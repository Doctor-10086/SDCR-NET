from __future__ import annotations
import math
from typing import Dict, Tuple, Union
import torch
import torch.nn.functional as F

def get_patch_size(patch_size: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    if isinstance(patch_size, int):
        return (patch_size, patch_size)
    if len(patch_size) != 2:
        raise ValueError(f'patch_size must be int or (h, w), got {patch_size}')
    return (int(patch_size[0]), int(patch_size[1]))

def check_vit_resolution(image_size: int, patch_size: Union[int, Tuple[int, int]]) -> Tuple[int, int, int]:
    (ph, pw) = get_patch_size(patch_size)
    if image_size <= 0:
        raise ValueError(f'vit_input_size must be positive, got {image_size}')
    if image_size % ph != 0 or image_size % pw != 0:
        raise ValueError(f'ViT input size {image_size} is incompatible with patch_size {(ph, pw)}. {image_size} must be divisible by both patch dimensions.')
    grid_h = image_size // ph
    grid_w = image_size // pw
    return (grid_h, grid_w, grid_h * grid_w)

def expected_keep_count(num_patches: int, keep_ratio: float) -> int:
    if keep_ratio <= 0 or keep_ratio > 1:
        raise ValueError(f'keep_ratio must be in (0, 1], got {keep_ratio}')
    return max(1, int(math.floor(keep_ratio * num_patches)))

def compute_l2_patch_scores(x_score: torch.Tensor, patch_size: Union[int, Tuple[int, int]]) -> torch.Tensor:
    if x_score.ndim != 4:
        raise ValueError(f'x_score must be (B, C, H, W), got {tuple(x_score.shape)}')
    if x_score.min() < -0.0001:
        raise ValueError('x_score appears ImageNet-normalized (negative values). L2 Token Keep must be computed on [0, 1] images, not on normalized tensors.')
    (ph, pw) = get_patch_size(patch_size)
    (b, c, h, w) = x_score.shape
    if h % ph != 0 or w % pw != 0:
        raise ValueError(f'x_score spatial size {(h, w)} is not divisible by patch_size {(ph, pw)}.')
    patches = F.unfold(x_score, kernel_size=(ph, pw), stride=(ph, pw))
    return patches.norm(p=2, dim=1)

def select_topk_indices(scores: torch.Tensor, keep_ratio: float) -> torch.Tensor:
    if scores.ndim != 2:
        raise ValueError(f'scores must be (B, N), got {tuple(scores.shape)}')
    k = expected_keep_count(scores.shape[1], keep_ratio)
    (_, indices) = torch.topk(scores, k=k, dim=1, largest=True, sorted=False)
    (indices, _) = torch.sort(indices, dim=1)
    return indices

def gather_tokens(tokens: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if tokens.ndim != 3:
        raise ValueError(f'tokens must be (B, N, D), got {tuple(tokens.shape)}')
    if indices.ndim != 2:
        raise ValueError(f'indices must be (B, k), got {tuple(indices.shape)}')
    if tokens.shape[0] == 1 and indices.shape[0] > 1:
        tokens = tokens.expand(indices.shape[0], -1, -1)
    if tokens.shape[0] != indices.shape[0]:
        raise ValueError(f'batch mismatch: tokens {tokens.shape[0]} vs indices {indices.shape[0]}')
    dim = tokens.shape[-1]
    gather_index = indices.unsqueeze(-1).expand(-1, -1, dim)
    return torch.gather(tokens, dim=1, index=gather_index)

def interpolate_abs_pos_embed(pos_embed: torch.Tensor, new_grid: Tuple[int, int], num_prefix_tokens: int) -> torch.Tensor:
    if pos_embed.ndim != 3 or pos_embed.shape[0] != 1:
        raise ValueError(f'pos_embed must be (1, L, D), got {tuple(pos_embed.shape)}')
    prefix = pos_embed[:, :num_prefix_tokens]
    patches = pos_embed[:, num_prefix_tokens:]
    n_old = patches.shape[1]
    old_gs = int(math.sqrt(n_old))
    if old_gs * old_gs != n_old:
        raise ValueError(f'Cannot restore patch pos_embed to a 2D grid: {n_old} is not a square.')
    (new_h, new_w) = (int(new_grid[0]), int(new_grid[1]))
    if old_gs == new_h and old_gs == new_w:
        return pos_embed
    patches_2d = patches.reshape(1, old_gs, old_gs, -1).permute(0, 3, 1, 2).float()
    patches_2d = F.interpolate(patches_2d, size=(new_h, new_w), mode='bicubic', align_corners=False)
    new_patches = patches_2d.to(dtype=pos_embed.dtype).permute(0, 2, 3, 1).reshape(1, new_h * new_w, -1)
    if num_prefix_tokens > 0:
        return torch.cat([prefix, new_patches], dim=1)
    return new_patches

def keep_and_gather(patch_tokens: torch.Tensor, patch_pos_embed: torch.Tensor, x_score: torch.Tensor, patch_size: Union[int, Tuple[int, int]], keep_ratio: float, selector_type: str='l2') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    if selector_type != 'l2':
        raise ValueError(f"selector_type={selector_type!r} is not implemented. Use 'l2'.")
    (b, n, _) = patch_tokens.shape
    with torch.no_grad():
        scores = compute_l2_patch_scores(x_score, patch_size)
        if scores.shape != (b, n):
            raise ValueError(f'Score count {tuple(scores.shape)} does not match patch tokens {(b, n)}.')
        indices = select_topk_indices(scores, keep_ratio)
    selected_tokens = gather_tokens(patch_tokens, indices)
    selected_pos = gather_tokens(patch_pos_embed, indices)
    info = {'scores': scores, 'indices': indices, 'num_patches': n, 'num_kept': indices.shape[1], 'keep_ratio': keep_ratio}
    return (selected_tokens, selected_pos, indices, info)
