from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
from timm import create_model
from models.token_keep import check_vit_resolution, get_patch_size, interpolate_abs_pos_embed, keep_and_gather

class ViTEncoder(nn.Module):

    def __init__(self, model_name: str='vit_base_patch16_224', freeze: bool=False, input_image_size: int=512, token_keep: bool=True, keep_ratio: float=1.0, selector_type: str='l2', pretrained: bool=True):
        super().__init__()
        self.model_name = model_name
        self.input_image_size = int(input_image_size)
        self.token_keep = bool(token_keep)
        self.keep_ratio = float(keep_ratio)
        self.selector_type = selector_type
        self.last_token_info: Dict = {}
        self.backbone = create_model(model_name, pretrained=pretrained)
        self.hidden_dim = self.backbone.embed_dim
        self.patch_size = self._read_patch_size()
        (self.grid_h, self.grid_w, self.num_patches) = check_vit_resolution(self.input_image_size, self.patch_size)
        self.num_prefix_tokens = self._read_num_prefix_tokens()
        self._adapt_input_size_and_pos_embed()
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    @property
    def hidden_size(self) -> int:
        return self.hidden_dim

    def _read_patch_size(self) -> Tuple[int, int]:
        return get_patch_size(self.backbone.patch_embed.patch_size)

    def _read_num_prefix_tokens(self) -> int:
        if hasattr(self.backbone, 'num_prefix_tokens'):
            return int(self.backbone.num_prefix_tokens)
        n = 0
        if getattr(self.backbone, 'cls_token', None) is not None:
            n += 1
        if getattr(self.backbone, 'dist_token', None) is not None:
            n += 1
        n += int(getattr(self.backbone, 'num_reg_tokens', 0) or 0)
        return n

    def _adapt_input_size_and_pos_embed(self) -> None:
        patch_embed = self.backbone.patch_embed
        patch_embed.img_size = (self.input_image_size, self.input_image_size)
        if hasattr(patch_embed, 'grid_size'):
            patch_embed.grid_size = (self.grid_h, self.grid_w)
        if hasattr(patch_embed, 'num_patches'):
            patch_embed.num_patches = self.num_patches
        pos = self.backbone.pos_embed
        no_embed_class = bool(getattr(self.backbone, 'no_embed_class', False))
        prefix_in_pos = 0 if no_embed_class else self.num_prefix_tokens
        new_pos = interpolate_abs_pos_embed(pos_embed=pos.data, new_grid=(self.grid_h, self.grid_w), num_prefix_tokens=prefix_in_pos)
        if tuple(new_pos.shape) != tuple(pos.shape):
            self.backbone.pos_embed = nn.Parameter(new_pos.clone())
        else:
            self.backbone.pos_embed.data.copy_(new_pos)

    def _split_pos_embed(self) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        pos = self.backbone.pos_embed
        no_embed_class = bool(getattr(self.backbone, 'no_embed_class', False))
        if no_embed_class:
            return (None, pos)
        return (pos[:, :self.num_prefix_tokens], pos[:, self.num_prefix_tokens:])

    def _build_prefix_tokens(self, batch_size: int, device, dtype) -> torch.Tensor:
        tokens = []
        cls_token = getattr(self.backbone, 'cls_token', None)
        if cls_token is not None:
            tokens.append(cls_token.expand(batch_size, -1, -1).to(device=device, dtype=dtype))
        dist_token = getattr(self.backbone, 'dist_token', None)
        if dist_token is not None:
            tokens.append(dist_token.expand(batch_size, -1, -1).to(device=device, dtype=dtype))
        reg_token = getattr(self.backbone, 'reg_token', None)
        if reg_token is not None:
            tokens.append(reg_token.expand(batch_size, -1, -1).to(device=device, dtype=dtype))
        if not tokens:
            return torch.empty(batch_size, 0, self.hidden_dim, device=device, dtype=dtype)
        return torch.cat(tokens, dim=1)

    def embed_tokens(self, x: torch.Tensor, x_score: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, Dict]:
        if x.shape[-2:] != (self.input_image_size, self.input_image_size):
            raise ValueError(f'ViT branch input is {tuple(x.shape[-2:])}, expected {(self.input_image_size, self.input_image_size)}.')
        patch_tokens = self.backbone.patch_embed(x)
        b = patch_tokens.shape[0]
        prefix_tokens = self._build_prefix_tokens(b, patch_tokens.device, patch_tokens.dtype)
        (prefix_pos, patch_pos) = self._split_pos_embed()
        keep_indices = None
        if self.token_keep:
            if x_score is None:
                raise ValueError('token_keep=True requires x_score: the same augmented image in [0, 1] before Normalize.')
            if x_score.shape[-2:] != (self.input_image_size, self.input_image_size):
                raise ValueError(f'x_score size {tuple(x_score.shape[-2:])} != ViT input {(self.input_image_size, self.input_image_size)}')
            (selected_tokens, selected_pos, keep_indices, select_info) = keep_and_gather(patch_tokens=patch_tokens, patch_pos_embed=patch_pos, x_score=x_score, patch_size=self.patch_size, keep_ratio=self.keep_ratio, selector_type=self.selector_type)
            patch_tokens = selected_tokens
            patch_pos = selected_pos
        else:
            select_info = {'scores': None, 'indices': None, 'num_patches': self.num_patches, 'num_kept': self.num_patches, 'keep_ratio': 1.0}
            if patch_pos.shape[0] == 1:
                patch_pos = patch_pos.expand(b, -1, -1)
        if prefix_pos is None:
            x_seq = patch_tokens + patch_pos
            if prefix_tokens.shape[1] > 0:
                x_seq = torch.cat([prefix_tokens, x_seq], dim=1)
        else:
            prefix_pos_b = prefix_pos.expand(b, -1, -1)
            x_seq = torch.cat([prefix_tokens, patch_tokens], dim=1)
            pos_sel = torch.cat([prefix_pos_b, patch_pos], dim=1)
            x_seq = x_seq + pos_sel
        x_seq = self.backbone.pos_drop(x_seq)
        token_info = {'keep_indices': keep_indices, 'num_patches': int(self.num_patches), 'num_kept_patches': int(patch_tokens.shape[1]), 'num_prefix_tokens': int(prefix_tokens.shape[1]), 'grid_size': (self.grid_h, self.grid_w), 'patch_size': self.patch_size, 'keep_ratio': self.keep_ratio if self.token_keep else 1.0, 'encoder_seq_len': int(x_seq.shape[1])}
        token_info.update(select_info)
        self.last_token_info = token_info
        return (x_seq, token_info)

def load_vit_pos_embed_from_state_dict(encoder: ViTEncoder, state_dict: dict, prefix: str='') -> dict:
    key = f'{prefix}backbone.pos_embed'
    if key not in state_dict:
        return state_dict
    ckpt_pos = state_dict[key]
    if ckpt_pos.shape == encoder.backbone.pos_embed.shape:
        return state_dict
    no_embed_class = bool(getattr(encoder.backbone, 'no_embed_class', False))
    prefix_in_pos = 0 if no_embed_class else encoder.num_prefix_tokens
    state_dict = dict(state_dict)
    state_dict[key] = interpolate_abs_pos_embed(pos_embed=ckpt_pos, new_grid=(encoder.grid_h, encoder.grid_w), num_prefix_tokens=prefix_in_pos)
    return state_dict
