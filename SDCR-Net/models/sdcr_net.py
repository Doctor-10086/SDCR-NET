from __future__ import annotations
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from models.convnext_encoder import ConvNeXtEncoder
from models.reinject import SpatialAlignReinject
from models.vit_encoder import ViTEncoder, load_vit_pos_embed_from_state_dict

class SDCRNet(nn.Module):

    def __init__(self, num_classes: int, image_size: int, vit_input_size: int=None, fast_model: str='vit_base_patch16_224', slow_model: str='convnext_base', dropout: float=0.1, multilabel: bool=False, token_keep: bool=True, keep_ratio: float=1.0, selector_type: str='l2', pretrained: bool=True, reinject_down_dim: int=256, reinject_num_heads: int=8, num_reinject: int=3, reinject_scale: float=0.1, pool_mode: str='cls', fusion_weight: float=0.2, freeze_fusion: bool=False):
        super().__init__()
        self.multilabel = bool(multilabel)
        self.image_size = int(image_size)
        self.vit_input_size = int(self.image_size if vit_input_size is None else vit_input_size)
        self.token_keep = bool(token_keep)
        self.keep_ratio = float(keep_ratio)
        self.selector_type = selector_type
        self.pool_mode = str(pool_mode)
        self.num_reinject = int(num_reinject)
        self.reinject_scale = float(reinject_scale)
        if self.pool_mode not in ('cls', 'mean'):
            raise ValueError(f"pool_mode must be 'cls' or 'mean', got {pool_mode}")
        if self.num_reinject < 0:
            raise ValueError(f'num_reinject must be >= 0, got {num_reinject}')
        self.fast_encoder = ViTEncoder(model_name=fast_model, input_image_size=self.vit_input_size, token_keep=self.token_keep, keep_ratio=self.keep_ratio, selector_type=self.selector_type, pretrained=pretrained)
        self.slow_encoder = ConvNeXtEncoder(model_name=slow_model, input_image_size=self.image_size)
        self.fast_dim = self.fast_encoder.hidden_size
        self.slow_dim = self.slow_encoder.hidden_size
        self.reinject = nn.ModuleList([SpatialAlignReinject(query_dim=self.fast_dim, cnn_dim=self.slow_dim, grid_h=self.fast_encoder.grid_h, grid_w=self.fast_encoder.grid_w, down_dim=reinject_down_dim, num_heads=reinject_num_heads, zero_init=True, residual_scale=self.reinject_scale) for _ in range(self.num_reinject)])
        self.fast_norm = nn.LayerNorm(self.fast_dim)
        self.fast_drop = nn.Dropout(dropout)
        self.fast_head = nn.Linear(self.fast_dim, num_classes)
        self.slow_norm = nn.LayerNorm(self.slow_dim)
        self.slow_drop = nn.Dropout(dropout)
        self.slow_head = nn.Linear(self.slow_dim, num_classes)
        if not 0.0 <= float(fusion_weight) <= 1.0:
            raise ValueError(f'fusion_weight must be in [0, 1], got {fusion_weight}')
        self.freeze_fusion = bool(freeze_fusion)
        self.fusion_weight = nn.Parameter(torch.tensor(float(fusion_weight), dtype=torch.float32), requires_grad=not self.freeze_fusion)

    def _stage_boundaries(self, n_blocks: int) -> List[int]:
        remain = n_blocks - 2
        if remain <= 0 or self.num_reinject <= 0:
            return []
        n_stages = self.num_reinject + 1
        step = max(1, remain // n_stages)
        boundaries = []
        for i in range(self.num_reinject):
            after = 2 + (i + 1) * step
            if after >= n_blocks:
                break
            boundaries.append(after)
        return boundaries

    def _run_vit_blocks(self, x_seq: torch.Tensor, cnn_feat: torch.Tensor) -> torch.Tensor:
        blocks = list(self.fast_encoder.backbone.blocks)
        inject_after = set(self._stage_boundaries(len(blocks)))
        inject_i = 0
        for (bi, blk) in enumerate(blocks):
            if self.training:
                x_seq = checkpoint(blk.__call__, x_seq, use_reentrant=False)
            else:
                x_seq = blk(x_seq)
            if bi + 1 in inject_after and inject_i < len(self.reinject):
                x_seq = self.reinject[inject_i](x_seq, cnn_feat)
                inject_i += 1
        return self.fast_encoder.backbone.norm(x_seq)

    def _pool_vit(self, x_seq: torch.Tensor, n_prefix: int) -> torch.Tensor:
        if self.pool_mode == 'cls' and n_prefix > 0:
            return x_seq[:, 0]
        patches = x_seq[:, n_prefix:] if n_prefix > 0 else x_seq
        return patches.mean(dim=1)

    def forward(self, x: torch.Tensor, vit_norm: torch.Tensor=None, vit_score: torch.Tensor=None, return_path_logits: bool=False):
        if x.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(f'CNN branch input is {tuple(x.shape[-2:])}, expected {(self.image_size, self.image_size)}.')
        if vit_norm is None:
            if self.vit_input_size != self.image_size:
                raise ValueError('CNN and ViT sizes differ; vit_norm (and vit_score if Token Keep is on) must come from the dataloader.')
            vit_norm = x
        if vit_norm.shape[-2:] != (self.vit_input_size, self.vit_input_size):
            raise ValueError(f'ViT input is {tuple(vit_norm.shape[-2:])}, expected {(self.vit_input_size, self.vit_input_size)}.')
        (x_seq, token_info) = self.fast_encoder.embed_tokens(vit_norm, x_score=vit_score)
        slow_feat = self.slow_encoder.forward_features(x)
        x_seq = self._run_vit_blocks(x_seq, slow_feat)
        n_prefix = int(token_info['num_prefix_tokens'])
        fast_pooled = self._pool_vit(x_seq, n_prefix)
        fast_logits = self.fast_head(self.fast_drop(self.fast_norm(fast_pooled)))
        slow_pooled = F.adaptive_avg_pool2d(slow_feat, 1).flatten(1)
        slow_logits = self.slow_head(self.slow_drop(self.slow_norm(slow_pooled)))
        w = self.fusion_weight
        final_logits = w * fast_logits + (1.0 - w) * slow_logits
        if return_path_logits:
            return (final_logits, fast_logits, slow_logits)
        return final_logits

def load_sdcr_checkpoint(model: SDCRNet, checkpoint, strict: bool=True):
    if isinstance(checkpoint, str):
        checkpoint = torch.load(checkpoint, map_location='cpu')
    state = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
    state = load_vit_pos_embed_from_state_dict(model.fast_encoder, state, prefix='fast_encoder.')
    return model.load_state_dict(state, strict=strict)
