from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ReInjectionAttention(nn.Module):

    def __init__(self, down_dim: int, kv_dim: int, num_heads: int):
        super().__init__()
        if down_dim % num_heads != 0:
            raise ValueError(f'down_dim={down_dim} must be divisible by num_heads={num_heads}')
        if kv_dim % num_heads != 0:
            raise ValueError(f'kv_dim={kv_dim} must be divisible by num_heads={num_heads}')
        self.num_heads = int(num_heads)
        self.down_dim = int(down_dim)
        self.kv_dim = int(kv_dim)
        self.down_head_dim = down_dim // num_heads
        self.v_head_dim = kv_dim // num_heads
        self.q_proj = nn.Linear(down_dim, down_dim, bias=True)
        self.k_proj = nn.Linear(down_dim, down_dim, bias=True)
        self.v_proj = nn.Linear(kv_dim, kv_dim, bias=True)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        (bsz, q_len, _) = query.size()
        k_len = key.size(1)
        v_len = value.size(1)
        q = self.q_proj(query).view(bsz, q_len, self.num_heads, self.down_head_dim).transpose(1, 2)
        k = self.k_proj(key).view(bsz, k_len, self.num_heads, self.down_head_dim).transpose(1, 2)
        v = self.v_proj(value).view(bsz, v_len, self.num_heads, self.v_head_dim).transpose(1, 2)
        attn_weights = torch.matmul(q.float(), k.float().transpose(2, 3)) / math.sqrt(self.down_head_dim)
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(v.dtype)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.kv_dim)
        return (attn_output, attn_weights)

class ReInjectionModule(nn.Module):

    def __init__(self, query_dim: int, kv_dim: int, down_dim: int=256, num_heads: int=8, zero_init_up: bool=True, residual_scale: float=0.1):
        super().__init__()
        self.query_dim = int(query_dim)
        self.kv_dim = int(kv_dim)
        self.down_dim = int(down_dim)
        self.residual_scale = float(residual_scale)
        self.down_q = nn.Linear(query_dim, down_dim, bias=True)
        self.down_k = nn.Linear(kv_dim, down_dim, bias=True)
        self.ln_q = nn.LayerNorm(down_dim)
        self.ln_k = nn.LayerNorm(down_dim)
        self.attn = ReInjectionAttention(down_dim=down_dim, kv_dim=kv_dim, num_heads=num_heads)
        self.up = nn.Linear(kv_dim, query_dim, bias=True)
        if zero_init_up:
            nn.init.zeros_(self.up.weight)
            nn.init.zeros_(self.up.bias)

    def forward(self, query: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        residual = query
        q = self.ln_q(self.down_q(query))
        k = self.ln_k(self.down_k(kv))
        (attn_out, _) = self.attn(query=q, key=k, value=kv)
        return residual + self.residual_scale * self.up(attn_out)

class SpatialAlignReinject(nn.Module):

    def __init__(self, query_dim: int, cnn_dim: int, grid_h: int, grid_w: int, down_dim: int=256, num_heads: int=8, zero_init: bool=True, residual_scale: float=0.1):
        super().__init__()
        self.grid_h = int(grid_h)
        self.grid_w = int(grid_w)
        self.slow_conv = nn.Conv2d(cnn_dim, cnn_dim, 1)
        self.slow_proj = nn.Conv2d(cnn_dim, query_dim, 1)
        nn.init.xavier_uniform_(self.slow_conv.weight)
        nn.init.zeros_(self.slow_conv.bias)
        if zero_init:
            nn.init.zeros_(self.slow_proj.weight)
            nn.init.zeros_(self.slow_proj.bias)
        else:
            nn.init.xavier_uniform_(self.slow_proj.weight)
            nn.init.zeros_(self.slow_proj.bias)
        self.reinject = ReInjectionModule(query_dim=query_dim, kv_dim=query_dim, down_dim=down_dim, num_heads=num_heads, zero_init_up=True, residual_scale=residual_scale)

    def align_kv(self, cnn_feat: torch.Tensor) -> torch.Tensor:
        aligned = F.interpolate(cnn_feat.float(), size=(self.grid_h, self.grid_w), mode='bilinear', align_corners=True).to(dtype=cnn_feat.dtype)
        aligned = self.slow_proj(F.gelu(self.slow_conv(aligned)))
        return aligned.flatten(2).transpose(1, 2).contiguous()

    def forward(self, query: torch.Tensor, cnn_feat: torch.Tensor) -> torch.Tensor:
        kv = self.align_kv(cnn_feat)
        return self.reinject(query, kv)
