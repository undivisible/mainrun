"""Optional experimental linear attention (L=128: softmax baseline preferred for val loss)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from rope import RotaryEmbedding, apply_rotary_pos_emb


def feature_map(x: torch.Tensor) -> torch.Tensor:
    return F.elu(x) + 1.0


def causal_linear_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    q = feature_map(F.normalize(q, dim=-1))
    k = feature_map(F.normalize(k, dim=-1))
    kv = torch.einsum("bhtd,bhte->bhtde", k, v).cumsum(dim=2)
    k_sum = k.cumsum(dim=2)
    num = torch.einsum("bhtd,bhtde->bhte", q, kv)
    den = torch.einsum("bhtd,bhtd->bht", q, k_sum).unsqueeze(-1)
    return num / (den + eps)


class LinearSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.rotary = RotaryEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim).transpose(1, 2)
        q, k, v = qkv.unbind(2)
        cos, sin = self.rotary.get_cos_sin(T, x.device, x.dtype)
        cos, sin = cos[None, None, :, :], sin[None, None, :, :]
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)
        y = causal_linear_attention(q, k, v)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)