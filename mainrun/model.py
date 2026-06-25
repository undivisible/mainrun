import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from rope import RotaryEmbedding, apply_rotary_pos_emb


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    d_model: int
    dropout: float
    eos_id: int = 0
    rope_theta: float = 10000.0
    rms_eps: float = 1e-5
    swiglu_mult: float = 8 / 3
    qk_norm: bool = False
    label_smoothing: float = 0.0


def title_boundary_attn_mask(idx: torch.Tensor, eos_id: int) -> torch.Tensor:
    """Causal + no attention across <eos> title boundaries (SuryaKannan-style)."""
    B, T = idx.shape
    eos = idx == eos_id
    seg = eos.cumsum(dim=1) - eos.long()
    same_seg = seg.unsqueeze(2) == seg.unsqueeze(1)
    causal = torch.tril(torch.ones(T, T, device=idx.device, dtype=torch.bool))
    allowed = causal.unsqueeze(0) & same_seg
    return torch.zeros(B, 1, T, T, device=idx.device, dtype=torch.float32).masked_fill(
        ~allowed.unsqueeze(1), float("-inf")
    )


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * x


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.d_model // cfg.n_head
        assert self.head_dim % 2 == 0
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        self.rotary = RotaryEmbedding(self.head_dim, base=cfg.rope_theta)
        self.q_norm = RMSNorm(self.head_dim, eps=cfg.rms_eps) if cfg.qk_norm else None
        self.k_norm = RMSNorm(self.head_dim, eps=cfg.rms_eps) if cfg.qk_norm else None

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim)
        q, k, v = (t.transpose(1, 2) for t in qkv.unbind(2))
        cos, sin = self.rotary.get_cos_sin(T, x.device, x.dtype)
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=False,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class SwiGLU(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        hidden = int(cfg.swiglu_mult * cfg.d_model)
        self.w_gate = nn.Linear(cfg.d_model, hidden, bias=False)
        self.w_up = nn.Linear(cfg.d_model, hidden, bias=False)
        self.w_out = nn.Linear(hidden, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w_out(F.silu(self.w_gate(x)) * self.w_up(x)))


class Block(nn.Module):
    """NeoX-style parallel: attn and FFN branches read from same residual x."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, eps=cfg.rms_eps)
        self.mlp_norm = RMSNorm(cfg.d_model, eps=cfg.rms_eps)
        self.attn = CausalSelfAttention(cfg)
        self.mlp = SwiGLU(cfg)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        return x + self.attn(self.attn_norm(x), attn_mask) + self.mlp(self.mlp_norm(x))


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = RMSNorm(cfg.d_model, eps=cfg.rms_eps)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)
        for block in self.blocks:
            nn.init.normal_(block.attn.proj.weight, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))
            nn.init.normal_(block.mlp.w_out.weight, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))
        self.head.weight = self.token_emb.weight

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        attn_mask = title_boundary_attn_mask(idx, self.cfg.eos_id)
        x = self.drop(self.token_emb(idx))
        for block in self.blocks:
            x = block(x, attn_mask)
        x = self.ln_f(x)
        logits = self.head(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            reduction="mean",
            label_smoothing=self.cfg.label_smoothing if self.training else 0.0,
        )
        return logits, loss

    def get_optimizer_param_groups(self):
        muon_params, adamw_params = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim == 2 and "token_emb" not in name and "head" not in name:
                muon_params.append(p)
            else:
                adamw_params.append(p)
        return muon_params, adamw_params