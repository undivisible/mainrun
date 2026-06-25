import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (rotate_half(x) * sin)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)

    def _update_cos_sin(self, seqlen: int, device: torch.device, dtype: torch.dtype):
        if (
            self._seq_len_cached >= seqlen
            and self.cos_cached.device == device
            and self.cos_cached.dtype == dtype
        ):
            return
        t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos().to(dtype)
        self.sin_cached = emb.sin().to(dtype)
        self._seq_len_cached = seqlen

    def get_cos_sin(self, seqlen: int, device: torch.device, dtype: torch.dtype):
        self._update_cos_sin(seqlen, device, dtype)
        return self.cos_cached[:seqlen], self.sin_cached[:seqlen]