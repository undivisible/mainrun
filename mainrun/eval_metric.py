"""Same validation definition as frozen evaluate() in train.py — for Burn/experiments."""

import torch
from torch.nn import functional as F


def validation_loss_per_char(
    model,
    val_ids: torch.Tensor,
    val_text: str,
    block_size: int,
    batch_size: int,
    device: torch.device,
    iter_full_split,
) -> float:
    model.eval()
    losses = 0.0
    with torch.no_grad():
        for xb, yb in iter_full_split(val_ids, block_size, batch_size, device):
            logits, _ = model(xb, yb)
            B, T, V = logits.size()
            loss = F.cross_entropy(logits.view(-1, V), yb.view(-1), reduction="sum")
            losses += loss.item()
    model.train()
    return losses / len(val_text)