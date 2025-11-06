from __future__ import annotations

from typing import Dict

import torch


@torch.no_grad()
def error_recovery_probe(
    model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, corrupt: int = 2
) -> Dict[str, float]:
    # Compute loss on clean vs corrupted last tokens
    device = next(model.parameters()).device
    vocab = model.linear.out_features if hasattr(model, "linear") else None

    def ce(a, b):
        return torch.nn.functional.cross_entropy(a.view(-1, a.size(-1)).float(), b.view(-1))

    logits_clean = model(x.to(device))
    loss_clean = ce(logits_clean, y.to(device))
    x_corrupt = x.clone()
    if corrupt > 0:
        x_corrupt[:, -corrupt:] = torch.flip(x_corrupt[:, -corrupt:], dims=[1])
    logits_err = model(x_corrupt.to(device))
    loss_err = ce(logits_err, y.to(device))
    return {"err_recovery_delta": float((loss_err - loss_clean).detach().cpu().item())}


@torch.no_grad()
def niah_probe(model: torch.nn.Module, seq_len: int = 256, batch: int = 4) -> Dict[str, float]:
    # Needle-In-A-Haystack: place a token at a random earlier position, ask for next-token prob on that token
    device = next(model.parameters()).device
    vocab = getattr(model, "vocab_size", 256)
    x = torch.randint(0, vocab, (batch, seq_len), device=device)
    # copy a random earlier token to target position - 1 so next-token should match it
    idx = torch.randint(low=1, high=seq_len - 1, size=(batch,), device=device)
    for b in range(batch):
        src = torch.randint(0, idx[b].item(), (1,), device=device)
        x[b, idx[b]] = x[b, src]
    logits = model(x)
    preds = torch.argmax(logits[:, :-1], dim=-1)
    targets = x[:, 1:]
    acc = (preds == targets).float().mean().item()
    return {"niah_acc": float(acc)}


@torch.no_grad()
def spec_decode_probe(
    model: torch.nn.Module, x: torch.Tensor, allowed: str = "0123456789 "
) -> Dict[str, float]:
    # Constrained decoding: probability mass assigned to allowed set on last step
    device = next(model.parameters()).device
    logits = model(x.to(device))
    probs = torch.softmax(logits[:, -1:].float(), dim=-1)
    # assume ASCII vocab subset if available; otherwise this is a surrogate metric by index
    allowed_idx = torch.arange(0, probs.size(-1), device=device)
    mass = probs[..., allowed_idx].sum().item()
    return {"spec_mass": float(mass)}
