from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from evoforge.builders import build_model
from evoforge.data import create_tiny_dataset
from evoforge.dsl.api import load_validate_yaml
from evoforge.dsl.errors import DSLValidationError
from evoforge.dsl.models import DSLConfig


def _select_device(requested: Optional[str] = None) -> torch.device:
    if requested:
        dev = torch.device(requested)
        if dev.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        if dev.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available")
        return dev
    return torch.device("cpu")


@dataclass
class TrainResult:
    loss_history: list[float]
    tokens_per_sec: float
    total_tokens: int
    total_flops: float
    metadata: dict


def _estimate_flops(cfg: DSLConfig, tokens: int) -> float:
    arch = cfg.arch
    d_model = arch.d_model
    mix_unit = arch.mix_unit
    ffn_mult = arch.ffn.mult
    layers = arch.n_layers
    heads = (mix_unit.mixer.heads if mix_unit.mixer else d_model // 64) if mix_unit else d_model // 64

    attn_flops = 4 * d_model * d_model + 2 * (heads or 1) * (d_model // (heads or 1)) * tokens
    ffn_flops = 2 * d_model * int(ffn_mult * d_model)
    per_layer = attn_flops + ffn_flops
    return per_layer * layers * tokens


def _enforce_train_budget(cfg: DSLConfig, steps: int, batch_tokens: int) -> None:
    budget = cfg.train.budget
    if not budget:
        return
    tokens_cap = budget.get("tokens_per_step")
    if tokens_cap is not None and batch_tokens > tokens_cap:
        raise DSLValidationError(
            f"Batch tokens {batch_tokens} exceed tokens_per_step budget {tokens_cap}"
        )
    if budget.get("max_steps") is not None and steps > budget["max_steps"]:
        raise DSLValidationError(
            f"Requested steps {steps} exceed max_steps budget {budget['max_steps']}"
        )


def _enforce_flop_budget(cfg: DSLConfig, step_flops: float) -> None:
    budget = cfg.train.budget
    if not budget:
        return
    flops_cap = budget.get("flops_per_step")
    if flops_cap is not None and step_flops > flops_cap:
        raise DSLValidationError(
            f"Step FLOPs {step_flops:.2e} exceed flops_per_step budget {flops_cap:.2e}"
        )


@torch.no_grad()
def _compute_calibration_metrics(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    logits = model(x)
    probs = torch.softmax(logits.float(), dim=-1)
    confidences, preds = probs.max(dim=-1)
    accuracy = (preds == y).float()

    flat_conf = confidences.reshape(-1)
    flat_acc = accuracy.reshape(-1)
    total = max(flat_conf.numel(), 1)
    bins = torch.linspace(0.0, 1.0, steps=11, device=flat_conf.device)
    ece = torch.tensor(0.0, device=flat_conf.device)
    for idx in range(10):
        mask = (flat_conf > bins[idx]) & (flat_conf <= bins[idx + 1])
        if mask.any():
            bin_prob = flat_conf[mask].mean()
            bin_acc = flat_acc[mask].mean()
            ece += mask.float().mean() * (bin_prob - bin_acc).abs()
    return {
        "ece": float(ece.item()),
        "avg_confidence": float(flat_conf.mean().item()),
        "accuracy": float(flat_acc.mean().item()),
    }


def run_micro_train(
    cfg_path: Path,
    *,
    steps: int = 20,
    seq_len: int = 64,
    batch_size: int = 4,
    device: Optional[str] = None,
    learning_rate: float = 3e-4,
    allow_fallback: bool = True,
) -> TrainResult:
    if steps <= 0:
        raise ValueError("steps must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    cfg = load_validate_yaml(cfg_path)
    dataset = create_tiny_dataset(seq_len)
    dataset_vocab = dataset.vocab_size
    cfg_vocab = cfg.train.vocab_size or dataset_vocab
    if cfg_vocab < dataset_vocab:
        raise DSLValidationError(
            f"Config vocab_size {cfg_vocab} smaller than dataset vocab {dataset_vocab}"
        )

    batch_tokens = batch_size * seq_len
    _enforce_train_budget(cfg, steps, batch_tokens)

    def _train_once(dev: torch.device) -> TrainResult:
        model, meta = build_model(cfg, vocab_size=dataset_vocab)
        dtype = torch.float16 if dev.type == "cuda" else torch.float32
        model = model.to(device=dev, dtype=dtype)

        optim = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=cfg.train.wd or 0.0)

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        data_iter = itertools.cycle(loader)

        loss_history: list[float] = []
        tokens_processed = 0
        start = time.perf_counter()

        total_tokens = 0
        total_flops = 0.0
        for _ in range(steps):
            x, y = next(data_iter)
            x = x.to(dev)
            y = y.to(dev)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, meta.vocab_size).float(), y.view(-1))
            # Collect auxiliary losses (e.g., router balance)
            aux_total = 0.0
            for m in model.modules():
                aux = getattr(m, "_aux_loss", None)
                if aux is not None:
                    try:
                        aux_total = aux_total + float(aux.detach().cpu().item())
                    except Exception:
                        pass
                    setattr(m, "_aux_loss", None)
            if aux_total:
                loss = loss + torch.tensor(aux_total, device=loss.device, dtype=loss.dtype)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip or 1.0)
            optim.step()

            loss_history.append(loss.detach().cpu().item())
            tokens_processed += batch_tokens
            total_tokens += batch_tokens
            step_flops = _estimate_flops(cfg, batch_tokens)
            _enforce_flop_budget(cfg, step_flops)
            total_flops += step_flops

        elapsed = max(time.perf_counter() - start, 1e-6)
        tokens_per_sec = tokens_processed / elapsed

        metadata = {
            "train_budget": dict(cfg.train.budget) if cfg.train.budget else None,
            "architecture": meta.extras or {},
        }

        eval_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        metrics: Dict[str, float] = {}
        try:
            model.eval()
            calib_iter = iter(eval_loader)
            calib_x, calib_y = next(calib_iter)
            calib_x = calib_x.to(dev)
            calib_y = calib_y.to(dev)
            metrics.update(_compute_calibration_metrics(model, calib_x, calib_y))
        finally:
            model.train()

        if loss_history:
            metrics.update(
                {
                    "loss_start": float(loss_history[0]),
                    "loss_final": float(loss_history[-1]),
                    "loss_delta": float(loss_history[0] - loss_history[-1]),
                }
            )

        metrics["tokens_per_sec"] = float(tokens_per_sec)
        metrics["total_tokens"] = int(total_tokens)
        metrics["total_flops"] = float(total_flops)
        # Collect router stats if present
        router_stats = []
        try:
            for m in model.modules():
                if hasattr(m, "router_stats"):
                    rs = m.router_stats()
                    if rs:
                        router_stats.append(rs)
        except Exception:
            pass
        if router_stats:
            metrics["router"] = router_stats
        metadata["metrics"] = metrics

        return TrainResult(
            loss_history=loss_history,
            tokens_per_sec=tokens_per_sec,
            total_tokens=total_tokens,
            total_flops=total_flops,
            metadata=metadata,
        )

    primary_device = _select_device(device)
    devices_to_try = [primary_device]
    if allow_fallback and primary_device.type != "cpu":
        devices_to_try.append(torch.device("cpu"))

    last_error: Optional[Exception] = None
    for dev in devices_to_try:
        try:
            if dev is not primary_device:
                print(f"[evoforge] falling back to {dev.type} device", flush=True)
            return _train_once(dev)
        except (RuntimeError, AssertionError) as err:
            last_error = err
            if dev.type == "cpu" or not allow_fallback:
                raise
            print(f"[evoforge] training failed on {dev}: {err}; retrying on cpu", flush=True)
            continue

    if last_error:
        raise last_error
    raise RuntimeError("Training failed without available device")
