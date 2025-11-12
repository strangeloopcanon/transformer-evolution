from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from evoforge.dsl.errors import DSLValidationError
from evoforge.dsl.models import DSLConfig, RecurrenceConfig

from .utils import summarize_architecture


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        inv_rms = torch.rsqrt(norm + self.eps)
        return self.weight * x * inv_rms


def build_norm(kind: str, dim: int) -> nn.Module:
    if kind == "RMSNorm":
        return RMSNorm(dim)
    if kind == "LayerNorm":
        return nn.LayerNorm(dim)
    if kind == "ScaleNorm":
        # ScaleNorm ~ RMSNorm without weights; simple implementation.
        return RMSNorm(dim)
    raise DSLValidationError(f"Unsupported norm kind '{kind}' for simple builder")


def build_activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    raise DSLValidationError(f"Unsupported activation '{name}' in simple builder")


class GatedFeedForward(nn.Module):
    def __init__(self, dim: int, mult: float, gate: str) -> None:
        super().__init__()
        hidden = int(dim * mult)
        hidden = max(hidden, dim)
        self.proj = nn.Linear(dim, hidden * 2)
        self.out = nn.Linear(hidden, dim)
        if gate == "swiglu":
            self.act = nn.SiLU()
        elif gate == "geglu":
            self.act = nn.GELU()
        else:
            raise DSLValidationError(f"Unsupported gated activation '{gate}'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.proj(x).chunk(2, dim=-1)
        return self.out(self.act(a) * b)


def build_ffn(dim: int, mult: float, act: str) -> nn.Module:
    if act in {"swiglu", "geglu"}:
        return GatedFeedForward(dim, mult, act)
    activation = build_activation(act)
    hidden = int(dim * mult)
    hidden = max(hidden, dim)
    return nn.Sequential(
        nn.Linear(dim, hidden),
        activation,
        nn.Linear(hidden, dim),
    )


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * -(torch.log(torch.tensor(10000.0)) / dim)
        )
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return self.pe[:seq_len].unsqueeze(0)


class SimpleBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ffn_mult: float, norm_kind: str, ffn_act: str) -> None:
        super().__init__()
        self.norm1 = build_norm(norm_kind, dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = build_norm(norm_kind, dim)
        self.ff = build_ffn(dim, ffn_mult, act=ffn_act)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=attn_mask, need_weights=False)
        x = residual + attn_out
        x = x + self.ff(self.norm2(x))
        return x


class TransformerStack(nn.Module):
    def __init__(
        self,
        dim: int,
        n_layers: int,
        heads: int,
        ffn_mult: float,
        ffn_act: str,
        norm_kind: str,
        max_seq_len: int,
        recurrence: Optional[RecurrenceConfig] = None,
        causal: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.causal = causal
        self.pos_embed = SinusoidalPositionalEncoding(dim, max_seq_len)
        self.blocks = nn.ModuleList(
            [SimpleBlock(dim, heads, ffn_mult, norm_kind, ffn_act) for _ in range(n_layers)]
        )
        self.final_norm = build_norm(norm_kind, dim)
        self.recurrence_cfg = recurrence
        if recurrence:
            self._rec_prelude = recurrence.prelude
            self._rec_body = recurrence.body
            self._rec_coda = recurrence.coda
            self._rec_adapter = (
                nn.Linear(dim * 2, dim) if recurrence.adapter == "concat_linear" else None
            )
        else:
            self._rec_prelude = 0
            self._rec_body = 0
            self._rec_coda = 0
            self._rec_adapter = None

    def _causal_mask(self, length: int, device: torch.device) -> Optional[torch.Tensor]:
        if not self.causal:
            return None
        mask = torch.full((length, length), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x = x + self.pos_embed(x)
        mask = self._causal_mask(seq_len, x.device)
        if not self.recurrence_cfg:
            for block in self.blocks:
                x = block(x, mask)
            return self.final_norm(x)

        idx = 0
        hidden = x
        for _ in range(self._rec_prelude):
            hidden = self.blocks[idx](hidden, mask)
            idx += 1
        anchor = hidden
        state = self._inject_recurrence_noise(anchor)
        loops = self._resolve_recurrence_loops()
        for _ in range(loops):
            step_state = state
            for offset in range(self._rec_body):
                step_state = self.blocks[idx + offset](step_state, mask)
            state = self._apply_recurrence_adapter(step_state, anchor)
        idx += self._rec_body
        hidden = state
        for _ in range(self._rec_coda):
            hidden = self.blocks[idx](hidden, mask)
            idx += 1
        return self.final_norm(hidden)

    def _resolve_recurrence_loops(self) -> int:
        if not self.recurrence_cfg or not self.recurrence_cfg.loops:
            return 1
        loops = self.recurrence_cfg.loops
        base = loops.train if self.training else (loops.eval or loops.train)
        return max(1, int(base))

    def _apply_recurrence_adapter(self, state: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        cfg = self.recurrence_cfg
        if not cfg:
            return state
        if cfg.adapter == "identity":
            return state
        if cfg.adapter == "residual":
            return state + anchor
        if cfg.adapter == "concat_linear" and self._rec_adapter is not None:
            combined = torch.cat([anchor, state], dim=-1)
            return self._rec_adapter(combined)
        return state

    def _inject_recurrence_noise(self, anchor: torch.Tensor) -> torch.Tensor:
        cfg = self.recurrence_cfg
        if not cfg or cfg.noise_std <= 0:
            return anchor
        noise = torch.randn_like(anchor) * cfg.noise_std
        return anchor + noise


class SimpleTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        n_layers: int,
        heads: int,
        ffn_mult: float,
        ffn_act: str,
        norm_kind: str,
        max_seq_len: int,
        recurrence: Optional[RecurrenceConfig] = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.recurrence_cfg = recurrence
        self.token_embed = nn.Embedding(vocab_size, dim)
        self.stack = TransformerStack(
            dim=dim,
            n_layers=n_layers,
            heads=heads,
            ffn_mult=ffn_mult,
            ffn_act=ffn_act,
            norm_kind=norm_kind,
            max_seq_len=max_seq_len,
            recurrence=recurrence,
            causal=True,
        )
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embed(input_ids)
        x = self.stack(x)
        logits = self.lm_head(x)
        return logits


@dataclass
class BuildMetadata:
    vocab_size: int
    max_seq_len: int
    dim: int
    n_layers: int
    extras: Optional[Dict[str, object]] = None


def _check_supported(cfg: DSLConfig) -> Tuple[int, int, int, float, str]:
    arch = cfg.arch
    mixer = arch.mix_unit.mixer
    if mixer is None and arch.mix_unit.choices:
        walker = arch.mix_unit.choices[0]
        mixer = walker
    if mixer is None:
        raise DSLValidationError("Simple builder requires an attention mixer")
    if mixer.kind != "Attention":
        raise DSLValidationError("Simple builder only supports Attention mixers")
    # Conditioning ops are ignored in the simple builder fallback.
    if arch.ffn.kind != "dense":
        raise DSLValidationError("Simple builder supports dense FFN only")
    if mixer.groups not in (None, mixer.heads, 1):
        mixer.groups = mixer.heads
    heads = mixer.heads or 1
    dim = arch.d_model
    if dim % heads != 0:
        # fall back to single-head attention if divisibility fails
        heads = 1
        mixer.heads = heads
    ffn_mult = arch.ffn.mult
    ffn_act = arch.ffn.act
    return dim, arch.n_layers, heads, ffn_mult, ffn_act


def build_simple_model(
    cfg: DSLConfig,
    *,
    vocab_size: Optional[int] = None,
) -> Tuple[nn.Module, BuildMetadata]:
    dim, n_layers, heads, ffn_mult, ffn_act = _check_supported(cfg)
    ctx_len = cfg.train.ctx_len
    vocab = vocab_size or cfg.train.vocab_size or 256
    model = SimpleTransformer(
        vocab_size=vocab,
        dim=dim,
        n_layers=n_layers,
        heads=heads,
        ffn_mult=ffn_mult,
        ffn_act=ffn_act,
        norm_kind=cfg.arch.norm,
        max_seq_len=ctx_len,
        recurrence=cfg.arch.recurrence,
    )
    meta = BuildMetadata(
        vocab_size=vocab,
        max_seq_len=ctx_len,
        dim=dim,
        n_layers=n_layers,
        extras=summarize_architecture(cfg),
    )
    return model, meta
