from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from evoforge.dsl.errors import DSLValidationError
from evoforge.dsl.models import (
    DSLConfig,
    MixUnit,
    Mixer,
    ModuleSpec,
    PipelineStage,
    RecurrenceConfig,
    StencilConfig,
)

from .simple_torch import BuildMetadata, build_ffn, build_norm
from .utils import summarize_architecture


def _pick_attention_mixer(mix_unit: MixUnit) -> Mixer:
    if mix_unit.mixer:
        return mix_unit.mixer
    if mix_unit.choices:
        for choice in mix_unit.choices:
            if choice.kind == "Attention":
                return choice
        return mix_unit.choices[0]
    raise DSLValidationError("Transformer module requires at least one mixer definition")


def _resolve_mix_unit(module: ModuleSpec, arch_default: MixUnit) -> MixUnit:
    if module.mix_unit is not None:
        return module.mix_unit
    return arch_default


def _resolve_ffn(module: ModuleSpec, arch_ffn) -> Tuple[float, str]:
    if module.ffn is not None:
        return module.ffn.mult, module.ffn.act
    return arch_ffn.mult, arch_ffn.act


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int) -> None:
        super().__init__()
        position = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * -(torch.log(torch.tensor(10000.0)) / dim)
        )
        pe = torch.zeros(max_seq_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return self.pe[:seq_len].unsqueeze(0)


class EmbeddingStage(nn.Module):
    def __init__(self, name: str, vocab_size: int, dim: int, max_seq_len: int) -> None:
        super().__init__()
        self.name = name
        self.embedding = nn.Embedding(vocab_size, dim)
        self.pos_embed = PositionalEncoding(dim, max_seq_len)

    def forward(self, input_ids: torch.Tensor, *_: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        return x + self.pos_embed(x)


def _build_attn_mask(
    seq_len: int,
    device: torch.device,
    causal: bool,
    stencil: Optional[StencilConfig],
) -> Optional[torch.Tensor]:
    if stencil is None and not causal:
        return None

    mask = torch.zeros(seq_len, seq_len, device=device)

    if causal:
        causal_mask = torch.triu(torch.full_like(mask, float("-inf")), diagonal=1)
        mask = mask + causal_mask

    if stencil:
        kind = stencil.kind or "full"
        idx = torch.arange(seq_len, device=device)
        diff = idx.unsqueeze(0) - idx.unsqueeze(1)

        def allow(condition: torch.Tensor) -> None:
            nonlocal mask
            allowed = torch.where(
                condition, torch.zeros_like(mask), torch.full_like(mask, float("-inf"))
            )
            mask = torch.maximum(mask, allowed)

        if kind in {"local", "sliding", "dilated", "hybrid"}:
            window = stencil.window or seq_len
            dilation = stencil.dilation or 1
            condition = (diff.abs() <= window) & (diff.abs() % dilation == 0)
            allow(condition)
        elif kind == "block":
            block = stencil.block or stencil.window or max(1, seq_len // 4)
            block_ids = idx // block
            condition = block_ids.unsqueeze(0) == block_ids.unsqueeze(1)
            allow(condition)
        elif kind == "ring":
            block = stencil.block or max(1, seq_len // 4)
            stride = stencil.stride or block
            blocks = max(1, (seq_len + block - 1) // block)
            block_ids = idx // block
            prev_block = (block_ids - 1) % blocks
            next_block = (block_ids + 1) % blocks
            source = block_ids.unsqueeze(0)
            target = block_ids.unsqueeze(1)
            condition = (
                (source == target)
                | (source == prev_block.unsqueeze(1))
                | (source == next_block.unsqueeze(1))
            )
            allow(condition)
        elif kind == "cross":
            # Allow all connections (mask already handles causality)
            pass

    if torch.all(mask == 0):
        return None
    mask = torch.clamp(mask, min=float("-1e9"))
    return mask


class AttentionMixerModule(nn.Module):
    def __init__(self, mixer: Mixer, dim: int, causal: bool, max_seq_len: int) -> None:
        super().__init__()
        heads = mixer.heads or max(1, dim // max(1, mixer.head_dim or (dim // (mixer.heads or 1))))
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.causal = causal
        self.max_seq_len = max_seq_len
        self.stencil = mixer.stencil if mixer else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        mask = _build_attn_mask(seq_len, x.device, self.causal, self.stencil)
        out, _ = self.attn(x, x, x, attn_mask=mask)
        return out


class RetentionMixerModule(nn.Module):
    def __init__(self, dim: int, chunk: Optional[int] = None, mode: Optional[str] = None) -> None:
        super().__init__()
        self.chunk = max(1, chunk or 256)
        self.mode = mode or "parallel"
        self.input_proj = nn.Linear(dim, dim, bias=False)
        self.output_proj = nn.Linear(dim, dim, bias=False)
        self.decay = nn.Parameter(torch.zeros(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self._last_state: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = x.shape
        decay = torch.sigmoid(self.decay).view(1, dim)
        state = torch.zeros(batch, dim, device=x.device, dtype=x.dtype)
        outputs = []
        projected = self.input_proj(x)
        for start in range(0, seq_len, self.chunk):
            segment = projected[:, start : start + self.chunk, :]
            for t in range(segment.size(1)):
                state = decay * state + (1.0 - decay) * segment[:, t, :]
                outputs.append(state.unsqueeze(1))
        self._last_state = state.detach()
        stacked = torch.cat(outputs, dim=1)
        return self.output_proj(stacked) + self.bias.view(1, 1, -1)


class ConvMixerModule(nn.Module):
    def __init__(self, dim: int, kernel: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            dim,
            dim,
            kernel_size=max(1, kernel),
            padding="same",
            groups=dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.transpose(1, 2)
        y = self.conv(y)
        y = y.transpose(1, 2)
        return y


class StateSpaceMixerModule(nn.Module):
    def __init__(
        self, dim: int, d_state: Optional[int] = None, expand: Optional[float] = None
    ) -> None:
        super().__init__()
        base_state = d_state or max(4, dim // 4)
        expand = expand or 1.0
        state_dim = max(1, int(base_state * expand))
        self.state_dim = state_dim
        self.A = nn.Parameter(torch.zeros(state_dim))
        self.B = nn.Linear(dim, state_dim)
        self.C = nn.Linear(state_dim, dim)
        self.D = nn.Linear(dim, dim)
        self._last_state: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        state = torch.zeros(batch, self.state_dim, device=x.device, dtype=x.dtype)
        A = torch.exp(self.A).view(1, self.state_dim)
        outputs = []
        for t in range(seq_len):
            u = x[:, t, :]
            state = torch.tanh(state * A + self.B(u))
            y = self.C(state) + self.D(u)
            outputs.append(y.unsqueeze(1))
        self._last_state = state.detach()
        return torch.cat(outputs, dim=1)


class ValueGLUWrapper(nn.Module):
    def __init__(self, module: nn.Module, dim: int) -> None:
        super().__init__()
        self.module = module
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.module(x)
        gate = torch.sigmoid(self.proj(x))
        return base * gate


def build_single_mixer(mixer: Mixer, dim: int, causal: bool, max_seq_len: int) -> nn.Module:
    if mixer.kind == "Attention":
        module = AttentionMixerModule(mixer, dim, causal, max_seq_len)
    elif mixer.kind == "Retention":
        module = RetentionMixerModule(dim, chunk=mixer.chunk, mode=mixer.mode)
    elif mixer.kind == "SSM":
        module = StateSpaceMixerModule(dim, d_state=mixer.d_state, expand=mixer.expand)
    elif mixer.kind == "LongConv":
        kernel = mixer.kernel_len or 16
        module = ConvMixerModule(dim, kernel)
    else:
        raise DSLValidationError(f"Unsupported mixer kind '{mixer.kind}' for builder")

    if getattr(mixer, "value_glu", False):
        module = ValueGLUWrapper(module, dim)
    return module


class ParallelMixerModule(nn.Module):
    def __init__(self, mix_unit: MixUnit, dim: int, causal: bool, max_seq_len: int) -> None:
        super().__init__()
        assert mix_unit.choices, "Parallel mixer requires choices"
        self.mixers = nn.ModuleList(
            [build_single_mixer(choice, dim, causal, max_seq_len) for choice in mix_unit.choices]
        )
        self.merge = mix_unit.merge or "Add"
        if self.merge == "WeightedAdd":
            self.weights = nn.Parameter(torch.ones(len(self.mixers)))
        elif self.merge == "Concat":
            self.proj = nn.Linear(dim * len(self.mixers), dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [mixer(x) for mixer in self.mixers]
        if self.merge == "Add":
            return torch.stack(outputs, dim=0).sum(dim=0) / len(outputs)
        if self.merge == "WeightedAdd":
            weights = torch.softmax(self.weights, dim=0)
            stacked = torch.stack(outputs, dim=0)
            return (stacked * weights.view(-1, 1, 1, 1)).sum(dim=0)
        if self.merge == "Concat":
            concat = torch.cat(outputs, dim=-1)
            return self.proj(concat)
        raise DSLValidationError(f"Unknown merge mode '{self.merge}'")


class RouterMixerModule(nn.Module):
    def __init__(self, mix_unit: MixUnit, dim: int, causal: bool, max_seq_len: int) -> None:
        super().__init__()
        assert mix_unit.choices, "Router mixer requires choices"
        self.mixers = nn.ModuleList(
            [build_single_mixer(choice, dim, causal, max_seq_len) for choice in mix_unit.choices]
        )
        self.router_proj = nn.Linear(dim, len(self.mixers))
        self.temperature = (mix_unit.router.temp if mix_unit.router else 1.0) or 1.0
        self.topk = mix_unit.router.topk if mix_unit.router and mix_unit.router.topk else None
        self.balance = float((mix_unit.router.balance if mix_unit.router else 0.0) or 0.0)
        # stats accumulators
        self.register_buffer("_usage_sum", torch.zeros(len(self.mixers)), persistent=False)
        self.register_buffer("_entropy_sum", torch.zeros(1), persistent=False)
        self._usage_count: int = 0
        self._collapsed: bool = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = self.router_proj(x) / self.temperature
        if self.topk and self.topk < scores.size(-1):
            topk_vals, topk_idx = torch.topk(scores, self.topk, dim=-1)
            mask = torch.full_like(scores, float("-inf"))
            mask.scatter_(-1, topk_idx, topk_vals)
            scores = mask
        weights = torch.softmax(scores, dim=-1)
        # accumulate stats
        with torch.no_grad():
            mean_w = weights.mean(dim=(0, 1))  # average over batch and time -> (num_mixers,)
            self._usage_sum += mean_w.detach().to(self._usage_sum.dtype)
            ent = -(mean_w * (mean_w.clamp_min(1e-8)).log()).sum()
            self._entropy_sum += ent.unsqueeze(0)
            self._usage_count += 1
            self._collapsed = bool(mean_w.max().item() > 0.95)
        outputs = torch.stack([mixer(x) for mixer in self.mixers], dim=-2)  # (..., num_mixers, dim)
        weights = weights.unsqueeze(-1)
        mixed = (outputs * weights).sum(dim=-2)
        # Aux loss: encourage balanced routing (maximize entropy)
        if self.balance > 0.0:
            with torch.no_grad():
                mean_w = (weights.squeeze(-1)).mean(dim=(0, 1))  # (num_mixers,)
            balance_loss = -(mean_w * (mean_w.clamp_min(1e-8)).log()).sum()
            # stash auxiliary loss on module for trainer to consume
            setattr(self, "_aux_loss", balance_loss * self.balance)
        return mixed

    def router_stats(self) -> Optional[dict]:
        if self._usage_count == 0:
            return None
        usage = (self._usage_sum / float(self._usage_count)).detach().cpu().tolist()
        entropy = self._entropy_sum.item() / float(self._usage_count)
        return {"usage": usage, "entropy": entropy, "collapsed": self._collapsed}


def build_mix_module(mix_unit: MixUnit, dim: int, causal: bool, max_seq_len: int) -> nn.Module:
    if mix_unit.kind == "single":
        mixer = mix_unit.mixer or mix_unit.choices[0]
        return build_single_mixer(mixer, dim, causal, max_seq_len)
    if mix_unit.kind == "par":
        return ParallelMixerModule(mix_unit, dim, causal, max_seq_len)
    if mix_unit.kind == "route":
        return RouterMixerModule(mix_unit, dim, causal, max_seq_len)
    raise DSLValidationError(f"Unsupported mix_unit kind '{mix_unit.kind}'")


class GenericTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        mix_unit: MixUnit,
        norm_kind: str,
        ffn_mult: float,
        ffn_act: str,
        max_seq_len: int,
        causal: bool,
    ) -> None:
        super().__init__()
        self.norm1 = build_norm(norm_kind, dim)
        self.mixer = build_mix_module(mix_unit, dim, causal, max_seq_len)
        self.norm2 = build_norm(norm_kind, dim)
        self.ff = build_ffn(dim, ffn_mult, ffn_act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = residual + self.mixer(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class GenericTransformerStack(nn.Module):
    def __init__(
        self,
        dim: int,
        n_layers: int,
        mix_unit: MixUnit,
        norm_kind: str,
        ffn_mult: float,
        ffn_act: str,
        max_seq_len: int,
        causal: bool,
        recurrence: Optional[RecurrenceConfig] = None,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                GenericTransformerBlock(
                    dim=dim,
                    mix_unit=mix_unit,
                    norm_kind=norm_kind,
                    ffn_mult=ffn_mult,
                    ffn_act=ffn_act,
                    max_seq_len=max_seq_len,
                    causal=causal,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = build_norm(norm_kind, dim)
        self.recurrence = recurrence
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.recurrence:
            for block in self.blocks:
                x = block(x)
            return self.norm(x)

        idx = 0
        hidden = x
        for _ in range(self._rec_prelude):
            hidden = self.blocks[idx](hidden)
            idx += 1
        anchor = hidden
        state = self._inject_recurrence_noise(anchor)
        loops = self._resolve_recurrence_loops()
        for _ in range(loops):
            loop_state = state
            for offset in range(self._rec_body):
                loop_state = self.blocks[idx + offset](loop_state)
            state = self._apply_recurrence_adapter(loop_state, anchor)
        idx += self._rec_body
        hidden = state
        for _ in range(self._rec_coda):
            hidden = self.blocks[idx](hidden)
            idx += 1
        return self.norm(hidden)

    def _resolve_recurrence_loops(self) -> int:
        if not self.recurrence or not self.recurrence.loops:
            return 1
        loops = self.recurrence.loops
        base = loops.train if self.training else (loops.eval or loops.train)
        return max(1, int(base))

    def _apply_recurrence_adapter(self, state: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        cfg = self.recurrence
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
        cfg = self.recurrence
        if not cfg or cfg.noise_std <= 0:
            return anchor
        noise = torch.randn_like(anchor) * cfg.noise_std
        return anchor + noise


class TransformerStage(nn.Module):
    def __init__(
        self,
        name: str,
        module: ModuleSpec,
        arch_cfg: DSLConfig,
        max_seq_len: int,
    ) -> None:
        super().__init__()
        self.name = name
        dim = module.d_model or arch_cfg.arch.d_model
        mix_unit = _resolve_mix_unit(module, arch_cfg.arch.mix_unit)
        attention = _pick_attention_mixer(mix_unit)
        heads = attention.heads or 1
        ffn_mult, ffn_act = _resolve_ffn(module, arch_cfg.arch.ffn)
        norm_kind = module.norm or arch_cfg.arch.norm
        causal = True
        if module.params and isinstance(module.params, dict):
            causal = module.params.get("causal", True)
        self.stack = GenericTransformerStack(
            dim=dim,
            n_layers=module.n_layers or 1,
            mix_unit=mix_unit,
            norm_kind=norm_kind,
            ffn_mult=ffn_mult,
            ffn_act=ffn_act,
            max_seq_len=max_seq_len,
            causal=causal,
            recurrence=module.recurrence,
        )

    def forward(self, hidden: torch.Tensor, *_: torch.Tensor) -> torch.Tensor:
        if hidden is None:
            raise DSLValidationError(f"Stage '{self.name}' requires hidden input")
        return self.stack(hidden)


class LatentSamplerStage(nn.Module):
    def __init__(self, name: str, dim: int) -> None:
        super().__init__()
        self.name = name
        self.project = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, dim),
        )

    def forward(self, hidden: torch.Tensor, *_: torch.Tensor) -> torch.Tensor:
        if hidden is None:
            raise DSLValidationError(f"Stage '{self.name}' requires hidden input")
        return self.project(hidden)


class ReadoutStage(nn.Module):
    def __init__(self, name: str, dim: int, vocab_size: int) -> None:
        super().__init__()
        self.name = name
        self.linear = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, hidden: torch.Tensor, *_: torch.Tensor) -> torch.Tensor:
        if hidden is None:
            raise DSLValidationError(f"Stage '{self.name}' requires hidden input")
        return self.linear(hidden)


class PipelineModel(nn.Module):
    def __init__(
        self,
        cfg: DSLConfig,
        vocab_size: int,
    ) -> None:
        super().__init__()
        arch = cfg.arch
        if not arch.modules or not arch.pipeline:
            raise DSLValidationError(
                "Pipeline configuration requires modules and pipeline definitions"
            )
        self.max_seq_len = cfg.train.ctx_len
        self.vocab_size = vocab_size
        self.dim = arch.d_model
        self.stage_order: List[str] = []
        self.stage_map = nn.ModuleDict()
        self.stage_specs: Dict[str, PipelineStage] = {}

        modules = arch.modules
        for stage in arch.pipeline:
            self.stage_order.append(stage.name)
            self.stage_specs[stage.name] = stage
            module_kind = stage.kind
            module_spec: Optional[ModuleSpec] = None
            if stage.module:
                module_spec = modules.get(stage.module)
                if module_spec is None:
                    raise DSLValidationError(
                        f"Pipeline stage '{stage.name}' references unknown module '{stage.module}'"
                    )
            if module_kind is None and module_spec is not None:
                module_kind = module_spec.kind
            if module_kind is None:
                module_kind = "module"
            if module_kind == "embedding":
                dim = module_spec.d_model if module_spec and module_spec.d_model else arch.d_model
                runner = EmbeddingStage(stage.name, vocab_size, dim, cfg.train.ctx_len)
            elif module_kind == "latent_sampler":
                dim = module_spec.d_model if module_spec and module_spec.d_model else arch.d_model
                runner = LatentSamplerStage(stage.name, dim)
            elif module_kind == "readout":
                dim = module_spec.d_model if module_spec and module_spec.d_model else arch.d_model
                runner = ReadoutStage(stage.name, dim, vocab_size)
            else:
                if module_spec is None:
                    raise DSLValidationError(
                        f"Pipeline stage '{stage.name}' missing module reference"
                    )
                runner = TransformerStage(stage.name, module_spec, cfg, cfg.train.ctx_len)
            self.stage_map[stage.name] = runner

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        activations: Dict[str, torch.Tensor] = {}
        last_output: Optional[torch.Tensor] = None

        for stage_name in self.stage_order:
            stage = self.stage_specs[stage_name]
            runner = self.stage_map[stage_name]

            repeats = stage.repeat or 1
            output = last_output
            for _ in range(repeats):
                if isinstance(runner, EmbeddingStage):
                    output = runner(input_ids)
                else:
                    sources: List[torch.Tensor] = []
                    for name in stage.inputs or []:
                        if name not in activations:
                            raise DSLValidationError(
                                f"Stage '{stage_name}' expected input '{name}' not available"
                            )
                        sources.append(activations[name])
                    for name in stage.kv_from or []:
                        if name not in activations:
                            raise DSLValidationError(
                                f"Stage '{stage_name}' expected kv_from '{name}' not available"
                            )
                        sources.append(activations[name])
                    if stage.mem_from:
                        for name in stage.mem_from:
                            if name not in activations:
                                raise DSLValidationError(
                                    f"Stage '{stage_name}' expected mem_from '{name}' not available"
                                )
                            sources.append(activations[name])
                    if not sources:
                        if output is None:
                            raise DSLValidationError(
                                f"Stage '{stage_name}' has no available inputs"
                            )
                        source_tensor = output
                    else:
                        if len(sources) == 1:
                            source_tensor = sources[0]
                        else:
                            source_tensor = torch.stack(sources, dim=0).mean(dim=0)
                    output = runner(source_tensor)
            activations[stage_name] = output
            last_output = output

        if last_output is None:
            raise DSLValidationError("Pipeline produced no output")
        return last_output


def build_pipeline_model(
    cfg: DSLConfig,
    *,
    vocab_size: Optional[int] = None,
) -> Tuple[nn.Module, BuildMetadata]:
    vocab = vocab_size or cfg.train.vocab_size or 256
    model = PipelineModel(cfg, vocab)
    meta = BuildMetadata(
        vocab_size=vocab,
        max_seq_len=cfg.train.ctx_len,
        dim=model.dim,
        n_layers=cfg.arch.n_layers,
        extras=summarize_architecture(cfg),
    )
    return model, meta
