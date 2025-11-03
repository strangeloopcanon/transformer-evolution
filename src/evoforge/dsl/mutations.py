from __future__ import annotations

import random
from typing import Callable, Iterable, List, Optional

from .errors import DSLValidationError

from .models import DSLConfig, MixUnit, Mixer, ModuleSpec, PipelineStage
from .validators import run_additional_checks


MutationFn = Callable[[DSLConfig], Optional[DSLConfig]]


def _clone(cfg: DSLConfig) -> DSLConfig:
    # model_copy with deep=True works on Pydantic v2.
    return cfg.model_copy(deep=True)


def _iter_attention_mixers(cfg: DSLConfig) -> Iterable[Mixer]:
    mu = cfg.arch.mix_unit
    if mu.kind == "single":
        if mu.mixer and mu.mixer.kind == "Attention":
            yield mu.mixer
    else:
        if not mu.choices:
            return
        for choice in mu.choices:
            if choice.kind == "Attention":
                yield choice


def _heads_mutation(cfg: DSLConfig, delta: int) -> Optional[DSLConfig]:
    clone = _clone(cfg)
    changed = False
    for mixer in _iter_attention_mixers(clone):
        heads = mixer.heads or 4
        new_heads = max(1, heads + delta)
        if new_heads != heads:
            mixer.heads = new_heads
            if mixer.groups and mixer.groups > new_heads:
                mixer.groups = new_heads
            changed = True
    if not changed:
        return None
    run_additional_checks(clone)
    return clone


def _groups_mutation(cfg: DSLConfig) -> Optional[DSLConfig]:
    clone = _clone(cfg)
    changed = False
    for mixer in _iter_attention_mixers(clone):
        heads = mixer.heads or 4
        current = mixer.groups or heads
        if current == heads and heads > 1:
            mixer.groups = max(1, heads // 2)
        elif current == 1 and heads > 1:
            mixer.groups = min(heads, 2)
        else:
            mixer.groups = heads
        changed = True
    if not changed:
        return None
    run_additional_checks(clone)
    return clone


def _rope_dims_mutation(cfg: DSLConfig) -> Optional[DSLConfig]:
    pos = cfg.arch.pos
    if pos.kind != "rope" or not pos.rope:
        return None
    dims = pos.rope.dims or 64
    next_dims = 32 if dims > 32 else 64
    clone = _clone(cfg)
    clone.arch.pos.rope = clone.arch.pos.rope or cfg.arch.pos.rope
    clone.arch.pos.rope.dims = next_dims
    run_additional_checks(clone)
    return clone


def _router_temp_mutation(cfg: DSLConfig) -> Optional[DSLConfig]:
    mu = cfg.arch.mix_unit
    if mu.router is None:
        return None
    clone = _clone(cfg)
    router = clone.arch.mix_unit.router
    assert router is not None
    router.temp = (router.temp or 1.0) * 0.8
    run_additional_checks(clone)
    return clone


def _router_topk_mutation(cfg: DSLConfig) -> Optional[DSLConfig]:
    mu = cfg.arch.mix_unit
    if mu.kind != "route" or not mu.router:
        return None
    clone = _clone(cfg)
    router = clone.arch.mix_unit.router
    if router.topk is None:
        router.topk = max(1, (clone.arch.mix_unit.mixer.heads or 2) // 2)
    else:
        router.topk = max(1, router.topk - 1)
    run_additional_checks(clone)
    return clone


def _local_window_mutation(cfg: DSLConfig) -> Optional[DSLConfig]:
    clone = _clone(cfg)
    changed = False
    for mixer in _iter_attention_mixers(clone):
        stencil = getattr(mixer, "stencil", None)
        if stencil and getattr(stencil, "kind", None) in {"local", "sliding"}:
            window = getattr(stencil, "window", None) or 256
            new_window = max(32, int(window * 1.5))
            stencil.window = new_window
            changed = True
    if not changed:
        return None
    run_additional_checks(clone)
    return clone


def _module_ffn_mutation(cfg: DSLConfig) -> Optional[DSLConfig]:
    modules = cfg.arch.modules
    if not modules:
        return None
    clone = _clone(cfg)
    for module in clone.arch.modules.values():
        if module.ffn:
            module.ffn.mult = round(module.ffn.mult * 1.1, 2)
            run_additional_checks(clone)
            return clone
    return None


def _pipeline_rewire_mutation(cfg: DSLConfig) -> Optional[DSLConfig]:
    if not cfg.arch.pipeline or len(cfg.arch.pipeline) < 3:
        return None
    clone = _clone(cfg)
    stages = clone.arch.pipeline
    # Swap last two stages if possible
    stages[-1], stages[-2] = stages[-2], stages[-1]
    run_additional_checks(clone)
    return clone


def _add_cross_skip_mutation(cfg: DSLConfig) -> Optional[DSLConfig]:
    if not cfg.arch.pipeline or len(cfg.arch.pipeline) < 3:
        return None
    clone = _clone(cfg)
    stages = {s.name: s for s in clone.arch.pipeline}
    # pick a later stage (not embedding) and add an input from an earlier one
    later = clone.arch.pipeline[-2]
    earlier = clone.arch.pipeline[1]
    if later.kind == "embedding":
        return None
    inputs = list(later.inputs) if later.inputs else []
    if earlier.name not in inputs:
        inputs.append(earlier.name)
        later.inputs = inputs
        run_additional_checks(clone)
        return clone
    return None


def _ensure_modules(clone: DSLConfig) -> None:
    if clone.arch.modules is None:
        clone.arch.modules = {}


def _ensure_pipeline(clone: DSLConfig) -> None:
    if clone.arch.pipeline is None:
        clone.arch.pipeline = []


def _make_retention_module(clone: DSLConfig, heads: int) -> ModuleSpec:
    retention = Mixer(kind="Retention", heads=heads, chunk=512, mode="parallel")
    mix_unit = MixUnit(kind="single", mixer=retention)
    return ModuleSpec(
        kind="transformer",
        d_model=clone.arch.d_model,
        n_layers=1,
        mix_unit=mix_unit,
        ffn=clone.arch.ffn,
        norm=clone.arch.norm,
        pos=clone.arch.pos,
    )


def _add_memory_stage(cfg: DSLConfig) -> Optional[DSLConfig]:
    if not cfg.arch.pipeline:
        return None
    clone = _clone(cfg)
    _ensure_modules(clone)
    _ensure_pipeline(clone)
    if "memory_auto" in clone.arch.modules:
        return None
    heads = clone.arch.mix_unit.mixer.heads if clone.arch.mix_unit.mixer else 4
    clone.arch.modules["memory_auto"] = _make_retention_module(clone, heads)
    stage = PipelineStage(
        name="memory_auto",
        module="memory_auto",
        inputs=[clone.arch.pipeline[0].name],
        kv_from=[clone.arch.pipeline[0].name],
    )
    insert_idx = 1 if len(clone.arch.pipeline) > 1 else 0
    clone.arch.pipeline.insert(insert_idx, stage)
    run_additional_checks(clone)
    return clone


def _swap_to_route_mutation(cfg: DSLConfig) -> Optional[DSLConfig]:
    mu = cfg.arch.mix_unit
    if mu.kind != "single" or not mu.mixer:
        return None
    clone = _clone(cfg)
    base = clone.arch.mix_unit.mixer
    choices = [base.model_copy(deep=True)]
    choices.append(Mixer(kind="Retention", heads=base.heads, chunk=512, mode="parallel"))
    choices.append(Mixer(kind="SSM", heads=base.heads, d_state=16, expand=1.5))
    clone.arch.mix_unit = MixUnit(kind="route", choices=choices, merge="Add")
    run_additional_checks(clone)
    return clone


def _toggle_latent_sampler(cfg: DSLConfig) -> Optional[DSLConfig]:
    clone = _clone(cfg)
    _ensure_modules(clone)
    _ensure_pipeline(clone)
    if clone.arch.modules and "latent_sampler" in clone.arch.modules:
        clone.arch.modules.pop("latent_sampler")
        clone.arch.pipeline = [stage for stage in clone.arch.pipeline if stage.module != "latent_sampler"]
    else:
        heads = clone.arch.mix_unit.mixer.heads if clone.arch.mix_unit.mixer else 4
        attn = Mixer(kind="Attention", heads=heads)
        mix_unit = MixUnit(kind="single", mixer=attn)
        clone.arch.modules["latent_sampler"] = ModuleSpec(kind="latent_sampler", mix_unit=mix_unit)
        # Insert after the first non-embedding stage if present, else after stage 0
        insert_after = 0
        for i, st in enumerate(clone.arch.pipeline):
            if st.kind and st.kind != "embedding":
                insert_after = i
                break
        # Wire latent sampler to the previous stage output
        input_stage = clone.arch.pipeline[insert_after].name if clone.arch.pipeline else None
        latent_stage = PipelineStage(
            name="latent_sampler_auto",
            module="latent_sampler",
            inputs=[input_stage] if input_stage else None,
        )
        clone.arch.pipeline.insert(insert_after + 1, latent_stage)
    run_additional_checks(clone)
    return clone


MUTATORS: List[MutationFn] = [
    lambda cfg: _heads_mutation(cfg, +2),
   lambda cfg: _heads_mutation(cfg, -1),
   _groups_mutation,
   _rope_dims_mutation,
   _router_temp_mutation,
   _router_topk_mutation,
   _local_window_mutation,
   _module_ffn_mutation,
    _pipeline_rewire_mutation,
    _add_cross_skip_mutation,
    _add_memory_stage,
    _swap_to_route_mutation,
    _toggle_latent_sampler,
]


def generate_mutations(cfg: DSLConfig, *, rng: Optional[random.Random] = None) -> List[DSLConfig]:
    """Produce a small set of mutated configs that remain valid."""

    rng = rng or random.Random()
    variants: List[DSLConfig] = []
    for mut in MUTATORS:
        try:
            mutated = mut(cfg)
        except DSLValidationError:
            continue
        if mutated is None:
            continue
        variants.append(mutated)

    rng.shuffle(variants)
    return variants


def pick_random_mutation(cfg: DSLConfig, *, rng: Optional[random.Random] = None) -> DSLConfig:
    variants = generate_mutations(cfg, rng=rng)
    if not variants:
        raise DSLValidationError("No valid mutations available for config")
    rng = rng or random.Random()
    return rng.choice(variants)
