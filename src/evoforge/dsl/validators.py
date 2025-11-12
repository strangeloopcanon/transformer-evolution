from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from .errors import DSLValidationError
from .models import (
    Arch,
    Cond,
    DSLConfig,
    MixUnit,
    Mixer,
    ModuleSpec,
    PipelineStage,
    RecurrenceSchedule,
)


def _iter_mixers(mix_unit: MixUnit) -> Iterable[Mixer]:
    if mix_unit.kind == "single":
        if mix_unit.mixer is not None:
            yield mix_unit.mixer
    else:
        if mix_unit.choices:
            for choice in mix_unit.choices:
                yield choice


def _ensure_mix_unit_shape(arch: Arch) -> None:
    mu = arch.mix_unit
    if mu.kind == "single":
        if mu.mixer is None:
            raise DSLValidationError("mix_unit.kind=single requires 'mixer'")
    else:
        if not mu.choices:
            raise DSLValidationError(f"mix_unit.kind={mu.kind} requires non-empty 'choices'")
        if mu.kind == "route" and mu.router is None:
            raise DSLValidationError("mix_unit.kind=route requires 'router'")
        if mu.kind == "par" and mu.merge is None:
            raise DSLValidationError("mix_unit.kind=par requires 'merge'")


def _validate_attention_rules(arch: Arch) -> None:
    has_attention = False
    for mixer in _iter_mixers(arch.mix_unit):
        if mixer.kind != "Attention":
            continue
        has_attention = True
        if mixer.groups is not None and mixer.heads is not None:
            if mixer.groups > mixer.heads:
                raise DSLValidationError("Attention mixer requires groups <= heads")
        if mixer.mode and mixer.mode != "parallel":
            raise DSLValidationError("Attention mixer cannot declare mode other than 'parallel'")
        if mixer.projection and mixer.projection.type == "low_rank":
            if mixer.projection.rank is None or mixer.projection.rank <= 0:
                raise DSLValidationError("Low-rank projection requires positive 'rank'")
        if mixer.value_glu and mixer.kind != "Attention":
            raise DSLValidationError("value_glu only valid for Attention mixers")
        if mixer.stencil and mixer.stencil.kind == "ring":
            if mixer.stencil.block is None:
                raise DSLValidationError("Ring stencil requires 'block' size")
        if mixer.softmax and mixer.softmax.type == "kernel":
            kernel = mixer.softmax.kernel
            if kernel is None or kernel.features is None:
                raise DSLValidationError("Kernel softmax requires kernel.features")
    if arch.pos.kind == "alibi" and not has_attention:
        raise DSLValidationError("Positional encoding 'alibi' requires an Attention mixer")


def _validate_recurrent_rules(arch: Arch) -> None:
    for mixer in _iter_mixers(arch.mix_unit):
        if mixer.mode == "recurrent" and mixer.kind not in {"Retention", "SSM"}:
            raise DSLValidationError("Only Retention or SSM mixers may set mode='recurrent'")


def _validate_cond_rules(cond: Optional[Cond]) -> None:
    if cond is None or not cond.ops:
        return
    for op in cond.ops:
        if op.op == "lora" and (op.r is None or op.r <= 0):
            raise DSLValidationError("Cond op 'lora' requires positive rank 'r'")


def _validate_mixer_general(arch: Arch) -> None:
    for mixer in _iter_mixers(arch.mix_unit):
        if mixer.value_glu and mixer.kind != "Attention":
            raise DSLValidationError("value_glu only supported for Attention mixers")
        if mixer.stencil and mixer.stencil.kind == "cross":
            if mixer.stencil.query is None or mixer.stencil.key is None:
                raise DSLValidationError("Cross stencil requires 'query' and 'key'")


def _validate_kv_policy(arch: Arch) -> None:
    kv = arch.kv_policy
    if kv is None:
        return
    if kv.cache == "latent" and kv.latent is None:
        raise DSLValidationError("cache='latent' requires latent configuration")
    if kv.latent is not None and kv.cache != "latent":
        raise DSLValidationError("latent KV config requires cache='latent'")


def _validate_residual(arch: Arch) -> None:
    res = arch.residual
    if res is None:
        return
    if res.kind == "dual":
        if not (res.pre_ln and res.post_ln):
            raise DSLValidationError("ResiDual requires both pre_ln and post_ln enabled")
    if res.kind == "deepnet" and res.scale is None:
        raise DSLValidationError("DeepNet residual requires 'scale'")


def _validate_depth_router(arch: Arch) -> None:
    dr = arch.depth_router
    if dr is None:
        return
    if dr.kind != "none" and dr.budget is None:
        raise DSLValidationError("Depth router with kind != 'none' requires budget")


def _validate_budget_dict(name: str, budget: Optional[Dict[str, float]]) -> None:
    if budget is None:
        return
    for key, value in budget.items():
        if value is None:
            continue
        if not isinstance(value, (int, float)):
            raise DSLValidationError(f"{name} budget '{key}' must be numeric")
        if value < 0:
            raise DSLValidationError(f"{name} budget '{key}' must be non-negative")


def _validate_module_budgets(modules: Optional[Dict[str, ModuleSpec]]) -> None:
    if not modules:
        return
    for module_name, module in modules.items():
        _validate_budget_dict(f"module '{module_name}'", module.budget)


def _validate_pipeline_budgets(pipeline: Optional[List[PipelineStage]]) -> None:
    if not pipeline:
        return
    for stage in pipeline:
        _validate_budget_dict(f"stage '{stage.name}'", stage.budget)


def _validate_hierarchy(arch: Arch) -> None:
    hierarchy = arch.hierarchy
    if hierarchy is None:
        return
    if not hierarchy.levels:
        raise DSLValidationError("Hierarchy requires at least one level")


def _validate_modules(modules: Optional[Dict[str, ModuleSpec]]) -> None:
    if not modules:
        return
    for name, module in modules.items():
        if not isinstance(module, ModuleSpec):
            continue
        if module.kind == "transformer":
            if (
                module.mix_unit is None
                or module.ffn is None
                or module.norm is None
                or module.pos is None
            ):
                raise DSLValidationError(f"Transformer module '{name}' missing core fields")
            if module.recurrence:
                _validate_module_recurrence(name, module)


def _validate_pipeline(arch: Arch) -> None:
    pipeline = arch.pipeline
    if not pipeline:
        return
    modules = arch.modules or {}
    seen: List[str] = []
    for stage in pipeline:
        if stage.name in seen:
            raise DSLValidationError(f"Duplicate pipeline stage name '{stage.name}'")
        seen.append(stage.name)
        if stage.module and stage.module not in modules:
            raise DSLValidationError(
                f"Pipeline stage '{stage.name}' references missing module '{stage.module}'"
            )
        if not stage.module and not stage.kind:
            raise DSLValidationError(
                f"Pipeline stage '{stage.name}' requires either 'module' or 'kind'"
            )
        for ref_list in (stage.inputs or []) + (stage.kv_from or []) + (stage.mem_from or []):
            if ref_list not in seen:
                raise DSLValidationError(
                    f"Pipeline stage '{stage.name}' references '{ref_list}' before it is defined"
                )


def _validate_module_recurrence(name: str, module: ModuleSpec) -> None:
    rec = module.recurrence
    if rec is None:
        return
    if module.n_layers is None:
        raise DSLValidationError(f"Module '{name}' with recurrence must set n_layers")
    total = rec.prelude + rec.body + rec.coda
    if total != module.n_layers:
        raise DSLValidationError(
            f"Module '{name}' recurrence layers ({total}) must equal module.n_layers ({module.n_layers})"
        )
    loops = rec.loops
    if loops:
        train_loops = loops.train
        eval_loops = loops.eval if loops.eval else train_loops
        if train_loops < 1 or eval_loops < 1:
            raise DSLValidationError(
                f"Module '{name}' recurrence loops must be >= 1 (got train={train_loops}, eval={eval_loops})"
            )
        if loops.schedule:
            _validate_recurrence_schedule(loops.schedule)


def _validate_recurrence(arch: Arch) -> None:
    recurrence = arch.recurrence
    if recurrence is None:
        return
    if arch.pipeline:
        raise DSLValidationError("recurrence is not supported alongside pipeline definitions yet")
    total = recurrence.prelude + recurrence.body + recurrence.coda
    if total != arch.n_layers:
        raise DSLValidationError("recurrence layers must sum to arch.n_layers")
    loops = recurrence.loops
    train_loops = loops.train if loops else 1
    eval_loops = loops.eval if loops and loops.eval else train_loops
    if train_loops < 1 or eval_loops < 1:
        raise DSLValidationError("recurrence loops must be >= 1")
    schedule = loops.schedule if loops else None
    if schedule:
        _validate_recurrence_schedule(schedule)


def _validate_recurrence_schedule(schedule: RecurrenceSchedule) -> None:
    if schedule.kind != "fixed":
        if schedule.mean is None or schedule.mean <= 0:
            raise DSLValidationError("recurrence schedule requires positive 'mean'")
        if schedule.sigma is not None and schedule.sigma < 0:
            raise DSLValidationError("recurrence schedule sigma must be >= 0")
    if schedule.min is not None and schedule.max is not None:
        if schedule.min > schedule.max:
            raise DSLValidationError("recurrence schedule min cannot exceed max")


def run_additional_checks(cfg: DSLConfig) -> None:
    arch = cfg.arch
    _ensure_mix_unit_shape(arch)
    _validate_attention_rules(arch)
    _validate_recurrent_rules(arch)
    _validate_cond_rules(arch.cond)
    _validate_mixer_general(arch)
    _validate_kv_policy(arch)
    _validate_residual(arch)
    _validate_depth_router(arch)
    _validate_recurrence(arch)
    _validate_hierarchy(arch)
    _validate_modules(arch.modules)
    _validate_pipeline(arch)
    _validate_module_budgets(arch.modules)
    _validate_pipeline_budgets(arch.pipeline)
    _validate_budget_dict("train", cfg.train.budget)
