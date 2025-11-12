from __future__ import annotations

import random
from typing import Callable, Iterable, List, Optional

from .errors import DSLValidationError

from .models import (
    DSLConfig,
    MixUnit,
    Mixer,
    ModuleSpec,
    PipelineStage,
    Router,
    StencilConfig,
    PosConfig,
    PosRope,
    RopeScaling,
    KVPolicy,
    HierarchyConfig,
    HierarchyLevel,
    DepthRouter,
    RecurrenceConfig,
    RecurrenceLoops,
    RecurrenceSchedule,
)
from .validators import run_additional_checks


MutationFn = Callable[[DSLConfig], Optional[DSLConfig]]


def _clone(cfg: DSLConfig) -> DSLConfig:
    # model_copy with deep=True works on Pydantic v2.
    return cfg.model_copy(deep=True)


def _random_recurrence_config(total_layers: int) -> RecurrenceConfig:
    total_layers = max(1, total_layers)
    if total_layers == 1:
        prelude = 0
        body = 1
    else:
        prelude = random.randint(0, total_layers - 1)
        remaining = total_layers - prelude
        body = random.randint(1, remaining)
    coda = max(0, total_layers - prelude - body)
    adapter = random.choice(["identity", "residual", "concat_linear"])
    noise_std = 0.0
    if random.random() < 0.4:
        noise_std = round(random.uniform(0.01, 0.08), 3)
    train_loops = random.choice([1, 2, 3, 4])
    eval_loops = train_loops + random.choice([0, 1, 2])
    loops = RecurrenceLoops(train=train_loops, eval=eval_loops if eval_loops != train_loops else None)
    if random.random() < 0.35:
        loops.schedule = RecurrenceSchedule(
            kind="poisson_lognormal",
            mean=round(random.uniform(1.5, 3.5), 2),
            sigma=round(random.uniform(0.2, 0.6), 2),
            min=1,
            max=max(train_loops * 2, 4),
            curriculum=random.choice(["linear", "sqrt"]),
            warmup_steps=random.randint(200, 2000),
            backprop=random.randint(2, 8),
        )
    return RecurrenceConfig(
        prelude=prelude,
        body=body,
        coda=coda,
        adapter=adapter,
        noise_std=noise_std,
        loops=loops,
    )


def _reshuffle_recurrence(rec: RecurrenceConfig, total_layers: Optional[int] = None) -> None:
    total = total_layers or (rec.prelude + rec.body + rec.coda)
    new_cfg = _random_recurrence_config(total)
    rec.prelude = new_cfg.prelude
    rec.body = new_cfg.body
    rec.coda = new_cfg.coda
    rec.adapter = new_cfg.adapter
    rec.noise_std = new_cfg.noise_std
    rec.loops = new_cfg.loops


def _mutate_recurrence_config(rec: RecurrenceConfig, total_layers: Optional[int] = None) -> None:
    if random.random() < 0.5:
        _reshuffle_recurrence(rec, total_layers=total_layers)
        return
    loops = rec.loops or RecurrenceLoops(train=1)
    loops.train = max(1, loops.train + random.choice([-1, 0, 1]))
    if loops.eval is None or random.random() < 0.4:
        loops.eval = max(1, loops.train + random.choice([0, 1, 2]))
    else:
        loops.eval = max(1, loops.eval + random.choice([-1, 0, 1]))
    if random.random() < 0.3:
        rec.adapter = random.choice(["identity", "residual", "concat_linear"])
    if random.random() < 0.3:
        rec.noise_std = round(max(0.0, rec.noise_std + random.choice([-0.02, 0.0, 0.02])), 3)
    rec.loops = loops


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


def _stencil_toggle_mutation(cfg: DSLConfig) -> Optional[DSLConfig]:
    """Flip between stencil families and set sensible defaults.

    Chooses among: full, local, sliding, ring. Adds window/stride/block when needed.
    """
    choices = ["full", "local", "sliding", "ring"]
    clone = _clone(cfg)
    changed = False
    for mixer in _iter_attention_mixers(clone):
        new_kind = random.choice(choices)
        st = mixer.stencil or StencilConfig()
        st.kind = new_kind
        if new_kind == "full":
            st.window = None
            st.stride = None
            st.block = None
        elif new_kind == "local":
            st.window = max(128, int((st.window or 256)))
            st.stride = None
            st.block = None
        elif new_kind == "sliding":
            st.window = max(128, int((st.window or 256)))
            st.stride = max(32, int((st.stride or 64)))
            st.block = None
        elif new_kind == "ring":
            st.block = max(128, int((st.block or 256)))
            st.stride = max(64, int((st.stride or 128)))
            st.window = None
        mixer.stencil = st
        changed = True
    if not changed:
        return None
    run_additional_checks(clone)
    return clone


def _pos_toggle_mutation(cfg: DSLConfig) -> Optional[DSLConfig]:
    """Toggle positional encoding between rope and alibi; tweak rope settings."""
    clone = _clone(cfg)
    pos = clone.arch.pos
    if pos.kind == "rope":
        # Occasionally flip to alibi
        if random.random() < 0.4:
            clone.arch.pos = PosConfig(kind="alibi")
        else:
            rope = pos.rope or PosRope()
            rope.dims = 32 if (rope.dims or 64) > 32 else 64
            rope.theta = 12000.0 if (rope.theta or 10000.0) > 12000 else 25000.0
            if random.random() < 0.5:
                rope.scaling = RopeScaling(type="yarn", factor=1.5)
            pos.rope = rope
            clone.arch.pos = pos
    else:
        # Switch to rope with sensible defaults
        rope = PosRope(theta=12000.0, dims=32)
        if random.random() < 0.5:
            rope.scaling = RopeScaling(type="yarn", factor=1.5)
        clone.arch.pos = PosConfig(kind="rope", rope=rope)
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


def _add_or_mutate_hierarchy(cfg: DSLConfig) -> Optional[DSLConfig]:
    """Add a two-level hierarchy when missing, or perturb existing schedule."""
    clone = _clone(cfg)
    if clone.arch.hierarchy is None:
        levels = [
            HierarchyLevel(every=3, downsample=0.5, up_proj=False),
            HierarchyLevel(every=6, downsample=0.25, up_proj=True),
        ]
        clone.arch.hierarchy = HierarchyConfig(levels=levels)
    else:
        # Perturb levels
        for lvl in clone.arch.hierarchy.levels:
            if random.random() < 0.6:
                lvl.every = max(1, int(lvl.every + random.choice([-1, 0, +1])))
            if random.random() < 0.6 and lvl.downsample is not None:
                lvl.downsample = max(
                    0.25, min(0.75, round(lvl.downsample * random.choice([0.8, 1.0, 1.2]), 2))
                )
            if random.random() < 0.3 and lvl.up_proj is not None:
                lvl.up_proj = not lvl.up_proj
    run_additional_checks(clone)
    return clone


def _add_or_mutate_depth_router(cfg: DSLConfig) -> Optional[DSLConfig]:
    """Inject a token-level depth router or tweak its budget/temperature."""
    clone = _clone(cfg)
    dr = clone.arch.depth_router
    if dr is None or dr.kind == "none":
        clone.arch.depth_router = DepthRouter(
            kind="token",
            budget=round(random.uniform(0.5, 0.8), 2),
            tau=round(random.uniform(0.6, 0.9), 2),
            min_layers=random.choice([2, 3, 4]),
        )
    else:
        if dr.budget is not None and random.random() < 0.7:
            dr.budget = max(0.2, min(0.95, round(dr.budget * random.choice([0.85, 1.0, 1.15]), 2)))
        if dr.tau is not None and random.random() < 0.7:
            dr.tau = max(0.3, min(1.5, round(dr.tau * random.choice([0.8, 1.0, 1.2]), 2)))
        if dr.min_layers is not None and random.random() < 0.5:
            dr.min_layers = max(0, dr.min_layers + random.choice([-1, 0, +1]))
        clone.arch.depth_router = dr
    run_additional_checks(clone)
    return clone


def _toggle_arch_recurrence(cfg: DSLConfig) -> Optional[DSLConfig]:
    if cfg.arch.pipeline:
        return None
    clone = _clone(cfg)
    if clone.arch.recurrence is None:
        clone.arch.recurrence = _random_recurrence_config(clone.arch.n_layers)
    else:
        if random.random() < 0.3:
            clone.arch.recurrence = None
        else:
            _mutate_recurrence_config(clone.arch.recurrence, clone.arch.n_layers)
    run_additional_checks(clone)
    return clone


def _module_recurrence_mutation(cfg: DSLConfig) -> Optional[DSLConfig]:
    modules = cfg.arch.modules
    if not modules:
        return None
    candidates = [
        (name, module)
        for name, module in modules.items()
        if module.kind == "transformer" and module.n_layers
    ]
    if not candidates:
        return None
    clone = _clone(cfg)
    name, module = random.choice(candidates)
    target = clone.arch.modules[name]
    assert target.n_layers is not None
    if target.recurrence is None:
        target.recurrence = _random_recurrence_config(target.n_layers)
    else:
        if random.random() < 0.25:
            target.recurrence = None
        else:
            _mutate_recurrence_config(target.recurrence, target.n_layers)
    run_additional_checks(clone)
    return clone


def _tune_existing_recurrence(cfg: DSLConfig) -> Optional[DSLConfig]:
    targets = []
    if cfg.arch.recurrence is not None:
        targets.append(("arch", None))
    if cfg.arch.modules:
        for name, module in cfg.arch.modules.items():
            if module.kind == "transformer" and module.recurrence is not None:
                targets.append(("module", name))
    if not targets:
        return None
    clone = _clone(cfg)
    target_kind, name = random.choice(targets)
    if target_kind == "arch":
        rec = clone.arch.recurrence
        if rec is None:
            return None
        _mutate_recurrence_config(rec, clone.arch.n_layers)
    else:
        assert name is not None
        module = clone.arch.modules[name]
        if module.recurrence is None or module.n_layers is None:
            return None
        _mutate_recurrence_config(module.recurrence, module.n_layers)
    run_additional_checks(clone)
    return clone


def _toggle_kv_policy(cfg: DSLConfig) -> Optional[DSLConfig]:
    """Add or mutate a global KV policy for the arch."""
    clone = _clone(cfg)
    kv = clone.arch.kv_policy
    if kv is None:
        clone.arch.kv_policy = KVPolicy(
            cache="window", window=random.choice([4096, 6144, 8192]), quant="nf4"
        )
    else:
        # Flip quant or window size
        if random.random() < 0.5:
            kv.quant = random.choice(
                ["nf4", "fp8", "int8", "none"]
            )  # keep variety; validators accept these
        if random.random() < 0.7:
            w = kv.window or 4096
            kv.window = max(1024, min(16384, int(w * random.choice([0.5, 1.0, 1.5]))))
        clone.arch.kv_policy = kv
    run_additional_checks(clone)
    return clone


def _toggle_to_par_mutation(cfg: DSLConfig) -> Optional[DSLConfig]:
    """Switch single -> par with Attention + Retention (+optional SSM); or perturb existing par."""
    mu = cfg.arch.mix_unit
    clone = _clone(cfg)
    heads = None
    if mu.kind == "single" and mu.mixer:
        heads = mu.mixer.heads or 8
        attn = mu.mixer.model_copy(deep=True)
        retn = Mixer(kind="Retention", heads=heads, chunk=512, mode="parallel")
        choices = [attn, retn]
        if random.random() < 0.6:
            choices.append(Mixer(kind="SSM", heads=heads, d_state=16, expand=1.5))
        clone.arch.mix_unit = MixUnit(
            kind="par", choices=choices, merge=random.choice(["Add", "WeightedAdd"])
        )
    elif mu.kind == "par" and mu.choices:
        # Probabilistically add/remove SSM or change merge
        choices = [c.model_copy(deep=True) for c in mu.choices]
        heads = choices[0].heads or 8
        if random.random() < 0.5:
            # toggle SSM presence
            have_ssm = any(c.kind == "SSM" for c in choices)
            if have_ssm:
                choices = [c for c in choices if c.kind != "SSM"]
            else:
                choices.append(Mixer(kind="SSM", heads=heads, d_state=16, expand=1.5))
        merge = mu.merge or "Add"
        if random.random() < 0.5:
            merge = "WeightedAdd" if merge == "Add" else "Add"
        clone.arch.mix_unit = MixUnit(kind="par", choices=choices, merge=merge)
    else:
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
    # router required for 'route'
    router = Router(topk=max(1, (base.heads or 4) // 2), temp=0.7, balance=0.01)
    clone.arch.mix_unit = MixUnit(kind="route", choices=choices, merge="Add", router=router)
    run_additional_checks(clone)
    return clone


def _toggle_latent_sampler(cfg: DSLConfig) -> Optional[DSLConfig]:
    clone = _clone(cfg)
    _ensure_modules(clone)
    _ensure_pipeline(clone)
    if clone.arch.modules and "latent_sampler" in clone.arch.modules:
        clone.arch.modules.pop("latent_sampler")
        clone.arch.pipeline = [
            stage for stage in clone.arch.pipeline if stage.module != "latent_sampler"
        ]
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
    _stencil_toggle_mutation,
    _pos_toggle_mutation,
    _router_temp_mutation,
    _router_topk_mutation,
    _local_window_mutation,
    _add_or_mutate_hierarchy,
    _add_or_mutate_depth_router,
    _toggle_arch_recurrence,
    _module_recurrence_mutation,
    _tune_existing_recurrence,
    _toggle_kv_policy,
    _toggle_to_par_mutation,
    _module_ffn_mutation,
    _pipeline_rewire_mutation,
    _add_cross_skip_mutation,
    _add_memory_stage,
    _swap_to_route_mutation,
    _toggle_latent_sampler,
]


def generate_mutations(cfg: DSLConfig, *, rng: Optional[random.Random] = None) -> List[DSLConfig]:
    """Produce a small set of mutated configs that remain valid."""

    external_rng = rng
    rng = rng or random.Random()
    restore_random_state = False
    global_state = None
    if external_rng is not None:
        restore_random_state = True
        global_state = random.getstate()
        random.seed(external_rng.getrandbits(64))

    base_dump = cfg.model_dump()
    variants: List[DSLConfig] = []
    try:
        for mut in MUTATORS:
            try:
                mutated = mut(cfg)
            except DSLValidationError:
                continue
            if mutated is None:
                continue
            if mutated.model_dump() == base_dump:
                continue
            variants.append(mutated)
    finally:
        if restore_random_state and global_state is not None:
            random.setstate(global_state)

    rng.shuffle(variants)
    return variants


def _apply_random(
    mutators: List[MutationFn], seed: DSLConfig, rng: random.Random, k: int
) -> Optional[DSLConfig]:
    cfg = seed.model_copy(deep=True)
    tried = 0
    for mut in rng.sample(mutators, min(k, len(mutators))):
        try:
            out = mut(cfg)
            if out is not None:
                cfg = out
        except DSLValidationError:
            pass
        tried += 1
    return cfg


def generate_macro_mutations(
    cfg: DSLConfig, *, rng: Optional[random.Random] = None, width: int = 3
) -> List[DSLConfig]:
    """Produce a few radical variants by composing several mutators."""
    rng = rng or random.Random()
    radical_pool = [
        _stencil_toggle_mutation,
        _pos_toggle_mutation,
        _add_or_mutate_hierarchy,
        _add_or_mutate_depth_router,
        _toggle_kv_policy,
        _toggle_to_par_mutation,
        _swap_to_route_mutation,
    ]
    out: List[DSLConfig] = []
    for _ in range(width):
        combo_k = rng.choice([2, 3, 4])
        mutated = _apply_random(radical_pool, cfg, rng, combo_k)
        try:
            run_additional_checks(mutated)  # type: ignore[arg-type]
            out.append(mutated)  # type: ignore[arg-type]
        except DSLValidationError:
            continue
    rng.shuffle(out)
    return out


def pick_random_mutation(cfg: DSLConfig, *, rng: Optional[random.Random] = None) -> DSLConfig:
    variants = generate_mutations(cfg, rng=rng)
    if not variants:
        raise DSLValidationError("No valid mutations available for config")
    rng = rng or random.Random()
    return rng.choice(variants)
