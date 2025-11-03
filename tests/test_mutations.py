from __future__ import annotations

from pathlib import Path
import random

import pytest
import yaml

from evoforge.dsl.api import load_validate_yaml
from evoforge.dsl.mutations import generate_mutations, pick_random_mutation


BASE_CFG = load_validate_yaml(Path("examples/route.yaml"))


def _prune_none(obj):
    if isinstance(obj, dict):
        return {k: _prune_none(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_prune_none(v) for v in obj if v is not None]
    return obj


def test_generate_mutations_non_empty() -> None:
    rng = random.Random(0)
    variants = generate_mutations(BASE_CFG, rng=rng)
    assert variants, "expected at least one mutation"
    original = BASE_CFG.model_dump()
    for variant in variants:
        assert variant.model_dump() != original


def test_pick_random_mutation_deterministic_seed() -> None:
    rng = random.Random(123)
    variant1 = pick_random_mutation(BASE_CFG, rng=rng)
    rng = random.Random(123)
    variant2 = pick_random_mutation(BASE_CFG, rng=rng)
    assert variant1.model_dump() == variant2.model_dump()


def test_mutations_preserve_validation(tmp_path: Path) -> None:
    for variant in generate_mutations(BASE_CFG, rng=random.Random(1)):
        # Serialize to YAML and re-validate via public loader.
        yaml_path = tmp_path / "variant.yaml"
        yaml_dict = _prune_none(variant.model_dump(mode="python"))
        with yaml_path.open("w") as fh:
            yaml.safe_dump(yaml_dict, fh)
        loaded = load_validate_yaml(yaml_path)
        assert loaded.model_dump() == variant.model_dump()


def test_pipeline_mutations(tmp_path: Path) -> None:
    cfg = load_validate_yaml(Path("examples/free_transformer_pipeline.yaml"))

    variants = []

    # Variant 1: switch decoder_lower attention to ring pattern
    ring = cfg.model_copy(deep=True)
    lower = ring.arch.modules["decoder_lower"]
    assert lower.mix_unit and lower.mix_unit.router
    target_mixer = None
    if lower.mix_unit.mixer is not None:
        target_mixer = lower.mix_unit.mixer
    elif lower.mix_unit.choices:
        target_mixer = lower.mix_unit.choices[0]
    assert target_mixer is not None
    target_mixer.stencil.kind = "ring"
    target_mixer.stencil.block = 512
    target_mixer.stencil.stride = 128
    variants.append(ring)

    # Variant 2: add depth router to decoder_upper
    from evoforge.dsl.models import DepthRouter

    depth = cfg.model_copy(deep=True)
    upper = depth.arch.modules["decoder_upper"]
    upper.depth_router = DepthRouter(kind="token", budget=0.55, tau=0.6, min_layers=2)
    variants.append(depth)

    # Variant 3: insert hierarchical memory module feeding decoder_upper
    from evoforge.dsl.models import ModuleSpec, MixUnit, Mixer, FFN, SoftmaxConfig, StencilConfig

    memory_cfg = cfg.model_copy(deep=True)
    memory_cfg.arch.modules["memory_module"] = ModuleSpec(
        kind="transformer",
        d_model=640,
        n_layers=1,
        mix_unit=MixUnit(
            kind="single",
            mixer=Mixer(
                kind="Attention",
                heads=4,
                groups=2,
                stencil=StencilConfig(kind="block", block=256),
                softmax=SoftmaxConfig(qk_norm="rms"),
            ),
        ),
        ffn=FFN(kind="dense", mult=2.0, act="gelu"),
        norm="RMSNorm",
        pos=cfg.arch.pos,
    )

    pipeline_list = list(memory_cfg.arch.pipeline or [])
    inserted = False
    for idx, stage in enumerate(pipeline_list):
        if stage.name == "decoder_upper":
            pipeline_list.insert(
                idx,
                stage.__class__(
                    name="memory",
                    module="memory_module",
                    kv_from=["encoder"],
                ),
            )
            inserted = True
            break
    assert inserted, "decoder_upper stage not found"
    memory_cfg.arch.pipeline = pipeline_list
    for stage in memory_cfg.arch.pipeline:
        if stage.name == "decoder_upper":
            existing = stage.kv_from or []
            if "memory" not in existing:
                stage.kv_from = existing + ["memory"]
    variants.append(memory_cfg)

    for idx, variant in enumerate(variants):
        yaml_dict = _prune_none(variant.model_dump(mode="python"))
        yaml_path = tmp_path / f"variant_pipeline_{idx}.yaml"
        with yaml_path.open("w") as fh:
            yaml.safe_dump(yaml_dict, fh)
        loaded = load_validate_yaml(yaml_path)
        assert loaded.arch.pipeline is not None


def test_mutation_suite_generates() -> None:
    cfg = load_validate_yaml(Path("examples/free_transformer_pipeline.yaml"))
    mutated = generate_mutations(cfg, rng=random.Random(2))
    assert mutated
    for variant in mutated:
        assert variant.arch.d_model == cfg.arch.d_model


def test_add_memory_stage_mutation() -> None:
    cfg = load_validate_yaml(Path("examples/free_transformer_pipeline.yaml"))
    from evoforge.dsl.mutations import _add_memory_stage

    variant = _add_memory_stage(cfg)
    assert variant is not None
    assert any(stage.name == "memory_auto" for stage in variant.arch.pipeline)
