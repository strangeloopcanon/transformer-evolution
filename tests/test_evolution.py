from __future__ import annotations

from pathlib import Path

from evoforge.search.evolution import EvolutionConfig, run_evolution
from evoforge.search.asha import ASHAConfig
from evoforge.search.pdh import PDHConfig


def test_run_evolution_smoke(tmp_path):
    seeds = [Path("examples/nanogpt_tiny.yaml"), Path("examples/transformer_vaswani.yaml")]
    evo_cfg = EvolutionConfig(
        generations=1,
        population_size=4,
        top_k=2,
        seq_len=64,
        batch_size=2,
        asha=ASHAConfig(min_steps=2, max_steps=4, reduction_factor=2),
        pdh=PDHConfig(base_steps=2, max_stages=1),
    )
    archive = run_evolution(seeds, tmp_path / "evo", evo_cfg=evo_cfg)
    assert archive
    assert all(c.path.exists() for c in archive if c.path.exists())
