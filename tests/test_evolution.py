from __future__ import annotations

from pathlib import Path

from evoforge.search.evolution import EvolutionConfig, run_evolution
from evoforge.search.asha import ASHAConfig, Candidate, run_asha
from evoforge.search.pdh import PDHConfig
from evoforge.dsl.api import load_validate_yaml


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


def test_run_asha_handles_runtime_failure(monkeypatch):
    cfg_path = Path("examples/nanogpt_tiny.yaml")
    cfg = load_validate_yaml(cfg_path)
    cand = Candidate(cfg_path=cfg_path, config=cfg, train_steps=2)

    def _boom(*args, **kwargs):
        raise RuntimeError("bad linear dimensions")

    monkeypatch.setattr("evoforge.search.asha._run_candidate", _boom)
    stats = run_asha(
        [cand],
        asha_cfg=ASHAConfig(min_steps=2, max_steps=2, reduction_factor=2),
        device="cpu",
        seq_len=16,
        batch_size=2,
    )
    assert stats.evaluated == 1
    assert stats.best_score == float("inf")
