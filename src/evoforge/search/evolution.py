from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from evoforge.dsl.api import load_validate_yaml
from evoforge.dsl.mutations import generate_mutations
from evoforge.dsl.models import DSLConfig
from evoforge.search.runner import run_search
from evoforge.search.asha import ASHAConfig
from evoforge.search.pdh import PDHConfig


@dataclass
class EvolutionCandidate:
    path: Path
    config: DSLConfig
    score: float = float("inf")
    metadata: Dict = field(default_factory=dict)


@dataclass
class EvolutionConfig:
    generations: int = 2
    population_size: int = 6
    top_k: int = 3
    seq_len: int = 128
    batch_size: int = 4
    device: Optional[str] = None
    asha: ASHAConfig = ASHAConfig(min_steps=20, max_steps=80, reduction_factor=2)
    pdh: PDHConfig = PDHConfig(base_steps=60, max_stages=2)


def _prune_none(value):
    if isinstance(value, dict):
        return {k: _prune_none(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_prune_none(v) for v in value if v is not None]
    return value


def _write_config(cfg: DSLConfig, path: Path) -> None:
    data = _prune_none(cfg.model_dump(mode="python"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)


def _score_from_report(report: Dict) -> float:
    best = report.get("best_score")
    if best is None:
        return float("inf")
    return best


def run_evolution(
    seed_paths: List[Path],
    output_dir: Path,
    *,
    evo_cfg: EvolutionConfig,
    rng: Optional[random.Random] = None,
) -> List[EvolutionCandidate]:
    rng = rng or random.Random()
    population: List[EvolutionCandidate] = []
    for path in seed_paths:
        cfg = load_validate_yaml(path)
        population.append(EvolutionCandidate(path=path, config=cfg))

    archive: List[EvolutionCandidate] = []

    for gen in range(evo_cfg.generations):
        gen_dir = output_dir / f"gen_{gen}"
        gen_dir.mkdir(parents=True, exist_ok=True)
        cfg_files = [cand.path for cand in population]
        result = run_search(
            cfg_files,
            asha_config=evo_cfg.asha,
            pdh_config=evo_cfg.pdh,
            device=evo_cfg.device,
            seq_len=evo_cfg.seq_len,
            batch_size=evo_cfg.batch_size,
        )

        cand_results: Dict[Path, float] = {}
        for idx, state in enumerate(result.pdh_states):
            best = state.best
            score = float("inf")
            if best is not None:
                score = best.get("score", float("inf"))
            cand_results[population[idx].path] = score
            population[idx].score = score
            population[idx].metadata = {
                "history": state.history,
                "best": state.best,
            }

        archive.extend(population)
        archive.sort(key=lambda c: c.score)
        archive = archive[: evo_cfg.population_size]

        parents = sorted(population, key=lambda c: c.score)[: evo_cfg.top_k]
        next_gen: List[EvolutionCandidate] = []
        next_gen.extend(parents)

        while len(next_gen) < evo_cfg.population_size:
            parent = rng.choice(parents)
            variants = generate_mutations(parent.config, rng=rng)
            if not variants:
                break
            variant_cfg = rng.choice(variants)
            variant_path = gen_dir / f"variant_{len(next_gen)}.yaml"
            _write_config(variant_cfg, variant_path)
            next_gen.append(EvolutionCandidate(path=variant_path, config=variant_cfg))

        population = next_gen

    return archive
