from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from evoforge.dsl.api import load_validate_yaml
from evoforge.dsl.mutations import generate_macro_mutations, generate_mutations
from evoforge.dsl.models import DSLConfig
from evoforge.search.runner import run_search
from evoforge.search.novelty import embed_architecture, novelty_score
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
    asha: ASHAConfig = field(
        default_factory=lambda: ASHAConfig(min_steps=20, max_steps=80, reduction_factor=2)
    )
    pdh: PDHConfig = field(default_factory=lambda: PDHConfig(base_steps=60, max_stages=2))
    immigrants: int = 2
    # exploration knobs
    macro_prob: float = 0.35
    crossover_prob: float = 0.35
    novelty_extra: int = 2


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
    # lineage tracking for visualization
    lineage_records: List[Dict[str, object]] = []

    archive_vecs: List[List[float]] = []
    scores_map: Dict[str, float] = {}
    for gen in range(evo_cfg.generations):
        gen_dir = output_dir / f"gen_{gen}"
        gen_dir.mkdir(parents=True, exist_ok=True)
        cfg_files = [cand.path for cand in population]
        print(
            f"[evoforge] generation {gen}: evaluating {len(cfg_files)} candidates",
            flush=True,
        )
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
            scores_map[str(population[idx].path)] = score
            population[idx].metadata = {
                "history": state.history,
                "best": state.best,
            }

        # Update archive and novelty info
        for cand in population:
            try:
                archive_vecs.append(embed_architecture(cand.config))
            except Exception:
                pass
        archive.extend(population)
        archive.sort(key=lambda c: c.score)
        archive = archive[: evo_cfg.population_size]

        # Parent selection: top by score plus a couple by novelty to keep diversity
        sorted_by_score = sorted(population, key=lambda c: c.score)
        parents = sorted_by_score[: evo_cfg.top_k]
        # compute novelty for the rest
        novelty_pairs = []
        for cand in population:
            try:
                vec = embed_architecture(cand.config)
                nov = novelty_score(vec, archive_vecs)
                novelty_pairs.append((nov, cand))
            except Exception:
                pass
        novelty_pairs.sort(key=lambda t: t[0], reverse=True)
        for _, cand in novelty_pairs:
            if cand not in parents:
                parents.append(cand)
            if len(parents) >= evo_cfg.top_k + evo_cfg.novelty_extra:
                break
        next_gen: List[EvolutionCandidate] = []
        next_gen.extend(parents)

        # Children from mutations
        while len(next_gen) < max(0, evo_cfg.population_size - evo_cfg.immigrants):
            if len(parents) >= 2 and rng.random() < evo_cfg.crossover_prob:
                # simple crossover: pick two parents, blend arch fields and borrow modules/pipeline
                p1, p2 = rng.sample(parents, 2)
                child = p1.config.model_copy(deep=True)
                # blend a few scalars
                try:
                    child.arch.ffn.mult = round(
                        (p1.config.arch.ffn.mult + p2.config.arch.ffn.mult) / 2.0, 2
                    )
                    # choose mix_unit from either
                    if rng.random() < 0.5 and p2.config.arch.mix_unit:
                        child.arch.mix_unit = p2.config.arch.mix_unit.model_copy(deep=True)
                    # swap rope dims occasionally
                    if (
                        child.arch.pos.kind == "rope"
                        and child.arch.pos.rope
                        and p2.config.arch.pos.kind == "rope"
                        and p2.config.arch.pos.rope
                        and rng.random() < 0.5
                    ):
                        child.arch.pos.rope.dims = p2.config.arch.pos.rope.dims
                    # borrow a module from p2
                    if p2.config.arch.modules and rng.random() < 0.5:
                        child.arch.modules = child.arch.modules or {}
                        for name, mod in p2.config.arch.modules.items():
                            if rng.random() < 0.5:
                                child.arch.modules[name] = mod.model_copy(deep=True)
                except Exception:
                    pass
                variant_cfg = child
                op_kind = "crossover"
            else:
                parent = rng.choice(parents)
                # Occasionally apply a macro (radical) mutation composed of multiple edits
                used_macro = rng.random() < evo_cfg.macro_prob
                if used_macro:
                    variants = generate_macro_mutations(parent.config, rng=rng, width=3)
                else:
                    variants = generate_mutations(parent.config, rng=rng)
                if not variants:
                    break
                variant_cfg = rng.choice(variants)
                op_kind = "macro" if used_macro else "mutation"
            variant_path = gen_dir / f"variant_{len(next_gen)}.yaml"
            _write_config(variant_cfg, variant_path)
            next_gen.append(EvolutionCandidate(path=variant_path, config=variant_cfg))
            # lineage entry
            if op_kind == "crossover":
                lineage_records.append(
                    {
                        "gen": gen,
                        "child": str(variant_path),
                        "op": op_kind,
                        "parents": [str(p1.path), str(p2.path)],
                    }
                )
            else:
                lineage_records.append(
                    {
                        "gen": gen,
                        "child": str(variant_path),
                        "op": op_kind,
                        "parents": [str(parent.path)],
                    }
                )

        # Random immigrants from original seeds to maintain diversity
        for i in range(evo_cfg.immigrants):
            base = rng.choice(seed_paths)
            base_cfg = load_validate_yaml(base)
            # Optionally mutate once to avoid duplicates
            muts = generate_mutations(base_cfg, rng=rng)
            immigrant_cfg = rng.choice(muts) if muts else base_cfg
            immigrant_path = gen_dir / f"immigrant_{i}.yaml"
            _write_config(immigrant_cfg, immigrant_path)
            next_gen.append(EvolutionCandidate(path=immigrant_path, config=immigrant_cfg))
            lineage_records.append(
                {
                    "gen": gen,
                    "child": str(immigrant_path),
                    "op": "immigrant",
                    "parents": [str(base)],
                }
            )

        population = next_gen
        # backfill scores into lineage records for any known paths in this generation
        for rec in lineage_records:
            if "score" not in rec:
                s = scores_map.get(rec.get("child", ""))
                if s is not None:
                    rec["score"] = s
        best_score = archive[0].score if archive else float("inf")
        score_msg = f"{best_score:.6f}" if math.isfinite(best_score) else "inf"
        print(
            f"[evoforge] generation {gen}: completed with best score {score_msg}",
            flush=True,
        )

    # Write lineage file for visualization
    try:
        import json

        lineage_path = output_dir / "lineage.json"
        with lineage_path.open("w") as fh:
            json.dump(
                {
                    "macro_prob": evo_cfg.macro_prob,
                    "crossover_prob": evo_cfg.crossover_prob,
                    "novelty_extra": evo_cfg.novelty_extra,
                    "records": lineage_records,
                },
                fh,
                indent=2,
            )
    except Exception:
        pass

    return archive
