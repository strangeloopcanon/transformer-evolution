#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evoforge.search.asha import ASHAConfig
from evoforge.search.evolution import EvolutionConfig, run_evolution
from evoforge.search.pdh import PDHConfig


def main() -> int:
    parser = argparse.ArgumentParser(description="Evolutionary search driver")
    parser.add_argument("configs", nargs="+", help="Seed config files or directories")
    parser.add_argument("--output", type=Path, default=Path("results/evolution"))
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--population", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seq-len", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--asha-min", type=int, default=20)
    parser.add_argument("--asha-max", type=int, default=80)
    parser.add_argument("--asha-reduction", type=int, default=2)
    parser.add_argument("--pdh-base", type=int, default=60)
    parser.add_argument("--pdh-stages", type=int, default=2)
    parser.add_argument("--immigrants", type=int, default=2)
    parser.add_argument(
        "--macro-prob",
        type=float,
        default=0.35,
        help="Probability to use radical macro-mutations when generating children [0-1]",
    )
    parser.add_argument(
        "--crossover-prob",
        type=float,
        default=0.35,
        help="Probability to use parent crossover when generating children [0-1]",
    )
    parser.add_argument(
        "--novelty-extra",
        type=int,
        default=2,
        help="Number of additional novelty-picked parents beyond top-k",
    )
    parser.add_argument(
        "--replacement",
        choices=["score", "aging"],
        default="score",
        help="Replacement policy when overflowing population (default: score)",
    )
    parser.add_argument(
        "--overflow-factor",
        type=float,
        default=1.0,
        help="Allow generating up to population*factor then trim (use with --replacement)",
    )
    parser.add_argument(
        "--top-index",
        type=Path,
        default=Path("docs/results_index.json"),
        help="Optional JSON index with prior runs whose top candidates can seed evolution",
    )
    parser.add_argument(
        "--top-count",
        type=int,
        default=0,
        help="If >0, pull up to this many top candidate configs from --top-index and add as seeds",
    )
    args = parser.parse_args()

    base_seeds = []
    for item in args.configs:
        path = Path(item)
        if path.is_dir():
            base_seeds.extend(sorted(path.glob("*.yaml")))
        else:
            base_seeds.append(path)
    seeds = [p for p in base_seeds if p.exists()]

    if args.top_count > 0 and args.top_index:
        top_paths: list[Path] = []
        if args.top_index.exists():
            try:
                data = json.loads(args.top_index.read_text())
                for run in data.get("runs", []):
                    for cand in run.get("top_candidates", []):
                        cand_path = Path(cand)
                        if not cand_path.is_absolute():
                            cand_path = Path(cand)
                        if cand_path.exists():
                            top_paths.append(cand_path)
                        if len(top_paths) >= args.top_count:
                            break
                    if len(top_paths) >= args.top_count:
                        break
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[warn] failed to parse {args.top_index}: {exc}")
        seeds.extend(top_paths)

    deduped = []
    seen = set()
    for path in seeds:
        resolved = Path(path).resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(resolved)
    seeds = deduped
    if not seeds:
        raise SystemExit("No seed configs found")

    evo_cfg = EvolutionConfig(
        generations=args.generations,
        population_size=args.population,
        top_k=args.top_k,
        device=args.device,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        asha=ASHAConfig(
            min_steps=args.asha_min,
            max_steps=args.asha_max,
            reduction_factor=args.asha_reduction,
        ),
        pdh=PDHConfig(
            base_steps=args.pdh_base,
            max_stages=args.pdh_stages,
        ),
        immigrants=args.immigrants,
        macro_prob=args.macro_prob,
        crossover_prob=args.crossover_prob,
        novelty_extra=args.novelty_extra,
        replacement=args.replacement,
        overflow_factor=args.overflow_factor,
    )

    archive = run_evolution(seeds, args.output, evo_cfg=evo_cfg)
    print("Evolution completed. Top candidates:")
    seen = set()
    printed = 0
    for cand in archive:
        p = str(cand.path)
        if p in seen:
            continue
        print(f"  {p}: score={cand.score}")
        seen.add(p)
        printed += 1
        if printed >= 5:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
