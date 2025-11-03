#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
    args = parser.parse_args()

    seeds = []
    for item in args.configs:
        path = Path(item)
        if path.is_dir():
            seeds.extend(sorted(path.glob("*.yaml")))
        else:
            seeds.append(path)
    seeds = [p for p in seeds if p.exists()]
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
    )

    archive = run_evolution(seeds, args.output, evo_cfg=evo_cfg)
    print(f"Evolution completed. Top candidates:")
    for cand in archive[:5]:
        print(f"  {cand.path}: score={cand.score}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
