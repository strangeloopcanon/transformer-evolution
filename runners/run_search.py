#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from evoforge.search.asha import ASHAConfig
from evoforge.search.pdh import PDHConfig
from evoforge.search.runner import run_search


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ASHA + PDH search over DSL configs")
    parser.add_argument("configs", nargs="+", help="List of config files or directories")
    parser.add_argument("--min-steps", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--reduction", type=int, default=3)
    parser.add_argument("--base-steps", type=int, default=5)
    parser.add_argument("--max-stages", type=int, default=3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    cfg_paths: List[Path] = []
    for item in args.configs:
        path = Path(item)
        if path.is_dir():
            cfg_paths.extend(sorted(path.glob("*.yaml")))
        else:
            cfg_paths.append(path)

    if not cfg_paths:
        raise SystemExit("No configs provided")

    asha_cfg = ASHAConfig(
        min_steps=args.min_steps, max_steps=args.max_steps, reduction_factor=args.reduction
    )
    pdh_cfg = PDHConfig(base_steps=args.base_steps, max_stages=args.max_stages)
    result = run_search(
        cfg_paths,
        asha_config=asha_cfg,
        pdh_config=pdh_cfg,
        device=args.device,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
    )

    print("ASHA evaluated:", result.asha.evaluated)
    print("Promoted:", result.asha.promoted)
    print("Best score:", result.asha.best_score)
    print("PDH states:", len(result.pdh_states))
    for idx, state in enumerate(result.pdh_states):
        best = state.best["score"] if state.best else None
        print(f"  Candidate {idx}: stages={len(state.history)} best={best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
