#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

from evoforge.search.asha import ASHAConfig
from evoforge.search.pdh import PDHConfig
from evoforge.search.runner import load_candidates, run_search


def compute_qpc(loss_history: List[float], total_flops: float) -> float:
    if not loss_history or total_flops <= 0:
        return float("nan")
    return (loss_history[0] - loss_history[-1]) / total_flops


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ASHA+PDH experiment and log results")
    parser.add_argument("configs", nargs="+", help="Config YAMLs or directories")
    parser.add_argument("--output", type=Path, default=Path("results/search_report.json"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--asha-min", type=int, default=10)
    parser.add_argument("--asha-max", type=int, default=60)
    parser.add_argument("--asha-reduction", type=int, default=3)
    parser.add_argument("--pdh-base", type=int, default=20)
    parser.add_argument("--pdh-stages", type=int, default=3)
    args = parser.parse_args()

    cfg_paths: List[Path] = []
    for item in args.configs:
        path = Path(item)
        if path.is_dir():
            cfg_paths.extend(sorted(path.glob("*.yaml")))
        else:
            cfg_paths.append(path)
    cfg_paths = [p for p in cfg_paths if p.exists()]
    if not cfg_paths:
        raise SystemExit("No valid configs provided")

    asha_cfg = ASHAConfig(min_steps=args.asha_min, max_steps=args.asha_max, reduction_factor=args.asha_reduction)
    pdh_cfg = PDHConfig(base_steps=args.pdh_base, max_stages=args.pdh_stages)

    search_result = run_search(
        cfg_paths,
        asha_config=asha_cfg,
        pdh_config=pdh_cfg,
        device=args.device,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
    )

    summary: Dict[str, Dict] = {
        "asha": {
            "evaluated": search_result.asha.evaluated,
            "promoted": search_result.asha.promoted,
            "best_score": search_result.asha.best_score,
            "history": search_result.asha.history,
        },
        "candidates": [],
    }

    for idx, state in enumerate(search_result.pdh_states):
        best = state.best or {}
        result = best.get("result")
        entry = {
            "candidate_index": idx,
            "stages": state.history,
            "best_score": best.get("score"),
            "best_steps": best.get("steps"),
            "qpc": compute_qpc(result.loss_history if result else [], result.total_flops if result else 0.0)
            if result
            else None,
            "tokens_per_sec": result.tokens_per_sec if result else None,
            "total_tokens": result.total_tokens if result else None,
            "total_flops": result.total_flops if result else None,
        }
        summary["candidates"].append(entry)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(f"Wrote report to {args.output}")
    print("ASHA evaluated", summary["asha"]["evaluated"], "configs")
    for cand in summary["candidates"]:
        print(
            f"Candidate {cand['candidate_index']}: best_score={cand['best_score']} steps={cand['best_steps']} qpc={cand['qpc']} tokens={cand['total_tokens']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
