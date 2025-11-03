#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from evoforge.train.simple_trainer import run_micro_train


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a tiny training loop for a DSL config")
    parser.add_argument("--cfg", required=True, type=Path, help="Path to DSL YAML config")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    result = run_micro_train(
        args.cfg,
        steps=args.steps,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        device=args.device,
    )
    print("Tiny training complete")
    print(f"Steps: {args.steps}")
    print(f"Loss history: {result.loss_history}")
    print(f"Tokens/sec: {result.tokens_per_sec:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

