#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from evoforge.dsl.api import DSLValidationError, load_validate_yaml


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate a DSL YAML config")
    ap.add_argument("--cfg", type=str, required=True, help="Path to YAML config")
    args = ap.parse_args()
    path = Path(args.cfg)
    try:
        cfg = load_validate_yaml(path)
    except DSLValidationError as e:
        print("INVALID\n---")
        print(e)
        return 1
    print("VALID\n---")
    print(cfg.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
