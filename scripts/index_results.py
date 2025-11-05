#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class RunInfo:
    name: str
    path: str
    generations: int
    lineage: Dict[str, bool]
    top_candidates: List[str]
    log: Optional[str]
    size_mb: float


def count_generations(run_dir: Path) -> int:
    gens = [p for p in run_dir.glob("gen_*") if p.is_dir()]
    return len(gens)


def locate_log(run_dir: Path) -> Optional[Path]:
    logs_dir = Path("results/logs")
    if not logs_dir.exists():
        return None
    stem = run_dir.name
    cands = sorted(logs_dir.glob(f"*{stem}*.log"))
    return cands[-1] if cands else None


def parse_top_from_log(log: Path) -> List[str]:
    try:
        text = log.read_text(errors="ignore")
    except Exception:
        return []
    lines = [l.strip() for l in text.splitlines()]
    out: List[str] = []
    capture = False
    for ln in lines[-1000:]:
        if ln.endswith("Top candidates:"):
            capture = True
            continue
        if capture:
            if ln.startswith("results/") or ln.startswith("  results/"):
                path = ln.split(":")[0].strip()
                out.append(path)
            elif ln == "" or not ln.startswith("  "):
                break
    # dedupe keep order
    seen = set()
    uniq = []
    for p in out:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def dir_size_mb(path: Path) -> float:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except Exception:
                pass
    return round(total / (1024 * 1024), 2)


def main() -> int:
    ap = argparse.ArgumentParser(description="Index evolution results into results/index.json")
    ap.add_argument("root", type=Path, nargs="?", default=Path("results"))
    args = ap.parse_args()

    runs = []
    for p in sorted(args.root.iterdir()):
        if not p.is_dir():
            continue
        if not re.match(r"^evolution_", p.name):
            continue
        gens = count_generations(p)
        lineage = {
            "lineage.json": (p / "lineage.json").exists(),
            "lineage_reconstructed.json": (p / "lineage_reconstructed.json").exists(),
            "lineage_focus.json": (p / "lineage_focus.json").exists(),
            "lineage_focus.png": (p / "lineage_focus.png").exists(),
        }
        log = locate_log(p)
        top = parse_top_from_log(log) if log else []
        runs.append(
            RunInfo(
                name=p.name,
                path=str(p),
                generations=gens,
                lineage=lineage,
                top_candidates=top[:5],
                log=str(log) if log else None,
                size_mb=dir_size_mb(p),
            )
        )

    index = {"runs": [asdict(r) for r in runs]}
    out = args.root / "index.json"
    out.write_text(json.dumps(index, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
