#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Set


def collect_keep_nodes(lineage_json: Path) -> Set[str]:
    data = json.loads(lineage_json.read_text())
    nodes: Set[str] = set()
    for rec in data.get("records", []):
        ch = str(rec.get("child"))
        if ch:
            nodes.add(ch)
        for p in rec.get("parents", []) or []:
            ps = str(p)
            if ps:
                nodes.add(ps)
    return nodes


def find_yaml_files(root: Path) -> List[Path]:
    return sorted(root.glob("gen_*/**/*.yaml"))


def prune_run(run_dir: Path, keep_nodes: Set[str], apply: bool) -> List[Path]:
    removed: List[Path] = []
    archive = run_dir / "_archive"
    if apply:
        archive.mkdir(exist_ok=True)
    for yp in find_yaml_files(run_dir):
        sp = str(yp)
        if sp in keep_nodes:
            continue
        removed.append(yp)
        if apply:
            dest = archive / yp.name
            try:
                shutil.move(str(yp), str(dest))
            except Exception:
                pass
    return removed


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Prune evolution run folder to a minimal subset using lineage"
    )
    ap.add_argument("run_dir", type=Path)
    ap.add_argument(
        "--lineage",
        type=Path,
        default=None,
        help="Path to lineage_focus.json / reconstructed.json / lineage.json",
    )
    ap.add_argument(
        "--apply", action="store_true", help="Move pruned YAMLs to _archive (default: dry-run)"
    )
    args = ap.parse_args()

    run_dir = args.run_dir
    if args.lineage and args.lineage.exists():
        lineage = args.lineage
    else:
        # prefer focus, then reconstructed, then live
        cand = [
            run_dir / "lineage_focus.json",
            run_dir / "lineage_reconstructed.json",
            run_dir / "lineage.json",
        ]
        lineage = next((p for p in cand if p.exists()), None)
        if not lineage:
            raise SystemExit("No lineage json found; pass --lineage explicitly")

    keep = collect_keep_nodes(lineage)
    removed = prune_run(run_dir, keep, apply=args.apply)
    print(f"keep_count={len(keep)} pruned_count={len(removed)} apply={args.apply}")
    for p in removed[:10]:
        print(f"pruned_sample: {p}")
    if removed and not args.apply:
        print("(dry-run) Re-run with --apply to move pruned files to _archive/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
