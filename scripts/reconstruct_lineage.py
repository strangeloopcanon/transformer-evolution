#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    try:
        return yaml.safe_load(path.read_text()) or {}
    except Exception:
        return {}


def flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten(v, key))
        else:
            out[key] = v
    return out


NUM_KEYS_HINTS = (
    ".heads",
    ".groups",
    ".dims",
    ".theta",
    ".window",
    ".stride",
    ".block",
    ".mult",
    ".budget",
    ".n_layers",
    ".d_model",
    ".chunk",
    ".d_state",
)


def as_num(key: str, v: Any) -> Tuple[bool, float]:
    if isinstance(v, (int, float)):
        return True, float(v)
    # Heuristic: only convert strings that look numeric and keys that hint numeric
    if isinstance(v, str) and any(h in key for h in NUM_KEYS_HINTS):
        try:
            return True, float(v)
        except Exception:
            return False, 0.0
    return False, 0.0


def feature_dict(cfg: Dict[str, Any]) -> Dict[str, Any]:
    arch = cfg.get("arch", {})
    # focus on architecture; ignore train budgets for distance
    return flatten({"arch": arch})


def distance(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    # Mixed distance: numeric L1 on overlapping numeric keys + categorical mismatch penalty
    keys = set(a.keys()) | set(b.keys())
    num_sum = 0.0
    num_count = 0
    cat_pen = 0.0
    for k in keys:
        va = a.get(k)
        vb = b.get(k)
        is_na, na = as_num(k, va)
        is_nb, nb = as_num(k, vb)
        if is_na and is_nb:
            denom = max(1.0, abs(na) + abs(nb))
            num_sum += abs(na - nb) / denom
            num_count += 1
        else:
            if va is None and vb is None:
                continue
            if va != vb:
                cat_pen += 1.0
    return (num_sum / max(1, num_count)) + 0.05 * cat_pen


def guess_op_kind(parent_feat: Dict[str, Any], child_feat: Dict[str, Any]) -> str:
    # Heuristic: if many categorical fields changed and several numeric changes -> macro
    keys = set(parent_feat.keys()) | set(child_feat.keys())
    cat_changes = 0
    num_changes = 0
    for k in keys:
        va = parent_feat.get(k)
        vb = child_feat.get(k)
        is_na, na = as_num(k, va)
        is_nb, nb = as_num(k, vb)
        if is_na and is_nb:
            if abs(na - nb) > 0.2 * max(1.0, abs(na)):
                num_changes += 1
        else:
            if va != vb:
                cat_changes += 1
    if cat_changes >= 4 or (cat_changes >= 2 and num_changes >= 3):
        return "reconstructed_macro"
    return "reconstructed_mutation"


def collect_configs(root: Path) -> List[Path]:
    return sorted(root.glob("gen_*/**/*.yaml"))


def main() -> int:
    ap = argparse.ArgumentParser(description="Reconstruct lineage post-hoc from gen_* YAMLs")
    ap.add_argument(
        "output_dir", type=Path, help="Evolution output directory containing gen_* folders"
    )
    ap.add_argument("--seed", nargs="*", default=[], help="Additional seed files or directories")
    ap.add_argument(
        "--k", type=int, default=2, help="Consider up to k parents (to flag crossovers)"
    )
    args = ap.parse_args()

    out = args.output_dir
    gens = sorted(
        [p for p in out.glob("gen_*") if p.is_dir()], key=lambda p: int(p.name.split("_")[-1])
    )
    if not gens:
        raise SystemExit(f"No gen_* folders found under {out}")

    # Load seeds
    seed_paths: List[Path] = []
    for s in args.seed:
        p = Path(s)
        if p.is_dir():
            seed_paths.extend(sorted(p.glob("*.yaml")))
        elif p.exists():
            seed_paths.append(p)
    seeds = {str(p): feature_dict(load_yaml(p)) for p in seed_paths}

    # Index previous gen variants progressively
    prev: Dict[str, Dict[str, Any]] = dict(seeds)
    records: List[Dict[str, Any]] = []
    for gen_path in gens:
        gen_idx = int(gen_path.name.split("_")[-1])
        variants = sorted(gen_path.glob("*.yaml"))
        # Build feature dicts for current gen
        curr_feats: Dict[str, Dict[str, Any]] = {
            str(p): feature_dict(load_yaml(p)) for p in variants
        }
        # For each variant, pick nearest parents from prev
        for path_str, feat in curr_feats.items():
            if prev:
                scored = sorted(
                    ((distance(feat, pf), pstr) for pstr, pf in prev.items()), key=lambda t: t[0]
                )
                parents = [p for _, p in scored[: max(1, args.k)]]
            else:
                parents = []
            op = (
                "immigrant"
                if Path(path_str).name.startswith("immigrant_")
                else ("crossover" if len(parents) >= 2 else "reconstructed_mutation")
            )
            if op == "reconstructed_mutation" and parents:
                op = guess_op_kind(prev[parents[0]], feat)
            rec: Dict[str, Any] = {
                "gen": gen_idx,
                "child": path_str,
                "op": op,
                "parents": parents[:2] if parents else [],
            }
            records.append(rec)
        # advance
        prev.update(curr_feats)

    lineage = {"records": records}
    (out / "lineage_reconstructed.json").write_text(json.dumps(lineage, indent=2))
    print(f"wrote {out/'lineage_reconstructed.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
