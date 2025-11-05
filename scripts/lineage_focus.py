#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set


def load_lineage(path: Path) -> Dict:
    return json.loads(path.read_text())


def find_max_gen(records: List[Dict]) -> int:
    gens = [
        int(r.get("gen", -1)) for r in records if isinstance(r.get("gen", None), (int, float, str))
    ]
    return max((int(g) for g in gens if g is not None and int(g) >= 0), default=0)


def pick_leaves(records: List[Dict], gen: int, k: int) -> List[str]:
    # Prefer variant_* in the target generation
    leaves = [
        r["child"]
        for r in records
        if int(r.get("gen", -1)) == gen and "variant_" in str(r.get("child", ""))
    ]
    leaves = sorted(set(leaves))
    return leaves[:k] if k > 0 else leaves


def build_parent_map(records: List[Dict]) -> Dict[str, List[str]]:
    pm: Dict[str, List[str]] = {}
    for r in records:
        ch = str(r.get("child"))
        parents = [str(p) for p in (r.get("parents") or [])]
        pm[ch] = parents
    return pm


def collect_ancestors(parent_map: Dict[str, List[str]], leaves: List[str]) -> Set[str]:
    keep: Set[str] = set()
    stack: List[str] = list(leaves)
    while stack:
        node = stack.pop()
        if node in keep:
            continue
        keep.add(node)
        for p in parent_map.get(node, []):
            if p not in keep:
                stack.append(p)
    return keep


def filter_records(records: List[Dict], keep_nodes: Set[str]) -> List[Dict]:
    out: List[Dict] = []
    for r in records:
        ch = str(r.get("child"))
        pars = [str(p) for p in (r.get("parents") or [])]
        if ch in keep_nodes:
            # keep edges even if parents are external seeds (not in keep_nodes)
            out.append(
                {
                    "gen": int(r.get("gen", -1)),
                    "child": ch,
                    "op": r.get("op", "edge"),
                    "parents": pars,
                }
            )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract top-k leaf lineages from lineage json")
    ap.add_argument("lineage_json", type=Path)
    ap.add_argument(
        "--k", type=int, default=3, help="Number of leaf variants from final generation"
    )
    ap.add_argument("--gen", type=int, default=None, help="Target generation (default: max)")
    ap.add_argument("--out", type=Path, default=None, help="Output filtered lineage json path")
    args = ap.parse_args()

    data = load_lineage(args.lineage_json)
    records: List[Dict] = data.get("records", [])
    target_gen = args.gen if args.gen is not None else find_max_gen(records)
    leaves = pick_leaves(records, gen=target_gen, k=args.k)
    parent_map = build_parent_map(records)
    keep_nodes = collect_ancestors(parent_map, leaves)
    filtered = filter_records(records, keep_nodes)
    out = {"records": filtered}
    out_path = args.out or (args.lineage_json.parent / "lineage_focus.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
