#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def node_id(path: str) -> str:
    # Make a Mermaid-safe identifier
    return re.sub(r"[^A-Za-z0-9_]", "_", path)


def label_from_path(path: str) -> str:
    # Show last two components like gen_23/variant_13.yaml
    p = Path(path)
    parent = p.parent.name
    return f"{parent}/{p.name}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert evolution lineage.json to Mermaid flowchart")
    ap.add_argument("lineage", type=Path, help="Path to lineage.json produced by evolution")
    ap.add_argument("--direction", default="LR", choices=["LR", "TB"], help="Graph direction")
    args = ap.parse_args()

    data = json.loads(args.lineage.read_text())
    records = data.get("records", [])

    print("flowchart", args.direction)
    # Group by generation for optional subgraphs
    by_gen: dict[int, list[dict]] = {}
    for rec in records:
        by_gen.setdefault(int(rec.get("gen", -1)), []).append(rec)

    # Declare nodes
    nodes = set()
    for rec in records:
        child = rec["child"]
        nid = node_id(child)
        if nid not in nodes:
            print(f'    {nid}["{label_from_path(child)}"]')
            nodes.add(nid)

    # Edges with operation labels
    for rec in records:
        child = rec["child"]
        op = rec.get("op", "mut")
        for par in rec.get("parents", []) or []:
            print(f"    {node_id(par)} -->|{op}| {node_id(child)}")

    # Optionally: subgraphs per generation (commented to keep output simple)
    # for g in sorted(by_gen.keys()):
    #     print(f"    subgraph gen_{g}")
    #     for rec in by_gen[g]:
    #         print(f"        {node_id(rec['child'])}")
    #     print("    end")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
