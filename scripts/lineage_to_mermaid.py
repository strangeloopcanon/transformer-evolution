#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import yaml


def node_id(path: str) -> str:
    # Make a Mermaid-safe identifier
    return re.sub(r"[^A-Za-z0-9_]", "_", path)


def label_from_path(path: str) -> str:
    # Show last two components like gen_23/variant_13.yaml
    p = Path(path)
    parent = p.parent.name
    return f"{parent}/{p.name}"


def load_traits(cfg_path: Path) -> List[str]:
    try:
        data = yaml.safe_load(cfg_path.read_text())
    except Exception:
        return []
    arch = (data or {}).get("arch", {})
    traits: List[str] = []
    mu = arch.get("mix_unit") or {}
    k = mu.get("kind")
    if k in {"single", "par", "route"}:
        traits.append(k)

    def _st(x: Optional[Dict]) -> Optional[str]:
        if not isinstance(x, dict):
            return None
        st = x.get("stencil")
        if isinstance(st, dict):
            return st.get("kind")
        return None

    st = _st(mu.get("mixer"))
    if st:
        traits.append(st)
    for ch in mu.get("choices") or []:
        st2 = _st(ch)
        if st2 and st2 not in traits:
            traits.append(st2)
    if arch.get("hierarchy"):
        traits.append("hier")
    dr = arch.get("depth_router")
    if isinstance(dr, dict) and dr.get("kind") and dr.get("kind") != "none":
        traits.append("dr")
    pos = arch.get("pos", {})
    if pos.get("kind") == "alibi":
        traits.append("alibi")
    elif pos.get("kind") == "rope":
        rope = pos.get("rope", {})
        dims = rope.get("dims")
        if isinstance(dims, int) and dims <= 32:
            traits.append("rope32")
        else:
            traits.append("rope")
    kv = arch.get("kv_policy") or {}
    if kv.get("cache") == "window" or kv.get("quant") in {"nf4", "fp8", "int8"}:
        traits.append("kv")
    return traits[:3]


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert evolution lineage.json to Mermaid flowchart")
    ap.add_argument("lineage", type=Path, help="Path to lineage.json produced by evolution")
    ap.add_argument("--direction", default="LR", choices=["LR", "TB"], help="Graph direction")
    ap.add_argument("--with-subgraphs", action="store_true", help="Group nodes by generation")
    ap.add_argument("--genesis-label", type=str, default="Transformer (Vaswani, 2017)")
    ap.add_argument(
        "--attach-genesis", action="store_true", help="Attach external seeds to genesis node"
    )
    ap.add_argument("--gen-start", type=int, default=None, help="Only include records with gen >= this")
    ap.add_argument("--gen-end", type=int, default=None, help="Only include records with gen <= this")
    args = ap.parse_args()

    data = json.loads(args.lineage.read_text())
    records = data.get("records", [])
    if args.gen_start is not None:
        records = [r for r in records if int(r.get("gen", -1)) >= args.gen_start]
    if args.gen_end is not None:
        records = [r for r in records if int(r.get("gen", -1)) <= args.gen_end]
    out_dir = args.lineage.parent

    print("flowchart", args.direction)
    # Class styles
    print("    classDef genesis fill:#FFFFFF,stroke:#000,stroke-width:2px;")
    print("    classDef seed fill:#FFFFFF,stroke:#666,stroke-dasharray: 5 5;")
    print("    classDef route fill:#FFE8C2,stroke:#C88;")
    print("    classDef par fill:#E8F7FF,stroke:#07C;")
    print("    classDef single fill:#F7F7F7,stroke:#999;")
    print("    classDef sliding fill:#EAF7EA,stroke:#3A3;")
    print("    classDef local fill:#F3FFF3,stroke:#6A6;")
    print("    classDef ring fill:#F3F3FF,stroke:#66A;")
    print("    classDef hier fill:#FFF1DB,stroke:#C83;")
    print("    classDef dr fill:#FBEFFF,stroke:#A5A;")
    print("    classDef rope32 fill:#D8F0FF,stroke:#08A;")
    print("    classDef rope fill:#EEF7FF,stroke:#66A;")
    print("    classDef alibi fill:#EEE,stroke:#999;")
    print("    classDef kv fill:#FFFCD1,stroke:#AA7;")

    # Group by generation
    by_gen: dict[int, list[dict]] = {}
    for rec in records:
        by_gen.setdefault(int(rec.get("gen", -1)), []).append(rec)

    nodes = set()
    internal_prefix = str(out_dir)
    if args.attach_genesis:
        print(f'    genesis(["{args.genesis_label}"]) ')
        print("    class genesis genesis")
    seed_nodes = set()

    def add_node(path: str):
        nid = node_id(path)
        if nid in nodes:
            return
        label = label_from_path(path)
        print(f'    {nid}["{label}"]')
        p = Path(path)
        classes = []
        if p.exists():
            classes = load_traits(p)
        if classes:
            print(f"    class {nid} {','.join(classes)}")
        nodes.add(nid)

    for rec in records:
        add_node(rec["child"])
        for par in rec.get("parents", []) or []:
            if str(par).startswith(internal_prefix):
                add_node(par)
            elif args.attach_genesis:
                sp = str(par)
                sid = node_id(sp)
                if sid not in seed_nodes:
                    print(f'    {sid}(["{label_from_path(sp)}"]) ')
                    print(f"    class {sid} seed")
                    seed_nodes.add(sid)

    # Edges with operation labels
    for rec in records:
        child = rec["child"]
        op = rec.get("op", "mut")
        for par in rec.get("parents", []) or []:
            par_str = str(par)
            if not par_str.startswith(internal_prefix) and args.attach_genesis:
                print(f"    genesis -->|seed| {node_id(par_str)}")
            print(f"    {node_id(par_str)} -->|{op}| {node_id(child)}")

    if args.with_subgraphs:
        for g in sorted(by_gen.keys()):
            print(f"    subgraph gen_{g}")
            for rec in by_gen[g]:
                print(f"        {node_id(rec['child'])}")
            print("    end")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
