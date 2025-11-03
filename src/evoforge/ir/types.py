from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple


Shape = Tuple[int, int, int]  # (T, B, D)


@dataclass
class StateSpec:
    kind: Literal["none", "kv", "recurrent"] = "none"
    size_bytes: int = 0


@dataclass
class Node:
    kind: Literal["Mixer", "FFN", "Norm", "Pos", "Cond", "Router", "Merge"]
    attrs: Dict[str, object]
    in_shape: Shape
    out_shape: Shape
    state_spec: Optional[StateSpec] = None


Graph = List[Node]

