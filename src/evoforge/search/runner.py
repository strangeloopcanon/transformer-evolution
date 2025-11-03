from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from evoforge.dsl.api import load_validate_yaml
from evoforge.dsl.models import DSLConfig
from evoforge.search.asha import ASHAConfig, ASHAStats, Candidate, run_asha
from evoforge.search.pdh import PDHConfig, ProgressiveDynamicHurdles, PDHState


@dataclass
class SearchResult:
    asha: ASHAStats
    pdh_states: List[PDHState]


def load_candidates(paths: Iterable[Path], base_steps: int) -> List[Candidate]:
    candidates: List[Candidate] = []
    for path in paths:
        cfg = load_validate_yaml(path)
        candidates.append(Candidate(cfg_path=path, config=cfg, train_steps=base_steps))
    return candidates


def run_search(
    cfg_paths: Iterable[Path],
    *,
    asha_config: Optional[ASHAConfig] = None,
    pdh_config: Optional[PDHConfig] = None,
    device: Optional[str] = None,
    seq_len: int = 64,
    batch_size: int = 4,
) -> SearchResult:
    asha_config = asha_config or ASHAConfig()
    pdh_config = pdh_config or PDHConfig()

    candidates = load_candidates(cfg_paths, base_steps=asha_config.min_steps)
    asha_stats = run_asha(
        candidates,
        asha_cfg=asha_config,
        device=device,
        seq_len=seq_len,
        batch_size=batch_size,
    )

    pdh = ProgressiveDynamicHurdles(
        pdh_config,
        device=device,
        seq_len=seq_len,
        batch_size=batch_size,
    )
    pdh_states: List[PDHState] = []

    for candidate in candidates:
        pdh_states.append(pdh.run(candidate.cfg_path))

    return SearchResult(asha=asha_stats, pdh_states=pdh_states)
