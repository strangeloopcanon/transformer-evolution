from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

from evoforge.dsl.errors import DSLValidationError
from evoforge.dsl.models import DSLConfig
from evoforge.train.simple_trainer import TrainResult, run_micro_train


@dataclass
class Candidate:
    cfg_path: Path
    config: DSLConfig
    train_steps: int
    score: float = math.inf
    result: Optional[TrainResult] = None


@dataclass
class ASHAConfig:
    max_steps: int = 100
    reduction_factor: int = 3
    min_steps: int = 5
    bracket: Optional[int] = None


@dataclass
class ASHAStats:
    evaluated: int
    promoted: int
    best_score: float
    history: List[Dict]


ScoreFn = Callable[[TrainResult], float]

DEFAULT_SCORE_FN: ScoreFn = lambda result: result.loss_history[-1]


def _compute_qpc(result: TrainResult) -> float:
    if not result.loss_history:
        return float("nan")
    if result.total_flops <= 0:
        return float("nan")
    return (result.loss_history[0] - result.loss_history[-1]) / result.total_flops


def _run_candidate(
    candidate: Candidate, steps: int, *, device: Optional[str], seq_len: int, batch_size: int
) -> TrainResult:
    return run_micro_train(
        candidate.cfg_path,
        steps=steps,
        device=device,
        seq_len=seq_len,
        batch_size=batch_size,
    )


def run_asha(
    candidates: Iterable[Candidate],
    *,
    asha_cfg: ASHAConfig,
    score_fn: ScoreFn = DEFAULT_SCORE_FN,
    device: Optional[str] = None,
    seq_len: int = 64,
    batch_size: int = 4,
) -> ASHAStats:
    brackets: Dict[int, List[Candidate]] = {}
    evaluated = 0
    history: List[Dict] = []

    for candidate in candidates:
        bracket = asha_cfg.bracket or candidate.train_steps
        brackets.setdefault(bracket, []).append(candidate)

    promoted = 0
    best_score = math.inf

    for bracket, bucket in brackets.items():
        bucket.sort(key=lambda c: c.train_steps)
        milestones: List[int] = []
        steps = asha_cfg.min_steps
        while steps <= asha_cfg.max_steps:
            milestones.append(steps)
            steps *= asha_cfg.reduction_factor
        for milestone in milestones:
            results = []
            for candidate in bucket:
                try:
                    result = _run_candidate(
                        candidate,
                        steps=milestone,
                        device=device,
                        seq_len=seq_len,
                        batch_size=batch_size,
                    )
                    candidate.result = result
                    candidate.score = score_fn(result)
                    history_entry = {
                        "cfg_path": str(candidate.cfg_path),
                        "steps": milestone,
                        "score": candidate.score,
                        "tokens": candidate.result.total_tokens,
                        "loss_start": result.loss_history[0] if result.loss_history else None,
                        "loss_final": result.loss_history[-1] if result.loss_history else None,
                        "qpc": _compute_qpc(result),
                        "metrics": result.metadata.get("metrics", {}),
                    }
                except DSLValidationError as err:
                    candidate.result = None
                    candidate.score = math.inf
                    history_entry = {
                        "cfg_path": str(candidate.cfg_path),
                        "steps": milestone,
                        "score": candidate.score,
                        "tokens": 0,
                        "error": str(err),
                    }
                except Exception as err:
                    candidate.result = None
                    candidate.score = math.inf
                    history_entry = {
                        "cfg_path": str(candidate.cfg_path),
                        "steps": milestone,
                        "score": candidate.score,
                        "tokens": 0,
                        "error": f"{err.__class__.__name__}: {err}",
                    }
                    print(
                        f"[evoforge] candidate {candidate.cfg_path} failed at {milestone} steps: {err}",
                        flush=True,
                    )
                evaluated += 1
                history.append(history_entry)
                results.append(candidate)
            results.sort(key=lambda c: c.score)
            survivors = math.floor(len(results) / asha_cfg.reduction_factor)
            survivors = max(survivors, 1)
            bucket = results[:survivors]
            promoted += len(bucket)
            if bucket[0].score < best_score:
                best_score = bucket[0].score

    return ASHAStats(
        evaluated=evaluated,
        promoted=promoted,
        best_score=best_score,
        history=history,
    )
