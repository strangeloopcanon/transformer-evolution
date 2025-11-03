from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

from evoforge.dsl.errors import DSLValidationError
from evoforge.train.simple_trainer import TrainResult, run_micro_train


@dataclass
class PDHConfig:
    base_steps: int = 10
    hurdle_growth: float = 1.3
    max_stages: int = 4
    patience: int = 1


ScoreFn = Callable[[TrainResult], float]
DEFAULT_SCORE_FN: ScoreFn = lambda result: result.loss_history[-1]


@dataclass
class PDHState:
    hurdles: List[float] = field(default_factory=list)
    history: List[Dict] = field(default_factory=list)
    best: Optional[Dict] = None


class ProgressiveDynamicHurdles:
    def __init__(
        self,
        cfg: PDHConfig,
        *,
        score_fn: ScoreFn = DEFAULT_SCORE_FN,
        device: Optional[str] = None,
        seq_len: int = 64,
        batch_size: int = 4,
    ) -> None:
        self.cfg = cfg
        self.score_fn = score_fn
        self.device = device
        self.seq_len = seq_len
        self.batch_size = batch_size

    def run(self, cfg_path: Path) -> PDHState:
        state = PDHState()
        steps = self.cfg.base_steps
        best_score = math.inf

        for stage in range(self.cfg.max_stages):
            error_msg: Optional[str] = None
            try:
                result = run_micro_train(
                    cfg_path,
                    steps=steps,
                    device=self.device,
                    seq_len=self.seq_len,
                    batch_size=self.batch_size,
                )
                score = self.score_fn(result)
            except DSLValidationError as err:
                result = None
                score = math.inf
                error_msg = str(err)
            state.history.append(
                {
                    "stage": stage,
                    "steps": steps,
                    "score": score,
                    "tokens": result.total_tokens if result else 0,
                    "flops": result.total_flops if result else 0.0,
                    "error": error_msg,
                }
            )
            if score == math.inf:
                state.hurdles.append(state.hurdles[-1] if state.hurdles else math.inf)
                break
            hurdle = score if not state.hurdles else min(state.hurdles[-1], score)
            state.hurdles.append(hurdle)
            if score < best_score:
                best_score = score
                state.best = {
                    "stage": stage,
                    "steps": steps,
                    "score": score,
                    "result": result,
                }
            # dynamic hurdle: must beat previous hurdle by growth factor
            target = hurdle / self.cfg.hurdle_growth
            if score > target:
                break
            steps = int(steps * self.cfg.hurdle_growth)
            steps = max(steps, self.cfg.base_steps)
        return state
