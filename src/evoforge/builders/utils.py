from __future__ import annotations

from collections import Counter
from typing import Dict

from evoforge.dsl.models import DSLConfig, MixUnit


def _tally_mix_unit(counter: Counter, mix_unit: MixUnit) -> None:
    if mix_unit is None:
        return
    if mix_unit.mixer is not None:
        counter[mix_unit.mixer.kind] += 1
    if mix_unit.choices:
        for choice in mix_unit.choices:
            counter[choice.kind] += 1
    # Nested mix-units are not currently supported, but we guard defensively.


def summarize_architecture(cfg: DSLConfig) -> Dict[str, object]:
    """
    Produce lightweight architecture metadata (used for logging and analysis).
    """
    counter: Counter = Counter()
    arch = cfg.arch
    _tally_mix_unit(counter, arch.mix_unit)

    has_router = bool(arch.mix_unit and arch.mix_unit.kind == "route")
    has_latent = False

    module_recurrence: Dict[str, dict] = {}

    if arch.modules:
        for module_name, module in arch.modules.items():
            if module.mix_unit:
                _tally_mix_unit(counter, module.mix_unit)
                if module.mix_unit.kind == "route":
                    has_router = True
            if module.kind == "latent_sampler" or module.cond is not None:
                has_latent = True
            if module.recurrence:
                loops = module.recurrence.loops
                module_recurrence[module_name] = {
                    "layout": {
                        "prelude": module.recurrence.prelude,
                        "body": module.recurrence.body,
                        "coda": module.recurrence.coda,
                    },
                    "adapter": module.recurrence.adapter,
                    "loops": {
                        "train": loops.train if loops else 1,
                        "eval": (loops.eval if loops and loops.eval else (loops.train if loops else 1)),
                    },
                }

    if arch.pipeline:
        for stage in arch.pipeline:
            if stage.kind == "latent_sampler":
                has_latent = True
            if stage.kind == "transformer":
                spec = arch.modules.get(stage.module) if stage.module else None
                if spec and spec.mix_unit and spec.mix_unit.kind == "route":
                    has_router = True

    recurrence = None
    if arch.recurrence:
        loops = arch.recurrence.loops
        recurrence = {
            "layout": {
                "prelude": arch.recurrence.prelude,
                "body": arch.recurrence.body,
                "coda": arch.recurrence.coda,
            },
            "adapter": arch.recurrence.adapter,
            "loops": {
                "train": loops.train if loops else 1,
                "eval": (loops.eval if loops and loops.eval else (loops.train if loops else 1)),
            },
        }

    return {
        "mixer_counts": dict(counter),
        "has_router": has_router,
        "has_latent": has_latent,
        "pipeline_stages": len(arch.pipeline) if arch.pipeline else 0,
        "recurrence": recurrence,
        "module_recurrence": module_recurrence or None,
    }
