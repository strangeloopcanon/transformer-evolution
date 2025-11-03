from __future__ import annotations

from pathlib import Path

import pytest

from evoforge.search.asha import ASHAConfig
from evoforge.search.pdh import PDHConfig
from evoforge.search.runner import run_search


def test_run_search_smoke(tmp_path):
    cfgs = [Path("examples/nanogpt_tiny.yaml"), Path("examples/transformer_vaswani.yaml")]
    result = run_search(
        cfgs,
        asha_config=ASHAConfig(min_steps=1, max_steps=2, reduction_factor=2),
        pdh_config=PDHConfig(base_steps=1, max_stages=2),
        device="cpu",
        seq_len=32,
        batch_size=2,
    )
    assert result.asha.evaluated > 0
    assert len(result.pdh_states) == len(cfgs)
    for state in result.pdh_states:
        assert state.history
