from __future__ import annotations

from pathlib import Path

import textwrap

import pytest

from evoforge.dsl.errors import DSLValidationError
from evoforge.train.simple_trainer import run_micro_train


def test_run_micro_train_smoke(tmp_path):
    cfg_path = Path("examples/nanogpt_tiny.yaml")
    result = run_micro_train(cfg_path, steps=2, seq_len=32, batch_size=2, device="cpu")
    assert len(result.loss_history) == 2
    assert result.tokens_per_sec > 0
    assert result.total_tokens == 2 * 2 * 32
    assert result.total_flops > 0
    assert "metrics" in result.metadata
    metrics = result.metadata["metrics"]
    assert "loss_final" in metrics
    assert "ece" in metrics
    arch_meta = result.metadata.get("architecture", {})
    assert isinstance(arch_meta, dict)


def test_run_micro_train_budget_guard(tmp_path):
    cfg_text = textwrap.dedent(
        """
        arch:
          d_model: 192
          n_layers: 2
          norm: RMSNorm
          mix_unit:
            kind: "single"
            mixer: { kind: "Attention", heads: 3, groups: 3, pos: "rope" }
          ffn: { kind: "dense", mult: 2.0, act: "gelu" }
          pos: { kind: "rope", rope: { theta: 10000, dims: 32 } }
        train:
          ctx_len: 64
          dtype: fp16
          vocab_size: 64
          budget: { tokens_per_step: 32, max_steps: 10 }
        """
    )
    cfg_path = tmp_path / "budget.yaml"
    cfg_path.write_text(cfg_text)
    with pytest.raises(DSLValidationError):
        run_micro_train(cfg_path, steps=2, seq_len=32, batch_size=2, device="cpu")
