from __future__ import annotations

from pathlib import Path

import torch

from evoforge.builders import build_model
from evoforge.dsl.api import load_validate_yaml


def test_pipeline_builder_forward() -> None:
    cfg = load_validate_yaml(Path("examples/free_transformer_pipeline.yaml"))
    model, meta = build_model(cfg, vocab_size=32)
    assert meta.vocab_size == 32
    input_ids = torch.randint(0, meta.vocab_size, (2, 16))
    logits = model(input_ids)
    assert logits.shape == (2, 16, meta.vocab_size)
