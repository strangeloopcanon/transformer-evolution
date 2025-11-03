from __future__ import annotations

from pathlib import Path

import pytest
import torch

from evoforge.builders import build_simple_model
from evoforge.dsl.api import DSLValidationError, load_validate_yaml


def test_simple_builder_forward_shape() -> None:
    cfg = load_validate_yaml(Path("examples/nanogpt_tiny.yaml"))
    model, meta = build_simple_model(cfg)
    vocab = meta.vocab_size
    input_ids = torch.randint(0, vocab, (2, 16))
    logits = model(input_ids)
    assert logits.shape == (2, 16, vocab)


def test_simple_builder_conditional_warning() -> None:
    cfg = load_validate_yaml(Path("examples/latent.yaml"))
    model, meta = build_simple_model(cfg)
    assert meta.dim == cfg.arch.d_model
