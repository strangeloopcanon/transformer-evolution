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
    assert torch.isfinite(logits).all()


def test_retention_and_ssm_mixers(tmp_path) -> None:
    cfg_text = """
    arch:
      d_model: 128
      n_layers: 2
      norm: RMSNorm
      mix_unit:
        kind: single
        mixer: { kind: Attention, heads: 4, groups: 2, pos: rope }
      ffn: { kind: dense, mult: 2.0, act: gelu }
      pos: { kind: rope, rope: { theta: 10000, dims: 32 } }
      modules:
        route_block:
          kind: transformer
          d_model: 128
          n_layers: 2
          mix_unit:
            kind: route
            choices:
              - { kind: Retention, chunk: 8, mode: parallel }
              - { kind: SSM, d_state: 16, expand: 1.5 }
              - { kind: Attention, heads: 4, groups: 2, stencil: { kind: local, window: 16 } }
            router: { topk: 2, temp: 0.8 }
            merge: Add
          ffn: { kind: dense, mult: 2.0, act: gelu }
          norm: RMSNorm
          pos: { kind: rope, rope: { theta: 10000, dims: 32 } }
      pipeline:
        - name: embeddings
          kind: embedding
        - name: decoder
          kind: module
          module: route_block
        - name: readout
          kind: readout
    train:
      ctx_len: 128
      dtype: fp16
      vocab_size: 64
    """
    cfg_path = tmp_path / "ret_ssm.yaml"
    cfg_path.write_text(cfg_text)
    cfg = load_validate_yaml(cfg_path)
    model, meta = build_model(cfg, vocab_size=64)
    input_ids = torch.randint(0, 64, (2, 16))
    logits = model(input_ids)
    assert logits.shape == (2, 16, meta.vocab_size)
    assert torch.isfinite(logits).all()
    loss = logits.mean()
    loss.backward()
