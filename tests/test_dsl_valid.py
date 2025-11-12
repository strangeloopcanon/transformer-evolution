from __future__ import annotations

from pathlib import Path

import pytest

from evoforge.dsl.api import DSLValidationError, load_validate_yaml


EXAMPLES = [
    "examples/plain.yaml",
    "examples/latent.yaml",
    "examples/retnet.yaml",
    "examples/route.yaml",
    "examples/nanogpt_tiny.yaml",
    "examples/transformer_vaswani.yaml",
    "examples/free_transformer.yaml",
    "examples/residual_dual.yaml",
    "examples/linformer.yaml",
    "examples/performer.yaml",
    "examples/ring_attention.yaml",
    "examples/mixture_depth.yaml",
    "examples/mla.yaml",
    "examples/hierarchical.yaml",
    "examples/free_transformer_pipeline.yaml",
    "examples/recurrence.yaml",
    "examples/pipeline_recurrence.yaml",
]


@pytest.mark.parametrize("path", EXAMPLES)
def test_examples_validate(path: str) -> None:
    cfg = load_validate_yaml(Path(path))
    assert cfg.arch.d_model >= 64
    assert cfg.train.dtype == "fp16"


def test_missing_required_fails(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("train: {ctx_len: 512, dtype: fp16}")
    with pytest.raises(DSLValidationError):
        load_validate_yaml(p)


def test_route_requires_router(tmp_path: Path) -> None:
    p = tmp_path / "route_bad.yaml"
    p.write_text(
        """
arch:
  d_model: 256
  n_layers: 2
  norm: RMSNorm
  mix_unit:
    kind: "route"
    choices:
      - { kind: "Attention", heads: 4 }
      - { kind: "SSM", d_state: 8 }
  ffn: { kind: "dense", mult: 2.0, act: "relu" }
  pos: { kind: "rope" }
train: { ctx_len: 512, dtype: fp16 }
        """
    )
    with pytest.raises(DSLValidationError):
        load_validate_yaml(p)


def test_lora_requires_rank(tmp_path: Path) -> None:
    p = tmp_path / "lora_bad.yaml"
    p.write_text(
        """
arch:
  d_model: 256
  n_layers: 2
  norm: RMSNorm
  mix_unit:
    kind: "single"
    mixer: { kind: "Attention", heads: 4 }
  ffn: { kind: "dense", mult: 2.0, act: "relu" }
  pos: { kind: "rope" }
  cond:
    ops:
      - { where: "proj_q", op: "lora" }
train: { ctx_len: 512, dtype: fp16 }
        """
    )
    with pytest.raises(DSLValidationError):
        load_validate_yaml(p)


def test_latent_requires_cache(tmp_path: Path) -> None:
    p = tmp_path / "latent_bad.yaml"
    p.write_text(
        """
arch:
  d_model: 256
  n_layers: 4
  norm: RMSNorm
  mix_unit:
    kind: "single"
    mixer: { kind: "Attention", heads: 4 }
  ffn: { kind: "dense", mult: 2.0, act: "relu" }
  pos: { kind: "rope" }
  kv_policy:
    cache: "full"
    latent: { dim: 64 }
train: { ctx_len: 512, dtype: fp16 }
        """
    )
    with pytest.raises(DSLValidationError):
        load_validate_yaml(p)


def test_depth_router_needs_budget(tmp_path: Path) -> None:
    p = tmp_path / "depth_router_bad.yaml"
    p.write_text(
        """
arch:
  d_model: 256
  n_layers: 4
  norm: RMSNorm
  mix_unit:
    kind: "single"
    mixer: { kind: "Attention", heads: 4 }
  ffn: { kind: "dense", mult: 2.0, act: "relu" }
  pos: { kind: "rope" }
  depth_router: { kind: "token" }
train: { ctx_len: 512, dtype: fp16 }
        """
    )
    with pytest.raises(DSLValidationError):
        load_validate_yaml(p)


def test_pipeline_missing_module(tmp_path: Path) -> None:
    p = tmp_path / "pipeline_bad.yaml"
    p.write_text(
        """
arch:
  d_model: 256
  n_layers: 4
  norm: RMSNorm
  mix_unit:
    kind: "single"
    mixer: { kind: "Attention", heads: 4 }
  ffn: { kind: "dense", mult: 2.0, act: "relu" }
  pos: { kind: "rope" }
  modules: {}
  pipeline:
    - { name: start, module: "missing" }
train: { ctx_len: 512, dtype: fp16 }
        """
    )
    with pytest.raises(DSLValidationError):
        load_validate_yaml(p)


def test_recurrence_layer_sum_guard(tmp_path: Path) -> None:
    p = tmp_path / "rec_bad.yaml"
    p.write_text(
        """
arch:
  d_model: 256
  n_layers: 4
  norm: RMSNorm
  mix_unit:
    kind: "single"
    mixer: { kind: "Attention", heads: 4 }
  ffn: { kind: "dense", mult: 2.0, act: "relu" }
  pos: { kind: "rope" }
  recurrence:
    prelude: 1
    body: 1
    coda: 1
train: { ctx_len: 512, dtype: fp16 }
        """
    )
    with pytest.raises(DSLValidationError):
        load_validate_yaml(p)


def test_recurrence_disallowed_with_pipeline(tmp_path: Path) -> None:
    p = tmp_path / "rec_pipeline.yaml"
    p.write_text(
        """
arch:
  d_model: 256
  n_layers: 4
  norm: RMSNorm
  mix_unit:
    kind: "single"
    mixer: { kind: "Attention", heads: 4 }
  ffn: { kind: "dense", mult: 2.0, act: "relu" }
  pos: { kind: "rope" }
  recurrence:
    prelude: 1
    body: 2
    coda: 1
  modules:
    enc: { kind: "transformer", n_layers: 2, mix_unit: { kind: "single", mixer: { kind: "Attention", heads: 4 } }, ffn: { kind: "dense", mult: 2.0, act: "relu" }, norm: "RMSNorm", pos: { kind: "rope" } }
  pipeline:
    - { name: tok, kind: "embedding", module: enc }
train: { ctx_len: 512, dtype: fp16 }
        """
    )
    with pytest.raises(DSLValidationError):
        load_validate_yaml(p)


def test_module_recurrence_requires_layers(tmp_path: Path) -> None:
    p = tmp_path / "module_rec_bad.yaml"
    p.write_text(
        """
arch:
  d_model: 256
  n_layers: 4
  norm: RMSNorm
  mix_unit:
    kind: "single"
    mixer: { kind: "Attention", heads: 4 }
  ffn: { kind: "dense", mult: 2.0, act: "relu" }
  pos: { kind: "rope" }
  modules:
    enc:
      kind: "transformer"
      mix_unit: { kind: "single", mixer: { kind: "Attention", heads: 4 } }
      ffn: { kind: "dense", mult: 2.0, act: "relu" }
      norm: RMSNorm
      pos: { kind: "rope" }
      recurrence: { prelude: 1, body: 1, coda: 1 }
  pipeline:
    - { name: tok, kind: "embedding" }
train: { ctx_len: 512, dtype: fp16 }
        """
    )
    with pytest.raises(DSLValidationError):
        load_validate_yaml(p)
