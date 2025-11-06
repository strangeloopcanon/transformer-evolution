from typing import Optional, Tuple

from torch import nn

from evoforge.dsl.models import DSLConfig

from .pipeline_torch import build_pipeline_model
from .simple_torch import BuildMetadata, build_simple_model


def build_model(
    cfg: DSLConfig, *, vocab_size: Optional[int] = None
) -> Tuple[nn.Module, BuildMetadata]:
    if cfg.arch.pipeline:
        return build_pipeline_model(cfg, vocab_size=vocab_size)
    return build_simple_model(cfg, vocab_size=vocab_size)


__all__ = ["build_simple_model", "build_pipeline_model", "build_model", "BuildMetadata"]
