from __future__ import annotations

from typing import Iterable, List

import math

from evoforge.dsl.models import DSLConfig


def embed_architecture(cfg: DSLConfig) -> List[float]:
    """A lightweight, stable feature embedding of the DSL for novelty.

    Keeps it simple to avoid bias: counts of mixer kinds, router/latent flags,
    pipeline length, heads/groups ratio, rope dims.
    """
    arch = cfg.arch
    d = []
    # scalar basics
    d.append(float(arch.d_model))
    d.append(float(arch.n_layers))
    # attention shape if available
    mu = arch.mix_unit
    heads = 0.0
    groups = 0.0
    if mu:
        if mu.mixer and mu.mixer.kind == "Attention":
            heads = float(mu.mixer.heads or 0)
            groups = float(mu.mixer.groups or (mu.mixer.heads or 1))
        elif mu.choices:
            for ch in mu.choices:
                if ch.kind == "Attention":
                    heads = float(ch.heads or 0)
                    groups = float(ch.groups or (ch.heads or 1))
                    break
    d.append(heads)
    d.append(groups / (heads if heads else 1.0))
    # rope dims
    rope_dims = 0.0
    if arch.pos and arch.pos.kind == "rope" and arch.pos.rope and arch.pos.rope.dims:
        rope_dims = float(arch.pos.rope.dims)
    d.append(rope_dims)
    # mixer counts from modules/pipeline
    attn = 0.0
    retn = 0.0
    ssm = 0.0
    if arch.modules:
        for m in arch.modules.values():
            mu2 = m.mix_unit
            if mu2:
                if mu2.mixer:
                    kind = mu2.mixer.kind
                    if kind == "Attention":
                        attn += 1
                    elif kind == "Retention":
                        retn += 1
                    elif kind == "SSM":
                        ssm += 1
                if mu2.choices:
                    for ch in mu2.choices:
                        if ch.kind == "Attention":
                            attn += 1
                        elif ch.kind == "Retention":
                            retn += 1
                        elif ch.kind == "SSM":
                            ssm += 1
    d.extend([attn, retn, ssm])
    # flags + pipeline length
    has_router = 1.0 if (arch.mix_unit and arch.mix_unit.kind == "route") else 0.0
    has_latent = 1.0 if (arch.cond is not None) else 0.0
    if arch.modules:
        if any(m.kind == "latent_sampler" for m in arch.modules.values()):
            has_latent = 1.0
    d.append(has_router)
    d.append(has_latent)
    d.append(float(len(arch.pipeline) if arch.pipeline else 0))
    return d


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def novelty_score(vec: List[float], archive: Iterable[List[float]], k: int = 5) -> float:
    pool = list(archive)
    if not pool:
        return 0.0
    sims = sorted((_cosine(vec, v) for v in pool), reverse=True)
    # take k nearest (highest cosine), novelty = 1 - average similarity
    k = min(k, len(sims))
    if k == 0:
        return 0.0
    return 1.0 - (sum(sims[:k]) / k)
