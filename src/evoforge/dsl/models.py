from __future__ import annotations

from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class Router(BaseModel):
    topk: Optional[int] = None
    temp: Optional[float] = None
    balance: Optional[float] = None


class LatentUpdate(BaseModel):
    kind: Optional[Literal["linear", "ema"]] = None
    freq: Optional[int] = None
    tau: Optional[float] = None


class LatentKV(BaseModel):
    dim: Optional[int] = None
    heads: Optional[int] = None
    update: Optional[LatentUpdate] = None


class KVPolicy(BaseModel):
    cache: Optional[Literal["full", "window", "ring", "none", "latent"]] = None
    window: Optional[int] = None
    quant: Optional[Literal["none", "fp8", "nf4", "int8"]] = None
    evict: Optional[str] = None
    latent: Optional[LatentKV] = None


class LongRopeParams(BaseModel):
    alpha: Optional[float] = None
    beta: Optional[float] = None


class RopeScaling(BaseModel):
    type: Literal["ntk", "linear", "yarn", "longrope"]
    factor: Optional[float] = None
    longrope: Optional[LongRopeParams] = None


class PosRope(BaseModel):
    theta: Optional[float] = None
    dims: Optional[int] = None
    scaling: Optional[RopeScaling] = None


class PosConfig(BaseModel):
    kind: Literal["rope", "alibi", "relbias", "learned", "xpos"]
    rope: Optional[PosRope] = None


class SoftmaxKernel(BaseModel):
    name: Optional[Literal["favor", "gaussian", "laplace"]] = None
    features: Optional[int] = None
    redraw: Optional[int] = None
    orthogonal: Optional[bool] = None


class SoftmaxConfig(BaseModel):
    type: Optional[Literal["standard", "kernel", "scaled"]] = None
    qk_scale: Optional[Union[float, str]] = None
    qk_norm: Optional[Literal["none", "rms", "layer"]] = None
    softcap: Optional[float] = None
    kernel: Optional[SoftmaxKernel] = None


class ProjectionConfig(BaseModel):
    type: Optional[Literal["low_rank", "none"]] = None
    rank: Optional[int] = None
    shared: Optional[bool] = None


class StencilConfig(BaseModel):
    kind: Optional[
        Literal[
            "full",
            "local",
            "dilated",
            "block",
            "ring",
            "hybrid",
            "sliding",
            "cross",
        ]
    ] = None
    window: Optional[int] = None
    dilation: Optional[int] = None
    block: Optional[int] = None
    stride: Optional[int] = None
    globals: Optional[int] = None
    query: Optional[str] = None
    key: Optional[str] = None


class Mixer(BaseModel):
    kind: Literal["Attention", "Retention", "SSM", "LongConv"]
    heads: Optional[int] = None
    groups: Optional[int] = Field(default=None, ge=1)
    head_dim: Optional[int] = None
    stencil: Optional[StencilConfig] = None
    softmax: Optional[SoftmaxConfig] = None
    pos: Optional[str] = None
    chunk: Optional[int] = None
    mode: Optional[Literal["parallel", "recurrent"]] = None
    d_state: Optional[int] = None
    expand: Optional[float] = None
    kernel_len: Optional[int] = None
    projection: Optional[ProjectionConfig] = None
    value_glu: Optional[bool] = None


class MixUnit(BaseModel):
    kind: Literal["single", "par", "route"]
    mixer: Optional[Mixer] = None
    choices: Optional[List[Mixer]] = None
    merge: Optional[Literal["Add", "WeightedAdd", "Concat"]] = None
    router: Optional[Router] = None


class FFN(BaseModel):
    kind: Literal["dense", "moe"]
    mult: float
    act: Literal["relu", "gelu", "silu", "swiglu", "geglu"]
    experts: Optional[int] = None
    topk: Optional[int] = None
    capacity: Optional[float] = None


class CondSource(BaseModel):
    kind: Literal["pool-mlp", "segment"]
    H: Optional[int] = None
    segment: Optional[int] = None


class CondOp(BaseModel):
    where: Literal[
        "pre_mixer",
        "post_mixer",
        "ln",
        "proj_q",
        "proj_v",
        "q",
        "v",
        "token",
    ]
    op: Literal["film", "add", "scale", "lora"]
    share: Optional[Literal["global", "per_channel", "per_head"]] = None
    r: Optional[int] = None


class CondReg(BaseModel):
    kind: Literal["freebits", "none"]
    kappa: Optional[float] = None


class Cond(BaseModel):
    source: Optional[CondSource] = None
    reg: Optional[CondReg] = None
    ops: Optional[List[CondOp]] = None


class ResidualConfig(BaseModel):
    kind: Literal["single", "dual", "deepnet"] = "single"
    pre_ln: Optional[bool] = None
    post_ln: Optional[bool] = None
    scale: Optional[float] = None


class HierarchyLevel(BaseModel):
    every: int = Field(..., ge=1)
    downsample: Optional[float] = None
    up_proj: Optional[bool] = None


class HierarchyConfig(BaseModel):
    levels: List[HierarchyLevel]


class DepthRouter(BaseModel):
    kind: Literal["token", "layer", "none"] = "none"
    budget: Optional[float] = None
    tau: Optional[float] = None
    min_layers: Optional[int] = Field(default=None, ge=0)


class ModuleSpec(BaseModel):
    kind: Literal["transformer", "latent_sampler", "readout", "embedding", "custom"] = "transformer"
    d_model: Optional[int] = None
    n_layers: Optional[int] = None
    mix_unit: Optional[MixUnit] = None
    ffn: Optional[FFN] = None
    norm: Optional[Literal["LayerNorm", "RMSNorm", "ScaleNorm"]] = None
    pos: Optional[PosConfig] = None
    cond: Optional[Cond] = None
    kv_policy: Optional[KVPolicy] = None
    residual: Optional[ResidualConfig] = None
    hierarchy: Optional[HierarchyConfig] = None
    depth_router: Optional[DepthRouter] = None
    latent: Optional[dict] = None
    output_dim: Optional[int] = None
    params: Optional[dict] = None
    budget: Optional[Dict[str, float]] = None


class PipelineStage(BaseModel):
    name: str
    module: Optional[str] = None
    kind: Optional[Literal["module", "latent_sampler", "readout", "embedding", "custom"]] = None
    repeat: Optional[int] = Field(default=None, ge=1)
    mode: Optional[Literal["all", "train", "prefill", "decode", "train_prefill", "inference"]] = (
        None
    )
    inputs: Optional[List[str]] = None
    kv_from: Optional[List[str]] = None
    mem_from: Optional[List[str]] = None
    train_only: Optional[bool] = None
    prefill_only: Optional[bool] = None
    params: Optional[dict] = None
    budget: Optional[Dict[str, float]] = None


class Arch(BaseModel):
    d_model: int
    n_layers: int
    mix_unit: MixUnit
    ffn: FFN
    norm: Literal["LayerNorm", "RMSNorm", "ScaleNorm"]
    pos: PosConfig
    cond: Optional[Cond] = None
    kv_policy: Optional[KVPolicy] = None
    residual: Optional[ResidualConfig] = None
    hierarchy: Optional[HierarchyConfig] = None
    depth_router: Optional[DepthRouter] = None
    modules: Optional[Dict[str, ModuleSpec]] = None
    pipeline: Optional[List[PipelineStage]] = None


class Train(BaseModel):
    ctx_len: int
    lr: Optional[float] = None
    wd: Optional[float] = None
    betas: Optional[List[float]] = None
    clip: Optional[float] = None
    dtype: Literal["fp16"] = "fp16"  # keep it simple, per user request
    vocab_size: Optional[int] = Field(default=None, ge=8)
    budget: Optional[Dict[str, float]] = None


class Budget(BaseModel):
    params_target: Optional[float] = None
    flops_target: Optional[Union[float, str]] = None


class DSLConfig(BaseModel):
    arch: Arch
    train: Train
    budget: Optional[Budget] = None
