from __future__ import annotations

# Minimal JSON Schema mirroring the spec (subset, focused on flexibility).

DSL_JSON_SCHEMA: dict = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Transformer DSL v0.1",
    "type": "object",
    "required": ["arch", "train"],
    "properties": {
        "arch": {
            "type": "object",
            "properties": {
                "d_model": {"type": "integer", "minimum": 64},
                "n_layers": {"type": "integer", "minimum": 1},
                "mix_unit": {
                    "type": "object",
                    "properties": {
                        "kind": {"enum": ["single", "par", "route"]},
                        "mixer": {"$ref": "#/definitions/mixer"},
                        "choices": {
                            "type": "array",
                            "items": {"$ref": "#/definitions/mixer"},
                        },
                        "merge": {"enum": ["Add", "WeightedAdd", "Concat"]},
                        "router": {"$ref": "#/definitions/router"},
                    },
                    "required": ["kind"],
                },
                "ffn": {"$ref": "#/definitions/ffn"},
                "norm": {"enum": ["LayerNorm", "RMSNorm", "ScaleNorm"]},
                "pos": {"$ref": "#/definitions/pos"},
                "cond": {"$ref": "#/definitions/cond"},
                "kv_policy": {"$ref": "#/definitions/kv"},
                "residual": {"$ref": "#/definitions/residual"},
                "hierarchy": {"$ref": "#/definitions/hierarchy"},
                "depth_router": {"$ref": "#/definitions/depth_router"},
                "modules": {
                    "type": "object",
                    "additionalProperties": {"$ref": "#/definitions/module"},
                },
                "pipeline": {
                    "type": "array",
                    "items": {"$ref": "#/definitions/pipeline_stage"},
                },
            },
            "required": ["d_model", "n_layers", "mix_unit", "ffn", "norm", "pos"],
            "additionalProperties": True,
        },
        "train": {
            "type": "object",
            "properties": {
                "lr": {"type": "number"},
                "wd": {"type": "number"},
                "betas": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 2,
                    "maxItems": 2,
                },
                "clip": {"type": "number"},
                "ctx_len": {"type": "integer", "minimum": 128},
                "dtype": {"enum": ["fp16"]},
                "vocab_size": {"type": "integer", "minimum": 8},
                "budget": {
                    "type": "object",
                    "properties": {
                        "tokens_per_step": {"type": "number"},
                        "max_steps": {"type": "integer", "minimum": 1},
                        "flops_per_step": {"type": "number"},
                    },
                },
            },
            "required": ["ctx_len"],
        },
        "budget": {
            "type": "object",
            "properties": {
                "params_target": {"type": "number"},
                "flops_target": {"type": ["number", "string"]},
            },
        },
    },
    "definitions": {
        "mixer": {
            "type": "object",
            "properties": {
                "kind": {"enum": ["Attention", "Retention", "SSM", "LongConv"]},
                "heads": {"type": "integer", "minimum": 1},
                "groups": {"type": "integer", "minimum": 1},
                "head_dim": {"type": "integer", "minimum": 8},
                "stencil": {"$ref": "#/definitions/stencil"},
                "softmax": {"$ref": "#/definitions/softmax"},
                "pos": {"type": "string"},
                "chunk": {"type": "integer"},
                "mode": {"enum": ["parallel", "recurrent"]},
                "d_state": {"type": "integer"},
                "expand": {"type": "number"},
                "kernel_len": {"type": "integer"},
                "projection": {"$ref": "#/definitions/projection"},
                "value_glu": {"type": "boolean"},
            },
            "required": ["kind"],
        },
        "ffn": {
            "type": "object",
            "properties": {
                "kind": {"enum": ["dense", "moe"]},
                "mult": {"type": "number"},
                "act": {"enum": ["relu", "gelu", "silu", "swiglu", "geglu"]},
                "experts": {"type": "integer"},
                "topk": {"type": "integer"},
                "capacity": {"type": "number"},
            },
            "required": ["kind", "mult", "act"],
        },
        "pos": {
            "type": "object",
            "properties": {
                "kind": {
                    "enum": ["rope", "alibi", "relbias", "learned", "xpos"],
                },
                "rope": {
                    "type": "object",
                    "properties": {
                        "theta": {"type": "number"},
                        "dims": {"type": "integer"},
                        "scaling": {
                            "type": "object",
                            "properties": {
                                "type": {
                                    "enum": ["ntk", "linear", "yarn", "longrope"],
                                },
                                "factor": {"type": "number"},
                                "longrope": {
                                    "type": "object",
                                    "properties": {
                                        "alpha": {"type": "number"},
                                        "beta": {"type": "number"},
                                    },
                                },
                            },
                        },
                    },
                },
            },
            "required": ["kind"],
        },
        "cond": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "object",
                    "properties": {
                        "kind": {"enum": ["pool-mlp", "segment"]},
                        "H": {"type": "integer"},
                        "segment": {"type": "integer"},
                    },
                },
                "reg": {
                    "type": "object",
                    "properties": {
                        "kind": {"enum": ["freebits", "none"]},
                        "kappa": {"type": "number"},
                    },
                },
                "ops": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "where": {
                                "enum": [
                                    "pre_mixer",
                                    "post_mixer",
                                    "ln",
                                    "proj_q",
                                    "proj_v",
                                    "q",
                                    "v",
                                    "token",
                                ]
                            },
                            "op": {"enum": ["film", "add", "scale", "lora"]},
                            "share": {
                                "enum": ["global", "per_channel", "per_head"],
                            },
                            "r": {"type": "integer"},
                        },
                        "required": ["where", "op"],
                    },
                },
            },
        },
        "router": {
            "type": "object",
            "properties": {
                "topk": {"type": "integer"},
                "temp": {"type": "number"},
                "balance": {"type": "number"},
            },
        },
        "kv": {
            "type": "object",
            "properties": {
                "cache": {"enum": ["full", "window", "ring", "none", "latent"]},
                "window": {"type": "integer"},
                "quant": {"enum": ["none", "fp8", "nf4", "int8"]},
                "evict": {"type": "string"},
                "latent": {"$ref": "#/definitions/latent_kv"},
            },
        },
        "softmax": {
            "type": "object",
            "properties": {
                "type": {"enum": ["standard", "kernel", "scaled"]},
                "qk_scale": {"type": ["number", "string"]},
                "qk_norm": {"enum": ["none", "rms", "layer"]},
                "softcap": {"type": "number"},
                "kernel": {
                    "type": "object",
                    "properties": {
                        "name": {"enum": ["favor", "gaussian", "laplace"]},
                        "features": {"type": "integer"},
                        "redraw": {"type": "integer"},
                        "orthogonal": {"type": "boolean"},
                    },
                },
            },
        },
        "projection": {
            "type": "object",
            "properties": {
                "type": {"enum": ["low_rank", "none"]},
                "rank": {"type": "integer"},
                "shared": {"type": "boolean"},
            },
        },
        "stencil": {
            "type": "object",
            "properties": {
                "kind": {
                    "enum": [
                        "full",
                        "local",
                        "dilated",
                        "block",
                        "ring",
                        "hybrid",
                        "sliding",
                        "cross",
                    ]
                },
                "window": {"type": "integer"},
                "dilation": {"type": "integer"},
                "block": {"type": "integer"},
                "stride": {"type": "integer"},
                "globals": {"type": "integer"},
                "query": {"type": "string"},
                "key": {"type": "string"},
            },
        },
        "residual": {
            "type": "object",
            "properties": {
                "kind": {"enum": ["single", "dual", "deepnet"]},
                "pre_ln": {"type": "boolean"},
                "post_ln": {"type": "boolean"},
                "scale": {"type": "number"},
            },
        },
        "hierarchy": {
            "type": "object",
            "properties": {
                "levels": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "every": {"type": "integer", "minimum": 1},
                            "downsample": {"type": "number"},
                            "up_proj": {"type": "boolean"},
                        },
                    },
                },
            },
        },
        "depth_router": {
            "type": "object",
            "properties": {
                "kind": {"enum": ["token", "layer", "none"]},
                "budget": {"type": "number"},
                "tau": {"type": "number"},
                "min_layers": {"type": "integer"},
            },
        },
        "latent_kv": {
            "type": "object",
            "properties": {
                "dim": {"type": "integer"},
                "heads": {"type": "integer"},
                "update": {
                    "type": "object",
                    "properties": {
                        "kind": {"enum": ["linear", "ema"]},
                        "freq": {"type": "integer"},
                        "tau": {"type": "number"},
                    },
                },
            },
        },
        "module": {
            "type": "object",
            "properties": {
                "kind": {
                    "enum": [
                        "transformer",
                        "latent_sampler",
                        "readout",
                        "embedding",
                        "custom",
                    ]
                },
                "d_model": {"type": "integer"},
                "n_layers": {"type": "integer"},
                "mix_unit": {"$ref": "#/definitions/mix_unit_ref"},
                "ffn": {"$ref": "#/definitions/ffn"},
                "norm": {"enum": ["LayerNorm", "RMSNorm", "ScaleNorm"]},
                "pos": {"$ref": "#/definitions/pos"},
                "cond": {"$ref": "#/definitions/cond"},
                "kv_policy": {"$ref": "#/definitions/kv"},
                "residual": {"$ref": "#/definitions/residual"},
                "hierarchy": {"$ref": "#/definitions/hierarchy"},
                "depth_router": {"$ref": "#/definitions/depth_router"},
                "latent": {"type": "object"},
                "output_dim": {"type": "integer"},
                "budget": {
                    "type": "object",
                    "properties": {
                        "tokens_per_step": {"type": "number"},
                        "flops_per_token": {"type": "number"},
                        "latency_ms": {"type": "number"},
                    },
                },
            },
            "additionalProperties": True,
        },
        "mix_unit_ref": {
            "type": "object",
            "properties": {
                "kind": {"enum": ["single", "par", "route"]},
                "mixer": {"$ref": "#/definitions/mixer"},
                "choices": {
                    "type": "array",
                    "items": {"$ref": "#/definitions/mixer"},
                },
                "merge": {"enum": ["Add", "WeightedAdd", "Concat"]},
                "router": {"$ref": "#/definitions/router"},
            },
            "required": ["kind"],
        },
        "pipeline_stage": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "module": {"type": "string"},
                "kind": {
                    "enum": [
                        "module",
                        "latent_sampler",
                        "readout",
                        "embedding",
                        "custom",
                    ]
                },
                "repeat": {"type": "integer", "minimum": 1},
                "mode": {
                    "enum": [
                        "all",
                        "train",
                        "prefill",
                        "decode",
                        "train_prefill",
                        "inference",
                    ]
                },
                "inputs": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "kv_from": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "mem_from": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "train_only": {"type": "boolean"},
                "prefill_only": {"type": "boolean"},
                "params": {"type": "object"},
                "budget": {
                    "type": "object",
                    "properties": {
                        "tokens_multiplier": {"type": "number"},
                        "compute_multiplier": {"type": "number"},
                    },
                },
            },
            "required": ["name"],
        },
    },
}
