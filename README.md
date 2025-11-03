# EvoForge Search Toolkit

## Objective

EvoForge lets us describe Transformer-family architectures in a single DSL, mutate them, and evaluate candidates automatically. The current goal is to run evolutionary sweeps (ASHA for breadth, PDH for depth) over that DSL and surface high-quality pipeline designs such as Free Transformer variants.

## Setup

1. Create a virtual env and install requirements:
   ```bash
   python3 -m venv .venv
   . .venv/bin/activate
   pip install -r requirements.txt -r requirements-dev.txt
   ```
2. Run the smoke tests:
   ```bash
   make test
   ```

## Key Commands

* Validate a config: `PYTHONPATH=src .venv/bin/python runners/validate.py --cfg examples/free_transformer_pipeline.yaml`
* Micro-train a config: `PYTHONPATH=src .venv/bin/python runners/train_tiny.py --cfg examples/nanogpt_tiny.yaml --steps 20`
* Full ASHA → PDH sweep: `PYTHONPATH=src .venv/bin/python runners/run_experiment.py <cfgs…> [options]`
* Mutation-based evolution: `PYTHONPATH=src .venv/bin/python runners/run_evolution.py configs examples --generations 4 --population 8 --top-k 4`

Results (metrics, tokens, FLOPs) are written to `results/search_report.json`, and mutated DSL snapshots live under `results/evolution/gen_*/variant_*.yaml`.

## Latest Results (CPU, seq_len=192, batch=6)

| Config | Loss@120 steps | QPC (Δloss/FLOPs) | Tokens/sec |
| --- | --- | --- | --- |
| `results/evolution/gen_7/variant_7.yaml` | **8.27e-5** | 5.30e-13 | 2.10k |
| `results/evolution/gen_2/variant_4.yaml` | 1.70e-4 | 5.52e-13 | 2.17k |
| `results/evolution/gen_8/variant_7.yaml` | 1.22e-4 | 5.46e-13 | 2.17k |
| `configs/free_transformer_alt.yaml` | 2.30e-3 | 5.35e-13 | 2.05k |

The best architecture (`gen_7/variant_7`) is an evolved Free Transformer pipeline: encoder → latent sampler → routed lower decoder (Attention/Retention/SSM mix, top‑k gating) → full-attention upper decoder with RMS QK-norm and windowed KV caching.

### Architecture sketch (`gen_7/variant_7`)

```mermaid
graph TD
    E[Embeddings<br/>RoPE dims=128] --> ENC[Encoder<br/>Non-causal Attention<br/>4 layers]
    ENC --> LAT[Latent Sampler<br/>FiLM + LoRA r=4]
    ENC --> DECLOW
    LAT --> DECLOW
    DECLOW[Decoder Lower<br/>Router top-k=2<br/>Attention (local 1024)<br/>Retention chunk 1024<br/>SSM d_state=32] --> DECUP
    LAT --> DECUP
    DECUP[Decoder Upper<br/>Attention + RMS QK-norm] --> OUT[Readout]
```

* Lower decoder keeps a windowed KV cache (`window=8192`, `nf4` quantisation).
* Router balances attention, retention, and SSM experts with temperature 0.7.
* Upper decoder relies on standard causal attention with RMS QK-norm.

## Next Steps

* Improve mixer fidelity further (GPU-friendly Retention/SSM kernels with state reuse).
* Scale mutation/evolution runs (more generations, larger populations, periodic long budgets) once GPU resources are available.
