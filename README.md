# Transformer Evolution Toolkit

## Objective

Transformer Evolution lets us describe Transformer-family architectures in a single DSL, mutate them, and evaluate candidates automatically. The current goal is to run evolutionary sweeps (ASHA for breadth, PDH for depth) over that DSL and surface high-quality pipeline designs such as Free Transformer variants.

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
| `results/evolution_overnight/gen_8/variant_9.yaml` | **1.14e-2** | 5.44e-13 | 2.05k |
| `results/evolution/gen_7/variant_7.yaml` | 8.27e-5 | 5.30e-13 | 2.10k |
| `results/evolution/gen_2/variant_4.yaml` | 1.70e-4 | 5.52e-13 | 2.17k |
| `configs/free_transformer_alt.yaml` | 2.30e-3 | 5.35e-13 | 2.05k |

The overnight sweep (10 generations, population 10 on MPS with CPU fallback) promoted `gen_8/variant_9` as the current frontier: still a Free Transformer skeleton, but with 9-way grouped query attention at the trunk, tighter RoPE (dims=32) in the shared positional core, and the routed lower decoder rebalanced across attention/retention/SSM experts. PDH deep-evaluation at 400 steps (`seq_len=256`, batch 6) finished with loss `1.17e-2`, ECE `3.2e-3`, and 0.998 accuracy on the calibration slice.

### Architecture sketch (`gen_8/variant_9`)

```mermaid
flowchart TD
    E["Embeddings + RoPE (dims=32 core / 128 module)"]
    ENC["Encoder (non-causal attention, 4 layers, GQA)"]
    LAT["Latent sampler (FiLM + LoRA r=4, H=16)"]
    DECLOW["Decoder lower:\nrouter top-k=2\nlocal attention (window 1024)\nretention chunk 1024\nSSM d_state=32"]
    DECUP["Decoder upper (attention with RMS QK-norm)"]
    OUT["Readout"]
    E --> ENC
    ENC --> LAT
    ENC --> DECLOW
    LAT --> DECLOW
    DECLOW --> DECUP
    LAT --> DECUP
    DECUP --> OUT
```

* Trunk attention runs with grouped queries (`heads=9`, `groups=9`) and tightened shared RoPE dims (32 core) while module-specific RoPE stays at 128.
* Lower decoder keeps a windowed KV cache (`window=8192`, `nf4` quantisation).
* Router balances attention, retention, and SSM experts with temperature 0.7.
* Upper decoder relies on standard causal attention with RMS QK-norm.

### Variant_9 vs. "Attention Is All You Need"

- **Latent sampler:** adds a FiLM + LoRA latent path (H=16) that modulates every decoder block; the Vaswani baseline has no learned latent conditioning.
- **Grouped-query attention + local windows:** uses per-head KV (groups=heads) and windowed attention in the lower decoder to cut KV memory; the baseline keeps full MHA everywhere.
- **Mixture-of-architectures:** decoder_lower routes across Attention/Retention/SSM experts with top-k gating, extending contextual memory; baseline decoder is attention-only.
- **Structured KV policy:** windowed + NF4-quantised cache vs. baseline’s full-precision KV store.
- **Normalization upgrades:** RMSNorm throughout and explicit QK-norm in the upper decoder for stability, replacing the baseline’s LayerNorm + raw QK.
- **RoPE + YaRN scaling:** rotates embeddings with NTK/YaRN scaling to support 8k contexts, whereas the baseline used learned/sinusoidal positional encodings.

## Evolution Creative Sweep (seq_len=192, batch=6, 30 generations)

The creative search (population 16, top-k 5, 3 immigrants) promoted a deeper sliding-window stack and a hybrid attention/retention path as the current front-runners:

| Config | Score | Highlights |
| --- | --- | --- |
| `results/evolution_creative/gen_28/variant_7.yaml` | **0.0143** | 18-layer sliding attention trunk with hierarchical downsampling and a token-level depth router |
| `results/evolution_creative/gen_22/variant_8.yaml` | 0.0152 | Parallel attention/retention mixer with FiLM + LoRA conditioning and NF4 windowed KV caching |

### Architecture sketch (`gen_28/variant_7`)

```mermaid
flowchart TD
    INPUT["Token embeddings<br/>RMSNorm + RoPE theta=12000 dims=32"]
    TRUNK["18-layer trunk<br/>Sliding-window attention<br/>window=256 stride=64<br/>heads=11 groups=8"]
    FFN["Dense FFN<br/>SwiGLU mult=3.2<br/>RMSNorm"]
    HIER["Hierarchy scheduler<br/>Level1 every 3 -> downsample x0.5<br/>Level2 every 6 -> downsample x0.25 + up-proj"]
    ROUTER["Depth router (token)<br/>budget=0.6 tau=0.7<br/>min_layers=3"]
    OUTPUT["Readout"]
    INPUT --> TRUNK --> FFN --> HIER --> ROUTER --> OUTPUT
```

- Downsampling compresses intermediate states aggressively (0.5 then 0.25) and re-expands with `up_proj=true`, letting later layers focus compute on summarised context.
- Sliding attention keeps receptive fields dense within a 256-token window while striding 64 tokens to balance memory and coverage.
- The token-level depth router prunes residual layers when per-token confidence is high, holding average depth close to the 0.6 budget.

### Architecture sketch (`gen_22/variant_8`)

```mermaid
flowchart TD
    INPUT["Token embeddings<br/>RMSNorm + RoPE theta=25000 dims=32<br/>YaRN scaling factor=1.5"]
    COND["Conditioning path<br/>Pool-MLP H=16<br/>Freebits kappa=0.5"]
    OPS["Latent ops<br/>FiLM at pre_mixer<br/>LoRA r=4 on proj_q"]
    MIX["Parallel mixers per layer<br/>Attention: heads=10 groups=4 window=3456 (ALiBi)<br/>Retention: heads=8 chunk=1024<br/>merge=Add"]
    KV["KV policy<br/>Window cache=8192<br/>NF4 quantisation"]
    FFN["Dense FFN<br/>SwiGLU mult=3.04<br/>RMSNorm"]
    OUTPUT["Readout"]
    INPUT --> MIX --> FFN --> OUTPUT
    COND --> OPS --> MIX
    MIX --> KV
```

- The conditioner injects global statistics (Pool-MLP) while Freebits regularisation caps information collapse before FiLM/LoRA apply per-layer modulation.
- Attention and Retention run side-by-side each layer, merging additively to mix long-context memory (retention chunk 1024) with windowed ALiBi attention.
- Windowed KV caching (8192 tokens, NF4 quant) keeps memory bounded while still supporting 8k token contexts via YaRN-scaled RoPE.

## Next Steps

* Improve mixer fidelity further (GPU-friendly Retention/SSM kernels with state reuse).
* Scale mutation/evolution runs (more generations, larger populations, periodic long budgets) once GPU resources are available.

## Overnight Sweep Playbook

- **Seed population**: `configs/free_transformer.yaml`, `configs/free_transformer_alt.yaml`, `examples/nanogpt_tiny.yaml`, plus evolved seeds in `results/evolution/gen_7/variant_7.yaml` and `results/evolution/gen_8/variant_7.yaml`.
- **Stage B micro-distill**: start runs with `--seq-len 192 --batch-size 6`; this keeps budgets within the current CPU envelope.
- **Deep promotions**: run `runners/run_experiment.py … --deep-steps 400 --deep-seq-len 256 --deep-batch-size 6 --deep-top-k 2` to re-evaluate the current Pareto front every few generations without touching earlier ASHA/PDH checkpoints.
- **Variant_7 note**: promising steerability/coherence at fixed FLOPs, but latent usefulness and router stability remain open engineering risks—monitor FiLM/LoRA activations and gating entropy during long runs.
- **MPS safety**: `run_micro_train` now retries on CPU if MPS throws (see `allow_fallback=True`), so overnight sweeps can run on GPU without babysitting. Logged history flags any invalid configs that ASHA pruned.
