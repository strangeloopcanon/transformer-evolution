# DSL Capability Map

This note turns the DSL in `src/evoforge/dsl/` into a single, GitHub-friendly visual so you can show at a glance what a config covers.

## DSL at a Glance

```mermaid
flowchart TB
  CFG["DSL Config (YAML specimen)"]
  CFG --> ARCH["arch section\n(core stack + modules/pipeline)"]
  CFG --> TRAIN["train section\n(ctx_len, optimizer, run budget)"]
  CFG --> BUDGET["budget\n(params_target, flops_target)"]

  subgraph ARCH_GRAPH["arch breakdown"]
    direction TB
    CORE["Core stack\n- d_model / n_layers\n- Norm (Layer/RMS/Scale)\n- Residual (single/dual/deepnet)"]
    MIX["Mix units\n- single | parallel | route\n- merge (Add / Weighted / Concat)\n- router (topk, temp, balance)"]
    ATT["Attention mixer\nheads / groups / head_dim\nstencils (full/local/dilated/...)\nsoftmax tweaks + KV policy (cache, window, ring, latent, quant)"]
    ALT["Alternate mixers\nRetention (chunk, parallel)\nSSM (d_state, expand, projection)\nLongConv (kernel_len, value_glu)"]
    FFN["FFN\nDense (mult x act)\nMoE (experts, topk, capacity)"]
    POS["Positional signals\nRoPE (theta, dims, scaling: NTK/linear/YaRN/LongRoPE)\nALiBi / RelBias / Learned / XPOS"]
    COND["Conditioning\nSources (pool-mlp, segment)\nOps (FiLM/add/scale/LoRA)\nReg (freebits, kappa)"]
    STRUCT["Structure\nHierarchy levels (downsample/up_proj)\nDepth router (token/layer, budget, tau, min_layers)\nResidual knobs"]
    MODULES["modules{}\ntransformer / embedding / latent_sampler / readout / custom\nper-module params/budgets"]
    PIPE["pipeline[]\nstage graph (repeat/mode)\ninputs, kv_from, mem_from\ntrain_only / prefill_only / budgets"]
  end

  CORE --> MIX
  MIX --> ATT
  MIX --> ALT
  MIX --> FFN
  CORE --> POS
  POS --> COND
  STRUCT --> PIPE
  MODULES --> PIPE

  TRAIN --> PIPE
  TRAIN --> COND
  TRAIN --> STRUCT

  BUDGET --> MODULES
  BUDGET --> PIPE
```

- Start from `arch`: choose the core stack, give it mixers (attention, retention, SSM, long conv), FFNs, positional encodings, conditioning, and structural knobs such as hierarchy or depth routing.
- Add optional `modules{}` for embeddings, latent samplers, readouts, or extra transformer stacks, then wire them via `pipeline[]` with explicit stage modes, repeats, and memory/KV routing.
- The `train` block anchors optimizer assumptions (ctx length, lr, wd, betas, clip, dtype) plus per-run budgets, while the top-level `budget` sets global params/flops targets that guide search and pruning.

## Rendering Notes

- GitHub, VS Code, Obsidian, and most Markdown previewers render Mermaid blocks inline.  
- To export a static asset, copy the diagram into a `.mmd` file and run `npx @mermaid-js/mermaid-cli -i <file>.mmd -o <file>.png`.
