# Smart Resident Scheduler (LMSL)

Future-aware smart resident scheduler for ComfyUI low VRAM mode.

Reduces model swap thrashing on 12GB GPUs (e.g. RTX 3060) when running with
`--lowvram` / `--enable-dynamic-vram`.

## Features

- Reserves resident models in VRAM to avoid reload thrashing
- Future-aware scheduling based on queue preview
- Lightweight: no extra dependencies beyond ComfyUI itself
- TDS SDK selection bridge (v0.2.0): model health audit, intent-based
  recommendation nodes, and read-only HTTP endpoints under `/lmsl_sdk/*`
  (`status` / `recommend` / `advise`). All bridge paths fail soft and never
  block startup or workflow execution.

## Installation

Install from ComfyUI Registry (recommended):

```
comfy node install lmsl-smart-offload
```

Or install from source:

```
git clone https://github.com/mxh678988/lmsl-smart-offload.git
custom_nodes/lmsl-smart-offload
```

Then restart ComfyUI.

## Usage

The node appears in the node list after installation. Connect it into your
workflow to enable future-aware resident scheduling under low VRAM mode.

## License

MIT
