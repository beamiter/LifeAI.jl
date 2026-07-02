# LifeAI.jl

LifeAI.jl is organized as a Julia package skeleton for sequence models and text generation research.

## Layout

- `src/LifeAI.jl` entrypoint.
- `src/core/` core building blocks:
  - `attention.jl`
  - `transformer.jl`
  - `rope.jl`
  - `sampling.jl`
- `src/data/` data pipeline modules:
  - `tokenizer.jl`
  - `dataset.jl`
- `src/train/` training scripts:
  - `train_gpt.jl`
- `examples/minigpt.jl` quick smoke example.
- `notes/week01_transformer.md` initial milestone notes.

## Run example

```bash
julia examples/minigpt.jl
```
