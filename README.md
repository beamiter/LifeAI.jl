# LifeAI.jl

LifeAI.jl is a small Julia/Lux research project for understanding and building
decoder-only language models end to end.

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
- `src/models/gpt.jl` decoder-only GPT model.
- `src/train/train_gpt.jl` next-token loss and training loop.
- `src/generation/text_generation.jl` greedy, temperature, and top-k generation.
- `examples/minigpt.jl` tiny character-level training and generation demo.
- `notes/week01_transformer.md` initial milestone notes.

## Install dependencies

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Run tests

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

## Train and generate

```bash
julia --project=. examples/minigpt.jl
```
