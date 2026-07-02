# Week 01 — Project Skeleton and Transformer Foundations

## Completed Layout

- `src/LifeAI.jl` entrypoint module now wires core, data, and training modules.
- Core modules moved under `src/core/`.
- Data and training helpers are organized under `src/data` and `src/train`.

## TODO

- Implement `MultiHeadAttention` in `src/core/attention.jl`.
- Implement `TransformerBlock` in `src/core/transformer.jl`.
- Implement `RoPE` in `src/core/rope.jl`.
- Implement datasets/tokenization in `src/data/tokenizer.jl` and `src/data/dataset.jl`.
- Implement GPT trainer in `src/train/train_gpt.jl`.
- Replace `examples/minigpt.jl` with an executable training/inference demo.
