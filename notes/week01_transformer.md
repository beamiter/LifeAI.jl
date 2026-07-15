# Week 01 — Project Skeleton and Transformer Foundations

> 周期：2026-07-01 — 2026-07-07
>
> 历史记录：本文保留 Week 01 当时的目标与 TODO。后续多项 TODO 已经完成，最新状态请查看 [`current_status.md`](current_status.md)。

## Completed Layout

- `src/LifeAI.jl` entrypoint module now wires core, data, and training modules.
- Core modules moved under `src/core/`.
- Data and training helpers are organized under `src/data` and `src/train`.

## Completed Components

- Implemented manual scaled dot-product attention for learning and correctness checks.
- Implemented batched scaled dot-product attention using `NNlib.batched_mul` and `softmax`.
- Implemented `MultiHeadAttention` with Q/K/V projections, head reshape, causal masking, head merge, and output projection.
- Implemented `RoPE` with precomputed `cos_cache` / `sin_cache`.
- Integrated RoPE into `MultiHeadAttention`, applying rotation only to Q/K and leaving V unchanged.
- Added RoPE tests covering shape, finite values, position-0 identity, pairwise norm preservation, `start_pos`, invalid odd `head_dim`, and mismatched dimensions.
- Implemented a minimal GPT-style `TransformerBlock` in `src/core/transformer.jl`.

## TransformerBlock Design

Current block structure is pre-norm GPT style:

```text
x
│
├── x + MultiHeadAttention(LayerNorm(x))
│
└── x + MLP(LayerNorm(x))
```

Tensor convention:

```text
x: (d_model, seq_len, batch)
y: (d_model, seq_len, batch)
```

The MLP is intentionally simple for now:

```text
Dense(d_model => 4d_model)
GELU
Dense(4d_model => d_model)
```

RoPE remains an attention-level option and is enabled through `TransformerBlock(...; use_rope=true)`.

## Suggested Immediate Tests

Add a `test/test_transformer.jl` with these basic checks:

- `TransformerBlock` forward keeps shape unchanged.
- Output values are finite.
- Returned state contains `norm1`, `attn`, `norm2`, and `mlp`.
- A RoPE-enabled block works when `max_seq_len >= seq_len`.
- A RoPE-enabled block throws when `seq_len > max_seq_len`.

Then include it from `test/runtests.jl`:

```julia
@testset "TransformerBlock" begin
    include("test_transformer.jl")
end
```

## TODO

- Add `test/test_transformer.jl` and run the full test suite.
- Implement a minimal GPT model that stacks multiple `TransformerBlock`s.
- Implement token embedding and LM head.
- Implement `Tokenizer` in `src/data/tokenizer.jl`.
- Implement `DatasetLoader` in `src/data/dataset.jl`.
- Implement GPT training utilities in `src/train/train_gpt.jl`.
- Replace `examples/minigpt.jl` with an executable training/inference demo.

## Next Milestone

The next milestone is no longer “understand attention”; it is now:

> Build the smallest complete GPT that can overfit a tiny text file.

A good target is a character-level GPT over a small corpus. Once it can overfit, the model stack, loss path, gradients, and sampling loop are all proven end-to-end.
