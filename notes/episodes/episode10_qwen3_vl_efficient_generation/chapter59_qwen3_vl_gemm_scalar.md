# Chapter 59 — Qwen3-VL decode gemmEx 的持久 alpha/beta

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-08
>
> 关闭：2026-10-08

## 结果

Chapter 58 留下的 per-head `mul!` 和各线性层的 `gemmEx!` 每次都给 alpha/beta
分配 `8` bytes。RTX 4090 D 上，同一条 `cublasGemmEx`（`CUBLAS_COMPUTE_32F`、
`CUBLAS_GEMM_DEFAULT`）改用进程内持久的 device `Float32` 标量 `1` 和 `0` 之后，
单 token 的 Q/K/V、gate/up、down、attention output、tied embedding 转置，以及
per-head `(2, 128) × (128, seq)`（`seq` 为 `1`、`80`、`330`、`331`，含 prefix
view）都与 `mul!` 逐位一致。标量存在之后，复用调用是 `0` GPU bytes。

这不是 Chapter 58 的 strided batched GEMM。attention 仍按原来的 head 顺序逐个
调用，只是不再为每次调用分配 alpha/beta。CPU 的 chunked Float32 和 per-head
循环顺序没有改。

一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_gemm_scalar.json`，SHA256 为
`c0bdb3d4fa6427bce41d8f42a95f075ec6c53331c10eed1d00a2aa5014a6298c`。
`closed=false`，`acceptance_fixture=false`。position `107 / 203 / 331` 的最后
一步都是 `1,168` bytes、`59` 次分配，归因覆盖 exact。相对 Chapter 57 的
`6,328 / 1,349`，正好少了 `5,160` bytes 和 `1,290` 次分配：

| 阶段 | Chapter 57 | Chapter 59 |
| --- | --- | --- |
| `attention` | `3,584 / 896` | `0 / 0` |
| `qkv_projection` | `672 / 168` | `0 / 0` |
| `mlp_gate_up_projection` | `448 / 112` | `0 / 0` |
| `attention_output_projection_residual` | `224 / 56` | `0 / 0` |
| `mlp_down_projection_residual` | `224 / 56` | `0 / 0` |
| `vocab_logits` | `8 / 2` | `0 / 0` |

前四枚是 `[1987, 2169, 375, 265]`。32-token 没有分叉。128/256 仍在 step 70
分叉：LifeAI zero-based `21239`，HF `64692`，margin `0.02797699`。profiled
logits 与 tokens 都和 baseline 一致，storage identity 通过。没有新的线性头。

最大并列阶段是 `pre_attention_norm` 和 `post_attention_norm`，各 `448` bytes。
`mrope_prepare` 是 `256` bytes，`final_norm` 是 `16` bytes。Episode 10 保持
Open。最终 logits 没有升成 Float32。
