# Chapter 89 — Qwen3-VL decode 单 token GEMM

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-09
>
> 关闭：2026-10-09

## 结果

Chapter 88 之后，单 token 线性层把 `(out, 1, 1)` 和 `(in, 1, 1)`
`reshape` 成矩阵再交给 `cublasGemmEx`。一次调用是 `176` bytes。

batch 维是 1 时，这两维的 stride 已经是矩阵的 leading dimension。这条路径
直接把原数组交给同一次 `cublasGemmEx`。`2048×2048`、`1024×2048`、
`6144×2048` 和 `128×64` 的输出与 reshape 路径都是 0 个不一致。多 token
仍走 reshape。CPU 仍走原来的分块乘法。单 token 调用降到 `16` bytes。

一层少 `1,088` bytes，28 层是 `30,464`。

一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_gemm_token.json`，
SHA256 为
`4c20846e26ec45e49530b9e5568a7f0f75f15741a84154ac5834030cd1d896a6`。
`closed=false`，`acceptance_fixture=false`。position `107 / 203 / 331`
的最后一步 GPU 仍是 `0` bytes、`0` 次分配。CPU 都是 `312,208`。
256-token 稳态一步从 Chapter 88 的 `342,672` 降到 `312,208`。第一次 decode
的 GPU 分配仍是 `2,018,180` bytes、`56` 次。前四枚是 `[1987, 2169, 375, 265]`。
32-token 没有分叉。128/256 仍在 step 70 分叉：LifeAI zero-based `21239`，
HF `64692`，margin `0.02797699`。

这次 profiled 最后一步里，最大阶段是 attention。Episode 10 保持 Open。
最终 logits 没有升成 Float32。
