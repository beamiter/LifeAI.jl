# Chapter 114 — Qwen3-VL decode tied logits 复用 reshape

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-09
>
> 关闭：2026-10-09

## 结果

tied logits 的 GEMM 把 `(vocab, 1, 1)` 的输出和 `(hidden, 1, 1)` 的
hidden 收成两张矩阵。这两块存储每次 decode 都不变，原来每次现做
reshape。

现在按对象留住这两张矩阵，再走原来的 `transA='T'` GEMM。CPU 数组仍走
原来的投影。换一组 hidden 之后，结果和每次现做 reshape 的 GEMM 逐位一致。
整步下降是 `160` bytes。

一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_tied_matrix.json`，
SHA256 为
`791f7e29a744f4402f061e12dc81ab1c6cc01ee386fa227ff7a04c8415387cb3`。
`closed=false`，`acceptance_fixture=false`。position `107 / 203 / 331`
的最后一步 GPU 仍是 `0` bytes、`0` 次分配。CPU 分别是 `4,192`、
`4,192` 和 `4,176`。256-token 稳态一步从 Chapter 113 的 `4,336`
降到 `4,176`。第一次 decode 的 GPU 分配仍是 `2,018,180` bytes、`56` 次。
前四枚是 `[1987, 2169, 375, 265]`。32-token 没有分叉。128/256 仍在
step 70 分叉：LifeAI zero-based `21239`，HF `64692`，margin `0.02797699`。

profiler 采样的最大阶段仍是 mlp_gate_up。Episode 10 保持 Open。
最终 logits 没有升成 Float32。
