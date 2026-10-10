# Chapter 99 — Qwen3-VL decode QKV 的 reshape

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-09
>
> 关闭：2026-10-09

## 结果

Chapter 98 之后，QKV 投影每次把 query、key、value 收成
`(head_dim, heads, 1, 1)`。三次 reshape 合计 `288` bytes。

共享 scratch 上这三份矩阵只做一次。两组输入的结果和每次现做的
reshape 都是 0 个不一致，形状是 `(128, 16, 1, 1)` 与 `(128, 8, 1, 1)`。
编译后的一次调用是 `0` bytes。28 层少 `288` bytes，合计 `8,064`。

一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_qkv_matrix.json`，
SHA256 为
`e1f0049928c88d1aa76b452ac8ca9400266bbd526717d9218eb572384036f1bf`。
`closed=false`，`acceptance_fixture=false`。position `107 / 203 / 331`
的最后一步 GPU 仍是 `0` bytes、`0` 次分配。CPU 都是 `223,232`。
256-token 稳态一步从 Chapter 98 的 `231,296` 降到 `223,232`。第一次 decode
的 GPU 分配仍是 `2,018,180` bytes、`56` 次。前四枚是 `[1987, 2169, 375, 265]`。
32-token 没有分叉。128/256 仍在 step 70 分叉：LifeAI zero-based `21239`，
HF `64692`，margin `0.02797699`。

这次 profiled 最后一步里，最大阶段仍是 mlp_gate_up。Episode 10 保持 Open。
最终 logits 没有升成 Float32。
