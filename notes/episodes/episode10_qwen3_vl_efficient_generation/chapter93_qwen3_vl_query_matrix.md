# Chapter 93 — Qwen3-VL decode attention query 的 reshape

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-09
>
> 关闭：2026-10-09

## 结果

Chapter 92 之后，attention 准备 kernel 每次把 5 维 query 收成
`(groups, head_dim, kv_heads)`。这次 reshape 是 `96` bytes。共享的
decode scratch 只做一次，后面的层直接用那份数组。

sequence `1 / 80 / 331` 的 query、整段 K 和整段 V 与原来的载入加拷贝
都是 0 个不一致。缓存的矩阵和每次现做的 reshape 也是 0 个不一致。
attention 从 `672` bytes 降到 `576`。28 层少 `96` bytes，合计 `2,688`。

一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_query_matrix.json`，
SHA256 为
`5337c63eef2bf9bfaacdd1929ee173744b7acc6cfac967f87310299341895fca`。
`closed=false`，`acceptance_fixture=false`。position `107 / 203 / 331`
的最后一步 GPU 仍是 `0` bytes、`0` 次分配。CPU 都是 `275,536`。
256-token 稳态一步从 Chapter 92 的 `278,224` 降到 `275,536`。第一次 decode
的 GPU 分配仍是 `2,018,180` bytes、`56` 次。前四枚是 `[1987, 2169, 375, 265]`。
32-token 没有分叉。128/256 仍在 step 70 分叉：LifeAI zero-based `21239`，
HF `64692`，margin `0.02797699`。

这次 profiled 最后一步里，最大阶段仍是 mlp_gate_up。Episode 10 保持 Open。
最终 logits 没有升成 Float32。
