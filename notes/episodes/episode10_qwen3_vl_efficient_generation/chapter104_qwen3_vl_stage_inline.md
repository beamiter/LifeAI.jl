# Chapter 104 — Qwen3-VL decode stage 不再包闭包

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-09
>
> 关闭：2026-10-09

## 结果

Chapter 103 之后，每个 decoder stage 都先包成闭包再调用。没有
profiler 时这个闭包没有别的用途。

没有 runner 时，stage 的表达式直接执行。有 runner 时仍把同一个表达式
交给 profiler。一层的 11 个 stage 都会被 runner 调用到。计算和改前是
同一段表达式。

一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_stage_inline.json`，
SHA256 为
`942e68a633d89d896a587e7c7786f03da1a293ef0da916df20f0cf35e2e9e125`。
`closed=false`，`acceptance_fixture=false`。position `107 / 203 / 331`
的最后一步 GPU 仍是 `0` bytes、`0` 次分配。CPU 分别是 `129,616`、
`129,616` 和 `129,600`。256-token 稳态一步从 Chapter 103 的 `196,544`
降到 `129,600`。第一次 decode 的 GPU 分配仍是 `2,018,180` bytes、`56` 次。
前四枚是 `[1987, 2169, 375, 265]`。32-token 没有分叉。128/256 仍在
step 70 分叉：LifeAI zero-based `21239`，HF `64692`，margin `0.02797699`。

这次 profiled 最后一步里，最大阶段仍是 mlp_gate_up。Episode 10 保持 Open。
最终 logits 没有升成 Float32。
