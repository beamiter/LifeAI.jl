# Chapter 74 — Qwen3-VL decode attention maximum

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-09
>
> 关闭：2026-10-09

## 结果

Chapter 73 之后，一次 attention 调用是 `5,552` bytes。里面最大的一块是
`maximum!`，`2,272` bytes。它把 `(groups, seq, kv_heads)` 收成
`(groups, 1, kv_heads)`。

有限分数上 `max` 可结合。GPU 路径改成每个 query group 和 KV head 一个线程，
从 token `1` 顺序扫到 `seq`。CPU 仍走 `maximum!`。

RTX 4090 D 上，sequence `1 / 80 / 330 / 331` 的 16 个最大值与 `maximum!`
都是 0 个不一致，包括一组前 10 个 token 全是 `1.25` 的输入。一次 attention
调用降到 `3,888` bytes，sequence `80` 和 `331` 相同。

一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_attention_max.json`，
SHA256 为
`030d937ac838355235bed889beac2720c496a1707b6e0db364a06421e6c28d2a`。
`closed=false`，`acceptance_fixture=false`。position `107 / 203 / 331`
的最后一步 GPU 仍是 `0` bytes、`0` 次分配。CPU 是 `617,440`、`617,504`、
`617,440`。256-token 稳态一步从 Chapter 73 的 `677,920` 降到 `617,440`。
第一次 decode 的 GPU 分配仍是 `2,018,180` bytes、`56` 次。前四枚是
`[1987, 2169, 375, 265]`。32-token 没有分叉。128/256 仍在 step 70 分叉：
LifeAI zero-based `21239`，HF `64692`，margin `0.02797699`。

profiled 最后一步里，最大阶段仍是 attention。Episode 10 保持 Open。最终
logits 没有升成 Float32。
