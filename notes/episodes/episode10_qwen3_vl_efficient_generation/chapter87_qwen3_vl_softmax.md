# Chapter 87 — Qwen3-VL decode attention softmax 的同一次启动

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-09
>
> 关闭：2026-10-09

## 结果

Chapter 86 之后，softmax 仍是 scale、顺序 `maximum!`、exp、warp `sum!`、
权重置换五次启动。每个 head 的顺序不变。

一次启动按这个顺序做完。`sum!` 仍只在单 block 能覆盖时替换；更长的序列
退回五次 kernel。sequence `1 / 2 / 31 / 32 / 33 / 80 / 330 / 331` 的
f32 分数、scaled、分母、exp、整段权重和整段置换都是 0 个不一致。CPU
仍走原来的广播和 `permutedims!`。

一层少 `768` bytes，28 层是 `21,504`。

一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_softmax.json`，
SHA256 为
`9d41f7461c0387d88d2333f6c9184108380140695af958ca2c6940af7673c331`。
`closed=false`，`acceptance_fixture=false`。position `107 / 203 / 331`
的最后一步 GPU 仍是 `0` bytes、`0` 次分配。CPU 都是 `344,144`。
256-token 稳态一步从 Chapter 86 的 `365,648` 降到 `344,144`。第一次 decode
的 GPU 分配仍是 `2,018,180` bytes、`56` 次。前四枚是 `[1987, 2169, 375, 265]`。
32-token 没有分叉。128/256 仍在 step 70 分叉：LifeAI zero-based `21239`，
HF `64692`，margin `0.02797699`。

这次 profiled 最后一步里，最大阶段是 `mlp_gate_up_projection`。Episode 10
保持 Open。最终 logits 没有升成 Float32。
