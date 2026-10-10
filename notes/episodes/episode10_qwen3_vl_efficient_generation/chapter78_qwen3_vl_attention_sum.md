# Chapter 78 — Qwen3-VL decode attention softmax 分母

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-09
>
> 关闭：2026-10-09

## 结果

Chapter 77 之后，编译过的 attention 里最大的一块是 softmax 分母
`sum!`，`448` bytes。它把 `(groups, seq, kv_heads)` 收成 `(groups, 1, kv_heads)`。
顺序相加和 `sum!` 不是逐位一致。

GPU 在 `sum!` 仍是单 block 时改走同一条 warp 树：每个 `(group, head)` 一块，
线程数是 `nextwarp(seq)`，块内先按线程跨步累加，再做 warp shuffle 和 32 槽
shared memory。超出这条 kernel 的 launch limit 时，`sum!` 会拆成多 block，
两种顺序对不上，所以调用退回 `sum!`。CPU 也仍走 `sum!`。

这块 RTX 4090 D 上，该 `sum!` kernel 的 launch limit 是 `576` 线程。
sequence `1:331` 和 `576` 的 16 个分母与 `sum!` 都是 0 个不一致。`577` 和
`640` 退回 `sum!`。换过长度之后，这条 kernel 是 `144` bytes，`sum!` 是
`448` bytes。

一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_attention_sum.json`，
SHA256 为
`ea2340ed9ab2231f79a1e93490d72c2cd17b5411e51b8a2f2dda00c93aed0290`。
`closed=false`，`acceptance_fixture=false`。position `107 / 203 / 331`
的最后一步 GPU 仍是 `0` bytes、`0` 次分配。CPU 都是 `424,896`。
256-token 稳态一步从 Chapter 77 的 `441,024` 降到 `424,896`。第一次 decode
的 GPU 分配仍是 `2,018,180` bytes、`56` 次。前四枚是 `[1987, 2169, 375, 265]`。
32-token 没有分叉。128/256 仍在 step 70 分叉：LifeAI zero-based `21239`，
HF `64692`，margin `0.02797699`。

profiled 最后一步里，最大阶段仍是 attention。Episode 10 保持 Open。最终
logits 没有升成 Float32。
