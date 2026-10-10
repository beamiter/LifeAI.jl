# Chapter 128 — Qwen3-VL 第一次 decode 不再包一层 @sync

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-09
>
> 关闭：2026-10-09

## 结果

第一次 decode 仍用 `CUDA.@timed`。它在窗口里放 `@sync`，这次调用上面
多了 `4,144` bytes。计时改成和稳态步一样，直接记 `gc_num` 和
`alloc_stats`。同步仍在窗口外面。计算没有变。

一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_first_timer.json`，
SHA256 为
`ee02b9b657eaef91346f255c6c2401ff1ba2a4e92f5756d6f8ef4952e7ce4016`。
`closed=false`，`acceptance_fixture=false`。position `107 / 203 / 331`
的最后一步 GPU 仍是 `0` bytes、`0` 次分配。CPU 仍是 `0`。
32-token 的第一次 decode CPU 从 `31,616` 降到 `27,472`。128/256 从
`25,856` 降到 `21,712`。下降都是 `4,144` bytes。256-token 第一次
decode 的 GPU 字节仍是 `2,018,180`，分配次数仍是 `21`。前四枚是
`[1987, 2169, 375, 265]`。32-token 没有分叉。128/256 仍在 step 70
分叉：LifeAI zero-based `21239`，HF `64692`，margin `0.02797699`。

这 `4,144` bytes 是计时器的 `@sync`。第一次 decode 本身仍分配。
不声称整条 decode 零分配。profiler 采样的最大阶段仍是 mlp_gate_up。
Episode 10 保持 Open。最终 logits 没有升成 Float32。
