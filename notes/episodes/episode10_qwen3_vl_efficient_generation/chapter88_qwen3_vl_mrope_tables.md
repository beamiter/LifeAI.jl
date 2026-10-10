# Chapter 88 — Qwen3-VL decode mRoPE 频率表

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-09
>
> 关闭：2026-10-09

## 结果

Chapter 87 之后，每个 decode 步都重新算 inv frequency、分配频率数组，再
广播 `cos`/`sin`。这一次是 `1,536` bytes。三行 position 都是同一个
coordinate，所以每一 lane 是 `inv_frequency * coordinate`。

inv frequency 和 host 上的频率、cos、sin 缓冲只建一次。之后写入同一块
缓冲，再 `copyto!` 到 device。coordinate `0 / 1 / 76 / 100 / 331 / 10000`
的 cos 和 sin 与原来的频率函数都是 0 个不一致。Prefill 仍走原来的函数。
复用之后是 `64` bytes。

一步少 `1,472` bytes。

一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_mrope.json`，
SHA256 为
`9d3a146875250b9dcfa4ded7b7c617620903cd20f21279d71973660b0a9965f5`。
`closed=false`，`acceptance_fixture=false`。position `107 / 203 / 331`
的最后一步 GPU 仍是 `0` bytes、`0` 次分配。CPU 都是 `342,672`。
256-token 稳态一步从 Chapter 87 的 `344,144` 降到 `342,672`。第一次 decode
的 GPU 分配仍是 `2,018,180` bytes、`56` 次。前四枚是 `[1987, 2169, 375, 265]`。
32-token 没有分叉。128/256 仍在 step 70 分叉：LifeAI zero-based `21239`，
HF `64692`，margin `0.02797699`。

这次 profiled 最后一步里，最大阶段是 `mlp_gate_up_projection`。Episode 10
保持 Open。最终 logits 没有升成 Float32。
