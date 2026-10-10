# Chapter 115 — Qwen3-VL 稳态循环只进入一次具体包装

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-09
>
> 关闭：2026-10-09

## 结果

具体包装上的 ready 步本身是 `0` bytes。每步还要先从 cache 的无类型槽里
取出这个包装，这次分派是 `32` bytes。

长生成的第一步仍走原来的入口，并装好包装。后面的步在一个已经知道
包装类型的循环里计时。分派只发生一次，不在稳态计时里。计算没有变。

一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_ready_loop.json`，
SHA256 为
`4caaf5d2593f99549eb761f7285bd1d9e4911bba253ed76e33739a2c6145bd3c`。
`closed=false`，`acceptance_fixture=false`。position `107 / 203 / 331`
的最后一步 GPU 仍是 `0` bytes、`0` 次分配。CPU 都是 `4,144`。
256-token 稳态一步从 Chapter 114 的 `4,176` 降到 `4,144`。整步下降是
`32` bytes。第一次 decode 的 GPU 分配仍是 `2,018,180` bytes、`56` 次。
前四枚是 `[1987, 2169, 375, 265]`。32-token 没有分叉。128/256 仍在
step 70 分叉：LifeAI zero-based `21239`，HF `64692`，margin `0.02797699`。

这 `4,144` bytes 是计时器，不是 ready 步里的分配。不声称整条 decode
零分配。profiler 采样的最大阶段仍是 mlp_gate_up。Episode 10 保持 Open。
最终 logits 没有升成 Float32。
