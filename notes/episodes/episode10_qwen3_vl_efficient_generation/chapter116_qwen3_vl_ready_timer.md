# Chapter 116 — Qwen3-VL 稳态计时不再包一层 @sync

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-09
>
> 关闭：2026-10-09

## 结果

`CUDA.@timed` 在窗口里放 `@sync`。空调用是 `32` bytes。ready 步本身是
`0` bytes，包上这次真实调用后，计时器是 `4,144` bytes。

稳态步改成直接记 `gc_num` 和 `alloc_stats`。同步仍在窗口外面。计算没有变。
第一步仍用原来的计时，用来装好具体包装。

一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_ready_timer.json`，
SHA256 为
`36246c5cc9e2d2e47fc1392db105143c6375c87f46109ac803cb65f41699576a`。
`closed=false`，`acceptance_fixture=false`。position `107 / 203 / 331`
的最后一步 GPU 仍是 `0` bytes、`0` 次分配。CPU 都是 `0`。
256-token 稳态一步从 Chapter 115 的 `4,144` 降到 `0`。整步下降是
`4,144` bytes。第一次 decode 的 GPU 分配仍是 `2,018,180` bytes、`56` 次。
前四枚是 `[1987, 2169, 375, 265]`。32-token 没有分叉。128/256 仍在
step 70 分叉：LifeAI zero-based `21239`，HF `64692`，margin `0.02797699`。

这 `4,144` bytes 是计时器的 `@sync`，不是 ready 步里的分配。第一次
decode 仍分配。不声称整条 decode 零分配。profiler 采样的最大阶段仍是
mlp_gate_up。Episode 10 保持 Open。最终 logits 没有升成 Float32。
