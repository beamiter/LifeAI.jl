# Chapter 95 — Qwen3-VL decode attention softmax 的直接启动

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-09
>
> 关闭：2026-10-09

## 结果

Chapter 94 之后，attention softmax 的一次启动是 `224` bytes。那是启动
配置，以及每个参数查询 stream 是否正在 capture。

参数现在留在具体盒子里，直接调用 `cuLaunchKernel`。kernel 读完整的
scratch，仍只使用 `seq` 以内的元素。sequence
`1 / 2 / 31 / 32 / 33 / 80 / 330 / 331` 的 output、整段 weights、
`weight_perm`、`maxima`，以及 scaled、exponents、`scores_f32` 的前缀，
与五次 kernel 都是 0 个不一致。超过 `sum!` 单 block 上限时仍走原来的
五次 kernel。attention 从 `576` bytes 降到 `352`。28 层少 `224` bytes，
合计 `6,272`。

一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_softmax_raw.json`，
SHA256 为
`c1f8e35ac1c364dd29d19c2814e1198406e3133fee7cec71d215bdea99463d9b`。
`closed=false`，`acceptance_fixture=false`。position `107 / 203 / 331`
的最后一步 GPU 仍是 `0` bytes、`0` 次分配。CPU 都是 `260,144`。
256-token 稳态一步从 Chapter 94 的 `266,416` 降到 `260,144`。第一次 decode
的 GPU 分配仍是 `2,018,180` bytes、`56` 次。前四枚是 `[1987, 2169, 375, 265]`。
32-token 没有分叉。128/256 仍在 step 70 分叉：LifeAI zero-based `21239`，
HF `64692`，margin `0.02797699`。

这次 profiled 最后一步里，最大阶段仍是 mlp_gate_up。Episode 10 保持 Open。
最终 logits 没有升成 Float32。
