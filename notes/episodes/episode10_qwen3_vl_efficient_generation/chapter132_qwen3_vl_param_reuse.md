# Chapter 132 — Qwen3-VL 复用 kernel 参数盒

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-09
>
> 关闭：2026-10-09

## 结果

新的 scratch 原来会把 kernel 参数盒整份重建。盒子在第一次启动后
还在，之后只改里面的设备指针。计算没有变。

一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_param_reuse.json`，
SHA256 为
`1fb6be933ff633abbcd6c6126cf3924496534843d9871915d929efd07d97138b`。
`closed=false`，`acceptance_fixture=false`。position `107 / 203 / 331`
的最后一步 GPU 仍是 `0` bytes、`0` 次分配。CPU 仍是 `0`。
32-token 的第一次 decode CPU 从 `20,672` 降到 `9,168`。128/256 从
`14,912` 降到 `9,168`。256-token 第一次 decode 的 GPU 字节仍是
`2,018,180`，分配次数仍是 `2`。前四枚是 `[1987, 2169, 375, 265]`。
32-token 没有分叉。128/256 仍在 step 70 分叉：LifeAI zero-based
`21239`，HF `64692`，margin `0.02797699`。

第一次 decode 仍要创建 scratch。不声称整条 decode 零分配。profiler
采样的最大阶段仍是 mlp_gate_up。Episode 10 保持 Open。最终 logits
没有升成 Float32。
