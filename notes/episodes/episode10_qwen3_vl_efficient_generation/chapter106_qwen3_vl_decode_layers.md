# Chapter 106 — Qwen3-VL decode 28 层一次调用

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-09
>
> 关闭：2026-10-09

## 结果

Chapter 105 的热路径本身是 `0` bytes。它仍被大函数逐层调用，
每次调用留下分配。28 层改由一个具体类型的函数循环调用，
大函数只进入一次。层内计算没有变。有 profiler 时仍逐层走原来的
stage 函数。

这 28 层放在具体函数里是 `0` bytes。整步下降是 `5,312` bytes。
一次调用仍然留下分配，所以不按逐层调用的字节数乘 28 记。

一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_decode_layers.json`，
SHA256 为
`8e06c526e0bd0c6a12572979034dccd8c0a38475d10a525411b5c1dab547bb6b`。
`closed=false`，`acceptance_fixture=false`。position `107 / 203 / 331`
的最后一步 GPU 仍是 `0` bytes、`0` 次分配。CPU 分别是 `16,784`、
`16,784` 和 `16,768`。256-token 稳态一步从 Chapter 105 的 `22,080`
降到 `16,768`。第一次 decode 的 GPU 分配仍是 `2,018,180` bytes、`56` 次。
前四枚是 `[1987, 2169, 375, 265]`。32-token 没有分叉。128/256 仍在
step 70 分叉：LifeAI zero-based `21239`，HF `64692`，margin `0.02797699`。

profiler 采样仍走原来的 stage 函数，最大阶段仍是 mlp_gate_up。
Episode 10 保持 Open。最终 logits 没有升成 Float32。
