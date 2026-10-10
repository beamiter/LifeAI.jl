# Chapter 103 — Qwen3-VL decode MLP SiLU 的直接启动

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-09
>
> 关闭：2026-10-09

## 结果

Chapter 102 之后，MLP 的 SiLU 乘积一次是 `160` bytes。

参数留在具体盒子里，直接调用 `cuLaunchKernel`。每次调用换上当前的
gate 和 up。两组输入与
`BFloat16(Float32(silu) * Float32(up))` 都是 0 个不一致，其中
`silu` 是 `BFloat16(Float32(gate) / (1 + exp(-Float32(gate))))`。
编译后的一次调用是 `0` bytes。28 层少 `160` bytes，合计 `4,480`。

一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_mlp_raw.json`，
SHA256 为
`7b293b12258a3be217777064f9d1b4ad2d3e534c03ff2fb05a952eb0c7d8d576`。
`closed=false`，`acceptance_fixture=false`。position `107 / 203 / 331`
的最后一步 GPU 仍是 `0` bytes、`0` 次分配。CPU 都是 `196,544`。
256-token 稳态一步从 Chapter 102 的 `201,024` 降到 `196,544`。第一次 decode
的 GPU 分配仍是 `2,018,180` bytes、`56` 次。前四枚是 `[1987, 2169, 375, 265]`。
32-token 没有分叉。128/256 仍在 step 70 分叉：LifeAI zero-based `21239`，
HF `64692`，margin `0.02797699`。

这次 profiled 最后一步里，最大阶段仍是 mlp_gate_up。Episode 10 保持 Open。
最终 logits 没有升成 Float32。
