# Chapter 98 — Qwen3-VL decode residual 加法的直接启动

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-09
>
> 关闭：2026-10-09

## 结果

Chapter 97 之后，hidden residual 的加法 kernel 一次是 `160` bytes。GEMM
本身已经是 `0`。

参数留在具体盒子里，直接调用 `cuLaunchKernel`。每次调用换上当前的
residual。两组输入的输出与 `BFloat16(Float32(residual) + Float32(projected))`
都是 0 个不一致。编译后的一次调用是 `0` bytes。attention 出口和 MLP
出口各一次，28 层共 56 次，每次少 `160` bytes，合计 `8,960`。

一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_residual_raw.json`，
SHA256 为
`3db90548a5f9e730f543a5db2dd4885817d8b325d2b198407a090d404c3a9d2b`。
`closed=false`，`acceptance_fixture=false`。position `107 / 203 / 331`
的最后一步 GPU 仍是 `0` bytes、`0` 次分配。CPU 都是 `231,296`。
256-token 稳态一步从 Chapter 97 的 `240,256` 降到 `231,296`。第一次 decode
的 GPU 分配仍是 `2,018,180` bytes、`56` 次。前四枚是 `[1987, 2169, 375, 265]`。
32-token 没有分叉。128/256 仍在 step 70 分叉：LifeAI zero-based `21239`，
HF `64692`，margin `0.02797699`。

这次 profiled 最后一步里，最大阶段仍是 mlp_gate_up。Episode 10 保持 Open。
最终 logits 没有升成 Float32。
