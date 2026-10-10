# Chapter 101 — Qwen3-VL decode QK RoPE 的直接启动

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-09
>
> 关闭：2026-10-09

## 结果

Chapter 100 之后，Q/K RoPE 的一次启动是 `208` bytes。

参数留在具体盒子里，直接调用 `cuLaunchKernel`。每层换上当前的 Q、K
和 cos/sin。三组 cos/sin、两组 Q/K 的输出与两条单张量 kernel 都是
0 个不一致。编译后的一次调用是 `0` bytes。28 层少 `208` bytes，合计
`5,824`。

一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_rope_raw.json`，
SHA256 为
`4eb335b744abf30c86693f3faf560c4d6d45b6a4908ac878d414d67bc37fb104`。
`closed=false`，`acceptance_fixture=false`。position `107 / 203 / 331`
的最后一步 GPU 仍是 `0` bytes、`0` 次分配。CPU 都是 `205,952`。
256-token 稳态一步从 Chapter 100 的 `211,776` 降到 `205,952`。第一次 decode
的 GPU 分配仍是 `2,018,180` bytes、`56` 次。前四枚是 `[1987, 2169, 375, 265]`。
32-token 没有分叉。128/256 仍在 step 70 分叉：LifeAI zero-based `21239`，
HF `64692`，margin `0.02797699`。

这次 profiled 最后一步里，最大阶段仍是 mlp_gate_up。Episode 10 保持 Open。
最终 logits 没有升成 Float32。
