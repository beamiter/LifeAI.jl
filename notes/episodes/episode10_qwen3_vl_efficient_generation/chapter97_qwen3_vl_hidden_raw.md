# Chapter 97 — Qwen3-VL decode hidden RMSNorm 的直接启动

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-09
>
> 关闭：2026-10-09

## 结果

Chapter 96 之后，长度 `2048` 的 hidden RMSNorm 一次启动是 `176` bytes。

参数留在具体盒子里，直接调用 `cuLaunchKernel`。每次调用换上当前的
输入和 scale。四组 scale、两组输入的 output 和 `mean_square` 与升精度、
`sum!(abs2)`、归一化都是 0 个不一致。编译后的一次调用是 `0` bytes。
pre、post 和 final 一共 57 次，每次少 `176` bytes，合计 `10,032`。

一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_hidden_raw.json`，
SHA256 为
`dc1a6db24f75ade00da6e847ca52996b11d3770537823b755cf2519003ab08db`。
`closed=false`，`acceptance_fixture=false`。position `107 / 203 / 331`
的最后一步 GPU 仍是 `0` bytes、`0` 次分配。CPU 都是 `240,256`。
256-token 稳态一步从 Chapter 96 的 `250,288` 降到 `240,256`。第一次 decode
的 GPU 分配仍是 `2,018,180` bytes、`56` 次。前四枚是 `[1987, 2169, 375, 265]`。
32-token 没有分叉。128/256 仍在 step 70 分叉：LifeAI zero-based `21239`，
HF `64692`，margin `0.02797699`。

这次 profiled 最后一步里，最大阶段仍是 mlp_gate_up。Episode 10 保持 Open。
最终 logits 没有升成 Float32。
