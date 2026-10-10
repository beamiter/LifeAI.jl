# Chapter 94 — Qwen3-VL decode hidden RMSNorm 的一次启动

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-09
>
> 关闭：2026-10-09

## 结果

Chapter 93 之后，长度 `2048` 的 hidden RMSNorm 仍是两次 kernel。编译后的
一次调用是 `336` bytes。

现在每个 block 按原来的 4×576 下标算出 4 个 partial，再用 576 线程的
warp 树把它们加起来，然后写 output 和 `mean_square`。只在现有 abs2 计划
仍是 4 个 block、每块 576 线程时启用。计划不同，或 hidden 不是 `2048`，
仍走原来的三次 kernel。CPU 仍走原来的广播。

六组随机输入的 output 和 `mean_square` 与升精度、`sum!(abs2)`、归一化
都是 0 个不一致。编译后的一次调用是 `176` bytes。第 60 个 scale 命中后
仍是 `176`。pre、post 和 final 一共 57 次，每次少 `160` bytes，合计
`9,120`。

一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_hidden_single.json`，
SHA256 为
`9e122b995b02a9c2d85b6b1ae67491aa8ea541adb581160493d00d30f398bd10`。
`closed=false`，`acceptance_fixture=false`。position `107 / 203 / 331`
的最后一步 GPU 仍是 `0` bytes、`0` 次分配。CPU 都是 `266,416`。
256-token 稳态一步从 Chapter 93 的 `275,536` 降到 `266,416`。第一次 decode
的 GPU 分配仍是 `2,018,180` bytes、`56` 次。前四枚是 `[1987, 2169, 375, 265]`。
32-token 没有分叉。128/256 仍在 step 70 分叉：LifeAI zero-based `21239`，
HF `64692`，margin `0.02797699`。

这次 profiled 最后一步里，最大阶段仍是 mlp_gate_up。Episode 10 保持 Open。
最终 logits 没有升成 Float32。
