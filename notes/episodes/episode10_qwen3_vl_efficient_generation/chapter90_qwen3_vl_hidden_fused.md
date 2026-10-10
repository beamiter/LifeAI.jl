# Chapter 90 — Qwen3-VL decode hidden RMSNorm 的两次 kernel

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-09
>
> 关闭：2026-10-09

## 结果

Chapter 89 之后，长度 `2048` 的 hidden RMSNorm 仍是升精度、4×576 的
partial abs2、32 线程尾部和归一化 kernel。编译后的一次调用是 `704` bytes。

这条路径只在现有 abs2 计划是 4 个 block、每块 576 线程时启用。第一次
kernel 按同一套下标从 BF16 做 abs2。第二次用 32 线程、64 个 block 重算
这 4 个数，再按原来的公式写 output 和 `mean_square`。计划不同，或 hidden
不是 `2048`，仍走原来的三次 kernel。CPU 仍走原来的广播。一维 scale 的
reshape 留在具体数组里。

四组随机输入的 output 和 `mean_square` 都是 0 个不一致。编译后的一次调用
是 `336` bytes。第 60 个 scale 命中后仍是 `336`。

一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_hidden_fused.json`，
SHA256 为
`5cea2add288724b3126c025c3ca3f0ca05352ea5d6f387288516a292d77f6ccf`。
`closed=false`，`acceptance_fixture=false`。position `107 / 203 / 331`
的最后一步 GPU 仍是 `0` bytes、`0` 次分配。CPU 都是 `297,952`。
256-token 稳态一步从 Chapter 89 的 `312,208` 降到 `297,952`。第一次 decode
的 GPU 分配仍是 `2,018,180` bytes、`56` 次。前四枚是 `[1987, 2169, 375, 265]`。
32-token 没有分叉。128/256 仍在 step 70 分叉：LifeAI zero-based `21239`，
HF `64692`，margin `0.02797699`。

这次 profiled 最后一步里，最大阶段是 mlp_gate_up。Episode 10 保持 Open。
最终 logits 没有升成 Float32。
