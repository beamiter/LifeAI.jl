# Chapter 130 — Qwen3-VL 九块 BF16 scratch 共用一块存储

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-09
>
> 关闭：2026-10-09

## 结果

QKV、gate/up、mRoPE、token embedding 和 tied logits 这 9 块 BF16
数组原来各自 `similar`。现在第一次 decode 把它们切自同一块存储。
CPU 路径仍是原来的 `similar`。计算没有变。

一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_bf16_tails.json`，
SHA256 为
`31ee2e77b9a8bb327e803e73521791ba66eff6e5abb6e0b6770b89ca7b89c469`。
`closed=false`，`acceptance_fixture=false`。position `107 / 203 / 331`
的最后一步 GPU 仍是 `0` bytes、`0` 次分配。CPU 仍是 `0`。
32-token 的第一次 decode CPU 从 `24,784` 降到 `23,632`。128/256 从
`19,024` 降到 `17,872`。下降都是 `1,152` bytes。256-token 第一次
decode 的 GPU 字节仍是 `2,018,180`，分配次数从 `21` 降到 `13`。前四枚是
`[1987, 2169, 375, 265]`。32-token 没有分叉。128/256 仍在 step 70
分叉：LifeAI zero-based `21239`，HF `64692`，margin `0.02797699`。

第一次 decode 仍要创建 scratch。不声称整条 decode 零分配。profiler
采样的最大阶段仍是 mlp_gate_up。Episode 10 保持 Open。最终 logits
没有升成 Float32。
