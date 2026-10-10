# Chapter 131 — Qwen3-VL decode scratch 共用两块存储

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-09
>
> 关闭：2026-10-09

## 结果

第一次 decode 原来为每组 scratch 各分配一块父存储，一共 13 块。
现在全部 BF16 切自一块，全部 Float32 切自另一块。CPU 路径仍是
原来的 `similar`。计算没有变。

一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_decode_workspace.json`，
SHA256 为
`2bc4e279c7355bd436203dd0982ce47b88b07c837d99911eecab08a0f49b8487`。
`closed=false`，`acceptance_fixture=false`。position `107 / 203 / 331`
的最后一步 GPU 仍是 `0` bytes、`0` 次分配。CPU 仍是 `0`。
32-token 的第一次 decode CPU 从 `23,632` 降到 `20,672`。128/256 从
`17,872` 降到 `14,912`。下降都是 `2,960` bytes。256-token 第一次
decode 的 GPU 字节仍是 `2,018,180`，分配次数从 `13` 降到 `2`。前四枚是
`[1987, 2169, 375, 265]`。32-token 没有分叉。128/256 仍在 step 70
分叉：LifeAI zero-based `21239`，HF `64692`，margin `0.02797699`。

第一次 decode 仍要创建 scratch。不声称整条 decode 零分配。profiler
采样的最大阶段仍是 mlp_gate_up。Episode 10 保持 Open。最终 logits
没有升成 Float32。
