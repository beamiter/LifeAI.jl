# Chapter 121 — Qwen3-VL attention scratch 共用两块存储

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-09
>
> 关闭：2026-10-09

## 结果

attention scratch 的 13 块数组不再各调一次 `similar`。BF16 和
Float32 各分配一块，再切成原来的形状。CPU 路径仍是 `similar`。
切出来的数组互不重叠，字节公式没有变。

一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_attention_pack.json`，
SHA256 为
`bab6018c177ba1083c6aa1a7e956180d6156b582e0bdd317fa53d65f1f800708`。
`closed=false`，`acceptance_fixture=false`。position `107 / 203 / 331`
的最后一步 GPU 仍是 `0` bytes、`0` 次分配。CPU 仍是 `0`。
32-token 的第一次 decode CPU 从 `41,664` 降到 `40,064`。128/256 从
`35,904` 降到 `34,304`。下降都是 `1,600` bytes。256-token 第一次
decode 的 GPU 字节仍是 `2,018,180`，分配次数从 `56` 降到 `45`。
前四枚是 `[1987, 2169, 375, 265]`。32-token 没有分叉。128/256 仍在
step 70 分叉：LifeAI zero-based `21239`，HF `64692`，margin `0.02797699`。

第一次 decode 仍要创建其余 scratch。不声称整条 decode 零分配。
profiler 采样的最大阶段仍是 mlp_gate_up。Episode 10 保持 Open。
最终 logits 没有升成 Float32。
