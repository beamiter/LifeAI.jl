# Chapter 119 — Qwen3-VL 第一次 decode 不再重拷 blocks

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-09
>
> 关闭：2026-10-09

## 结果

第一次 BF16 decode 先把 blocks 放进 pack，再从无类型参数里把同一份
tuple 拷进 ready。现在 ready 直接沿用 pack 里的那份。计算没有变。

一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_ready_install.json`，
SHA256 为
`ec735216c159c84be0712a9bbe46c35df4600d4bf40e5a5eb7c35667f0d1bbc2`。
`closed=false`，`acceptance_fixture=false`。position `107 / 203 / 331`
的最后一步 GPU 仍是 `0` bytes、`0` 次分配。CPU 仍是 `0`。
32-token 的第一次 decode CPU 从 `46,128` 降到 `43,424`。128/256 从
`40,368` 降到 `37,664`。下降都是 `2,704` bytes。第一次 decode 的 GPU
分配仍是 `2,018,180` bytes、`56` 次。前四枚是
`[1987, 2169, 375, 265]`。32-token 没有分叉。128/256 仍在 step 70
分叉：LifeAI zero-based `21239`，HF `64692`，margin `0.02797699`。

第一次 decode 仍要创建 scratch。不声称整条 decode 零分配。profiler
采样的最大阶段仍是 mlp_gate_up。Episode 10 保持 Open。最终 logits
没有升成 Float32。
