# Chapter 138 — Qwen3-VL 复用父缓冲

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-10
>
> 关闭：2026-10-10

## 结果

后来的请求在已有 BF16 和 Float32 父缓冲装得下时，不再 `similar` 新的两块。
切片仍按这次容量改指针和形状。装不下才重新分配。warmup 已经按最长
容量分配，所以 32/128/256 的采样第一次 decode 不再向 GPU 要内存。
计算没有变。

一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_parent_reuse.json`，
SHA256 为
`1111c98b93bca2cd344bb4cc4468898b8d2a77fb9b5dc4ceebb21e48b80a3ccd`。
`closed=false`，`acceptance_fixture=false`。position `107 / 203 / 331`
的最后一步 GPU 仍是 `0` bytes、`0` 次分配。CPU 仍是 `0`。
32/128/256 的第一次 decode CPU 都从 `800` 降到 `352`。下降都是
`448` bytes。这三次的第一次 decode GPU 字节都从原来的
`1,043,332 / 1,461,124 / 2,018,180` 降到 `0`，分配次数都从 `2` 降到
`0`。前四枚是 `[1987, 2169, 375, 265]`。32-token 没有分叉。128/256
仍在 step 70 分叉：LifeAI zero-based `21239`，HF `64692`，margin
`0.02797699`。

第一次 decode 仍要分配 attention scratch。父缓冲的 GPU 内存在 warmup
里，不在采样的第一次 decode 里。不声称整条 decode 零分配。profiler
采样的最大阶段仍是 mlp_gate_up。Episode 10 保持 Open。最终 logits
没有升成 Float32。
