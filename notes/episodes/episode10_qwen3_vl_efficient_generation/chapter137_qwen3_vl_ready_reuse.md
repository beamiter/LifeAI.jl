# Chapter 137 — Qwen3-VL 复用 Ready

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-10
>
> 关闭：2026-10-10

## 结果

第一次 decode 仍创建 Ready。之后的请求留着这一个对象，只换上新容量的
attention scratch。其余 scratch 还是原来的字段。计算没有变。

一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_ready_reuse.json`，
SHA256 为
`72409ccf67d9ceb0848d2a1de2fc82e7aa0a2c654ac81a66d336faa3a19e1365`。
`closed=false`，`acceptance_fixture=false`。position `107 / 203 / 331`
的最后一步 GPU 仍是 `0` bytes、`0` 次分配。CPU 仍是 `0`。
32/128/256 的第一次 decode CPU 都从 `1,312` 降到 `800`。下降都是
`512` bytes。256-token 第一次 decode 的 GPU 字节仍是 `2,018,180`，
分配次数仍是 `2`。前四枚是 `[1987, 2169, 375, 265]`。32-token 没有
分叉。128/256 仍在 step 70 分叉：LifeAI zero-based `21239`，HF
`64692`，margin `0.02797699`。

第一次 decode 仍要分配两块父缓冲和新的 attention scratch。不声称整条
decode 零分配。profiler 采样的最大阶段仍是 mlp_gate_up。Episode 10
保持 Open。最终 logits 没有升成 Float32。
