# Chapter 61 — Qwen3-VL decode mRoPE cos/sin 表

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-08
>
> 关闭：2026-10-08

## 结果

Chapter 60 之后，profile 里只剩 `mrope_prepare` 的 `256` bytes、`2` 次分配。
单 token 的 host 频率仍按 `_qwen3_vl_text_mrope` 计算。搬上 GPU 的是两张
BF16 cos/sin，形状 `(64, 1, 1)`，各 `128` bytes。把它们写进按 `head_dim`
预分配的 scratch 之后，与原来的 `to_device` 结果逐位一致。表已经存在时，
`copyto!` 是 `0` GPU bytes。28 层共用这一对表。prefill 和 Float32 仍走原来的
`_qwen3_vl_text_mrope`。驻留 scratch 是 `256` bytes。

一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_mrope.json`，SHA256 为
`8411b055efea353da282dc295558b5db99b6d088b7814a62673ce54ecc7d46b9`。
`closed=false`，`acceptance_fixture=false`。position `107 / 203 / 331` 的最后
一步都是 `0` bytes、`0` 次分配，归因覆盖 exact。`mrope_prepare` 是 `0`。相对
Chapter 60 的 `256 / 2`，正好少了 `256` bytes 和 `2` 次分配。前四枚是
`[1987, 2169, 375, 265]`。32-token 没有分叉。128/256 仍在 step 70 分叉：
LifeAI zero-based `21239`，HF `64692`，margin `0.02797699`。没有新的线性头。

同一次生成里，只有第一次 decode 分配。256-token 那次是 `2,018,180` bytes、
`56` 次分配，随后 `254` 步都是 `0`。这 `56` 次是按 capacity 创建的 scratch，
不是每步流量。三次 sample 之间 CUDA pool used 没有增加。

Episode 10 保持 Open。这不是整条 decode 零分配：prefill、vision、第一次
scratch 创建和 host greedy 仍在。最终 logits 没有升成 Float32。
