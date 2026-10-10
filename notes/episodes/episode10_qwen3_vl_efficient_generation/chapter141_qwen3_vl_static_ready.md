# Chapter 141 — Qwen3-VL 静态 Ready 调用

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-10
>
> 关闭：2026-10-10

## 结果

BF16 Ready 的具体类型写在入口上。`isa` 能收窄到这个类型，steady 调用不再
装箱 token。返回的 `(logits, cache)` 元组还在。计算没有变。

一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_static_ready.json`，
SHA256 为
`f9a66705ccb388606f478839ac20770733c0c008905ccdcd53f35ee599362831`。
`closed=false`，`acceptance_fixture=false`。position `107 / 203 / 331`
的最后一步 GPU 仍是 `0` bytes、`0` 次分配。CPU 仍是 `0`。
32/128/256 的第一次 decode CPU 都从 `48` 降到 `32`。下降都是
`16` bytes。第一次 decode 的 GPU 字节仍是 `0`，分配次数仍是 `0`。
前四枚是 `[1987, 2169, 375, 265]`。32-token 没有分叉。128/256 仍在
step 70 分叉：LifeAI zero-based `21239`，HF `64692`，margin
`0.02797699`。

第一次 decode 仍有返回元组的 `32` bytes。warmup 里的第一次构建仍要分配
父缓冲和 Ready。不声称整条 decode 零分配。profiler 采样的最大阶段仍是
mlp_gate_up。Episode 10 保持 Open。最终 logits 没有升成 Float32。
