# Chapter 144 — Qwen3-VL GPU 的 BF16 转 Float32 复用 kernel

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-10
>
> 关闭：2026-10-10

## 结果

连续的 GPU 数组不再每次为转换启动一套新的 launch。复用的 kernel
按 `1.0f0 * Float32(x[i])` 写入。不连续的数组仍走 `copyto!`。CPU
路径仍是原来的广播。数值没有变。

一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_promote.json`，
SHA256 为
`25f498529f6e84e7a8e2902b5e02c79e5fb62519a1472e5e015d8b0fcb9ab0a7`。
`closed=false`，`acceptance_fixture=false`。position `107 / 203 / 331`
的最后一步 GPU 仍是 `0` bytes、`0` 次分配。CPU 仍是 `0`。
32/128/256 的第一次 decode CPU 仍是 `0`，GPU 字节仍是 `0`，分配次数
仍是 `0`。prefill GPU 仍是 `858,377,949` bytes、`4240` 次。32 的
prefill CPU 从 `6,941,832` 降到 `5,680,008`。128 从 `6,926,328` 降到
`5,654,808`。256 从 `6,918,616` 降到 `5,657,880`。前四枚是
`[1987, 2169, 375, 265]`。32-token 没有分叉。128/256 仍在 step 70
分叉：LifeAI zero-based `21239`，HF `64692`，margin `0.02797699`。

输出数组仍是每次 `similar`。warmup 的第一次构建和第一次 prefill 仍要
分配。不声称整条 decode 零分配。profiler 采样的最大阶段仍是
mlp_gate_up。Episode 10 保持 Open。最终 logits 没有升成 Float32。
