# Chapter 143 — Qwen3-VL GPU 的 BF16 转 Float32 直接拷贝

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-10
>
> 关闭：2026-10-10

## 结果

GPU 上的 `_bf16a_f32` 改成 `copyto!`。原来的 `1.0f0 .* x` 是精确的
BF16 到 Float32 转换，广播会分配一次 launch。CPU 路径仍是原来的广播。
数值没有变。

一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_f32_copy.json`，
SHA256 为
`f132e3624ccb19b175babe19eafd447c3300356e738edeada965fc11a6a4f0ad`。
`closed=false`，`acceptance_fixture=false`。position `107 / 203 / 331`
的最后一步 GPU 仍是 `0` bytes、`0` 次分配。CPU 仍是 `0`。
32/128/256 的第一次 decode CPU 仍是 `0`，GPU 字节仍是 `0`，分配次数
仍是 `0`。prefill GPU 仍是 `858,377,949` bytes、`4240` 次。32 的
prefill CPU 从 `8,034,056` 降到 `6,941,832`。128 从 `8,009,544` 降到
`6,926,328`。256 从 `8,007,336` 降到 `6,918,616`。前四枚是
`[1987, 2169, 375, 265]`。32-token 没有分叉。128/256 仍在 step 70
分叉：LifeAI zero-based `21239`，HF `64692`，margin `0.02797699`。

输出数组仍是每次 `similar`。warmup 的第一次构建和第一次 prefill 仍要
分配。不声称整条 decode 零分配。profiler 采样的最大阶段仍是
mlp_gate_up。Episode 10 保持 Open。最终 logits 没有升成 Float32。
