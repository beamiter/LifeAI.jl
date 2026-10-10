# Chapter 85 — Qwen3-VL decode attention weight 的置换

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-09
>
> 关闭：2026-10-09

## 结果

Chapter 84 之后，softmax 权重和把它置换到 `(token, group, head)` 是两次
kernel。有效 token 的值是 `BFloat16(exp / max)`，其余位置只是原样拷过去。

一次启动写完这两处。sequence `1 / 80 / 331` 的整段权重和整段置换结果与
原来的两次 kernel 都是 0 个不一致。CPU 仍是广播再 `permutedims!`。

一层少 `128` bytes，28 层是 `3,584`。

一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_weight_perm.json`，
SHA256 为
`922a55bc3533657cc5a09278239cd6e64d20c443147215edd441ab69b58fcda7`。
`closed=false`，`acceptance_fixture=false`。position `107 / 203 / 331`
的最后一步 GPU 仍是 `0` bytes、`0` 次分配。CPU 都是 `377,744`。
256-token 稳态一步从 Chapter 84 的 `381,328` 降到 `377,744`。第一次 decode
的 GPU 分配仍是 `2,018,180` bytes、`56` 次。前四枚是 `[1987, 2169, 375, 265]`。
32-token 没有分叉。128/256 仍在 step 70 分叉：LifeAI zero-based `21239`，
HF `64692`，margin `0.02797699`。

profiled 最后一步里，最大阶段仍是 attention。Episode 10 保持 Open。最终
logits 没有升成 Float32。
