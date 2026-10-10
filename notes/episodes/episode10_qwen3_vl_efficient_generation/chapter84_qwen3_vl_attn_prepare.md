# Chapter 84 — Qwen3-VL decode attention 的准备启动

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-09
>
> 关闭：2026-10-09

## 结果

Chapter 83 之后，attention 先单独载入 query 布局，再单独把 K/V 拷进
scratch。两件事都是下标赋值，互不依赖。

一次启动做完这两件事。sequence `1 / 80 / 331` 的 query、整段 K 和整段 V
与原来的两次 kernel 都是 0 个不一致。CPU 仍走 `permutedims!` 和分开的
拷贝。

一层少 `112` bytes，28 层是 `3,136`。

一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_attn_prepare.json`，
SHA256 为
`fe36b8825d6582c932c9bf264db617d04a1276bff0210987320c78f001d54a35`。
`closed=false`，`acceptance_fixture=false`。position `107 / 203 / 331`
的最后一步 GPU 仍是 `0` bytes、`0` 次分配。CPU 都是 `381,328`。
256-token 稳态一步从 Chapter 83 的 `384,464` 降到 `381,328`。第一次 decode
的 GPU 分配仍是 `2,018,180` bytes、`56` 次。前四枚是 `[1987, 2169, 375, 265]`。
32-token 没有分叉。128/256 仍在 step 70 分叉：LifeAI zero-based `21239`，
HF `64692`，margin `0.02797699`。

profiled 最后一步里，最大阶段仍是 attention。Episode 10 保持 Open。最终
logits 没有升成 Float32。
