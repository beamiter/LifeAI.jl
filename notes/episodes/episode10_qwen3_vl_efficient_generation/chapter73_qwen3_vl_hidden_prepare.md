# Chapter 73 — Qwen3-VL decode hidden RMSNorm 的准备

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-09
>
> 关闭：2026-10-09

## 结果

Chapter 72 之后，pre/post hidden norm 是 attention 旁边最大的两段。一次
hidden norm 复用调用是 `4,864` bytes：升精度 `1,904`，缓存的 `sum!`
`1,008`，除以 `hidden_size` `1,680`，归一化 kernel `272`。

GPU 路径和 Chapter 71 的 QK 一样，归约不动：

`xf = 1.0f0 * f32(x)` 收成 kernel。`sqrt(sum / hidden_size + eps)` 写进原来的
归一化 kernel。`sum!(abs2, ...)` 仍走 Chapter 60 缓存的
`partial_mapreduce_grid`。CPU 仍是原来的广播。

RTX 4090 D 上，`2,048` 个元素与改前的 GPU 路径是 0 个不一致。一次复用调用
降到 `3,456` bytes。

一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_hidden_prepare.json`，
SHA256 为
`aa375cd1d1ea25041f1bf50259838af4c0f176253c77b90b4a6d794d34861fa0`。
`closed=false`，`acceptance_fixture=false`。position `107 / 203 / 331`
的最后一步 GPU 仍是 `0` bytes、`0` 次分配，CPU 都是 `677,920`。256-token
稳态一步从 Chapter 72 的 `873,104` 降到 `677,920`。`pre_attention_norm` 从
`172,528` 降到 `59,840`，`post_attention_norm` 从 `165,760` 降到 `59,264`，
`final_norm` 从 `5,568` 降到 `2,144`。第一次 decode 的 GPU 分配仍是
`2,018,180` bytes、`56` 次。前四枚是 `[1987, 2169, 375, 265]`。32-token
没有分叉。128/256 仍在 step 70 分叉：LifeAI zero-based `21239`，HF `64692`，
margin `0.02797699`。

profiled 最后一步里，最大阶段仍是 attention。Episode 10 保持 Open。最终
logits 没有升成 Float32。
