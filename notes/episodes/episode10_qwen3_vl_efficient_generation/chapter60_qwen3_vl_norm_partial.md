# Chapter 60 — Qwen3-VL decode hidden RMSNorm partial

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-08
>
> 关闭：2026-10-08

## 结果

Chapter 59 之后，`pre_attention_norm` 和 `post_attention_norm` 各剩 `448`
bytes，`final_norm` 剩 `16` bytes。这是 `sum!(abs2, dest, x)` 在归约长度
`2048` 上为多 block kernel 分配的 partial：`4` 个 Float32，`16` bytes、`1`
次分配。长度 `16` 和 `128` 仍是单 block，本来就是 `0`。

GPU 的 `sum!` 把 init 传成 `zero(Float32)`，不是 `nothing`。按同一条
`partial_mapreduce_grid`、同一个 launch configuration 复用这块 partial 之后，
长度 `2048` 的 `abs2` 归约与 `sum!` 逐位一致，复用调用是 `0` GPU bytes。完整
hidden RMSNorm 也与 `_bf16a_rmsnorm` 逐位一致。CPU 仍走 `sum!`。

一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_norm_partial.json`，SHA256 为
`b9e6d0ab4331d689d448ff8d03b7714d283c0553454f37f89b37abb43276361e`。
`closed=false`，`acceptance_fixture=false`。position `107 / 203 / 331` 的最后
一步都是 `256` bytes、`2` 次分配，归因覆盖 exact。相对 Chapter 59 的
`1,168 / 59`，正好少了 `912` bytes 和 `57` 次分配。三处 hidden norm 都是 `0`。
前四枚是 `[1987, 2169, 375, 265]`。32-token 没有分叉。128/256 仍在 step 70
分叉：LifeAI zero-based `21239`，HF `64692`，margin `0.02797699`。没有新的
线性头。

剩下的阶段是 `mrope_prepare` 的 `256` bytes、`2` 次分配。Episode 10 保持
Open。最终 logits 没有升成 Float32。
