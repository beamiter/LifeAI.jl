# Chapter 91 — Qwen3-VL decode attention softmax 的启动对象

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-09
>
> 关闭：2026-10-09

## 结果

Chapter 90 之后，attention softmax 的一次调用是 `288` bytes。其中 `64`
bytes 是把 f32 scores 从函数里返回。kernel 本身不变。

编译好的 kernel 留在具体 `Ref` 里，成功时返回 `Bool`。sequence
`1 / 2 / 31 / 32 / 33 / 80 / 330 / 331` 的 output、整段 weights、
`weight_perm`、`maxima`，以及 scaled、exponents、`scores_f32` 的前缀，
与五次 kernel 都是 0 个不一致。超过 `sum!` 单 block 上限时仍走原来的
五次 kernel。编译后的 softmax 调用是 `224` bytes。attention 从 `992`
降到 `928`。

一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_softmax_launch.json`，
SHA256 为
`00534ab94c2fa492f615082b041b007f042cf72e0f907bb2a5f19ec8d0d4ea09`。
`closed=false`，`acceptance_fixture=false`。position `107 / 203 / 331`
的最后一步 GPU 仍是 `0` bytes、`0` 次分配。CPU 都是 `288,544`。
256-token 稳态一步从 Chapter 90 的 `297,952` 降到 `288,544`。第一次 decode
的 GPU 分配仍是 `2,018,180` bytes、`56` 次。前四枚是 `[1987, 2169, 375, 265]`。
32-token 没有分叉。128/256 仍在 step 70 分叉：LifeAI zero-based `21239`，
HF `64692`，margin `0.02797699`。

这次 profiled 最后一步里，最大阶段仍是 mlp_gate_up。Episode 10 保持 Open。
最终 logits 没有升成 Float32。
