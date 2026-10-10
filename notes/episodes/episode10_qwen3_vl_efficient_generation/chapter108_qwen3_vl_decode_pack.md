# Chapter 108 — Qwen3-VL decode 不再拷贝整份参数

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-09
>
> 关闭：2026-10-09

## 结果

28 层函数每次调用都会把整份参数元组拷走。玩具模型里这份元组是
`2,608` bytes。

blocks、embedding、final norm 和 spec 现在放进 cache 上的一个可变包装。
包装按 embedding 数组认出同一份参数。稳态调用传这个包装，不再拷元组。
层内计算没有变。

一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_decode_pack.json`，
SHA256 为
`192477eaa948f93a095264f64205930cbaa87b75f8570e9eee22c25256b4f22b`。
`closed=false`，`acceptance_fixture=false`。position `107 / 203 / 331`
的最后一步 GPU 仍是 `0` bytes、`0` 次分配。CPU 分别是 `10,288`、
`10,288` 和 `10,272`。256-token 稳态一步从 Chapter 107 的 `13,344`
降到 `10,272`。整步下降是 `3,072` bytes。第一次 decode 的 GPU 分配仍是
`2,018,180` bytes、`56` 次。前四枚是 `[1987, 2169, 375, 265]`。
32-token 没有分叉。128/256 仍在 step 70 分叉：LifeAI zero-based
`21239`，HF `64692`，margin `0.02797699`。

profiler 采样的最大阶段仍是 mlp_gate_up。Episode 10 保持 Open。
最终 logits 没有升成 Float32。
