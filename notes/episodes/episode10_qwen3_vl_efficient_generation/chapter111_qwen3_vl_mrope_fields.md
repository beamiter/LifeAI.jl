# Chapter 111 — Qwen3-VL decode mRoPE 不再传递整个 spec

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-09
>
> 关闭：2026-10-09

## 结果

mRoPE 只用 `head_dim`、`rope_theta` 和 `mrope_section`。原来把整个
spec 传进去，这个结构体里有字符串，调用时被装箱。

热路径现在只传这三个值。频率公式没有变。原 spec 入口仍转给这三个值。

一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_mrope_fields.json`，
SHA256 为
`a0cd3dc851bc37e61cf7e6d978514651fbcf51ae340b3e91ee461bc5d1f6778d`。
`closed=false`，`acceptance_fixture=false`。position `107 / 203 / 331`
的最后一步 GPU 仍是 `0` bytes、`0` 次分配。CPU 分别是 `5,264`、
`5,264` 和 `5,248`。256-token 稳态一步从 Chapter 110 的 `5,344`
降到 `5,248`。整步下降是 `96` bytes。第一次 decode 的 GPU 分配仍是
`2,018,180` bytes、`56` 次。前四枚是 `[1987, 2169, 375, 265]`。
32-token 没有分叉。128/256 仍在 step 70 分叉：LifeAI zero-based
`21239`，HF `64692`，margin `0.02797699`。

profiler 采样的最大阶段仍是 mlp_gate_up。Episode 10 保持 Open。
最终 logits 没有升成 Float32。
