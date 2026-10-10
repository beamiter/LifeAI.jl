# Chapter 113 — Qwen3-VL decode mRoPE 主机拷贝不再查询 context

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-09
>
> 关闭：2026-10-09

## 结果

mRoPE 的 cos/sin 仍在主机上按原来的频率公式算好，再抄到设备。
原来的 `copyto!` 每次都会查询 CUDA context。

设备数组现在直接调用 `cuMemcpyHtoDAsync`。CPU 数组仍走 `copyto!`。
坐标 `0 / 1 / 97 / 331` 的设备和 `copyto!` 逐位一致。整步下降是
`64` bytes。

一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_mrope_copy.json`，
SHA256 为
`c29c0a498bb5eb7f421effdc7212f3205599d968fedc39e13fce7670602def1d`。
`closed=false`，`acceptance_fixture=false`。position `107 / 203 / 331`
的最后一步 GPU 仍是 `0` bytes、`0` 次分配。CPU 分别是 `4,352`、
`4,352` 和 `4,336`。256-token 稳态一步从 Chapter 112 的 `4,400`
降到 `4,336`。第一次 decode 的 GPU 分配仍是 `2,018,180` bytes、`56` 次。
前四枚是 `[1987, 2169, 375, 265]`。32-token 没有分叉。128/256 仍在
step 70 分叉：LifeAI zero-based `21239`，HF `64692`，margin `0.02797699`。

profiler 采样的最大阶段仍是 mlp_gate_up。Episode 10 保持 Open。
最终 logits 没有升成 Float32。
