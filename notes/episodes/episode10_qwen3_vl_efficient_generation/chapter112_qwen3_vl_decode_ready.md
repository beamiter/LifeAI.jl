# Chapter 112 — Qwen3-VL decode 复用具体 scratch

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-09
>
> 关闭：2026-10-09

## 结果

每一步都会从 cache 的 `Any` 字段里取出各块 scratch，再确认类型和设备。
这些字段每次读取都会装箱。

第一步 BF16 decode 之后，blocks、embedding、final norm、spec 和全部
scratch 留在一个具体类型的包装里。后面的步直接用这个包装，不再走
ensure。计算和原来的热路径是同一批函数。有 profiler 时仍走原来的
ensure。

一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_decode_ready.json`，
SHA256 为
`232a29ef3afb41f19c443d3aa2a5006cf13691c8e232afb609ea7b2422ed644d`。
`closed=false`，`acceptance_fixture=false`。position `107 / 203 / 331`
的最后一步 GPU 仍是 `0` bytes、`0` 次分配。CPU 分别是 `4,416`、
`4,416` 和 `4,400`。256-token 稳态一步从 Chapter 111 的 `5,248`
降到 `4,400`。整步下降是 `848` bytes。第一次 decode 的 GPU 分配仍是
`2,018,180` bytes、`56` 次。前四枚是 `[1987, 2169, 375, 265]`。
32-token 没有分叉。128/256 仍在 step 70 分叉：LifeAI zero-based
`21239`，HF `64692`，margin `0.02797699`。

profiler 采样的最大阶段仍是 mlp_gate_up。Episode 10 保持 Open。
最终 logits 没有升成 Float32。
