# Chapter 76 — Qwen3-VL decode abs2 归约计划的具体类型

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-09
>
> 关闭：2026-10-09

## 结果

Chapter 75 之后，profiled 最后一步里最大的阶段是 QK norm。一次 query 加
key 的复用调用是 `4,480` bytes。其中 `sum!(abs2, ...)` 的缓存计划把 kernel
和 CartesianIndices 存在 `Any` 字段里，每次启动都重新装箱。

计划改成带具体字段类型的可变对象，按数组类型和形状分开缓存。启动仍是原来的
`partial_mapreduce_grid`，launch configuration 不变。顺序扫描的 `abs2` 和
`sum!` 不是同一条归约，没有换。

RTX 4090 D 上，query 16 head、key 的 8 head 前缀、hidden 长度 `2048` 的
`abs2` 归约与 `sum!` 都是 0 个不一致。query 那次复用调用是 `144` bytes，
key 是 `384` bytes。一次 query 加 key 降到 `2,752` bytes。

一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_abs2_plan.json`，
SHA256 为
`7aad1e4a7a720fe31136200fd016125e0ed48d002e073b62ed4e16f1060d6fd1`。
`closed=false`，`acceptance_fixture=false`。position `107 / 203 / 331`
的最后一步 GPU 仍是 `0` bytes、`0` 次分配，CPU 都是 `513,152`。256-token
稳态一步从 Chapter 75 的 `581,600` 降到 `513,152`。第一次 decode 的 GPU
分配仍是 `2,018,180` bytes、`56` 次。前四枚是 `[1987, 2169, 375, 265]`。
32-token 没有分叉。128/256 仍在 step 70 分叉：LifeAI zero-based `21239`，
HF `64692`，margin `0.02797699`。

profiled 最后一步里，最大阶段回到 attention，其后是 K/V 写入。Episode 10
保持 Open。最终 logits 没有升成 Float32。
