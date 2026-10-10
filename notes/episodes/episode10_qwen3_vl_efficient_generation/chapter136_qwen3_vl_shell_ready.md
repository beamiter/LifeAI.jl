# Chapter 136 — Qwen3-VL 复用请求直接安装 Ready

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-10
>
> 关闭：2026-10-10

## 结果

后来的请求不再把 11 个 scratch 装进 cache 的 `Any` 字段。切片改指到新的
父缓冲之后，直接装进具体类型的 Ready，再走 steady 入口。profiled
warmup 会再编译这条入口，所以 32-token 的第一次 decode 不把那次编译算进去。
计算没有变。

一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_shell_ready.json`，
SHA256 为
`1cad8aca2a56b0149cd8533d45bb3d3230e1b16529216408bf1c8caa408454bf`。
`closed=false`，`acceptance_fixture=false`。position `107 / 203 / 331`
的最后一步 GPU 仍是 `0` bytes、`0` 次分配。CPU 仍是 `0`。
32/128/256 的第一次 decode CPU 都从 `3,264` 降到 `1,312`。下降都是
`1,952` bytes。256-token 第一次 decode 的 GPU 字节仍是 `2,018,180`，
分配次数仍是 `2`。前四枚是 `[1987, 2169, 375, 265]`。32-token 没有
分叉。128/256 仍在 step 70 分叉：LifeAI zero-based `21239`，HF
`64692`，margin `0.02797699`。

第一次 decode 仍要分配两块父缓冲和 attention scratch。不声称整条
decode 零分配。profiler 采样的最大阶段仍是 mlp_gate_up。Episode 10
保持 Open。最终 logits 没有升成 Float32。
