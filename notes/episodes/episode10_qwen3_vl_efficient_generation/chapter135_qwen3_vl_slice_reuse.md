# Chapter 135 — Qwen3-VL 复用 scratch 切片

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-10
>
> 关闭：2026-10-10

## 结果

第一次 decode 仍 carve 出 56 张 CuArray。之后的请求不再新建它们，
只改 data、offset 和 dims，指向新的 BF16 父缓冲和 Float32 父缓冲。
`capacity` 不能改，attention scratch 仍按新容量重建，其余 scratch
沿用原对象。objectid 不变，kernel 参数盒和 reshape 在设备指针或形状
变化时重绑。两块父缓冲留在 holder 上。计算没有变。

一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_slice_reuse.json`，
SHA256 为
`526066a3cdf57673e5384c36a625c1acd2861a451a5dd47a24c1c5f9af48640b`。
`closed=false`，`acceptance_fixture=false`。position `107 / 203 / 331`
的最后一步 GPU 仍是 `0` bytes、`0` 次分配。CPU 仍是 `0`。
32/128/256 的第一次 decode CPU 都从 `6,768` 降到 `3,264`。下降都是
`3,504` bytes。256-token 第一次 decode 的 GPU 字节仍是 `2,018,180`，
分配次数仍是 `2`。前四枚是 `[1987, 2169, 375, 265]`。32-token 没有
分叉。128/256 仍在 step 70 分叉：LifeAI zero-based `21239`，HF
`64692`，margin `0.02797699`。

第一次 decode 仍要创建 attention scratch 和两块父缓冲。不声称整条
decode 零分配。profiler 采样的最大阶段仍是 mlp_gate_up。Episode 10
保持 Open。最终 logits 没有升成 Float32。
