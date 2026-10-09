# Chapter 70 — Qwen3-VL decode attention permute kernel

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-09
>
> 关闭：2026-10-09

## 结果

Chapter 69 之后，一次复用的 attention 调用是 `25,488` bytes。其中 query 装载
`5,232`，K/V 各 `2,512`，weight 置换 `1,760`，context 写回 `2,736`。这些都是
`permutedims!` 的 Julia 启动对象，数值只是换下标。

GPU 路径改成四次拷贝 kernel，下标和原来的置换相同：

- query：`q[group, lane, kv] = query[lane, group + (kv - 1) * groups, 1, 1]`
- K/V：`dest[lane, token, head] = src[lane, head, token, 1]`，整段 capacity
- weight：`dest[token, group, head] = src[group, token, head]`
- context：`output[lane, group + (kv - 1) * groups, 1, 1] = context[lane, group, kv]`

CPU 仍走 `permutedims!`。

RTX 4090 D 上，sequence `1 / 80 / 330 / 331` 的 query、整段 K、整段 V、weight
置换和 2048 个 attention 输出，与 `permutedims!` 路径都是 0 个不一致。K/V
各 `338,944` 个元素。一次复用的 attention 调用降到 `11,632` bytes，sequence
`1` 和 `331` 相同。

一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_permute_kernel.json`，
SHA256 为
`269aa7775f10056a8abe10a70f0180288db4b995b862e07a2f00761903e8244f`。
`closed=false`，`acceptance_fixture=false`。position `107 / 203 / 331`
的最后一步 GPU 仍是 `0` bytes、`0` 次分配，CPU 都是 `1,422,800`。256-token
稳态一步从 Chapter 69 的 `1,817,040` 降到 `1,422,800`。第一次 decode 的 GPU
分配仍是 `2,018,180` bytes、`56` 次。前四枚是 `[1987, 2169, 375, 265]`。
32-token 没有分叉。128/256 仍在 step 70 分叉：LifeAI zero-based `21239`，
HF `64692`，margin `0.02797699`。

profiled 最后一步里，最大阶段变成 QK norm。attention 里剩下的是 per-head
`cublasGemmEx` 启动对象和 `maximum!`。Episode 10 保持 Open。最终 logits
没有升成 Float32。
