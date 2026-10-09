# Chapter 69 — Qwen3-VL decode attention softmax 元素 kernel

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-09
>
> 关闭：2026-10-09

## 结果

Chapter 68 把 scale、exp 和 weight 收成一次 kernel。sequence `1`、`80`、
`330`、`331` 的 2048 个 attention 输出全部和广播路径不同，所以没有留下。

这一章仍把 `maximum!` 和 `sum!` 留在原来的位置，只替换它们前后的元素广播。
scale 按这个顺序写回同一块 Float32 buffer：

`f32 = 1.0f0 * f32(score)`，`scaled = bf16(f32 * scaling)`，`f32 = 1.0f0 * f32(scaled)`

exp 是 `exp(f32 - max)`。weight 是 `bf16(exp / sum)`。CPU 仍走原来的广播。

RTX 4090 D 上，sequence `1 / 80 / 330 / 331` 的 scaled、Float32 score、exp、
weight、maxima，以及 2048 个 attention 输出，和广播路径相比都是 0 个不一致。
一次复用的 attention 调用在 sequence `331` 从 `38,080` bytes 降到 `25,680`。
sequence `1` 和 `80` 的复用调用也是 `25,680`。

一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_softmax_kernel.json`，
SHA256 为
`1d463083013c0076f62032731f14a02129e0eaced160ee09e4cdb87e79a5d638`。
`closed=false`，`acceptance_fixture=false`。position `107 / 203 / 331`
的最后一步 GPU 仍是 `0` bytes、`0` 次分配。这三步的 CPU 分配是
`1,816,592`、`1,815,696`、`1,817,040`。256-token 稳态一步从 Chapter 67 的
`2,168,720` 降到 `1,817,040`。第一次 decode 的 GPU 分配仍是 `2,018,180`
bytes、`56` 次。前四枚是 `[1987, 2169, 375, 265]`。32-token 没有分叉。
128/256 仍在 step 70 分叉：LifeAI zero-based `21239`，HF `64692`，margin
`0.02797699`。

剩下的大头仍在 attention：query/K/V permute，以及 per-head `cublasGemmEx`
的 Julia 启动对象。其后是 QK norm。Episode 10 保持 Open。最终 logits 没有
升成 Float32。
