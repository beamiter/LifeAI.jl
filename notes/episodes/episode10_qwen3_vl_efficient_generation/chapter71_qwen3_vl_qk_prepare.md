# Chapter 71 — Qwen3-VL decode QK RMSNorm 的准备

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-09
>
> 关闭：2026-10-09

## 结果

Chapter 70 之后，profiled 最后一步里最大的阶段是 QK norm。一次 query 加
key 的复用调用是 `13,760` bytes。升精度广播、`sum!` 和除以 `head_dim`
各占大约 `3,000` bytes。归一化 kernel 本身是 `272` bytes。

GPU 路径改三处，归约顺序不动：

- `xf = 1.0f0 * f32(x)` 收成 kernel
- `sum!(abs2, mean, xf)` 改走 Chapter 60 那条缓存的
  `partial_mapreduce_grid`。query 16 head 和 key 的 8 head 前缀各一份
  plan，长度 `128` 仍是单 block
- `sqrt(mean / head_dim + eps)` 写进原来的归一化 kernel，不再先广播除法

CPU 仍是原来的广播和 `sum!`。

RTX 4090 D 上，query 的 `2,048` 个元素和 key 的 `1,024` 个元素与改前的 GPU
路径都是 0 个不一致。一次 query 加 key 的复用调用降到 `4,480` bytes。

一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_qk_prepare.json`，
SHA256 为
`ab40b3fee9c35b43385c32d5c70dbffff01f44ccfc548b76cd2ceac1bd8d1da6`。
`closed=false`，`acceptance_fixture=false`。position `107 / 203 / 331`
的最后一步 GPU 仍是 `0` bytes、`0` 次分配，CPU 都是 `1,126,224`。256-token
稳态一步从 Chapter 70 的 `1,422,800` 降到 `1,126,224`。第一次 decode 的 GPU
分配仍是 `2,018,180` bytes、`56` 次。前四枚是 `[1987, 2169, 375, 265]`。
32-token 没有分叉。128/256 仍在 step 70 分叉：LifeAI zero-based `21239`，
HF `64692`，margin `0.02797699`。

profiled 最后一步里，最大阶段回到 attention。其后是 pre/post hidden norm
的升精度和除法广播。Episode 10 保持 Open。最终 logits 没有升成 Float32。
