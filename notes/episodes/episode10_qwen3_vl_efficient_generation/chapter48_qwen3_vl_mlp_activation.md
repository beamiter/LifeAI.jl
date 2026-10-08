# Chapter 48 — Qwen3-VL decode mlp_activation 常量分配

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-08
>
> 关闭：2026-10-08

## Open：核心问题

Chapter 47 把随 position 增长的 BF16 decode attention 临时数组收进了有界
scratch。重测后，单 token decode 的最大 GPU allocation 变成 `mlp_activation`，
而且它不随 position 变化。当前最需要回答的是：

> 这个常量分配的 shape、dtype 和 byte cap 是什么，能不能在不改变 BF16 数值的
> 前提下复用？

## 预期结果

本章 Close 时，应当可以展示或验证：

1. `mlp_activation` 的 shape、dtype、byte cap 和它在 28 层里的重复方式。
2. 若复用，复用前后的单 token decode logits 与 K/V 逐位一致。
3. 若不复用，写明剩余分配为什么必须留下。

## 已核对的分配

Qwen3-VL-2B 的 text `intermediate_size` 是 `6144`。单 token、batch 1 的 gate
和 up 都是 `(6144, 1, 1)` BF16。实现是：

```julia
gate_f32 = _bf16a_f32(gate)
activated = BFloat16.(gate_f32 ./ (1.0f0 .+ exp.(.-gate_f32)))
BFloat16.(_bf16a_f32(activated) .* _bf16a_f32(up))
```

RTX 4090 D 上，把这三行合成一次调用，CUDA.jl 报告 `98,304` bytes 和 `5` 次
分配。28 层是 `2,752,512` bytes 和 `140` 次分配，与 Chapter 47 的
position `107 / 203 / 331` 剖面逐项相同。语句拆开会得到更多中间数组，合计
`221,184` bytes；融合后只留下 5 个：

| 数组 | shape | dtype | bytes |
| --- | --- | --- | ---: |
| gate Float32 | `(6144, 1, 1)` | Float32 | 24,576 |
| SiLU 结果 | `(6144, 1, 1)` | BFloat16 | 12,288 |
| activated Float32 | `(6144, 1, 1)` | Float32 | 24,576 |
| up Float32 | `(6144, 1, 1)` | Float32 | 24,576 |
| gate-up 乘积 | `(6144, 1, 1)` | BFloat16 | 12,288 |

`24,576 × 3 + 12,288 × 2 = 98,304`。公式是
`intermediate_size * (3 * sizeof(Float32) + 2 * sizeof(BFloat16))`。这些数组
不依赖 cache position。

## 复用

BF16 单 token static decode 现在把这 5 块 buffer 放进
`Qwen3VLMlpActivationScratch`。scratch 按 `intermediate_size` 一次分配，28 层
顺序共用，`reset` 保留 storage。prefill 和 Float32 decode 不创建它。shape
不匹配时 fail closed，不回退到另一套数值。

同进程里，原地 broadcast 与原来的融合表达式逐位一致。RTX 4090 D 上，预热后的
一次复用调用是 `0` GPU bytes；原来的融合调用仍是 `98,304` bytes。
`intermediate_size = 6144` 的驻留 scratch 是 `98,304` bytes，
`intermediate_size = 32` 是 `512` bytes。

## 计划

| 工作项 | 所属主线 | 交付物 | 验收方式 | 状态 |
| --- | --- | --- | --- | --- |
| 融合分配账 | 高效推理 | 5 个数组的 shape/dtype/bytes | 与 Chapter 47 的 `2,752,512 / 140` 一致 | 已完成 |
| 跨层 scratch | 高效推理 | 按 `intermediate_size` 预分配的 5 块 buffer | 容量、dtype、reset、不与输入 alias | 已完成 |
| 数值不变 | 正确性 | 复用前后的 BF16 SiLU 乘积 | 逐位一致 | 已完成 |
| 重测 | 高效推理 | position 107/203/331 的 stage bytes | `mlp_activation` 下降，且不出现新的线性头 | 已完成 |

## Close 条件

- 账目来自 Chapter 47 已冻结的 position `107 / 203 / 331` 剖面，或同协议的新剖面。
- 不把最终 logits 升成 Float32，也不把 BF16 与 HF Float32 的 step 70 分叉改写成 strict parity。
- 不声称整条 decode 零分配、稳定吞吐、batch/padding、multi-image/video 或 sampling。

## Close 回顾

Chapter 48 于 2026-10-08 Closed。一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_mlp_scratch.json`，SHA256 为
`4aa2057607d940a12901bb3d71f8498a9fff1c6d548e597d07c8d7c3a22906de`。
`closed=false`，`acceptance_fixture=false`，因为这是 allocation 记录，不是
BF16 token acceptance。

- **完成了什么**：核对了融合 `mlp_activation` 的 5 个数组，并把它们收成不随
  position 增长、28 层共用的 scratch。复用后该 stage 的 GPU allocation 为 0。
- **验证证据**：离线回归覆盖 byte 公式、逐位 SiLU 乘积、shape mismatch、
  prefill 不分配、decode 后 object identity、reset 保留 storage，以及两个
  cache 不共享 buffer。Chapter 46 static cache 回归仍通过。RTX 4090 D 上
  position `107 / 203 / 331` 的最后一步 profiled/unprofiled bytes 与 allocation
  count 都是 `7,811,209 / 3,628`，归因覆盖 exact，`kv_write` 为 0，storage
  identity 通过。`mlp_activation` 三次都是 `0` bytes、`0` 次分配、`28` 次调用。
  相对 Chapter 47 关闭剖面的 `10,563,721 / 3,768`，正好少了 `2,752,512` bytes
  和 `140` 次分配。attention 仍是 `7,168` bytes。前四枚都是
  `[1987, 2169, 375, 265]`。32-token 没有分叉；128、256 和 warmup 仍在
  generated step 70 分叉，LifeAI zero-based `21239`，HF `64692`，margin
  `0.02797699`。vision load / text load / vision forward 为
  `2.570 / 16.584 / 12.316` 秒。`CUDA.reclaim()` 回收 device free
  `2,627,731,456` bytes，释放 pool cached `2,617,245,696` bytes，pool used
  释放 `0`。
- **没有完成及原因**：128/256 的 BF16 自由生成仍与 HF Float32 oracle 在
  step 70 分叉。最终 logits 没有升成 Float32，也不把这次分叉写成 strict
  parity。整条 decode 仍非零分配。
- **最重要的认知变化**：融合后的 MLP activation 是每层 5 个固定形状数组，
  不依赖 cache position。28 层共用一块 `98,304` byte scratch 后，这个 stage
  从单步最大头变成 `0`。没有出现新的随 position 增长的头。
- **是否满足 Close 条件**：shape、dtype、byte cap、逐位复用、reset/ownership
  和同协议重测都已满足。BF16 128/256 完整 prefix exact 仍不满足，并继续作为
  跨 dtype 边界记录。
- **带到下一问的问题**：最大的常量 stage 现在是 `qk_rope`，三个 position 都是
  `2,265,088` bytes、`1,064` 次分配、`28` 次调用。其后是 `qk_norm` 的
  `1,066,240` bytes。本章不开始复用它们，也不因此开启下一章。

Episode 10 保持 Open。本章不声称 BF16 HuggingFace strict parity、整条 decode
零分配、稳定吞吐、batch/padding、multi-image/video 或 sampling。
