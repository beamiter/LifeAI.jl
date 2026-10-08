# Chapter 50 — Qwen3-VL decode qk_norm 常量分配

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-08
>
> 关闭：2026-10-08

## Open：核心问题

Chapter 49 把 `qk_rope` 收进跨层 scratch 之后，单 token decode 的最大 GPU
allocation 变成 `qk_norm`。它在 position `107 / 203 / 331` 上不变。当前要回答的是：

> 这个常量分配的 shape、dtype 和 byte cap 是什么，能不能在不改变 BF16 舍入的
> 前提下复用？

## 已核对的分配

Chapter 49 的关闭剖面里，`qk_norm` 是 `1,066,240` bytes、`392` 次分配、`28`
次调用。每层是 `38,080` bytes 和 `14` 次分配。

Q 是 `(128, 16, 1, 1)`，K 是 `(128, 8, 1, 1)`，scale 长度是 `head_dim = 128`。
实现是 `_bf16a_rmsnorm`。RTX 4090 D 上一次 Q 调用是 `25,216` bytes、`7` 次分配，
一次 K 调用是 `12,864` bytes、`7` 次分配。合计正好是每层的 `38,080 / 14`。

Q 的 7 个数组是：

| 数组 | shape | dtype | bytes |
| --- | --- | --- | ---: |
| 输入 Float32 | `(128, 16, 1, 1)` | Float32 | 8,192 |
| `sum(abs2)` | `(1, 16, 1, 1)` | Float32 | 64 |
| mean square | `(1, 16, 1, 1)` | Float32 | 64 |
| normalized | `(128, 16, 1, 1)` | BFloat16 | 4,096 |
| scale Float32 | `(128, 1, 1, 1)` | Float32 | 512 |
| normalized Float32 | `(128, 16, 1, 1)` | Float32 | 8,192 |
| 输出 | `(128, 16, 1, 1)` | BFloat16 | 4,096 |

`8,192 + 64 + 64 + 4,096 + 512 + 8,192 + 4,096 = 25,216`。K 沿 head 轴减半，
scale 仍是 `512` bytes，所以是 `12,864` 而不是 Q 的一半。

Q 和 K 的结果要同时活到 RoPE，所以两块输出不能共用。中间数组按 query heads
分配，K 使用 head 前缀。驻留 scratch 是 `27,200` bytes。`sum!(abs2, dest, x)`
和随后的 broadcast 在预分配 buffer 上是 `0` GPU bytes，并且与
`_bf16a_rmsnorm` 逐位一致。

## 计划

| 工作项 | 所属主线 | 交付物 | 验收方式 | 状态 |
| --- | --- | --- | --- | --- |
| Q/K 分配账 | 高效推理 | 单次 Q 与单次 K 的 shape/dtype/bytes | 两者合计等于每层 `38,080 / 14` | 已完成 |
| 跨层 scratch | 高效推理 | 按 head 形状预分配的 buffer | 容量、dtype、reset、不与输入 alias | 已完成 |
| 数值不变 | 正确性 | 复用前后的 BF16 Q/K | 与 `_bf16a_rmsnorm` 逐位一致 | 已完成 |
| 重测 | 高效推理 | position 107/203/331 的 stage bytes | `qk_norm` 下降，且不出现新的线性头 | 已完成 |

## Close 条件

- 账目来自 Chapter 49 的 position `107 / 203 / 331` 剖面，或同协议的新剖面。
- 不把最终 logits 升成 Float32，也不把 step 70 分叉改写成 strict parity。
- 不声称整条 decode 零分配、稳定吞吐、batch/padding、multi-image/video 或 sampling。
- prefill、Float32 和多 token 仍走 `_qwen3_vl_text_rmsnorm`。

## Close 回顾

Chapter 50 于 2026-10-08 Closed。一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_qk_norm.json`，SHA256 为
`8f415d759aca2cf27a402c484c2c6c2c0a06624a25c64bceb6f74681f25afdad`。
`closed=false`，`acceptance_fixture=false`。

- **完成了什么**：拆开了单次 Q/K RMSNorm 的 7 个数组，并把同一串舍入收进 28 层
  共用的 scratch。复用后 `qk_norm` 的 GPU allocation 为 0。
- **验证证据**：离线回归覆盖 byte 公式、与 `_bf16a_rmsnorm` 逐位一致、
  head/token mismatch、Float32 cache 保持 `nothing`、prefill 不分配、reset
  保留 storage，以及两个 cache 不共享输出。Chapter 46 回归仍通过。position
  `107 / 203 / 331` 的最后一步都是 `4,479,881` bytes、`2,172` 次分配，归因
  覆盖 exact，`kv_write` 为 0。`qk_norm` 是 `0` bytes、`0` 次分配、`28` 次调用。
  相对 Chapter 49 的 `5,546,121 / 2,564`，正好少了 `1,066,240` bytes 和 `392`
  次分配。`qk_rope` 与 `mlp_activation` 仍是 `0`。驻留 scratch 是 `27,200`
  bytes。前四枚是 `[1987, 2169, 375, 265]`。32-token 没有分叉；128/256 仍在
  step 70 分叉，margin `0.02797699`。vision load / text load / vision forward
  为 `2.715 / 16.773 / 12.479` 秒。`CUDA.reclaim()` 回收 device free
  `2,627,731,456` bytes，释放 pool cached `2,617,245,696` bytes，pool used
  释放 `0`。
- **没有完成及原因**：128/256 仍在 step 70 与 HF Float32 分叉。最终 logits
  仍是 BF16。整条 decode 仍非零分配。
- **最重要的认知变化**：QK RMSNorm 的 7 个数组都不依赖 cache position。
  `sum!` 加预分配 broadcast 可以逐位复现，并且复用调用是 `0` GPU bytes。
- **是否满足 Close 条件**：shape、dtype、byte cap、逐位复用和同协议重测都已
  满足。三个 position 的 stage bytes 仍然相同，没有新的线性头。
- **带到下一问的问题**：`pre_attention_norm` 与 `post_attention_norm` 并列，
  都是 `918,176` bytes、`224` 次分配、`28` 次调用。单次 hidden RMSNorm 是
  `(2048, 1, 1)` 的 `32,792` bytes、`8` 次分配，`28 × 32,792 = 918,176`。
  `final_norm` 是同一次调用，`32,792` bytes、`8` 次分配。本章不开始复用它。

Episode 10 保持 Open。本章不声称 BF16 HuggingFace strict parity、整条 decode
零分配、稳定吞吐、batch/padding、multi-image/video 或 sampling。
