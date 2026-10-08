# Chapter 49 — Qwen3-VL decode qk_rope 常量分配

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-08
>
> 关闭：2026-10-08

## Open：核心问题

Chapter 48 把不随 position 增长的 `mlp_activation` 收进了跨层 scratch。重测后，
单 token decode 的最大 GPU allocation 变成 `qk_rope`，而且它在 position
`107 / 203 / 331` 上保持不变。当前最需要回答的是：

> 这个常量分配的 shape、dtype 和 byte cap 是什么，能不能在不改变 BF16 舍入的
> 前提下复用？

## 预期结果

本章 Close 时，应当可以展示或验证：

1. `qk_rope` 的 shape、dtype、byte cap，以及 Q 与 K 在 28 层里的重复方式。
2. 若复用，复用前后的 BF16 Q/K 与原来的 `_bf16a_apply_rope` 逐位一致。
3. 若不复用，写明剩余分配为什么必须留下。

## 已核对的分配

Chapter 48 的关闭剖面里，position `107 / 203 / 331` 的 `qk_rope` 都是
`2,265,088` bytes、`1,064` 次分配、`28` 次调用。每层是 `80,896` bytes 和
`38` 次分配。decode 只有 1 个 token，所以它不随 cache position 增长。

Qwen3-VL-2B 的 text `head_dim` 是 `128`，query heads 是 `16`，KV heads 是
`8`。单 token、batch 1 的 Q 是 `(128, 16, 1, 1)`，K 是 `(128, 8, 1, 1)`。
实现是 `_bf16a_apply_rope`：half 切片、Float32 提升、中间 BF16 舍入，再
`cat` 回 BF16。这个函数也被 dense Qwen3、MoE 和 XLA 使用，本章不改它。

RTX 4090 D 上，一次 Q 调用是 `53,760` bytes 和 `19` 次分配，一次 K 调用是
`27,136` bytes 和 `19` 次分配。`53,760 + 27,136 = 80,896`，`19 + 19 = 38`，
与每层剖面一致。Q 的 19 次分配是：

| 语句 | dtype | bytes | 次数 |
| --- | --- | ---: | ---: |
| 两个 half 切片 | BFloat16 | 2,048 × 2 | 2 |
| 两个 half Float32 | Float32 | 4,096 × 2 | 2 |
| cos / sin Float32 | Float32 | 256 × 2 | 2 |
| upper 融合表达式 | 混合 | 16,384 | 5 |
| lower 融合表达式 | 混合 | 16,384 | 5 |
| upper / lower 的 BF16 | BFloat16 | 2,048 × 2 | 2 |
| `cat` 输出 | BFloat16 | 4,096 | 1 |

K 的形状沿 head 轴减半，cos/sin 仍各是 `256` bytes，所以不是 Q 的一半：
`27,136`。`reshape` 本身是 `0` bytes。把 half 写进预分配输出的 view，以及把
K 写进 query 宽度 buffer 的 head 前缀，也都是 `0` GPU bytes。

Q 和 K 的结果要同时活到 `kv_write` 和 attention，所以两块输出不能共用。
中间数组可以在 Q/K 之间以及 28 层之间复用。驻留 scratch 按 query heads 分配
7 个 half-width Float32、1 个 half-width BF16、cos/sin 两张 Float32 表，再加
Q/K 两个输出。`head_dim = 128`、`16 / 8` heads 时是 `37,376` bytes。

## 计划

| 工作项 | 所属主线 | 交付物 | 验收方式 | 状态 |
| --- | --- | --- | --- | --- |
| Q/K 分配账 | 高效推理 | 单次 Q 与单次 K 的 shape/dtype/bytes | 两者合计等于每层 `80,896 / 38` | 已完成 |
| 跨层 scratch | 高效推理 | 按 head 形状预分配的 buffer | 容量、dtype、reset、不与输入 alias | 已完成 |
| 数值不变 | 正确性 | 复用前后的 BF16 Q/K | 与 `_bf16a_apply_rope` 逐位一致 | 已完成 |
| 重测 | 高效推理 | position 107/203/331 的 stage bytes | `qk_rope` 下降，且不出现新的线性头 | 已完成 |

## Close 条件

- 账目来自 Chapter 48 已冻结的 position `107 / 203 / 331` 剖面，或同协议的新剖面。
- 不把最终 logits 升成 Float32，也不把 BF16 与 HF Float32 的 step 70 分叉改写成 strict parity。
- 不声称整条 decode 零分配、稳定吞吐、batch/padding、multi-image/video 或 sampling。
- 不修改共享的 `_bf16a_apply_rope`。prefill、Float32 和多 token 仍走原路径。

## Close 回顾

Chapter 49 于 2026-10-08 Closed。一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_qk_rope.json`，SHA256 为
`d8726e41c1171fe42b823254bb0ccb3fde40baf547c700dd87bda228f0c398d8`。
`closed=false`，`acceptance_fixture=false`，因为这是 allocation 记录，不是
BF16 token acceptance。

- **完成了什么**：拆开了单次 Q/K RoPE 的 19 次分配，并把同一串 BF16 舍入收进
  28 层共用的 scratch。复用后该 stage 的 GPU allocation 为 0。
- **验证证据**：离线回归覆盖 byte 公式、Q/K 与 `_bf16a_apply_rope` 逐位一致、
  head/token mismatch、prefill 不分配、Float32 cache 保持 `nothing`、decode
  后 object identity、reset 保留 storage，以及两个 cache 不共享输出。Chapter
  46 static cache 回归仍通过。RTX 4090 D 上 position `107 / 203 / 331` 的
  最后一步 profiled/unprofiled bytes 与 allocation count 都是 `5,546,121 /
  2,564`，归因覆盖 exact，`kv_write` 为 0，storage identity 通过。`qk_rope`
  三次都是 `0` bytes、`0` 次分配、`28` 次调用。相对 Chapter 48 关闭剖面的
  `7,811,209 / 3,628`，正好少了 `2,265,088` bytes 和 `1,064` 次分配。
  `mlp_activation` 仍是 `0`。attention 仍是 `7,168` bytes。驻留 scratch 是
  `37,376` bytes。前四枚都是 `[1987, 2169, 375, 265]`。32-token 没有分叉；
  128、256 和 warmup 仍在 generated step 70 分叉，LifeAI zero-based `21239`，
  HF `64692`，margin `0.02797699`。vision load / text load / vision forward
  为 `2.861 / 16.530 / 12.671` 秒。`CUDA.reclaim()` 回收 device free
  `2,627,272,704` bytes，释放 pool cached `2,617,245,696` bytes，pool used
  释放 `0`。
- **没有完成及原因**：128/256 的 BF16 自由生成仍与 HF Float32 oracle 在
  step 70 分叉。最终 logits 没有升成 Float32。整条 decode 仍非零分配。共享的
  `_bf16a_apply_rope` 没有修改。
- **最重要的认知变化**：融合表达式本身仍会留下中间数组；把每个舍入点写成
  预分配 buffer 上的 broadcast 之后，Q 与 K 都是 `0` GPU bytes，并且与原函数
  逐位一致。没有出现新的随 position 增长的头。
- **是否满足 Close 条件**：shape、dtype、byte cap、逐位复用、reset/ownership
  和同协议重测都已满足。BF16 128/256 完整 prefix exact 仍不满足，并继续作为
  跨 dtype 边界记录。
- **带到下一问的问题**：最大的常量 stage 现在是 `qk_norm`，三个 position 都是
  `1,066,240` bytes、`392` 次分配、`28` 次调用。其后是 `pre_attention_norm`
  和 `post_attention_norm`，各 `918,176` bytes。本章不开始复用它们。

Episode 10 保持 Open。本章不声称 BF16 HuggingFace strict parity、整条 decode
零分配、稳定吞吐、batch/padding、multi-image/video 或 sampling。
