# Chapter 52 — Qwen3-VL decode gate/up projection 常量分配

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-08
>
> 关闭：2026-10-08

## Open：核心问题

Chapter 51 之后，最大的常量阶段是 `mlp_gate_up_projection`。当前要回答的是：

> gate 和 up 两次 GEMM 的输出能不能在不改变 BF16 数值的前提下复用？

## 已核对的分配

Chapter 51 的关闭剖面里，这个阶段是 `688,576` bytes、`168` 次分配、`28` 次调用。
每层 `24,592` bytes、`6` 次分配。

权重是 `(6144, 2048)` BF16，输入是 `(2048, 1, 1)`。RTX 4090 D 上一次分配型
GEMM 是 `12,296` bytes、`3` 次分配。其中输出是 `(6144, 1)` BF16 的 `12,288`
bytes。`mul!` 写进预分配输出后仍有 `8` bytes、`2` 次分配，并且与分配型 GEMM
逐位一致。两次调用是 `24,592` bytes，`28 × 24,592 = 688,576`。

gate 和 up 要同时活到 SiLU，所以输出是两块。驻留 scratch 是
`2 × 6144 × 2 = 24,576` bytes。CPU 不用 BF16 `mul!`，因为那条路径的累加和
`_bf16_linear` 不一致；CPU 仍按 chunk 把 Float32 GEMM 写进同一块输出。

## 计划

| 工作项 | 所属主线 | 交付物 | 验收方式 | 状态 |
| --- | --- | --- | --- | --- |
| 两次 GEMM 账 | 高效推理 | 单次输出 bytes 与 `mul!` 剩余 | 合计等于每层 `24,592 / 6` | 已完成 |
| 跨层 scratch | 高效推理 | gate 与 up 两块输出 | reset、不与输入 alias | 已完成 |
| 数值不变 | 正确性 | CPU 对 `_bf16a_linear`，GPU 对分配型 GEMM | 逐位一致 | 已完成 |
| 重测 | 高效推理 | position 107/203/331 的 stage bytes | 阶段下降，且不出现新的线性头 | 已完成 |

## Close 回顾

Chapter 52 于 2026-10-08 Closed。一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_gate_up.json`，SHA256 为
`e39f1e52b76a0c965175c7591f8f2a9aad60255d3d0fdf434ce91a9c30d10399`。

- **完成了什么**：gate 和 up 的 BF16 输出改由预分配 buffer 接收。GPU 上
  `mul!` 与分配型 GEMM 逐位一致。CPU 仍走 chunked Float32，与 `_bf16_linear`
  一致。
- **验证证据**：position `107 / 203 / 331` 的最后一步都是 `1,923,521` bytes、
  `1,717` 次分配。`mlp_gate_up_projection` 从 `688,576 / 168` 降到 `448 / 112`。
  `448 = 28 × 2 × 8`，每次 `mul!` 留下 `8` bytes、`2` 次分配。相对 Chapter 51
  的 `2,611,649 / 1,773`，正好少了 `688,128` bytes 和 `56` 次分配。驻留 scratch
  是 `24,576` bytes。前四枚是 `[1987, 2169, 375, 265]`。32-token 没有分叉；
  128/256 仍在 step 70 分叉，margin `0.02797699`。
- **留下的分配**：每次 GEMM 的 `8` bytes 来自 `mul!` 本身，不是输出张量。
- **带到下一问的问题**：`mlp_down_projection_residual` 与
  `attention_output_projection_residual` 并列，各 `688,352` bytes。

Episode 10 保持 Open。最终 logits 没有升成 Float32。
