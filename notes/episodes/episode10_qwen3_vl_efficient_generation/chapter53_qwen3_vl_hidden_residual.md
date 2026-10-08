# Chapter 53 — Qwen3-VL decode hidden residual 常量分配

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-08
>
> 关闭：2026-10-08

## Open：核心问题

Chapter 52 之后，`mlp_down_projection_residual` 和
`attention_output_projection_residual` 并列成为最大的常量阶段。两者都是一次
hidden 宽度的 BF16 GEMM，再加同一串 residual 舍入。

## 复用

两处调用在一层里是先后发生的，所以共用一块 projected buffer 和一块 residual
输出。输出按 `hidden_size` 预分配，`28` 层共用。GPU 用 `mul!`，CPU 用和
`_bf16_linear` 相同的 chunked Float32。residual 先把输入抄进 Float32，因此
下一步即使读到同一块输出也不会覆盖还没读完的值。

驻留 scratch 是 `24,576` bytes：
`hidden_size * (2 * sizeof(Float32) + 2 * sizeof(BFloat16))`。

## Close 回顾

Chapter 53 于 2026-10-08 Closed。一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_hidden_residual.json`，SHA256 为
`73f44165347e1eae6628be1cc9c9b05f51bf86b8d496b34cb0a40474c08317a9`。

- **验证证据**：position `107 / 203 / 331` 的最后一步都是 `547,265` bytes、
  `1,493` 次分配，归因覆盖 exact，profiled/unprofiled token 一致，storage
  identity 通过。两个 residual 阶段都从 `688,352 / 168` 降到 `224 / 56`。
  `224 = 28 × 8`，每次 `mul!` 留下 `8` bytes。相对 Chapter 52 的
  `1,923,521 / 1,717`，正好少了 `1,376,256` bytes 和 `224` 次分配。前四枚是
  `[1987, 2169, 375, 265]`。32-token 没有分叉；128/256 仍在 step 70 分叉，
  margin `0.02797699`。vision load / text load / vision forward 为
  `2.726 / 17.437 / 12.738` 秒。`CUDA.reclaim()` 回收 device free
  `2,632,318,976` bytes，释放 pool cached `2,617,245,696` bytes。
- **带到下一问的问题**：最大阶段现在是 `vocab_logits`，`303,880` bytes、`3`
  次分配、`1` 次调用。其后是 `qkv_projection` 的 `230,048` bytes。

Episode 10 保持 Open。最终 logits 没有升成 Float32。本章不声称整条 decode
零分配。
