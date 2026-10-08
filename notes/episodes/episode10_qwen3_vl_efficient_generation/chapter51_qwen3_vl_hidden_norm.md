# Chapter 51 — Qwen3-VL decode hidden RMSNorm 常量分配

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-08
>
> 关闭：2026-10-08

## Open：核心问题

Chapter 50 把 `qk_norm` 收进 scratch 之后，最大的常量阶段变成并列的
`pre_attention_norm` 和 `post_attention_norm`。它们和 `final_norm` 是同一次
hidden RMSNorm。当前要回答的是：

> 这次调用的 shape、dtype 和 byte cap 是什么，大数组能不能在不改变 BF16
> 舍入的前提下复用？

## 已核对的分配

Chapter 50 的关闭剖面里，`pre_attention_norm` 和 `post_attention_norm` 各是
`918,176` bytes、`224` 次分配、`28` 次调用。`918,176 / 28 = 32,792`，
`224 / 28 = 8`。`final_norm` 是 `32,792` bytes、`8` 次分配、`1` 次调用。

输入是 `(2048, 1, 1)` BF16，scale 长度是 `hidden_size = 2048`。RTX 4090 D 上
一次 `_bf16a_rmsnorm` 是 `32,792` bytes、`8` 次分配。把大数组放进 scratch 后，
同进程结果与原函数逐位一致。`sum!` 在长度为 `2048` 时仍会分配 CUDA 多 block
归约的 partial，测得每次调用 `16` bytes、`1` 次分配。长度为 `128` 的 QK
归约可以放进一个 block，所以 Chapter 50 没有这笔 partial。

驻留 scratch 是 `32,772` bytes：3 个 hidden-sized Float32、2 个 hidden-sized
BF16，再加 1 个 Float32 mean。公式是
`hidden_size * (3 * sizeof(Float32) + 2 * sizeof(BFloat16)) + sizeof(Float32)`。
prefill、post-attention 和 final norm 共用这一块输出，因为同一时刻只有一个
结果还活着。

## 计划

| 工作项 | 所属主线 | 交付物 | 验收方式 | 状态 |
| --- | --- | --- | --- | --- |
| 单次分配账 | 高效推理 | hidden RMSNorm 的 shape/dtype/bytes | 等于 `32,792 / 8` | 已完成 |
| 跨调用 scratch | 高效推理 | 按 `hidden_size` 预分配的 buffer | dtype、reset、不与输入 alias | 已完成 |
| 数值不变 | 正确性 | 复用前后的 BF16 hidden | 与 `_bf16a_rmsnorm` 逐位一致 | 已完成 |
| 重测 | 高效推理 | position 107/203/331 的 stage bytes | 三个 norm stage 下降，且不出现新的线性头 | 已完成 |

## Close 条件

- 账目来自 Chapter 50 的 position `107 / 203 / 331` 剖面，或同协议的新剖面。
- 不把最终 logits 升成 Float32，也不把 step 70 分叉改写成 strict parity。
- 不声称整条 decode 零分配。若 `sum!` 的 partial 留着，写明它的 byte cap。
- prefill 和 Float32 仍走 `_qwen3_vl_text_rmsnorm`。

## Close 回顾

Chapter 51 于 2026-10-08 Closed。一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_hidden_norm.json`，SHA256 为
`cb5b785c94655583dfe361923817ccb515ad2673bee849b164df2c80ec8a6be0`。
`closed=false`，`acceptance_fixture=false`。

- **完成了什么**：把 prefill norm、post-attention norm 和 final norm 的大数组
  收进一块按 `hidden_size` 预分配的 scratch。三个阶段下降，并且与
  `_bf16a_rmsnorm` 逐位一致。
- **验证证据**：position `107 / 203 / 331` 的最后一步都是 `2,611,649` bytes、
  `1,773` 次分配，归因覆盖 exact。`pre_attention_norm` 和
  `post_attention_norm` 都从 `918,176 / 224` 降到 `448 / 28`。`final_norm`
  从 `32,792 / 8` 降到 `16 / 1`。相对 Chapter 50 的 `4,479,881 / 2,172`，
  正好少了 `1,868,232` bytes 和 `399` 次分配。驻留 scratch 是 `32,772` bytes。
  前四枚是 `[1987, 2169, 375, 265]`。32-token 没有分叉；128/256 仍在 step 70
  分叉，margin `0.02797699`。vision load / text load / vision forward 为
  `2.709 / 15.370 / 12.034` 秒。`CUDA.reclaim()` 回收 device free
  `2,656,108,544` bytes，释放 pool cached `2,617,245,696` bytes，pool used
  释放 `0`。
- **留下的分配**：每次 hidden RMSNorm 仍有 `16` bytes。这是 `sum!` 在归约长度
  `2048` 时为多 block 归约分配的 partial。QK norm 的归约长度是 `128`，能放进
  一个 block，所以没有这笔。改归约顺序会破坏与 `_bf16a_rmsnorm` 的逐位一致。
- **最重要的认知变化**：三个 hidden norm 可以共用一块输出。剩下的常量头不再是
  norm，而是 `mlp_gate_up_projection` 的 `688,576` bytes。
- **是否满足 Close 条件**：大数组已复用，partial 的 byte cap 已写明，三个
  position 没有新的线性头。
- **带到下一问的问题**：`mlp_gate_up_projection` 是 `688,576` bytes、`168` 次
  分配、`28` 次调用。本章不开始复用它。

Episode 10 保持 Open。本章不声称 BF16 HuggingFace strict parity、整条 decode
零分配、稳定吞吐、batch/padding、multi-image/video 或 sampling。
