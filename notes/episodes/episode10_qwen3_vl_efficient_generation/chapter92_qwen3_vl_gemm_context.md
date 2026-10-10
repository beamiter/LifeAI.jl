# Chapter 92 — Qwen3-VL decode cublasGemmEx 的 context

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-09
>
> 关闭：2026-10-09

## 结果

Chapter 91 之后，每次 `cublasGemmEx` 仍先查询当前 context。那次查询分配
`16` bytes。参数、BF16 数据类型和 `CUBLAS_COMPUTE_32F` 都不变，调用改成
直接进入同一条函数。

`N` 和 `T`、单 token 的 `2048×2048`、`1024×2048`、`6144×2048`、`128×64`，
以及 8 头 batched mul，输出与原来的包装都是 0 个不一致。单 token 调用从
`16` bytes 降到 `0`。QKV 从 `336` 降到 `288`。gate/up 从 `32` 降到 `0`。
residual 从 `176` 降到 `160`。attention 从 `928` 降到 `672`。

一层有 23 次 GEMM：QKV 3 次，attention 16 次，O projection 1 次，gate/up
2 次，down projection 1 次。28 层是 644 次，再加上最后的 tied logits，
一共 645 次。每次少 `16` bytes，合计 `10,320`。

一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_gemm_context.json`，
SHA256 为
`5ca785fc6469ce8cd9abc963e9a7a362126eb2f01bf3fb490e5bcf272ab708e3`。
`closed=false`，`acceptance_fixture=false`。position `107 / 203 / 331`
的最后一步 GPU 仍是 `0` bytes、`0` 次分配。CPU 都是 `278,224`。
256-token 稳态一步从 Chapter 91 的 `288,544` 降到 `278,224`。第一次 decode
的 GPU 分配仍是 `2,018,180` bytes、`56` 次。前四枚是 `[1987, 2169, 375, 265]`。
32-token 没有分叉。128/256 仍在 step 70 分叉：LifeAI zero-based `21239`，
HF `64692`，margin `0.02797699`。

这次 profiled 最后一步里，最大阶段仍是 mlp_gate_up。Episode 10 保持 Open。
最终 logits 没有升成 Float32。
