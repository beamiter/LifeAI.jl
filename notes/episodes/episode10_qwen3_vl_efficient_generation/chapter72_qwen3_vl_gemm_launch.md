# Chapter 72 — Qwen3-VL decode cublasGemmEx 的启动对象

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-09
>
> 关闭：2026-10-09

## 结果

Chapter 71 之后，一次复用的 attention 调用是 `11,632` bytes。其中两次
per-head GEMM 仍是大头。直接调用 `cublasGemmEx` 一次是 `384` bytes：
`pointer` 会走 `convert(::CuPtr, ::Managed)` 的 stream 记账，`@checked`
再包一层 closure。

GPU 路径改成同一条 `cublasGemmEx`（`CUBLAS_COMPUTE_32F`、
`CUBLAS_GEMM_DEFAULT`、device alpha `1`、beta `0`），但地址直接从
`CuArray` 的 byte offset 取出，和 `pointer` 是同一个值。handle、alpha、
beta 和 BF16 的 `cudaDataType` 都是进程内持久对象。失败时仍看 status。

稠密矩阵一次调用是 `16` bytes。sequence 比 capacity 短的 head 切片是
`80` bytes。这两种，加上 `(6144, 2048)` 的线性层和 tied embedding 的转置，
都与原来的 `cublasGemmEx` 逐位一致。一次 attention 调用降到 `5,744`
bytes，sequence `80` 和 `331` 相同。

一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_gemm_launch.json`，
SHA256 为
`33a5f3d04f0bdc2cf405358585cec60a26607e0fe259453f331ca9452b36c355`。
`closed=false`，`acceptance_fixture=false`。position `107 / 203 / 331`
的最后一步 GPU 仍是 `0` bytes、`0` 次分配。CPU 是 `873,104`、`873,168`、
`873,104`。256-token 稳态一步从 Chapter 71 的 `1,126,224` 降到 `873,104`。
第一次 decode 的 GPU 分配仍是 `2,018,180` bytes、`56` 次。前四枚是
`[1987, 2169, 375, 265]`。32-token 没有分叉。128/256 仍在 step 70 分叉：
LifeAI zero-based `21239`，HF `64692`，margin `0.02797699`。

profiled 最后一步里，attention 仍是单阶段最大，pre/post hidden norm 的
升精度和除法紧随其后。Episode 10 保持 Open。最终 logits 没有升成 Float32。
