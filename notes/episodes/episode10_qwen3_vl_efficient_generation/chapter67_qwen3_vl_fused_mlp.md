# Chapter 67 — Qwen3-VL decode MLP SiLU 合成 kernel

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-09
>
> 关闭：2026-10-09

## 结果

融合 SiLU 乘积原先是五次中间维广播。GPU 路径收成一次 kernel：

`bf16(f32(bf16(g / (1 + exp(-g)))) * f32(up))`

CPU 仍走广播。中间维 `6144` 的 3 组随机输入与 CPU 路径逐位一致，复用
调用是 `0` GPU bytes。

Chapter 62—67 的联合关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_fused_elementwise.json`，
SHA256 为
`ebf007aafe96f34972d45a8429899acdcfd8781468e4f9b24611396ec9ab8b4a`。
`closed=false`，`acceptance_fixture=false`。position `107 / 203 / 331`
的最后一步 GPU 仍是 `0` bytes、`0` 次分配。256-token 稳态一步的 CPU
分配从 Chapter 61 的 `7,586,128` bytes 降到 `2,168,720`。前四枚是
`[1987, 2169, 375, 265]`。32-token 没有分叉。128/256 仍在 step 70
分叉：LifeAI zero-based `21239`，HF `64692`，margin `0.02797699`。

剩下的大头是 per-head `cublasGemmEx` 的 Julia 启动对象，以及 attention
softmax 那几条广播。Episode 10 保持 Open。最终 logits 没有升成 Float32。
