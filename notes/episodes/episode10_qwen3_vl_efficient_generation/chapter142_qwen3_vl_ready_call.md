# Chapter 142 — Qwen3-VL 第一次 decode 直接调用 Ready

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-10
>
> 关闭：2026-10-10

## 结果

第一次 decode 的计时窗口改成和稳态步一样。小函数里直接调用具体
Ready。`hf_` 没有内联时，返回元组的 `32` bytes 落在窗口里。计算没有变。

一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_ready_call.json`，
SHA256 为
`c838c5f037c16a35820eb45aa8fbe7d93139a519ab7fac1f3946e6947053ccc6`。
`closed=false`，`acceptance_fixture=false`。position `107 / 203 / 331`
的最后一步 GPU 仍是 `0` bytes、`0` 次分配。CPU 仍是 `0`。
32/128/256 的第一次 decode CPU 都从 `32` 降到 `0`。下降都是
`32` bytes。第一次 decode 的 GPU 字节仍是 `0`，分配次数仍是 `0`。
前四枚是 `[1987, 2169, 375, 265]`。32-token 没有分叉。128/256 仍在
step 70 分叉：LifeAI zero-based `21239`，HF `64692`，margin
`0.02797699`。

这 `32` bytes 是未内联的 `hf_` 返回元组，不是 ready 步里的分配。
warmup 里的第一次构建仍要分配父缓冲和 Ready。不声称整条 decode
零分配。profiler 采样的最大阶段仍是 mlp_gate_up。Episode 10 保持
Open。最终 logits 没有升成 Float32。
