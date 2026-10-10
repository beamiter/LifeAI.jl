# Chapter 133 — Qwen3-VL scratch 切片共用父引用

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-10
>
> 关闭：2026-10-10

## 结果

decode scratch 的切片原来各拷一份 `DataRef`。现在它们共用父缓冲
那一份，两块父缓冲留在 cache 上。CPU 路径仍是原来的 `similar`。
计算没有变。

一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_shared_ref.json`，
SHA256 为
`103dc1d18457d27054ddad1215f1d0b777b98694747e9b426f5101e0cc9230dd`。
`closed=false`，`acceptance_fixture=false`。position `107 / 203 / 331`
的最后一步 GPU 仍是 `0` bytes、`0` 次分配。CPU 仍是 `0`。
32/128/256 的第一次 decode CPU 都从 `9,168` 降到 `7,408`。下降都是
`1,760` bytes。256-token 第一次 decode 的 GPU 字节仍是 `2,018,180`，
分配次数仍是 `2`。前四枚是 `[1987, 2169, 375, 265]`。32-token 没有
分叉。128/256 仍在 step 70 分叉：LifeAI zero-based `21239`，HF
`64692`，margin `0.02797699`。

第一次 decode 仍要创建 scratch。不声称整条 decode 零分配。profiler
采样的最大阶段仍是 mlp_gate_up。Episode 10 保持 Open。最终 logits
没有升成 Float32。
