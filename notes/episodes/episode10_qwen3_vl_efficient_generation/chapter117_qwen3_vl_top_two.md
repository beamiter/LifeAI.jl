# Chapter 117 — Qwen3-VL 主机选 token 不再做全长排序

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-09
>
> 关闭：2026-10-09

## 结果

`partialsortperm(values, 1:2; rev=true)` 会做一条长度为 vocab 的下标，
再做一块同样长度的 quicksort scratch。两边都是 `1,215,488` bytes。

主机选择改成一次线性扫描。顺序和原来的排序一样：值更大的在前，
相等时下标更小的在前。decode 计算没有变。

一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_top_two.json`，
SHA256 为
`978dd531c9f64d4f0deda0237ce67d62db75712d312bb6406826da636112a02e`。
`closed=false`，`acceptance_fixture=false`。position `107 / 203 / 331`
的最后一步 GPU 仍是 `0` bytes、`0` 次分配。CPU 仍是 `0`。
256-token 主机选择从 Chapter 116 的 `2,739,400` 降到 `308,232`。
下降是 `2,431,168` bytes。前两次选择是 `304,120`。第一次 decode
的 GPU 分配仍是 `2,018,180` bytes、`56` 次。前四枚是
`[1987, 2169, 375, 265]`。32-token 没有分叉。128/256 仍在 step 70
分叉：LifeAI zero-based `21239`，HF `64692`，margin `0.02797699`。

剩下的主机分配是把 logits 抄回来。不声称整条 decode 零分配。
profiler 采样的最大阶段仍是 mlp_gate_up。Episode 10 保持 Open。
最终 logits 没有升成 Float32。
