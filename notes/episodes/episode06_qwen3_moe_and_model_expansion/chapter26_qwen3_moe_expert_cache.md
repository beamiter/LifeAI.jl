# Chapter 26 — Qwen3 MoE active-expert device cache

> 所属 Episode：Episode 06 — Qwen3 MoE 与模型架构扩展
>
> 状态：Closed

## Open：核心问题

Chapter 25 的 30B-A3B session 已能在 24 GB 显存上运行，但相同请求每次仍会
从 safetensors 读取并经 PCIe 上传全部 active experts。怎样利用路由局部性，
在不改变 BF16 数值路径、40K KV 容量和默认纯 streaming 行为的前提下，用有
明确 byte budget 的设备缓存消除重复 I/O？

## 预期结果

本章 Close 时，应当可以展示或验证：

1. 以 `(one-based layer, one-based expert)` 为 key 的设备 LRU，显存占用不超过
   调用方给定的 byte budget；
2. request reset 保留 expert cache，并提供显式清空接口；默认预算为零时保持
   Chapter 25 的逐层纯 streaming 路径；
3. 官方 30B-A3B 的真实重复请求不再读取或上传已缓存 experts，logits 与未命中
   请求逐位一致，并保持 Transformers BF16 reference argmax；
4. 在 RTX 4090 D 上冻结缓存容量、命中数、I/O 和端到端 latency，而不是只用
   tiny fixture 推断收益。

## 计划与结果

| 工作项 | 所属主线 | 交付物 | 验收方式 | 状态 |
| --- | --- | --- | --- | --- |
| byte-budgeted device LRU | 模型 / 工程 | session expert cache | tiny 容量与淘汰测试 | 已完成 |
| 生命周期与观测 | 工程 | stats/reset/clear API | 跨请求命中与显式清空 | 已完成 |
| 真实 30B cache benchmark | 模型 / 工程 | benchmark 与冻结摘要 | 8 GiB miss/hit 对照 | 已完成 |
| 数值不变性 | 模型 | fill/hit logits parity | exact repeat + HF argmax | 已完成 |

## Close 条件

- 缓存预算按实际三投影 BF16 tensor bytes 计量，峰值不越界；
- 同一路由重复请求的 expert read/upload 均降为零；
- 缓存命中前后 logits 逐位一致，真实 prompt/decode 的 reference argmax 一致；
- 冻结的真实 30B case 在 24 GB GPU 上保留可用显存，并展示端到端收益；
- 不把单个请求的零淘汰结果表述成任意自然文本都能零淘汰。

## 学习重点

- **缓存单元应跟物理读取单元一致**：每个 key 对应一层里的一个 expert，包含
  gate/up/down 三个投影。Qwen3-30B-A3B 的单项为 `9,437,184` bytes，LRU
  因而可以用实际 payload byte 精确预算。
- **请求 reset 不等于资源 reset**：KV position 和请求级计数需要复位，但跨
  请求复用的专家权重应保留；显式 clear 才撤销 expert tensor 的逻辑所有权。
- **命中数必须和 I/O 一起看**：一次 miss 会产生等量的磁盘读取和设备上传，
  hit 则两者都为零。只记录 hit rate 无法证明 PCIe 流量真的消失。
- **allocator free 不等于逻辑 cache bytes**：清空 Julia/CUDA 对象后，CUDA
  allocator pool 可能保留已申请显存。本章分别记录逻辑 cache bytes 和设备
  free memory，不把二者混为一谈。

## 风险与取舍

- loader 默认 `expert_cache_budget_bytes=0`，因此兼容并保留 Chapter 25 的低
  常驻纯 streaming 行为；8 GiB 是冻结请求的推荐值，不是通用默认值。
- 当前 LRU 在 miss 时同步读取 safetensors 并上传，尚未实现 pinned-memory
  staging、异步预取或计算/I/O 双缓冲。
- 每层计算前仍会把命中的独立 expert tensors 拼成 active-local 三维张量；本章
  消除了磁盘和 PCIe 重复流量，但还没有消除这一步设备内重组。
- 冻结 case 只有 2-token prompt + 1-token decode。更长自然文本可能覆盖超过
  8 GiB 的 expert-layer working set 并触发淘汰。

## 实验与过程记录

### 2026-08-12：8 GiB 真实 30B-A3B device LRU

- 在 Chapter 25 相同 RTX 4090 D、官方 revision
  `ad44e777…85d39`、40,960-token BF16 static KV 和 scalar production
  dispatch 上运行。session resident load 为 `15.81 s`。
- 缓存预算 `8,589,934,592` bytes；冻结请求实际形成 892 个 expert-layer
  entries、`8,417,968,128` bytes（预算利用率 `97.998%`），峰值相同、淘汰
  为零。运行结束仍有 `4,345,298,944` bytes GPU free。
- warm fill 的 prefill 激活 734 个 entries，读取/上传
  `6,926,893,056` bytes；decode 先命中 prompt 留下的 226 个 entries，再 miss
  158 个并读取/上传 `1,491,075,072` bytes。两段的唯一 working set 正好等于
  最终 cache bytes。
- 随后的 warm hit 中，prefill 734 次、decode 384 次全部命中，磁盘读取和设备
  上传均为零。prefill/decode/request 从 `18.738 / 4.537 / 23.274 s` 降至
  `10.558 / 2.959 / 13.517 s`，加速分别为
  `1.775× / 1.533× / 1.722×`。
- fill 与 hit 的 prompt/decode logits 均逐位一致。相对 Chapter 24 BF16
  reference 的 max-abs 为 `0.3125 / 0.25`，两个 argmax 均一致。
- 原始报告保存在本机 `/tmp/qwen3_moe_cuda_expert_cache_8g.json`；仓库冻结
  `benchmark_results/qwen3_moe_cuda_expert_cache/summary.json`，并以源码
  SHA256 绑定 benchmark 与实现。

## Close 回顾

- **完成了什么**：给 30B-A3B offload session 增加可选的 byte-budgeted
  device expert LRU、统计与显式 clear，并用真实 40K-capacity session 验证。
- **验证证据**：tiny 缓存/淘汰与真实结果 contract `54 / 54`，默认全套
  `6,043 / 6,043`；8 GiB 冻结 case 892 entries、零淘汰、重复请求 expert
  I/O 为零，端到端加速 `1.722×`，fill/hit exact 且 reference argmax 一致。
- **没有完成及原因**：没有异步预取、pinned-memory 双缓冲或长自然文本 cache
  trace；这些需要独立的调度与 workload 设计，不和同步 LRU 的最小闭环混写。
- **最重要的认知变化**：在单卡 MoE offload 中，8 GiB active-expert cache
  已能覆盖这个短请求跨 prompt/decode 的完整路由集合，且重复 I/O 确实是主要
  延迟来源；下一步优化应转向 miss path 的重叠和更真实的路由工作集。
- **是否满足 Close 条件**：是。
- **带到下一章的问题**：怎样用长自然文本与多轮 trace 确定合理 cache budget，
  并以 pinned host buffers、异步 H2D 和 layer-ahead prefetch 隐藏剩余 miss
  latency？
