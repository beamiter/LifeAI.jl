# Chapter 25 — Qwen3-30B-A3B GPU resident/offload session

> 所属 Episode：Episode 06 — Qwen3 MoE 与模型架构扩展
>
> 状态：Closed

## Open：核心问题

怎样把 Chapter 24 已验证的 BF16 active-expert streamer、静态 KV cache 和
CUDA grouped tensor-core dispatch 组合成 RTX 4090 D 上可运行的官方
Qwen3-30B-A3B session，而不把 61 GB BF16 权重树假装成能常驻 24 GB 显存？

## 预期结果

本章 Close 时，应当可以展示或验证：

1. attention/router/norm/LM head 与 40,960-token KV cache 常驻 GPU，expert
   权重按层、按实际路由从 safetensors 读取和上传；
2. 官方 30B-A3B 的真实 prompt 与 cached decode 在 BF16 容差下保持
   Transformers reference argmax；
3. grouped WMMA 与 scalar production dispatch 有端到端对照，明确小
   route batch 和宽 prefill 各自应走哪条路径。

## 计划与结果

| 工作项 | 所属主线 | 交付物 | 验收方式 | 状态 |
| --- | --- | --- | --- | --- |
| resident/offload 容量规划 | 模型 / 工程 | `qwen3_moe_offload_plan` | 官方 config 逐 byte 门禁 | 已完成 |
| 活跃 expert 局部映射 | 模型 | global→local route remap | 形状、编号、越界测试 | 已完成 |
| GPU streamed session | 模型 / 工程 | `HFQwen3MoEOffloadSession` | 真实 30B prefill/decode | 已完成 |
| 40K KV 容量 | 工程 | BF16 static layer caches | 40,960 容量实际分配 | 已完成 |
| grouped 端到端对照 | 学习 / 工程 | benchmark 与冻结摘要 | 2/32-token steady 对照 | 已完成 |

## Close 条件

- 真实 30B session 不构造全 expert 参数树，单层只上传当前路由覆盖的
  experts；
- 40,960-token BF16 KV cache 与常驻非 expert 参数在 RTX 4090 D 上实际
  分配成功；
- 2-token prompt + 1-token cached decode 与上一章 Transformers BF16
  reference 的 argmax 一致；
- 至少一个宽 prefill case 证明 grouped WMMA 的端到端收益，并保留小
  batch 的反例。

## 学习重点

- **生命周期比 dtype 更关键**：30.5B 参数即使全部 BF16 仍约 61 GB；可部署
  性来自非 expert 常驻、expert 稀疏读取和单层上传，而不是只把 Float32 改成
  BF16。
- **路由编号必须重映射**：磁盘上的 global expert id 不能直接索引只包含
  active experts 的局部三维参数张量；session 在宿主读取极小的 top-k route
  table，排序 active ids 后构造 local ids，再把小表传回设备。
- **kernel 加速不等于 session 加速**：真实端到端包含 safetensors 读取、host
  拼接、PCIe 上传、attention、router、expert 和 LM head。必须在同一已加载
  session 中区分 cold 与 steady。

## 风险与取舍

- 当前 active route table 会从 GPU 回读到宿主，以决定要读取哪些 checkpoint
  tensors；这是 offload 的物理 I/O 决策，不是把 expert 计算回退到 CPU。
- LM head 与 attention/router 常驻约 `2.291 GiB`，40K BF16 KV 为
  `3.75 GiB`。最坏单层 128 experts 为 `1.125 GiB`，三者硬下限合计
  `7.166 GiB`，未计 allocator slack、attention scores 和 dispatch workspace。
- session 支持填充到 40,960 positions，但本章只实际分配全容量 cache，并
  实跑 2/32-token prompt；没有声称完成 40K-token full-window prefill 或长序列
  生成质量。

## 实验与过程记录

### 2026-08-12：真实 30B-A3B 40K-capacity session

- 宿主机为 RTX 4090 D，24,564 MiB，compute capability 8.9。加载官方 revision
  `ad44e777…85d39` 的 48 层 attention/router、final norm 与 LM head，实际常驻
  参数 `2,459,856,896` bytes；同时预分配 48 层、40,960-token BF16 KV
  `4,026,531,840` bytes。常驻加载约 `15.82 s`。
- `load_hf_qwen3_moe_offload_session` 不加载 embedding 全矩阵；prompt token
  rows 直接从 safetensors gather。每层先在 GPU 算 attention/router，再回读
  `top_k × tokens` route ids，只读取 active expert 的 gate/up/down，映射成局部
  expert tensor 后上传。
- 2-token grouped pass 每层激活 14—16 experts，prefill/decode 分别从磁盘读取
  `6,936,330,240 / 3,623,878,656` bytes。steady 为 `12.15 / 5.58 s`；相对
  Transformers BF16 reference 的 logits max-abs 为 `0.421875 / 0.2890625`，
  两个 argmax 均一致，重复请求输出逐位一致。
- 同 case 的 scalar production steady 为 `11.39 / 5.02 s`，因此 grouped 只有
  `0.938× / 0.899×`，小 route batch 明确不应走 WMMA padding 路径。

### 2026-08-12：32-token grouped 端到端边界

- 使用 `repeat([1, 9707], 16)` 构造确定性 32-token prompt；每层只激活
  21—38 experts，平均 `29.375`，远少于 128。grouped 每 pass 读取约
  `13.31 GB` expert payload，steady prefill/decode 为 `23.82 / 4.63 s`。
- 相同 token、相同 40K cache 的 scalar production 为 `26.02 / 6.18 s`；
  grouped 端到端加速 `1.092× / 1.335×`。两路最终 prompt/decode argmax
  分别同为 1-based `2 / 359`。
- 结论是沿用 32-token 宽 prefill 边界：小 prompt/decode 走 scalar，达到已
  验证宽度后才启用 grouped BF16 WMMA。原始运行报告保留在本机 `/tmp`，仓库
  冻结 `benchmark_results/qwen3_moe_cuda_offload/summary.json`。

## Close 回顾

- **完成了什么**：完成官方 30B-A3B BF16 GPU resident/offload session，连接
  真实 safetensors active-expert streaming、40K static KV 与 grouped WMMA。
- **验证证据**：40K cache 实际分配；真实 2-token prompt/decode argmax 与
  Transformers 一致；32-token grouped steady 相对 scalar 为
  `1.092× / 1.335×`；默认全套 `5,989 / 5,989`、Chapter 25 contract
  `41 / 41`、CUDA 专项 `38 / 38`。
- **没有完成及原因**：没有填满 40K window，也没有验证长序列生成质量；当前
  逐请求仍重复读取/upload active experts，I/O 是主要延迟，不属于本章最小
  session 闭环。
- **最重要的认知变化**：MoE 在单卡上的真实优势不是“30B 模型变成 3B 常驻”，
  而是参数容量与每 token 计算量可以分开管理；能否部署取决于 expert 生命周期
  和路由局部性，能否更快则取决于 I/O cache 与 route batch 宽度。
- **是否满足 Close 条件**：是。
- **带到下一章的问题**：怎样加入跨请求/跨层 active-expert host/device cache、
  异步预取与 pinned-memory 双缓冲，并用更长自然文本量化命中率和 I/O 隐藏率？
