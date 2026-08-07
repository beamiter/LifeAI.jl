# Chapter 24 — Qwen3 MoE 架构支持

> 所属 Episode：Episode 06 — Qwen3 MoE 与模型架构扩展
>
> 状态：Open

## Open：核心问题

LifeAI.jl 能否严格复现原始 Qwen3 MoE 的 top-k routing、expert SwiGLU、HuggingFace 权重布局与 cached decode，并把 correctness reference 推进为不计算未选 expert 的可部署实现？

目标契约以 `Qwen/Qwen3-30B-A3B` 的官方 `qwen3_moe` 配置和 Transformers 4.51.0 实现为准：128 experts、每 token 选择 8 experts、Float32 router softmax、选中概率重新归一化。这个原始架构没有 shared expert；其他后续 Qwen MoE 变体不能未经验证复用本章结论。

## 预期结果

本章 Close 时，应当可以展示或验证：

1. tiny Qwen3 MoE 的 router、expert mixture、full logits 与 HuggingFace reference 在显式容差内对齐。
2. 官方 `config.json` 与逐 expert safetensors 名称可以严格加载，missing、unexpected、shape 和不支持的 dense/sparse 混合 schedule 均 fail closed。
3. prefill、dynamic cache 与 static cache decode 一致，并形成 CUDA/XLA 可编译的 sparse dispatch 路径。
4. 至少一个官方真实 checkpoint 的资产 provenance、逐层 parity、内存与性能结果被冻结。

## 计划

| 工作项 | 所属主线 | 交付物 | 验收方式 | 状态 |
| --- | --- | --- | --- | --- |
| top-k router 与 expert SwiGLU correctness oracle | 模型 | `Qwen3SparseMoE`、Float32 routing | 独立手写 reference 对拍 | 已完成 |
| GPT decoder 与 KV cache 集成 | 模型 / 推理 | `GPTModel(..., mlp_type=:qwen3_moe)` | full/dynamic/static logits 一致 | 已完成 |
| `qwen3_moe` config 与权重映射 | 工程 | strict config parser、逐 expert stack loader | tiny HF 命名 fixture、错误输入 fail closed | 已完成 |
| 独立 Transformers tiny reference | 模型 | hidden/router/logits fixture | 跨框架逐层 parity | 已完成 |
| CPU sparse token dispatch | 高效推理 | 只执行被选 token-expert pair 的 gather/compute/combine | 与 dense oracle 对齐，inactive expert 不执行 | 已完成 |
| CUDA/XLA sparse dispatch | 高效推理 | 无 host routing fallback 的设备路径 | correctness 不变，编译与性能证据 | 计划中 |
| 官方真实权重验证 | 模型 / 工程 | streamed loader、资产清单、parity 报告 | checksum、逐层/logits/cache 与资源实测 | 计划中 |

## Close 条件

只有以下条件满足后才能关闭本章：

- router top-k、`norm_topk_prob` 和 tie-break 契约有独立 reference。
- tiny checkpoint 的 config、全部 expert 权重、full logits 和 cached decode 与 Transformers 对齐。
- CPU、CUDA、XLA 至少各有一条可验证执行路径；若某后端不支持，必须给出可复现边界而不是静默回退。
- 生产路径不会对每个 token 计算全部 128 experts。
- 真实 checkpoint 采用流式/按需 expert 加载，不要求把 30B Float32 state dict 全量驻留内存。
- Dense Qwen3 和 GPT-2 既有测试无回归。

## 学习重点

- 要理解的概念：router softmax、top-k gating、选中概率重归一化、token-to-expert dispatch、expert 容量与负载均衡。
- 要亲手实现的关键组件：router、expert 参数布局、dispatch/combine、MoE checkpoint loader 与 cache 集成。
- 要验证的假设：先建立全 expert correctness oracle，能够降低 sparse accelerator 实现的数值调试风险。

## 风险与取舍

- CPU 默认路径已经按 expert gather 被选 token；全 expert masked 版本保留为 `qwen3_dense_expert_reference` 数值 oracle。CUDA/XLA 仍未实现，不能从 CPU 结果推断 accelerator 已支持。
- 初始 loader 会把 expert 权重 stack 成三维数组，只适合 tiny fixture；真实 30B/235B 必须使用 streamed reader 和按需生命周期。
- host `partialsortperm` 不是 CUDA/XLA 路由实现，不能把现有 Float32 CPU 通过写成 accelerator 已支持。
- Qwen3 后续系列可能使用融合 expert tensor、shared expert、不同 attention 或 hybrid layer；必须按各自配置重新建立契约。

## 实验与过程记录

### 2026-08-07：最小 MoE 架构切片

- 新增 `Qwen3SparseMoE`：router 参数 `(experts, hidden)`，expert gate/up/down 按第三维 stack。
- router 在 Float32 上 softmax，保留固定 top-k，并支持 `norm_topk_prob`。
- `GPTModel` / `TransformerBlock` 增加 MoE 构造配置；现有 full、dynamic cache、static cache 路径无需专用分支。
- 新增 strict `load_hf_qwen3_moe_config` 与原始逐 expert 权重名映射。
- 四个按测试内容命名的专项共 `43 / 43` 通过：router 13、expert mixture 6、config/weight mapping 18、cached decode 6。
- Dense 回归：Qwen3 HF weight loading `54 / 54`、dense family `91 / 91` 通过。

### 2026-08-07：独立 Transformers tiny parity

- 新增 `scripts/export_qwen3_moe_tiny_reference.py`，使用项目 `.venv` 中的 Python 3.10.12、PyTorch 2.7.1+cpu、Transformers 4.51.3，从官方 `Qwen3MoeForCausalLM` 独立生成 2-layer / 4-expert / top-2 Float32 checkpoint。
- config、generation config、模型权重、reference tensors 和生成脚本的 SHA256 全部冻结在 `reference.json`；重新生成命令：

  ```bash
  .venv/bin/python scripts/export_qwen3_moe_tiny_reference.py \
    test/fixtures/qwen3_moe_tiny_parity
  ```

- `hf_qwen3_moe_forward_trace` 显式观测 post-attention RMSNorm 后、expert dispatch 前的 router logits；没有用 LifeAI 结果生成 reference。
- 两层共 8 个 token-router 决策的 top-2 expert 顺序和归一化权重全部一致。最大绝对误差：router `5.96e-8`、block `7.45e-9`、final hidden `3.58e-7`、full logits `8.94e-8`、prompt/decode logits `5.96e-8`；embedding 逐位相同。
- Transformers parity 专项 `44 / 44`，Chapter 24 累计 `87 / 87`；默认全套 `5,750 / 5,750` 通过。

### 2026-08-07：CPU sparse token dispatch

- `qwen3_sparse_expert_dispatch` 按 expert gather 非零 routing token，完成 SwiGLU 后按原 token index combine；`Qwen3SparseMoE` 默认前向不再计算未选 token-expert pair。
- `Qwen3MoEDispatchStats` 显式记录 active experts、逐 expert token 数、实际 routed pairs 和 dense oracle pairs，避免仅凭实现名称宣称稀疏。
- inactive expert 测试把三个未选 expert 的 gate/up/down 全部写成 `NaN`：sparse 输出保持全 finite，而 dense oracle 被 `NaN × 0` 污染，证明未选 expert 没有进入矩阵乘法。
- 128 experts / top-8 / 64 tokens / `d_model=128` / expert hidden 64、BLAS 单线程的 7 次 steady CPU 对照：执行 pair 从 `8,192` 降到 `512`（16×），sparse median `1.416 ms`，dense median `6.637 ms`，加速 `4.69×`，max-abs `3.58e-7`。原始结果位于 `benchmark_results/qwen3_moe_sparse_dispatch/cpu_reference.json`。
- sparse dispatch 新增 `16 / 16`，Transformers parity 仍为 `44 / 44`；Chapter 24 累计 `103 / 103`，默认全套 `5,766 / 5,766`。

## Close 回顾

- **完成了什么**：CPU Float32 correctness、独立 Transformers tiny parity 和真实 CPU sparse dispatch；本章仍 Open。
- **验证证据**：MoE 专项 `103 / 103`，默认全套 `5,766 / 5,766`，缩小 128-expert CPU 基准 `4.69×`。
- **没有完成及原因**：CUDA/XLA sparse dispatch 和真实大权重尚未执行。
- **最重要的认知变化**：原始 Qwen3 MoE 没有 shared expert；不能把后续 Qwen MoE 变体的结构预设到本章。
- **是否满足 Close 条件**：否。
- **带到下一阶段的问题**：怎样在不产生动态 host 控制流的前提下表达 XLA top-k dispatch，并控制 128 experts 的编译图规模？
