# Week 11 — Qwen3 Dense Family Completion

> 状态：Closed
>
> 开启记录：2026-07-23
>
> 关闭记录：2026-07-23
>
> 依赖基线：[`Week 10 — GPT-2 Architecture, HuggingFace Weights and Text Parity`](week10_gpt2_hf_parity.md) 已 Closed，保持历史内容不变。
>
> 近期主线：把 Week 06—09 在 Qwen3-0.6B 上建立的结构、权重和生成闭环扩展为显式的 Qwen3 dense family contract，同时严格区分“结构/loader 已覆盖”和“真实大权重已实跑”。

## 核心问题

> 现有通用 `GPTModel` 与 Qwen3 HuggingFace adapter 是否真的覆盖
> 0.6B / 1.7B / 4B / 8B / 14B / 32B 六个官方 dense 尺寸，而不是只对
> 0.6B 的一组 shape、tied embedding 和参数量成立？

六个尺寸共享 RMSNorm、SwiGLU、GQA、QK-Norm 和 rotate-half RoPE，但仍有两类必须单独钉死的结构差异：

1. 0.6B、4B、32B 的 query projection width 大于 residual hidden size；不能假设 `num_heads * head_dim == hidden_size`。
2. 0.6B / 1.7B / 4B tied embedding，8B / 14B / 32B 使用独立 LM head；不能只验证 tied 权重路径。

## 已冻结的官方 family contract

以下规格来自 2026-07-23 查询并冻结的 Hugging Face 官方仓库
`config.json`。仓库中的小 fixture 记录完整 revision、config SHA256 和原始字段，
默认测试不联网。

| variant | hidden | MLP | layers | Q / KV | head dim | Q width | tied | 精确参数量 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| Qwen3-0.6B | 1,024 | 3,072 | 28 | 16 / 8 | 128 | 2,048 | 是 | 596,049,920 |
| Qwen3-1.7B | 2,048 | 6,144 | 28 | 16 / 8 | 128 | 2,048 | 是 | 1,720,574,976 |
| Qwen3-4B | 2,560 | 9,728 | 36 | 32 / 8 | 128 | 4,096 | 是 | 4,022,468,096 |
| Qwen3-8B | 4,096 | 12,288 | 36 | 32 / 8 | 128 | 4,096 | 否 | 8,190,735,360 |
| Qwen3-14B | 5,120 | 17,408 | 40 | 40 / 8 | 128 | 5,120 | 否 | 14,768,307,200 |
| Qwen3-32B | 5,120 | 25,600 | 64 | 64 / 8 | 128 | 8,192 | 否 | 32,762,123,264 |

六个冻结 checkpoint 的 `max_position_embeddings` 均为 40,960。官方发布材料中
8B+ 的 128K 能力涉及部署侧 RoPE scaling / YaRN 配置；LifeAI 当前仍对非空
`rope_scaling` fail closed，因此本 Week 不把 128K 扩展写成已支持。

## 实现范围

- 新增 `Qwen3DenseSpec`、`qwen3_dense_specs()`、`qwen3_dense_spec(...)` 和
  `qwen3_dense_parameter_count(...)`，公开六个尺寸的不可变规格、revision 与
  config checksum。
- `load_hf_qwen3_config` 自动识别官方尺寸，并返回 `qwen3_variant` 与
  `source_max_seq_len`；传入 `variant` 时，shape、RoPE 或 norm 语义不完全
  匹配立即失败。
- `load_hf_qwen3_model` / `load_hf_qwen3_bundle` 透传 `variant`，加载结果返回
  已识别的 `Qwen3DenseSpec`；兼容的自定义 dense config 仍返回 `nothing`，
  不被误标成官方尺寸。
- 六个完整官方 topology 都以缩短 RoPE cache 的方式构造，并让
  `Lux.parameterlength(model)` 与独立冻结参数量逐项一致。
- 增加 32B 形态的缩小 fixture：`Q width > hidden`、untied LM head、GQA
  KV storage，并验证 full / dynamic / static cache logits 一致。
- 修复 `Lux.parameterlength(::MultiHeadAttention)` 漏算每层 Q/K-Norm scale
  的旧问题；此前 0.6B 少报 7,168，32B 少报 16,384 个参数。

## 验证分层

| 证据层 | 六个尺寸的状态 |
| --- | --- |
| 官方 config revision / checksum | 六个尺寸均已冻结 |
| config 识别与显式 variant 校验 | 六个尺寸默认离线覆盖 |
| 完整 depth/width topology 与精确参数量 | 六个尺寸默认离线覆盖 |
| tied / untied HF 参数树映射 | 两类路径均覆盖 |
| `Q width > hidden` attention 与 GQA cache | 缩小 32B 形态 full/dynamic/static 覆盖 |
| 真实 checkpoint 逐层/logits/text parity | 仅 0.6B 已完成 |
| 真实大权重内存/吞吐 | 1.7B—32B 未执行 |

最后两行是刻意保留的证据边界：family contract 完成不等于在当前机器上加载并
运行了 32B Float32 权重。

## Close 条件

- 六个官方 dense config 的 immutable revision、config SHA256、结构字段和精确
  参数量进入仓库 fixture，并与公开 API 一致。
- `load_hf_qwen3_config(...; variant=...)` 对六个尺寸正确识别，对错配尺寸和
  RoPE/RMSNorm 语义漂移 fail closed；未知 variant 与 MoE 也必须拒绝。
- 0.6B/4B/32B 的宽 attention，以及 tied/untied LM head 差异均有结构证据。
- 至少一个 untied + `Q width > hidden` 缩小模型通过参数映射、full forward、
  dynamic cache 和 static cache 数值一致性测试。
- QK-Norm 参数计数修复不影响 Week 06—10 的 logits、checkpoint 或默认测试。
- 文档明确说明 1.7B—32B 没有真实权重逐层 parity、native BF16、量化、MoE
  或 128K YaRN 支持。

## 非目标

- 不下载或提交 1.7B—32B 巨型权重，不伪造逐层 logits / text parity。
- 不实现 Qwen3 MoE、FP8/AWQ/GPTQ/GGUF、native BF16、tensor parallel 或
  distributed checkpoint loading。
- 不实现 YaRN / dynamic NTK 等 RoPE scaling，也不把 checkpoint 的 40,960
  原生位置表述成 128K 已验证。
- 不扩展 Qwen3 embedding、reranker、ASR、VL 或 Qwen3.5/3.6 等不同架构。

## 过程记录

### 2026-07-23：Open 与 family fixture

- Week 10 保持 Closed；Week 11 单独承接 Qwen3 dense family 补全。
- 从六个 Hugging Face 官方仓库读取 HEAD immutable revision，只下载各自
  `config.json` 到临时目录核对，并把精简但完整的 reference 写入默认离线
  fixture。
- 首次运行六尺寸参数量测试时发现所有模型都固定少报
  `num_layers * 2 * head_dim`，定位到 QK-Norm scale 由自定义参数初始化加入，
  但 `Lux.parameterlength` 仍只遍历四个 projection child。修复后六个尺寸
  与独立公式全部一致。

### 2026-07-23：验证与 Close

- Week 11 专项 `91 / 91` 通过：family contract 80 项，untied + 宽 attention
  权重/cache 路径 11 项。
- 默认全套 `4284 / 4284` 通过；Week 06—10 的结构、HF adapter、tokenizer、
  sampling 和 GPT-2 回归均未受 QK-Norm 参数统计修复影响。
- 可选 Reactant-XLA CPU 的 untied + 宽 attention prefill/decode smoke
  `4 / 4` 通过。
- 六个官方完整 topology 均实际构造到目标 depth/width，但没有分配 1.7B—32B
  权重；精确参数量由模型自身 `Lux.parameterlength` 与独立公式双向核对。
- `qwen3_moe`、未知 variant、显式 variant/config 错配均 fail closed；未命名
  但结构兼容的自定义 dense config 继续支持，并明确标记为 `nothing`。

## Close 回顾

- **完成了什么**：把“Qwen3 dense 可配置”提升为六个官方尺寸的显式 family
  contract，补齐 untied LM head 和宽 attention 分支的可执行证据，并修复
  QK-Norm 参数量漏计。
- **验证证据**：六份 immutable revision/config checksum fixture、六套完整
  topology 的精确参数量、缩小 32B 形态 full/dynamic/static cache，以及默认
  全套 `4284 / 4284` 和 Week 11 XLA smoke `4 / 4`。
- **没有完成及原因**：没有下载 1.7B—32B 权重或生成逐层 reference；当前
  Float32 loader 的内存成本随尺寸增长明显，这属于后续低精度/分片流式加载和
  硬件验证，不应由 shape 测试冒充。
- **最重要的认知变化**：family 支持的风险不只在“更多层”；Q projection
  width 与 residual width 解耦、tied/untied head 分界，以及自定义参数未进入
  框架默认 introspection，都会产生只在其他尺寸暴露的问题。
- **是否满足 Close 条件**：是。所有 Close 条件已由默认 fixture、专项测试、
  全套回归和明确的真实权重边界覆盖。
