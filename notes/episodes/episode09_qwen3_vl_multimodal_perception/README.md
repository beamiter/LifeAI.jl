# Episode 09 — Qwen3-VL 多模态感知

> 状态：Open
>
> 收录章节：Chapter 43–45

## 本卷主题

Episode 06 已完成 Qwen3 MoE 主线，本卷回到 Qwen3 系列并补上图像感知。
推进顺序仍以官方不可变资产和独立 Transformers oracle 为准：先关闭
Qwen3-VL 的 checkpoint、图像 patch/grid 边界和 vision tower，再把视觉 token
接入 chat template、multimodal RoPE、decoder prefill 与 KV cache，最后验收真实
image-to-text generation。单独跑通 vision encoder 不等于多模态生成已经可用。

## 章节目录

1. [`Chapter 43 — Qwen3-VL 2B vision architecture 与真实权重 parity`](chapter43_qwen3_vl_vision_architecture.md)（Closed）
2. Chapter 44 — processor/chat 与 mRoPE decoder prefill（计划中）
3. Chapter 45 — multimodal cache 与完整 image-to-text generation（计划中）

## 预期能力变化

- **模型基本组件**：严格解析 Qwen3-VL text/vision config，复现 3D patch
  embedding、learned position interpolation、vision RoPE、24 层 vision
  Transformer、三个 DeepStack 分支和主 merger。
- **训练与推理**：Float32/BF16 vision 参数可流式读取到 CPU 或 accelerator；
  先以 Float32 跨框架逐 stage parity 关闭 correctness，再单独记录 BF16
  跨后端数值边界。
- **多模态感知**：建立 resize geometry、image grid、token count、patchify 与
  已预处理 `pixel_values` 的明确边界；后续再接 raw image、content-list chat 和
  decoder image-token layout。
- **工程与测试**：checkpoint 必须同时绑定 ModelScope 国内下载 commit、
  Hugging Face immutable revision、逐文件 checksum、完整 tensor name/shape/dtype
  契约和 Transformers 4.57.0 oracle。

## 当前进展

Chapter 43 已关闭独立 vision tower：官方 2B checkpoint 的 625 个 BF16 tensor、
`2,127,532,032` 个参数和 `4,255,064,064` payload bytes 已完整校验；RTX 4090 D
Float32 `visual_embeddings` 对 Transformers 4.57.0 的 max-abs / mean-abs /
relative L2 / cosine 为
`6.771088e-5 / 2.2492045e-6 / 5.461097e-6 / 1.0`。
BF16 同组指标为
`0.890625 / 0.02120505 / 0.05465893 / 0.9985048`；它只作为
CPU oracle 与 CUDA LifeAI 的跨后端归约/舍入边界，不升级为严格 parity。

## Episode Close 条件

- 官方 raw image processor 和 content-list chat template 有独立 reference，
  image placeholder 数量、grid 与 token layout 一致。
- Qwen3-VL decoder 的 multimodal RoPE、DeepStack visual feature injection 和
  image-conditioned prefill 完成逐层/logits parity。
- dynamic/static KV cache 至少一条路径完成真实 image-to-text greedy generation，
  token ids、停止条件和文本与冻结 reference 一致。
- 默认回归、真实 checkpoint 校验和 accelerator 验收全部通过；未覆盖的 video、
  多图、长上下文或性能边界必须显式记录。

## 本卷回顾

- **形成的能力闭环**：目前只完成“官方 checkpoint → 预处理后 patch/grid →
  vision tower → `patch_hidden_state` → merge 后的 DeepStack features /
  `visual_embeddings`”，Episode 仍为 Open。
- **关键实验与指标**：Chapter 43 默认 `178 passed + 1 explicit skip`，设置
  真实模型目录后 `190 / 190`；Float32 GPU strict parity 与 BF16 GPU full
  forward 均在 RTX 4090 D 完成。
- **失败、偏差与未解决问题**：BF16 跨框架 `visual_embeddings` cosine 低于
  Float32，说明
  24 层归约与舍入误差会累积；当前不据此声称 BF16 strict parity。
- **重要架构决策**：国内 ModelScope 用于快速恢复，身份同时绑定 ModelScope
  commit 与 Hugging Face immutable revision；raw decode/resize 和 decoder 集成
  不偷渡进 vision-only API。
- **最重要的认知变化**：Qwen3-VL 的感知闭环不止是 ViT 输出；processor 的
  merge-order、DeepStack 中间层特征和 decoder mRoPE/injection 都是模型语义。
- **进入下一章的问题**：Chapter 44 如何把 raw image/content-list chat、视觉
  placeholder 和 mRoPE decoder prefill 接成可逐层验证的单次多模态前向？
