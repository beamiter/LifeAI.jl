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
2. [`Chapter 44 — Qwen3-VL processor/chat 与 mRoPE decoder prefill`](chapter44_qwen3_vl_multimodal_prefill.md)（Closed）
3. Chapter 45 — multimodal cache 与完整 image-to-text generation（计划中）

## 预期能力变化

- **模型基本组件**：严格解析 Qwen3-VL text/vision config，复现 3D patch
  embedding、learned position interpolation、vision RoPE、24 层 vision
  Transformer、三个 DeepStack 分支和主 merger。
- **训练与推理**：Float32/BF16 vision 与 text 参数可流式读取到 CPU 或
  accelerator；先以 Float32 跨框架逐 stage parity 关闭 correctness，再单独记录
  BF16 跨后端数值边界。
- **多模态感知**：从 raw image decode、官方 fast resize/normalize/patchify，
  经 content-list chat、placeholder expansion、T/H/W mRoPE、main visual
  replacement 和三层 DeepStack injection，接到 cache-free decoder prefill；
  Chapter 45 再加入请求级 cache/decode 与 image-to-text generation。
- **工程与测试**：checkpoint 必须同时绑定 ModelScope 国内下载 commit、
  Hugging Face immutable revision、逐文件 checksum、完整 tensor name/shape/dtype
  契约和 Transformers 4.57.0 oracle。

## 当前进展

Chapter 43 已关闭独立 vision tower；Chapter 44 又把同一官方 2B checkpoint
接成 raw image、content-list chat、64-token image placeholder、三轴 mRoPE、
main visual replacement、DeepStack injection 和完整 28-layer decoder prefill。
冻结的 `256×256` image 形成 sequence `76` / image tokens `64`，raw processor
max-abs `0`。Float32/BF16 reference SHA256 分别为
`d7d3b58cea35cf90806bdd14ade7e453e1b486355b190d094ec95f852f6b60f5` 和
`711749d9cb0d2c33b34c6fc87a4f9dd06bbf7cc52b589daf01f0210bd58cb5ae`。

RTX 4090 D 上两种 dtype 的 80-stage gate 全部通过。Float32 final hidden / logits
max-abs 为 `0.00094986 / 0.00083363`，combined warm prefill 为 `0.703 s`；BF16
对应 `6.9375 / 7.4375` 与 `0.751 s`，仍只作为 CPU oracle 对 CUDA LifeAI 的
跨后端边界。全模型 logical parameter bytes 分别为 Float32 `8,510,128,128`、
BF16 `4,255,064,064`。

## Episode Close 条件

- 官方 raw image processor 和 content-list chat template 有独立 reference，
  image placeholder 数量、grid 与 token layout 一致。（Chapter 44 已满足）
- Qwen3-VL decoder 的 multimodal RoPE、DeepStack visual feature injection 和
  image-conditioned prefill 完成逐层/logits parity。（Chapter 44 已满足）
- dynamic/static KV cache 至少一条路径完成真实 image-to-text greedy generation，
  token ids、停止条件和文本与冻结 reference 一致。
- 默认回归、真实 checkpoint 校验和 accelerator 验收全部通过；未覆盖的 video、
  多图、长上下文或性能边界必须显式记录。

## 本卷回顾

- **形成的能力闭环**：当前已完成“raw image/content-list message → exact
  pixels/grid/tokens/mRoPE → vision tower → main/DeepStack features → 28-layer
  decoder prefill → logits”；Episode 因尚无 cache/generation 仍为 Open。
- **关键实验与指标**：Chapter 43 的 vision-only结果继续有效；Chapter 44 默认
  专项 `139 / 139`，真实 RTX 4090 D Float32/BF16 的 80-stage gate全部通过，
  raw processor max-abs `0`。
- **失败、偏差与未解决问题**：BF16 final/logits max-abs 为
  `6.9375 / 7.4375`，不能据此声称 BF16 strict 或 greedy parity；KV cache、
  decode、EOS 和输出文本尚未实现。
- **重要架构决策**：国内 ModelScope 用于快速恢复，身份同时绑定 ModelScope
  commit 与 Hugging Face immutable revision；Chapter 43 的 vision-only API 与
  Chapter 44 的 raw processor/decoder prefill边界保持可独立验证。
- **最重要的认知变化**：Qwen3-VL 的感知闭环不止是 ViT 输出；processor 的
  UInt8 resize舍入、merge-order、placeholder数量、DeepStack 中间层特征和 decoder
  mRoPE/injection 都是模型语义。
- **进入下一章的问题**：Chapter 45 如何把物理 cache position 与 prompt-specific
  `rope_delta` 分离，完成 dynamic KV、单 token decode 和真实 greedy
  image-to-text generation？video、batch/padding generation与长上下文仍不在
  Chapter 44 结论内。
