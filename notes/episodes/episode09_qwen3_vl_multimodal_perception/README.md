# Episode 09 — Qwen3-VL 多模态感知

> 状态：Closed
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
3. [`Chapter 45 — Qwen3-VL dynamic KV cache 与 image-to-text generation`](chapter45_qwen3_vl_dynamic_decode.md)（Closed）

## 预期能力变化

- **模型基本组件**：严格解析 Qwen3-VL text/vision config，复现 3D patch
  embedding、learned position interpolation、vision RoPE、24 层 vision
  Transformer、三个 DeepStack 分支和主 merger。
- **训练与推理**：Float32/BF16 vision 与 text 参数可流式读取到 CPU 或
  accelerator；先以 Float32 跨框架逐 stage parity 关闭 correctness，再单独记录
  BF16 跨后端数值边界。
- **多模态感知**：从 raw image decode、官方 fast resize/normalize/patchify，
  经 content-list chat、placeholder expansion、T/H/W mRoPE、main visual
  replacement 和三层 DeepStack injection，接到 cache-free/cached decoder
  prefill、dynamic KV、single-token decode 与 greedy image-to-text generation。
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

Chapter 45 把物理 cache position 与请求级 `rope_delta` 分开，缓存每层 mRoPE 后的
K 和 raw V，layout 为 `(head_dim, kv_heads, tokens, batch)`，不保存 GQA repeat。
tiny `DynamicCache` fixture SHA256 为
`7b20111e43aa9efd2aae0be3f4a740ab1fdeaff9bd0a6ffd0bfef49adfeeffd8`；
真实 Float32 `reference.json` / `reference.safetensors` SHA256 为
`569fe3666b65ee2f497327e9ce9931f81652d5bdc32d44dfb9fb774435caccfc` /
`a98812e25efb44c02ab9c06e974ab718724f35f2f1c686e4bdc395d856c03e81`。

真实 prompt physical length `76`、`rope_delta=-56`，三次 decode 的 mRoPE T/H/W
坐标为 `20 / 21 / 22`。RTX 4090 D 上 prefill 与三次 decode 的完整 logits
max-abs 分别为 `3.86238e-5 / 4.00543e-5 / 2.95639e-5 / 2.90871e-5`，四个
argmax 全部一致。HF 0-based ids `[1986,2168,374,264]` 对应 LifeAI 1-based ids
`[1987,2169,375,265]`，completion 为 `"This image is a"`。最终 cache length
`79`、Float32 K/V logical bytes `18,120,704`。Chapter 45 专项 `331 / 331`。
最终隔离 `Pkg.test()` 为 `8,994 passed`、`1 intentional broken`、零 failure/error；
唯一 broken 是未提供真实 checkpoint 时的 opt-in 门禁。

## Episode Close 条件

- 官方 raw image processor 和 content-list chat template 有独立 reference，
  image placeholder 数量、grid 与 token layout 一致。（Chapter 44 已满足）
- Qwen3-VL decoder 的 multimodal RoPE、DeepStack visual feature injection 和
  image-conditioned prefill 完成逐层/logits parity。（Chapter 44 已满足）
- dynamic/static KV cache 至少一条路径完成真实 image-to-text greedy generation，
  token ids、停止条件和文本与冻结 reference 一致。（Chapter 45 dynamic 路径已满足）
- 默认回归、真实 checkpoint 校验和 accelerator 验收全部通过；未覆盖的 video、
  多图、长上下文或性能边界必须显式记录。（Chapter 43—45 专项与真实验收已满足）

## 本卷回顾

- **形成的能力闭环**：已完成“raw image/content-list message → exact
  pixels/grid/tokens/mRoPE → vision tower → main/DeepStack features → cached
  28-layer decoder prefill → dynamic KV single-token decode → greedy ids/text”；
  Episode 09 Closed。
- **关键实验与指标**：Chapter 43 的 vision-only 结果与 Chapter 44 的
  Float32/BF16 prefill 证据继续有效；Chapter 45 专项 `331 / 331`，真实 RTX
  4090 D Float32 四阶段 logits max-abs 均小于 `4.01e-5`，生成
  `"This image is a"`。
- **失败、偏差与未解决问题**：dynamic cache 每 token/每层通过 `cat` 重分配并复制
  prefix，累计 copy 为 `O(T²)`，不是长上下文部署方案；默认 BF16 CUDA 高层 smoke
  虽生成相同 ids/text，但尚无 BF16 strict oracle。static cache、video、多图、
  batch/padding、sampling 和长上下文仍未验收。
- **重要架构决策**：国内 ModelScope 用于快速恢复，身份同时绑定 ModelScope
  commit 与 Hugging Face immutable revision；Chapter 43 的 vision-only API 与
  Chapter 44 的 raw processor/cache-free prefill 与 Chapter 45 的 request cache/
  generation 边界保持可独立验证。
- **最重要的认知变化**：Qwen3-VL decode 同时有 physical cache timeline 与 mRoPE
  coordinate timeline；下一坐标必须是 `cache.position + rope_delta`。第一个输出
  token 来自 prefill logits，生成 N 个 token 只需 N−1 次 decode。
- **带往后续的问题**：如何把 correctness-first dynamic `cat` 换成 static/preallocated
  cache，并为 BF16、padding/batch、多图/video、sampling 与长上下文分别建立独立
  oracle？这些扩展不影响本 Episode 已关闭的单图 Float32 greedy 最小闭环。
