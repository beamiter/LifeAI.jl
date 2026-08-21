# Chapter 44 — Qwen3-VL processor/chat 与 mRoPE decoder prefill

> 所属 Episode：Episode 09 — Qwen3-VL 多模态感知
>
> 状态：Closed
>
> 日期：2026-08-21
>
> 真实资产：`Qwen/Qwen3-VL-2B-Instruct`
>
> ModelScope revision：`ae9985b208c074c10cfbe3a61b5cb7268cdc9c53`
>
> Hugging Face revision：`78448d793a7eb2f7a987a1da76d464384aa1becd`
>
> Oracle：Transformers `4.57.0` / PyTorch `2.7.1+cpu` / torchvision
> `0.22.1+cu126`，eager attention、显式全一 attention mask
>
> 设备：NVIDIA GeForce RTX 4090 D

## Closed：核心问题

Chapter 43 已经证明 LifeAI 能读取同一官方 checkpoint，并把预处理后的 image
patch 送过 vision tower、DeepStack mergers 和 main merger。本章继续回答：能否
从原始 RGB 图像和官方 content-list message 开始，严格复现 fast image processor、
chat template、image placeholder expansion、multimodal RoPE 与完整 28 层 text
decoder prefill，并把 main visual embeddings 和三份 DeepStack features 放到官方
定义的 decoder 位置？

本章关闭的是一次完整、cache-free、batch-1 image-conditioned prefill。它已经从
raw image 走到 vocabulary logits，但没有 KV cache、增量 decode、greedy generation
或停止条件，因而还不能称为 image-to-text generation API。上述闭环留给 Chapter 45。

## Close 条件

1. 原始 PNG/JPEG 或 RGB array 经 decode、RGB conversion、smart resize、CPU UInt8
   bicubic、normalize 和 patchify 后，与官方 `Qwen2VLImageProcessorFast` reference
   对齐；不以 geometry-only 测试代替真实 pixels。
2. Qwen3-VL 专用 tokenizer profile、content-list chat template、image placeholder
   expansion 与 token ids 对官方 frozen assets fail closed。
3. mRoPE 输出明确区分 token id 和零基 T/H/W 坐标，覆盖 visual grid、text continuation、
   `rope_deltas`、padding 输入和异常 placeholder 边界。
4. decoder 实现 QK-Norm、interleaved mRoPE、GQA causal attention、SwiGLU、tied
   projection，以及 main visual replacement 和三层 DeepStack injection。
5. deterministic tiny Float32 decoder 以完整 tensor oracle 验证逐层 raw/post
   hidden、final norm 和 logits；不能只比较 argmax 或 checksum。
6. 官方 2B Float32/BF16 权重在 RTX 4090 D、`CUDA.allowscalar(false)` 下完成
   raw processor、vision 与 decoder 的统一 verifier；Float32 作为 strict gate，
   BF16 只据实记录 CPU oracle 与 CUDA LifeAI 的跨后端边界。
7. 文档明确 cache、generation、video、batch/padding generation 仍未实现。

## 不可变资产与 reference

本章继续使用 Chapter 43 冻结的同一 checkpoint 身份：ModelScope commit
`ae9985b208c074c10cfbe3a61b5cb7268cdc9c53` 用于国内恢复，Hugging Face revision
`78448d793a7eb2f7a987a1da76d464384aa1becd` 绑定上游模型语义。13 个必要文件、
625 个 BF16 tensor、`2,127,532,032` 个参数和 `4,255,064,064` tensor payload
bytes 的资产契约没有变化。

`scripts/export_qwen3_vl_prefill_reference.py` 重新从 raw deterministic image 和
content-list message 出发，不消费 LifeAI 的中间输出。两份仓库外 reference 的
`reference.safetensors` SHA256 为：

- Float32：`d7d3b58cea35cf90806bdd14ade7e453e1b486355b190d094ec95f852f6b60f5`；
- BF16：`711749d9cb0d2c33b34c6fc87a4f9dd06bbf7cc52b589daf01f0210bd58cb5ae`。

对应 `reference.json` SHA256 为 Float32
`1141ef4c503800607d60d8fa23eb795b00d94a7ff879f97ac230abf05681362c`
与 BF16 `7a941b837e7c6df13490386dc42160b9ac6f7d5befcc6bba148cc10e1e81afb5`。
verifier 同时硬编码 metadata/tensor SHA，并核对 80 个 tensor 的 name、shape、
storage dtype 与语义 dtype；不能通过修改 JSON 自报一份新 reference。

exporter 固定 eager attention，并显式传入全一 attention mask。Transformers
4.57 对重复的 temporal mRoPE 坐标存在 packed-sequence 自动探测分支；若把 mask
省略为 `nothing`，会冻结另一套 attention 语义，因此“显式全一”是 oracle
contract，而不是无关紧要的调用细节。

## raw image processor

`qwen3_vl_process_image` 接受 RGB `UInt8` HWC/CHW array，也可从路径 decode
PNG/JPEG；输出 `Qwen3VLProcessedImage`，其中 `pixel_values` 使用 LifeAI
`(1536, patches)` layout，`grid_thw` 使用 host `3×image_count` layout。

官方 fast processor 的 CPU bicubic 并不是普通 Float32 插值后再 round。冻结的
torchvision `0.22.1+cu126` 路径对 UInt8 image 做 horizontal、vertical 两次量化 pass，
每次都按 fixed-point coefficient 舍入并 clamp 回 UInt8。本章按同一顺序实现，避免
在高频图像上出现肉眼不可见但逐 pixel 为 `±1` 的偏差。测试包含：

- identity resize；
- 同时覆盖上采样、下采样和混合方向 resize；
- `17×31 → 192×352` 的非方形高频图像；
- grayscale / alpha decode 后的明确 RGB conversion；
- normalize、patch width、grid 与非 finite 输入的 fail-closed 边界。

真实 oracle 使用确定性 `256×256` RGB image，得到未 merge grid
`(1,16,16)`、256 个 vision patches 和 64 个 merge 后 image tokens。Julia raw
processor 对冻结 `pixel_values` 的 max-abs 为 `0`；这项 exact 结果覆盖了 resize、
normalize 和 patch ordering 的组合，而不只是最终 shape。

## content-list chat、placeholder 与 tokenizer

`load_hf_qwen3_vl_tokenizer` 使用独立的 `:qwen3_vl_generation` profile，严格读取
`tokenizer.json`、`tokenizer_config.json` 与 `generation_config.json`，并拒绝把
普通 Qwen3 或 embedding profile 静默当作 VL tokenizer。官方 chat template 的
SHA256 也是执行 content-list renderer 的准入条件。

`apply_qwen3_vl_chat_template` 覆盖 system/user/assistant/tool、content-list 中的
text/image/video marker、tool calls、tool responses、generation prompt 与可选
vision id。`qwen3_vl_expand_image_placeholders` 再按每张 image 的未 merge grid 和
`2×2` spatial merge，把每个单一 `<|image_pad|>` 展开成正确数量；placeholder 数、
grid 数和连续 run 不一致时立即报错。

本章真实输入为一张 image 加文本 `Describe.`。渲染、展开和 tokenization 后：

| 项目 | 冻结值 |
| --- | ---: |
| sequence length | `76` |
| image placeholder tokens | `64` |
| image grid | `(1,16,16)` |
| spatial merge | `2×2` |

token ids 使用 LifeAI 公共 1-based 约定；checkpoint metadata 和 reference JSON 中
标为 Hugging Face id 的值仍是 0-based，两者不会在 API 内静默混用。

## multimodal RoPE

`qwen3_vl_rope_layout` 返回 `Qwen3VLRopeLayout`：

- `position_ids` shape 为 `(3, sequence, batch)`，三行依次为 temporal、height、
  width；坐标保持零基；
- `visual_mask` 标出 main visual embeddings 替换的 64 个 token rows；
- `attention_mask` 保留 prompt 有效位置；
- `rope_deltas` 保存未来 cache decode 所需的多模态坐标压缩量。

静态 image 的 visual coordinates 按 merge 后 H/W grid 展开，后续 text 从所有
visual axis 的最大坐标加一继续。真实 76-token prompt 的 `rope_delta=-56`；tiny
8-token prompt 的 delta 为 `-2`。这说明未来 decode 的物理 cache position 不能
直接当成 mRoPE coordinate，也是 Chapter 45 必须单独设计 VL cache 的原因。

text decoder 的 head dim 为 `128`，half-head mRoPE section 为 `(24,20,20)`。
height/width frequency lanes 按官方 interleaving 选取，剩余 lanes 使用 temporal
axis，再与自身拼接成 rotate-half 所需的 128 lanes。默认测试同时覆盖纯文本、
单图、left padding，以及 missing vision-end、额外 image-pad、错误 run length、
video 和 batch 边界。

## decoder prefill 与视觉注入

`load_hf_qwen3_vl_text_parameters` 从同一 safetensors 流式读取 language tower，
支持 Float32/BFloat16 并让每个 tensor 完成语义 layout 后立即经 `to_device`
转移。它保留官方 tied embedding/output projection，不构造第二份 LM-head weight。

`hf_qwen3_vl_text_prefill` 执行完整 28 层 cache-free decoder：

1. gather token embeddings；
2. 在 layer 0 前，仅在 `visual_mask` rows 用 main `visual_embeddings` **替换**
   image-token embeddings；
3. 对每层执行 RMSNorm、Q/K/V projection、per-head QK-Norm、interleaved mRoPE、
   16 Q heads / 8 KV heads GQA causal attention、output projection 与 residual；
4. 执行第二个 RMSNorm、SwiGLU MLP 与 residual；
5. decoder layers `0 / 1 / 2` 的 raw block output 分别加上 vision blocks
   `5 / 11 / 17` 对应的 merge 后 DeepStack features，且只改 visual rows；
6. final RMSNorm 后使用 tied embedding 投影 vocabulary logits。

capture 使用官方 0-based decoder layer index，并分别保存 raw block output 与
post-DeepStack layer output。HF `_deepstack_process` 会原地修改 hidden，因此 Python
hook 在捕获时立即 `detach().clone().cpu().contiguous()`；否则所谓 raw oracle 会被
后续 injection 污染。

`hf_qwen3_vl_prefill` 是薄组合层：先运行 Chapter 43 vision tower，再把 main/
DeepStack features 交给 text prefill。它仍然不保留 K/V，`logits_to_keep=1` 可让
普通调用只投影最后一个 prompt position；真实 parity verifier 为完整 logits 明确传
`0`。

## tiny oracle 与错误定位

默认回归使用 hidden `16`、4 layers、2 Q heads / 1 KV head、head dim `8`、
`mrope_section=(2,1,1)` 的确定性 Float32 模型。权重、norm、visual embeddings 和
DeepStack features 都由固定正弦公式生成；fixture 保存完整 input embedding、每层
raw/post hidden、final hidden 和 logits，不以摘要或 argmax 代替 tensor。

这一 oracle 同时钉死 Q/K-Norm 所在轴、RoPE half rotation、mRoPE lane mapping、
GQA head grouping、SwiGLU 舍入、DeepStack 在 block 后而非 block 前注入、以及 tied
projection。最终 tiny 各层误差处于 Float32 rounding 量级，Chapter 44 默认专项
`139 / 139` 通过。

## 真实 RTX 4090 D 结果

`scripts/verify_qwen3_vl_prefill_cuda.jl` 重新执行 raw processor、chat、tokenizer、
mRoPE、vision tower 和 text decoder，并在 `CUDA.allowscalar(false)` 下检查所有
compute tensor 的 device residency。Float32 与 BF16 两次运行的 80 个
stage/semantic gates 全部通过；两次均为 sequence `76`、image tokens `64`，raw
processor max-abs `0`。

| compute | 数值口径 | 全模型 logical parameter bytes | final hidden max-abs | logits max-abs | combined warm prefill |
| --- | --- | ---: | ---: | ---: | ---: |
| Float32 | strict parity | `8,510,128,128` | `0.00094986` | `0.00083363` | `0.703 s` |
| BF16 | cross-backend boundary | `4,255,064,064` | `6.9375` | `7.4375` | `0.751 s` |

logical bytes 是全部 vision + text 参数按 compute dtype 的精确 payload，不包含
CUDA allocator metadata、activation 或 workspace。warm 数字是相同输入的完整
vision+decoder复跑，并在计时边界显式同步；不是只测最后一层，也不是 generation
tokens/s。

Float32 的 final hidden 和 logits 均在 strict gate 内，80 个 stage 全部通过，作为
本章 correctness 结论。BF16 reference 在 CPU Transformers 生成，而 LifeAI 在
CUDA 上执行；vision 与 28 层 decoder 的 BF16 matmul、归约和逐算子舍入会累积，
因此 `6.9375 / 7.4375` 只冻结为真实跨后端边界。Chapter 44 没有执行 token
selection，不能从该数值边界外推 greedy token parity。

复现方式：

```bash
MODEL_DIR=/home/ubuntu/models/modelscope/Qwen/Qwen3-VL-2B-Instruct
REFERENCE_DIR=/tmp/qwen3-vl-prefill-f32
ORACLE_PYTHONPATH=/tmp/lifeai-qwen3vl-oracle/lib/python3.10/site-packages:/tmp/lifeai-qwen3vl-uv-cache/archive-v0/SNUjiORDNkYR55Or

PYTHONPATH="$ORACLE_PYTHONPATH" \
  .venv/bin/python scripts/export_qwen3_vl_prefill_reference.py \
  "$MODEL_DIR" "$REFERENCE_DIR" float32

julia --project=. --startup-file=no \
  scripts/verify_qwen3_vl_prefill_cuda.jl \
  "$MODEL_DIR" "$REFERENCE_DIR" cuda
```

将两个目录名和 exporter dtype 改为 `qwen3-vl-prefill-bf16` / `bfloat16` 可重跑
BF16 boundary。reference 保持在仓库外；完整依赖与本机 torchvision import 注意
事项见 [`notes/local_model_assets.md`](../../local_model_assets.md)。

## 已知边界与决策

- **没有 KV cache 或 decode**：本章每次运行完整 prompt，不能复用历史 K/V。
- **没有 generation**：没有 greedy/sampling、EOS、stop reason、completion text
  或逐 token trace；获得 prefill logits 不等于完成 image-to-text generation。
- **batch/padding generation 未覆盖**：mRoPE builder 对 left padding 有独立测试，
  但 decoder prefill 和真实 GPU oracle 限定 batch 1、全一 attention mask。
- **video 未实现**：chat renderer 能识别 video marker，vision tower 也能表达
  `grid_t>1` 的预处理后输入，但缺少 frame decode/sampling、timestamp mRoPE 和
  端到端 video processor。
- **多图未做真实权重验收**：placeholder/mRoPE 支持多个 image grid，但本章真实
  reference 只含单图。
- **无长上下文结论**：真实序列长度为 76；没有测 262K context、内存上限或
  attention 性能。
- **BF16 不是 strict parity**：真实执行和 finite 输出已证明，但 strict 结论仅来自
  Float32；未来生成还必须逐 step 检查 argmax margin。

## Close 回顾

- **完成了什么**：把 Chapter 43 的预处理后 vision 边界向两端扩展为 raw image、
  Qwen3-VL tokenizer/content-list chat、placeholder expansion、T/H/W mRoPE、main
  visual replacement、DeepStack injection 和完整 28 层 decoder prefill。
- **验证证据**：默认专项 `139 / 139`；官方 2B Float32/BF16 reference SHA256
  分别为 `d7d3b58c…b60f5` / `711749d9…cb5ae`；RTX 4090 D 两种 dtype 的
  80-stage gate 全部通过，raw max-abs `0`，Float32 final/logits max-abs 为
  `0.00094986 / 0.00083363`。
- **没有完成及原因**：cache position 与 mRoPE coordinate 在多模态 prompt 后不再
  相同，不能把普通 Qwen3 cache 机械套用；KV cache、decode、greedy/EOS 和完整
  image-to-text 被留给单独可验证的 Chapter 45。
- **最重要的认知变化**：Qwen3-VL 的 decoder 语义不只是“把 image embedding 塞进
  token 序列”。raw UInt8 resize 舍入、placeholder 数量、三轴坐标、main replacement、
  三次 post-block DeepStack addition 和显式 attention mask 都是可独立出错的契约。
- **是否满足 Close 条件**：是。raw processor、chat/token、mRoPE、tiny 完整 tensor、
  真实 Float32 strict parity、BF16 boundary 和能力边界均已冻结。
- **带到下一章的问题**：Chapter 45 实现请求级 `rope_delta`、dynamic KV cache、
  单 token decode 与真实 image-to-text greedy generation，并以逐 step logits、
  token ids、EOS 和文本冻结 reference 关闭 Episode 09。
