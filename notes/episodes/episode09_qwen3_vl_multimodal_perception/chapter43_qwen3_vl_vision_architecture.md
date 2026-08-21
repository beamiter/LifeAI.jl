# Chapter 43 — Qwen3-VL 2B vision architecture 与真实权重 parity

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
> Oracle：Transformers `4.57.0` / PyTorch `2.7.1+cpu`，eager attention
>
> 设备：NVIDIA GeForce RTX 4090 D

## Closed：核心问题

在不先实现整套 image-to-text generation 的前提下，LifeAI.jl 能否以官方
Qwen3-VL-2B-Instruct 为不可变契约，严格校验 checkpoint 与图像 patch/grid
布局，复现 vision tower、DeepStack 和 merger，并用真实权重逐 stage 对齐独立
Transformers 4.57.0 reference？

本章只关闭视觉编码器这一层。text config 会作为未来 decoder 集成的 shape
契约被严格解析，但 text weights 不进入本章前向；raw image、chat template、
decoder mRoPE/visual injection、generation 和 video 都不属于本章能力。

## 目标契约

| 部分 | 官方 Qwen3-VL-2B-Instruct 契约 |
| --- | --- |
| text | vocabulary `151,936`，hidden `2,048`，28 layers，16 Q heads / 8 KV heads，head dim `128`，SwiGLU，tied embedding |
| text position | native context `262,144`，RoPE theta `5,000,000`，interleaved mRoPE section `(24, 20, 20)` |
| vision | hidden `1,024`，24 layers，16 heads，MLP `4,096`，GELU-tanh |
| patch | channels `3`，temporal patch `2`，spatial patch `16×16`，merge `2×2` |
| position | `2,304` learned positions，经目标 H/W grid 双线性插值；attention 使用 H/W vision RoPE |
| output | `patch_hidden_state` 为未 merge 的 `1,024×N`；main merger 产生供 decoder 消费的 `visual_embeddings`，shape `2,048×N/4`；block `5 / 11 / 17` 各产生一份同样 merge 后的 DeepStack feature |
| special ids | vision start/end `151652 / 151653`，image/video `151655 / 151656` |

## Close 条件

1. ModelScope 国内 checkpoint 同时绑定精确 ModelScope commit 和 Hugging Face
   immutable revision；全部必要文件逐 byte 校验。
2. strict nested config、processor config、完整 safetensors name/shape/BF16
   dtype、参数量与 payload bytes 全部 fail closed。
3. resize geometry、image grid/token count、patchify merge-order 和 temporal
   broadcast 有手算或官方 processor reference。
4. tiny Float32/BF16 完整 vision forward 覆盖 patch embedding、position、
   per-temporal-frame packed attention、24 blocks、DeepStack 和 main merger。
5. 官方真实权重在 RTX 4090 D 上完成 Float32 逐 stage strict parity；BF16
   必须实跑并据实记录跨后端误差，不以宽松容差伪装为严格 parity。
6. 文档明确区分 vision-only 能力与尚未完成的 raw processor、decoder 和
   generation，既有默认回归不被破坏。

## 不可变资产与 checkpoint 验证

国内下载使用 ModelScope 官方仓库，当前 commit 为
`ae9985b208c074c10cfbe3a61b5cb7268cdc9c53`；模型语义同时绑定 Hugging Face
revision `78448d793a7eb2f7a987a1da76d464384aa1becd`。ModelScope 的移动分支名
不能替代这两个 provenance 字段。

完整验证器对 13 个必要文件顺序计算 SHA256，再检查 safetensors header 中的
每个 tensor name、shape、dtype 和 byte range。真实结果为：

| 项目 | 冻结结果 |
| --- | ---: |
| `model.safetensors` 文件大小 | `4,255,140,312` bytes |
| `model.safetensors` SHA256 | `7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0` |
| tensor 数量 / dtype | `625` / 全部 BF16 |
| 参数量 | `2,127,532,032` |
| tensor payload | `4,255,064,064` bytes |
| 其中 vision 参数 | `406,957,056`，BF16 payload `813,914,112` bytes |

`qwen3_vl_expected_tensor_shapes` 从冻结 text/vision config 独立推导全部 625
个名称与 shape；验证器不会只相信 safetensors metadata 中自报的总量。config
还严格检查 mRoPE、DeepStack indexes、special ids、tied head 和 text/vision
width 接口，未知字段或 shape 漂移直接拒绝。

## processor 与输入边界

官方 `preprocessor_config.json` SHA256 为
`27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516`。
冻结参数是 min/max pixels `65,536 / 16,777,216`、patch `16`、temporal patch
`2`、merge `2`、RGB mean/std 均为 `0.5`。

本章提供三层确定性边界：

1. `qwen3_vl_smart_resize` 只计算满足 factor/pixel budget 的目标 H/W，**不执行**
   像素插值；
2. `qwen3_vl_image_grid` 与 `qwen3_vl_image_token_count` 计算 `(t,h,w)` grid
   和 merger 后 image-token 数；
3. `qwen3_vl_patchify` 接受已经 decode、resize、rescale、normalize 的 CHW
   浮点 image，按官方 temporal/channel/patch/merge 顺序输出
   `(C×T×P×P, total_patches)`。

静态图像的 temporal patch 通过确定性 broadcast 构造。`Qwen3VLVisionInput`
再要求 patch width 与 grid 总量完全一致、值 finite，并保留 host grid metadata；
它不接受 PNG/JPEG bytes，也不会静默执行 resize 或 normalize。

Transformers oracle 使用一张确定性 `256×256` RGB 图，经官方 image processor
得到 `pixel_values=(256,1536)`、`grid_thw=(1,16,16)`；Julia 从同一份已归一化
CHW image 独立 patchify，max-abs 为 `0`。这项 exact 对照钉死了最容易被
普通 shape test 漏掉的 `2×2` merge-group 内 patch 顺序。

## vision tower 实现

- 3D patch projection 将官方 `(out, channel, temporal, height, width)`
  Conv3D 权重显式转换为 LifeAI column-major linear layout；不依赖 shape
  恰好可 reshape 的偶然性。
- `2,304 = 48×48` learned position table 按每个 media 的目标 H/W
  双线性插值；H/W vision RoPE 依照 merge-group 顺序生成。
- 24 个 pre-LayerNorm blocks 使用 fused QKV、per-temporal-frame packed
  attention、exact attention mask、GELU-tanh MLP 和 residual；`grid_t` 中的
  每个 temporal frame 形成独立 attention chunk。
- block `5 / 11 / 17` 的 post-block hidden 分别进入三个 DeepStack merger；
  block 23 后的 hidden 进入 main merger。两类 merger 的 norm/group 顺序按
  官方实现分别保留，并以 exact-erf GELU 投影到 text hidden width `2,048`。
- vision 参数可以按需读为 Float32 或 BFloat16，并通过 callback 放到 CPU/CUDA；
  loader 只读取 `model.visual.*` 的 `406,957,056` 个参数，不为 vision-only
  验证常驻完整 text decoder。

`hf_qwen3_vl_vision_forward` 返回未 merge 的 `patch_hidden_state`
（`1,024×N`）、main merger 后供 decoder 消费的 `visual_embeddings`
（`2,048×N/4`）、三份同为 merge 后输出的 DeepStack features，以及显式请求的
post-block checkpoints。capture layer
使用官方 0-based index，避免文档、checkpoint 名称和 Julia one-based 容器之间
出现静默偏移。

## 独立 reference 与真实 GPU 结果

`scripts/export_qwen3_vl_vision_reference.py` 固定 Transformers `4.57.0`、
eager attention、两条 registry revision 和确定性图像；reference 捕获
patch embedding、position-added、blocks `0 / 5 / 11 / 17 / 23`、三个
DeepStack output 与 main `visual_embeddings`。reference 在 Python/Transformers 中
生成，LifeAI verifier 只消费冻结 tensor，不用自身输出反向制造 oracle。

最终 reference SHA256 为：

- Float32：`480d988d9f679c8090f8c80c8e5cd007e5a41c47e6bb5cc7ad2f16541cbe5f88`；
- BF16：`ecd904b8a110169c73c9814d23d43eabcc5a2593d0a746bbbda8bb9c308b36b8`。

RTX 4090 D 上最终 `visual_embeddings` 的跨框架指标是：

| compute | 数值口径 | max-abs | mean-abs | relative L2 | cosine |
| --- | --- | ---: | ---: | ---: | ---: |
| Float32 | strict parity | `6.771088e-5` | `2.2492045e-6` | `5.461097e-6` | `1.0` |
| BF16 | cross-backend boundary | `0.890625` | `0.02120505` | `0.05465893` | `0.9985048` |

Float32 全部 stage 均通过逐 stage strict gate；这证明 position interpolation、
RoPE、24 层 residual、DeepStack/main merger 的组合没有只在 tiny shape 上成立。

对应生命周期与同步计时为：

| compute | logical vision parameter bytes | load | cold forward | warm forward |
| --- | ---: | ---: | ---: | ---: |
| Float32 | `1,627,828,224` | `5.934235821 s` | `11.319966811 s` | `0.024733474 s` |
| BF16 | `813,914,112` | `3.188056864 s` | `11.916055118 s` | `0.056884242 s` |

cold forward 包含 Julia/CUDA 首次编译，**不是 steady-state
benchmark**；warm 是完全相同输入的第二次前向，并在计时区间内显式
`CUDA.synchronize()`。logical parameter bytes 只按 vision tensor dtype 计数，
不是进程总 VRAM。

BF16 reference 在 CPU Transformers 生成，而 LifeAI 在 CUDA 上执行，24 层中的
BF16 matmul、归约和逐算子舍入顺序会累积差异。该结果证明真实 BF16 路径可完整
执行且输出 finite，但只冻结为**跨后端数值边界**，不是 strict BF16 parity；
本章 correctness gate 以 Float32 结果为准。

复现流程：

```bash
MODEL_DIR=/home/ubuntu/models/modelscope/Qwen/Qwen3-VL-2B-Instruct
REFERENCE_DIR=/tmp/qwen3-vl-vision-float32

PYTHONPATH=/tmp/lifeai-qwen3vl-oracle/lib/python3.10/site-packages \
  .venv/bin/python scripts/export_qwen3_vl_vision_reference.py \
  "$MODEL_DIR" "$REFERENCE_DIR" float32

julia --project=. --startup-file=no \
  scripts/verify_qwen3_vl_vision_cuda.jl \
  "$MODEL_DIR" "$REFERENCE_DIR" cuda
```

把最后一个 exporter 参数改为 `bfloat16` 并使用独立 output directory，可重跑
BF16 diagnostic。两类 reference 均在仓库外，不提交 checkpoint 或大 tensor。

## 测试与验证

Chapter 43 未设置真实模型目录时为 `178 passed + 1 explicit skip`；设置
`LIFEAI_QWEN3_VL_2B_MODEL_DIR` 后为 `190 / 190`：

- immutable asset/tensor/config/processor contract；
- smart resize、grid/token count、手算 patchify merge-order；
- input shape、dtype、finite、多 media boundary 与多 temporal-frame 独立
  attention chunks；
- deterministic tiny Float32/BF16 full forward；
- post-block capture 与 DeepStack 0-based order oracle；
- `LIFEAI_QWEN3_VL_2B_MODEL_DIR` opt-in 的真实 13-file SHA256、625 tensor
  全量 checkpoint 验证。

真实 checkpoint 验证不会自动下载模型；环境变量未设置时明确跳过。下载与完整
校验命令见 [`notes/local_model_assets.md`](../../local_model_assets.md)。

## 已知边界与决策

- **没有 raw image processor**：当前没有 PNG/JPEG decode、pixel resize、
  rescale 或 normalize 实现；smart resize 只返回 geometry，patchify 从已处理
  CHW image 开始。
- **没有 multimodal chat**：content-list schema、chat template、vision
  placeholder expansion 和多图用户消息尚未接入。
- **没有 decoder multimodal semantics**：text config 只是冻结契约；mRoPE
  position ids、DeepStack injection、image-conditioned decoder prefill 都未实现。
- **没有 generation/cache**：不能从图像生成文本，也没有 multimodal KV
  cache、stop/EOS 或质量结论。
- **没有 video pipeline**：vision input/attention 已用预处理后的 `grid_t > 1`
  验证 temporal frames 各自形成独立 attention chunk，但没有 video decode、
  frame sampling、temporal patchifier、chat 或端到端 video forward API。
- BF16 `visual_embeddings` cosine 低于 Float32 是已记录的跨后端边界；未来若要求 strict BF16
  parity，应冻结相同 device/kernel/math mode 的 oracle，而不是放宽本章标题。

## Close 回顾

- **完成了什么**：冻结 Qwen3-VL-2B-Instruct 双 registry provenance 和完整
  checkpoint；实现严格 config/processor/tensor contract、预处理后 patch/grid
  边界，以及真实可加载的 24-layer vision tower、DeepStack 和 merger。
- **验证证据**：13 个资产与 625 个 BF16 tensor 全量校验；Chapter 专项在
  默认边界为 `178 passed + 1 explicit skip`、真实 checkpoint 为 `190 / 190`；
  Transformers 4.57.0 对照的 Float32 `visual_embeddings` max/mean/relative
  L2/cosine 为
  `6.771088e-5 / 2.2492045e-6 / 5.461097e-6 / 1.0`；同步 cold/warm 为
  `11.319966811 / 0.024733474 s`。
- **没有完成及原因**：raw processor、chat template、decoder mRoPE/injection、
  KV cache、image-to-text generation 和 video 被刻意留在 vision-only 边界之外，
  不能用 encoder parity 代替端到端多模态验证。
- **最重要的认知变化**：DeepStack 是 decoder 将消费的三份中间视觉表示，不是
  可忽略的调试输出；processor 的 merge-order 与 position/RoPE order 也必须在
  encoder 前就钉死，否则最终 shape 正确仍可能语义错位。
- **是否满足 Close 条件**：是。Float32 strict parity、真实 BF16 execution、
  checkpoint/processor contract、tiny/真实测试和能力边界均已冻结。
- **带到下一章的问题**：Chapter 44 实现官方 raw image/content-list processor
  与 chat template，并把 visual features、DeepStack 和 interleaved mRoPE 接入
  decoder prefill；Chapter 45 再完成 KV cache 和真实 image-to-text generation。
