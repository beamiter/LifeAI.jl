# Chapter 151 — Qwen3-VL 多帧视频理解与图文混合生成

> 状态：Closed（2026-11-25）
>
> 所属卷：[Episode 10 — Qwen3-VL 高效生成](README.md)

## 目标

将 Qwen3-VL 推理链路从单图/多图扩展到**多帧视频**与**图文混合（image + video）**生成，并对照官方 Transformers `4.57.6` 对真实 `Qwen3-VL-2B-Instruct` 权重完成 Blackwell GPU 端到端验收。

## 实现

1. **视频预处理与时间戳契约（`src/data/hf_qwen3_vl_processor.jl`）**
   - 冻结官方 `video_preprocessor_config.json` SHA-256（`7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13`），新增 `qwen3_vl_video_processor_spec` 与 `load_hf_qwen3_vl_video_processor_config`（`shortest_edge=4096`、`longest_edge=25165824`、`patch_size=16`、`temporal_patch_size=2`、`merge_size=2`）。
   - 实现 `qwen3_vl_video_smart_resize`（按 `t_factor = temporal_patch_size` 和空间 `factor = patch_size * merge_size` 联合缩放）、`qwen3_vl_video_grid`、`qwen3_vl_video_token_count`、奇数帧末帧补齐与时间分块 `qwen3_vl_patchify_video`、以及官方中点时间戳 `qwen3_vl_video_timestamps`。
   - 实现 `qwen3_vl_process_video`（支持本地帧路径向量、3D `UInt8` 帧序列、以及 4D `UInt8` 张量 `:thwc/:hwct/:tchw`）和 `qwen3_vl_expand_video_placeholders`（按时间组展开为 `<{t:.1f} seconds><|vision_start|><|video_pad|>...<|vision_end|>`）。
   - 扩展 `qwen3_vl_rope_layout(...; video_grid_thw=nothing)`：按官方 `get_rope_index` 将每个 `(grid_t, h, w)` 视频网格拆成 `grid_t` 组 `(1, h, w)`，同时保持缺省 `video_grid_thw=nothing` 时对既有 image-only 拒绝契约完全向后兼容。

2. **Decoder / Cache / 高层生成（`src/models/qwen3_vl_decoder.jl`、`src/models/qwen3_vl_cache.jl`、`src/generation/qwen3_vl_generation.jl`）**
   - `hf_qwen3_vl_text_prefill`、`hf_qwen3_vl_text_prefill_cached`、`hf_qwen3_vl_text_prefill_static` 及 `generate_hf_qwen3_vl_tokens` 新增显式 `allow_video::Bool=false` 参数，未显式启用时保持 Chapter 44–46 的 fail-closed 契约。
   - `generate_hf_qwen3_vl` 统一支持纯图片、纯视频及图文混合消息，完整支持 dynamic/static/显式复用 cache 与 greedy/sample 策略，并在返回结果中补充 `video_grid_thw`、`original_video_sizes`、`resized_video_sizes`、`video_timestamps`。

3. **独立 Oracle 导出与真实 GPU 验收脚本**
   - `scripts/export_qwen3_vl_video_reference.py`：调用官方 `AutoProcessor` 与 `Qwen3VLForConditionalGeneration.generate(..., do_sample=False, output_scores=True)` 在 CPU Float32 导出确定性 4 帧 `96×128` 运动方块视频的参考张量与 8 步 logits（`tensor_sha256=e174bd66f20d42366615350ae10be47d48a8002d1c88c297414f011c4bd5fcd7`）。
   - `scripts/verify_qwen3_vl_video_cuda.jl`：端到端核验预处理像素、视频网格、chat template 展开、token IDs、3D mRoPE、`rope_deltas`，以及真实 2B 权重在 CUDA Float32（严格 HF gate）和 BFloat16（同 dtype dynamic/static/reuse 逐位一致）下的生成。

## 验证结果（2026-11-25，NVIDIA RTX PRO 5000 72GB Blackwell，Julia 1.12.6 `-O1`）

- **离线 CPU 专项与 Episode 09/10 回归**：`Episode 09/10 Qwen3-VL Complete CPU + Chapter 151 Video` 共 **5,072 / 5,072** 项通过（覆盖 Chapter 43–47、145、147、148、149 及 Chapter 151 视频/图文混合 Float32 与 BF16、dynamic/static/reuse）。
- **真实 `Qwen3-VL-2B-Instruct` Blackwell GPU 视频验收**（结果冻结于 [`benchmark_results/qwen3_vl_video/summary.json`](../../../benchmark_results/qwen3_vl_video/summary.json)，SHA-256 `ee2a24d2a82e74b8f461100b620811fae76b877a07c6a5d7b08b272acd8be9fe`）：
  - 输入：4 帧 `96×128` RGB 视频，`fps=2.0`，`video_grid_thw=[2, 6, 8]`，`prompt_length=58`，`rope_delta=-16`。
  - **像素预处理最大绝对误差**：`0.0`（与 Transformers 4.57.6 逐位一致）；token IDs、`position_ids`、`rope_deltas` 全部 exact。
  - **Float32 CUDA 严格 HF 对照**：8/8 greedy token 完全一致（`"A colorful square moves across the screen."`），dynamic == static == reused 逐位一致；8 步全词表 logits 最大绝对误差为 **`3.7670e-5`**（每步：`3.05e-5, 2.91e-5, 3.15e-5, 3.77e-5, 2.79e-5, 3.10e-5, 3.43e-5, 2.44e-5`，均小于 `1e-4`）。
  - **BFloat16 CUDA 内部一致性**：dynamic == static == reused 全部 8 步逐位一致；前 7 个 token 与 Float32 相同，第 8 个 token 为 `,`（id 11）而非 `.`（id 13），属于已记录的 BF16/Float32 跨 dtype 边界，不声称 BF16 strict HF parity。

## 明确边界

- 当前视频入口直接消费帧序列（图像路径/3D `UInt8` 帧）或 4D `UInt8` 张量，尚未内置容器解复用（MP4/MKV 音频/视频解码仍由调用方提供帧与 `fps`/`frame_indices`）。
- 未覆盖长视频（>100 帧）显存压力测试、batch > 1 padding 或整图编译；Episode 10 保持 Open。
