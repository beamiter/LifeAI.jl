# Chapter 152 — Qwen3-VL 视频容器解码、采样生成与图文混合 Static Cache 复用

> 状态：Closed（2026-11-25）
>
> 所属卷：[Episode 10 — Qwen3-VL 高效生成](README.md)

## 目标

闭合 Chapter 151 留下的两项实用边界：
1. 支持直接消费本地视频容器文件（MP4/MKV/WebM/AVI），按 Qwen3-VL 官方 `smart_nframes` / linspace 策略均匀采样并保留真实源帧索引与 wall-clock 时间戳。
2. 在真实 `Qwen3-VL-2B-Instruct` Blackwell GPU 上验证：真实 `.mp4` 路径输入、可重放采样（`strategy=:sample`）、图文混合（image + MP4）、16 帧长视频，以及跨全部请求复用同一块预分配 `Qwen3VLStaticKVCache` 存储（56 块 K/V 设备指针不变、输出与 fresh `:dynamic` 逐位一致）。

## 实现

- **视频容器 I/O 与帧采样（[`src/data/hf_qwen3_vl_video_io.jl`](../../../src/data/hf_qwen3_vl_video_io.jl)）**：
  - `qwen3_vl_sample_frame_indices`：严格校验 `total_frames`/`source_fps`/`target_fps`/`min_frames`/`max_frames`/`frame_factor`，按官方时间因子对齐并返回 0-based 均匀采样索引。
  - `qwen3_vl_probe_video_file`、`qwen3_vl_decode_video_file`、`qwen3_vl_encode_video_file`：自动解析系统 `ffmpeg`/`ffprobe`、`LIFEAI_FFMPEG`/`LIFEAI_FFPROBE` 或已安装的 `FFMPEG_jll`（通过 `Base.invokelatest` 安全跨 world-age 调用），对缺失/空/损坏/超文件字节预算/超解码字节预算的容器 fail closed，无临时文件流式输出 RGB `UInt8` 帧与 `Qwen3VLDecodedVideo`。
- **高层管线无缝接入（`src/data/hf_qwen3_vl_processor.jl`、`src/generation/qwen3_vl_generation.jl`）**：
  - `qwen3_vl_process_video` 与 `generate_hf_qwen3_vl` 直接接受本地视频文件路径字符串和 `Qwen3VLDecodedVideo`，支持在 content item 中指定 `target_fps`、`min_frames`、`max_frames`；不存在的路径仍先于 cache 变更抛出 `ArgumentError`（保持 Chapter 149 契约）。

## 验证结果（2026-11-25，NVIDIA RTX PRO 5000 72GB Blackwell，Julia 1.12.6 `-O1`）

- **离线 CPU 专项与回归**：[`test_qwen3_vl_video_container.jl`](../../../test/episodes/episode10_qwen3_vl_efficient_generation/chapter152_qwen3_vl_video_container/test_qwen3_vl_video_container.jl) 覆盖帧采样边界、缺失/损坏/超预算容器拒绝、真实 MP4 编解码往返、采样生成及图文混合 static cache 复用；DSW 上 Episode 09/10/11 离线回归 **5,986** 项通过（Episode 09 `1,878` 项 + 1 opt-in broken skip，Episode 10 `4,069/4,069`，Episode 11 `39/39`）。
- **真实 `Qwen3-VL-2B-Instruct` Blackwell GPU 验收**（[`scripts/verify_qwen3_vl_video_container_cuda.jl`](../../../scripts/verify_qwen3_vl_video_container_cuda.jl)，报告冻结于 [`benchmark_results/qwen3_vl_video_container/summary.json`](../../../benchmark_results/qwen3_vl_video_container/summary.json)，SHA-256 `db6e3e609556b601704dce5e439e70acb51853655341d38df92f501cc97d1c64`）：
  - 在 Float32 与 BFloat16 下各预分配 1 块 `capacity=256` 的共享 `Qwen3VLStaticKVCache`，依次执行：单图（`"Based on the image provided,"`）→ 8 帧 MP4（`14,647` B，`grid=[2,6,8]`）greedy（`"The video starts with a colorful"`）→ 8 帧 MP4 `strategy=:sample`（固定 uniform，`"A vibrant array of colors,"`）→ 图文混合（`image + MP4`，`prompt_len=132`，`"Yes, the square appears in"`）→ 16 帧 MP4（`28,487` B，`grid=[8,6,8]`，`prompt_len=176`，`"The square moves from the top"`）。
  - 全部 56 块 K/V 设备指针全程保持不变（`kv_pointer_identity_preserved=true`），Float32 与 BFloat16 下全部 5 类请求与 fresh `:dynamic` 逐步 logits **逐位一致**（`all_dynamic_vs_reused_bitwise_exact=true`）。

## 明确边界

- 当前仅解复用主视频轨（`-map 0:v:0`），不包含音频轨理解。
- 超长视频（>100 帧）显存压力测试、batch > 1 padding 与整图编译仍未关闭；Episode 10 保持 Open。
