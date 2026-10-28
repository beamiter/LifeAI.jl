# Flint 第二轮解码优化（2026-10-28）

本轮把完整模型稳定解码中位数从 **19.04 提升到 30.75 tok/s（1.61 倍）**。
已部署到同一台 RTX PRO 5000 72GB Blackwell，保留原生 FP4 W4A4。
结果与源文件哈希见 [summary.json](../benchmark_results/flint_decode_optimized_v3/summary.json)。
前一轮从 1.24 提升到约 18 tok/s 的记录见 [第一轮报告](flint_performance.md)。
本轮重新测量了基线，波动范围与上一轮略有不同。

## 完整模型实测

基线为 `5d61acf` 的已预热服务。在所有独立 GPU 测试退出后测量三轮。
候选版重启并额外预热一个相同的 64-token 请求，再测三轮。
固定提示词“请用中文解释什么是递归，并给出一个简单例子。”，温度 0，
单请求、15 prompt tokens、64 output tokens；经过与 Lumen 相同的流式路由。

| 指标 | v2 三轮 | v3 三轮 |
| --- | --- | --- |
| 稳定解码，tok/s | 19.04 / 16.34 / 19.23 | 26.51 / 31.02 / 30.75 |
| 请求总耗时，秒 | 4.97 / 6.65 / 4.72 | 3.58 / 3.35 / 3.47 |
| 首 token，秒 | 1.65 / 2.77 / 1.42 | 1.18 / 1.30 / 1.40 |

解码为 `(输出数-1)/(生成总时间-首token时间)`，不含加载与 prefill。
总耗时为客户端墙钟时间。短样本仍有 GC、调度和时钟波动；不代表所有
提示词、并发或长上下文的速度。首次排除的预热请求约 6.83 秒。

## 本轮改动

- **共享输入量化**：`native_nvfp4_gemm_multi` 支持最多 16 个投影，
  Q/K/V 共用一份激活；六个路由专家与共享专家的 14 个 gate/up 投影
  也一次量化。每个权重保留独立 device alpha，各专家仍按原顺序累加。
- **融合 RMSNorm 与缓存 RoPE**：归一化减少为一个 CUDA kernel。
  RoPE 使用与旧实现相同的 CPU Float32 三角函数表，每 context 最多
  8 个 2048-position 窗口，常见解码只启动一个旋转 kernel；跨窗口 prefill
  保留通用实现。
- **复用 KV**：全注意力缓存按实际需要倍增；滑窗使用双份环形存储，
  一个 kernel 同时追加 K/V，避免每 token 拼接整个前缀。保留逻辑顺序、
  窗口长度及失败重试时上一状态的可见内容。长期保留缓存快照时应复制视图。
- **设备端贪心采样**：温度为 0 时在 GPU 做 argmax 与有限值检查，只回传
  一个 token ID；保留首索引 tie 及正负零顺序。其他温度仍使用原采样器。
- **主机开销**：具体化热路径类型、跳过未改变的 B-scale 指针更新，
  为每个权重保留 decode 和最近一个 prefill plan，统计改为单遍累加。

CUDA 算子与自定义 kernel 的实现接口参考
[CUDA.jl kernel 文档](https://cuda.juliagpu.org/stable/development/kernel/)。
本轮没有使用 CUDA graph，也尚未将专家路由迁到 GPU。

## 微基准与内核证据

| 操作 | 原实现 | 优化后 |
| --- | --- | --- |
| RMSNorm，2048 | 17.85 µs | 4.60 µs |
| RoPE，128×32×1 | 30.67 µs | 3.94 µs |
| 150144 词表贪心采样 | 305.19 µs | 25.64 µs |
| 滑窗 KV 追加，511 tokens | 103.54 µs | 3.39 µs |
| 14 个 gate/up，七次 pair 对比一次 batch | 244.66 µs | 183.53 µs |

微基准使用预热后的独占测量；不能直接作为完整模型倍率。
CUPTI 对十次 14 投影调用观察到 20 个量化 kernel 和 140 个真实 SM120
E2M1 Tensor Core GEMM，即每组只做一次 amax 和一次 pack。

## 数值与功能验证

最终组合包含全部候选扩展与 MoE 批处理：CPU 模型/量化/流式 420 项、
CUDA attention 25 项、多投影 1874 项、既有 FP4 decode 185 项、融合
kernel 126 项、完整/缓存 tiny native Flint 81 项、长 KV 52 项通过。
另有 CPU/GPU KV 不变量 15928/15929 项和 HTTP/UTF-8 43 项，覆盖扩容、
W=1/2/5/17/512、重复绕环、prefill/decode 切换和失败重试。

tiny native Flint 对 CPU W4A4 oracle 的相对 L2 误差约 `6.68e-8`，
长 KV teacher-forced 测试最大约 `3.15e-7`。测试输入中的多投影结果与
分别调用一致，RoPE 与旧 GPU 实现逐位一致，GPU greedy token ID 与
CPU 采样器一致。RMSNorm 的并行归约顺序会改变浮点计算；完整模型
贪心文本与 v2 不同，三轮 v3 文本一致。这些检查不替代完整任务质量评测。

真实服务的三个基础功能检查通过：`17+25=42`、判断 37 为质数，及
生成 Python `add(a,b)` 并验证正负数输入。实际 Lumen PTY 生成、退出和
终端恢复也通过。服务当前为 `lifeai_flint_decode_v3`。

## 复现与记录

```sh
python3 scripts/benchmark_flint_service.py --tokens 64 --runs 1 --output /path/warmup.json
python3 scripts/benchmark_flint_service.py --tokens 64 --runs 3 --gpu-sample --output /path/measured.json
julia -O1 --project=. test/test_nvfp4_multi_projection.jl
julia -O1 --project=. test/test_flint_decode_kernels.jl
LIFEAI_FLINT_KV_CUDA_TESTS=true julia -O1 --project=. test/test_flint_kv_decode.jl
```

完整原始记录位于
`/jfs/application/user/bgl/benchmarks/flint-performance-v3-20261028/`。
当前仍有 CPU 专家路由、逐专家下投影和算子调度开销；尚未达到官方融合
引擎的速度。硬件、量化产物与官方测速条件的区别见第一轮报告。
