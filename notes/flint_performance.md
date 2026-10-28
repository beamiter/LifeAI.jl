# Flint 解码性能分析与优化（2026-10-28）

完整模型稳定解码从 **1.24 tok/s 提升到 18.02 tok/s（14.54 倍）**。
优化版已部署到 DSW，保留原生 cuBLASLt FP4 W4A4 计算。
机器为 RTX PRO 5000 72GB Blackwell，驱动 580.126.09，Julia 1.12.6。
完整结果、源文件哈希与测试记录见
[summary.json](../benchmark_results/flint_decode_optimized/summary.json)。

## 完整模型实测

固定提示词“请用中文解释什么是递归，并给出一个简单例子。”，单请求、
温度 0、输出 64 tokens，走与 Lumen 相同的路由和流式接口。
基线为提交 `eb34f9d` 的已预热驻留服务；优化版额外预热一次同样的
64-token 请求，然后测量两轮。计时不包含权重加载，decode 单独排除首 token。

| 指标 | 原实现，两轮 | 优化后，两轮 |
| --- | --- | --- |
| 稳定 decode，tok/s | 1.245 / 1.234 | 17.661 / 18.376 |
| 请求总耗时，秒 | 55.08 / 57.23 | 5.01 / 4.94 |
| 服务端首 token，秒 | 4.44 / 6.17 | 1.43 / 1.49 |
| 端到端吞吐，tok/s | 1.162 / 1.119 | 12.807 / 13.010 |
| GPU 平均利用率 | 18.2% | 34.9% |

decode 的计算方式为 `(输出 tokens - 1) / (总生成时间 - 首 token 时间)`。
GPU 通过 nvidia-smi 每 0.5 秒采样，短请求的利用率统计只作辅助。
两轮样本不能覆盖所有上下文、并发、专家路由或首次 JIT 的性能。
单独诊断请求“你好”生成 209 tokens 后正常停止，约 12.43 秒；该请求含
一次 1 秒 CPU 采样，不参与上表。重新加载与预热权重耗时 244.31 秒，也不计入吞吐。

## 改动及原因

原实现已经使用原生 FP4 GEMM，但每次小矩阵乘都调用通用激活量化器，
多次验证/读取 GPU 标量，再同步 GEMM。模型还逐个计算 32 个注意力头。
这些操作产生大量小 kernel、临时数组与主机往返，GPU 无法连续工作。

- 注意力改为两次 GQA 分组批量矩阵乘，保留 mask、learned sink 和 KV head
  对应关系，无需复制 KV 到所有 query heads。
- 单 token FP4 以两个 CUDA kernel 完成 amax、量化和 scale swizzle，
  cuBLASLt alpha 留在 GPU；同 stream 按执行顺序复用 scratch，跨 stream
  才交接等待。公共调用默认保留有限值检查；模型内部在最终采样前统一检查 logits。
- 单 token MoE 复用相同输入，去除逐专家 gather、单元素系数上传和 scatter；
  避免重复 Float32 转换和不必要的完整 KV 拷贝。
- 服务新增独立的 prefill/decode 指标；Lumen 使用精确 decode 速率，
  `eval_duration` 原有含义保留以兼容其他客户端。

注意力核心微基准为 33.5–39.9 倍，三个单 token FP4 GEMM 微基准为
6.7–11.9 倍；完整模型加速以上表为准。CUPTI 观察到每次 FP4 调用为
两个量化 kernel 和一个真实 SM120 E2M1 Tensor Core GEMM。

## 数值与验证范围

CPU 模型 oracle/量化/流式检查 104 项、HTTP/UTF-8 检查 43 项通过；
CUDA 注意力/MoE 25 项、融合 FP4 185 项、原有 native 回归 81 项通过。
测试覆盖量化字节、scale、零值/次正规数、padding、异步复用、跨 stream、
完整与缓存 tiny Flint、生成和产物重载。CUDA 注意力最大绝对误差约
`7.16e-7`，tiny native Flint 对 CPU W4A4 oracle 的相对 L2 误差约 `1.13e-7`。

分组 FP32 注意力改变了累加顺序，完整模型贪心文本与基线不完全一致；
两轮优化版文本一致。测试输入的量化字节一致不代表完整模型逐位等价，
也不能替代代码任务质量评测。本次未重跑完整质量基准。

## 与官方宣传的差距

官方目前描述的是 **RTX Spark 64GB、短上下文、优化引擎超过 150 tok/s**；
另一张 128k 会话平均速度图排除了 prefill。这些条件与本机短上下文测试
不同，官方页面也没有给出短上下文 headline 的完整输入/输出长度与重复次数。
[官方公告](https://recursive.com/updates/introducing-recursive-flint)

官方 NVIDIA 路径提供单独准备的约 22.9GB NVFP4 GGUF 和 llama.cpp，
与 LifeAI 当前 BF16 在线 maxabs 量化的产物不同。
[官方量化产物](https://huggingface.co/RecursiveSI/flint-nvfp4-gguf)、
[NVIDIA 部署文档](https://docs.recursive.com/local/nvidia)

当前开源 `rec` 主要是 Apple 后端，其完整前向 megakernel、设备端路由和
多 token 驻留采样展示了完整引擎优化的方向，不能直接用于这台 CUDA GPU。
[rec 源码](https://github.com/recursive-oss/rec)

本次尚未达到官方速度。独立 CPU 采样仍观察到 launch、临时分配、
cuBLAS 调用及主机读取时的等待；等待计数包含之前排队的 GPU 工作，
不能直接解释为拷贝本身耗时。47 层 MoE 仍在 CPU 路由并逐专家调度。
进一步提升需要设备端路由/融合 MoE、缓冲区复用和 CUDA graph 等完整解码链路优化。

## 复现

在 Flint 服务已就绪且空闲时运行：

```sh
python3 scripts/benchmark_flint_service.py --tokens 64 --runs 1 --output /path/warmup.json
python3 scripts/benchmark_flint_service.py --tokens 64 --runs 2 --gpu-sample --output /path/measured.json
LIFEAI_FLINT_OPT_CUDA_TESTS=true julia -O1 --project=. test/test_flint_decode_optimized.jl
julia -O1 --project=. test/test_nvfp4_decode_optimized.jl
julia -O1 --project=. scripts/verify_flint_nvfp4_native.jl /path/native.json
```

服务仅在收到 Linux `SIGUSR1` 时采集一次 CPU profile 并写入 worker 日志，
正常推理不启用周期性采样。分析记录保存在
`/jfs/application/user/bgl/benchmarks/flint-performance-20261028/`。
