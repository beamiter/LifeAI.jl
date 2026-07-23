# Week 09 Qwen3-0.6B 真实推理基线

原始数据：

- [`qwen3_0.6b_cpu.json`](qwen3_0.6b_cpu.json)
- [`qwen3_0.6b_cuda.json`](qwen3_0.6b_cuda.json)
- [`qwen3_0.6b_xla_gpu.json`](qwen3_0.6b_xla_gpu.json)

## 共同条件

- 模型：`Qwen/Qwen3-0.6B` revision `c1899de289a04d12100db370d81485cdf75e47ca`
- 权重/计算：BF16 safetensors storage → Float32 parameters/compute
- Julia：1.12.6，`JULIA_NUM_THREADS=1`
- 计时同步：每次模型调用后将 logits 物化到 host；因此结果包含当前 host 采样边界所需的数据传输
- steady-state：3 个原始样本并保留逐样本时间；表格使用中位数

## CPU

- CPU：Intel Core Ultra 7 270K Plus，24 logical CPUs
- OpenBLAS：24 threads
- prompt：16 / 64 / 256 tokens；每组 decode 8 tokens
- 每次 sample 前在计时窗外执行 GC，且不保留前一次 sample 的 cache
- static cache：`max_seq_len=264`

模型加载耗时 `12.517 s`；加载后 peak RSS `5.01 GiB`，整个 benchmark peak RSS `6.13 GiB`。

### Decode 结果

| Prompt tokens | Full recompute tok/s | Dynamic tok/s | Dynamic speedup | Static tok/s | Static speedup | Dynamic cache | Static cache | Cache max-abs |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 7.22 | 14.28 | 1.98× | 11.03 | 1.53× | 5.25 MiB | 57.75 MiB | `2.19e-5` |
| 64 | 3.97 | 14.77 | 3.72× | 9.90 | 2.49× | 15.75 MiB | 57.75 MiB | `1.01e-4` |
| 256 | 1.06 | 10.95 | 10.33× | 12.49 | 11.79× | 57.75 MiB | 57.75 MiB | `5.96e-5` |

三种长度的 full/dynamic/static correctness 均通过；dynamic/static prefill 与 full forward 在当前实现中 max-abs 为 0，逐 token decode 的全局最大误差为 `1.01447105e-4`。

### CPU 结论

- full recompute 的 decode 成本随已有上下文迅速上升；KV cache 在 256-token prompt 时已经带来约 10–12× 吞吐提升。
- dynamic cache 在短 prompt 更快且按实际 token 数占用内存；static cache 为固定 shape 预留完整 264-token 空间，在 16-token prompt 时用约 11× cache 内存换取编译友好的布局，但 CPU 下并未更快。
- 到 256-token prompt 时 dynamic cache 已增长到 static 的完整容量，static decode 反而领先约 14%。
- raw samples 仍显示少量系统级波动，尤其是 full prefill/decode 和 static decode；因此这里只使用中位数并保留每次原始值，不把 3 samples 当生产性能结论。

## CUDA GPU

- GPU：NVIDIA GeForce RTX 5080，compute capability 12.0，15.45 GiB
- CUDA driver/runtime：13.3 / 13.3
- OpenBLAS：12 threads，仅用于 host 加载/reference 工作
- prompt：16 / 64 / 256 tokens；每组 decode 8 tokens；static `max_seq_len=264`
- 模型加载 `11.720 s`；Float32 参数传输 `3.064 s`；传输后 CUDA used memory `2.22 GiB`

| Prompt tokens | Full tok/s | Dynamic tok/s | Dynamic speedup | Static tok/s | Static speedup | Cache max-abs |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 44.25 | 86.06 | 1.94× | 81.55 | 1.84× | `2.96e-5` |
| 64 | 22.61 | 84.60 | 3.74× | 82.17 | 3.63× | `9.92e-5` |
| 256 | 7.26 | 67.30 | 9.27× | 81.99 | 11.29× | `9.44e-5` |

三种长度的 dynamic/static correctness 均通过，decode global max-abs `9.918213e-5`。GPU 上 static decode 在三种长度都约为 82 tok/s；dynamic 在 256-token prompt 下因增长/布局成本降至 67.30 tok/s。`observed_summarysize` 只反映 Julia CuArray wrapper，不代表设备 buffer；cache 显存应使用 JSON 中的 theoretical bytes 和 CUDA allocator 指标。

## Reactant-XLA GPU

- GPU/模型/dtype 与 CUDA 实验相同
- prompt 16 tokens、decode 2 tokens、static cache `max_seq_len=18`
- XLA 初始化使用 11.59 GiB BFC allocator；decoder setup `2.182 s`
- prefill/decode compile + first run：`64.745 / 24.771 s`
- steady prefill 中位数 `15.03 ms`；steady decode `14.50 ms / 2 tokens`，即 `137.89 tok/s`
- 相对 CPU Float32 reference：prefill/decode max-abs `0.0160928 / 0.0115070`；冻结容差 `atol=2e-2, rtol=5e-3`，prefill 与所有 decode argmax 一致

XLA 的 steady-state 明显快于当前 CUDA eager/static 路径，但本组只使用 16+2 的固定形状，不能与 CUDA 16+8 表格直接做严格吞吐排名。更重要的工程结论是：真实 0.6B XLA 静态 decoder 可执行且 token 决策一致，但 cold compile 约 90 秒，BFC allocator 固定占用也接近整卡显存的 75%。

## 校验和与边界

- CPU JSON SHA256：`f45b363b8aeb3afd1dac52b4909f2f22d92099cc842caddef72bc4faae6f84e6`
- CUDA JSON SHA256：`53212443b6691ec870013fb15313736ecf5e2b67f3d0145cbe67a1459ee52597`
- XLA GPU JSON SHA256：`cb185126dcf4d0be2ad8732e97f0a4a1d4c51438ee52731b116b33bf34d89b03`
- 三组都使用 Float32 parameters/compute；这些结果不能作为 native BF16、量化或生产 serving 性能结论。
- CUDA 与 XLA 的输出同步包含 host materialization，符合当前 token-by-token generation 实现，但与完全 device-resident sampler 的指标不同。

## 复现命令

```bash
LIFEAI_QWEN3_REVISION=c1899de289a04d12100db370d81485cdf75e47ca \
julia --startup-file=no --project=. scripts/benchmark_qwen3_inference.jl \
  /home/yj/models/huggingface/Qwen/Qwen3-0.6B/c1899de289a04d12100db370d81485cdf75e47ca \
  benchmark_results/week09/qwen3_0.6b_cpu.json 3

julia --startup-file=no --project=. scripts/benchmark_qwen3_accelerator.jl \
  /home/yj/models/huggingface/Qwen/Qwen3-0.6B/c1899de289a04d12100db370d81485cdf75e47ca \
  benchmark_results/week09/qwen3_0.6b_cuda.json cuda 3

julia --startup-file=no --project=. scripts/benchmark_qwen3_accelerator.jl \
  /home/yj/models/huggingface/Qwen/Qwen3-0.6B/c1899de289a04d12100db370d81485cdf75e47ca \
  benchmark_results/week09/qwen3_0.6b_xla_gpu.json xla_gpu 3
```
