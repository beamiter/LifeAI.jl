# Week 09 Qwen3-0.6B CPU 推理基线

原始数据：[`qwen3_0.6b_cpu.json`](qwen3_0.6b_cpu.json)

## 固定条件

- 模型：`Qwen/Qwen3-0.6B` revision `c1899de289a04d12100db370d81485cdf75e47ca`
- 权重/计算：BF16 safetensors storage → Float32 parameters/compute
- CPU：Intel Core Ultra 7 270K Plus，24 logical CPUs
- Julia：1.12.6，`JULIA_NUM_THREADS=1`
- OpenBLAS：24 threads
- prompt：16 / 64 / 256 tokens；每组 decode 8 tokens
- steady-state：3 个原始样本，中位数；每次 sample 前在计时窗外执行 GC，且不保留前一次 sample 的 cache
- static cache：`max_seq_len=264`

模型加载耗时 `12.517 s`；加载后 peak RSS `5.01 GiB`，整个 benchmark peak RSS `6.13 GiB`。

## Decode 结果

| Prompt tokens | Full recompute tok/s | Dynamic tok/s | Dynamic speedup | Static tok/s | Static speedup | Dynamic cache | Static cache | Cache max-abs |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 7.22 | 14.28 | 1.98× | 11.03 | 1.53× | 5.25 MiB | 57.75 MiB | `2.19e-5` |
| 64 | 3.97 | 14.77 | 3.72× | 9.90 | 2.49× | 15.75 MiB | 57.75 MiB | `1.01e-4` |
| 256 | 1.06 | 10.95 | 10.33× | 12.49 | 11.79× | 57.75 MiB | 57.75 MiB | `5.96e-5` |

三种长度的 full/dynamic/static correctness 均通过；dynamic/static prefill 与 full forward 在当前实现中 max-abs 为 0，逐 token decode 的全局最大误差为 `1.01447105e-4`。

## 结论与边界

- full recompute 的 decode 成本随已有上下文迅速上升；KV cache 在 256-token prompt 时已经带来约 10–12× 吞吐提升。
- dynamic cache 在短 prompt 更快且按实际 token 数占用内存；static cache 为固定 shape 预留完整 264-token 空间，在 16-token prompt 时用约 11× cache 内存换取编译友好的布局，但 CPU 下并未更快。
- 到 256-token prompt 时 dynamic cache 已增长到 static 的完整容量，static decode 反而领先约 14%；这只是当前 CPU/Float32/shape 的测量，不外推到 CUDA/XLA。
- raw samples 仍显示少量系统级波动，尤其是 full prefill/decode 和 static decode；因此这里只使用中位数并保留每次原始值，不把 3 samples 当生产性能结论。
- JSON SHA256：`f45b363b8aeb3afd1dac52b4909f2f22d92099cc842caddef72bc4faae6f84e6`。
