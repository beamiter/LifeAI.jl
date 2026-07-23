# Week 10 GPT-2 124M 复现与真实性能记录

原始记录：

- [`gpt2_124m_parity.json`](gpt2_124m_parity.json)
- [`gpt2_124m_cpu.json`](gpt2_124m_cpu.json)
- [`gpt2_124m_cuda.json`](gpt2_124m_cuda.json)

## 冻结资产

- 模型：`openai-community/gpt2`
- immutable revision：`607a30d783dfa663caf39e06633721c8d4cfcd7e`
- 持久目录：`/home/yj/models/huggingface/openai-community/gpt2/607a30d783dfa663caf39e06633721c8d4cfcd7e`
- reference：上述目录的 `lifeai_week10_reference/`，不使用 `/tmp`
- Python reference 环境：Python 3.12.13、PyTorch 2.7.1+cpu、Transformers 4.51.0、Tokenizers 0.21.4、Safetensors 0.5.3
- 模型随附 `generation_config.json` 声明的历史 Transformers 版本：`4.26.0.dev0`
- reference 条件：Float32、eager attention、eval mode、deterministic algorithms、单 CPU thread

冻结文件 SHA256：

| 文件 | SHA256 |
| --- | --- |
| `config.json` | `0daed7749b4f02b8f76240d5444551d7b08712dab4d0adb8239c56ba823bb7b4` |
| `generation_config.json` | `ed0b32ac72c0f5f44a719abb2d7786ea5146c871f83717b7f2018065954de02b` |
| `tokenizer.json` | `8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6` |
| `tokenizer_config.json` | `5e04eb606e3a1583530a42e36c2a6b6615c86f34fe77e44d9ddeb43ff940931f` |
| `vocab.json` | `196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783` |
| `merges.txt` | `1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5` |
| `model.safetensors` | `248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707` |

`load_hf_gpt2_bundle` 默认 fail closed：revision 和以上七个 checksum 必须全部匹配。

## 架构与权重映射

LifeAI 的同一 `GPTModel` 现在可显式选择 `:rope`、`:none` 或
`:learned_absolute`。GPT-2 配置为 learned absolute position、pre-LayerNorm
(`eps=1e-5`)、标准 12-head MHA、GELU-New、attention/MLP/LayerNorm bias、无
LM-head bias 的 tied embedding。旧模型仍保持原来的参数/state tree；Qwen3
继续使用 RoPE、RMSNorm、SwiGLU、GQA/QK-Norm。

HF `Conv1D.weight` 的 `(in, out)` 布局在 adapter 中显式转换：

- `c_attn.weight (768, 2304)` 按输出列切成 Q/K/V，再分别 transpose 成 Lux
  Dense `(out, in)`；
- attention `c_proj`、MLP `c_fc/c_proj` 同样显式 transpose；
- `wte/wpe` 从 `(vocab/position, hidden)` transpose 到 embedding 参数布局；
- 12 个 `(1, 1, 1024, 1024)` causal buffer 验证为严格下三角，但不进入参数树；
- tied `lm_head.weight` 若出现，只允许与 `wte.weight` 完全相等。

全 context 的参数量为 124,439,808。Dropout 值从 config 验证并记录，但本
Week adapter 是 eval/inference 复现，不声称训练时 dropout/RNG parity。

## Transformers parity

prompt：`The meaning of life is`，token ids 为
`[464, 3616, 286, 1204, 318]`（HF 0-based）。

| 检查点 | global max-abs | 冻结上限 |
| --- | ---: | ---: |
| token + position embedding | `0` | `0` |
| 12 block residual（逐层最大值的最大值） | `4.8828125e-4` | `5e-4` |
| final hidden | `7.05719e-5` | `1e-4` |
| full logits | `1.0681152e-4` | `1.5e-4` |

逐层 block max-abs 完整数组保存在 parity JSON，而不是只报告最终 argmax。

Tokenizer corpus 共 10 组，覆盖 ASCII、中文 Unicode、连续/尾随空白、LF/CRLF、
tab、NUL/控制字节、组合重音、emoji、多种 contraction 和
`<|endoftext|>`。所有 ids、token bytes、普通 decode 和 skip-special decode
均与冻结 HF reference 完全一致。

8-step greedy reference：

```text
The meaning of life is not the same as the meaning of death
```

生成 ids 为 `[407, 262, 976, 355, 262, 3616, 286, 1918]`（HF 0-based）。
full、dynamic cache 和 static cache 的 ids、停止位置、completion、完整文本
完全一致；三条路径的逐 step logits global max-abs 都是 `1.2207031e-4`。

## CPU 性能

条件：Intel Core Ultra 7 270K Plus，Julia 1.12.6，1 Julia thread，
12 BLAS threads，Float32；每组 decode 8 tokens，3 个 steady-state 原始样本。
load `6.986 s`，load 后 peak RSS `3.42 GB`，进程 peak RSS `3.66 GB`。

| Prompt | Full tok/s | Dynamic tok/s | Static tok/s | Dynamic speedup | Static speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 34.46 | 58.71 | 45.34 | 1.70× | 1.32× |
| 64 | 13.67 | 55.76 | 40.23 | 4.08× | 2.94× |
| 256 | 2.58 | 33.86 | 44.20 | 13.13× | 17.13× |

三档 prefill 与 full forward 的 max-abs 都为 0；decode 的全局 cache max-abs
为 `5.569458e-4`，所有 correctness 检查通过。static 为 264-token 固定容量；
dynamic cache 随实际长度增长，因此短 prompt 内存更省，长 prompt 下 static
布局在当前 CPU 实现中更快。

## CUDA 性能

条件：NVIDIA GeForce RTX 5080，CUDA driver/runtime 13.3，Float32，
3 个 steady-state 原始样本；参数传输 `2.597 s`，转移后 CUDA used memory
约 `495 MB`。计时通过 host logits materialization 包含同步边界。

| Prompt | Full tok/s | Dynamic tok/s | Static tok/s | Dynamic speedup | Static speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 160.46 | 352.13 | 329.42 | 2.19× | 2.05× |
| 64 | 82.95 | 339.24 | 339.46 | 4.09× | 4.09× |
| 256 | 23.26 | 269.92 | 321.67 | 11.61× | 13.83× |

三档 correctness 全部通过，decode global max-abs `6.1035156e-4`。过程中发现
GELU-New 不能只写成 host array 广播，也不能把 scalar method 限定为
`AbstractFloat`：前者使 CUDA fused Dense 产生 invalid IR，后者不接受
Reactant traced scalar。最终使用后端可追踪的通用 scalar formula，加
`AbstractArray` 广播入口，CUDA 和 XLA 均通过。

## Reactant-XLA

完成 GPT-2 同构缩小模型（vocab 31、hidden 16、4 heads、1 layer、MLP 64、
learned absolute position、GELU-New、全 bias/tied-head 语义）的 XLA CPU
prefill/decode smoke。4/4 tests 通过；prefill/decode 与 eager 在
`atol=rtol=1e-4` 内一致，固定 decode executable 和 position 递增均正常。

真实 124M XLA 性能不是 Week 10 Close 的必要结论，本次没有把缩小 smoke
外推为真实模型 XLA 吞吐。

## 原始记录 SHA256

- CPU JSON：`435aeb4726eecca21da59395c25b528bd3bc9b1655c70bf3d0653f598264fa8c`
- CUDA JSON：`a514bbc71f699c7e38c27ecd437da09c8569bcf6a3dab2ef8f8c06c5680d5589`
- parity JSON：`5e7c985e2a8a4d96633e818600fc3eeb27bc5a94e182e261dab15dd61a994181`
- persistent `reference.json`：`dea176e2926feda723ac7329abe582889824f71aaa0c558fc5d46ee2dde61965`
- persistent `reference.safetensors`：`d02cde726aa93ed074826079a6e3e585235e893e16ace64b0123c69c6f391527`

## 复现命令

```bash
/home/yj/projects/jwm/.venv/bin/python scripts/export_gpt2_reference.py \
  --model-dir /home/yj/models/huggingface/openai-community/gpt2/607a30d783dfa663caf39e06633721c8d4cfcd7e \
  --revision 607a30d783dfa663caf39e06633721c8d4cfcd7e \
  --output-dir /home/yj/models/huggingface/openai-community/gpt2/607a30d783dfa663caf39e06633721c8d4cfcd7e/lifeai_week10_reference \
  --steps 8

julia --project=. --startup-file=no scripts/verify_gpt2_parity.jl \
  /home/yj/models/huggingface/openai-community/gpt2/607a30d783dfa663caf39e06633721c8d4cfcd7e \
  /home/yj/models/huggingface/openai-community/gpt2/607a30d783dfa663caf39e06633721c8d4cfcd7e/lifeai_week10_reference \
  benchmark_results/week10/gpt2_124m_parity.json

LIFEAI_GPT2_REVISION=607a30d783dfa663caf39e06633721c8d4cfcd7e \
julia --project=. --startup-file=no scripts/benchmark_gpt2_inference.jl \
  /home/yj/models/huggingface/openai-community/gpt2/607a30d783dfa663caf39e06633721c8d4cfcd7e \
  benchmark_results/week10/gpt2_124m_cpu.json 3

LIFEAI_GPT2_REVISION=607a30d783dfa663caf39e06633721c8d4cfcd7e \
julia --project=. --startup-file=no scripts/benchmark_gpt2_cuda.jl \
  /home/yj/models/huggingface/openai-community/gpt2/607a30d783dfa663caf39e06633721c8d4cfcd7e \
  benchmark_results/week10/gpt2_124m_cuda.json 3
```
