# LifeAI.jl test suite

测试目录与 `notes/episodes/` 使用相同的 Episode / Chapter 层级。一个 Chapter
产生的测试与冻结 fixture 放在同名目录中，因此可以从研发记录直接定位到验收证据：

```text
test/
├── runtests.jl
├── support/                       # 跨 Chapter 复用的测试构造器与资产定位工具
└── episodes/
    ├── episode01_transformer_and_training_foundations/
    │   ├── chapter01_transformer/
    │   └── ...
    └── episode06_qwen3_moe_and_model_expansion/
        ├── chapter24_qwen3_moe_architecture/
        ├── chapter25_qwen3_moe_gpu_offload/
        ├── chapter26_qwen3_moe_expert_cache/
        ├── chapter27_qwen3_moe_layer_balanced_cache/
        ├── chapter28_qwen3_moe_scattered_cache/
        ├── chapter29_qwen3_moe_scattered_reuse/
        └── chapter30_qwen3_moe_async_miss_pipeline/
```

## Chapter 索引

| Episode | Chapter | 主要测试内容 |
| --- | --- | --- |
| 01 | 01 | manual attention、MHA、RoPE、TransformerBlock |
| 01 | 02 | GPT、Tokenizer、DatasetLoader、训练、KV cache、XLA cache |
| 01 | 03 | 可复现训练、评估、checkpoint 与断点续训 |
| 01 | 04 | RMSNorm、SwiGLU、tied embedding 与现代 GPT 配置 |
| 01 | 05 | versioned tokenizer、中文数据管线、随机 UTF-8 与 XLA |
| 02 | 06 | Qwen3 GQA、QK-Norm 与 XLA |
| 02 | 07 | Hugging Face config、safetensors、权重映射与 XLA |
| 02 | 08 | Qwen3 tokenizer、chat/text generation 与 XLA |
| 02 | 09 | sampling、长位置 RoPE、cache correctness 与推理 fidelity |
| 03 | 10 | GPT-2 架构、Hugging Face parity 与 XLA |
| 03 | 11 | Qwen3 dense family contract 与 XLA |
| 03 | 12 | dense family 真实权重 parity |
| 03 | 13 | 8B/14B/32B streamed parity |
| 04 | 14 | native BF16 mixed-precision compute |
| 04 | 15 | BF16 CUDA/XLA acceleration |
| 04 | 16 | XLA decode、INT8/INT4 与量化 round-trip |
| 04 | 17 | calibrated INT4 与预算化混合精度 |
| 04 | 18 | activation-aware INT4 calibration |
| 05 | 19 | Qwen3-8B CUDA 日常部署 |
| 05 | 20 | Qwen3-8B XLA single-residency 部署 |
| 05 | 21 | XLA resident HTTP service |
| 05 | 22 | Qwen3 Embedding 与 exact semantic memory |
| 05 | 23 | XLA device-resident sampling |
| 06 | 24 | Qwen3 MoE router、Float32/BF16 streaming、CPU/XLA/CUDA sparse dispatch、真实资产与 30B parity |
| 06 | 25 | 30B-A3B resident/offload 内存预算、全局→活跃 expert 路由映射与 GPU session contract |
| 06 | 26 | byte-budgeted device expert LRU、跨请求命中、显式清空与真实 30B I/O/latency 证据 |
| 06 | 27 | scan-resistant layer-balanced LRU、运行时预算重配与自然文本 cache sweep |
| 06 | 28 | CUDA scattered pointer-table dispatch、active tensor materialization/GC 消除 |
| 06 | 29 | scattered pointer-table/workspace 有界复用、cache clear 失效与长期命中生命周期 |
| 06 | 30 | bounded parallel safetensors reads、CUDA pinned async upload 与真实 30B miss benchmark |

`fixtures/` 只存在于拥有该证据的 Chapter 目录中。跨 Chapter 复用 fixture 时，
测试显式引用其原始 Chapter，不复制第二份数据。`support/` 只存放无独立产品能力
归属的构造器和定位工具。

## 运行

默认离线测试：

```bash
julia --project=. test/runtests.jl
```

启用 Reactant/XLA 专项：

```bash
LIFEAI_TEST_XLA=true julia --project=. test/runtests.jl
```

启用 CUDA sparse accelerator 专项：

```bash
LIFEAI_TEST_CUDA=true julia --project=. test/runtests.jl
```

`runtests.jl` 的 testset 输出同样按 Episode / Chapter 分组；真实权重 integration
仍由各 Chapter 原有的显式环境变量控制，默认测试不会联网。
