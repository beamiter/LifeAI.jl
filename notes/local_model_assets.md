# 本地模型资产与持久化约定

大模型权重、下载缓存和真实 reference 不放在 `/tmp`，也不提交到
LifeAI.jl 仓库。Week 09—16 的历史 RTX 5080 资产机统一使用：

```text
/home/yj/models/huggingface/<organization>/<model>/<revision>/
```

每个 revision 自包含配置、tokenizer 和权重；LifeAI 生成的真实 reference 放在模型 revision 目录内的 `lifeai-references/<week-or-purpose>/`。仓库只保存生成/验证脚本、checksum、可重复命令和小型 benchmark 结果。

Week 17 的 RTX 4090 D 验证机用户目录不同，ModelScope 官方下载放在
`/home/ubuntu/models/modelscope/`；该例外在本页单独记录。无论下载源和
宿主路径如何，能否复用资产都以冻结 revision 的逐文件 SHA256 为准。

## 当前 Qwen3-0.6B 资产

固定 revision：

```text
c1899de289a04d12100db370d81485cdf75e47ca
```

模型目录：

```text
/home/yj/models/huggingface/Qwen/Qwen3-0.6B/c1899de289a04d12100db370d81485cdf75e47ca/
```

Week 09 sampled reference：

```text
/home/yj/models/huggingface/Qwen/Qwen3-0.6B/c1899de289a04d12100db370d81485cdf75e47ca/lifeai-references/week09-sampling/
```

Week 09 long-position RoPE reference：

```text
/home/yj/models/huggingface/Qwen/Qwen3-0.6B/c1899de289a04d12100db370d81485cdf75e47ca/lifeai-references/week09-rope/
```

### 文件校验和

| 文件 | SHA256 |
| --- | --- |
| `config.json` | `660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd` |
| `tokenizer.json` | `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` |
| `tokenizer_config.json` | `d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101` |
| `generation_config.json` | `2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2` |
| `model.safetensors` | `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b` |
| Week 09 `reference.json` | `b879c6f8203ec1d45134534b0f9f6185e6db0d78415f3989e757fc1a9caf64d1` |
| Week 09 `reference.safetensors` | `0d3d2ed57f7edcb820a376979489dbac27951d3714a3bf39c410d25b7c3d6581` |
| Week 09 RoPE `reference.json` | `2158095b305ced45cc7c9d03ddb0cb9f77d246fde9f66dbe1aa9d31062799fb5` |
| Week 09 RoPE `reference.safetensors` | `3e42d148d9553ff691751c02c306b4f8f12c687f4743cb9c443ad296af996c65` |

## 当前 Qwen3-1.7B 与 Qwen3-4B 资产（Week 12）

Week 12 下载了 1.7B 与 4B 的完整权重资产，目录遵循顶部约定：

```text
/home/yj/models/huggingface/Qwen/Qwen3-1.7B/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e/
/home/yj/models/huggingface/Qwen/Qwen3-4B/1cfa9a7208912126459214e8b04321603b3df60c/
```

Week 12 逐层 parity reference 位于各自 revision 目录内的
`lifeai-references/week12-parity/`。两个 revision 的
`tokenizer.json`、`tokenizer_config.json`、`generation_config.json` 与
0.6B 资产字节相同（SHA256 一致）。

### Qwen3-1.7B 文件校验和

| 文件 | SHA256 |
| --- | --- |
| `config.json` | `1ddb5b89ebc90dcb417a45c213d818577e65976454d29385c8f6140771d95197` |
| `model.safetensors.index.json` | `0d660e94b165eb912669a5249dff44b83188c4777a07ddb9611fb78d91b0578d` |
| `model-00001-of-00002.safetensors` | `169ad53ec313c3a34b06c0809216e4fc072cce444a5d4ff2b59690d064130ed5` |
| `model-00002-of-00002.safetensors` | `912becff8d60672aa8628ef08c05898d9adf17c2ad4ae3caf99b065622fdeff9` |

### Qwen3-4B 文件校验和

| 文件 | SHA256 |
| --- | --- |
| `config.json` | `8ba006f74fecfaaeb392872a60f4a480e7ec9860153d2e1b769ec81f9a147f8a` |
| `model.safetensors.index.json` | `6dc0981b8829fead746441f68f38f24c5ca4a3a66351f652c26c6df0efc43ab2` |
| `model-00001-of-00003.safetensors` | `328a91d3122359d5547f9d79521205bc0a46e1f79a792dfe650e99fc2d651223` |
| `model-00002-of-00003.safetensors` | `6cd087b316306a68c562436b5492edbcf6e16c6dba3a1308279caa5a58e21ca5` |
| `model-00003-of-00003.safetensors` | `e4bf436957184f4eeb86a80e9db394503f1f56446b2e6b7edeac5b81470f4ca1` |

同一 reference 的离线副本（含 parity 结果）位于
`test/fixtures/week12_qwen3_dense_real_weights/assets.json`。

### 恢复下载与 reference

```bash
/home/yj/projects/jwm/.venv/bin/hf download Qwen/Qwen3-1.7B \
  config.json tokenizer.json tokenizer_config.json generation_config.json \
  model.safetensors.index.json \
  model-00001-of-00002.safetensors model-00002-of-00002.safetensors \
  --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
  --local-dir /home/yj/models/huggingface/Qwen/Qwen3-1.7B/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e

/home/yj/projects/jwm/.venv/bin/hf download Qwen/Qwen3-4B \
  config.json tokenizer.json tokenizer_config.json generation_config.json \
  model.safetensors.index.json \
  model-00001-of-00003.safetensors model-00002-of-00003.safetensors \
  model-00003-of-00003.safetensors \
  --revision 1cfa9a7208912126459214e8b04321603b3df60c \
  --local-dir /home/yj/models/huggingface/Qwen/Qwen3-4B/1cfa9a7208912126459214e8b04321603b3df60c

# 各尺寸 reference 与 parity（MODEL_DIR 换成上面两个目录之一）
/home/yj/projects/jwm/.venv/bin/python scripts/export_qwen3_reference.py \
  MODEL_DIR MODEL_DIR/lifeai-references/week12-parity --revision REVISION

julia --project=. --startup-file=no scripts/verify_qwen3_parity.jl \
  MODEL_DIR MODEL_DIR/lifeai-references/week12-parity
```

## 当前 Qwen3-8B / 14B / 32B 资产（Week 13）

Week 13 下载了三个 untied 尺寸的完整权重资产：

```text
/home/yj/models/huggingface/Qwen/Qwen3-8B/b968826d9c46dd6066d109eabc6255188de91218/
/home/yj/models/huggingface/Qwen/Qwen3-14B/40c069824f4251a91eefaf281ebe4c544efd3e18/
/home/yj/models/huggingface/Qwen/Qwen3-32B/9216db5781bf21249d130ec9da846c4624c16137/
```

三个 revision 的 `tokenizer.json`、`tokenizer_config.json`、
`generation_config.json` 与 0.6B 资产字节相同；`config.json` SHA256 与
Week 11 冻结值一致。Week 13 逐层 parity reference 位于各自 revision 目录
的 `lifeai-references/week13-parity/`。全部权重分片 SHA256 的离线副本在
`test/fixtures/week13_qwen3_streamed_large_weights/assets.json`；下表只列
index 与首末分片，完整清单以 fixture 为准。

| 模型 | 文件 | SHA256 |
| --- | --- | --- |
| 8B | `model.safetensors.index.json` | `f9fdbcb91c23971c13ec5d5f2573d2349e8f61f2f049371ec699281748fdb1bc` |
| 8B | `model-00001-of-00005.safetensors` | `31d6a825ae35f11fb85b195b4c42c146c051e446433125a215336abdf95cbf5f` |
| 8B | `model-00005-of-00005.safetensors` | `20c2d6366ab85c90786ccdd829cd2b9e7d30ef3b2ebbb998280e7e4014b542ff` |
| 14B | `model.safetensors.index.json` | `62d7ad35757bae5e7baa452cb1483178b7daa50e869e923226b8da10871f7ebc` |
| 14B | `model-00001-of-00008.safetensors` | `e942bdbdf08857d16a8fef7d1dae9fceabeb4e84def6043485fe2f6f085dab0e` |
| 14B | `model-00008-of-00008.safetensors` | `0d6b92296e326d39bbbaeb32c3ec454ac606da843d4c8ffa8edf010b62b8c9e0` |
| 32B | `model.safetensors.index.json` | `bed42c6c55274bc08a1f616bceb3bcb84b3f02cb6584c573bd18c6519291ecd0` |
| 32B | `model-00001-of-00017.safetensors` | `52562b2ff97b61764260273e71bf5b4cf8a66f569399398f26dec0300fcf1316` |
| 32B | `model-00017-of-00017.safetensors` | `1f47c318fcd7797c0f85b4233cb754438b10e795b8bc874889090c416a94bd38` |

### 恢复下载与 reference（Week 13）

```bash
# 下载（8B 5 分片 / 14B 8 分片 / 32B 17 分片，文件名见各自 index）
/home/yj/projects/jwm/.venv/bin/hf download Qwen/Qwen3-8B \
  config.json tokenizer.json tokenizer_config.json generation_config.json \
  model.safetensors.index.json model-0000{1..5}-of-00005.safetensors \
  --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --local-dir /home/yj/models/huggingface/Qwen/Qwen3-8B/b968826d9c46dd6066d109eabc6255188de91218

# Float32 参数超出 RAM 的尺寸用 accelerate disk offload 生成 reference
/home/yj/projects/jwm/.venv/bin/python scripts/export_qwen3_reference.py \
  MODEL_DIR MODEL_DIR/lifeai-references/week13-parity --revision REVISION \
  --offload-dir MODEL_DIR/lifeai-references/tmp-offload-week13 \
  --max-cpu-memory 12GiB   # 用后删除 tmp-offload-week13

# Julia 侧流式逐层 parity（不加载完整参数树）
julia --project=. --startup-file=no scripts/verify_qwen3_streamed_parity.jl \
  MODEL_DIR MODEL_DIR/lifeai-references/week13-parity
```

注意：opt-in 测试时 `LIFEAI_QWEN3_1_7B/4B_MODEL_DIR`（全量加载）与
`LIFEAI_QWEN3_8B/14B/32B_MODEL_DIR`（流式）必须分两个 `Pkg.test` 进程
运行；五个 integration 同进程会因 Julia 堆跨 testset 累积被 OOM KILL。

## Week 14 BF16 reference 与验证

Week 14 的 HF BF16 逐层 reference（含 16 步 greedy token）位于四个
revision 目录的 `lifeai-references/week14-bf16-parity/`：

```bash
/home/yj/projects/jwm/.venv/bin/python scripts/export_qwen3_reference.py \
  MODEL_DIR MODEL_DIR/lifeai-references/week14-bf16-parity --revision REVISION \
  --compute-dtype bfloat16 --greedy-steps 16

julia --project=. --startup-file=no scripts/verify_qwen3_bf16_parity.jl \
  MODEL_DIR MODEL_DIR/lifeai-references/week14-bf16-parity
```

BF16 opt-in 进程协议：`LIFEAI_QWEN3_BF16_0_6B/1_7B/4B_MODEL_DIR` 可在
一次 `Pkg.test` 内同进程运行；8B（参数树 15.3 GiB，峰值 ≈ 19 GiB）
连"完整套件 + 8B integration"都可能在共享内存的机器上越界，必须用
独立的测试文件进程：

```bash
LIFEAI_QWEN3_BF16_8B_MODEL_DIR=/home/yj/models/huggingface/Qwen/Qwen3-8B/b968826d9c46dd6066d109eabc6255188de91218 \
julia --project=. --startup-file=no --heap-size-hint=2G -e \
  'using Test, JSON3, Lux; import LifeAI; include("test/test_week14.jl")'
```

`--heap-size-hint=2G` 必需：它强制激进 GC，把峰值 RSS 压到 ≈ 18.5 GiB
（实测 125/125 通过）；不带 hint 时 GC 滞后会在共享内存机器上触发
OOM KILL。

## Week 15 CUDA / XLA BF16 验证

复用 Week 14 的 `week14-bf16-parity` reference（不重复导出）。注意加载
顺序：**脚本必须先 `using LuxCUDA` / `using Reactant` 再 `using
LifeAI`**，否则 cuDNN 初始化失败。

```bash
# CUDA eager（0.6B/1.7B/4B；8B BF16 树 15.3 GiB 超出 16.3 GiB VRAM）
julia --project=. --startup-file=no scripts/verify_qwen3_bf16_cuda.jl \
  MODEL_DIR MODEL_DIR/lifeai-references/week14-bf16-parity

# Reactant XLA BF16 编译 prefill（0.6B）
julia --project=. --startup-file=no scripts/verify_qwen3_bf16_xla.jl \
  MODEL_DIR MODEL_DIR/lifeai-references/week14-bf16-parity
```

CUDA opt-in 测试（`Pkg.test` 内可三尺寸同进程，累计 GPU 树 ≈ 11.9 GiB）：
`LIFEAI_QWEN3_BF16_CUDA_0_6B/1_7B/4B_MODEL_DIR`。

## Week 16 XLA compiled decode 与量化验证

14B 的 HF BF16 reference（offload 导出）位于
`Qwen3-14B/<revision>/lifeai-references/week16-bf16-parity/`。

```bash
# XLA BF16 static-cache compiled decode（0.6B，含设备端 greedy 快路径）
julia --project=. --startup-file=no scripts/verify_qwen3_bf16_xla_decode.jl \
  MODEL_DIR MODEL_DIR/lifeai-references/week14-bf16-parity

# 量化 GPU 驻留验证（8B 用 int8 + week14 reference；14B 用 int4 + week16 reference）
julia --project=. --startup-file=no --heap-size-hint=3G scripts/verify_qwen3_quant_cuda.jl \
  MODEL_DIR REFERENCE_DIR int8|int4
```

量化验证是 GPU 独占任务，且宿主内存紧张时（本机常有其他负载）14B 加载
可能被 OOM KILL——重试前用 `free -g` 确认 ≥ 15 GiB 可用。

## Week 17 RTX 4090 D / Qwen3-14B 量化验证

验证机：

```text
GPU: NVIDIA GeForce RTX 4090 D
driver: 570.153.02
CUDA.jl: 6.2.1
visible VRAM: 25,238,568,960 bytes
```

ModelScope 官方仓库下载目录：

```text
/home/ubuntu/models/modelscope/Qwen/Qwen3-14B/
```

8 个权重分片合计 `29,536,665,640` bytes；配置、tokenizer、index 与全部
分片 SHA256 均和 HuggingFace 冻结 revision
`40c069824f4251a91eefaf281ebe4c544efd3e18` 一致。完整期望 checksum 继续
以 `test/fixtures/week13_qwen3_streamed_large_weights/assets.json` 为准，
不为下载源复制第二份清单。

BF16 reference：

```text
/home/ubuntu/models/modelscope/Qwen/Qwen3-14B/lifeai-references/week17-bf16-parity/
reference.json        35755f087cd09313c5e2cffd80bb49bc8adee18b8845f2aa8546b01e7e1b3294
reference.safetensors e76e8bb6a782c3bc4a4db2bca8f250230d4460a1b70c04d81796ecb3278dde5c
```

生成环境为 Transformers 4.51.0 / Torch 2.7.1+cpu：

```bash
python scripts/export_qwen3_reference.py \
  /home/ubuntu/models/modelscope/Qwen/Qwen3-14B \
  /home/ubuntu/models/modelscope/Qwen/Qwen3-14B/lifeai-references/week17-bf16-parity \
  --revision 40c069824f4251a91eefaf281ebe4c544efd3e18 \
  --compute-dtype bfloat16 --greedy-steps 16
```

三组 GPU 对照：

```bash
MODEL_DIR=/home/ubuntu/models/modelscope/Qwen/Qwen3-14B
REFERENCE_DIR=$MODEL_DIR/lifeai-references/week17-bf16-parity

# 全 INT8：16/16 greedy；tensor tree 14.487 GiB，但仅余约 54 MiB VRAM
julia --threads=auto --project=. --startup-file=no --heap-size-hint=3G \
  scripts/verify_qwen3_quant_cuda.jl "$MODEL_DIR" "$REFERENCE_DIR" int8

# mixed MSE：4/16；full-logits error 低于同布局 RTN，但序列 fidelity 更差
julia --threads=auto --project=. --startup-file=no --heap-size-hint=3G \
  scripts/verify_qwen3_quant_cuda.jl "$MODEL_DIR" "$REFERENCE_DIR" int4 \
  test/fixtures/week17_qwen3_calibrated_int4/plan_mixed_24g.json

# 同布局 mixed RTN：16/16；tensor tree 12.093 GiB
julia --threads=auto --project=. --startup-file=no --heap-size-hint=3G \
  scripts/verify_qwen3_quant_cuda.jl "$MODEL_DIR" "$REFERENCE_DIR" int4 \
  test/fixtures/week17_qwen3_calibrated_int4/plan_mixed_24g_rtn.json
```

精确 bytes、load/cold/warm、误差与 greedy 指标冻结在
`test/fixtures/week17_qwen3_calibrated_int4/assets.json`。三组必须用独立
Julia 进程，避免 CUDA allocator 的跨组状态污染；全 INT8 是极限驻留证据，
不是有安全余量的部署配置。

## Week 11 Qwen3 dense family config reference

Week 11 冻结全部六个官方 config contract；Week 12 下载 1.7B/4B 权重，
Week 13 下载 8B/14B/32B 权重，至此六个尺寸的完整资产均在本地。六个
config 的不可变 revision 与 SHA256 为：

| model | revision | `config.json` SHA256 |
| --- | --- | --- |
| Qwen3-0.6B | `c1899de289a04d12100db370d81485cdf75e47ca` | `660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd` |
| Qwen3-1.7B | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | `1ddb5b89ebc90dcb417a45c213d818577e65976454d29385c8f6140771d95197` |
| Qwen3-4B | `1cfa9a7208912126459214e8b04321603b3df60c` | `8ba006f74fecfaaeb392872a60f4a480e7ec9860153d2e1b769ec81f9a147f8a` |
| Qwen3-8B | `b968826d9c46dd6066d109eabc6255188de91218` | `f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30` |
| Qwen3-14B | `40c069824f4251a91eefaf281ebe4c544efd3e18` | `e73c3664ca09b10a673fef0c22e8a6b456201d49bd4713c9691f775720e8857a` |
| Qwen3-32B | `9216db5781bf21249d130ec9da846c4624c16137` | `97e295b63283935788fac5e4f8860862a56d4089538cafc93f0431f2ebe483bb` |

同一 reference 的小型离线副本位于
`test/fixtures/week11_qwen3_dense_family/specs.json`。未来若下载其他尺寸权重，
仍应遵守本页顶部的持久目录布局，并为真实逐层 reference 单独记录模型权重和
分片 index checksum。

## 恢复下载

使用 `/home/yj/projects/jwm/.venv` 中的 HuggingFace CLI，下载目标 revision 的五个必要文件：

```bash
/home/yj/projects/jwm/.venv/bin/hf download Qwen/Qwen3-0.6B \
  config.json tokenizer.json tokenizer_config.json generation_config.json model.safetensors \
  --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --local-dir /home/yj/models/huggingface/Qwen/Qwen3-0.6B/c1899de289a04d12100db370d81485cdf75e47ca
```

`--local-dir` 内的 `.cache/huggingface/` 保存下载元数据和断点续传状态；不要把该目录移动到 `/tmp`。

## Week 09 reference 与验证

```bash
/home/yj/projects/jwm/.venv/bin/python scripts/export_qwen3_sampling_reference.py \
  --model-dir /home/yj/models/huggingface/Qwen/Qwen3-0.6B/c1899de289a04d12100db370d81485cdf75e47ca \
  --output-dir /home/yj/models/huggingface/Qwen/Qwen3-0.6B/c1899de289a04d12100db370d81485cdf75e47ca/lifeai-references/week09-sampling \
  --revision c1899de289a04d12100db370d81485cdf75e47ca

julia --project=. --startup-file=no scripts/verify_qwen3_sampling_parity.jl \
  /home/yj/models/huggingface/Qwen/Qwen3-0.6B/c1899de289a04d12100db370d81485cdf75e47ca \
  /home/yj/models/huggingface/Qwen/Qwen3-0.6B/c1899de289a04d12100db370d81485cdf75e47ca/lifeai-references/week09-sampling

/home/yj/projects/jwm/.venv/bin/python scripts/export_qwen3_rope_reference.py \
  --model-dir /home/yj/models/huggingface/Qwen/Qwen3-0.6B/c1899de289a04d12100db370d81485cdf75e47ca \
  --output-dir /home/yj/models/huggingface/Qwen/Qwen3-0.6B/c1899de289a04d12100db370d81485cdf75e47ca/lifeai-references/week09-rope \
  --revision c1899de289a04d12100db370d81485cdf75e47ca
```

仓库内 `test/fixtures/week09_qwen3_rope/` 保留上述 RoPE reference 的同 checksum 小型副本，使 position 0/2048/32767/40959 的独立 Transformers 对照可进入默认离线测试；它不是模型权重或下载缓存。

## 当前 GPT-2 124M 资产

模型与 immutable revision：

```text
openai-community/gpt2
607a30d783dfa663caf39e06633721c8d4cfcd7e
```

模型目录：

```text
/home/yj/models/huggingface/openai-community/gpt2/607a30d783dfa663caf39e06633721c8d4cfcd7e/
```

Week 10 Transformers reference：

```text
/home/yj/models/huggingface/openai-community/gpt2/607a30d783dfa663caf39e06633721c8d4cfcd7e/lifeai_week10_reference/
```

文件与 reference SHA256 见
[`benchmark_results/week10/summary.md`](../benchmark_results/week10/summary.md)；
`load_hf_gpt2_bundle` 也内置同一 revision/checksum 契约并默认 fail closed。

### 恢复下载与 reference

```bash
/home/yj/projects/jwm/.venv/bin/hf download openai-community/gpt2 \
  config.json generation_config.json tokenizer.json tokenizer_config.json \
  vocab.json merges.txt model.safetensors \
  --revision 607a30d783dfa663caf39e06633721c8d4cfcd7e \
  --local-dir /home/yj/models/huggingface/openai-community/gpt2/607a30d783dfa663caf39e06633721c8d4cfcd7e

/home/yj/projects/jwm/.venv/bin/python scripts/export_gpt2_reference.py \
  --model-dir /home/yj/models/huggingface/openai-community/gpt2/607a30d783dfa663caf39e06633721c8d4cfcd7e \
  --revision 607a30d783dfa663caf39e06633721c8d4cfcd7e \
  --output-dir /home/yj/models/huggingface/openai-community/gpt2/607a30d783dfa663caf39e06633721c8d4cfcd7e/lifeai_week10_reference \
  --steps 8

julia --project=. --startup-file=no scripts/verify_gpt2_parity.jl \
  /home/yj/models/huggingface/openai-community/gpt2/607a30d783dfa663caf39e06633721c8d4cfcd7e \
  /home/yj/models/huggingface/openai-community/gpt2/607a30d783dfa663caf39e06633721c8d4cfcd7e/lifeai_week10_reference \
  benchmark_results/week10/gpt2_124m_parity.json
```

## 维护边界

- `/home/yj/models/` 是本机持久资产目录，不是 Git 仓库的一部分；系统备份策略需要单独覆盖它。
- 每个实验必须记录 model id、完整 revision、文件 checksum、reference 环境版本和计算 dtype。
- 删除或替换某个 revision 前，先确认没有 weekly reference、benchmark 或 checkpoint 指向它。
- 仓库内的 `artifacts/` 与该模型目录没有关系；保持用户已有内容不受自动 staging 影响。
