# 本地模型资产与持久化约定

大模型权重、下载缓存和真实 reference 不放在 `/tmp`，也不提交到
LifeAI.jl 仓库。Chapter 09—16 的历史 RTX 5080 资产机统一使用：

```text
/home/yj/models/huggingface/<organization>/<model>/<revision>/
```

每个 revision 自包含配置、tokenizer 和权重；LifeAI 生成的真实 reference 放在模型 revision 目录内的 `lifeai-references/<week-or-purpose>/`。仓库只保存生成/验证脚本、checksum、可重复命令和小型 benchmark 结果。

Chapter 17 的 RTX 4090 D 验证机用户目录不同，ModelScope 官方下载放在
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

Chapter 09 sampled reference：

```text
/home/yj/models/huggingface/Qwen/Qwen3-0.6B/c1899de289a04d12100db370d81485cdf75e47ca/lifeai-references/week09-sampling/
```

Chapter 09 long-position RoPE reference：

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
| Chapter 09 `reference.json` | `b879c6f8203ec1d45134534b0f9f6185e6db0d78415f3989e757fc1a9caf64d1` |
| Chapter 09 `reference.safetensors` | `0d3d2ed57f7edcb820a376979489dbac27951d3714a3bf39c410d25b7c3d6581` |
| Chapter 09 RoPE `reference.json` | `2158095b305ced45cc7c9d03ddb0cb9f77d246fde9f66dbe1aa9d31062799fb5` |
| Chapter 09 RoPE `reference.safetensors` | `3e42d148d9553ff691751c02c306b4f8f12c687f4743cb9c443ad296af996c65` |

## 当前 Qwen3-1.7B 与 Qwen3-4B 资产（Chapter 12）

Chapter 12 下载了 1.7B 与 4B 的完整权重资产，目录遵循顶部约定：

```text
/home/yj/models/huggingface/Qwen/Qwen3-1.7B/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e/
/home/yj/models/huggingface/Qwen/Qwen3-4B/1cfa9a7208912126459214e8b04321603b3df60c/
```

Chapter 12 逐层 parity reference 位于各自 revision 目录内的
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
`test/episodes/episode03_model_family_and_large_weights/chapter12_qwen3_dense_real_weights/fixtures/qwen3_dense_real_weights/assets.json`。

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

## 当前 Qwen3-8B / 14B / 32B 资产（Chapter 13）

Chapter 13 下载了三个 untied 尺寸的完整权重资产：

```text
/home/yj/models/huggingface/Qwen/Qwen3-8B/b968826d9c46dd6066d109eabc6255188de91218/
/home/yj/models/huggingface/Qwen/Qwen3-14B/40c069824f4251a91eefaf281ebe4c544efd3e18/
/home/yj/models/huggingface/Qwen/Qwen3-32B/9216db5781bf21249d130ec9da846c4624c16137/
```

三个 revision 的 `tokenizer.json`、`tokenizer_config.json`、
`generation_config.json` 与 0.6B 资产字节相同；`config.json` SHA256 与
Chapter 11 冻结值一致。Chapter 13 逐层 parity reference 位于各自 revision 目录
的 `lifeai-references/week13-parity/`。全部权重分片 SHA256 的离线副本在
`test/episodes/episode03_model_family_and_large_weights/chapter13_qwen3_streamed_large_weights/fixtures/qwen3_streamed_large_weights/assets.json`；下表只列
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

### 恢复下载与 reference（Chapter 13）

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

## Chapter 14 BF16 reference 与验证

Chapter 14 的 HF BF16 逐层 reference（含 16 步 greedy token）位于四个
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
  'using Test, JSON3, Lux; import LifeAI; include("test/episodes/episode04_efficient_inference_and_quantization/chapter14_qwen3_bf16_compute/test_qwen3_bf16_compute.jl")'
```

`--heap-size-hint=2G` 必需：它强制激进 GC，把峰值 RSS 压到 ≈ 18.5 GiB
（实测 125/125 通过）；不带 hint 时 GC 滞后会在共享内存机器上触发
OOM KILL。

## Chapter 15 CUDA / XLA BF16 验证

复用 Chapter 14 的 `week14-bf16-parity` reference（不重复导出）。注意加载
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

## Chapter 16 XLA compiled decode 与量化验证

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

## Chapter 17 RTX 4090 D / Qwen3-14B 量化验证

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
以 `test/episodes/episode03_model_family_and_large_weights/chapter13_qwen3_streamed_large_weights/fixtures/qwen3_streamed_large_weights/assets.json` 为准，
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
  test/episodes/episode04_efficient_inference_and_quantization/chapter17_qwen3_calibrated_int4/fixtures/qwen3_calibrated_int4/plan_mixed_24g.json

# 同布局 mixed RTN：16/16；tensor tree 12.093 GiB
julia --threads=auto --project=. --startup-file=no --heap-size-hint=3G \
  scripts/verify_qwen3_quant_cuda.jl "$MODEL_DIR" "$REFERENCE_DIR" int4 \
  test/episodes/episode04_efficient_inference_and_quantization/chapter17_qwen3_calibrated_int4/fixtures/qwen3_calibrated_int4/plan_mixed_24g_rtn.json
```

精确 bytes、load/cold/warm、误差与 greedy 指标冻结在
`test/episodes/episode04_efficient_inference_and_quantization/chapter17_qwen3_calibrated_int4/fixtures/qwen3_calibrated_int4/assets.json`。三组必须用独立
Julia 进程，避免 CUDA allocator 的跨组状态污染；全 INT8 是极限驻留证据，
不是有安全余量的部署配置。

## Chapter 18 activation-aware INT4 验证

Chapter 18 复用上述 Qwen3-14B 权重和 BF16 reference，不新增模型下载。
独立校准 token 由仓库自有的多语种/代码/数学短语生成，不包含 Chapter 17
evaluation token sequence：

```bash
/tmp/lifeai-week17-venv/bin/python \
  scripts/export_qwen3_calibration_tokens.py \
  --revision 40c069824f4251a91eefaf281ebe4c544efd3e18 \
  /home/ubuntu/models/modelscope/Qwen/Qwen3-14B \
  test/episodes/episode04_efficient_inference_and_quantization/chapter18_qwen3_activation_calibration/fixtures/qwen3_activation_aware_int4/calibration_tokens.json
```

fixture SHA256 为
`0c61c44e8dcebf0eaea75ff320ebdf9f9ab377f20276f5430abf54c654cbb6f2`，
包含 8×32=256 tokens。GPU 校准和验证命令：

```bash
MODEL_DIR=/home/ubuntu/models/modelscope/Qwen/Qwen3-14B
REFERENCE_DIR=$MODEL_DIR/lifeai-references/week17-bf16-parity

julia --project=. scripts/verify_qwen3_quant_cuda.jl \
  "$MODEL_DIR" "$REFERENCE_DIR" int4 \
  test/episodes/episode04_efficient_inference_and_quantization/chapter18_qwen3_activation_calibration/fixtures/qwen3_activation_aware_int4/plan_mixed_24g_activation.json \
  test/episodes/episode04_efficient_inference_and_quantization/chapter18_qwen3_activation_calibration/fixtures/qwen3_activation_aware_int4/calibration_tokens.json
```

实测 calibration `248.32 s`、load `489.23 s`、tree `12.093 GiB`、VRAM
`21.474 GiB`；logits max/mean error `3.4238 / 0.42865`，greedy 4/16，
首次分歧第 5 token。精确 provenance 与指标位于
`test/episodes/episode04_efficient_inference_and_quantization/chapter18_qwen3_activation_calibration/fixtures/qwen3_activation_aware_int4/assets.json`。这是
diagonal activation second-moment clipping 的负结果，不代表完整
AWQ/GPTQ 已验证。

## Chapter 11 Qwen3 dense family config reference

Chapter 11 冻结全部六个官方 config contract；Chapter 12 下载 1.7B/4B 权重，
Chapter 13 下载 8B/14B/32B 权重，至此六个尺寸的完整资产均在本地。六个
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
`test/episodes/episode03_model_family_and_large_weights/chapter11_qwen3_dense_family/fixtures/qwen3_dense_family/specs.json`。未来若下载其他尺寸权重，
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

## Chapter 19 RTX 4090 D / Qwen3-8B 日常部署

Chapter 19 复用 Chapter 13 冻结的官方 revision：

```text
model_id: Qwen/Qwen3-8B
revision: b968826d9c46dd6066d109eabc6255188de91218
recommended runtime path:
  /home/ubuntu/models/modelscope/Qwen/Qwen3-8B
```

Hugging Face 原始恢复命令：

```bash
/home/ubuntu/.local/bin/hf download Qwen/Qwen3-8B \
  --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --local-dir /home/ubuntu/models/huggingface/Qwen/Qwen3-8B/b968826d9c46dd6066d109eabc6255188de91218 \
  --max-workers 2
```

当前验证机从 ModelScope 官方 Qwen 镜像恢复，以改善下载连通性：

```bash
MODELSCOPE_DOWNLOAD_PARALLEL_WORKERS=8 \
MODELSCOPE_DOWNLOAD_TIMEOUT=600 \
modelscope download Qwen/Qwen3-8B \
  --revision master \
  --local-dir /home/ubuntu/models/modelscope/Qwen/Qwen3-8B \
  --max-workers 5
```

ModelScope 的 `master` 不是 provenance 身份。下载后必须逐文件匹配 Chapter 13
冻结的 Hugging Face revision checksum；任一权重、config 或 tokenizer
不匹配，都不能运行 deployment benchmark。

日常 profile：

```text
configs/deployment/qwen3_8b_4090d_bf16_daily.json
SHA256 93e7bde699fad4f0c93153e8d1c0458326c1ba848127cc14758fff066d944e4b
asset manifest configs/deployment/qwen3_8b_frozen_assets.json
asset manifest SHA256 f4737c1aca92b3cbf046da7861af88fc2d4650552397b7d6f4b7edade5040e91
context 4096 = prompt/history 3584 + output 512
prefill chunk 64
prefill explicit full GC + CUDA.reclaim every chunk
decode explicit incremental GC + CUDA.reclaim every 8 tokens
workspace reserve 5 GiB
```

`CUDA.reclaim()` 在当前 CUDA.jl 中自身会执行 full GC、同步、清理
task-local library state 和 pool trim；上述 full/incremental 指调用
`CUDA.reclaim()` 之前的显式 GC pass，不表示 decode 路径没有 full GC。

启动与硬件验收：

```bash
julia --project=. --startup-file=no \
  scripts/run_qwen3_cuda_chat.jl \
  /home/ubuntu/models/modelscope/Qwen/Qwen3-8B

julia --project=. --startup-file=no \
  scripts/benchmark_qwen3_cuda_deployment.jl \
  /home/ubuntu/models/modelscope/Qwen/Qwen3-8B \
  configs/deployment/qwen3_8b_4090d_bf16_daily.json \
  benchmark_results/week19/qwen3_8b_4090d_bf16_daily.json
```

2026-07-30 的 RTX 4090 D 验收已通过：3,584-token prefill `46.849 s`，
3,584+512 整窗 decode `10.246 tok/s`，CUDA 内部最低 free `2.190 GiB`，
200 ms `nvidia-smi` 物理采样最低 free `2,099 MiB`。本地结果文件 SHA256
为 `5c01dd5e218167778255b71f2b1053e153cbef279b60806e14ab5911fe2fa0c2`；
2,856 行外部采样 CSV 的 SHA256 为
`018d873a77e6d19ca6727fb9b2fdd00801e5cd4a6b6de10ddb0ec75be530555d`。

权重不进入 Git。Git 提交保存 profile、checksum contract、脚本和文档化
结果；原始 benchmark/采样保存在本地被忽略的 `benchmark_results/`
目录，并用上述 SHA256 标识。

`--local-dir` 内的 `.cache/huggingface/` 保存下载元数据和断点续传状态；不要把该目录移动到 `/tmp`。

## Chapter 09 reference 与验证

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

仓库内 `test/episodes/episode02_qwen3_end_to_end_parity/chapter09_qwen3_sampling_performance/fixtures/qwen3_rope_long_context/` 保留上述 RoPE reference 的同 checksum 小型副本，使 position 0/2048/32767/40959 的独立 Transformers 对照可进入默认离线测试；它不是模型权重或下载缓存。

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

Chapter 10 Transformers reference：

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
- 删除或替换某个 revision 前，先确认没有 Chapter reference、benchmark 或 checkpoint 指向它。
- 仓库内的 `artifacts/` 与该模型目录没有关系；保持用户已有内容不受自动 staging 影响。

## Chapter 20 RTX 4090 D / Qwen3-8B XLA 单驻留部署

Chapter 20 继续使用 Chapter 19 已校验的同一份本地 Qwen3-8B 资产：

```text
model_id: Qwen/Qwen3-8B
revision: b968826d9c46dd6066d109eabc6255188de91218
model directory:
  /home/ubuntu/models/modelscope/Qwen/Qwen3-8B
verified files: 10
verified bytes: 16,392,983,007
```

资产的下载/provenance 边界与 Chapter 19 相同：ModelScope `master` 只作为
下载通道，身份仍由冻结的 Hugging Face revision 和逐文件 SHA256 决定。
任一文件不匹配时，reference、benchmark 和默认 CLI 都应 fail closed。

### 冻结配置与结果摘要

```text
XLA daily profile:
  configs/deployment/qwen3_8b_4090d_bf16_xla_daily.json
profile SHA256:
  0638eecce7864d261770c8af1698575f055cf3149d6e5d70605c7cf35dbb8d01
asset manifest:
  configs/deployment/qwen3_8b_frozen_assets.json
asset manifest SHA256:
  f4737c1aca92b3cbf046da7861af88fc2d4650552397b7d6f4b7edade5040e91

context:
  4096 = prompt/history 3584 + output 512
prefill chunk:
  64
decode strategy:
  greedy
XLA_REACTANT_GPU_MEM_FRACTION:
  0.87
XLA_REACTANT_GPU_PREALLOCATE:
  false
```

最终验收使用 Reactant `0.2.275`。compact Qwen3-8B tree 为 291 个 tensor
leaves、16,381,470,720 logical bytes；设备上只有这一棵参数树且只 transfer
一次。4K BF16 KV 为 603,979,776 bytes。RTX 4090 D 实测：

- host load / tree transfer：`108.445 / 3.266 s`；
- prefill / decode compile：`71.696 / 25.204 s`；
- ready wall / ready free：`213.219 s / 6,909 MiB`；
- 7 个 steady 64-token request 中位 prefill：`0.02733 s`；
- steady decode 中位数：`40.421 tok/s`；
- 3,584-token prefill：`1.49764 s`；
- 3,584+512 wall / decode：`13.921 s / 41.132 tok/s`；
- logical sequence / cache：`4,096 / 4,095`；
- 8 次复用请求 allocator drift：`234,752 bytes`；
- 200 ms 连续 `nvidia-smi` 共 1,225 点，最低 free `2,301 MiB`。

三个 CUDA BF16 oracle case 分别覆盖 64-token 单 chunk、65-token
左补齐到 bucket 128，以及 3,584-token 最大 prompt；每组冻结 32 个
generated tokens，XLA 合计 **96 / 96** 完全一致。

### 可确定性 CUDA BF16 oracle

先在独立 CUDA eager 进程导出 reference：

```bash
julia --project=. --startup-file=no \
  scripts/export_qwen3_8b_greedy_reference.jl \
  /home/ubuntu/models/modelscope/Qwen/Qwen3-8B \
  configs/deployment/qwen3_8b_4090d_bf16_xla_daily.json \
  benchmark_results/week20/qwen3_8b_cuda_greedy_reference.json
```

reference 使用 schema 2，只冻结 provenance、prompt 与 generated token
payload，不写入与运行时钟有关的 elapsed 字段，因此相同资产和代码可确定
性再生成：

```text
benchmark_results/week20/qwen3_8b_cuda_greedy_reference.json
SHA256 83f62afbbb470b695b6990a3b86a8860407a37874354d6b039e1ce19917e2747
```

XLA benchmark 源码钉住这个 exact digest，并继续核对 schema/source、
model/revision、profile/manifest digest、资产数和三个 case shape；不能用
任意 caller-supplied JSON 替代 oracle。

### 最终 benchmark

显式复现最终 allocator 配置：

```bash
XLA_REACTANT_GPU_MEM_FRACTION=0.87 \
XLA_REACTANT_GPU_PREALLOCATE=false \
julia --project=. --startup-file=no \
  scripts/benchmark_qwen3_xla_deployment.jl \
  /home/ubuntu/models/modelscope/Qwen/Qwen3-8B \
  configs/deployment/qwen3_8b_4090d_bf16_xla_daily.json \
  benchmark_results/week20/qwen3_8b_cuda_greedy_reference.json \
  benchmark_results/week20/qwen3_8b_4090d_bf16_xla_daily.json
```

脚本在资产校验后自行启动 GPU 0 的 200 ms `nvidia-smi` monitor，覆盖
XLA load、compile、8 次复用短请求、65-token padding case 与
3,584+512 整窗。最终两个结果文件为：

```text
benchmark_results/week20/qwen3_8b_4090d_bf16_xla_daily.json
SHA256 075dc76a023a8143213f640bfb354d6e38c10b5747b5ef3e8c7e6baeb1c730dc

benchmark_results/week20/qwen3_8b_4090d_bf16_xla_daily_nvidia_smi.csv
SHA256 72fd3fe80b56647714604a85499a3a9d8e5833412f7a50864d8ea1aa6588b586
```

JSON 的全部 acceptance 为 true，顶层 `closed=true`。CSV 连续采样最大
used 为 21,769 MiB、最低 free 为 2,301 MiB；后者是
2,412,773,376 bytes（2.247 GiB），通过冻结的 2 GiB 门槛。

显存 fraction 的失败过程同样保留：

- `0.95`：3,215 个外部样本最低 free 仅 330 MiB；旧版 JSON 虽在尚未
  纳入物理 trace 的门禁下 nominal `closed=true`，但不能作为日常验收。
- `0.89`：1,517 个外部样本最低 free 为 1,848 MiB，仍未达到 2 GiB。
- `0.87`：性能门槛保持通过，连续最低 free 提升到 2,301 MiB，最终关闭。

### 日常聊天入口

默认 CLI 已内置 `0.87 / preallocate=false`，也可由环境显式覆盖：

```bash
julia --project=. --startup-file=no \
  scripts/run_qwen3_xla_chat.jl \
  /home/ubuntu/models/modelscope/Qwen/Qwen3-8B
```

指定 frozen profile 的一次性 smoke：

```bash
julia --project=. --startup-file=no \
  scripts/run_qwen3_xla_chat.jl \
  /home/ubuntu/models/modelscope/Qwen/Qwen3-8B \
  --profile configs/deployment/qwen3_8b_4090d_bf16_xla_daily.json \
  --prompt "用三点解释 KV Cache" --greedy --max-new-tokens 64
```

当前 XLA 日常 profile 只冻结 batch-1 greedy。CLI 保留 chat template、
system/thinking、history 裁剪、流式 tokenizer bytes、`/clear` 与 EOF
处理；sampling、40K context、FlashAttention、persistent executable
cache 和单次 device-side decode loop 均不在 Chapter 20 的关闭范围内。

权重仍保存在仓库外；Chapter 20 的 deterministic CUDA oracle、最终 JSON
和 200 ms 显存 CSV 体积较小，作为关闭证据进入 Git。0.95/0.89 的失败
中间产物继续只保存在本机，不提交。

## Chapter 21 Qwen3-8B XLA 常驻本地服务

Chapter 21 继续使用上节完全相同的 model directory、revision、asset
manifest 和 4K XLA profile，不复制或修改模型文件。常驻 server：

```bash
julia --project=. --startup-file=no \
  scripts/run_qwen3_xla_server.jl \
  /home/ubuntu/models/modelscope/Qwen/Qwen3-8B
```

默认只监听 `127.0.0.1:11435`。首次 ready 仍需验证资产、读取 15.27 GiB
权重、上传唯一参数树并编译两个 executable；不要为“更快启动”跳过 hash
验证，除非明确接受本地资产未校验风险。服务 ready 后，其他终端只启动
轻量 client：

```bash
julia --project=. --startup-file=no \
  scripts/qwen3_xla_client.jl \
  /home/ubuntu/models/modelscope/Qwen/Qwen3-8B \
  --prompt "解释静态 KV cache 为什么不能并发写" \
  --max-new-tokens 128
```

client 只加载 tokenizer 并套 Qwen3 chat template。服务只支持冻结的
batch-1 greedy 4K profile；sampling、错误 model、非 4K `num_ctx`、
未知 options、超过 1 MiB body 和超窗口请求都会明确拒绝。

最终硬件验收：

```bash
julia --threads=auto --project=. --startup-file=no \
  scripts/benchmark_qwen3_xla_service.jl \
  /home/ubuntu/models/modelscope/Qwen/Qwen3-8B \
  configs/deployment/qwen3_8b_4090d_bf16_xla_daily.json \
  benchmark_results/week20/qwen3_8b_cuda_greedy_reference.json \
  benchmark_results/week21/qwen3_8b_4090d_bf16_xla_service.json
```

结果为 `load_count=1`、15/15 HTTP generation、96/96 frozen CUDA tokens；
9 个 steady 短请求中位 decode 41.351 tok/s，3,584+512 为
1.4828 s prefill / 41.351 tok/s decode。1,228 个 200 ms 显存样本最低
free 2,416,967,680 bytes（2.251 GiB）。

```text
benchmark_results/week21/qwen3_8b_4090d_bf16_xla_service.json
SHA256 73c1bf7a0a7dbabbd2ee1b4ae246022e05d2296e11d8e1d665624fb4cc4b6152

benchmark_results/week21/qwen3_8b_4090d_bf16_xla_service_nvidia_smi.csv
SHA256 e006940214ecabb3802dda178faaad994491cfeae2fc2cfd3425a0d71c2d960b
```

Reactant 0.2.275 的 persistent kernel/autotune cache 不保存完整 PJRT
executable；server 退出后仍会冷启动。Chapter 21 的承诺是一个长寿命进程
服务多个客户端，不是把 kernel cache 文件描述成跨进程 executable cache。

## Chapter 22 Qwen3-Embedding-0.6B

本机模型目录：

```text
/home/ubuntu/models/modelscope/Qwen/Qwen3-Embedding-0.6B
```

RTX 5080 资产机也已按同一冻结 HF revision 恢复一份完整副本：

```text
/home/yj/models/huggingface/Qwen/Qwen3-Embedding-0.6B/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3
```

2026-08-16 逐文件验证下表全部 8 个 bytes/SHA256 一致，并用于 Chapter 39 的真实 retrieval
与三臂验收。该宿主机为 NVIDIA GeForce RTX 5080 16 GiB，driver `595.71.05`；embedding 在 CPU
运行，Qwen3-4B generation 在 CUDA BF16 运行。沙箱内看不到设备不代表宿主机驱动不可用。

下载使用 ModelScope 镜像 commit
`6a58e49965123c0a3012d9576414b8c920faef7e`；验收身份仍冻结为官方
Hugging Face revision
`97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`。镜像只作为传输来源，
以下 8 个内容 hash 必须与该 HF revision 完全一致：

| 资产 | bytes | SHA256 |
| --- | ---: | --- |
| `config.json` | 727 | `b5bf1f51fc45be473a54718cef92448d90a1be001bf9b9a44b8c7f10a19feaa9` |
| `tokenizer.json` | 11,423,705 | `def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a` |
| `tokenizer_config.json` | 9,706 | `253153d0738ceb4c668d2eff957714dd2bea0b56de772a9fdccd96cbf517e6a0` |
| `generation_config.json` | 117 | `28396d421a2108acce96383f6a7de78008f7f1b17f807958f3c14c51dbfb65fb` |
| `modules.json` | 349 | `84e40c8e006c9b1d6c122e02cba9b02458120b5fb0c87b746c41e0207cf642cf` |
| `config_sentence_transformers.json` | 215 | `10667c72ddb772627bf1780cb7f86af8e2ae0032b8c243c731172064105c6961` |
| `1_Pooling/config.json` | 313 | `37bf193fa101f19101bfad9c31d3eb0f786e247b7b1e5cb7f007d730eed1ddbd` |
| `model.safetensors` | 1,191,586,416 | `0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd` |

机器可读清单为
`test/episodes/episode05_deployment_memory_and_sampling/chapter22_qwen3_embedding_memory/fixtures/qwen3_embedding_memory/assets.json`，SHA256
`f02a10758da8b561a9d111823e26d0f4cca05ad905408d3737842c2342bf7782`。
模型权重不进入 Git。

恢复资产：

```bash
git clone --depth 1 \
  https://www.modelscope.cn/Qwen/Qwen3-Embedding-0.6B.git \
  /home/ubuntu/models/modelscope/Qwen/Qwen3-Embedding-0.6B
```

Python oracle 环境严格放在仓库 `.venv`，不会写入已有 Conda、系统 Python
或其他项目环境：

```bash
/home/ubuntu/.local/bin/uv venv --python /usr/bin/python3 .venv
/home/ubuntu/.local/bin/uv pip install \
  --python .venv/bin/python \
  -r requirements/week22-reference.txt
```

冻结版本为 Python `3.10.12`、NumPy `1.26.4`、safetensors `0.5.3`、
tokenizers `0.21.4`、PyTorch `2.7.1+cpu` 与 Transformers `4.51.3`。
导出并验证：

```bash
.venv/bin/python scripts/export_qwen3_embedding_reference.py \
  --model-dir /home/ubuntu/models/modelscope/Qwen/Qwen3-Embedding-0.6B \
  --output test/episodes/episode05_deployment_memory_and_sampling/chapter22_qwen3_embedding_memory/fixtures/qwen3_embedding_memory/reference.json

julia --threads=8 --project=. --startup-file=no \
  scripts/verify_qwen3_embedding_parity.jl \
  /home/ubuntu/models/modelscope/Qwen/Qwen3-Embedding-0.6B \
  test/episodes/episode05_deployment_memory_and_sampling/chapter22_qwen3_embedding_memory/fixtures/qwen3_embedding_memory/reference.json \
  benchmark_results/week22/qwen3_embedding_0_6b_cpu.json

LIFEAI_WEEK22_EMBEDDING_DEVICE=cuda \
julia --threads=8 --project=. --startup-file=no \
  scripts/verify_qwen3_embedding_parity.jl \
  /home/ubuntu/models/modelscope/Qwen/Qwen3-Embedding-0.6B \
  test/episodes/episode05_deployment_memory_and_sampling/chapter22_qwen3_embedding_memory/fixtures/qwen3_embedding_memory/reference.json \
  benchmark_results/week22/qwen3_embedding_0_6b_cuda.json
```

reference SHA256 为
`669694c860798fb8496dd1be199fc648a5779fd7860000c81fe8ca3d0153b322`；
最终 CPU report SHA256 为
`1e2098c5cfc1ba08e941beb1fdd9c012688ab55a518c528228b2b6b2372fe668`，
顶层 `closed=true`。i9-14900K 上 8 条文本、35-token padded batch 的
asset verification / load / forward 为 `2.436 / 8.832 / 7.687 s`，
峰值 RSS `3361.09 MiB`。token/mask 完全一致；五档 MRL embedding 与
similarity max-abs 均低于 `0.01`，15 组 top-k 全一致。

宿主机 RTX 4090 D 的 NVIDIA driver 为 `570.153.02`，CUDA runtime 为
`12.9.0`。GPU report SHA256 为
`5d677d499794bab74c73b22ea0c8342a62fb96f72d942e58eb0655809d839be6`，
顶层 `closed=true`。参数上传 `1.103 s`，同一 8×35 batch 的 cold / warm
forward 为 `18.959 / 0.0641 s`，冷/热 embedding max-abs 为 `0`；
token/mask、五档数值、15 组 top-k 与 semantic memory 门禁全部通过。
普通沙箱内 `nvidia-smi` 因设备隔离失败，不代表宿主机驱动不可用。

## Chapter 43–47 Qwen3-VL-2B-Instruct 资产

本章使用 ModelScope 国内源恢复完整官方仓库：

```text
model_id: Qwen/Qwen3-VL-2B-Instruct
local path: /home/ubuntu/models/modelscope/Qwen/Qwen3-VL-2B-Instruct
ModelScope revision: ae9985b208c074c10cfbe3a61b5cb7268cdc9c53
Hugging Face revision: 78448d793a7eb2f7a987a1da76d464384aa1becd
```

两个 revision 分别描述下载仓库和上游模型身份，不能互相替代，也不能用移动的
`master`/`main` 代替。当前本机 `git rev-parse HEAD` 已验证为上述 ModelScope
commit；仓库内 13 个必要文件同时按以下内容 hash 冻结：

| file | bytes | SHA256 |
| --- | ---: | --- |
| `.gitattributes` | `2,227` | `74f188c115dbc9614c048d24500099ee572b14830cacdfded13c86aa4fb9c7c7` |
| `README.md` | `7,136` | `5fc5be1ca9a3910399bd6239ee5086ab5d82a2a59c5d2b00e887a8835cc110e4` |
| `chat_template.json` | `5,502` | `6f8a6a55027e3da5160105556cda5dd69f6423f1c32645f6730d32de7773d0c4` |
| `config.json` | `1,505` | `bec4b3d446efa05807365c9e1cec03ac590836879d02f3a6da879971154bdd3b` |
| `configuration.json` | `51` | `2d4464e2ead06bc9bc718c781309ad1e7baded626d66e8dcdc8b469ba185faf0` |
| `generation_config.json` | `269` | `1e241830b48b397cb0900101421df5450baddc7adf01e5fc86b5615865f3bae4` |
| `merges.txt` | `1,671,839` | `599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3` |
| `preprocessor_config.json` | `390` | `27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516` |
| `tokenizer_config.json` | `10,868` | `c2da771801886ad9ae98181793ffd3dfb7f1af30f6f7c6a4e15d7dbba52e2399` |
| `tokenizer.json` | `7,032,403` | `a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7` |
| `video_preprocessor_config.json` | `385` | `7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13` |
| `vocab.json` | `2,776,833` | `ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910` |
| `model.safetensors` | `4,255,140,312` | `7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0` |

权重文件是单个 BF16 safetensors：625 个 tensor、`2,127,532,032` 个参数，
tensor payload 为 `4,255,064,064` bytes；其中 `model.visual.*` 为
`406,957,056` 参数。文件大小大于 payload 的部分是 safetensors header，不能
把两者混成同一个指标。

### 国内源恢复与全量验证

新目录可直接从 ModelScope Git/LFS 恢复；clone 后显式 checkout 冻结 commit：

```bash
QWEN3_VL_MODEL_DIR=/home/ubuntu/models/modelscope/Qwen/Qwen3-VL-2B-Instruct

git clone --branch master \
  https://www.modelscope.cn/Qwen/Qwen3-VL-2B-Instruct.git \
  "$QWEN3_VL_MODEL_DIR"
git -C "$QWEN3_VL_MODEL_DIR" checkout --detach \
  ae9985b208c074c10cfbe3a61b5cb7268cdc9c53
git -C "$QWEN3_VL_MODEL_DIR" lfs pull
git -C "$QWEN3_VL_MODEL_DIR" rev-parse HEAD
```

`verify_qwen3_vl_checkpoint` 会重新读取全部 4.255 GB，逐个校验 13 个文件，
再独立推导并检查 625 个 tensor 的 name/shape/BF16 dtype/byte range、参数量和
payload：

```bash
julia --project=. --startup-file=no -e '
using LifeAI
model_dir = "/home/ubuntu/models/modelscope/Qwen/Qwen3-VL-2B-Instruct"
report = verify_qwen3_vl_checkpoint(model_dir)
@show report.modelscope_revision report.hf_revision
@show report.tensor_count report.parameter_count report.tensor_bytes
'
```

真实 checkpoint 测试使用同一验证器，不会自动联网：

```bash
LIFEAI_QWEN3_VL_2B_MODEL_DIR=/home/ubuntu/models/modelscope/Qwen/Qwen3-VL-2B-Instruct \
  julia --project=. --startup-file=no -e 'using Pkg; Pkg.test()'
```

Chapter 43 的 Transformers 4.57.0 oracle 与 CUDA verifier 分开运行。Python
reference 目录由调用方指定并保持在仓库外；Float32 是 strict parity gate，
BF16 只记录 CPU oracle 与 CUDA LifeAI 的跨后端边界：

```bash
QWEN3_VL_MODEL_DIR=/home/ubuntu/models/modelscope/Qwen/Qwen3-VL-2B-Instruct
QWEN3_VL_REFERENCE_DIR=/tmp/qwen3-vl-vision-float32

PYTHONPATH=/tmp/lifeai-qwen3vl-oracle/lib/python3.10/site-packages \
  .venv/bin/python scripts/export_qwen3_vl_vision_reference.py \
  "$QWEN3_VL_MODEL_DIR" "$QWEN3_VL_REFERENCE_DIR" float32

julia --project=. --startup-file=no \
  scripts/verify_qwen3_vl_vision_cuda.jl \
  "$QWEN3_VL_MODEL_DIR" "$QWEN3_VL_REFERENCE_DIR" cuda
```

最终 Float32/BF16 reference SHA256 分别为
`480d988d9f679c8090f8c80c8e5cd007e5a41c47e6bb5cc7ad2f16541cbe5f88` 和
`ecd904b8a110169c73c9814d23d43eabcc5a2593d0a746bbbda8bb9c308b36b8`。
RTX 4090 D Float32 strict `visual_embeddings` max/mean/relative L2/cosine 为
`6.771088e-5 / 2.2492045e-6 / 5.461097e-6 / 1.0`；load/cold/warm
为 `5.934235821 / 11.319966811 / 0.024733474 s`，logical parameter bytes
为 `1,627,828,224`。BF16 boundary 的四项指标为
`0.890625 / 0.02120505 / 0.05465893 / 0.9985048`；load/cold/warm 为
`3.188056864 / 11.916055118 / 0.056884242 s`，logical parameter bytes
为 `813,914,112`。cold 含 Julia/CUDA 首次编译，不是 steady benchmark；warm
是相同输入的显式同步复跑。BF16 结果不会被写成 strict parity。

### Chapter 44 processor/chat 与 multimodal prefill reference

Chapter 44 不引入第二份 checkpoint：raw processor、VL tokenizer/chat 和 text
decoder 继续消费上表同一 13-file immutable asset contract。独立 Transformers
4.57.0 exporter 从 deterministic `256×256` raw RGB image 与 content-list
`Describe.` message 出发，显式传入全一 attention mask，并在 DeepStack 原地修改
hidden 前 clone 每个 capture。生成的仓库外 reference 为：

| compute | reference directory | `reference.safetensors` SHA256 |
| --- | --- | --- |
| Float32 | `/tmp/qwen3-vl-prefill-f32` | `d7d3b58cea35cf90806bdd14ade7e453e1b486355b190d094ec95f852f6b60f5` |
| BF16 | `/tmp/qwen3-vl-prefill-bf16` | `711749d9cb0d2c33b34c6fc87a4f9dd06bbf7cc52b589daf01f0210bd58cb5ae` |

对应 `reference.json` SHA256 分别为 Float32
`1141ef4c503800607d60d8fa23eb795b00d94a7ff879f97ac230abf05681362c`
与 BF16 `7a941b837e7c6df13490386dc42160b9ac6f7d5befcc6bba148cc10e1e81afb5`；
verifier 将 tensor 与 metadata 两类 SHA 都硬编码为准入条件。

exporter 会 fail closed 地检查 CPython `3.10.12`、NumPy `1.26.4`、Pillow
`11.3.0`、safetensors `0.5.3`、tokenizers `0.22.1`、Jinja2 `3.1.6`、PyTorch
`2.7.1+cpu` 的 build revision/config、Transformers `4.57.0`、torchvision
`0.22.1+cu126`、AVX2/MKL/OpenMP/MKLDNN 与固定的 `24/24` intra/inter-op
threads。本机使用以下 frozen overlay：

```bash
QWEN3_VL_MODEL_DIR=/home/ubuntu/models/modelscope/Qwen/Qwen3-VL-2B-Instruct
QWEN3_VL_ORACLE_PYTHONPATH=/tmp/lifeai-qwen3vl-oracle/lib/python3.10/site-packages:/tmp/lifeai-qwen3vl-uv-cache/archive-v0/SNUjiORDNkYR55Or

PYTHONPATH="$QWEN3_VL_ORACLE_PYTHONPATH" \
  .venv/bin/python scripts/export_qwen3_vl_prefill_reference.py \
  "$QWEN3_VL_MODEL_DIR" /tmp/qwen3-vl-prefill-f32 float32

PYTHONPATH="$QWEN3_VL_ORACLE_PYTHONPATH" \
  .venv/bin/python scripts/export_qwen3_vl_prefill_reference.py \
  "$QWEN3_VL_MODEL_DIR" /tmp/qwen3-vl-prefill-bf16 bfloat16

julia --project=. --startup-file=no \
  scripts/verify_qwen3_vl_prefill_cuda.jl \
  "$QWEN3_VL_MODEL_DIR" /tmp/qwen3-vl-prefill-f32 cuda

julia --project=. --startup-file=no \
  scripts/verify_qwen3_vl_prefill_cuda.jl \
  "$QWEN3_VL_MODEL_DIR" /tmp/qwen3-vl-prefill-bf16 cuda
```

当前 cache 中的 torchvision `0.22.1+cu126` wheel 缺少可加载的 C++ NMS op，
但 Qwen3-VL image transform 不消费 NMS。exporter 只在 torchvision import
期间忽略这个唯一缺失 op 的 fake-registration；其他 import 错误仍是 fatal，且
resize/normalize 实现没有被替换。F32/BF16 reference 与 metadata 均已从头各重建
一次并逐字节复现下列冻结 SHA，不能用该 workaround 绕过 processor 数值契约。

RTX 4090 D verifier 在 `CUDA.allowscalar(false)` 下对 Float32/BF16 各跑通 80 个
stage/semantic gates。两次输入均为 sequence `76`、image tokens `64`，raw
processor max-abs `0`：

| compute | logical parameter bytes | final hidden max-abs | logits max-abs | combined warm prefill |
| --- | ---: | ---: | ---: | ---: |
| Float32 | `8,510,128,128` | `0.00094986` | `0.00083363` | `0.703 s` |
| BF16 | `4,255,064,064` | `6.9375` | `7.4375` | `0.751 s` |

Float32 是 Chapter 44 strict correctness gate；BF16 仍是 CPU oracle 与 CUDA
LifeAI 之间的跨后端舍入边界。`/tmp` 目录只代表本次机器上的工作副本，长期保留
时应把 reference 移到仓库外的持久资产目录并记录同一 SHA256。

Chapter 44 的边界停在 cache-free prefill；Chapter 45 使用同一份 checkpoint 和
processor/tokenizer 资产继续关闭 dynamic cache/decode，reference 契约如下。

### Chapter 45 dynamic decode 与 image-to-text reference

Chapter 45 没有下载第二份权重。`export_qwen3_vl_decode_reference.py real` 从同一
deterministic `256×256` image 与 `Describe.` content-list message 出发，以 HF
`DynamicCache` 做 cached prefill 和三次单 token decode。真实 K/V 很大，因此只在
metadata 中冻结每层 shape 和 raw SHA256；仓库外 safetensors 只保存每个决策阶段的
last hidden 与 logits：

| artifact | path | SHA256 |
| --- | --- | --- |
| Float32 tensors | `/tmp/qwen3-vl-decode-f32/reference.safetensors` | `a98812e25efb44c02ab9c06e974ab718724f35f2f1c686e4bdc395d856c03e81` |
| Float32 metadata | `/tmp/qwen3-vl-decode-f32/reference.json` | `569fe3666b65ee2f497327e9ce9931f81652d5bdc32d44dfb9fb774435caccfc` |

exporter 复用 Chapter 44 的完整 Python/NumPy/Pillow/safetensors/tokenizers/Jinja2/
PyTorch/torchvision build、CPU capability、MKL/OpenMP/MKLDNN 和 `24/24` thread
门禁，同时重新校验 ModelScope revision、clean tree 与必要资产 SHA。可复现命令：

```bash
QWEN3_VL_MODEL_DIR=/home/ubuntu/models/modelscope/Qwen/Qwen3-VL-2B-Instruct
QWEN3_VL_ORACLE_PYTHONPATH=/tmp/lifeai-qwen3vl-oracle/lib/python3.10/site-packages:/tmp/lifeai-qwen3vl-uv-cache/archive-v0/SNUjiORDNkYR55Or
QWEN3_VL_DECODE_REFERENCE_DIR=/tmp/qwen3-vl-decode-f32

PYTHONPATH="$QWEN3_VL_ORACLE_PYTHONPATH" \
  .venv/bin/python scripts/export_qwen3_vl_decode_reference.py real \
  "$QWEN3_VL_MODEL_DIR" "$QWEN3_VL_DECODE_REFERENCE_DIR" \
  --dtype float32 --device cpu --greedy-tokens 4

julia --project=. --startup-file=no \
  scripts/verify_qwen3_vl_decode_cuda.jl \
  "$QWEN3_VL_MODEL_DIR" "$QWEN3_VL_DECODE_REFERENCE_DIR" cuda
```

RTX 4090 D / CUDA runtime `12.9.0` 的最终 Float32 strict 结果为：

| stage | max-abs | mean-abs | relative L2 | cosine | argmax (LifeAI 1-based) |
| --- | ---: | ---: | ---: | ---: | ---: |
| prefill | `3.862380981445e-5` | `7.502852388181e-6` | `2.491764492934e-6` | `0.9999999999970042` | `1987` |
| decode 0 | `4.005432128906e-5` | `5.626603576536e-6` | `1.276814328535e-6` | `0.9999999999992562` | `2169` |
| decode 1 | `2.956390380859e-5` | `5.176316295721e-6` | `1.219367434425e-6` | `0.9999999999993456` | `375` |
| decode 2 | `2.908706665039e-5` | `5.168276760988e-6` | `1.653558204443e-6` | `0.9999999999986956` | `265` |

HF 0-based greedy ids 为 `[1986,2168,374,264]`，LifeAI 1-based ids 为
`[1987,2169,375,265]`，两边最终文本均为 `This image is a`。prompt/cache 轨迹为
`76→77→78→79`，image tokens `64`，request-local `rope_delta=-56`，三次 decode
使用 mRoPE 坐标 `20/21/22`。最终 F32 cache 为 `18,120,704` bytes；按 28 layers、
8 KV heads、128 head dim 计算，每个 token 为 `229,376` bytes（`224 KiB`），BF16
则为 `114,688` bytes（`112 KiB`）。默认 BF16 CUDA 高层 API 也实跑得到相同
token ids/text，但它不替代 Float32 的跨实现 strict 数值门禁。

本次记录的 vision/text load、vision cold、cached-prefill cold 分别约为
`5.938 / 27.140 / 11.719 / 6.681 s`；三次 decode 为
`0.759 / 0.0415 / 0.0159 s`，高层 generation 为 `3.174 s`。这些时间含首次
Julia/CUDA 编译和重新跑 vision，不能当 steady throughput。当前 dynamic cache
使用 `cat` 追加，每步都会重新分配并复制历史 K/V；本章关闭的是 correctness，
不是 static capacity、低分配部署或长上下文性能。尚未覆盖 video、multi-image、
batch/padding、sampled generation 和真实长上下文。

### Chapter 46 bounded static cache：无新增 checkpoint/reference

Chapter 46 继续使用本节开头冻结的同一 ModelScope/Hugging Face revision、13-file
checksum、Chapter 44 processor/prefill reference 与 Chapter 45 decode reference，
没有下载第二份权重，也没有把 LifeAI 自己的 static storage 输出伪装成新的外部
oracle。static 与 dynamic 的数学状态相同，变化的是 K/V ownership：Chapter 45
fixture 仍是独立 Transformers `DynamicCache` reference，Chapter 46 同时把 LifeAI
dynamic 与 static 的有效 prefix 对到它。

committed tiny fixture 仍位于：

```text
test/episodes/episode09_qwen3_vl_multimodal_perception/
  chapter45_qwen3_vl_dynamic_decode/fixtures/tiny_text_dynamic_decode.json
```

SHA256 仍为
`7b20111e43aa9efd2aae0be3f4a740ab1fdeaff9bd0a6ffd0bfef49adfeeffd8`。
它冻结 4-layer tiny model 的 prefill 与两次 decode、全部 K/V、hidden/logits 和
greedy timeline。Chapter 46 默认离线测试在 capacity `8/9/10` 下验证 static
有效 prefix 与该 fixture、LifeAI dynamic path 一致，同时验证固定 storage
identity/bytes、overflow 原子性与 reset/reuse。

低层 static cache 的 `capacity` 是必填 keyword：

```julia
using LifeAI

cache = init_qwen3_vl_static_kv_cache(
    text_parameters;
    capacity=79,
    batch_size=1,
)
```

不能省略它并按官方最大 context 隐式分配。高层 generation 已知 prompt length 和
`max_new_tokens`，因此 `cache=:static` 在未给出 `static_capacity` 时只推导本次请求
精确需要的：

```text
prompt_length + max(0, max_new_tokens - 1)
```

Chapter 45 的真实 sequence 76、4-token generation 对应 capacity 79。官方 2B
每个 capacity token 的 K/V payload 为 Float32 `229,376` bytes、BFloat16
`114,688` bytes，所以 capacity 79 分别预分配 `18,120,704` 与 `9,060,352`
bytes。该数字不含 allocator metadata、模型参数、vision features、attention/MLP/
RoPE/logits intermediates 或 CUDA workspace。

单独运行离线 static contract：

```bash
julia --project=. --startup-file=no -e '
using Test, LifeAI
include("test/episodes/episode10_qwen3_vl_efficient_generation/" *
        "chapter46_qwen3_vl_static_cache/test_qwen3_vl_static_cache.jl")
'
```

真实 checkpoint 恢复和 Python oracle 仍使用 Chapter 45 上方命令与相同 frozen
files；Chapter 46 新增的 CUDA verifier 接在 Chapter 45 verifier 后执行：

```bash
QWEN3_VL_MODEL_DIR=/home/ubuntu/models/modelscope/Qwen/Qwen3-VL-2B-Instruct
QWEN3_VL_DECODE_REFERENCE_DIR=/tmp/qwen3-vl-decode-f32

julia --project=. --startup-file=no \
  scripts/verify_qwen3_vl_static_cache_cuda.jl \
  "$QWEN3_VL_MODEL_DIR" "$QWEN3_VL_DECODE_REFERENCE_DIR" cuda float32
```

RTX 4090 D Float32 最终结果：capacity 128 的 56 个 K/V `CuArray` 在 prefill、
三次 decode 和 reset 间保持 object/device-pointer identity，且所有 buffers
互不 alias；static 与 dynamic 的有效 K/V prefix/logits bitwise 一致。相对同一
HF reference 的四阶段 max-abs 仍为
`3.86238e-5 / 4.00543e-5 / 2.95639e-5 / 2.90871e-5`，greedy ids/text 仍为
`[1987,2169,375,265]` / `This image is a`。capacity-128 Float32 payload 为
`29,360,128` bytes，四阶段后 valid 79-token prefix 为 `18,120,704` bytes。

32-token dynamic/static 对照最终 position 都是 107、ids exact。三样本中位 GPU
allocated bytes 为 `4,270,103,300 / 3,645,283,076`，static 少
`624,820,224` bytes；allocation count 为 `69,431 / 67,751`。理论 dynamic
`cat` outputs 为 `654,180,352` bytes，static 新 slice writes 为 `7,110,656`
bytes。latency 中位 `0.616681 / 0.585846 s` 只有三个交替顺序样本，verifier 明确不把它
设为 acceptance gate。

同一命令末尾使用 `bfloat16` 时，static/dynamic 四阶段 K/V prefix 与 logits
bitwise，ids 与 Float32 相同；但它只对 HF Float32 reference 报 cross-dtype
boundary，不声明 BF16 HF tensor strict parity。capacity 128 / valid 79 的 BF16
payload 为 `14,680,064 / 9,060,352` bytes。32-token dynamic/static GPU allocated
bytes 为 `1,881,657,676 / 1,569,247,564`，减少 `312,410,112`；理论 `cat`
outputs/static writes 为 `327,090,176 / 3,555,328` bytes。最终交替顺序样本的
latency 中位为 `0.849395 / 0.840655 s`，两者接近；固定顺序的先导样本曾给出相反
观感，说明三样本 latency 对顺序/GC 敏感。因此 latency 不作为本章 gate，也不据此
宣称 static 普遍更快。

`Qwen3VLStaticKVCache` 只保证 K/V backing arrays 在 prefill、
decode、reset 间固定，并消除 token-axis `cat` 与历史 prefix copy；它不保证整条
decode 执行零分配，也不代表 CUDA Graph/XLA static executable、长上下文、batch/
padding、multi-image/video 或 sampling 已验收。`reset_qwen3_vl_static_kv_cache!`
默认只清 logical `position/rope_delta`；`clear=true` 才会遍历并归零整个 capacity。

实现时还暴露了 Julia API 层面的可复现失败：keyword 参数及其类型不参与 method
dispatch。若只用 `cache::Qwen3VLKVCache` 与 `cache::Qwen3VLStaticKVCache` 两个
keyword 定义同名 prefill，后定义会覆盖前定义并触发 precompile method-overwrite。
最终保留显式 `hf_qwen3_vl_text_prefill_static` / `_decode_step_static` API，generation
内部只在 positional cache helper 上分派。资产没有变化，但这一语言契约决定了
reference verifier 必须同时回归 Chapter 45 dynamic 路径。

Chapter 47 继续使用同一 checkpoint、Chapter 45 decode reference、确定性 image
和 76-token prompt，不新增权重。它新增仓库外的 HF Float32/CPU 256-token 长轨迹
oracle；exporter 流式消费 DynamicCache，不保存逐步完整 K/V。生成命令为：

```bash
QWEN3_VL_MODEL_DIR=/home/ubuntu/models/modelscope/Qwen/Qwen3-VL-2B-Instruct
QWEN3_VL_DECODE_REFERENCE_DIR=/tmp/qwen3-vl-decode-f32
QWEN3_VL_ORACLE_PYTHONPATH=/tmp/lifeai-qwen3vl-oracle/lib/python3.10/site-packages:/tmp/lifeai-qwen3vl-uv-cache/archive-v0/SNUjiORDNkYR55Or
QWEN3_VL_LONG_REFERENCE="$QWEN3_VL_DECODE_REFERENCE_DIR/long_generation_reference.json"

PYTHONPATH="$QWEN3_VL_ORACLE_PYTHONPATH" \
  .venv/bin/python scripts/export_qwen3_vl_long_generation_reference.py \
  "$QWEN3_VL_MODEL_DIR" "$QWEN3_VL_LONG_REFERENCE" \
  --greedy-tokens 256 --checkpoint-lengths 32,128,256

QWEN3_VL_LONG_REFERENCE_SHA256="$(sha256sum "$QWEN3_VL_LONG_REFERENCE" | cut -d' ' -f1)"

LIFEAI_QWEN3_VL_LONG_REFERENCE="$QWEN3_VL_LONG_REFERENCE" \
LIFEAI_QWEN3_VL_LONG_REFERENCE_SHA256="$QWEN3_VL_LONG_REFERENCE_SHA256" \
julia --project=. --startup-file=no \
  scripts/benchmark_qwen3_vl_static_long_generation.jl \
  "$QWEN3_VL_MODEL_DIR" "$QWEN3_VL_DECODE_REFERENCE_DIR" \
  /tmp/qwen3_vl_long_profile.json
```

上面的即时 SHA 只适合首次 smoke；正式验收必须在两个独立 Python 进程得到逐字节相同
JSON，人工复核后再把固定 SHA 传给 benchmark。loader 会在模型加载前拒绝 hash/schema、
Float32 CPU claim、revision/assets、prompt/image/input IDs 或 256-step timeline 不一致。
若目标 BF16 run 与 oracle 分叉，同一输出路径会保存
`status="correctness_blocker"` JSON，包含首分叉 step、双方 token、HF top-2 margin
和完整对照 prefix；该 run 不会继续进入 allocation attribution。

2026-10-08 的持久副本不放在 `/tmp`，也不放进模型 git 工作树：

```text
oracle: /home/ubuntu/models/lifeai-oracles/qwen3vl-transformers-4.57.0
chapter45: /home/ubuntu/models/lifeai-references/Qwen3-VL-2B-Instruct/chapter45-decode-f32/
long oracle: /home/ubuntu/models/lifeai-references/Qwen3-VL-2B-Instruct/chapter47-long-oracle/long_generation_reference.json
blocker: /home/ubuntu/models/lifeai-references/Qwen3-VL-2B-Instruct/chapter47-long-oracle/qwen3_vl_long_profile.json
```

Chapter 45 `reference.safetensors` / `reference.json` 重新导出后仍是
`a98812e25efb44c02ab9c06e974ab718724f35f2f1c686e4bdc395d856c03e81` /
`569fe3666b65ee2f497327e9ce9931f81652d5bdc32d44dfb9fb774435caccfc`。两个独立
进程的 long oracle 逐字节相同，SHA256 为
`72461330a8c5a0e74e7489abfdd481223aeb47d6c161fdc12dc0048e82690140`。RTX 4090 D
BF16 static 256-token warmup 在 generated step 79 分叉，blocker 文件 SHA256 为
`90b10f6cb25489984314bfb02b727557fd25d71a1a1424c08bda98301d00c8d4`，
`closed=false`。这是 correctness blocker，不是 allocation acceptance fixture。
2026-10-08 的同前缀 dtype 拆分写在同目录 `step79_dtype_split.json`，SHA256 为
`ebd90347601b12abfbca79d656c7c8f39d859ea0858575db97624cfbcbdf0fbb`。LifeAI
Float32 CUDA 79/79 argmax 与 HF 相同；BF16 在 step 79 把 margin `0.04928589`
落成 `26 / 25.875` 后选中 runner-up。`classification="cross_dtype_boundary"`，
同样 `closed=false`，也不是 acceptance fixture。Teacher-forced 256 步写在
`step256_teacher_forced_dtype_split.json`，SHA256 为
`126bb11234031f20b7dcc1475c515c76ca76e2819b71d5ab58cab8495c6edf07`。Float32 CUDA
256/256 argmax 与 HF 相同；BF16 在正确前缀上只在 step 79 和 step 95 离开，
step 95 两个候选都是 `28.375`。它同样 `closed=false`，不是 acceptance fixture。
同日的非验收 allocation 报告是 `qwen3_vl_allocation_despite_divergence.json`，
SHA256 为 `b317f7fd4191f6f04a7f7023b3fda59fc59458174cb3a6f4dbde443038112a6f`。
它使用一次 repeat，并在记录 token 分叉后继续归因。position `107 / 203 / 331`
的最后一步 bytes/count 与无 hook 表面一致，`kv_write` 为 0，dominant stage 为
`attention`。32-token prefix exact，128/256 在 step 79 分叉，因此
`acceptance_fixture=false`。复用 decode attention scratch 之后的重测是
`qwen3_vl_allocation_attention_scratch.json`，SHA256 为
`66c9afd64bc81a10c89c11f76313b5f99061a254e4b325f7e76378a26924a4a7`。
三个 position 的 attention bytes 都是 `7,168`，整步合计都是 `10,563,721`；
`kv_write` 为 0。三次 repeat 的关闭采样是 `qwen3_vl_allocation_close.json`，
SHA256 为 `1a915560007e1424fedc77a0297ea66e910baa72de8aef9f1b23b6607f7954b2`。
32-token prefix exact，128/256 在 step 70 分叉，三次 timeline 各自 deterministic。
Chapter 47 因此 Closed。该文件是 allocation 关闭记录，不是 BF16 strict parity。
Chapter 48 复用 MLP activation scratch 之后的一次 repeat 重测是
`qwen3_vl_allocation_mlp_scratch.json`，SHA256 为
`4aa2057607d940a12901bb3d71f8498a9fff1c6d548e597d07c8d7c3a22906de`。
三个 position 的 `mlp_activation` 都是 `0` bytes，整步合计都是 `7,811,209`，
正好少了 `2,752,512`。32-token prefix exact，128/256 仍在 step 70 分叉。
Chapter 48 因此 Closed。该文件同样是 allocation 关闭记录，不是 BF16 strict parity。
Chapter 49 复用 QK RoPE scratch 之后的一次 repeat 重测是
`qwen3_vl_allocation_qk_rope.json`，SHA256 为
`d8726e41c1171fe42b823254bb0ccb3fde40baf547c700dd87bda228f0c398d8`。
三个 position 的 `qk_rope` 都是 `0` bytes，整步合计都是 `5,546,121`，
正好少了 `2,265,088`。32-token prefix exact，128/256 仍在 step 70 分叉。
Chapter 49 因此 Closed。该文件同样是 allocation 关闭记录，不是 BF16 strict parity。
Chapter 50 复用 QK norm scratch 之后的一次 repeat 重测是
`qwen3_vl_allocation_qk_norm.json`，SHA256 为
`8f415d759aca2cf27a402c484c2c6c2c0a06624a25c64bceb6f74681f25afdad`。
三个 position 的 `qk_norm` 都是 `0` bytes，整步合计都是 `4,479,881`，
正好少了 `1,066,240`。32-token prefix exact，128/256 仍在 step 70 分叉。
Chapter 50 因此 Closed。该文件同样是 allocation 关闭记录，不是 BF16 strict parity。
Chapter 51 复用 hidden RMSNorm scratch 之后的一次 repeat 重测是
`qwen3_vl_allocation_hidden_norm.json`，SHA256 为
`cb5b785c94655583dfe361923817ccb515ad2673bee849b164df2c80ec8a6be0`。
`pre_attention_norm` 和 `post_attention_norm` 各剩 `448` bytes，`final_norm`
剩 `16` bytes，整步合计 `2,611,649`。Chapter 51 因此 Closed。
Chapter 52 复用 gate/up 输出之后的一次 repeat 重测是
`qwen3_vl_allocation_gate_up.json`，SHA256 为
`e39f1e52b76a0c965175c7591f8f2a9aad60255d3d0fdf434ce91a9c30d10399`。
`mlp_gate_up_projection` 从 `688,576` 降到 `448`，整步合计 `1,923,521`。
Chapter 52 因此 Closed。这些文件都是 allocation 记录，不是 BF16 strict parity。
Chapter 53 复用 hidden residual scratch 之后的一次 repeat 重测是
`qwen3_vl_allocation_hidden_residual.json`，SHA256 为
`73f44165347e1eae6628be1cc9c9b05f51bf86b8d496b34cb0a40474c08317a9`。
两处 residual 都从 `688,352` 降到 `224`，整步合计 `547,265`。
Chapter 53 因此 Closed。该文件同样是 allocation 记录，不是 BF16 strict parity。
Chapter 54 复用 tied logits scratch 之后的一次 repeat 重测是
`qwen3_vl_allocation_tied_logits.json`，SHA256 为
`5123640b5cbd751eb5c7c794566b5c40027f9f7e0ae21b7654775568e7a0e429`。
`vocab_logits` 从 `303,880` 降到 `8`，整步合计 `243,393`。
Chapter 54 因此 Closed。该文件同样是 allocation 记录，不是 BF16 strict parity。
Chapter 55 复用 QKV scratch 之后的一次 repeat 重测是
`qwen3_vl_allocation_qkv.json`，SHA256 为
`5c27f76184dca24a0eb0ce48dec26bf92809baad5a92407affcd461c0aaf0995`。
`qkv_projection` 从 `230,048` 降到 `672`，整步合计 `14,017`。
Chapter 55 因此 Closed。该文件同样是 allocation 记录，不是 BF16 strict parity。
Chapter 56 把 attention softmax 的归约改成 `maximum!` / `sum!` 之后的一次
repeat 重测是 `qwen3_vl_allocation_attention_softmax.json`，SHA256 为
`3dda0a60100a6dbd73b0931b858b34e51205899daf941acf065ea16e9019ff49`。
`attention` 从 `7,168` 降到 `3,584`，整步合计 `10,433`。
Chapter 56 因此 Closed。该文件同样是 allocation 记录，不是 BF16 strict parity。
Chapter 57 复用 token embedding scratch 之后的一次 repeat 重测是
`qwen3_vl_allocation_token_embedding.json`，SHA256 为
`dfaa168759d4beddc9e8db405cbcdf7a3a0896a83831a3f2df6b0335fd575742`。
`token_embedding` 从 `4,105` 降到 `0`，整步合计 `6,328`。
Chapter 57 因此 Closed。该文件同样是 allocation 记录，不是 BF16 strict parity。
`qwen3_vl_allocation_strided_gemm.json` 的 SHA256 为
`b8542c41ada07f774c728e0f72457b6627098822759f1c55b2ed87caa7c50561`。
它是一次没有采纳的 strided GEMM 试验：`attention` 降到 `448` bytes，但
sequence `331` 不逐位，自由生成分叉回到 step 79。当前代码仍是 per-head
`batched_mul!`。Chapter 58 因此 Closed。该文件不是 acceptance fixture。
Chapter 59 把同一条 `cublasGemmEx` 的 alpha/beta 换成持久 device 标量之后的
一次 repeat 重测是 `qwen3_vl_allocation_gemm_scalar.json`，SHA256 为
`c0bdb3d4fa6427bce41d8f42a95f075ec6c53331c10eed1d00a2aa5014a6298c`。
`attention`、`qkv_projection`、`mlp_gate_up_projection`、两处 residual 和
`vocab_logits` 都是 `0` bytes，整步合计 `1,168`。32-token prefix exact，
128/256 仍在 step 70 分叉。Chapter 59 因此 Closed。该文件同样是 allocation
记录，不是 BF16 strict parity。
Chapter 60 复用 hidden RMSNorm 的 `sum!` partial 之后的一次 repeat 重测是
`qwen3_vl_allocation_norm_partial.json`，SHA256 为
`b9e6d0ab4331d689d448ff8d03b7714d283c0553454f37f89b37abb43276361e`。
三处 hidden norm 都是 `0` bytes，整步合计 `256`。32-token prefix exact，
128/256 仍在 step 70 分叉。Chapter 60 因此 Closed。该文件同样是 allocation
记录，不是 BF16 strict parity。
Chapter 61 复用单 token mRoPE cos/sin 之后的一次 repeat 重测是
`qwen3_vl_allocation_mrope.json`，SHA256 为
`8411b055efea353da282dc295558b5db99b6d088b7814a62673ce54ecc7d46b9`。
`mrope_prepare` 是 `0` bytes，position `107 / 203 / 331` 的最后一步都是 `0`。
32-token prefix exact，128/256 仍在 step 70 分叉。Chapter 61 因此 Closed。
该文件同样是 allocation 记录，不是 BF16 strict parity。
Chapter 62—67 把元素广播收成 kernel 之后的一次 repeat 重测是
`qwen3_vl_allocation_fused_elementwise.json`，SHA256 为
`ebf007aafe96f34972d45a8429899acdcfd8781468e4f9b24611396ec9ab8b4a`。
GPU 最后一步仍是 `0`。256-token 稳态一步的 CPU 分配是 `2,168,720` bytes。
32-token prefix exact，128/256 仍在 step 70 分叉。Chapter 62—67 因此 Closed。
该文件同样是 allocation 记录，不是 BF16 strict parity。
Chapter 69 把 attention softmax 的 scale、exp、weight 收成 kernel 之后的一次
repeat 重测是 `qwen3_vl_allocation_softmax_kernel.json`，SHA256 为
`1d463083013c0076f62032731f14a02129e0eaced160ee09e4cdb87e79a5d638`。
GPU 最后一步仍是 `0`。256-token 稳态一步的 CPU 分配是 `1,817,040` bytes。
32-token prefix exact，128/256 仍在 step 70 分叉。Chapter 69 因此 Closed。
该文件同样是 allocation 记录，不是 BF16 strict parity。
Chapter 70 把 attention 置换收成拷贝 kernel 之后的一次 repeat 重测是
`qwen3_vl_allocation_permute_kernel.json`，SHA256 为
`269aa7775f10056a8abe10a70f0180288db4b995b862e07a2f00761903e8244f`。
GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 都是 `1,422,800`
bytes。32-token prefix exact，128/256 仍在 step 70 分叉。Chapter 70 因此
Closed。该文件同样是 allocation 记录，不是 BF16 strict parity。
Chapter 71 把 QK norm 的升精度和均值除法收进 kernel 之后的一次 repeat 重测是
`qwen3_vl_allocation_qk_prepare.json`，SHA256 为
`ab40b3fee9c35b43385c32d5c70dbffff01f44ccfc548b76cd2ceac1bd8d1da6`。
GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 都是 `1,126,224`
bytes。32-token prefix exact，128/256 仍在 step 70 分叉。Chapter 71 因此
Closed。该文件同样是 allocation 记录，不是 BF16 strict parity。
Chapter 72 把 `cublasGemmEx` 的启动对象收成持久状态之后的一次 repeat 重测是
`qwen3_vl_allocation_gemm_launch.json`，SHA256 为
`33a5f3d04f0bdc2cf405358585cec60a26607e0fe259453f331ca9452b36c355`。
GPU 最后一步仍是 `0`。256-token 稳态一步的 CPU 是 `873,104` bytes。
32-token prefix exact，128/256 仍在 step 70 分叉。Chapter 72 因此 Closed。
该文件同样是 allocation 记录，不是 BF16 strict parity。
Chapter 73 把 hidden RMSNorm 的升精度和除法收进 kernel 之后的一次 repeat
重测是 `qwen3_vl_allocation_hidden_prepare.json`，SHA256 为
`aa375cd1d1ea25041f1bf50259838af4c0f176253c77b90b4a6d794d34861fa0`。
GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 都是 `677,920`
bytes。32-token prefix exact，128/256 仍在 step 70 分叉。Chapter 73 因此
Closed。该文件同样是 allocation 记录，不是 BF16 strict parity。
Chapter 74 把 attention `maximum!` 换成顺序扫描之后的一次 repeat 重测是
`qwen3_vl_allocation_attention_max.json`，SHA256 为
`030d937ac838355235bed889beac2720c496a1707b6e0db364a06421e6c28d2a`。
GPU 最后一步仍是 `0`。256-token 稳态一步的 CPU 是 `617,440` bytes。
32-token prefix exact，128/256 仍在 step 70 分叉。Chapter 74 因此 Closed。
该文件同样是 allocation 记录，不是 BF16 strict parity。
Chapter 75 用 stride 偏移代替 per-head GEMM view 之后的一次 repeat 重测是
`qwen3_vl_allocation_batched_pointer.json`，SHA256 为
`8cfa7b7f67eb1f0c241b751d6a047a68c997e1e25db81a7d58143eca6485beae`。
GPU 最后一步仍是 `0`。256-token 稳态一步的 CPU 是 `581,600` bytes。
32-token prefix exact，128/256 仍在 step 70 分叉。Chapter 75 因此 Closed。
该文件同样是 allocation 记录，不是 BF16 strict parity。
Chapter 76 把 abs2 归约计划收成具体类型之后的一次 repeat 重测是
`qwen3_vl_allocation_abs2_plan.json`，SHA256 为
`7aad1e4a7a720fe31136200fd016125e0ed48d002e073b62ed4e16f1060d6fd1`。
GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 都是 `513,152`
bytes。32-token prefix exact，128/256 仍在 step 70 分叉。Chapter 76 因此
Closed。该文件同样是 allocation 记录，不是 BF16 strict parity。
Chapter 77 把单 token K/V 写入换成拷贝 kernel 之后的一次 repeat 重测是
`qwen3_vl_allocation_kv_write.json`，SHA256 为
`20917f6fd9f56e3f2cfa8752c14f857d7d72ad99268c35ac6dc16a17347fc792`。
GPU 最后一步仍是 `0`。256-token 稳态一步的 CPU 是 `441,024` bytes。
`kv_write` 从 `110,976` 降到 `34,080`。32-token prefix exact，128/256 仍在
step 70 分叉。Chapter 77 因此 Closed。该文件同样是 allocation 记录，不是
BF16 strict parity。
Chapter 78 把 attention softmax 分母换成单 block warp 树之后的一次 repeat
重测是 `qwen3_vl_allocation_attention_sum.json`，SHA256 为
`ea2340ed9ab2231f79a1e93490d72c2cd17b5411e51b8a2f2dda00c93aed0290`。
GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 都是 `424,896`
bytes。32-token prefix exact，128/256 仍在 step 70 分叉。Chapter 78 因此
Closed。该文件同样是 allocation 记录，不是 BF16 strict parity。
Chapter 79 把 Q/K 的 abs2 计划放进具体 `Ref` 之后的一次 repeat 重测是
`qwen3_vl_allocation_qk_abs2.json`，SHA256 为
`10da9cdb7e18588ce9431f3a8dcc9b719a9062ea12d53cb107304f1d4396fcc2`。
GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 都是 `415,936`
bytes。32-token prefix exact，128/256 仍在 step 70 分叉。Chapter 79 因此
Closed。该文件同样是 allocation 记录，不是 BF16 strict parity。
Chapter 80 复用 Q/K scale reshape 之后的一次 repeat 重测是
`qwen3_vl_allocation_qk_scale.json`，SHA256 为
`9105965aca4d2c296b5f41b34f4820af00a60ee277153fb38db3dc0ed129fb50`。
GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 都是 `415,744`
bytes。32-token prefix exact，128/256 仍在 step 70 分叉。Chapter 80 因此
Closed。该文件同样是 allocation 记录，不是 BF16 strict parity。
Chapter 81 把 hidden abs2 的 4 块尾部换成 warp 树之后的一次 repeat 重测是
`qwen3_vl_allocation_hidden_tail.json`，SHA256 为
`73e26bcfc57fea24cd2ba006b1aa51fe2ca5b245bb73f00c32766f6151dd9721`。
GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 都是 `394,768`
bytes。32-token prefix exact，128/256 仍在 step 70 分叉。Chapter 81 因此
Closed。该文件同样是 allocation 记录，不是 BF16 strict parity。
Chapter 82 把 K/V 的两次拷贝和两次写回收成各一次启动之后的一次 repeat
重测是 `qwen3_vl_allocation_kv_pair.json`，SHA256 为
`adbfdda29a4aa59a751cd48f5547822b8d21fd947225efdd5b28225a68e9da6c`。
GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 都是 `388,496`
bytes。32-token prefix exact，128/256 仍在 step 70 分叉。Chapter 82 因此
Closed。该文件同样是 allocation 记录，不是 BF16 strict parity。
Chapter 83 把 Q/K RoPE 收成一次启动之后的一次 repeat 重测是
`qwen3_vl_allocation_rope_pair.json`，SHA256 为
`39f8c6ab8f0320a24c27c7f5af400ea98c45322fe91282be5893fbc96ca80fad`。
GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 都是 `384,464`
bytes。32-token prefix exact，128/256 仍在 step 70 分叉。Chapter 83 因此
Closed。该文件同样是 allocation 记录，不是 BF16 strict parity。
Chapter 84 把 query 载入和 K/V 拷贝收成一次启动之后的一次 repeat 重测是
`qwen3_vl_allocation_attn_prepare.json`，SHA256 为
`fe36b8825d6582c932c9bf264db617d04a1276bff0210987320c78f001d54a35`。
GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 都是 `381,328`
bytes。32-token prefix exact，128/256 仍在 step 70 分叉。Chapter 84 因此
Closed。该文件同样是 allocation 记录，不是 BF16 strict parity。
Chapter 85 把 softmax 权重和它的置换收成一次启动之后的一次 repeat 重测是
`qwen3_vl_allocation_weight_perm.json`，SHA256 为
`922a55bc3533657cc5a09278239cd6e64d20c443147215edd441ab69b58fcda7`。
GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 都是 `377,744`
bytes。32-token prefix exact，128/256 仍在 step 70 分叉。Chapter 85 因此
Closed。该文件同样是 allocation 记录，不是 BF16 strict parity。
Chapter 86 把 Q/K RMSNorm 收成一次启动之后的一次 repeat 重测是
`qwen3_vl_allocation_qk_fused.json`，SHA256 为
`1f6e990c7b7e060edd280755d60ff385279370538e3f81a038e71eeaa78d9f3c`。
GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 都是 `365,648`
bytes。32-token prefix exact，128/256 仍在 step 70 分叉。Chapter 86 因此
Closed。该文件同样是 allocation 记录，不是 BF16 strict parity。
Chapter 87 把 attention softmax 收成一次启动之后的一次 repeat 重测是
`qwen3_vl_allocation_softmax.json`，SHA256 为
`9d41f7461c0387d88d2333f6c9184108380140695af958ca2c6940af7673c331`。
GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 都是 `344,144`
bytes。32-token prefix exact，128/256 仍在 step 70 分叉。Chapter 87 因此
Closed。该文件同样是 allocation 记录，不是 BF16 strict parity。
Chapter 88 复用 mRoPE 频率表之后的一次 repeat 重测是
`qwen3_vl_allocation_mrope.json`，SHA256 为
`9d3a146875250b9dcfa4ded7b7c617620903cd20f21279d71973660b0a9965f5`。
GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 都是 `342,672`
bytes。32-token prefix exact，128/256 仍在 step 70 分叉。Chapter 88 因此
Closed。该文件同样是 allocation 记录，不是 BF16 strict parity。
Chapter 89 让单 token GEMM 直接使用原来的 stride 之后的一次 repeat 重测是
`qwen3_vl_allocation_gemm_token.json`，SHA256 为
`4c20846e26ec45e49530b9e5568a7f0f75f15741a84154ac5834030cd1d896a6`。
GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 都是 `312,208`
bytes。32-token prefix exact，128/256 仍在 step 70 分叉。Chapter 89 因此
Closed。该文件同样是 allocation 记录，不是 BF16 strict parity。
Chapter 90 把长度 `2048` 的 hidden RMSNorm 收成两次 kernel 之后的一次
repeat 重测是 `qwen3_vl_allocation_hidden_fused.json`，SHA256 为
`5cea2add288724b3126c025c3ca3f0ca05352ea5d6f387288516a292d77f6ccf`。
GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 都是 `297,952`
bytes。32-token prefix exact，128/256 仍在 step 70 分叉。Chapter 90 因此
Closed。该文件同样是 allocation 记录，不是 BF16 strict parity。
Chapter 91 把 attention softmax 的 kernel 留在具体 `Ref` 里之后的一次
repeat 重测是 `qwen3_vl_allocation_softmax_launch.json`，SHA256 为
`00534ab94c2fa492f615082b041b007f042cf72e0f907bb2a5f19ec8d0d4ea09`。
GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 都是 `288,544`
bytes。32-token prefix exact，128/256 仍在 step 70 分叉。Chapter 91 因此
Closed。该文件同样是 allocation 记录，不是 BF16 strict parity。
Chapter 92 让 `cublasGemmEx` 不再每次查询 context 之后的一次 repeat 重测是
`qwen3_vl_allocation_gemm_context.json`，SHA256 为
`5ca785fc6469ce8cd9abc963e9a7a362126eb2f01bf3fb490e5bcf272ab708e3`。
GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 都是 `278,224`
bytes。32-token prefix exact，128/256 仍在 step 70 分叉。Chapter 92 因此
Closed。该文件同样是 allocation 记录，不是 BF16 strict parity。
Chapter 93 复用 attention query 的 reshape 之后的一次 repeat 重测是
`qwen3_vl_allocation_query_matrix.json`，SHA256 为
`5337c63eef2bf9bfaacdd1929ee173744b7acc6cfac967f87310299341895fca`。
GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 都是 `275,536`
bytes。32-token prefix exact，128/256 仍在 step 70 分叉。Chapter 93 因此
Closed。该文件同样是 allocation 记录，不是 BF16 strict parity。
Chapter 94 把长度 `2048` 的 hidden RMSNorm 收成一次启动之后的一次 repeat
重测是 `qwen3_vl_allocation_hidden_single.json`，SHA256 为
`9e122b995b02a9c2d85b6b1ae67491aa8ea541adb581160493d00d30f398bd10`。
GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 都是 `266,416`
bytes。32-token prefix exact，128/256 仍在 step 70 分叉。Chapter 94 因此
Closed。该文件同样是 allocation 记录，不是 BF16 strict parity。
Chapter 95 直接启动 attention softmax 之后的一次 repeat 重测是
`qwen3_vl_allocation_softmax_raw.json`，SHA256 为
`c1f8e35ac1c364dd29d19c2814e1198406e3133fee7cec71d215bdea99463d9b`。
GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 都是 `260,144`
bytes。32-token prefix exact，128/256 仍在 step 70 分叉。Chapter 95 因此
Closed。该文件同样是 allocation 记录，不是 BF16 strict parity。
Chapter 96 直接启动 attention 的准备和 context 写回之后的一次 repeat
重测是 `qwen3_vl_allocation_attn_raw.json`，SHA256 为
`24cd654c35811cc9375573995687bbfc47fb207237a28d2b9b1d7d8f0b616896`。
GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 都是 `250,288`
bytes。32-token prefix exact，128/256 仍在 step 70 分叉。Chapter 96 因此
Closed。该文件同样是 allocation 记录，不是 BF16 strict parity。
Chapter 97 直接启动 hidden RMSNorm 之后的一次 repeat 重测是
`qwen3_vl_allocation_hidden_raw.json`，SHA256 为
`dc1a6db24f75ade00da6e847ca52996b11d3770537823b755cf2519003ab08db`。
GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 都是 `240,256`
bytes。32-token prefix exact，128/256 仍在 step 70 分叉。Chapter 97 因此
Closed。该文件同样是 allocation 记录，不是 BF16 strict parity。
Chapter 98 直接启动 residual 加法之后的一次 repeat 重测是
`qwen3_vl_allocation_residual_raw.json`，SHA256 为
`3db90548a5f9e730f543a5db2dd4885817d8b325d2b198407a090d404c3a9d2b`。
GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 都是 `231,296`
bytes。32-token prefix exact，128/256 仍在 step 70 分叉。Chapter 98 因此
Closed。该文件同样是 allocation 记录，不是 BF16 strict parity。
Chapter 99 复用 QKV 的 reshape 之后的一次 repeat 重测是
`qwen3_vl_allocation_qkv_matrix.json`，SHA256 为
`e1f0049928c88d1aa76b452ac8ca9400266bbd526717d9218eb572384036f1bf`。
GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 都是 `223,232`
bytes。32-token prefix exact，128/256 仍在 step 70 分叉。Chapter 99 因此
Closed。该文件同样是 allocation 记录，不是 BF16 strict parity。
Chapter 100 直接启动 QK RMSNorm 之后的一次 repeat 重测是
`qwen3_vl_allocation_qk_raw.json`，SHA256 为
`cc01142c0d717ac61f2e769e90e88ff6f0a6c41a34311fdd33833c0e8ef1e84d`。
GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 都是 `211,776`
bytes。32-token prefix exact，128/256 仍在 step 70 分叉。Chapter 100 因此
Closed。该文件同样是 allocation 记录，不是 BF16 strict parity。
Chapter 101 直接启动 QK RoPE 之后的一次 repeat 重测是
`qwen3_vl_allocation_rope_raw.json`，SHA256 为
`4eb335b744abf30c86693f3faf560c4d6d45b6a4908ac878d414d67bc37fb104`。
GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 都是 `205,952`
bytes。32-token prefix exact，128/256 仍在 step 70 分叉。Chapter 101 因此
Closed。该文件同样是 allocation 记录，不是 BF16 strict parity。
Chapter 102 直接启动 K/V 写入之后的一次 repeat 重测是
`qwen3_vl_allocation_kv_raw.json`，SHA256 为
`545067cbab352dc90671bfa184b221e01c9b51c7cebdd18154a3894459992cc2`。
GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 都是 `201,024`
bytes。32-token prefix exact，128/256 仍在 step 70 分叉。Chapter 102 因此
Closed。该文件同样是 allocation 记录，不是 BF16 strict parity。
Chapter 103 直接启动 MLP SiLU 之后的一次 repeat 重测是
`qwen3_vl_allocation_mlp_raw.json`，SHA256 为
`7b293b12258a3be217777064f9d1b4ad2d3e534c03ff2fb05a952eb0c7d8d576`。
GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 都是 `196,544`
bytes。32-token prefix exact，128/256 仍在 step 70 分叉。Chapter 103 因此
Closed。该文件同样是 allocation 记录，不是 BF16 strict parity。
Chapter 104 在没有 profiler 时直接执行 stage 之后的一次 repeat 重测是
`qwen3_vl_allocation_stage_inline.json`，SHA256 为
`942e68a633d89d896a587e7c7786f03da1a293ef0da916df20f0cf35e2e9e125`。
GPU 最后一步仍是 `0`。position `331` 的 CPU 是 `129,600` bytes，
`107 / 203` 是 `129,616` bytes。32-token prefix exact，128/256 仍在
step 70 分叉。Chapter 104 因此 Closed。该文件同样是 allocation 记录，
不是 BF16 strict parity。
Chapter 105 在没有 profiler 时让单 token 层只走热路径之后的一次 repeat
重测是 `qwen3_vl_allocation_decode_block.json`，SHA256 为
`414590cfcf94ced3dd4d15f83cdd5a392e8bd584d07e2d4f52d6dc516c894135`。
GPU 最后一步仍是 `0`。position `331` 的 CPU 是 `22,080` bytes，
`107 / 203` 是 `22,096` bytes。32-token prefix exact，128/256 仍在
step 70 分叉。Chapter 105 因此 Closed。该文件同样是 allocation 记录，
不是 BF16 strict parity。
Chapter 106 把 28 层收成一次调用之后的一次 repeat 重测是
`qwen3_vl_allocation_decode_layers.json`，SHA256 为
`8e06c526e0bd0c6a12572979034dccd8c0a38475d10a525411b5c1dab547bb6b`。
GPU 最后一步仍是 `0`。position `331` 的 CPU 是 `16,768` bytes，
`107 / 203` 是 `16,784` bytes。32-token prefix exact，128/256 仍在
step 70 分叉。Chapter 106 因此 Closed。该文件同样是 allocation 记录，
不是 BF16 strict parity。
Chapter 107 复用 cache 重叠检查的数组之后的一次 repeat 重测是
`qwen3_vl_allocation_seen_storage.json`，SHA256 为
`0e9e3ec5332667677e006ddda68de743ff39071effd1bdcdae4624ad7eae3273`。
GPU 最后一步仍是 `0`。position `331` 的 CPU 是 `13,344` bytes，
`107 / 203` 是 `13,360` bytes。32-token prefix exact，128/256 仍在
step 70 分叉。Chapter 107 因此 Closed。该文件同样是 allocation 记录，
不是 BF16 strict parity。
Chapter 108 不再在稳态调用里拷贝整份参数之后的一次 repeat 重测是
`qwen3_vl_allocation_decode_pack.json`，SHA256 为
`192477eaa948f93a095264f64205930cbaa87b75f8570e9eee22c25256b4f22b`。
GPU 最后一步仍是 `0`。position `331` 的 CPU 是 `10,272` bytes，
`107 / 203` 是 `10,288` bytes。32-token prefix exact，128/256 仍在
step 70 分叉。Chapter 108 因此 Closed。该文件同样是 allocation 记录，
不是 BF16 strict parity。
Chapter 109 把长生成的 decode 计时放进小函数之后的一次 repeat 重测是
`qwen3_vl_allocation_time_decode.json`，SHA256 为
`d82e5e953538c6d1631e09ac9bfe887325439f4055831627371c1378dbfa74aa`。
GPU 最后一步仍是 `0`。position `331` 的 CPU 是 `7,200` bytes，
`107 / 203` 是 `7,216` bytes。32-token prefix exact，128/256 仍在
step 70 分叉。Chapter 109 因此 Closed。该文件同样是 allocation 记录，
不是 BF16 strict parity。
Chapter 110 直接拷贝 token embedding 的一列之后的一次 repeat 重测是
`qwen3_vl_allocation_embedding_copy.json`，SHA256 为
`ddd117e0443de28401032c581f353bfcdf8f983894b3a2a42ec2d01a5bb3655b`。
GPU 最后一步仍是 `0`。position `331` 的 CPU 是 `5,344` bytes，
`107 / 203` 是 `5,360` bytes。32-token prefix exact，128/256 仍在
step 70 分叉。Chapter 110 因此 Closed。该文件同样是 allocation 记录，
不是 BF16 strict parity。
Chapter 111 让 mRoPE 只接收 `head_dim`、`rope_theta` 和 `mrope_section`
之后的一次 repeat 重测是 `qwen3_vl_allocation_mrope_fields.json`，
SHA256 为
`a0cd3dc851bc37e61cf7e6d978514651fbcf51ae340b3e91ee461bc5d1f6778d`。
GPU 最后一步仍是 `0`。position `331` 的 CPU 是 `5,248` bytes，
`107 / 203` 是 `5,264` bytes。32-token prefix exact，128/256 仍在
step 70 分叉。Chapter 111 因此 Closed。该文件同样是 allocation 记录，
不是 BF16 strict parity。
Chapter 112 把 scratch 留在具体类型的包装里之后的一次 repeat 重测是
`qwen3_vl_allocation_decode_ready.json`，SHA256 为
`232a29ef3afb41f19c443d3aa2a5006cf13691c8e232afb609ea7b2422ed644d`。
GPU 最后一步仍是 `0`。position `331` 的 CPU 是 `4,400` bytes，
`107 / 203` 是 `4,416` bytes。32-token prefix exact，128/256 仍在
step 70 分叉。Chapter 112 因此 Closed。该文件同样是 allocation 记录，
不是 BF16 strict parity。
Chapter 113 直接把 mRoPE 的主机表抄到设备之后的一次 repeat 重测是
`qwen3_vl_allocation_mrope_copy.json`，SHA256 为
`c29c0a498bb5eb7f421effdc7212f3205599d968fedc39e13fce7670602def1d`。
GPU 最后一步仍是 `0`。position `331` 的 CPU 是 `4,336` bytes，
`107 / 203` 是 `4,352` bytes。32-token prefix exact，128/256 仍在
step 70 分叉。Chapter 113 因此 Closed。该文件同样是 allocation 记录，
不是 BF16 strict parity。
Chapter 114 复用 tied logits 的两张矩阵之后的一次 repeat 重测是
`qwen3_vl_allocation_tied_matrix.json`，SHA256 为
`791f7e29a744f4402f061e12dc81ab1c6cc01ee386fa227ff7a04c8415387cb3`。
GPU 最后一步仍是 `0`。position `331` 的 CPU 是 `4,176` bytes，
`107 / 203` 是 `4,192` bytes。32-token prefix exact，128/256 仍在
step 70 分叉。Chapter 114 因此 Closed。该文件同样是 allocation 记录，
不是 BF16 strict parity。
Chapter 115 让稳态循环只进入一次具体包装之后的一次 repeat 重测是
`qwen3_vl_allocation_ready_loop.json`，SHA256 为
`4caaf5d2593f99549eb761f7285bd1d9e4911bba253ed76e33739a2c6145bd3c`。
GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 都是 `4,144`
bytes。32-token prefix exact，128/256 仍在 step 70 分叉。Chapter 115
因此 Closed。该文件同样是 allocation 记录，不是 BF16 strict parity。
Chapter 116 让稳态计时不再包一层 `@sync` 之后的一次 repeat 重测是
`qwen3_vl_allocation_ready_timer.json`，SHA256 为
`36246c5cc9e2d2e47fc1392db105143c6375c87f46109ac803cb65f41699576a`。
GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 都是 `0`
bytes。32-token prefix exact，128/256 仍在 step 70 分叉。Chapter 116
因此 Closed。该文件同样是 allocation 记录，不是 BF16 strict parity。
Chapter 117 用线性扫描代替主机选择的全长排序之后的一次 repeat 重测是
`qwen3_vl_allocation_top_two.json`，SHA256 为
`978dd531c9f64d4f0deda0237ce67d62db75712d312bb6406826da636112a02e`。
GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 仍是 `0`
bytes。256-token 主机选择从 `2,739,400` 降到 `308,232` bytes。
32-token prefix exact，128/256 仍在 step 70 分叉。Chapter 117
因此 Closed。该文件同样是 allocation 记录，不是 BF16 strict parity。
Chapter 118 复用一块主机 logits 缓冲之后的一次 repeat 重测是
`qwen3_vl_allocation_logit_copy.json`，SHA256 为
`1a5ee12eadee76b467864460e202d2a371532eb5637337e1c06056abc882722c`。
GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 仍是 `0`
bytes。256-token 主机选择从 `308,232` 降到 `0` bytes。
32-token prefix exact，128/256 仍在 step 70 分叉。Chapter 118
因此 Closed。该文件同样是 allocation 记录，不是 BF16 strict parity。
Chapter 119 让第一次 decode 沿用 pack 里的 blocks 之后的一次 repeat
重测是 `qwen3_vl_allocation_ready_install.json`，SHA256 为
`ec735216c159c84be0712a9bbe46c35df4600d4bf40e5a5eb7c35667f0d1bbc2`。
GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 仍是 `0`
bytes。128/256 的第一次 decode CPU 从 `40,368` 降到 `37,664` bytes。
32-token prefix exact，128/256 仍在 step 70 分叉。Chapter 119
因此 Closed。该文件同样是 allocation 记录，不是 BF16 strict parity。
Chapter 120 让 attention scratch 的字节核对不再装箱之后的一次 repeat
重测是 `qwen3_vl_allocation_attention_bytes.json`，SHA256 为
`16645b1138ca6369ca3a0b386ad53422680ae645765a2cc1addbf27d76abebcf`。
GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 仍是 `0`
bytes。128/256 的第一次 decode CPU 从 `37,664` 降到 `35,904` bytes。
32-token prefix exact，128/256 仍在 step 70 分叉。Chapter 120
因此 Closed。该文件同样是 allocation 记录，不是 BF16 strict parity。
Chapter 121 让 attention scratch 的 13 块数组共用两块存储之后的一次
repeat 重测是 `qwen3_vl_allocation_attention_pack.json`，SHA256 为
`bab6018c177ba1083c6aa1a7e956180d6156b582e0bdd317fa53d65f1f800708`。
GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 仍是 `0`
bytes。128/256 的第一次 decode CPU 从 `35,904` 降到 `34,304` bytes。
256-token 第一次 decode 的 GPU 字节仍是 `2,018,180`，分配次数从 `56`
降到 `45`。32-token prefix exact，128/256 仍在 step 70 分叉。
Chapter 121 因此 Closed。该文件同样是 allocation 记录，不是 BF16
strict parity。
Chapter 122 让 QK RoPE scratch 的 12 块数组共用两块存储之后的一次
repeat 重测是 `qwen3_vl_allocation_rope_pack.json`，SHA256 为
`1a3c49ac4089ae92e1365725899c42fdea28cc98328b0112816e47348944ad97`。
GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 仍是 `0`
bytes。128/256 的第一次 decode CPU 从 `34,304` 降到 `32,864` bytes。
256-token 第一次 decode 的 GPU 字节仍是 `2,018,180`，分配次数从 `45`
降到 `35`。32-token prefix exact，128/256 仍在 step 70 分叉。
Chapter 122 因此 Closed。该文件同样是 allocation 记录，不是 BF16
strict parity。
Chapter 123 让 QK norm scratch 的 7 块数组共用两块存储之后的一次
repeat 重测是 `qwen3_vl_allocation_qk_norm_pack.json`，SHA256 为
`802f5ab3a2f4499dfff4f5b7de66b639145ae70a950128ec592976e69edf434f`。
GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 仍是 `0`
bytes。128/256 的第一次 decode CPU 从 `32,864` 降到 `32,224` bytes。
256-token 第一次 decode 的 GPU 字节仍是 `2,018,180`，分配次数从 `35`
降到 `30`。32-token prefix exact，128/256 仍在 step 70 分叉。
Chapter 123 因此 Closed。该文件同样是 allocation 记录，不是 BF16
strict parity。
Chapter 124 让 hidden norm scratch 的 6 块数组共用两块存储之后的一次
repeat 重测是 `qwen3_vl_allocation_hidden_norm_pack.json`，SHA256 为
`02074e31b3194d537e593cc6289178fb862cd088544710a6d8958db55a75bccc`。
GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 仍是 `0`
bytes。128/256 的第一次 decode CPU 从 `32,224` 降到 `31,744` bytes。
256-token 第一次 decode 的 GPU 字节仍是 `2,018,180`，分配次数从 `30`
降到 `26`。32-token prefix exact，128/256 仍在 step 70 分叉。
Chapter 124 因此 Closed。该文件同样是 allocation 记录，不是 BF16
strict parity。
Chapter 125 让 MLP activation scratch 的 5 块数组共用两块存储之后的一次
repeat 重测是 `qwen3_vl_allocation_mlp_pack.json`，SHA256 为
`7ffc03b19d4421bd06c3a98c583664b2765261adeab657554d01bfdfde9353d2`。
GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 仍是 `0`
bytes。128/256 的第一次 decode CPU 从 `31,744` 降到 `31,424` bytes。
256-token 第一次 decode 的 GPU 字节仍是 `2,018,180`，分配次数从 `26`
降到 `23`。32-token prefix exact，128/256 仍在 step 70 分叉。
Chapter 125 因此 Closed。该文件同样是 allocation 记录，不是 BF16
strict parity。
Chapter 126 让 hidden residual scratch 的 4 块数组共用两块存储之后的一次
repeat 重测是 `qwen3_vl_allocation_residual_pack.json`，SHA256 为
`6db4aa2978fc8e80b064352f4968d6ccbeb236c6057f852244907eb84cff5336`。
GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 仍是 `0`
bytes。128/256 的第一次 decode CPU 从 `31,424` 降到 `31,264` bytes。
256-token 第一次 decode 的 GPU 字节仍是 `2,018,180`，分配次数从 `23`
降到 `21`。32-token prefix exact，128/256 仍在 step 70 分叉。
Chapter 126 因此 Closed。该文件同样是 allocation 记录，不是 BF16
strict parity。
Chapter 127 让 ready 只留着 pack、不再拷贝 blocks 之后的一次 repeat
重测是 `qwen3_vl_allocation_ready_pack.json`，SHA256 为
`e5f49acc926b4b205186f93ed9f42c8e3c21583ffc5f1b4c01085e6f6594ed73`。
GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 仍是 `0`
bytes。128/256 的第一次 decode CPU 从 `31,264` 降到 `25,856` bytes。
256-token 第一次 decode 的 GPU 字节仍是 `2,018,180`，分配次数仍是
`21`。32-token prefix exact，128/256 仍在 step 70 分叉。Chapter 127
因此 Closed。该文件同样是 allocation 记录，不是 BF16 strict parity。
Chapter 128 让第一次 decode 的计时不再包一层 `@sync` 之后的一次
repeat 重测是 `qwen3_vl_allocation_first_timer.json`，SHA256 为
`ee02b9b657eaef91346f255c6c2401ff1ba2a4e92f5756d6f8ef4952e7ce4016`。
GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 仍是 `0`
bytes。128/256 的第一次 decode CPU 从 `25,856` 降到 `21,712` bytes。
256-token 第一次 decode 的 GPU 字节仍是 `2,018,180`，分配次数仍是
`21`。32-token prefix exact，128/256 仍在 step 70 分叉。Chapter 128
因此 Closed。该文件同样是 allocation 记录，不是 BF16 strict parity。
Chapter 129 让参数 pack 随 cache 创建之后的一次 repeat 重测是
`qwen3_vl_allocation_cache_pack.json`，SHA256 为
`137d69319e472b2faaf15ca5e89b097ebf87d09142d668a64fec64ba12b677c5`。
GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 仍是 `0`
bytes。128/256 的第一次 decode CPU 从 `21,712` 降到 `19,024` bytes。
256-token 第一次 decode 的 GPU 字节仍是 `2,018,180`，分配次数仍是
`21`。32-token prefix exact，128/256 仍在 step 70 分叉。Chapter 129
因此 Closed。该文件同样是 allocation 记录，不是 BF16 strict parity。
Chapter 130 让 QKV、gate/up、mRoPE、token embedding 和 tied logits
这 9 块 BF16 数组共用一块存储之后的一次 repeat 重测是
`qwen3_vl_allocation_bf16_tails.json`，SHA256 为
`31ee2e77b9a8bb327e803e73521791ba66eff6e5abb6e0b6770b89ca7b89c469`。
GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 仍是 `0`
bytes。128/256 的第一次 decode CPU 从 `19,024` 降到 `17,872` bytes。
256-token 第一次 decode 的 GPU 字节仍是 `2,018,180`，分配次数从
`21` 降到 `13`。32-token prefix exact，128/256 仍在 step 70 分叉。Chapter 130
因此 Closed。该文件同样是 allocation 记录，不是 BF16 strict parity。
Chapter 131 让全部 decode scratch 切自一块 BF16 和一块 Float32 之后的一次
repeat 重测是 `qwen3_vl_allocation_decode_workspace.json`，SHA256 为
`2bc4e279c7355bd436203dd0982ce47b88b07c837d99911eecab08a0f49b8487`。
GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 仍是 `0`
bytes。128/256 的第一次 decode CPU 从 `17,872` 降到 `14,912` bytes。
256-token 第一次 decode 的 GPU 字节仍是 `2,018,180`，分配次数从
`13` 降到 `2`。32-token prefix exact，128/256 仍在 step 70 分叉。Chapter 131
因此 Closed。该文件同样是 allocation 记录，不是 BF16 strict parity。
Chapter 132 让新的 scratch 复用已经建好的 kernel 参数盒之后的一次
repeat 重测是 `qwen3_vl_allocation_param_reuse.json`，SHA256 为
`1fb6be933ff633abbcd6c6126cf3924496534843d9871915d929efd07d97138b`。
GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 仍是 `0`
bytes。128/256 的第一次 decode CPU 从 `14,912` 降到 `9,168` bytes。
256-token 第一次 decode 的 GPU 字节仍是 `2,018,180`，分配次数仍是
`2`。32-token prefix exact，128/256 仍在 step 70 分叉。Chapter 132
因此 Closed。该文件同样是 allocation 记录，不是 BF16 strict parity。
Chapter 133 让 scratch 切片共用父缓冲的引用之后的一次 repeat 重测是
`qwen3_vl_allocation_shared_ref.json`，SHA256 为
`103dc1d18457d27054ddad1215f1d0b777b98694747e9b426f5101e0cc9230dd`。
GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 仍是 `0`
bytes。32/128/256 的第一次 decode CPU 都从 `9,168` 降到 `7,408` bytes。
256-token 第一次 decode 的 GPU 字节仍是 `2,018,180`，分配次数仍是
`2`。32-token prefix exact，128/256 仍在 step 70 分叉。Chapter 133
因此 Closed。该文件同样是 allocation 记录，不是 BF16 strict parity。
Chapter 134 让 QKV、tied、query 和 attention 投影复用已经建好的
reshape 之后的一次 repeat 重测是 `qwen3_vl_allocation_reshape_reuse.json`，
SHA256 为
`92cce38e05d6123d6f55f4a520cbc4c7df36ce3be53339690296fafa904cb17a`。
GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 仍是 `0`
bytes。32/128/256 的第一次 decode CPU 都从 `7,408` 降到 `6,768` bytes。
256-token 第一次 decode 的 GPU 字节仍是 `2,018,180`，分配次数仍是
`2`。32-token prefix exact，128/256 仍在 step 70 分叉。Chapter 134
因此 Closed。该文件同样是 allocation 记录，不是 BF16 strict parity。
Chapter 135 让后来的请求复用第一次 decode 的 56 张 scratch 切片之后的
一次 repeat 重测是 `qwen3_vl_allocation_slice_reuse.json`，SHA256 为
`526066a3cdf57673e5384c36a625c1acd2861a451a5dd47a24c1c5f9af48640b`。
GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 仍是 `0`
bytes。32/128/256 的第一次 decode CPU 都从 `6,768` 降到 `3,264` bytes。
256-token 第一次 decode 的 GPU 字节仍是 `2,018,180`，分配次数仍是
`2`。32-token prefix exact，128/256 仍在 step 70 分叉。Chapter 135
因此 Closed。该文件同样是 allocation 记录，不是 BF16 strict parity。
Chapter 136 让后来的请求把复用的 scratch 直接装进 Ready 之后的一次
repeat 重测是 `qwen3_vl_allocation_shell_ready.json`，SHA256 为
`1cad8aca2a56b0149cd8533d45bb3d3230e1b16529216408bf1c8caa408454bf`。
GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 仍是 `0`
bytes。32/128/256 的第一次 decode CPU 都从 `3,264` 降到 `1,312` bytes。
256-token 第一次 decode 的 GPU 字节仍是 `2,018,180`，分配次数仍是
`2`。32-token prefix exact，128/256 仍在 step 70 分叉。Chapter 136
因此 Closed。该文件同样是 allocation 记录，不是 BF16 strict parity。
Chapter 137 让后来的请求复用同一个 Ready 之后的一次 repeat 重测是
`qwen3_vl_allocation_ready_reuse.json`，SHA256 为
`72409ccf67d9ceb0848d2a1de2fc82e7aa0a2c654ac81a66d336faa3a19e1365`。
GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 仍是 `0`
bytes。32/128/256 的第一次 decode CPU 都从 `1,312` 降到 `800` bytes。
256-token 第一次 decode 的 GPU 字节仍是 `2,018,180`，分配次数仍是
`2`。32-token prefix exact，128/256 仍在 step 70 分叉。Chapter 137
因此 Closed。该文件同样是 allocation 记录，不是 BF16 strict parity。
Chapter 138 让后来的请求在已有父缓冲装得下时不再重新分配之后的一次
repeat 重测是 `qwen3_vl_allocation_parent_reuse.json`，SHA256 为
`1111c98b93bca2cd344bb4cc4468898b8d2a77fb9b5dc4ceebb21e48b80a3ccd`。
GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 仍是 `0`
bytes。32/128/256 的第一次 decode CPU 都从 `800` 降到 `352` bytes。
这三次的第一次 decode GPU 字节都降到 `0`，分配次数都从 `2` 降到 `0`。
32-token prefix exact，128/256 仍在 step 70 分叉。Chapter 138
因此 Closed。该文件同样是 allocation 记录，不是 BF16 strict parity。
Chapter 139 让后来的请求原地改 attention scratch 的 `capacity` 之后的
一次 repeat 重测是 `qwen3_vl_allocation_attention_capacity.json`，
SHA256 为
`d3597accee924dba321e9252a6f9411fbd6838152971d5a9c28302ba1a3fca42`。
GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 仍是 `0`
bytes。32/128/256 的第一次 decode CPU 都从 `352` 降到 `192` bytes。
第一次 decode 的 GPU 字节仍是 `0`，分配次数仍是 `0`。32-token prefix
exact，128/256 仍在 step 70 分叉。Chapter 139 因此 Closed。该文件同样是
allocation 记录，不是 BF16 strict parity。
Chapter 140 让复用入口不再装箱 text spec 之后的一次 repeat 重测是
`qwen3_vl_allocation_reuse_entry.json`，SHA256 为
`65ce9f34941b20837d9aeae43badbf13e7bc03366a1ba0c1a26ce8303bda0bf4`。
GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 仍是 `0`
bytes。32/128/256 的第一次 decode CPU 都从 `192` 降到 `48` bytes。
第一次 decode 的 GPU 字节仍是 `0`，分配次数仍是 `0`。32-token prefix
exact，128/256 仍在 step 70 分叉。Chapter 140 因此 Closed。该文件同样是
allocation 记录，不是 BF16 strict parity。
Chapter 141 用 BF16 Ready 的具体类型把 steady 调用收成静态调用之后的
一次 repeat 重测是 `qwen3_vl_allocation_static_ready.json`，SHA256 为
`f9a66705ccb388606f478839ac20770733c0c008905ccdcd53f35ee599362831`。
GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 仍是 `0`
bytes。32/128/256 的第一次 decode CPU 都从 `48` 降到 `32` bytes。
第一次 decode 的 GPU 字节仍是 `0`，分配次数仍是 `0`。32-token prefix
exact，128/256 仍在 step 70 分叉。Chapter 141 因此 Closed。该文件同样是
allocation 记录，不是 BF16 strict parity。
Chapter 142 让第一次 decode 的计时窗口和稳态步一样，直接调用具体 Ready
之后的一次 repeat 重测是 `qwen3_vl_allocation_ready_call.json`，SHA256 为
`c838c5f037c16a35820eb45aa8fbe7d93139a519ab7fac1f3946e6947053ccc6`。
GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 仍是 `0`
bytes。32/128/256 的第一次 decode CPU 都从 `32` 降到 `0` bytes。
第一次 decode 的 GPU 字节仍是 `0`，分配次数仍是 `0`。32-token prefix
exact，128/256 仍在 step 70 分叉。Chapter 142 因此 Closed。该文件同样是
allocation 记录，不是 BF16 strict parity。
Chapter 143 让 GPU 上的 BF16 转 Float32 直接 `copyto!` 之后的一次 repeat
重测是 `qwen3_vl_allocation_f32_copy.json`，SHA256 为
`f132e3624ccb19b175babe19eafd447c3300356e738edeada965fc11a6a4f0ad`。
GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 仍是 `0`
bytes。32/128/256 的第一次 decode CPU 仍是 `0` bytes。256 的 prefill
CPU 从 `8,007,336` 降到 `6,918,616` bytes，prefill GPU 仍是
`858,377,949` bytes、`4240` 次。32-token prefix exact，128/256 仍在
step 70 分叉。Chapter 143 因此 Closed。该文件同样是 allocation 记录，
不是 BF16 strict parity。
Chapter 144 让连续 GPU 数组的 BF16 转 Float32 复用同一个 kernel 之后的
一次 repeat 重测是 `qwen3_vl_allocation_promote.json`，SHA256 为
`25f498529f6e84e7a8e2902b5e02c79e5fb62519a1472e5e015d8b0fcb9ab0a7`。
GPU 最后一步仍是 `0`。position `107 / 203 / 331` 的 CPU 仍是 `0`
bytes。32/128/256 的第一次 decode CPU 仍是 `0` bytes。256 的 prefill
CPU 从 `6,918,616` 降到 `5,657,880` bytes，prefill GPU 仍是
`858,377,949` bytes、`4240` 次。32-token prefix exact，128/256 仍在
step 70 分叉。Chapter 144 因此 Closed。该文件同样是 allocation 记录，
不是 BF16 strict parity。

默认 workload 为 32/128/256-token BF16 static generation、每个长度三次；可用
`LIFEAI_QWEN3_VL_PROFILE_LENGTHS` 和 `LIFEAI_QWEN3_VL_PROFILE_REPEATS`
显式缩小 smoke（每个长度至少 4，且不超过 oracle timeline），但缩小结果不能关闭
Chapter 47。脚本在 prefix exact 之后才输出累计 allocation traffic、
allocation count、CUDA pool high-water、host greedy selection、无 hook latency 和
最终 step 的逐 stage attribution。默认路径在 token 分叉时仍停在 warmup；
`LIFEAI_QWEN3_VL_PROFILE_RECORD_DIVERGENCE=true` 的非验收剖面会继续归因。

## Qwen3-30B-A3B 资产状态

截至 2026-08-12，本机已下载 `Qwen/Qwen3-30B-A3B` 完整权重，并按
官方 Hugging Face immutable revision 的冻结清单通过逐文件 SHA256：

```text
ad44e777bcd18fa416d9da3bd8f70d33ebb85d39
```

持久目录为：

```text
/home/ubuntu/models/huggingface/Qwen/Qwen3-30B-A3B/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39/
```

该 checkpoint 包含 16 个 BF16 safetensors 分片，文件共
`61,066,575,648` bytes；index 记录 `18,867` 个 tensor、
`61,064,245,248` tensor bytes，即 `30,532,122,624` 个参数。config、
index 以及全部分片的大小和 SHA256 位于
`test/episodes/episode06_qwen3_moe_and_model_expansion/chapter24_qwen3_moe_architecture/fixtures/qwen3_moe_real_checkpoint/assets.json`。

### 中国大陆下载路径

2026-08-12 在本机对同一个官方分片实测：Hugging Face Xet high-performance
模式等待约一分钟仍未开始传输；ModelScope 国内站单文件顺序下载稳定约
`9—13 MB/s`。增加文件并发或单文件 Range 并发会让代理链路频繁断流，综合
吞吐反而更低，因此本机采用单 worker。ModelScope `master` 上的
`config.json`、index 和 16 个 safetensors 分片已逐项与本仓库冻结的
Hugging Face revision 对比，18 个关键文件的 byte size 和 SHA256 全部一致。

安装轻量官方客户端下载：

```bash
python3 -m venv /path/to/modelscope-download-venv
/path/to/modelscope-download-venv/bin/pip install modelscope-hub==0.2.0
```

下载到持久模型目录；同一命令可安全断点续传：

```bash
MODELSCOPE_DOWNLOAD_PARALLEL_WORKERS=1 \
MODELSCOPE_DOWNLOAD_TIMEOUT=120 \
MODELSCOPE_DOWNLOAD_MAX_RETRIES=10 \
/path/to/modelscope-download-venv/bin/ms-hub download \
  Qwen/Qwen3-30B-A3B \
  --local-dir MODEL_DIR \
  --max-workers 1
```

ModelScope 没有对应 Hugging Face commit hash 的命名契约，所以不能只凭模型名
信任下载结果。完成后仍必须运行下面的仓库校验器；只有 18 个文件全部通过冻结
SHA256，才把它视为 revision
`ad44e777bcd18fa416d9da3bd8f70d33ebb85d39` 的等价本地资产。

下载到持久目录后，先运行快速结构检查，再运行完整 61 GB 顺序哈希：

```bash
julia --project=. --startup-file=no \
  scripts/verify_qwen3_moe_checkpoint.jl MODEL_DIR --fast

julia --project=. --startup-file=no \
  scripts/verify_qwen3_moe_checkpoint.jl MODEL_DIR asset-report.json
```

设置 `LIFEAI_QWEN3_30B_A3B_MODEL_DIR=MODEL_DIR` 后，默认测试会启用同一
完整资产校验。本次快速结构检查耗时 `4.20 s`；完整 16 分片 SHA256 检查
耗时 `133.35 s`，报告中的 `shard_checksums_verified=true`。完整报告保存在
该 revision 目录的 `lifeai-asset-report.json`。这证明本地资产与冻结 revision
等价。真实逐层/logits/cache parity 与峰值内存也已在同日完成，reference
保存在：

```text
MODEL_DIR/lifeai-references/chapter24-real-parity/bfloat16/
MODEL_DIR/lifeai-references/chapter24-real-parity/float32/
```

每个目录包含 `reference.json`、`reference.safetensors`、`parity.json` 与
`lifeai-time.txt`。reference 由 Transformers 4.51.3 的官方 decoder layer
逐层加载生成；LifeAI 也逐层只读取当前路由 active experts。Float32 的
`1,152 / 1,152` 路由槽位一致，prompt/decode logits max-abs 为
`4.39e-5 / 1.53e-5`；native BF16 路由槽位重合 `95.92%`，两组 logits
max-abs 均为 `0.3125`，两种口径 argmax 全一致。完整摘要与 checksum 位于
`benchmark_results/qwen3_moe_real_parity/summary.json`。

Chapter 25 的 RTX 4090 D resident/offload 复现实验直接复用 BF16 reference：

```bash
julia --project=. scripts/benchmark_qwen3_moe_cuda_offload.jl \
  MODEL_DIR \
  MODEL_DIR/lifeai-references/chapter24-real-parity/bfloat16 \
  /tmp/qwen3_moe_cuda_offload_grouped.json \
  true

julia --project=. scripts/benchmark_qwen3_moe_cuda_offload.jl \
  MODEL_DIR \
  MODEL_DIR/lifeai-references/chapter24-real-parity/bfloat16 \
  /tmp/qwen3_moe_cuda_offload_scalar.json \
  false
```

两条命令都会实际分配 40,960-token BF16 static KV。可再追加第 5 个参数
`PROMPT_REPEAT`，例如 `true 16` 把冻结的两 token pattern 扩为 32-token
宽 prefill。仓库只提交精简结果
`benchmark_results/qwen3_moe_cuda_offload/summary.json`；原始逐次报告仍留在
本机临时目录。

Chapter 26 在同一模型、reference 与 40K-capacity session 上复现 device
expert LRU；最后一个参数是 cache GiB：

```bash
julia --project=. scripts/benchmark_qwen3_moe_cuda_expert_cache.jl \
  MODEL_DIR \
  MODEL_DIR/lifeai-references/chapter24-real-parity/bfloat16 \
  /tmp/qwen3_moe_cuda_expert_cache_8g.json \
  8
```

该命令依次运行 cold fill/first hit、只清空 expert cache、warm fill/warm hit，
不会在中途重载 resident tree、40K KV、compiled kernels 或 allocator pool。冻结
摘要为 `benchmark_results/qwen3_moe_cuda_expert_cache/summary.json`：8 GiB
预算实际缓存 892 个 expert-layer entries、`8,417,968,128` bytes，重复请求
read/upload 为零，request 相对 warm fill 加速 `1.722×`。这个零淘汰结论只
绑定冻结的 2-token prompt + 1-token decode；任意请求不作零淘汰保证。

Chapter 27 直接使用模型目录里的冻结 tokenizer，运行一次 resident load 后在
同一 40K-capacity session 内切换 global 8 GiB 与 layer-balanced 4/6/8 GiB：

```bash
julia --project=. \
  scripts/benchmark_qwen3_moe_cuda_layer_balanced_cache.jl \
  MODEL_DIR \
  /tmp/qwen3_moe_cuda_layer_balanced_cache.json
```

默认 workload 固定为 `English32 → Chinese32 → English32`，每条 prompt 后
greedy decode 1 token；文本、token ids 与 tokenizer SHA256 均写入报告。脚本
先用 2-token prompt 编译路径，再只清空逻辑 expert cache 并依次 sweep，避免
重复 resident load/compile。精简摘要位于
`benchmark_results/qwen3_moe_cuda_layer_balanced_cache/summary.json`：同为
8 GiB 时 layer-balanced 相对 global 少读 `18.71%`、加速 `1.082×`；冻结
trace 推荐 4 GiB，因为它只比 balanced 8 GiB 慢 `1.48%`，却保留
`7,440,629,760` bytes GPU free。该推荐不泛化到任意 workload。

Chapter 28 复用同一 BF16 reference 和 8 GiB cache，在一次 resident load 中
依次比较 materialized/scattered dispatch 与 1/8/0 层 forced-GC interval：

```bash
julia --project=. \
  scripts/benchmark_qwen3_moe_cuda_scattered_cache.jl \
  MODEL_DIR \
  MODEL_DIR/lifeai-references/chapter24-real-parity/bfloat16 \
  /tmp/qwen3_moe_cuda_scattered_cache.json
```

脚本默认对 `scattered + gc0` 额外运行 100 次全命中请求，用于发现用显存回收
换最低单次延迟的问题；可通过 `LIFEAI_MOE_SCATTERED_REPEAT_HITS` 改变次数。
精简摘要位于 `benchmark_results/qwen3_moe_cuda_scattered_cache/summary.json`：
推荐 `scattered + gc8` 将冻结 hit request 从 `12.477 s` 降到 `0.151 s`
（`82.89×`），同时消除 `10,550,771,712` active materialization bytes。
`gc0` 的 100 次结果虽中位 `0.109 s`，但 allocator free 下降 `0.9375 GiB`，
因此只保留为负面证据，不作为长期服务建议。

Chapter 29 在同一 session contract 上验证有界 pointer-plan/workspace reuse；默认
运行 `gc8 × 100` 和 `gc0 × 500` 次全命中请求：

```bash
julia --project=. \
  scripts/benchmark_qwen3_moe_cuda_scattered_reuse.jl \
  MODEL_DIR \
  MODEL_DIR/lifeai-references/chapter24-real-parity/bfloat16 \
  /tmp/qwen3_moe_cuda_scattered_reuse.json
```

可用 `LIFEAI_MOE_REUSE_GC8_REPEATS` 和 `LIFEAI_MOE_REUSE_GC0_REPEATS`
调整窗口。每 1/10 窗口记录一次 GPU free，末尾同步后再显式 GC/reclaim，从而
区分 allocator pool 高水位和 live leak。冻结摘要位于
`benchmark_results/qwen3_moe_cuda_scattered_reuse/summary.json`：真实 hit 的
96 次调用均零 pointer build/upload 与 workspace allocation；`gc8` 100 次 free
不下降。`gc0` 500 次运行中 pool 扩张 `885,915,648` bytes，但 reclaim 后完全
恢复，说明没有 live leak、仍存在尾延迟/高水位，长期建议继续使用 `gc8`。

Chapter 30 继续复用同一模型/reference/session，要求 Julia 多线程，并比较
sequential、overlapped pageable 与 overlapped pinned 三条 cold device-cache
路径：

```bash
julia --threads=4 --project=. \
  scripts/benchmark_qwen3_moe_cuda_async_miss_pipeline.jl \
  MODEL_DIR \
  MODEL_DIR/lifeai-references/chapter24-real-parity/bfloat16 \
  /tmp/qwen3_moe_cuda_async_miss_pipeline.json
```

默认每种配置先 warmup，再交错运行 3 次；可用 `LIFEAI_MOE_READ_WORKERS` 和
`LIFEAI_MOE_MISS_REPETITIONS` 改变 worker/重复数。脚本不会 drop OS page
cache，因此只测 warm-page-cache miss path。冻结摘要位于
`benchmark_results/qwen3_moe_cuda_async_miss_pipeline/summary.json`：4-worker
pageable overlap 的 request median 为 `5.423 s`，相对 sequential 的
`10.677 s` 加速 `1.969×`；pinned 为 `5.481 s`，略慢于 pageable。三路
warmup/measured logits 全 exact，pinned 的 8.418 GB 注册/解绑生命周期也已验证。

Chapter 31 固定 Chapter 27 的 English32 自然文本，使用 4 GiB layer-balanced
cache 与 scattered dispatch，比较 1/2/4/8 workers。脚本要求 8 Julia threads：

```bash
julia --threads=8 --project=. \
  scripts/benchmark_qwen3_moe_cuda_read_worker_sweep.jl \
  MODEL_DIR \
  /tmp/qwen3_moe_cuda_read_worker_sweep.json
```

每个 cold 请求前对全部 16 个分片调用 `POSIX_FADV_DONTNEED`，并从
`/proc/self/io` 记录实际 storage-read bytes；随后的 revisit 只清 device cache。
冻结报告位于 `benchmark_results/qwen3_moe_cuda_read_worker_sweep/summary.json`：
cold 的 1/2/4/8-worker 中位为 `48.181 / 30.614 / 20.737 / 16.962 s`，
revisit 为 `37.349 / 23.316 / 17.015 / 14.553 s`。cold 每轮实际读存储约
`26.236 GB`，revisit 为 `0`；所有输出 exact。overlapped pipeline 未显式指定
worker 时现在使用 `min(8, Threads.nthreads())`，仍可通过 API 显式覆盖。

Chapter 32 在完全相同的 English32、4 GiB layer-balanced、scattered、gc8 与
8-worker 条件下，比较独立 tensor、shared-open 和严格相邻 coalesced reads：

```bash
julia --threads=8 --project=. \
  scripts/benchmark_qwen3_moe_cuda_coalesced_reads.jl \
  MODEL_DIR \
  /tmp/qwen3_moe_cuda_coalesced_reads.json
```

脚本按三种不同顺序各运行一轮，每个配置均包含
`POSIX_FADV_DONTNEED` cold 与零 storage-read revisit，并记录 `/proc/self/io`
及 Julia allocation/GC。冻结报告位于
`benchmark_results/qwen3_moe_cuda_coalesced_reads/summary.json`：coalesced
把 cold/revisit read syscalls 从 `9,413 / 9,368` 降到 `3,304 / 3,298`
（减少 `64.90% / 64.80%`），但 request median 从 `17.122 / 14.589 s`
恶化到 `21.221 / 19.488 s`。shared-open 为 `17.147 / 14.586 s`，与 tensor
无可归因差异。因此 `expert_read_mode` 默认保持 `:tensor`；另外两种模式只作
显式实验选项，不作为性能建议。

Chapter 33 保持 Chapter 32 的 tensor 粒度与 8-worker overlap，只移除多维
safetensors decode 中在最终 `permutedims` 前的线性同 dtype copy：

```bash
julia --threads=8 --project=. \
  scripts/benchmark_qwen3_moe_cuda_decode_copy_elision.jl \
  MODEL_DIR \
  /tmp/qwen3_moe_cuda_decode_copy_elision.json
```

脚本运行 3 个 cold/revisit pair，并以 Chapter 32 tensor 三轮结果及其报告
SHA256 为旧路径基线。冻结报告位于
`benchmark_results/qwen3_moe_cuda_decode_copy_elision/summary.json`：cold/revisit
allocation 从 `86.326 GB` 降为 `57.646 GB`，均减少 `33.22%`；消失的约
`28.680 GB` 与请求 logical BF16 payload 逐 byte 对齐。cold latency 为
`16.540 s`（基线 `17.122 s`），revisit 为 `14.882 s`（基线 `14.589 s`）；
因不是同进程交错 A/B，不把这组微小且方向不一致的 latency 变化泛化为 speedup。
