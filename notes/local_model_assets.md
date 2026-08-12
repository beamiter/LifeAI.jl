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
