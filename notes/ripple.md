# Ricursive Ripple-65B Preview

LifeAI 原生 Julia 推理实现，对照
[官方发布说明](https://www.ricursive.com/ripple) 和
[官方模型仓库](https://huggingface.co/ricursiveAI/Ripple-65B-Preview/tree/e27e171792f303d74c048184e8259ba663cfa0d5)。
参考版本固定为 `e27e171792f303d74c048184e8259ba663cfa0d5`。
这是 Ricursive Intelligence 的 Ripple，区别于 RecursiveSI 的 Flint。

## 实现

- `src/io/hf_ripple.jl`：`RippleConfig`、完整张量契约、单文件／分片 safetensors 流式加载。
- `src/models/ripple.jl`：四次 RMSNorm 的 PeriNorm、GQA、逐元素 sigmoid attention gate、局部 RoPE／全局 NoPE、非对称 FFN clipping、embedding scale、value residual、缓存生成。
- `src/data/hf_ripple_tokenizer.jl`：原始字节 BPE、`Removed + invert=true` 分词规则、与词表重叠的 added tokens、官方文本／工具／reasoning 模板。
- `src/models/ripple_offload.jl`：按字节预算保留 GPU 权重，其他权重按当前层上传；原始 BF16 checkpoint 也可按层流式运行。
- `src/io/ripple_quantized.jl`：完整 checkpoint 的一次性量化、可复用文件格式、逐文件 SHA-256 校验和按预算加载。
- `scripts/run_ripple.jl`：CPU/CUDA 命令行生成，支持保存后的量化模型和 `--resident-gib`。

Value residual 的来源是**经过 embed_scale 的原始 embedding**，在每层
`v_proj` **之前**与该层 pre-attention normalized hidden 混合：
`seed + sigmoid(value_res_lambda) * (normalized_hidden - seed)`。
它不是第一层投影后的 V，也不是上一层的 hidden。
FFN gate 只裁剪上界，up projection 同时裁剪上下界。
所有层的 K/V 均不重复存储 GQA query heads；局部层仅保留 `window - 1` 个历史位置。

复用既有设备投影、RoPE、RMSNorm 和 CUDA attention/KV primitives；
Ripple 的 softmax 不包含 Flint 的 learnable sinks，复用时把 sink 设为 `-Inf`。
CPU reference 的 attention 会建立 scores 数组；CUDA 复用有界 scratch 的 attention。
默认 chunk size 为 1024，cache reservation 由本次请求推导。

## DSW 运行

```bash
cd /jfs/application/user/bgl/LifeAI.jl
export JULIA_DEPOT_PATH=/jfs/application/user/bgl/envs/lifeai/depot
LIFEAI_JULIA=/jfs/application/user/bgl/envs/lifeai/julia-1.12.6/bin/julia

# 原始 BF16 checkpoint 已下载并核验。以下命令可在新环境重建同一版本。
/jfs/application/user/bgl/envs/flint-download/bin/hf download \
  ricursiveAI/Ripple-65B-Preview \
  --revision e27e171792f303d74c048184e8259ba663cfa0d5 \
  --local-dir /jfs/application/user/bgl/models/ricursiveAI/Ripple-65B-Preview

# 若需重建量化文件，执行一次；已有运行包时直接运行下一条命令。
"$LIFEAI_JULIA" -O1 --project=. scripts/quantize_ripple.jl \
  /jfs/application/user/bgl/models/ricursiveAI/Ripple-65B-Preview \
  /jfs/application/user/bgl/models/ricursiveAI/Ripple-65B-Preview-lifeai-int4-g128 \
  int4 128

# 共享 GPU 模式：12GiB 权重常驻，剩余 packed 权重在 CPU，逐层上传。
JULIA_CUDA_SOFT_MEMORY_LIMIT=20GiB "$LIFEAI_JULIA" -O1 --project=. scripts/run_ripple.jl \
  /jfs/application/user/bgl/models/ricursiveAI/Ripple-65B-Preview-lifeai-int4-g128 \
  --device cuda --resident-gib 12 \
  --prompt 'Reply with exactly OK.' --max-new-tokens 128 --temperature 0
```

`--scheme float|int8|int4|nvfp4` 是 LifeAI 从 BF16/F32 checkpoint 进行的可选量化。
embeddings、norms 和 value-residual scalars 保留浮点。
单 token 的 CUDA INT4 解码直接读取 packed 权重与 scales，以 Float32 activations／accumulation 执行 GEMV，避免大矩阵临时反量化。多 token prefill、INT8 和 NVFP4 reference 仍按 projection 分块重建浮点权重。
**不支持直接导入官方 ModelOpt NVFP4 checkpoint**；reader 对该配置明确报错。
运行时 activations／logits 为 Float32；`--dtype bf16` 只指定浮点权重存储。
没有声明与官方 BF16 activation 路径逐位一致。

官方发布说明声称最多 1M context，checkpoint config 的位置上限是 2,097,152；
实现读取 config 上限，不代表已验证或有显存运行该长度。
65B 完整 BF16 无法驻留此 DSW 的单张 72GB GPU。
INT4 也需要为 embeddings、scales、临时 projection、KV 和其他进程预留空间。
完整 65B 的原始权重校验、官方参考、Julia 数值对齐与生成实测见后续章节。

## Julia API

```julia
using LifeAI
using CUDA
bundle = load_ripple_quantized("/path/to/ripple-int4"; device=CUDA.cu, resident_bytes=12*2^30)
tokenizer = load_hf_ripple_tokenizer("/path/to/ripple-int4")
result = generate_ripple(bundle, tokenizer, [(role="user", content="Hello")];
    temperature=0, max_new_tokens=32, prefill_chunk_size=1024)
```

Config 中的 special-token IDs 沿用 HF 的 **0-based**；`encode`、forward、
generation 和 callbacks 使用 LifeAI 的 **1-based** IDs。
`hf_ripple_forward` 返回 `(vocab, tokens)` logits；`last_only=true` 只返回最后一列。
`RippleKVCache` 支持复用和 `reset_ripple_cache!`；`output_logits=false` 只处理缓存。
`generate_ripple_tokens` 的 `on_token(id)` / `on_prefill(done,total)` 返回 false 可取消。

工具 schemas／arguments 使用保留顺序的 `NamedTuple`、`OrderedJSONObject` 或
`parse_qwen3_json` 的结果，以复用项目已有的 CPython JSON serializer。
`apply_ripple_chat_template(...; tools, clear_thinking=false)` 与官方 Jinja 模板一致；
未知 template hash 会拒绝渲染。`chat_template.jinja` 优先于 tokenizer config 内嵌模板。

## 验证与参考生成

```bash
"$LIFEAI_JULIA" -O1 --project=. test/ripple/test_ripple.jl
"$LIFEAI_JULIA" -O1 --project=. scripts/verify_ripple_cuda.jl

# 如有完整官方 tokenizer 资产，可附加官方词表的精确 parity：
LIFEAI_RIPPLE_TOKENIZER_DIR=/path/to/pinned/metadata \
  "$LIFEAI_JULIA" -O1 --project=. test/ripple/test_ripple.jl
```

`test/ripple/fixtures/` 包含合成五层小模型，heads × head_dim 大于 hidden size，
同时覆盖 local/global attention、3-token window、非零 value-residual scalars、
非单位 norms、启用 clipping 的投影和 23-token prompt。
`reference.json` 来自固定版本**官方 PyTorch modeling 文件**，不是 Julia 的重新实现。
还包含逐项关闭 valres、gating、clipping、embedding scale，以及 window=1 的参考输出。
生成 oracle 时另行确认官方 token-by-token cache 与 full forward 一致；greedy oracle
用每步完整前缀重算，避免依赖官方 global attention 的 multi-token cached SDPA mask 行为。
Julia 的分块 prefill 按绝对位置做 causal mask，并与 full forward 对齐。

重新生成 fixture：在隔离 Python 环境安装 CPU PyTorch、`transformers==4.57.6`、
`safetensors`、`tokenizers` 和 `jinja2`，下载固定版本的 config／tokenizer／两个
modeling Python 文件（不需要 65B 权重），检查代码后执行：

```bash
python scripts/export_ripple_reference.py --upstream /path/to/pinned/metadata
```

参考模型与模板遵循上游 Apache-2.0 许可。fixture 权重为随机合成数据。
运行日志保存在 `benchmark_results/ripple/`。

## 第一阶段：合成小模型验证（2026-11-05）

Julia 1.12.6，RTX PRO 5000 72GB Blackwell，`-O1`：

- Ripple CPU **193** 项、CUDA **58** 项断言通过；包含完整官方 tokenizer 资产的精确 parity。
- 23-token、五层 Float32 合成模型，相对官方 PyTorch 的最大 logits 绝对误差：CPU `1.9073486e-5`，CUDA `1.04904175e-5`。缓存／分块／greedy token 验证通过。
- Flint checkpoint／tokenizer／NumPy reference／INT8／INT4 和采样模块的 **294** 项针对性回归通过；未全量执行所有 Chapter 测试。
- 官方 65B config 与 safetensors index 的 **1,203** 个张量名称完全一致。
- CPU Float32、CUDA 加载时 INT4 的 CLI 各生成 3 个 token，通过端到端 smoke；随机小模型输出不代表语言质量。

最终 Ripple 日志：`benchmark_results/ripple/validation.log`。
回归日志：`benchmark_results/ripple/regression.log`。
CLI 结果：`benchmark_results/ripple/cli-cpu.json`、`cli-cuda-int4.json`。
本节仅记录第一阶段的合成数据验证；完整模型结果单独记录如下。

## 完整 65B 的复现入口

完整原始模型位于 `/jfs/application/user/bgl/models/ricursiveAI/Ripple-65B-Preview`。
`verification.json` 记录 31 个文件共 129,572,862,161 bytes 的校验：
16 个分片匹配官方 LFS SHA-256，其余文件匹配官方 Git blob SHA-1。
实际参数总数为 64,782,689,360，包含全部 80 层、1,203 个参数张量。

- `scripts/export_ripple_full_reference.py`：直接运行固定版本官方 `RicursiveForCausalLM.forward`。只用 forward hooks 按模块加载／释放参数；不重写模型方程。保留原始 BF16 权重值，使用 Float32 activations／GEMM，并固定 `--attention-backend math`。DSW 通过 `/isaac-sim/python.sh` 提供 PyTorch 2.7.0+cu128 与 Transformers 4.57.6。
- `scripts/verify_ripple_full.jl`：完整原始权重按层流式运行，与上述官方参考比较全部词表的最后一列 logits、argmax、缓存解码 token，以及真实 prompt 的分词 IDs。
- `scripts/benchmark_ripple_full.jl`：复用完整 INT4 文件，warmup 后记录各请求 prefill、首 token、decode 速度、输出、EOS 与权重上传统计；同时验证 1,057-token 的整段／256-token 分块一致性。
- `scripts/check_ripple_full_quality.py`：检查精确 OK、17×23、生成的 Python `add` 函数、中文输出与连续解码长度。代码只在 AST 白名单通过后执行四组输入，属于小规模 smoke，而非综合质量 benchmark。

生成实测采用显式闭合的 `<think>` 前缀，便于测量直接回答；默认聊天入口单独验收。
显存与 RSS 由外部进程每 2 秒采样，报告采样峰值；CUDA allocator 的常驻缓存也计入进程显存。
性能来自共享 GPU、12GiB 权重常驻的具体设置，不代表独占 GPU 或官方服务的性能。
原始权重流式验证受磁盘读取和逐层传输影响，其耗时不用于估计正常驻留推理速度。

完整参考固定 PyTorch 的 `MATH` SDPA 后端。此 DSW 的 PyTorch 2.7.0+cu128 在
Blackwell 上自动选择 `EFFICIENT_ATTENTION` 时，固定随机种子的 28/64-token
Float32 注意力相对数学后端分别出现约 0.392／0.373 的最大绝对差异；
Julia CUDA 注意力与独立 CPU 数学参考的误差低于 8.4e-7。
可用 `scripts/diagnose_ripple_torch_sdpa.py` 复现诊断；实际记录位于
`benchmark_results/ripple/full/torch-sdpa-diagnostic.json`。该设置仅作用于参考导出进程。

## 完整模型实测结果（2026-11-05）

原始 65B、持久化 INT4 包、完整数学参考对齐、真实生成和普通 CLI 入口均已通过。
运行包的权重 payload 为 **35,176,223,040 bytes（32.76GiB）**；其中 641 个矩阵
使用 INT4/group128，embedding、norms 和 value-residual scalars 保留浮点。
来源校验记录也保存在运行包的 `source-verification.json`。

原始 BF16 权重、Float32 数学计算对官方 MATH 参考：

| 输入 | prompt tokens | 最大 logits 绝对误差 | relative L2 | argmax |
| --- | ---: | ---: | ---: | --- |
| 英文 OK 指令 | 19 | 9.536743e-05 | 4.707931e-06 | 一致 |
| 中文乘法指令 | 28 | 3.1232834e-05 | 1.5855205e-06 | 一致 |
| 跨滑动窗口输入 | 1057 | 0.00035095215 | 1.9921079e-06 | 一致 |

完整词表的最后位置 logits 均满足 `max_abs < 1e-3`、`relative_l2 < 1e-4`；
缓存贪心解码的 `OK` 与 EOS token 也与官方参考一致。

INT4 生成采用 **12GiB GPU 权重常驻 + 20.76GiB CPU 权重卸载**，单 token 直接
消费 packed INT4，Float32 accumulation，不进行激活量化。
完整 SHA-256 校验并加载权重耗时 **55.94s**，普通 CLI 另一次运行是 **64.86s**。
`checksum_backend=:auto` 优先使用系统 `sha256sum` 一次性校验所有 payload，
缺少该工具时使用 Julia SHA；`checksum_backend=:julia` 可显式选择可移植路径。
形状、dtype、文件长度、路径和有限值校验始终执行。
上述加载时间不包含 Julia 导入／预编译、tokenizer 初始化和生成。

| 实测任务 | 输出 tokens（含 EOS） | prefill 秒 | decode token/s | 结果 |
| --- | ---: | ---: | ---: | --- |
| 精确 OK | 2 | 15.85 | 0.849 | OK，EOS |
| 17 × 23 | 2 | 16.96 | 0.604 | 391，EOS |
| Python add | 12 | 19.75 | 0.657 | 函数执行验证通过，EOS |
| 中文二分查找 | 55 | 22.71 | 0.723 | 完整中文解释，EOS |
| 连续解码性能 | 96 | 20.93 | 0.727 | 达到 96-token 预算 |

96-token 用例的稳态 decode 耗时 **130.76s**，速度
**0.727 token/s**。它用于性能测量，达到 token
预算后停止；完整代码的功能验收使用单独的 `add` 用例，四组整数／浮点输入均通过。
默认聊天模板的独立 CLI 调用也返回 `OK` 与 EOS，不需要显式闭合思考前缀。

1,057-token 的 INT4 整段／256-token 分块计算最大 logits 差异为
**0.0002746582**，argmax 一致；对原始权重 Float32 logits 的
cosine 为 **0.9997452**，relative L2 为
**0.050933**。该差异属于权重量化影响。

整个 INT4 生成与长输入验收进程的显存采样峰值为 **37,024MiB
（36.16GiB）**，RSS 采样峰值为
**35.90GiB**。12GiB 是常驻权重预算；临时计算、KV 与
CUDA allocator 缓存另计，不能把它作为总显存上限。

最终回归共 **669** 项断言通过：Ripple CPU 251、CUDA 124，Flint／采样 294。
包含 INT4 GEMV 的 8／128／5,120／40,960 输入宽度、非整齐尾行、缓存解码、
BF16/F32 设备读取、两种 SHA 校验路径以及损坏 payload 拒绝测试。
这些单卡/INT4 回归未运行全部 Chapter 测试、行业综合 benchmark 或 1M-token 上下文验收；TP8 阶段性长上下文验收另见本文末节。

最终证据位于 `benchmark_results/ripple/full/`：

- `summary.json`：模型来源、数值验收、资源与性能汇总。
- `parity.json`、`official/reference.json`、`official/reference.safetensors`：完整原始权重的数学对齐。
- `generation.json`、`quality.json`、`generation-resources.json`：真实输出、功能检查和资源采样。
- `default-cli.json`：普通聊天 CLI 入口结果。
- `native-int4-tests.log`：最终 669 项回归结果。

模型数学参考与 INT4 生成使用不同权重精度；这些 smoke 不能替代综合质量评测。

## DLC 单节点 8 卡原始权重服务

通过 `scripts/dlc_ripple_tp8.py` 创建单 Worker、8 GPU 的推理任务，调度策略固定
`Binpack`，同时挂载 `/jfs` 与 `/mnt/public_data`。任务从已提交版本导出代码快照，
启动前核对 `CODE_REVISION` 与所有源码 SHA-256；模型仍读取共享 JFS 上已校验的原始分片。
持久化环境为 `configs/deployment/ripple_tp8`，固定 CUDA.jl 6.2.1、NCCL.jl 0.1.3、
NCCL 2.28.3 与 CUDA 12.9；不在容器启动时重新解析整个应用环境。

采用原始 BF16 矩阵驻留、Float32 norms/activations/GEMM；每卡保留 10 个 Q heads、
1 个 KV head、5,120 维 FFN 和 12,544 行输出词表。Embedding 与 norms 复制，
Q/K/V/g 与 gate/up 按输出行分片，O/down 按输入列分片；每层的 attention 和 FFN
输出均在 post norm/residual 前做求和，最终 logits 按词表 allgather。
八卡总驻留权重包含复制的 embedding；权重字节数不等于 GPU 总显存。

一个 Julia 进程用每卡常驻 worker 计算本地结果，复用任务内的 CUDA/cuBLAS 状态，
然后由单个主机线程用 NCCL group 提交全部 rank 的通信。初始版本每次 forward 前在主机侧
回收临时对象，后续优化改为预填充边界与有界解码间隔回收。每个计算/通信阶段完成后同步；计算中的 CUDA 分配与回收不与未
提交完整的 collective 交错。最初整段异步方案在 1,057-token 输入上停顿，SIGUSR1
栈确认某 rank 卡在 finalizer 的 `cuMemFreeAsync`，其他 GPU 等待通信；分阶段提交修复
保留相同数学方程。`test_ripple_nccl_transport.jl` 覆盖匹配该长输入的 20.6MiB
allreduce 与阶段间 GC，`test_ripple_tensor_parallel.jl` 对照固定版本官方 PyTorch
fixture 验证 TP1/2/4/8、完整 logits、缓存、分块和 greedy。

提交和查询（在 DSW 项目根目录执行）：

```bash
/isaac-sim/python.sh scripts/dlc_ripple_tp8.py create --name ripple65b-tp8
/isaac-sim/python.sh scripts/dlc_ripple_tp8.py status JOB_ID
python3 scripts/query_ripple_tp8.py RUN_DIR --prompt 'Reply with exactly OK.'
python3 scripts/verify_ripple_tp8_http.py RUN_DIR
/isaac-sim/python.sh scripts/verify_ripple_tp8_context.py RUN_DIR
/isaac-sim/python.sh scripts/dlc_ripple_tp8.py stop JOB_ID
```

`create --dry-run` 生成可检查的代码快照和 `request.json`，不创建任务。
每次实际提交都会创建独立 run directory；`job.json` 保存任务编号和控制台链接。
服务在数值与生成验收通过后写入 `results/ready.json`。DSW 查询脚本读取该文件、探测
私网地址并从 `credentials/api_key` 读取 Bearer key，不打印 key。

服务提供 `/healthz`、`/v1/models` 和基本 OpenAI 兼容的 `/v1/chat/completions`。
初始基线为单请求串行、非逐 token 文本流式聊天，预算上限 8,192 tokens、输出上限 512 tokens、
prefill chunk 256 tokens。后续阶段性长上下文与异步进度支持见本文末节；当前公开上限以部署收据及 `/healthz` 为准。
并发 batch 与结构化 tool calls 未实现。
`results/service.log`、`resources.csv`、`parity.json`、`generation.json`、`validation.json`
保存启动、数值、输出和资源记录。采样显存包括 CUDA allocator 缓存；加载时间不包含
Julia 导入/预编译。此节点为八张 RTX PRO 5000 72GB、PCIe PHB 互连。

### 历史：初始 8K 八卡实测（2026-11-05）

历史任务 [dlc1x35onfyb2m10](https://pai.console.aliyun.com/?regionId=cn-wulanchabu&workspaceId=253586#/dlc2/jobs/dlc1x35onfyb2m10) 记录了初始 8K 验收，现已退役。该次服务模型名为 `lifeai-ripple65b-tp8`，
当时 DSW 实测的 API base 为 `http://10.68.4.21:18085/v1`；它不是当前连接地址。
已加载的服务代码固定为 `d7340cb1fb57a379f5d18de891e60f458ca0ea2f`；后续仓库更新包含调用工具、地址元数据修正和本记录。
本节为历史基线；当前发布任务及公开预算以 [`deployment.json`](../configs/deployment/ripple_tp8/deployment.json) 为准，历史原始证据见本节运行目录。

原始 BF16 矩阵全部驻留八卡，norms 与计算使用 Float32，每卡权重为
**17,101,066,560 bytes（15.93GiB）**，无 CPU 权重卸载。
完整加载耗时 **91.64s**，不含 Julia 导入/预编译。
CPU TP1/2/4/8 共 **64** 项断言、真实 NCCL 八卡 **19** 项断言全部通过。
后者包括三轮 20.6MiB allreduce、阶段间 GC、常驻 worker 身份，以及官方小模型的
完整/缓存/分块 logits 与 greedy token 对齐。

| 官方完整权重参考 | 输入 tokens | 最大 logits 绝对误差 | relative L2 | argmax |
| --- | ---: | ---: | ---: | --- |
| exact_ok | 19 | 4.7683716e-05 | 7.7708626e-07 | 一致 |
| math | 28 | 2.861023e-05 | 1.3316906e-06 | 一致 |
| long_window | 1057 | 0.00037574768 | 3.386717e-06 | 一致 |

1,057-token 整段输入耗时 **2.44s**；全部 logits 误差均满足
`max_abs < 1e-3`、`relative_l2 < 1e-4`，缓存 OK/EOS token 与官方数学参考一致。
启动验收实际执行了 **27,040** 次逻辑 allreduce 与 **169** 次词表 allgather，
八卡 rank 调用计数分别为 **216,320** 与 **1,352**。

| 真实生成任务 | 输出 tokens（含 EOS） | prefill 秒 | decode token/s | 结果 |
| --- | ---: | ---: | ---: | --- |
| 精确 OK | 2 | 0.55 | 2.197 | OK，EOS |
| 17 × 23 | 9 | 0.51 | 2.294 | 17 × 23 = 391，EOS |
| Python add | 17 | 0.43 | 2.695 | 四组执行结果通过，EOS |
| 中文二分查找 | 41 | 0.44 | 2.627 | 完整中文解释，EOS |
| 连续解码 | 96 | 0.38 | 2.581 | 达到 96-token 预算 |

96-token 用例的稳态 decode 耗时 **36.81s**、速度 **2.581 token/s**；
到达预算后停止。完整 Python 函数使用独立的 `add` 用例验收。
此处为该硬件上原生 Float32 计算路径的单请求测量，五项生成检查不代表综合质量 benchmark。

DSW 的 **12 项 HTTP 检查**通过：未鉴权返回 401，非法请求/超限/不支持的流式请求返回 400，
普通聊天返回 `OK` 与 stop。另一个 **7,996-token 输入**经 256-token 分块，
生成 **21 tokens** 后以 EOS 结束，最终回答为 `OK`，端到端 **26.80s**。
长输入检查覆盖容量和最终回答；接口保留模型原始文本，包括可能出现的 `<think>` 标签。
`verify_ripple_tp8_http.py` 在完成真实鉴权调用后，可将已验证的私网地址写回 `ready.json`；
客户端跳过 link-local 地址并直接访问私网，后续服务启动也会过滤 link-local 地址。

从启动到上述 HTTP 验收，每 2 秒采样一次八卡设备显存。单卡采样峰值为
**59,057MiB（57.67GiB）**，八卡同时采样总和峰值
**431,434MiB**；包括 CUDA allocator 缓存、临时计算和 KV。
主 Julia 进程的 RSS 采样峰值为 **8.13GiB**，不包括预编译子进程。

历史基线运行目录：

```text
/jfs/application/user/bgl/deployments/ripple-dlc-tp8/ripple65b-tp8-service-20261105T070401Z
```

历史任务的复现入口如下；任务已退役，实际使用请读取当前部署收据：

```bash
python3 scripts/query_ripple_tp8.py /jfs/application/user/bgl/deployments/ripple-dlc-tp8/ripple65b-tp8-service-20261105T070401Z --prompt 'Reply with exactly OK.'
```

`results/http-validation.json`、`http-long-context.json`、`query-client.json` 和
`resources-summary.json` 分别保存接口、长输入、实际客户端与资源验收结果；
其余完整数值/生成证据见同目录。API key 保存在该运行目录的 `credentials/api_key`，
调用脚本自动读取，仓库收据只记录文件路径。

## Lumen CLI 接入（2026-11-05）

Lumen 已增加 `ripple` 模型、第三个 TUI 标签与独立会话。DSW 的已安装入口
`/usr/local/bin/lumen`、`lifeai` 和仓库 `scripts/lifeai` 均已更新。

```bash
lumen --model ripple
lumen --model ripple -n 8 'Reply with exactly OK.'
lumen --model ripple --no-stream '请用一句话解释二分查找。'
```

在 TUI 使用 `/model ripple` 或 Ctrl+T/F2 切换。默认模型仍为 Qwen，
可用 `LIFEAI_MODEL=ripple` 选择 Ripple。其他机器先按 [Lumen 说明](../tools/tui/README.md)
编译/安装更新后的客户端；自动 SSH 转发仍连接 DSW 的 11435 路由端口。

DSW 路由通过 11438 的 `ripple_worker.py` 适配器使用当前发布收据选定的 DLC 八卡服务，
完整保留 system/user/assistant 消息，密钥只在 DSW 读取。
切换或退出只释放本地适配器或客户端 SSH 连接，DLC 服务继续驻留。
模型文本完整生成后显示；异步任务可发布真实预填充进度和心跳。
上下文上限由当前已验证档位发布，输出上限仍为 512 tokens。
文件/标准输入与草稿上限 64MiB UTF-8、对话合计 64Mi 字符、JSON 请求 128MiB，响应上限仍为 16MiB。
生成超时默认 21600 秒（6 小时），加载超时独立保持 1800 秒。
完成的 `<think>` 内容保留显示，下一轮历史只发送最终答案。

本次传输与长请求更新：Rust **36** 项单元测试、**15** 项 CLI HTTP 测试、Python **61** 项协议/生命周期测试通过；
`cargo fmt --check` 与 Clippy（警告视为错误）通过，Python 61 项也在 DSW 仓库通过。
另有 Julia HTTP/队列/真实 TCP 边界 **78** 项检查和分阶段语料 harness **3** 项测试通过。
原始接入时，真实安装命令、仓库启动脚本和自动 SSH 单次调用均返回 `OK`。
真实交互终端两轮测试中，首轮记住暗号，次轮正确回答 `SQUIRREL`；
Ctrl+C 退出码为 0，SSH 转发关闭，路由及 DLC 仍 ready。输出预算 513 被正确拒绝。
上述初始接入证据作为历史保留。v14 公开 1Mi 后，真实安装的 `lumen` 单次调用 3.666 秒返回 `OK`，
交互终端两轮正确返回 `OK` / `SQUIRREL`，退出码 0 且 SSH tunnel 已关闭。
另以 `--system` / `--file` 读取 6,462,244-byte 文件，实际提示词 1,048,336 tokens，
20.750 秒返回 `OK`；stderr 显示真实已复用前缀 `1048320/1048336`。
该文件用嵌入角色标记的 synthetic transcript flatten 重现已验证多轮序列，验证大文件传输及缓存续问，
不作为任意文档的冷启动 1Mi prefill 性能证据。小证据位于 `benchmark_results/ripple/long-context/lumen-v14/{file,tui}/`。

部署验证收据见 [`lumen.json`](../configs/deployment/ripple_tp8/lumen.json)，包含二进制与源码
SHA-256；详细日志、终端截图文本和验证 JSON 保存在 `benchmark_results/ripple/lumen/`。

## 历史：TP8 短上下文解码性能优化（2026-11-05）

本节短上下文基准使用历史任务 [dlc14tk1zf50nw9d](https://pai.console.aliyun.com/?regionId=cn-wulanchabu&workspaceId=253586#/dlc2/jobs/dlc14tk1zf50nw9d)，服务代码固定为 `0921a3316dc3ee25b42793ab698f59c3f2ba10ed`。
该历史任务现已退役；Lumen 的 `lumen --model ripple` 使用当前部署收据选定的服务。
客户端自动读取服务发布的有效解码速率元数据。

保持原始 BF16 权重值与 Float32 激活/累加，没有进行权重或激活量化。
新增 direct BF16 split-K GEMV，直接读取 BF16 矩阵并用 Float32 合并局部和；
多 token prefill 继续使用 FP32 GEMM，临时转换在所在 CUDA stream 上显式释放。
解码将 FFN 后残差处理并入下一层 attention 阶段，缓存每卡的 -Inf sink 常量，
将 gate/up clamp 与 SiLU 乘积融合。预填充仍每批回收，解码可配置有界 GC 间隔。
所有 collective 仍由单个主机线程按 group 提交，在分配/回收前完成同步。

比较在同一个节点、同一组 8×RTX PRO 5000 72GB、同一 29-token 提示词上运行，
温度 0，固定 96 个输出 token；速率为首 token 之后的 95 次间隔。
**这是 29-token 输入的短请求稳态基准，6.155 token/s 不代表 128K—1Mi 的预填充速度，也不是长历史追问的端到端吞吐。**
原始 `reference` 与最终候选各做三次，其他方案只有一次 pilot：

| 方案 | 样本数 | median decode token/s |
| --- | ---: | ---: |
| bf16_gemv_gc16 | 3 | 6.155 |
| reference | 3 | 2.478 |
| bf16_gemv | 1 | 5.147 |
| fp32 | 1 | 4.113 |
| merged_reference | 1 | 2.609 |

最终选择 `bf16_gemv_gc16`（投影 `bf16_gemv`、调度 `merged`、解码 GC 间隔 `16`）。
三次原始结果为 `2.504, 2.478, 2.452`，优化后为 `6.155, 6.179, 6.088` token/s，
中位数 **2.478 → 6.155 token/s，2.48×**。
96-token 输出与基线逐 token 一致：**true**。
在 DSW 通过真实 HTTP（包含 prefill）重复三次，整体速率中位数为 **5.833 token/s**。

三组官方 PyTorch MATH 完整词表对齐继续满足 `max_abs < 1e-3`、`relative_l2 < 1e-4`；
另以原始实现保存的 16 个预测位置（prefill 末位及 15 个缓存解码位置）逐步对比全部词表，最大绝对误差 **8.392334e-05**，
argmax 全部一致。OK、乘法、Python 函数执行、中文与 96-token 连续生成检查通过。
两次 7,996-token HTTP 输入完整生成最终 `OK` 并以 EOS 结束，鉴权与请求边界检查通过。
每卡常驻权重仍为 **15.93GiB**；资源采样覆盖全部 benchmark 方案及长输入验收，
其中包括仅用于对照的 FP32 常驻方案，因此整体采样峰值不等于最终 BF16 服务的权重占用。

`prepare_ripple_tp_projections!(bundle; mode=:reference|:bf16_gemv|:fp32)` 仅在 idle 时调用；
从 Float32 转回 BF16 会检查是否无损。`hf_ripple_tp_forward` / `generate_ripple_tp_tokens`
可选择 `schedule=:staged|:merged`，`ripple_nccl_backend(...; gc_interval=N)` 配置回收间隔。
CLI 服务的对应参数为 `--projection-mode`、`--schedule`、`--gc-interval`；
DLC 配置可用 `benchmark_variants=true` 运行同机比较并选择已通过数值验证的最快方案。

部署和汇总数据见 [`deployment.json`](../configs/deployment/ripple_tp8/deployment.json)、
[`performance.json`](../configs/deployment/ripple_tp8/performance.json)。
本节历史短上下文的逐步 logits 参考、各方案 timings、阶段剖析、生成和接口证据保存在该历史任务运行目录的
`results/variants.json`、`decode-reference.jls`、`generation.json`、`http-throughput.json`、
`http-long-context.json`、`final-verification.log` 与 `resources.csv`。


## 阶段性长上下文支持（v14 已验证 1Mi）

> 128Ki pilot 与 native v14 的 256Ki、512Ki、1Mi 长测共同完成四档功能验收，公开验证上限为 **1,048,576 tokens**。
> 当前 HTTP v15 已通过独立接口及 installed Lumen 客户端验收，详见 [上下文收据](../configs/deployment/ripple_tp8/context.json)。

预算按**模板、全部消息历史与预留输出合计**，四档分别为 131,072、262,144、524,288、1,048,576 tokens。
模型仍为原始 BF16 权重、Float32 激活/累加与 FP32 KV，单节点 TP8。
16 层全局注意力的 1Mi KV 估算每卡 16GiB，64 层 1024-token 滑窗缓存保持有界，
加上权重每卡约 32GiB 常驻；该估算不包含临时激活、scratch 和 allocator 缓存。

### 方法与已知边界

- 长输入按 2048-token 分块，attention query microbatch 最大 2048，并受 1GiB scratch guard 限制。
  对 owning CuArray 的 view/reshape 别名及 tiled-attention 临时区显式释放，prefill 同步边界清除持久 worker 引用后 full GC，
  解决先前约 188K 位置的显存累积问题。
- Ripple 专用 RoPE 采用与官方 CUDA forward 一致的逆频率/三角函数计算。独立位置 oracle 覆盖
  131072、262144、524288、1048576 附近以及 2097151；实际 CUDA 位置检查最大绝对误差为 0。
- 一个有界、串行的 exact-token 前缀检查点只复用已预填充提示词。滑窗在提示词边界复制快照；
  全局 KV 共享不再改写的前缀区，生成后缀不加入可复用检查点。
  失配、运行策略变化、异常或取消会使检查点失效；长缓存记住 full-GC 需求。
  容量增长在重新 checkpoint 前完成，清除旧 snapshot/worker 引用后回收，避免持久双份完整 KV。
- `/v1/chat/completions/jobs` 使用共享 Bearer 鉴权，GET 轮询进度/结果；适配器按 ID 取消自身提交的 job。
  Lumen 转发真实 prefill 计数与心跳，最终文本仍完整返回；断开或超时不停止 DLC 服务。
  请求链路输入 64MiB、JSON 128MiB、响应 16MiB，生成超时默认 21600 秒。
- 后台配置的 `max_context=1048576` 与公开已验证上限分开。部署收据的 `validated_context_tokens`
  控制 Lumen/路由公布的有效上限；适配器转发 `context_limit`，模型在实际分词后校验输入+输出预算。
  提高配置上限本身不会把未验收档位标记为已通过。

每档在**新增 archive 文档段**约 5%/50%/95% 位置放置随机验证值，精确取回答案并要求 EOS；
后续请求保留历史并验证 exact-token 前缀复用计数。最终 1Mi 也已查询最早文档中此前未查询的隐藏锚点并正确返回。
这些是容量、检索与复用 smoke，不代表综合长上下文 benchmark。

### 已完成阶段的实际结果

以下 `seconds` 为测试客户端端到端耗时，包含 HTTP 轮询；不应与上文短请求 decode token/s 混用。
128Ki 来自 `e067d018a455f82afb985563d7cbebe6ea1f91be` 的 pilot；
256Ki/512Ki/1Mi 来自 native v14 `aea3e4f284f1c0e63a540ae31c59c70ae4cd874b`。

| 阶段 | 状态 | 首问输入 tokens | 首问复用 tokens | 首问端到端秒 | 中/尾短追问秒 | 精确预算输入 + 预留输出 |
| --- | --- | ---: | ---: | ---: | --- | --- |
| 128Ki pilot | passed | 126,976 | 0 | 136.61 | 6.05 / 6.05 | 130,816 + 256 = 131,072 |
| 256Ki v14 | passed | 258,048 | 0 | 657.09 | 8.07 / 8.05 | 261,888 + 256 = 262,144 |
| 512Ki v14 | passed | 520,192 | 261,888 | 820.83 | 12.08 / 12.07 | 524,032 + 256 = 524,288 |
| 1Mi v14 | passed | 1,044,480 | 524,032 | 2,593.71 | 22.16 / 24.15 | 1,048,320 + 256 = 1,048,576 |

512Ki 首问已复用前一阶段的 261,888-token 检查点，只新增预填充 258,304 tokens；
**820.83 秒不是冷启动处理整段 512Ki 的耗时**。中/尾短追问也复用了绝大多数历史，
其约 6/8/12 秒为历史复用后的端到端续问时间。
128Ki pilot 与 1Mi 均完成全局最早锚点查询；1Mi 该查询端到端 25.090 秒。
1Mi 首问新增预填充 520,448 tokens，复用 524,032 tokens，native prefill 为 2,589.504 秒；
2,593.71 秒为这次增量续算的客户端端到端时间。精确预算请求 43.123 秒返回 `OK`。
这次 1Mi 首问生成 10 个 output tokens，样本 decode 为 **3.889 token/s**；这是单次长历史样本，
与上文 29-token 输入、96-token 输出、三次重复的历史 6.155 token/s 基准分别记录。
冻结的资源采样记录单卡峰值 51,099MiB（49.90GiB）；这是设备采样峰值，非 KV 单项大小。

### 原生长测与最终 HTTP 版本分离

native v14 的长测运行目录：

```text
/jfs/application/user/bgl/deployments/ripple-dlc-tp8/ripple65b-tp8-context-bounded-20261105T114011Z
```

该 native 版本与当前 HTTP v15 `4c324b4968da2a775b8fc48b86fcb013b80c9c41` 的源码树完全一致：

```text
src tree: 78d19728505b8bb2ecc8b25247c810b7a82df9a4
ext tree: 5ffacf294e90015a32f231fff97f7baec77becd4
```

native v14 的完整阶段报告 `results/context-stages.json` 在 2026-11-05T13:12:10.743432+00:00 完成，
`all_passed=true`；SHA-256 为 `c7451237587248dd8ed0efcae8e0008675820e47605e4bc9f9736c7feff884c3`。
其实际长测任务为 `dlc1e0h0crzzqys0`，源码为 `aea3e4f284f1c0e63a540ae31c59c70ae4cd874b`。

当前 HTTP v15 任务为 `dlcdyvi8bn3txeib`，源码为
`4c324b4968da2a775b8fc48b86fcb013b80c9c41`。该版本的标准 HTTP、headers-only 401/413、
queued cancel 检查通过；DSW 安装命令及仓库启动脚本均返回 `OK`。
installed Lumen 单次调用 2.171 秒返回 `OK`，交互终端两轮记忆正确返回 `OK` / `SQUIRREL`，退出码 0，SSH tunnel 关闭。
6.46MB、1,048,848-token 输入加 8-token 输出预算为 1,048,856，实网在 19.795 秒返回 HTTP 400，
验证超预算拒绝。v14 的完整原生阶段报告与 v15 独立 HTTP/客户端证据分别记录，
共同覆盖当前发布的模型路径与接口边界。
证据见 `benchmark_results/ripple/long-context/lumen-final-dsw.json` 及
`benchmark_results/ripple/long-context/lumen-final/`；发布预算见
[context.json](../configs/deployment/ripple_tp8/context.json)。

```bash
lumen --status
lumen --model ripple --file /path/to/long-document.txt --timeout 21600
```

较长文件仍须满足当前已公开的 token 预算；完整消息由服务端分词，超限返回可识别的 400 类错误。
源证据：128Ki pilot 的 `results/context-128k-pilot.json`；v14 的 `results/context-stages.json`、
`context-stages.messages.json`、`context-stages.resume-proof.json`、`staged-run.resume-launch.json` 及 `resources.csv`。
原先 1Mi 语料拼接遇到 BPE 尾部合并的一 token 间隙，harness 已改为按真实编码长度 bracket/search；
resume 保留并核对已通过阶段的消息与 token 哈希。该 harness 构造失败不是模型 512Ki 失败，
最终 1Mi 模型验收以随后完成的真实阶段报告为准。
