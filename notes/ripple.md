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
未运行全部 Chapter 测试、行业综合 benchmark 或 1M-token 上下文验收。

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
然后由单个主机线程用 NCCL group 提交全部 rank 的通信。每次 forward 前在主机侧
回收上一轮临时对象。每个计算/通信阶段完成后同步；计算中的 CUDA 分配与回收不与未
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
当前为单请求串行、非流式文本聊天，输入与输出预算合计上限 8,192 tokens、输出上限 512 tokens、
prefill chunk 256 tokens；流式、并发 batch、结构化 tool calls 未实现。
`results/service.log`、`resources.csv`、`parity.json`、`generation.json`、`validation.json`
保存启动、数值、输出和资源记录。采样显存包括 CUDA allocator 缓存；加载时间不包含
Julia 导入/预编译。此节点为八张 RTX PRO 5000 72GB、PCIe PHB 互连。

### 八卡实测记录（2026-11-05）

任务 [dlc1x35onfyb2m10](https://pai.console.aliyun.com/?regionId=cn-wulanchabu&workspaceId=253586#/dlc2/jobs/dlc1x35onfyb2m10) 保持运行，服务模型名为 `lifeai-ripple65b-tp8`，
DSW 实测可达的 API base 为 `http://10.68.4.21:18085/v1`。
已加载的服务代码固定为 `d7340cb1fb57a379f5d18de891e60f458ca0ea2f`；后续仓库更新包含调用工具、地址元数据修正和本记录。
完整部署收据为 [`configs/deployment/ripple_tp8/deployment.json`](../configs/deployment/ripple_tp8/deployment.json)。

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

当前运行目录：

```text
/jfs/application/user/bgl/deployments/ripple-dlc-tp8/ripple65b-tp8-service-20261105T070401Z
```

在 DSW 的项目根目录直接调用：

```bash
python3 scripts/query_ripple_tp8.py /jfs/application/user/bgl/deployments/ripple-dlc-tp8/ripple65b-tp8-service-20261105T070401Z --prompt 'Reply with exactly OK.'
```

`results/http-validation.json`、`http-long-context.json`、`query-client.json` 和
`resources-summary.json` 分别保存接口、长输入、实际客户端与资源验收结果；
其余完整数值/生成证据见同目录。API key 保存在该运行目录的 `credentials/api_key`，
调用脚本自动读取，仓库收据只记录文件路径。
