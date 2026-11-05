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
