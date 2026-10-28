# Flint 256K 上下文

Flint 的原始配置已支持 `262144` tokens，本次沿用检查点的位置编码设置。
总预算包含模板、历史、输入与预留输出；DSW 服务输出上限为 `4096`，
可通过 `--max-new-tokens` 配置。
Qwen 的上下文配置保持 `40960`。

## 内存与计算

该模型有 14 个全注意力层和 35 个滑窗层，每层 4 个 KV heads、head dim 128。
FP32 的全注意力 KV 在最大长度下每层占 1 GiB，14 层共 14 GiB；滑窗只保留
后续计算需要的 512-token 历史。KV 精度保持 FP32。

原生 NVFP4 权重此前同时保留 GPU 上的 portable 数据和 cuBLASLt 布局。
`prepare_flint_nvfp4(bundle; portable_device=:cpu)` 将 portable 副本放入主机
内存，推理继续使用 GPU 原生 FP4 GEMM。服务释放加载时的原始 GPU bundle
并回收闲置池，给 KV 和预填充腾出空间；保存量化产物仍能还原原始字节。
加载器会清空其独占的原始参数字典，避免函数栈保留的旧引用拖延 GPU 副本回收。

`generate_flint_tokens` 默认按 4096 tokens 分块预填充。全注意力 KV 一次按
请求的输入加输出预算预留，后续追加复用同一存储；短请求不会分配 256K KV。
中间批次跳过最终 norm 和词表投影，最后一批只为末 token 计算 logits。

CUDA 注意力包含两条有界路径：预填充使用 FP32 分块 GEMM 和在线 softmax，
逐 token 解码使用 split-K 内核。两者都保留因果约束、GQA、滑窗和 learned
sink。预填充复用 K/V tile 与 score tile，避免创建完整的 N×N 分数矩阵或
转置整个 KV。解码的 scratch 上限为 256 MiB；预填充低于 1 GiB。

NVFP4 是 W4A4 路径。旧的 `activation_scaling=:global` 对整个激活矩阵求
一个缩放因子，批次边界和其他 token 的异常值都会改变当前 token 的量化。
服务改用 `:token`：在 GPU 上为每 token 求独立缩放因子，直接打包 FP4 与
E4M3 scales，原生 GEMM 后乘回各列的 Float32 因子。输入和权重仍执行
原生 FP4 GEMM；公开 API 默认保留 `:global` 以兼容已有结果。
不同 GEMM/注意力的浮点累加仍不要求逐位一致。FP32 fixture 的分块与完整 logits
以及独立的注意力 oracle 用于检查实现误差，不能代替模型任务质量评测。

## 使用

```sh
lumen --model flint -n 2048 --file long-document.txt
# 或在 TUI 中输入 /load /path/to/long-document.txt，再按 Enter 发送草稿
```

客户端文件和标准输入上限 4 MiB，支持 UTF-8。路由与 worker 的 JSON 上限
为 16 MiB、64 条消息、合计 4 Mi 字符；tokenizer 会再检查实际 token 预算。
预填充事件分别发布 `prefill_processed` 与 `prompt_eval_count`，Lumen 显示
百分比，不将进度混入回答。默认请求超时 7200 秒。

服务参数为 `--context-tokens 262144 --prefill-chunk-size 4096
--activation-scaling token --max-new-tokens 4096`。
Julia 接口可使用 `prefill_chunk_size`、`on_prefill(processed,total)`、
`cache_capacity` 和 `output_logits=false`。回调返回 `false` 或抛出异常时，
生成在下一批计算前退出；已提交的 KV 前缀在失败重试时保持有效。

## 验证记录

整模型 256K HTTP 验收正在进行，结果尚未写入本记录。

运行环境：NVIDIA RTX PRO 5000 72GB Blackwell、Julia 1.12.6 `-O1`、
原生 cuBLASLt NVFP4，模型 revision
`2a247be0c0aec895ccb0ac7ca92229ec6df37288`。

完整原始日志和验收输入位于 JuiceFS：
`/jfs/application/user/bgl/benchmarks/flint-context256k-20261028/`。
仓库记录见 [benchmark_results](../benchmark_results/flint_context256k/)。

注意力独立微基准在 256K keys、4096 queries 时从 split-K 的 `13.401 s/层`
降为 tiled GEMM 的 `0.926 s/层`，约 `14.47×`；实际 scratch 为 `560.2 MiB`。
这是单层测量，整模型吞吐还包含 MoE、投影、路由和分块调度。

回归覆盖独立 NumPy logits、CPU/GPU 分块缓存与失败重试、NVFP4 算术和
CPU portable 产物往返。长注意力专项 `108/108` 通过，包含实际 256K keys、
非零起始位置、全遮蔽查询、GQA、learned sinks、滑窗与跨 tile 边界；
单个真实尺寸 KV 层也完成了 1 GiB、262144-token 的分块写入。
路由/进程控制 `36/36`、Rust 客户端 `46/46` 和验收脚本 HTTP mock `3/3` 通过。
每 token 激活量化及既有 FP4 回归共 `2313/2313` 通过，包含独立 CPU 逐 token
量化器、原始打包字节、异常值隔离、分块与整批一致性、tiny Flint 的 logits。
额外的真实随机 256K KV 对照 `18/18` 通过，最大注意力误差 `4.65e-6`。
真实 HTTP 的 `262143+2` 越界请求在 worker/router 均返回 400。
实际 PTY 导入 `1,255,139` 字节草稿约 `0.10 s`，退出与终端恢复通过。

同一输入、温度0、单次请求的每 token 激活缩放对照如下。8K/32K 的总预算
包含512预留输出；完成回答时以整个最终JSON与三个标准值完全相等判定通过。

| 多样文本预算 | global 首字 | token 首字 | 比值 | 严格检索 |
| --- | ---: | ---: | ---: | --- |
| 8192 | 33.78 s | 12.68 s | 2.66× | 两者均3/3 |
| 32768 | 156.76 s | 57.18 s | 2.74× | 两者均3/3 |

每项只有一次测量，不能外推为所有输入的固定提速。短对话另有一次排除的
64-token预热与三次测量：新路径解码中位数 `34.07 tok/s`，首字中位数
`0.539 s`；同轮global基线为 `27.68 tok/s`、`0.991 s`。

重复8K文本在两条量化路径中都找到了三个标识，但耗尽512输出预算仍未完成
思考，严格最终答案失败。早期global模式的 `261632+512` 重复文本请求完成
了全部输入与512输出，首字约1111秒、解码9.85tok/s，输出出现重复且未通过
检索。该运行的显存采样峰值 `72376 MiB`，发生在加载器显式释放原始字典前。
因此，容量成功不能当作长文质量通过；后续使用独立的多样文本与2048输出预算。
这些检查不构成通用的256K模型能力评测。

最终部署的每 token 缩放版本完成了独立的多样文本验收：**260096输入tokens
+2048输出预算=262144**。输入全部token与官方tokenizer逐个一致，实际生成
214tokens后EOS结束；最终JSON正确返回文档开头、中段、末尾三个标识。
客户端首字 `885.35 s`（约14.8分钟），完整请求 `907.27 s`，稳定解码
`9.73 tok/s`，显存采样峰值 `66123 MiB`（约64.6 GiB）。原生FP4调用增量
为 `2439320`。首字时间包含约16秒的tokenizer/请求准备；服务内部首token
计时为 `868.97 s`。所有耗时均排除权重加载。

该256K输入、输出预算和早期重复文本不同，不能据此做直接提速归因。
真实容量、三处检索和数值回归通过；极端重复输入的退化仍作为已知限制保留。
最终运行、源码哈希和复现范围见
[`summary.json`](../benchmark_results/flint_context256k/summary.json)。
