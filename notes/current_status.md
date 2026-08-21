# LifeAI.jl 当前状态

## 一句话判断

项目已经形成一个可训练、可生成、可保存恢复、可评估比较，支持现代组件、KV Cache / XLA 路径，并具备版本化 Tokenizer 与文档级无泄漏中文数据管线的 decoder-only GPT；Qwen3 0.6B—32B **六个官方 dense 尺寸全部完成真实权重逐层 parity**，原始 Qwen3-30B-A3B MoE 也完成 61 GB 资产校验、Float32/native BF16 真实 parity，以及 RTX 4090 D 的 40K-capacity BF16 GPU resident/offload session。项目具备镜像 HF 语义的 **native BF16 混合精度推理路径**与可预算的 INT4/INT8/BF16 混合权重量化。Qwen3-Embedding-0.6B 的独立 checkpoint/tokenizer contract、五档 MRL 与 dense exact semantic memory 也已完成真实 BF16 parity。RTX 4090 D 上，dense 14B mixed RTN 是已实证生成上限；日常 8B BF16 已完成 XLA single-residency 4K greedy 部署与 loopback 常驻 HTTP 服务。30B-A3B 的 global/layer-balanced device cache 现在同时支持 scalar 与 grouped BF16 WMMA 直接消费分散 cache matrices，并有界复用 generation-safe pointer/workspace state；grouped-scattered 对相同 grouped 数值契约逐位一致，在 2/32-token request 相对 active-3D materialization 加速 `10.745× / 6.103×`。智能体侧已有官方 tool protocol、多 step loop、跨请求持久记忆、由环境终态评分并可联合 replay 的确定性 observation/action 闭环，以及 clean successful 环境事件的显式写回、fresh-load exact-spec 检索与无反馈后续执行。

## 当前活动阶段

[`Chapter 42 — 显式环境事件记忆写回与因果验收`](episodes/episode08_environment_action_loop/chapter42_environment_event_memory.md) 已于 2026-08-21 Closed。ModelScope Qwen3-8B 在 RTX 4090 D 上完成 8 个 full-feedback writer，全部 clean success 且与 BFS 合计同为 44 步；8 次显式 policy append 后 fresh-load journal 恰有 8 条。CPU Qwen3-Embedding raw top-1 / 生产 exact-spec context 均为 `8/8`，相关/干扰 prompt `8/8` 等 token，mirrored route `8/8` non-solving。三个 `feedback=:none` reader 臂为 no-memory `0/8`、retrieved `7/8`、matched-distractor `0/8`，相关记忆相对两组对照的双侧精确 McNemar p 均为 `0.015625`；`grid/08` 是正确检索后的 reader 执行失败。独立 replay 不加载 generation/embedding、不 append，重放 32 rows、304 prompts、273 tool outcomes、272 transitions，并重算 8 组 event/source/prefix evidence。环境事件记忆条目已关闭；Episode 08 保持 Open，下一步是跨 adapter timeout、execution failure、e-stop 和 idempotent action safety。

[`Chapter 41 — Qwen3 MoE grouped scattered expert-cache dispatch`](episodes/episode06_qwen3_moe_and_model_expansion/chapter41_qwen3_moe_grouped_scattered.md) 已于 2026-08-21 Closed，并关闭 [`Episode 06 — Qwen3 MoE 与模型架构扩展`](episodes/episode06_qwen3_moe_and_model_expansion/README.md)。新 pointer-backed `m32n8k16` / `m16n16k16` kernels 直接读取 generation-safe expert pointer tables；官方 30B 同进程 2-token 与 32-token 三路对照中，两条 grouped 路径的 prefill/decode logits、逐层 routes、每次 expert I/O 与 cache traffic 全部 exact。grouped-scattered 分别消除 `10.560 / 16.930 GB` logical active concat，请求从 `5.324 / 18.155 s` 降到 `0.495 / 2.975 s`；HF argmax 保持一致。模型线恢复时的下一架构已确定为 Qwen3-VL。

[`Chapter 40 — 确定性 GridWorld observation/action 与真实反馈闭环`](episodes/episode08_environment_action_loop/chapter40_deterministic_gridworld.md) 已于 2026-08-16 Closed，同时开启 [`Episode 08 — 环境与行动闭环`](episodes/episode08_environment_action_loop/README.md)。本章加入与模型无关的 observation/action/transition/environment 类型、hidden-wall GridWorld、唯一 allowlisted `move(direction)` 副作用面、动作预算、环境终态评分与联合 model/environment replay。冻结 8 题上 BFS oracle 和 Qwen3-4B full-feedback 均为 `8/8`、44 个动作、0 个非法动作，逐题都是最短路；首轮 prompt `8/8` 完全相同的 feedback-withheld 臂为 `0/8`、67 个非法/失败动作，配对 McNemar p=`0.0078125`。第二进程重算 148 / 148 prompts、140 / 140 tool outcomes、132 / 132 transitions 和 16 / 16 final states。环境事件记忆写回已由 Chapter 42 完成；Episode 08 只剩跨 adapter safety semantics，因此保持 Open。

[`Chapter 39 — 跨请求、可回放的语义记忆闭环`](episodes/episode07_agent_closed_loop/chapter39_persistent_semantic_memory.md) 已于 2026-08-16 Closed，同时关闭 [`Episode 07 — 智能体闭环`](episodes/episode07_agent_closed_loop/README.md)。版本化 append-only JSONL source journal、fresh-load 严格恢复、启动时 exact embedding index 重建、retrieval context 注入与 trace digest 均已接入真实模型。冻结 12 个 synthetic private-fact tasks 上，Qwen3-Embedding-0.6B top-1 / recall@1 均为 `12/12`；Qwen3-4B 的 no-memory / retrieved / token-matched distractor 为 `0/12 / 12/12 / 0/12`，两组精确 McNemar p 均为 `0.00048828125`。第二进程从同一 journal 重建的 context digest、完整 prompt digest 和 token 数为 `36/36` 一致。默认专项 `100 / 100`；真实 tokenizer 合计 `112 / 112`。

[`Chapter 38 — 工具到底帮不帮得上忙`](episodes/episode07_agent_closed_loop/chapter38_qwen3_tool_task_success.md) 已于 2026-08-16 Closed。Chapter 36 测「协议是否合法」、Chapter 37 测「单轮是否答对」，本章补上两章都留着的缺口：**工具闭环本身的任务成功率**。利用 Chapter 37 已冻结的无工具基线（Qwen3-4B 在 150 道 GSM8K 上 `141/150 = .940`），在**逐字节相同的用户消息**上做三臂配对对照：`tool-declared` 与 `tool-nudged` 都是 `137/150 = .913`，精确 McNemar p 分别为 `0.2891` / `0.3877`。

本章最重要的发现不是那 `2.67` 个点，而是**它不是工具造成的**：`tool-declared` 掉的 6 题里调用过工具的是 `0` 题，`tool-nudged` 掉的 8 题里只有 `1` 题；工具在被真正使用时基本中性（用工具的 28 题 `26/28`，同批题基线 `27/28`）。也就是说往 prompt 里加约 200 token 的 `# Tools` 头，在完全不触发工具的题上也会翻掉 4%–8% 的题。另一条：两个工具臂准确率完全相同（`137` vs `137`），却有 `8/150` 题在两个方向上互换——只看准确率会误判为「两臂等价」。工具调用的失败模式也很具体：`31` 次调用中 `3` 次失败全部集中在同一题，模型反复把**代数方程**丢给只会求值的计算器，连续三轮只换变量名重试。

[`Chapter 37 — Qwen3 dense 任务质量基线`](episodes/episode07_agent_closed_loop/chapter37_qwen3_task_quality.md) 已于 2026-08-16 Closed。这是项目第一次回答「复现出来的模型答对率是多少」，而不是「和 HuggingFace 一不一样」。冻结 MMLU 200 题（8 subject × 25，含许可与上游 row_idx provenance）与 GSM8K 150 题，跑 loglikelihood 与 generative 两种口径：0.6B/1.7B/4B 的 MMLU loglikelihood 为 `.365 / .395 / .555`，generative 全集为 `.295 / .305 / .530`，4B 的 GSM8K 为 `141/150 = .940`。

本章最重要的是两条负结果。其一，**MMLU loglikelihood 的逐题决策在 BF16 下不可复现**——既不跨 dtype，也不跨设备，甚至不跨同一实现里两条数学等价的写法：与 HuggingFace fp32 参照在协议对齐后一致率为 `193/200 = 96.5%`（协议不对齐的 fast 捷径是 `189/200`，其中约一半分歧其实来自捷径而非 dtype）；同一份权重同一协议，CUDA 给 `73`、CPU 给 `74`、fp32 给 `70`；我们自己 fast 与 general 两条路径在同一块 GPU 上也有 `6/200` 翻转，翻转题的 margin 全是 `0.125` 的整数倍，与 BF16 表示间距一致。其二，**generative 口径的数字被我们自己的 token 预算压低**：三个模型中每一个未解析的题都是被截断的题（`14/14`、`52/52`、`23/23`），因此全集数是下界、未截断子集是有偏上界，两个数必须一起报。另需记住位置偏置存在且各模型方向不同（0.6B 偏 A `128/24/24/24`、1.7B 偏 B `45/79/48/28`、4B 轻微偏 C），「永远选 A」在本题集上就有 `.270`，0.6B 的 `.365` 只高出 9.5 个点。

[`Chapter 36 — Qwen3 tools chat template HF parity 与 4B 工具调用闭环`](episodes/episode07_agent_closed_loop/chapter36_qwen3_tools_chat_template.md) 已于 2026-08-16 Closed，同时开启 [`Episode 07 — 智能体闭环`](episodes/episode07_agent_closed_loop/README.md)。本章第一次把参照物从「LifeAI 与 HF 的数值一致」换成**外部独立实现**：官方 Jinja 模板经 CPython + jinja2（不需要 torch/transformers）渲染的结果。30 / 30 case 逐字节相同，真实 Qwen3-4B tokenizer 下 token id 亦逐位相同。过程中修复了 Chapter 08 遗留的两条真实分歧——历史 assistant 的 `<think>` 块未按官方语义剥离、以及无 user 消息时 `last_query_index` 方向相反；前者会在 `--thinking` 多轮聊天的第 2 轮起真实发生。`tools` / `tool` 角色 / `tool_calls` 以官方模板 sha256 fail closed，`tojson` 按 CPython `json.dumps(ensure_ascii=False)` 复刻，含 CPython `repr(float)` 与任意精度整数。fixture 与 Close 前的对抗式复核共抓到四个真实 bug，其中最重要的一条是 `JSON3.read` 会把 `1.0` 窄化成 `Int64`，使原本的浮点 fail-closed 保护完全触发不到、静默渲染出与 HF 不同的 prompt；已改为用保留数字词法的 `parse_qwen3_json`，并拒绝 `JSON3.Object`。

[`Chapter 35 — Qwen3 MoE bounded host projection-buffer reuse`](episodes/episode06_qwen3_moe_and_model_expansion/chapter35_qwen3_moe_host_buffer_reuse.md) 已于 2026-08-12 Closed。in-place safetensors decode 与 8-slot/72 MiB final host matrix pool 只在 CUDA pageable ownership 明确安全时启用；CPU identity 和 pinned async 自动无池，上传失败也会完整归还 reader task leases。真实 30B-A3B English32 同进程 A/B 中 cold/revisit allocation 从 `28.967 GB` 降到约 `0.288 GB`（`99.01%`），latency 从 `13.770 / 11.358 s` 降到 `10.471 / 8.763 s`（`1.315× / 1.296×`），3,039 borrow/return 完整配对且所有输出 exact。

[`Chapter 34 — Qwen3 MoE bounded safetensors read-buffer reuse`](episodes/episode06_qwen3_moe_and_model_expansion/chapter34_qwen3_moe_read_buffer_reuse.md) 已于 2026-08-12 Closed。cache-backed tensor reads 现在默认按 reader worker 预分配有界 raw pool：真实 8-worker 只常驻 `25,165,824` bytes，3,039 misses 恰好借用 3,039 次。30B-A3B English32 同进程交错 A/B 中 cold/revisit allocation 从 `57.647 GB` 降到 `28.967 GB`（`49.75%`），消失的约 `28.680 GB` 与 logical raw payload 对齐；latency 从 `17.079 / 15.576 s` 降到 `14.052 / 11.653 s`（`1.215× / 1.337×`）。所有输出 exact；最终 host matrices 未复用，pinned H2D ownership 不变。

[`Chapter 33 — Qwen3 MoE safetensors decode copy elision`](episodes/episode06_qwen3_moe_and_model_expansion/chapter33_qwen3_moe_decode_copy_elision.md) 已于 2026-08-12 Closed。多维 BF16/F32 同 dtype safetensors decode 不再先复制线性数组再 `permutedims`；最终矩阵仍拥有独立存储，零维/一维仍显式 copy。真实 30B-A3B English32 cold/revisit allocation 从 `86.326 GB` 降到 `57.646 GB`（`33.22%`），消失的约 `28.680 GB` 与 logical BF16 payload 对齐；所有输出 exact。相邻实验 cold 为 `16.540 vs 17.122 s`、revisit 为 `14.882 vs 14.589 s`，因非同进程交错 A/B 而不声称通用 latency speedup。

[`Chapter 32 — Qwen3 MoE adjacent safetensors read experiment`](episodes/episode06_qwen3_moe_and_model_expansion/chapter32_qwen3_moe_coalesced_reads.md) 已于 2026-08-12 Closed。真实 checkpoint 的每个 expert 均以严格相邻的 `down → gate → up` 存放；coalesced 模式把 English32 cold/revisit 的 read syscall 从 `9,413 / 9,368` 降到 `3,304 / 3,298`（约 `64.8%`），但 request median 从 `17.122 / 14.589 s` 恶化到 `21.221 / 19.488 s`。shared-open 为 `17.147 / 14.586 s`，与原路径无可归因差异。三种模式所有输出 exact；默认保持 `expert_read_mode=:tensor`，避免合并大 read 破坏 8-worker 的 I/O 与 BF16 decode/transpose overlap。

[`Chapter 31 — Qwen3 MoE storage-aware read-worker sweep`](episodes/episode06_qwen3_moe_and_model_expansion/chapter31_qwen3_moe_read_worker_sweep.md) 已于 2026-08-12 Closed。固定 32-token English + greedy decode、4 GiB layer-balanced cache 与 scattered dispatch，每轮 cold 前对 61 GB checkpoint 执行 `POSIX_FADV_DONTNEED`，并以 `/proc/self/io` 证明实际 storage reads 约 `26.236 GB`；紧随 revisit 为零 storage bytes。1/2/4/8-worker cold 中位 `48.181 / 30.614 / 20.737 / 16.962 s`，revisit 为 `37.349 / 23.316 / 17.015 / 14.553 s`，8-worker 分别加速 `2.840× / 2.566×`，全部输出 exact。overlap 保持 opt-in，但未显式配置时 worker 改为 `min(8, Threads.nthreads())`。

[`Chapter 30 — Qwen3 MoE bounded async miss pipeline`](episodes/episode06_qwen3_moe_and_model_expansion/chapter30_qwen3_moe_async_miss_pipeline.md) 已于 2026-08-12 Closed。router 完成后，当前层 cache misses 可由 bounded Julia tasks 并行读取，再按确定 active-expert 顺序上传/写入 cache；entry generation、LRU 和容量语义不变。CUDA 另有按 worker window 有界、显式同步/解绑的 pinned transfer stream。真实 30B-A3B 4-thread warm-page-cache 实验中，pageable overlap 把请求中位从 `10.677 s` 降到 `5.423 s`（`1.969×`），全部 logits exact；pinned 为 `5.481 s`，没有额外收益，因此推荐显式 4-worker pageable overlap，默认仍保持 sequential。

[`Chapter 29 — Qwen3 MoE scattered dispatch state reuse`](episodes/episode06_qwen3_moe_and_model_expansion/chapter29_qwen3_moe_scattered_reuse.md) 已于 2026-08-12 Closed。CUDA session 现在按 ordered expert ids + 单调 entry generations 匹配 pointer plan，避免 allocator 地址重用造成 stale plan；每层最多保留 4 plans，全 session 最多保留 4 个 route-shape workspaces，cache clear/reconfigure 会同步失效。真实 30B-A3B hit 的 96 次调用均零 pointer build/upload、零 workspace allocation，只保留 `26,832 + 294,912` bytes 状态。`gc=8` 连续 100 次 exact/零 I/O 且 free checkpoints 不下降；`gc=0` 连续 500 次没有 live leak，reclaim 后完全恢复，但运行中 allocator pool 仍扩张 `885.9 MB`，p95/max 为 `283 / 758 ms`，所以长期建议仍为每 8 层 GC。

[`Chapter 28 — Qwen3 MoE scattered expert cache dispatch`](episodes/episode06_qwen3_moe_and_model_expansion/chapter28_qwen3_moe_scattered_cache.md) 已于 2026-08-12 Closed。旧 cache hit 虽为零 expert read/upload，仍在 48 层 prefill + 48 层 decode 中 materialize 96 次 active tensors，共 `10.551 GB` D2D copy，并每层强制 GC。新 CUDA indexed/bucketed kernels 只上传 `26,832` bytes device pointer tables，直接读取缓存持有的 BF16 matrices；`scattered + gc8` 的 hit prefill/decode/request 为 `0.078 / 0.073 / 0.151 s`，相对 materialized request 加速 `82.89×`，所有 logits exact 且 Transformers reference argmax 一致。`gc=0` 的 100 次重复虽有 `0.109 s` 中位数，却损失 `0.9375 GiB` allocator free 并出现 `0.729 s` 尾延迟，因此冻结推荐为每 8 层 GC；pointer/workspace reuse 与异步 miss path 后续由 Chapter 29–31 完成，grouped scattered 已由 Chapter 41 闭环。

[`Chapter 27 — Qwen3 MoE layer-balanced expert cache`](episodes/episode06_qwen3_moe_and_model_expansion/chapter27_qwen3_moe_layer_balanced_cache.md) 已于 2026-08-12 Closed。全局 LRU 在 48 层顺序扫描且工作集大于缓存时会 thrash：真实 30B `English32 → Chinese32 → English32` trace 即使给 8 GiB 也只有 `4.73%` 命中。按层公平保留 9/14/18 slots 的 4/6/8 GiB 配置将命中率提升到 `12.07% / 17.81% / 22.56%`，I/O 单调降到 `76.29 / 71.32 / 67.19 GB`。同 8 GiB 对照少读 `18.71%`、加速 `1.082×`；但 4 GiB 只比 8 GiB 慢 `1.48%`，最终保留 `7.44 GB` vs `1.88 GB` GPU free，因此成为该冻结 trace 的建议。所有 English replay logits/token exact；当时的 device concat/GC 与同步 miss path 已由 Chapter 28–31 继续优化。

[`Chapter 26 — Qwen3 MoE active-expert device cache`](episodes/episode06_qwen3_moe_and_model_expansion/chapter26_qwen3_moe_expert_cache.md) 已于 2026-08-12 Closed。`HFQwen3MoEOffloadSession` 新增默认关闭、按实际 tensor bytes 预算的 device LRU，key 为 one-based layer/expert；request reset 保留缓存，显式 clear 撤销缓存 tensor 的逻辑所有权。真实 30B-A3B 8 GiB case 容纳 892 entries、`8.418 GB`，零淘汰且最终仍有 `4.346 GB` GPU free。重复请求 prefill 734 次、decode 384 次全部命中，expert read/upload 从 `8.418 GB` 降到零；prefill/decode/request 相对 warm fill 加速 `1.775× / 1.533× / 1.722×`，fill/hit exact 且 Transformers BF16 reference argmax 一致。该章的零淘汰结论只适用于冻结短请求；长自然文本容量与 scan thrashing 已在 Chapter 27 单独量化，pinned-memory 与异步预取仍未完成。

[`Chapter 25 — Qwen3-30B-A3B GPU resident/offload session`](episodes/episode06_qwen3_moe_and_model_expansion/chapter25_qwen3_moe_gpu_offload.md) 已于 2026-08-12 Closed。真实官方 30B-A3B 现在以 attention/router/norm/LM head 常驻、active experts 逐层从 safetensors 上传的方式在 RTX 4090 D 运行；40,960-token BF16 static KV `3.75 GiB` 已实际分配，常驻参数 `2.291 GiB`，加最坏单层 experts 的工作集硬下限为 `7.166 GiB`。2-token prompt/decode 与 Transformers BF16 reference argmax 均一致；32-token grouped steady 为 `23.82 / 4.63 s`，相对 scalar production `26.02 / 6.18 s` 为 `1.092× / 1.335×`。小 2-token case 的 grouped 只有 `0.938× / 0.899×`，因此宽 prefill 才启用 WMMA。完整 40K window 填充和长序列质量仍未声称完成。

[`Chapter 24 — Qwen3 MoE 架构支持`](episodes/episode06_qwen3_moe_and_model_expansion/chapter24_qwen3_moe_architecture.md) 已于 2026-08-12 Closed。CPU Float32 correctness、Transformers tiny/官方 30B parity、Float32/native BF16 active-expert streaming、官方 30B-A3B immutable 资产契约、compact Reactant/XLA CPU 与 RTX 4090 D CUDA indexed/bucketed dispatch 已完成。官方 61 GB checkpoint 的 16 分片 SHA256 全部通过；Float32 的 `1,152 / 1,152` 路由槽位一致，prompt/decode logits max-abs `4.39e-5 / 1.53e-5`；native BF16 路由槽位重合 `95.92%`，两组 logits max-abs `0.3125`，两种口径 argmax 全一致。BF16/Float32 streamer 峰值 RSS `3.50 / 4.62 GiB`。该章关闭时默认全套 `5,948 / 5,948`，XLA `3 / 3`、CUDA `38 / 38`；当时留下的 40K GPU session 已由 Chapter 25–28 继续完成。

[`Chapter 23 — Qwen3 XLA 设备端采样`](episodes/episode05_deployment_memory_and_sampling/chapter23_qwen3_xla_device_sampling.md) 已于 2026-08-01 Closed。temperature / top-k / top-p / inverse-CDF 采样策略整体进入编译好的 prefill/decode executable：宿主每 token 只送一个 4 字节 uniform、取回一个整数，不再传回 151,936 维 logits。策略用 `top_k` 次 masked reduction 提取候选（不排序），nucleus 改写为「严格排在候选之前的质量 < top_p」，inverse-CDF 仍按词表 index 顺序走。真实 Qwen3-0.6B 在本机 RTX 5080 CUDA XLA 上以同一串 uniform replay，与宿主策略 **38/38 token 完全一致**，decode 从 `23.66` 提升到 `237.23` tok/s（`10.03×`），已贴近 greedy 的 246 tok/s。`top_k` 是编译期常量，temperature/top_p 是运行期设备标量。

[`Chapter 22 — Qwen3-Embedding-0.6B 与最小语义记忆`](episodes/episode05_deployment_memory_and_sampling/chapter22_qwen3_embedding_memory.md) 已于 2026-07-31 Closed。独立的 151,669-vocabulary / 32K embedding contract、官方尾 `<|endoftext|>` post-processor、base-model safetensors namespace、变长批 mask、last-token pooling、五档 MRL 和 dense exact cosine memory 已完成；真实 CPU 与 RTX 4090 D CUDA BF16 的 token/mask、15 组 top-k 全一致，embedding/similarity max-abs 均低于 0.01。

[`Chapter 19 — Qwen3-8B / RTX 4090 D 日常本地部署`](episodes/episode05_deployment_memory_and_sampling/chapter19_qwen3_8b_4090d_deployment.md) 已于 2026-07-30 Closed。本阶段明确区分容量上限与日常选择：14B mixed RTN 是同卡已实证生成上限，但仅 0.377 tok/s 且上下文余量很小；日常部署选择无量化误差的 8B BF16，冻结为 4K 总 context、3,584-token prompt/history、512-token output、64-token 分块 prefill。CUDA eager 路径的 3,584 prefill 为 46.85 s，3,584+512 整窗 decode 为 10.25 tok/s，保留 sampling 与多轮历史裁剪能力。

[`Chapter 18 — Qwen3 Activation-Aware INT4 校准`](episodes/episode04_efficient_inference_and_quantization/chapter18_qwen3_activation_calibration.md) 已于 2026-07-29 Closed。独立的 8×32 calibration token、native BF16 CPU/GPU 逐层 activation second-moment 采集、`:activation_mse` scale search 与 streamed/in-memory 一致性均已完成。RTX 4090 D 上同 12.093 GiB mixed layout 的真实结果为 4/16、首次分歧第 5 token，没有守住 Chapter 17 mixed RTN 16/16；该 diagonal 近似不称为完整 AWQ/GPTQ。

[`Chapter 17 — Qwen3 Reconstruction-Calibrated INT4 与预算化混合精度`](episodes/episode04_efficient_inference_and_quantization/chapter17_qwen3_calibrated_int4.md) 已于 2026-07-29 Closed。per-row/group MSE clipping search、按层/投影/LM head 覆盖的 INT4/INT8/BF16 计划，以及真实参数树/Qwen3 topology 的逐 byte 预算均已完成。RTX 4090 D 实测：14B 全 INT8（tree 14.487 GiB）和 12.093 GiB mixed RTN 均与 BF16 reference 16/16 greedy 一致；同布局 mixed MSE 虽降低 full-logits max/mean error，却只有 4/16，证明 reconstruction error 不能作为生成 fidelity 代理。

[`Chapter 16 — Qwen3 XLA BF16 Compiled Decode 与 INT8/INT4 量化`](episodes/episode04_efficient_inference_and_quantization/chapter16_qwen3_xla_decode_quant.md) 已于 2026-07-26 Closed。XLA BF16 static-cache decode 编译完成：设备端 greedy 闭环 steady **246 tok/s**（eager 的 16.1 倍），16 步 greedy 与 HF BF16 全对。RTN 量化让 8B（INT8，8.22 GiB）与 14B（INT4，8.38 GiB）首次驻留 16.3 GiB GPU：8B token 级行为近乎无损（greedy 14/16，仅近平局翻转），14B INT4 一致率 4/16 如实冻结——无校准 INT4 的生成保真是明确的下一个边界。

[`Chapter 15 — Qwen3 BF16 CUDA / XLA Accelerated Inference`](episodes/episode04_efficient_inference_and_quantization/chapter15_qwen3_bf16_accel.md) 已于 2026-07-26 Closed。设备通用向量化 BF16 路径（CPU 上与 Chapter 14 循环路径逐位相同）在 RTX 5080 上以原生 BF16 张量核运行：0.6B/1.7B/4B CUDA parity 与 16 步 greedy 全部与 HF BF16 一致，吞吐 15.3/14.1/8.1 tok/s（CPU 的 33—92 倍），VRAM ≤ 12.1 GiB；Reactant XLA BF16 编译 prefill 通过（编译 44.8 s、steady 1.36 ms）。推理验证主战场从 CPU 移至 CUDA/XLA；8B GPU 驻留超出 16.3 GiB VRAM 保持边界。

[`Chapter 14 — Qwen3 Native BF16 Mixed-Precision Compute`](episodes/episode04_efficient_inference_and_quantization/chapter14_qwen3_bf16_compute.md) 已于 2026-07-26 Closed。独立 BF16 推理路径逐算子镜像 Transformers 4.51.0 混合精度语义；0.6B/1.7B/4B/8B 与 HF BF16 逐层对齐、argmax 零失配、16 步 greedy token 序列完全一致；参数常驻内存减半，8B 完成本机首个 >4B 全量驻留 greedy 生成（峰值 RSS 19.0 GiB）。既有 F32 / 流式 / XLA 路径零改动。

[`Chapter 13 — Qwen3 Streamed Loading and 8B/14B/32B Real-Weight Parity`](episodes/episode03_model_family_and_large_weights/chapter13_qwen3_streamed_large_weights.md) 已于 2026-07-25 Closed。流式 / 逐层 safetensors 加载（与 in-memory 路径逐位一致）让 8B / 14B / 32B 在峰值 ≤ 8.9 GiB 内完成真实权重逐层 parity，untied LM head 获得真实证据；Qwen3 dense family 六尺寸的真实权重验证闭环就此完成。

[`Chapter 12 — Qwen3 Dense Family Real-Weight Parity`](episodes/episode03_model_family_and_large_weights/chapter12_qwen3_dense_real_weights.md) 已于 2026-07-25 Closed。1.7B 与 4B 真实权重经分片 safetensors index 加载并在显式容差内完成逐层 hidden/logits/dynamic/static decode parity，loader 零改动一次通过；tied embedding 的三个官方尺寸（0.6B/1.7B/4B）全部实跑。8B/14B/32B 因 Float32 全量加载需 30.5—122.1 GiB、超出本机 30 GiB RAM，保持显式未验证边界。

## 已实现能力

### 1. 模型基本组件

- scaled dot-product attention：同时保留手写版本与基于 `NNlib.batched_mul` 的批量版本（均支持 GQA/MQA 分组），便于原理对照和正确性验证。
- Multi-Head Attention：包括 Q/K/V/O 投影、head reshape / merge 和 causal mask；`head_dim` 可独立于 `d_model` 配置，`use_bias` 可关闭。
- GQA / MQA：`num_kv_heads` 独立可配，K/V 投影与 KV cache 按 KV head 数缩减；manual reference、无物化分组实现与 `repeat_kv` 展开三路等价性已测试钉死。
- QK-Norm（Qwen3 语义）：per-head RMSNorm、独立 q/k scale、位于 head reshape 之后 RoPE 之前，独立开关，关闭时参数树与 legacy 完全一致。
- RoPE：支持预计算 cos / sin cache、可配置 `rope_theta` 和增量解码所需的绝对起始位置；同时支持 legacy `:interleaved` 与 HF Qwen3 `:rotate_half` 配对。
- learned absolute position：full、dynamic/static KV cache 与 XLA decode 共用显式绝对位置；位置表上限 fail closed。
- TransformerBlock：采用 pre-norm、attention residual 和 MLP residual，可独立选择 LayerNorm / RMSNorm 与 GELU / GELU-New / SwiGLU / Qwen3 sparse MoE。
- GPTModel：包括 token/可选 position embedding、多层 TransformerBlock、final norm 和 LM head；支持 embedding / LM head 单 kernel 权重共享，并可分离 projection bias 与 LM-head bias。
- legacy 默认仍为 LayerNorm + GELU + untied；modern 配置可通过独立开关组合，不改变旧调用。
- HuggingFace Qwen3 dense 导入：冻结 0.6B / 1.7B / 4B / 8B / 14B / 32B 六个官方规格与 config checksum，可自动识别或显式要求 variant；严格解析 config，读取 BF16/F32 safetensors 单文件或 index 分片，完整映射 embedding、attention、QK-Norm、MLP、final norm 与 tied/untied LM head；missing、unexpected、duplicate、shape/dtype/config 错误均 fail closed。六个真实 checkpoint 全部实跑逐层 parity：0.6B—4B 全量加载，8B—32B 流式加载。
- Qwen3 MoE 导入（Chapter 24–35、41，Closed）：Float32 top-k routing、逐 expert SwiGLU、all-sparse decoder topology、原始 `mlp.experts.N.*` 权重名映射和 full/dynamic/static cache 已完成 Transformers tiny/官方 30B parity。`stream_hf_qwen3_moe_forward` 以 header-only index 在路由后只读取 active experts，可选择 Float32/native BF16；官方 immutable revision、30.53B 参数、config/index 与 16 分片 checksum 已形成代码级资产契约。Reactant/XLA CPU 使用 compact route-major fallback；RTX 4090 D 具备 indexed/bucketed/grouped WMMA，并由 `HFQwen3MoEOffloadSession` 将真实 streamer、全容量 static KV 和 global/layer-balanced device LRU 接成可运行的 30B session。scalar 与 grouped CUDA cache 都可用 device pointer tables 直接读取分散 BF16 expert matrices，有界复用 generation-safe pointer/workspace state，并独立配置 forced-GC cadence；当前层 post-router misses 可用 storage-verified bounded parallel reads/pinned upload pipeline。batch reader 可严格合并相邻 safetensors ranges，但实测默认仍应逐 tensor 读取；同 dtype decode copy、raw payload allocation 与 CUDA pageable final host-matrix allocation 已依次消除。
- Qwen3-Embedding-0.6B 导入（Chapter 22）：独立冻结 HF revision 和 8 个
  asset SHA256，严格区分 151,669 vocabulary、32K model context、
  SentenceTransformers base-model namespace 与 causal-LM contract；
  `load_hf_qwen3_embedding_bundle` 以 BF16 载入 595,776,512 参数，
  hidden-state 前向不投影 vocabulary logits、也不保留 KV cache。
- 流式 safetensors 加载（Chapter 13）：`open_safetensors_reader` header-only 索引 + `read_safetensors_tensor` 按需读取 + `stream_hf_qwen3_forward` 逐层 trace 与 dynamic decode；embedding 按 token 行从磁盘 gather，单层参数映射与 in-memory loader 共用；与全量路径逐位一致（合成 fixture 与真实 0.6B 验证），32B 峰值 RSS 8.9 GiB。
- native BF16 混合精度推理（Chapter 14）：`load_hf_qwen3_model(...; weight_dtype=BFloat16)` 位保真加载（参数内存减半）+ `hf_qwen3_bf16_forward` 独立路径，逐算子镜像 HF 语义（RMSNorm/QK-Norm/softmax F32 归一化、RoPE 表 F32 转 BF16、线性 BF16 存储 + 分块 F32 累加 + BF16 舍入）；BF16 cached decode 与全量前向逐位等价；0.6B—8B 与 HF BF16 argmax 零失配、16 步 greedy 完全一致。
- BF16 CUDA/XLA 加速推理（Chapter 15）：`hf_qwen3_bf16_accel_forward` 设备通用向量化路径（CPU 上与循环路径逐位相同），CUDA eager 用原生 BF16 张量核（CUBLAS/batched_mul，F32 累加），Reactant XLA 可编译同一实现；0.6B—4B GPU parity/greedy 全对，吞吐 8—15 tok/s；CPU batched matmul 显式分派为 F32 累加防止通用 fallback 破坏契约。
- XLA 设备端采样（Chapter 23）：`_device_sample_next_index` 只用 reduction、
  elementwise select 与标量算术表达 temperature/top-k/top-p/inverse-CDF，
  同一份实现既是宿主参考也被 trace 进 executable；`:device_sample` session
  策略把 `top_k` 固化为编译常量、temperature/top_p 作为运行期设备标量，
  `sample_uniforms` 支持与宿主策略逐 token 对拍。已知契约差异：宿主/HF
  保留与第 k 名并列的全部 token，设备保留恰好 `top_k` 个（并列取较大
  index，与 HF 稳定排序方向一致）。
- XLA BF16 compiled decode（Chapter 16）：static KV cache 的 traced prefill/decode/greedy（traced position、动态写、有效前缀掩码），设备端 argmax 闭环使宿主每 token 只取回一个整数；0.6B steady 246 tok/s（eager 16.1×），greedy 与 HF 全对。
- Qwen3-8B XLA single residency（Chapter 20）：`load_hf_qwen3_compact_model`
  直接以 BF16 流式读取并按层合并 Q/K/V 与 gate/up，不构造完整 unpacked
  参数树；`load_hf_qwen3_bf16_xla_session` 对最终 291-leaf tree 只调用一次
  `Reactant.to_rarray`。分块 K/V 写入每层各 lower 为一个
  `dynamic_update_slice`，替代 64 次逐 token 写入。
- Qwen3-8B XLA resident HTTP service（Chapter 21）：
  `Qwen3XLAHTTPService` 只调用一次 loader，默认仅监听 loopback；
  `/healthz` 与严格 `/api/generate` 支持 buffered/NDJSON、UTF-8 安全
  streaming、1 MiB body/context/options 门禁和 generation single-flight。
  日常 client 只加载 tokenizer，8B 参数与 executable 常驻 server。
- RTN 权重量化（Chapter 16）：INT8 per-channel / INT4 group 打包 + 混合精度选项；`load_hf_qwen3_quantized` 逐投影流式量化加载；分块反量化线性层；8B INT8 与 14B INT4 首次 GPU 驻留（各约 8.2/8.4 GiB），量化只改权重驻留格式不改计算契约。
- 校准与预算化量化计划（Chapter 17）：`LinearQuantizationSpec` /
  `QuantizationPlan` 统一流式与 in-memory 量化，支持 default、projection、
  one-based layer 和独立 LM-head override；INT4 可选确定性
  reconstruction-MSE clipping，`quantized_parameter_bytes` 与
  `estimate_qwen3_quantized_bytes` 对真实树/静态 topology 逐 byte 对齐。
  14B 实测同时冻结正负证据：全 INT8 和 mixed RTN 16/16，mixed MSE 4/16。
- activation-aware INT4（Chapter 18）：`ActivationCalibration` /
  `activation_second_moment` 以 one-based layer/target 绑定输入二阶矩；
  `calibrate_hf_qwen3_activations` 逐层读取权重，支持 CPU 位精确路径和
  无 CUDA 硬依赖的 device callback 加速路径；`:activation_mse` 缺统计、
  错维度、NaN/负值或空统计全部 fail closed。14B 实测 max-logit error
  `3.4238`，但 greedy 仍只有 4/16。
- HuggingFace GPT-2 导入：冻结 revision/checksum，严格映射 learned position、LayerNorm bias、fused QKV 与 HF Conv1D `(in, out)`，验证 causal buffers 与 tied LM head；完整 context 参数量 124,439,808。
- 显式 `hf_token_ids` 处理 HF 0-based 到 LifeAI 1-based 边界；逐层 trace 与 reference 脚本可验证 embedding、每个 block、final hidden、full logits 和 cache decode logits。`Lux.parameterlength` 已包含自定义 Q/K-Norm scale，六个 dense topology 的精确参数量与冻结 reference 一致。

### 2. Tokenizer 与数据

- `AbstractTokenizer` 统一接口：character、byte、byte-BPE、imported HF Qwen3 与 imported HF GPT-2 共用 encode / decode / vocab / special-token / fingerprint API，token id 保持 1-based。
- legacy character `Tokenizer` 完整保留，旧调用与旧 checkpoint 不受影响。
- `ByteTokenizer`：对任意有效 UTF-8 无 OOV、可精确 round-trip；`decode_bytes` 始终可逆，`decode` 提供显式 `:strict` / `:replace` 策略。
- `ByteBPETokenizer`：train-only 确定性训练，固定 tie-break，相同语料与配置产生相同 vocabulary、merge ranks 和 fingerprint。
- `HFQwen3Tokenizer`：严格导入目标 revision 的 NFC、regex、ByteLevel、151,643-token BPE、151,387 merges 与 26 个 added tokens；支持 HF character/Julia UTF-8 byte spans、byte-exact decode、special-token 语义和 1-based 公共 ids。
- Qwen3 embedding tokenizer profile：只接受官方
  `Sequence(ByteLevel, TemplateProcessing)` 的尾 `<|endoftext|>` 模板，
  与 generation profile 分开持久化和 fingerprint；两者不能静默互换。
- `HFGPT2Tokenizer`：严格导入 GPT-2 regex、ByteLevel、50,257-token BPE、50,000 merges 与 `<|endoftext|>`；10 组 ASCII/Unicode/空白/控制字节/special corpus 与 HF 完全一致。
- Qwen3 基础 chat template：支持无 tools 的 system/user/assistant、generation prompt 与 thinking 开关；模板 hash 与三份 tokenizer config checksum 纳入 provenance/fingerprint，未知 revision fail closed。
- Tokenizer artifact v1：显式 schema version、normalization、special tokens、vocabulary / merges 与内容指纹，可独立保存、加载与校验，篡改被拒绝。
- 中文数据管线：以 document 为单位记录来源、许可、checksum、变换配置；确定性文档级无泄漏 split；Tokenizer 只在 train split 上拟合；versioned dataset artifact 与显式 EOS 边界语义。
- 滑动窗口 DatasetLoader 与 DocumentDatasetLoader，支持 batch、stride 和 `drop_last`。
- 稀疏 next-token cross entropy；token-weighted validation loss、perplexity 与 `bits_per_byte` 等 byte-normalized 评估。
- checkpoint format v2：版本化、设备无关，支持全部五类 Tokenizer 的保存恢复，并显式迁移 v1 legacy checkpoint。
- 确定性 checkpoint resume、可配置 global gradient norm clipping、Zygote 常规训练路径与 Reactant + Enzyme 的 XLA 训练路径。

### 3. 生成与推理

- greedy、temperature、top-k 和 top-p sampling；基础生成入口对全部五类 Tokenizer 通用。
- 动态 KV Cache（prompt prefill、单 token decode、cached generation）与固定形状静态 KV Cache（面向编译后增量推理）。
- XLA prefill / decode 接口及编译后生成流程。
- Qwen3-8B XLA 4K batch-1 greedy session：固定 64-token prompt bucket、
  左填充 mask、4,095-slot K/V、设备端 argmax、连续请求 reset 与多轮
  history 裁剪；`scripts/run_qwen3_xla_chat.jl` 是 4090 D 日常入口。
  XLA sampling 尚未作为 Chapter 20 日常 profile 开放。
- full forward、动态 KV Cache、静态 KV Cache 的 correctness matrix 与 microbenchmark。
- CPU、CUDA GPU、XLA CPU、XLA GPU 独立进程 benchmark，可区分 cold compile、warm-up 和 steady-state。
- `load_hf_qwen3_bundle` / `generate_hf_text` 串联本地模型、tokenizer、EOS 与 greedy trace；Qwen3-0.6B 的 full、dynamic、static 生成及 host-tokenizer→XLA static 路径已验证。
- Qwen3 generation config 严格解析、`:sample` / `:config` 与固定 uniform CDF replay 已完成；真实 HF sampled integration 86 / 86 通过，16 步 token/candidate/文本完全一致，概率 global max-abs `5.90086e-6`。
- Qwen3 rotate-half RoPE 已用 Transformers 4.51.0 独立 fixture 验证到 position 40,959；真实 0.6B 的 CPU、CUDA 与 Reactant-XLA GPU cache correctness/benchmark 均有冻结条件和原始 JSON。
- `load_hf_gpt2_bundle` / `generate_hf_text` 串联冻结 GPT-2 模型/tokenizer；embedding、12 blocks、final hidden、full logits 与 full/dynamic/static 8-step greedy text 均通过 Transformers Float32 reference。

### 4. 最小语义记忆

- `embed_texts` 支持 instruction-aware query、left/right padding、right
  truncation、mask-aware last-token pooling，以及
  1024/512/256/128/64 维先截断后 L2 normalization。
- `Qwen3SemanticMemory` / `build_qwen3_semantic_memory` /
  `search_qwen3_semantic_memory` 提供内存内 dense exact cosine top-k；
  文档、embedding 列与 metadata 强校验，稳定按 document index
  处理同分。`examples/qwen3_embedding_memory.jl` 已实跑仓库 notes，
  默认 query 首项稳定返回 Chapter 22。
- Chapter 22 的 dense semantic baseline 本身仍是内存内 exact index；Chapter 39 已增加持久 source
  journal，Chapter 42 再增加 clean environment-event 的显式 policy 写回与 exact-spec retrieval。
  当前仍没有 ANN、reranker、通用摘要/遗忘或并发 writer，因此“已形成可回放链路”不等于“长期
  记忆系统完成”。

### 5. 学习与可视化记录

`notebook/` 已覆盖 Attention 结构、RoPE、prefill / decode、KV Cache 原理与常见错误、动态与静态 cache 等主题；这些 notebook 不只是展示结果，也是关键组件学习过程的一部分。

## 验证状态

运行默认测试套件：

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

2026-08-21 最新复核默认套件，共 `8,346 / 8,346` 项测试通过；Episode 06
为 `1,866 / 1,866`，其中 Chapter 41 真实结果 contract 为 `60 / 60`，Chapter
25–35 的 per-report historical provenance 与原有非 GPU contracts 合计
`1,521 / 1,521`。该计数包含官方 30B-A3B immutable 资产契约、Float32/BF16 tiny
streaming、真实 parity/offload/cache 冻结报告契约、device LRU 生命周期/淘汰、
scan-thrashing、scattered dispatch/GC、generation-safe pointer/workspace reuse、
bounded miss pipeline、storage-aware worker sweep、decode-copy/buffer ownership，
以及 grouped-scattered 真实时延/traffic/materialization 合约；Episode 08 为 `244 / 244`，其中
Chapter 40 / 42 分别为 `126 / 126` 与 `118 / 118`。默认计数不包含需显式启用的 XLA/CUDA
accelerator 专项。
Chapter 24 compact dispatch 的 Reactant/XLA CPU 专项另计 `3 / 3`：
128 experts/top-8/64 tokens 的 route pairs 为 `512 / 8,192`，编译
`32.126 s`、steady median `38.616 ms`，对 dense oracle max-abs `9.09e-7`。
RTX 4090 D CUDA 专项另计 `228 / 228`，其中 Chapter 28/29/30/35/41 分别为
`13 / 13`、`90 / 90`、`11 / 11`、`29 / 29` 与 `47 / 47`；indexed kernels 的单-token/64-token
steady median 为 `0.365 / 0.377 ms`，相对 route-major baseline 加速
`1.32× / 1.57×`，64-token 相对单线程 CPU sparse 加速 `4.06×`；
workspace 缩减 `118.15×`，两组 max-abs 均低于 `6.56e-7`。官方
`2,048→768` 投影宽度的 BF16 indexed dispatch 为 `0.180 / 0.689 ms`
（1/8 token）；32/64 token 自动切换到 bucketed，分别从
`2.813 / 6.730 ms` 降至 `1.896 / 2.052 ms`，输出逐位一致。
纯设备 grouped BF16 WMMA 使用 `m32n8k16` 后，单投影在 32/64/128/256
token 相对 scalar grouped 达到 `1.03× / 1.18× / 1.61× / 2.32×`；
共享 padded workspace 且直接 combine 的完整实验路径相对现有 bucketed
dispatch 达到 `1.12× / 1.08× / 1.55× / 2.60×`。grouped 两种实现
输出逐位相同；相对 Float32-activation bucketed 的 max-abs ≤ `1.86e-5`。
Chapter 23 的真实 0.6B CUDA XLA 验收（同 uniform replay）38/38 token 一致、
decode `237.23` vs `23.66` tok/s，报告顶层 `closed=true`。Chapter 22 加真实 Qwen3-Embedding-0.6B 权重为 `103 / 103`；Chapter 21 加真实 loopback socket opt-in 为 `116 / 116`；compact fixed-chunk prefill 在 Reactant CPU 编译执行 `5 / 5` 通过。

Chapter 22 的真实 CPU BF16 reference 使用 PyTorch `2.7.1+cpu` /
Transformers `4.51.3`。LifeAI token ids 与 attention mask 逐项相同；
1024/512/256/128/64 维 embedding max-abs 为
`0.00360 / 0.00502 / 0.00667 / 0.00826 / 0.00927`，similarity
max-abs 全部低于 `0.00830`；五档共 15 组完整 top-k、semantic memory
首项全部一致，验收 JSON 为 `closed=true`。i9-14900K 上 8 条/35-token
batch 的资产校验、BF16 load、forward 为 `2.436 / 8.832 / 7.687 s`。
RTX 4090 D（driver `570.153.02`、CUDA runtime `12.9.0`）上相同 batch
的参数上传、cold forward、warm forward 为
`1.103 / 18.959 / 0.0641 s`，冷/热 embedding max-abs 为 `0`；CUDA
token/mask、五档数值、15 组 top-k 和 semantic memory 同样全部通过。

Chapter 21 的真实 resident-service 验收在唯一 PID 内完成 15 个独立 HTTP
请求，最终 health 为 15 completed / 0 failed、`load_count=1`、
`max_active=1`。三组 CUDA BF16 oracle 为 `96 / 96`；steady decode
`41.351 tok/s`，3,584+512 prefill/decode 为 `1.483 s / 41.351 tok/s`。
10 次复用 allocator drift 167,680 bytes；1,228 个 200 ms 样本最低
free 2,304 MiB（2.251 GiB）。冷 service constructor 仍需 205.32 s，
但常驻后的每个新客户端不会重复 load/transfer/compile。

Chapter 20 的真实 8B XLA 验收使用冻结 revision 与 deterministic CUDA
BF16 oracle：64-token 单 chunk、65-token 左填充和 3,584-token 多 chunk
三组共 `96 / 96` greedy tokens 一致。唯一 compact 参数树 291 leaves、
16,381,470,720 logical bytes，只做一次递归 device transfer；3,584
prefill `1.49764 s`，3,584+512 decode `41.132 tok/s`。8 次复用请求
allocator drift 为 234,752 bytes；1,225 个 200 ms `nvidia-smi` 样本的
最低物理 free 为 2,301 MiB，最终报告 `closed=true`。

Chapter 19 CUDA eager 日常入口专项 `80 / 80`；同一 8B BF16/4K profile
的 3,584 prefill 为 `46.849 s`，整窗 decode 为 `10.246 tok/s`，外部
最低 free 为 2,099 MiB。它继续承担 XLA daily profile 尚未覆盖的
temperature/top-k/top-p sampling。

Chapter 15 的离线 `63 / 63` 覆盖 CPU batched matmul F32 累加契约、向量化 vs 循环路径逐位一致（tied/untied）与 CUDA/XLA 资产 contract；CUDA opt-in（`LIFEAI_QWEN3_BF16_CUDA_0_6B/1_7B/4B_MODEL_DIR` 同进程 `Pkg.test`）`173 / 173` 通过。CUDA 三尺寸 16 步 greedy 与 HF BF16 全部一致，吞吐 15.3 / 14.1 / 8.1 tok/s，VRAM 峰值 ≤ 12.1 GiB；XLA BF16 prefill 编译 44.8 s、steady 1.36 ms，argmax/greedy 首 token 一致（`scripts/verify_qwen3_bf16_xla.jl`）。

Chapter 16 离线专项覆盖 RTN round-trip/打包语义（48 项）、量化树驱动 accel 前向（13 项）与 XLA/量化资产 contract。GPU/XLA 实测经脚本冻结：XLA compiled decode（`scripts/verify_qwen3_bf16_xla_decode.jl`）设备端 greedy steady 4.06 ms/token = 246 tok/s，两条编译路径 16 步 greedy 与 HF 全对；8B INT8（`scripts/verify_qwen3_quant_cuda.jl`）树 8.22 GiB、argmax 全对、greedy 14/16（第 15 token 近平局）；14B INT4 g128 树 8.38 GiB、prefill argmax 对、greedy 4/16（第 5 token 分歧）。量化吞吐 0.11—0.61 tok/s（每 token 全量反量化，带宽瓶颈，驻留验证非吞吐目标）。

Chapter 17 专项 `83 / 83` 覆盖 MSE candidate 局部不劣于 max-abs、outlier
严格改善、计划校验/优先级、streamed vs in-memory 逐 tensor 一致、
legacy API 位兼容、tied/untied LM head 与 14B tensor-byte/硬件 fixture。
Qwen3-14B 全分片 SHA256 与冻结 revision 一致；RTX 4090 D 实测全 INT8
参数 tree 14.487 GiB、VRAM 23.453 GiB、16/16 greedy、0.816 tok/s，
mixed RTN tree 12.093 GiB、VRAM 22.597 GiB、16/16、0.377 tok/s；
同布局 mixed MSE VRAM 22.521 GiB、full-logits error 更低但 greedy 4/16。
全 INT8 仅余约 54 MiB 显存，不视为有部署余量。

Chapter 18 专项覆盖 activation-weighted candidate 局部不劣于 max-abs、
偏斜 activation fixture 严格改善、统计契约 fail-closed、CPU/accelerated
采集逐值一致、streamed/in-memory 逐 tensor 一致，以及 calibration/plan/
14B 硬件资产 checksum。RTX 4090 D 实测 256-token 校准 `248.32 s`、
量化加载 `489.23 s`、tree `12.093 GiB`、VRAM `21.474 GiB`；
full-logits max/mean error `3.4238 / 0.42865`，greedy 4/16、第 5 token
首次分歧。相较 mixed RTN，max/mean error 均略低但生成轨迹显著更差；
相较 weight-MSE，max 更低、mean 更高而 greedy 相同。

Chapter 14 的离线 `77 / 77` 覆盖 BF16 位保真加载、混合精度算子语义、路径确定性与 cached-decode 逐位等价、BF16 资产/parity contract。opt-in：0.6B/1.7B/4B BF16 integration 同进程 `Pkg.test` 中 Chapter 14 专项 `205 / 205` 通过；8B BF16 integration 用独立进程 + `--heap-size-hint=2G` 跑 `125 / 125` 通过（峰值 RSS 18.53 GiB）。四尺寸 16 步 greedy token 与 HF BF16 全部一致，logits max-abs ≤ 0.72（mean ≤ 0.073），blocks 尺度感知误差 ≤ 1.49e-2，embedding 位精确。

Chapter 13 的离线 `283 / 283` 覆盖流式 reader 严格性（44 项）、流式 vs in-memory 逐位一致（23 项）与 8B/14B/32B 资产/parity contract；带 `LIFEAI_QWEN3_8B/14B/32B_MODEL_DIR` 的 opt-in 全套 `689 / 689` 通过（integration 406 项，重算逐层流式 parity）。8B/14B/32B 流式峰值 RSS 分别 6.94 / 8.84 / 8.87 GiB（对照 Float32 全量 30.51 / 55.02 / 122.06 GiB），logits max-abs ≤ `3.05e-5`，decode ≤ `4.01e-5`，argmax 全一致。注意 opt-in 协议：tied 尺寸与 streamed 尺寸的 integration 必须分两个 `Pkg.test` 进程运行（同进程堆累积会 OOM）。block 级中间激活用尺度感知容差（scaled ≤ 1e-5，实测最差 6.9e-7），归一化输出用绝对容差。

Chapter 11 的 `91 / 91` 包含：六个 frozen config 的 revision/checksum、自动识别、错配/RoPE 语义漂移/MoE 拒绝，六套完整 depth/width topology 和精确参数量共 80 项；untied LM head、`Q width > hidden`、GQA dynamic/static cache 的缩小 32B 形态共 11 项。

Chapter 12 的离线 `85 / 85` 冻结 1.7B/4B 全部资产 checksum、分片清单与实测 parity 结果；带 `LIFEAI_QWEN3_1_7B_MODEL_DIR` / `LIFEAI_QWEN3_4B_MODEL_DIR` 的 opt-in integration 共 `209 / 209` 通过（含 1.7B 57 项、4B 67 项），重算文件尺寸、variant 识别、精确参数量与全部逐层/decode 断言。1.7B 的 final hidden / logits / decode max-abs 为 `1.72e-4 / 9.35e-5 / 3.05e-5`，4B 为 `1.07e-4 / 3.53e-5 / 2.00e-5`，argmax 全一致；4B 真实 3 分片 index 加载 32.0 s，峰值内存在 30 GiB 内。

使用 `Qwen/Qwen3-0.6B` revision `c1899de289a04d12100db370d81485cdf75e47ca` 的真实 BF16 权重和 Transformers Float32 reference，opt-in integration 35 / 35 通过。final hidden max-abs 为 `7.43866e-5`，full logits 为 `5.67436e-5`，dynamic/static decode logits 均为 `4.48227e-5`，下一 token argmax 全部一致。详细版本、容差、逐层误差、checksum 与内存记录见 Chapter 07 文档。

同一 revision 的 Chapter 08 真实 tokenizer/text integration 70 / 70 通过：6 组多语种/Unicode/代码/special-token corpus 和 4 组基础 chat 的 strings/spans/ids 完全一致；raw 与 chat greedy 共 6 step 的 token ids、停止位置和文本完全一致，global logits max-abs `5.054474e-5`。chat prompt 的 full/dynamic/static 输出均为 `"hello"` 并在相同 EOS 停止。

Chapter 09 的官方 sampling reference 使用 Transformers 4.51.0、16 个固定 uniforms 和同一 Float32 compute reference：sampled integration `86 / 86` 通过，raw/filtered/probability global max-abs 分别为 `6.67572e-5`、`3.05176e-5`、`5.90086e-6`。同版本的独立 RoPE fixture 覆盖 position 0/2048/32767/40959，默认专项 `30 / 30` 通过。

Chapter 10 GPT-2 124M opt-in integration `82 / 82` 通过：tokenizer artifact/checkpoint round-trip 与 10 组 corpus 的 strings/spans/ids/bytes 完全一致；embedding max-abs 为 0，12 blocks 全局最大 `4.8828125e-4`，final hidden `7.05719e-5`，full logits `1.0681152e-4`；full/dynamic/static 的 8-step greedy ids/text 完全一致，step logits global max-abs `1.2207031e-4`。

Qwen3-0.6B CPU benchmark 在 Intel Core Ultra 7 270K Plus 上完成；16/64/256-token prompt 的 dynamic cache decode 分别为 14.28/14.77/10.95 tok/s，256-token 时相对 full recompute 加速 10.33×。RTX 5080 CUDA 的相同三组 dynamic 为 86.06/84.60/67.30 tok/s，static 为 81.55/82.17/81.99 tok/s，三组 correctness 全通过。Reactant-XLA GPU 的 16+2 静态 cache steady decode 为 137.89 tok/s，prefill/decode max-abs `0.01609 / 0.01151`，在 `atol=2e-2, rtol=5e-3` 下通过且 argmax 全一致；cold compile 和 11.59 GiB BFC allocator 成本单独记录。完整 raw samples 见 `benchmark_results/week09/`。

GPT-2 124M 的 16/64/256-token CPU dynamic decode 为 58.71/55.76/33.86 tok/s，CUDA dynamic 为 352.13/339.24/269.92 tok/s；对应 CUDA static 为 329.42/339.46/321.67 tok/s。所有 full/dynamic/static correctness 均通过，完整 warm-up、steady samples、RSS/cache bytes 与同步口径见 `benchmark_results/week10/`。

Chapter 05 三 seed（20260720–22）跨 tokenizer 对照记录于 `benchmark_results/week05/`：character / byte / byte_bpe 的 tokens per byte 为 0.3717 / 1.0000 / 0.7139，final BPB 3.0753 / 8.1890 / 6.7614；byte 与 byte-BPE 对 unseen UTF-8 lossless 且 validation unknown 率为 0，character 为 19.6%（其 BPB 不可与 lossless tokenizer 直接排名）。

Chapter 06 GQA benchmark（CPU）记录于 `benchmark_results/week06/`：固定形状下 KV cache 内存严格按 `num_kv_heads / num_heads` 缩减（8 / 4 / 1 heads 对应 1024 / 512 / 128 KiB），dynamic decode 吞吐 2097 / 2484 / 3131 tok/s，全部配置 correctness 为 true。默认测试、XLA 专项测试和硬件 benchmark 仍是三类不同证据。

## 当前边界

以下能力尚未实现，不应从现有 GPT demo 或已完成的结构 parity 推断为已经具备：

- GPT-2 的 WebText 从零训练、论文 zero-shot quality、其他尺寸和非 causal-LM heads；Chapter 10 只完成 124M 官方 checkpoint 的 Float32 推理/架构复现。
- 通用 Jinja chat template engine 仍未实现；Chapter 36 已按冻结官方模板逐分支完成 Qwen3
  tools/tool-role、JSON schema 工具注入与 agent loop，但该实现只接受已知模板 SHA256。
- BF16/量化训练、FP8、完整 GPTQ/AWQ/Hessian/block reconstruction、
  KV cache/激活量化与完整生产级 MoE sparse accelerator；32B GPU 驻留
  （INT4 约 16.4 GiB）仍出界。Chapter 24–35、41 已完成 CPU Float32 correctness、
  tiny/官方 30B Transformers parity、Float32/native BF16 active-expert streaming、
  Reactant/XLA CPU compact fallback、RTX 4090 D CUDA indexed/bucketed/grouped
  kernels、40K-capacity resident/offload session、device cache 与 scalar/grouped
  scattered paths、bounded pointer/workspace reuse、raw/final host buffers 与
  current-layer parallel miss reads；但 full-window 40K prefill/长生成、
  layer-ahead prefetch、grouped workspace byte cap 和 concurrent-session dispatch
  尚未完成。严格相邻 shard coalescing 已实现，但因破坏
  I/O/decode overlap 而不作为默认；多维 decode 的冗余同 dtype copy 已移除；
  pinned async 已可用，但本机没有胜过 pageable；cold/revisit worker sweep 已完成到 8 workers。
  Chapter 18 只实现 diagonal activation second-moment 加权，并且与 Chapter 17
  weight-MSE 一样不保证 greedy fidelity；量化推理仍每 token 全量反量化。
  XLA 日常部署目前为 Qwen3-8B、batch 1、greedy、4K 总窗口；device-side
  sampling、通用动态 batch、40K 长窗和 fused/FlashAttention 未完成。
  resident service 只提供 loopback 单模型 strict subset，不包含 TLS、
  authentication、多租户或动态 batching；server 退出后仍需重新加载编译。
  Reactant kernel/autotune cache 不是完整 executable cache。
- Qwen3 128K YaRN / RoPE scaling；六个冻结 checkpoint 的原生 `max_position_embeddings` 均为 40,960，非空 `rope_scaling` 仍 fail closed。
- 长期运行的模型质量与较大规模真实语料训练；Chapter 37 已建立 MMLU/GSM8K 基线，
  但只覆盖 0.6B/1.7B/4B、8 个 subject、0-shot、greedy 单次，没有 5-shot、多 seed
  方差，也没有 8B 及以上。
- 适合 tied embedding 的统一初始化基线、低精度专项与真实规模组件对照。
- 实验注册、超参数搜索、分布式训练和面向生产的性能评估。
- 更大、更自然的跨请求记忆任务、失败/部分经验 schema、ANN/reranker、通用 redaction/摘要/遗忘、
  并发 writer 与完整 agent memory policy；Chapter 39 的 source journal 和 Chapter 42 的显式 clean
  environment-event 写回/fresh-load exact-spec 检索已完成，但仍是 single writer。同 run ID 幂等，
  不同 run ID 仍可重复记录同一事实。
- 任务规划与反思，以及 tools 版 `/api/generate`；工具调用协议已在 Chapter 36
  完成 HF 逐字节 parity 与单请求内闭环，任务级质量对照已由 Chapter 38 完成，
  但当前仍是按冻结官方模板手写的 renderer，不是通用 Jinja 引擎。
- 图像、音频、非符号空间状态或机器人传感器输入；Chapter 40/42 observation 只有结构化 GridWorld 坐标。
- 连续动作、动力学、外部 simulator 和真实设备适配器；Chapter 40/42 只有确定性离散 GridWorld、
  enum action、硬预算与 clean-success memory policy。生产 retrieval 强制 exact-spec，cross-spec 只用于
  diagnostic distractor。
- 机器人运行所需的实时性、容错和物理安全机制；timeout、execution failure、e-stop 与 idempotent
  action safety 尚未形成跨 adapter 契约。
- 在线学习、持续学习与个体长期成长。

## 建议的近期里程碑

### Milestone A：建立可恢复、可评估、可比较的实验基线（已完成）

完成记录：Chapter 03 已于 2026-07-18 Closed；默认测试 654 / 654 通过，四后端基线均完成 correctness 与性能记录。

### Milestone B：推进模型组件、Tokenizer 与中文训练（已完成主体）

- RMSNorm、SwiGLU、embedding / lm_head 权重共享独立开关与对照实验。（Chapter 04 已完成，2026-07-19 Closed）
- 无 OOV、完全可逆的 byte-level baseline，deterministic byte-BPE、版本化 Tokenizer artifact 与 fingerprint。（Chapter 05 已完成，2026-07-21 Closed）
- 来源、许可、checksum、文档级切分可追踪的中文语料训练流程与 bits-per-byte 评估口径。（Chapter 05 已完成）

### Milestone B'：复现 Qwen3 并以 HF 权重验证（已完成）

- 实现 GQA 与 QK-Norm，使模型结构与 Qwen3 dense 同构；复用 KV Cache correctness / benchmark 验证 cache 布局与 decode 收益。（Chapter 06 已完成，2026-07-22 Closed）
- 解决 RoPE rotate_half 适配，实现 safetensors / bfloat16 权重加载、HF `config.json` 解析与参数名映射；用 token-id fixture 对齐 Qwen3-0.6B 逐层 hidden states、logits 与 KV Cache decode。（Chapter 07 已完成，2026-07-22 Closed）
- 导入 HF `tokenizer.json`（byte-level BPE、byte↔unicode 映射、regex pre-tokenization、special tokens），完成基础 chat template 与 text→text 端到端一致性验证。（Chapter 08 已完成，2026-07-22 Closed）

完成标准已满足：LifeAI.jl 能从本地加载 Qwen3-0.6B 官方权重和同 revision tokenizer，在明确 Float32 容差下与 HF logits 对齐，并以 full / dynamic / static KV Cache 产生完全相同的 greedy token 序列和文本。

### Milestone B''：深化 Qwen3 真实生成与框架性能（已完成）

- 复现官方 temperature/top-k/top-p sampling，比较候选分布并用固定 uniform 流跨框架重放。（Chapter 09 已完成）
- 验证长位置 RoPE 边界和多 prompt length 的 full/dynamic/static cache correctness。（Chapter 09 已完成）
- 建立 Qwen3-0.6B CPU、CUDA/XLA 的 load/prefill/decode/RSS/cache benchmark。（Chapter 09 已完成）

完成标准已满足：官方 sampling 的候选 ids、filtered logits/probabilities 和固定 uniform token 与 Transformers 对齐；真实模型性能结论有原始实验条件和可重复命令，且没有把 BF16 storage 误写为 native BF16 compute。

### Milestone B'''：验证第二种经典 decoder 架构（已完成）

- 以 GPT-2 124M 验证 learned absolute position、LayerNorm + GELU-New、带 bias MHA/MLP 与 HF Conv1D/fused-QKV 权重布局。（Chapter 10 已完成）
- 严格导入 GPT-2 byte-level BPE，并完成 tokenizer→逐层 logits→KV cache→greedy text parity。（Chapter 10 已完成）
- 复用 CPU/CUDA/XLA 验证体系，区分官方 checkpoint 推理复现与未执行的 WebText 从零训练/论文质量复现。（Chapter 10 已完成）

完成标准已满足：checkpoint revision/reference 环境已冻结；GPT-2 124M tokenizer、逐层 hidden/logits、full/dynamic/static generation 与 HF 对齐；默认回归、CUDA 和缩小 XLA smoke 均通过。

### Milestone B''''：补齐 Qwen3 dense family contract（已完成）

- 冻结六个官方 dense 尺寸的 revision、config checksum、width/depth、tied head 与精确参数量。（Chapter 11）
- 显式识别官方 variant，同时保留兼容 custom dense config；错配 variant fail closed。（Chapter 11）
- 用完整 topology 和缩小的 untied + 宽 attention 模型覆盖 0.6B 单尺寸真实验证没有触达的结构分支。（Chapter 11）

完成标准：六个官方 config/topology/参数量进入默认离线测试，tied/untied 与
`Q width > hidden` 的参数映射和 cache 路径均通过；文档不得把 1.7B—32B
结构覆盖写成真实大权重 parity。

完成标准已满足：六个官方尺寸均有 immutable config fixture、完整 topology
和精确参数量；untied + 宽 attention 的缩小 32B 形态通过三路 cache，默认
全套 `4284 / 4284` 通过。

### Milestone B''''': 真实权重 parity 扩展到可实跑的 family 尺寸（已完成）

- 下载 1.7B / 4B 冻结 revision 完整资产并记录全文件 checksum。（Chapter 12 已完成）
- 用同一 token-id fixture 生成 Transformers Float32 reference，验证逐层
  hidden、logits 与 dynamic/static decode 对齐；两尺寸均走真实分片
  index，1.7B 覆盖 `Q width == hidden` 真实分支。（Chapter 12 已完成）
- loader 零改动一次通过，未发现尺寸相关 bug；0.6B / GPT-2 既有 parity
  无回归。（Chapter 12 已完成）

完成标准已满足：两个尺寸在显式容差内逐层对齐且 argmax 一致（默认全套
`4369 / 4369`，opt-in `209 / 209`）；离线 checksum/parity fixture 进入
默认测试；8B+ 不可实跑的边界保持明确。

### Milestone B'''''': 流式加载完成 8B/14B/32B 真实权重验证（已完成）

- header-only safetensors 索引、按需单 tensor 读取与逐层流式
  forward/decode，与 in-memory 路径逐位一致。（Chapter 13 已完成）
- accelerate disk offload 生成 Float32 同口径 reference；8B/14B/32B
  逐层 parity 全部通过，untied LM head 获真实证据。（Chapter 13 已完成）
- 尺度感知 block 容差与分进程测试协议进入默认套件与文档。（Chapter 13
  已完成）

完成标准已满足：六尺寸真实权重逐层验证闭环完成（默认全套
`4652 / 4652`，week13 opt-in `689 / 689`）；32B 峰值 RSS 8.9 GiB 实测
记录在案。

### Milestone B''''''': native BF16 混合精度推理（已完成）

- BF16 位保真加载与参数内存减半；独立推理路径逐算子镜像 HF 混合精度
  契约。（Chapter 14 已完成）
- 0.6B/1.7B/4B/8B 与 HF BF16 逐层对齐、argmax 零失配、16 步 greedy
  token 序列完全一致。（Chapter 14 已完成）
- 8B 全量驻留 greedy 生成与内存实测；分块升精度 + 显式 GC + heap
  hint 的内存工程协议。（Chapter 14 已完成）

完成标准已满足：默认全套 `4729 / 4729`，BF16 opt-in `205 / 205` 与
8B 独立进程 `125 / 125` 全绿；容差、greedy 序列与内存实测冻结进
离线 fixture。

### Milestone B'''''''': BF16 推理落地 CUDA/XLA（已完成）

- 设备通用向量化 BF16 路径，CPU 上与循环路径逐位相同。（Chapter 15
  已完成）
- 0.6B/1.7B/4B CUDA parity/greedy 全对，吞吐 8—15 tok/s（CPU 的
  33—92 倍）。（Chapter 15 已完成）
- Reactant XLA BF16 编译 prefill 通过（steady 1.36 ms）。（Chapter 15
  已完成）

完成标准已满足：默认全套 `4792 / 4792`、CUDA opt-in `173 / 173`
全绿；吞吐/VRAM/编译耗时冻结进离线 fixture；推理主战场移至
CUDA/XLA。

### Milestone B''''''''': XLA 编译生成与量化驻留（已完成）

- XLA BF16 static-cache decode 编译，设备端 greedy 闭环 246 tok/s
  （eager 16.1×），greedy 与 HF 全对。（Chapter 16 已完成）
- RTN INT8/INT4 量化：8B 与 14B 首次 GPU 驻留；8B token 行为近乎
  无损，14B INT4 漂移精确量化记录。（Chapter 16 已完成）

完成标准已满足：XLA 吞吐目标 ≥10× 达成；量化离线语义测试、GPU 驻留
实测与行为记录冻结进 fixture；默认全套无回归。

### Milestone B'''''''''': 校准实验与预算化混合精度（已完成）

- reconstruction-MSE INT4 scale search 保证每个 row/group 在冻结
  candidates 中不劣于 max-abs baseline，Chapter 16 RTN 默认保持位兼容。
- 一份 `QuantizationPlan` 同时驱动 streamed/in-memory 路径，可按层、
  投影与 LM head 选择 INT4/INT8/BF16；真实树统计与 topology 估算逐
  byte 相同。
- RTX 4090 D 上用冻结 Qwen3-14B/BF16 reference 比较全 INT8、mixed
  MSE 与同布局 mixed RTN；全 INT8/mixed RTN 16/16，MSE 4/16。

完成标准已满足：默认 `4953 / 4953`、Chapter 17 `83 / 83`；全模型 checksum、
reference checksum、GPU 驻留/误差/greedy 指标进入 fixture。MSE 降低
full-logits 全局误差却损害 greedy 的负结果被保留，未包装成质量提升。

### Milestone B2：独立 token 的 diagonal activation-aware 校准（已完成）

- 冻结与评测 prompt 分离的多语种/代码/数学 token fixture、tokenizer
  revision 与 checksum。
- native BF16 流式采集每层 Q/K/V/O、gate/up/down 和 untied LM head
  输入二阶矩；CPU 数组上的 loop 与 device-generic accelerated 路径逐值一致。
- `:activation_mse` 共用 Chapter 17 packed INT4/预算与 plan；缺失或非法
  stats fail closed，streamed/in-memory 逐 tensor 相同。
- 14B/RTX 4090 D 同布局对照按负结果关闭：activation-aware 4/16，
  没有保住 mixed RTN 16/16。

完成标准已满足：实现、离线测试、校准 provenance、模型/reference
checksum、GPU 驻留/误差/greedy 指标均进入 fixture；没有把 diagonal
近似包装成完整 AWQ/GPTQ 或质量提升。

### Milestone B3：Qwen3-8B XLA single-residency 日常部署（已完成）

- 逐层 streamed BF16 loader 直接产生 compact packed 参数树，应用侧只
  保留一棵 device 参数树、一次递归 transfer。（Chapter 20 已完成）
- 固定形状 64-token prefill 和单 token decode 共用 4K 静态 KV；分块
  cache 写入从 64 次 DUS 合并为每层 K/V 各一次。（Chapter 20 已完成）
- RTX 4090 D 完成 3,584+512 整窗、8 次复用、96-token CUDA parity 与
  200 ms 连续显存验收。（Chapter 20 已完成）

完成标准已满足：prefill 1.498 s、decode 41.13 tok/s，最低物理 free
2.247 GiB；benchmark 只有全部门槛通过才以零状态退出。

### Milestone B4：Qwen3-8B XLA 常驻服务（已完成）

- 唯一 compiled session 接入 loopback HTTP，loader 只运行一次，静态 KV
  由 single-flight lock 保护。（Chapter 21 已完成）
- strict `/api/generate` 支持 buffered/NDJSON、UTF-8 安全 token streaming、
  1 MiB body 和 4K context/options fail-closed 门禁。（Chapter 21 已完成）
- 4090 D 完成 15 个跨连接请求、96-token CUDA parity、并发串行化、
  3,584+512 整窗和 200 ms 显存验收。（Chapter 21 已完成）

完成标准已满足：load_count 1，15/15 请求成功，整窗 41.35 tok/s，
最低物理 free 2.251 GiB；常驻摊销冷启动，但不宣称跨进程 executable
cache 已实现。

### Milestone C：建立最小有状态智能体与环境闭环（主体已完成）

- Qwen3-Embedding-0.6B 与内存内 exact semantic memory baseline 已完成，
  但尚未接入跨 step 状态或 policy。（Chapter 22 已完成）
- 工具协议与单请求内多 step 闭环已完成：官方 chat template 全分支逐字节
  parity、沙箱化工具注册、`<tool_call>` 解析与合法性判定，以及不加载模型
  即可复算每轮 prompt sha256 的 replay 测试。（Chapter 36 已完成）
- 跨请求 source journal、exact retrieval policy、memory-aware trace 与三臂任务集
  已实现；相关/干扰 prompt 在真实 Qwen3-4B tokenizer 下 12 / 12 等 token。
  冻结 Qwen3-Embedding/Qwen3-4B 的真实 retrieval 为 `12/12`，三臂为
  `0/12 / 12/12 / 0/12`，36 / 36 轨迹跨进程 replay。（Chapter 39 已完成）
- 与具体机器人无关的 observation/action/transition/environment 接口、hidden-wall GridWorld、BFS oracle、
  环境终态评分和联合 replay 已完成。（Chapter 40 已完成）
- clean successful episode 的显式 policy writeback、authoritative-spec/source binding、fresh-load CPU exact
  retrieval 与无反馈三臂已完成；真实 Qwen3-8B 为 `0/8 / 7/8 / 0/8`，32 条轨迹 tokenizer-only replay。
  （Chapter 42 已完成）
- 保持模型后端可替换，使当前小 GPT、Qwen3 复现权重或后续多模态模型都能接入。

工具、记忆和环境三个主体部件已经具备：Chapter 39 提供独立 `AgentMemory` source journal；
Chapter 40 提供与模型无关的 observation/action/transition/environment 契约和 BFS policy oracle，
Qwen3 adapter 在确定性 simulator 中根据新 observation 改变动作；Chapter 42 把 clean successful
episode 经显式 policy 写入同一 journal，fresh-load 后再以 exact-spec context 影响后续无反馈决策。
完整 writer/reader prompt、工具、环境和 writeback evidence 均可由第二进程 replay。“感知 → 记忆 →
决策”的最小确定性链已经闭合，但真实传感器、动力学，以及 timeout/execution failure/e-stop/idempotent
action 等跨 adapter 安全仍不在当前结论内。

## 长期能力地图

| 主线 | 当前状态 | 下一关键缺口 |
| --- | --- | --- |
| 模型基本组件 | Qwen3 六尺寸真实权重 parity 全闭环 + native BF16 推理 + CUDA/XLA 加速 + 8B XLA single-residency 4K greedy 部署/常驻服务 + 可预算 INT4/INT8/BF16 计划与 diagonal activation-aware 校准（14B RTN 16/16，weight/activation MSE 均 4/16）；30B-A3B MoE grouped/scalar scattered cache；GPT-2 真实 parity；流式加载；五类版本化 Tokenizer 与中文数据管线 | 模型线恢复时进入 Qwen3-VL；另有完整 AWQ/GPTQ 或量化 GEMM |
| 高效训练与推理 | modern / GQA / rotate_half 已兼容 Zygote / XLA 与两类 KV Cache；Qwen3-0.6B compiled decode、Qwen3-8B 4K XLA single-residency/service 与 30B-A3B BF16 offload/cache/grouped-scattered + bounded reuse/storage-verified parallel miss 已在 GPU 实证；adjacent coalescing 负结果与 decode copy-elision 正结果已冻结 | route/attention 临时数组复用、grouped workspace byte cap、fused/FlashAttention、动态 batch、低精度 kernel 与长上下文专项 |
| 智能体核心 | 官方 chat template 全分支 HF 逐字节 parity + 沙箱化工具与单请求多 step + 工具任务成功率配对测量 + append-only source journal/exact retrieval + clean environment-event 显式写回/exact-spec context + observation/action/transition 联合 replay | ANN/reranker、通用 redaction/forgetting、并发 writer、planning 与反思 |
| 多模态感知 | 尚未开始 | vision / audio / sensor representation |
| 具身闭环 | 确定性 hidden-wall GridWorld、allowlisted enum action、硬预算、环境终态评分；Qwen3-4B full-feedback 8/8 最短路；Qwen3-8B clean writer 8/8、无反馈 no/retrieved/distractor 0/8 / 7/8 / 0/8 | 跨 adapter timeout/execution failure/e-stop/idempotent action safety；随后才是连续状态/动作、外部 simulator 与 device adapter |
| 持续学习与生命感 | 处于愿景阶段 | 长期状态、适应、主动性与安全边界 |
| 学习记录 | Chapter 01—42 已 Closed；Episode 06/07 Closed，Episode 08 Open | Episode 08 继续跨 adapter action safety；保持 oracle/因果对照/replay 门禁 |
