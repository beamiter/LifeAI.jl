# Chapter 42 — 显式环境事件记忆写回与因果验收

> 所属 Episode：Episode 08 — 环境与行动闭环
>
> 状态：Closed
>
> 日期：2026-08-21

## 本章问题

Chapter 39 已有跨请求 append-only memory journal，Chapter 40 已有由环境终态判分的 action loop，
但两者仍是两条分开的链。本章回答：**一次真实、成功且没有非法动作的环境 episode，能否在显式
policy 准入后写成可验证的 memory event；fresh load 后真实检索到的事件，能否在没有中间反馈的后续
episode 中提高任务成功率；整条 writer → journal → retrieval → reader 链能否不加载模型地重放？**

## 范围

- 复用 Chapter 40 冻结的 8 个 hidden-wall GridWorld task；不修改地图或 BFS oracle。
- writer 使用真实 Qwen3 generation session 与 full feedback，只有 clean successful episode 才能写回。
- 写回是 episode 结束后的**显式 policy 调用**，不是每个 transition 的隐式副作用。
- journal 严格 fresh-load 后，用 CPU Qwen3-Embedding-0.6B 建 exact index；同时验收通用 raw top-1 和
  生产路径的 exact-spec context。
- reader 使用 no-memory / retrieved-memory / token-matched mirrored distractor 三臂；三臂均
  `feedback=:none`、相同 system prompt、non-thinking greedy，并按 task 交错运行。
- tokenizer-only replay 重建 writer/reader prompt、环境轨迹、memory context、writer event bytes 与
  source binding；不加载 generation/embedding 权重，也不追加 journal。
- 不做并发 writer、ANN、reranker、通用 redaction/forgetting、连续控制或真实设备安全。

## 关键设计

### 1. 写回由环境证据和显式 policy 共同决定

`verified_successful_episode_v1` 只接纳 full-feedback、终态成功、每个工具调用和 transition 都可重放，
且 blocked/非法动作数为 0 的 episode。模型自述成功不算证据；即使最终到达 goal，只要曾撞墙，也会
fail closed。事件派生必须同时接收 authoritative `GridWorldSpec`，再核对 spec、初始/最终 observation、
动作链、source trace SHA256 与 transition-chain SHA256，不能只相信 trace 自带的可伪造字段。

每个 writer 使用稳定 `run_id=chapter42/write/grid-NN`。同一个 `run_id` 重试相同事件时幂等返回
existing，不再追加；相同 ID 但 bytes 不一致会拒绝。不过不同 `run_id` 仍可为同一事实生成重复事件，
当前还没有内容级去重策略。

### 2. writer 与 reader 之间有真实持久化边界

8 个 writer 逐题运行并显式 append 后，runner 丢弃 writer store，再从 JSONL fresh-load journal，严格
校验 sequence、record checksum、environment-event schema 和完整 store fingerprint。reader 不复用
进程内临时 `AgentMemoryContext`；相关 context 由新加载的 journal 和真实 embedding query 生成。

通用 `retrieve_agent_memory(...; top_k=1)` 用于报告没有预先指定 ID 的 raw top-1；生产注入再通过
`retrieve_gridworld_memory_context` 做 exact-spec 过滤与验证。跨 spec context 默认拒绝。只有
matched-distractor 诊断臂显式 opt-in cross-spec，并选择镜像 task 的 canonical event；它不是生产
retrieval policy。

### 3. 因果主对照是相关记忆与等 token 非求解干扰

只比较 no-memory 与 retrieved 会混入 prompt 变长效应。每个相关 context 都配一个镜像 task 的
distractor context，并用真实 Qwen3-8B tokenizer 验证完整首 prompt token 数相同；同时把 distractor
route 机械执行到当前 spec，确认它不能解题。这样 retrieved 与 distractor 的差别是内容相关性，而不是
token 数或一条碰巧也能到达 goal 的替代路线。

三个 reader 臂都看不到中间位置、legal actions 或 blocked 原因，只收到与 Chapter 40 相同的
`feedback=:none`。它们共用 `gridworld_memory_system_prompt`；区别只有是否注入相关、无关或不注入
memory context。

## 已实现

- `AgentEnvironmentMemoryPolicy` / `AgentEnvironmentMemoryEvent` /
  `AgentEnvironmentMemoryWriteback`，以及 clean successful GridWorld 准入 policy。
- `agent_environment_memory_events(trace, spec; ...)` 与
  `append_agent_environment_events!(store, trace, spec; ...)`：authoritative-spec 验证、稳定 ID、显式 append
  和同 run ID 幂等。
- environment-event canonical JSON/text/metadata、source-trace/transition-chain binding、strict record
  validation 和 fresh-load fingerprint。
- `retrieve_gridworld_memory_context` / `select_gridworld_memory_context`：生产 exact-spec 边界；通用
  selector 只供显式 cross-spec diagnostic control。
- `run_qwen3_environment_memory_eval.jl`：8 writers、fresh load、CPU embedding、三 reader arms、逐题
  交错、token/non-solving 门禁，以及 trace/items/summary/run manifest。
- `replay_qwen3_environment_memory_eval.jl`：只读 tasks、journal、trace 与 tokenizer，从 record ID 重建
  context，并重新派生 writer events，不调用 append。

## 默认验证

Chapter 42 离线专项为 `118 / 118`（`38 + 32 + 41 + 7`）；Chapter 40 回归为 `126 / 126`，
全量 `Pkg.test` 为 `8,346 / 8,346`。测试覆盖：

- success/terminal/full-feedback/clean-action 的准入矩阵，尤其是“最终成功但曾 blocked”必须拒绝；
- trace 与 authoritative spec 不一致、transition/source hash 漂移、非 canonical record 均 fail closed；
- 同 `run_id` 幂等、同 ID 内容冲突拒绝，以及不同 `run_id` 可产生独立记录的当前边界；
- fresh-load、exact-spec filter、stale/cross-spec context 拒绝与显式 diagnostic opt-in；
- 8 个 task 的 relevant/distractor token match、mirrored route non-solving 和 tokenizer-only replay。

## 真实 Qwen3-8B 结果

2026-08-21 在 NVIDIA GeForce RTX 4090 D 上完成正式验收。generation checkpoint 从 ModelScope 本地源
加载 Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`，CUDA native BF16、4,096-token
context、每 turn 96 个新 token、non-thinking greedy。retrieval 使用 ModelScope
Qwen3-Embedding-0.6B revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`，在 CPU 以 1,024 维、
256-token query 建 exact index。

8 个 writer 全部 clean success，合计 `44` 个动作、`0` 个非法动作，与各题 BFS shortest path 完全
一致。8 次显式写回均 admitted/appended；fresh-load journal 恰有 8 条 environment event，文件与 store
fingerprint 都是
`9a6d550e493bb1a0ef09ec06f929355a1618dd9d4fbae723efde5640c84c70e1`。

raw semantic top-1 和生产 exact-spec context 均为 `8/8` 正确。相关/干扰首 prompt 为 `8/8` 等 token，
8 条 mirrored distractor route 在目标 spec 中均不能解题。

同一 CPU embedding checkpoint 和 journal 的复核重建仍保持 raw/exact-spec top-1 ID `8/8`，但
Float32 cosine score 有约 `7e-5`—`2.7e-3` 漂移。score 不进入 context render 或 prompt SHA256，
因此不改变这次冻结 reader 输入；本章只主张 top-1 ID 与 exact-spec selection，不主张 score bit-exact。
当前 runner 也没有记录 Julia、BLAS 与 thread runtime provenance。

| 实验臂 | 成功 | 动作 | 非法/失败动作 | 成功 episode 相对 BFS 多余动作 |
| --- | ---: | ---: | ---: | ---: |
| full-feedback writer | `8/8` | `44` | `0` | `0` |
| no-memory reader | `0/8` | `88` | `70` | 不适用 |
| retrieved-memory reader | `7/8` | `52` | `13` | `0` |
| token-matched distractor reader | `0/8` | `88` | `59` | 不适用 |

retrieved 相对 no-memory 是 `7` 个 variant-only、`0` 个 baseline-only、`1` 个 neither；相对
matched distractor 也是同一张配对表。两组成功率差均为 `+0.875`，双侧精确 McNemar
p=`0.015625`。no-memory 与 distractor 没有不一致对，p=`1`。

唯一 end-to-end 失败是 `grid/08` 的 retrieved reader：它的 raw top-1 与 exact-spec context 都命中正确
事件，但模型在无中间反馈执行阶段失败。因此 retrieval 证据是 `8/8`，任务成功率只能报告 `7/8`，
不能把后者改写成 retrieval miss，也不能外推到开放域或不同环境。

## 独立 replay

第二进程只加载冻结 tasks、8 条 journal、32 条 trace 和 Qwen3-8B tokenizer。它重放 `8` 条 writer 与
`24` 条 reader episode，共核对：

- `32 / 32` rows 和 final states；
- `304 / 304` prompt SHA256、prompt token counts 与 recorded generated IDs；
- `273 / 273` tool outcomes、`272 / 272` transition hashes；
- `16 / 16` memory contexts、`8 / 8` token-matched pairs；
- `8 / 8` writer event bytes、source bindings 与逐次 writeback-prefix fingerprints。

replay 明确记录 `generation_loaded=false`、`embedding_loaded=false`、`journal_appended=false`。它重新执行
环境 transition，并从 journal ID 重建 reader context；writer event 也从 trace + authoritative spec
重新派生后逐 byte 对照，但不会再次写入 store。

## pilot 失败与协议冻结

正式结果之前的 `pilot_attempt001` 必须与指标分开。pilot 的 `grid/01`—`grid/04` 产生了 4 条合法、
可严格恢复的 journal prefix，SHA256 为
`9b215e853c208214ca04f19615f25cfa22c0ecee8961d5dea82cfd504c90f870`；`grid/05` 的环境终态虽为
success，但 episode 曾有 blocked action，因此 `verified_successful_episode_v1` 正确以
`policy_rejects_blocked_actions` 拒绝。pilot runner 当时只在所有 writers 通过后才落完整 wrapper trace，
所以该题的逐动作 trace 不存在，文档不重构或猜测它。

观察 pilot 后，writer prompt 才增加 task-agnostic 的 clean-admission 说明：`legal_actions` 是权威集合，
每次只复制一个合法方向，禁止 probe/batch/guess/retry。该 prompt 在正式空 journal 八题重跑前冻结，
因此正式结果不是预注册、未观察数据上的确认性实验，泛化证据有限。最终 run manifest 明确 supersedes
pilot failure manifest；后者 SHA256 为
`68f4f6716e9e5a4f212d4c82154cfe272b8842d1906116e14f6f1672c34b203f`。pilot 的 4 条 prefix 没有复用，
也没有混入任何正式 retrieval、reader 或配对指标。

## 产物与复现

本地 `benchmark_results/chapter42_qwen3_environment_memory/` 的冻结 SHA256：

- summary：`c4e5846e0b7e57830f82f6dcceea73641c7395024eac8fa480ba702a5945d875`
- trace JSONL：`46b26f92e4d431a83b9244013d0c4e8fab98ff51ea0b74f2669cada52ef7029f`
- item JSONL：`b99c48462fe4bdc880f62db26ea61fb5f770e1dbeb64a7d4037cbe7b18277969`
- journal JSONL：`9a6d550e493bb1a0ef09ec06f929355a1618dd9d4fbae723efde5640c84c70e1`
- replay JSON：`ebacf9da8b5a7b890acd43ce2ef74668002193180f7c6ace03856e31023fed26`
- run start / complete manifest：
  `9b8d6d03e3b4143a7819685a0bdb9ac3f0901a7ffc55f225f9c52d0bed231fd7` /
  `37aa715926bea17aa11aaf82c14fd14579bf0efea08f16a2f34724b4b77afdef`

另保留 8 份 task-scoped writer-attempt JSON，以及 pilot 的 4-record prefix 和 failure manifest，便于从
正式 summary 追到每次准入和 supersession 证据。

正式运行：

```bash
julia --project=. --startup-file=no scripts/run_qwen3_environment_memory_eval.jl \
  /path/to/Qwen3-8B/<revision> \
  /path/to/Qwen3-Embedding-0.6B/<revision> \
  --out benchmark_results/chapter42_qwen3_environment_memory \
  --label qwen3-8b \
  --variant qwen3_8b \
  --revision <qwen3-8b-revision> \
  --supersedes benchmark_results/chapter42_qwen3_environment_memory_pilot_attempt001/failure.json
```

tokenizer-only replay：

```bash
julia --project=. --startup-file=no scripts/replay_qwen3_environment_memory_eval.jl \
  test/episodes/episode08_environment_action_loop/chapter40_deterministic_gridworld/fixtures/gridworld_tasks.json \
  benchmark_results/chapter42_qwen3_environment_memory/environment-memory.jsonl \
  benchmark_results/chapter42_qwen3_environment_memory/qwen3-8b_environment_memory_trace.jsonl \
  /path/to/Qwen3-8B/<revision> \
  --out benchmark_results/chapter42_qwen3_environment_memory/replay.json
```

上述小于 1 MiB 的精选可审计证据随仓库进入版本控制；checkpoint、模型缓存和未列入 allowlist 的临时
benchmark 产物仍保持忽略。

## 能力边界

- 只有一个 writer，journal 仍需外部协调；同 run ID 幂等不等于跨不同 run 的内容去重。
- 准入只覆盖 clean、成功、可完整重放的确定性 GridWorld episode；失败经验、部分轨迹和非 GridWorld
  event 还没有通用 schema/policy。
- 生产 context 强制 exact-spec；cross-spec 只存在于明确标记的 diagnostic distractor 臂。
- 没有通用 redaction、forgetting、摘要、ANN/reranker 或 prompt-injection 防护。
- CPU embedding score 不是 bit-exact 复现，且 artifact 缺少 Julia/BLAS/thread runtime provenance；当前
  只验证 ID 稳定和 score 不进入渲染后的 context/prompt。
- 样本只有 8 个 task、一个 writer generation、一次 greedy run；`grid/08` 也证明正确 context 不保证模型
  稳定执行。

## Close 条件

- [x] 环境事件只有经显式、可审计 policy 才能写入 Chapter 39 journal；blocked-success fail closed。
- [x] stable run ID、authoritative spec/source binding、同 run ID 幂等与 fresh-load 完整性均有回归测试。
- [x] 真实 Qwen3-8B 写出 8 条 clean BFS-shortest event，并由 CPU Qwen3-Embedding raw top-1 / exact-spec
      `8/8` 检索。
- [x] no-memory / retrieved / token-matched non-solving distractor 三臂完成，相关记忆同时显著胜过两组
      对照。
- [x] 第二进程不加载 generation/embedding model、不追加 journal，重放全部 32 条 writer/reader 轨迹
      与 8 条写回证据。
- [x] pilot 的 policy rejection、post-pilot prompt freeze 与 supersession 独立披露，不混入正式指标。

环境事件记忆条目至此关闭，Chapter 42 于 2026-08-21 Closed。Episode 08 仍保持 Open；下一步为跨
adapter 的 timeout、execution failure、e-stop 与 idempotent action safety semantics。
