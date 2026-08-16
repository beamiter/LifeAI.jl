# Chapter 37 — Qwen3 dense 任务质量基线

> 所属 Episode：Episode 07 — 智能体闭环
>
> 状态：Closed
>
> 日期：2026-08-16
>
> 真实资产：`Qwen/Qwen3-0.6B@c1899de2…`、`Qwen3-1.7B@70d244cc…`、`Qwen3-4B@1cfa9a72…`
>
> 设备：NVIDIA GeForce RTX 5080（16 GiB），本地 NVMe，Julia 8 threads

## 本章问题

前 36 章都在问「LifeAI 算出来的数和 HuggingFace 一样吗」。本章问一个从未问过的问题：

- **Q1**：这些复现出来的模型，在冻结的公开任务集上**答对率**是多少？
- **Q2**：我们的评测管线与 HuggingFace 在**逐题**层面是否一致？
- **Q3**：同一批题目上，经典 loglikelihood 口径与真实部署的 generative 口径差多少？

## 任务集与冻结

`scripts/export_qwen3_eval_tasks.py` 只用标准库通过 HuggingFace datasets-server 的 JSON API
抓取（不引入 pyarrow/pandas/datasets）：

- **MMLU**（`cais/mmlu`，MIT，Hendrycks et al. 2021）：8 个 subject × 每 subject 前 25 题 = 200 题。
- **GSM8K**（`openai/gsm8k`，MIT，Cobbe et al. 2021）：test split 前 150 题，答案取 `####` 之后的数值。

`eval_tasks.json` sha256 `b1b9a368c1c3afd6e0b6493598d68ff958a37e3b78927514a2fc80849035fff8`；
`eval_tasks_provenance.json` 另记录抓取时间、每个 API URL、上游 `num_rows_total`、**每题的上游全局
row_idx**、许可与引用。重跑（不同 timeout/retries）得到逐字节相同的 `eval_tasks.json`。
`answer_index` 分布 `A/B/C/D = 54/48/45/53`。

## 三种协议

三种口径跑在**同一批题目**上，差异因此可归因于协议而非样本：

1. `mmlu_loglikelihood` — base-model prompt，**不套 chat template**，比较 ` A`/` B`/` C`/` D`
   四个续写的未归一化 log likelihood。四个续写在 Qwen3 词表下都是单 token（1-based
   `363/426/357/423`），故一次前向即可读出四个值；实现保留通用多 token 路径。
2. `mmlu_generative` — 走 Chapter 36 冻结的官方 chat template，greedy，从模型真正写出的文本解析字母。
3. `gsm8k_generative` — 同样走 chat template，要求最后一行给出 `#### <number>`。

抽取规则机械且写死：MMLU 取「最后一次出现 `answer` 之后的第一个独立 A/B/C/D」，无 `answer`
则取全文第一个独立字母；GSM8K 有 `####` 取其后第一个数字，否则取全文最后一个数字。解析失败记为
答错但**单独计数**；截断（`stop_reason == "length"`）与「是否以字母开头」（格式合规）也单独计数。

## 真实结果

### Q1 — 三个尺寸的答对率

MMLU 200 题，`max_new_tokens=256`，greedy，non-thinking：

| 模型 | loglikelihood | generative（全集） | generative（未截断子集） | 未解析 | 截断 | 格式合规 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | `73/200 = .365` (W `.301–.434`) | `59/200 = .295` (W `.236–.362`) | `57/175 = .326` | 14 | 25 | 150/200 |
| Qwen3-1.7B | `79/200 = .395` (W `.330–.464`) | `61/200 = .305` (W `.245–.372`) | `48/108 = .444` | 64 | 92 | 45/200 |
| Qwen3-4B | `111/200 = .555` (W `.486–.622`) | `106/200 = .530` (W `.461–.598`) | `95/158 = .601` | 23 | 42 | 158/200 |

GSM8K 150 题（仅 4B，`max_new_tokens=512`）：`141/150 = .940`（Wilson `.890–.968`），
0 未解析，2 截断，耗时 `1808 s`。

**generative 的全集数字是下界，未截断子集是有选择偏差的上界，真值在两者之间。**
证据是干净的：未解析几乎全部落在截断组（0.6B `14/14`、4B `23/23`，1.7B `62/64`），
且截断组正确率显著更低。也就是说这一列很大程度上
在测我们给的 token 预算，而不是模型。第一版用 `max_new_tokens=64` 时更糟：0.6B 有 `104/200`
被截断、accuracy 只有 `.260`；放宽到 256 后截断降到 25、accuracy 升到 `.295`，那一版数据已作废。

**1.7B 是异常点**：格式合规只有 `45/200`（0.6B 150、4B 158），`92/200` 被截断，未解析 `64/200`。
它最爱写长推导，被同一个预算伤得最重。所以「1.7B generative 只比 0.6B 高 1.0 个点」主要是格式
行为差异，不能读成知识差异——同一批题的 loglikelihood 口径下两者差 3.0 个点。

**必须同时记住的对照基线**：位置偏置存在，但**每个模型偏的方向不同**，不能一概而论：

| 模型 | loglikelihood 预测分布 A/B/C/D | 多数类平凡基线 | 实测 | 高出基线 |
| --- | --- | ---: | ---: | ---: |
| Qwen3-0.6B | `128/24/24/24`（严重偏 A） | 永远选 A `= .270` | `.365` | `9.5` 点 |
| Qwen3-1.7B | `45/79/48/28`（偏 B，A 反而低于均匀） | 永远选 A `= .270` | `.395` | `12.5` 点 |
| Qwen3-4B | `57/43/61/39`（轻微偏 C） | 永远选 A `= .270` | `.555` | `28.5` 点 |

gold 分布为 `A/B/C/D = 54/48/45/53`。HF fp32 参照对 0.6B 给出的分布是 `126/23/23/28`，与我们的
BF16 结果同样偏 A。0-shot 未归一化 loglikelihood 对 instruct checkpoint 的位置偏置是已知现象；
0.6B 的 `.365` 只比平凡基线高 9.5 个点这一点必须和数字一起说，不能把它当成判别力。

### Q2 — 与 HuggingFace 的逐题对照（本章最重要的结果）

`scripts/export_qwen3_mmlu_loglikelihood_reference.py`（torch 2.7.1+cpu / transformers 4.51.3，
float32 CPU）对 Qwen3-0.6B 跑同样 200 题，得到 `70/200 = .350`。

**先说一个必须先厘清的口径问题**：参照实现走的是 general 协议（forward `prompt+continuation`，
在 continuation 位置取值）。我们的 loglikelihood 默认走 fast 捷径（只 forward prompt，读最后一列）。
两者在精确算术下等价，在 BF16 下不等价。因此拿 fast 去和参照比，比出来的既有 dtype 差异也有我们
自己的捷径差异。**协议对齐后的对照才是 dtype 对照**：

| 对照 | 正确数 / 200 | 与 fp32 决策一致 | 翻转数 | 翻转题的 fp32 margin 最大值 |
| --- | ---: | ---: | ---: | ---: |
| BF16 general vs fp32（协议对齐） | `71` | `193 / 200 = 96.5%` | `7` | `0.1507` |
| BF16 fast vs fp32（协议不对齐） | `73` | `189 / 200 = 94.5%` | `11` | `0.3341` |
| BF16 fast vs BF16 general（同 dtype，仅协议差） | — | `194 / 200 = 97.0%` | `6` | — |

fast 相对参照多出的 4 次分歧里有 5 题与 fast/general 互相分歧的那 6 题重合，也就是说
「BF16 vs fp32 分歧」里约有一半其实来自我们自己的捷径。协议对齐后 general 的 `71` 也比 fast 的
`73` 更贴近 fp32 的 `70`。

设备之间同样不一致：同一份权重、同一 fast 协议，CUDA 给 `73`（一致 `189/200`，logprob 最大差
`0.8980`），CPU 给 `74`（一致 `192/200`，最大差 `0.7879`，且有一题在 margin `0.5349` 处翻转）。

fast 与 general 六次翻转的 general margin 全部是 `0.125` 的整数倍——与 BF16 在该量级上的表示间距
一致，即**决策边界本身被量化到了与噪声同一个格点**。

结论：**MMLU loglikelihood 的逐题决策在 BF16 下不可复现，既不跨 dtype，也不跨设备，甚至不跨
同一份实现里两条等价写法**。三个 accuracy（`70 / 71 / 73 / 74`）的极差是 4 题 = 2 个点，远小于
Wilson 区间宽度（约 ±6.5 个点），标题数字仍然可用；但任何「逐题一致」式的主张都不成立。

**一次失败的测试设计，记在这里**：本章先后写过两版 opt-in 断言，两版都是错的。第一版断言
「fp32 margin `> 0.5` 的题不允许分歧」——那是拿 CUDA 数据拟合出的常数，换到 CPU 立刻被
margin `0.5349` 的一题证伪。第二版改成「每次翻转都满足 `margin <= 2 × deviation`」，看起来更
本质，实际上是**恒真式**：设参照 argmax 为 `a`、翻转到 `c`，则 `s_c >= s_a`、`s_c <= e_c + d`、
`s_a >= e_a - d`，得 `e_a - e_c <= 2d`，而 `margin = e_a - e_(2) <= e_a - e_c`。它对任意 scores
都成立，包括把 scores 换成全 0 或随机数，因此检出能力为 0——把 continuation 的 B 与 C 对调（典型
索引写错）会造成 46 次翻转，该断言依然全绿。真正抓住它的是一致率（跌到 `154/200 = .77`）。
最终断言因此只保留两条有检出力的：`prompt_token_count` 逐题相等（prompt 漂移会先动它），
以及协议对齐下的一致率 `>= 0.9`。

### Q3 — 两种口径的差距

| 模型 | loglikelihood | generative 全集 | 差 |
| --- | ---: | ---: | ---: |
| 0.6B | `.365` | `.295` | `7.0` 点 |
| 1.7B | `.395` | `.305` | `9.0` 点 |
| 4B | `.555` | `.530` | `2.5` 点 |

差距随模型变大而缩小，与「大模型更能遵守输出格式」一致（格式合规 4B 158/200）。但 1.7B 的
`9.0` 点里有相当部分来自截断而非能力，见上。**两种口径的数字不可互相引用**，公开榜单上的
MMLU 几乎都是 loglikelihood（且多为 5-shot）口径，本章是 0-shot。

## Close 前对抗式复核抓到的问题

与 Chapter 36 一样，Close 前对整份改动做了多视角对抗复核。四条 high 全部经我复现确认，
其中两条直接推翻了本章初稿的结论：

1. **抽取规则把英文冠词当成选项 A（high）**。`extract_mmlu_letter` 的「无 `answer` 锚点时取全文
   第一个独立字母」规则无法区分选项 `A` 与冠词 `A`、集合名 `A`、`options A-D` 这类写法。
   实测：`"A factor group of a non-Abelian group is non-Abelian."` 返回 `A`。在 1.7B 的冻结轨迹上，
   有 24 题走到这条分支，其中 18 题被判成 `A`，而 gold 的 A 占比只有 27%。修复后 1.7B 的 generative
   从 `64/200` 降到 `61/200`，未解析从 `52` 升到 `64`，13 个判定翻转；0.6B、4B 与 GSM8K 完全不变。
2. **`margin <= 2 × deviation` 是恒真式（high）**。详见 Q2，测试断言检出能力为 0，初稿却把它
   写成「能抓住索引写错这类真 bug」。
3. **「预测分布严重偏 A」对 1.7B 是错的（high）**。1.7B 实际偏 B（`45/79/48/28`），A 的预测频次
   低于均匀基线。初稿只用 0.6B 的数据就推广到了「4B 之外的两个模型」。已改为逐模型给出分布。
4. **Q2 的 `189/200` 把协议差异算进了 dtype 差异（high）**。详见 Q2，协议对齐后是 `193/200`。
5. `load_eval_tasks` 接受 `"nan"`/`"inf"`/`"0x10"`（medium）——`tryparse(Float64, "nan")` 返回 `NaN`
   而非 `nothing`，这样的题会干净地加载进分母却永远不可能判对。已改为只接受十进制定点写法。
6. runner 的默认 `--label` 用 `dirname` 取到了父目录（medium）。在 revision 嵌套布局下恰好正确，
   换布局就会让多模型运行互相覆盖输出。已改为按 basename 是否像 40 位 sha 来决定。

由此新增 `scripts/rescore_qwen3_eval.jl`：per-item JSONL 保留了每条原始 completion，抽取规则变更
后可以**不重跑模型**重新打分，并打印 before/after 与每一条判定变化。上面第 1 条的重打分就是这么做的。

**本章的教训**：初稿有两个数字级结论（口径混淆、偏置方向）和一个方法级结论（恒真断言）是错的，
而默认测试全绿、与外部参照的对照也全绿。全绿只说明没有触发已写下的检查，不说明结论正确。

## 决策

- generative 口径同时报「全集」与「未截断子集」两个数，并强制报出未解析/截断/格式合规三个计数；
  任何单独引用其中一个数字都是误导。
- loglikelihood 默认走 fast 路径（快 4 倍），但把它与 general 路径的差异实测冻结进本章；
  两者在 BF16 下不等价这一点写进文档，不假装等价。
- opt-in 的 HF 逐题参照**不断言全量一致**——那是假的；也不用任何拟合出来的 margin 阈值，
  也不用恒真的不等式（两次失败的设计都记在 Q2 里）。只保留有检出力的两条：`prompt_token_count`
  逐题相等，以及协议对齐（`path=:general`）下的一致率 `>= 0.9`。
- 与外部参照对照必须**协议对齐**：`mmlu_loglikelihood_scores` 因此新增 `path=:auto|:fast|:general`，
  参照对照强制走 `:general`，日常评测仍用快 4 倍的 `:fast`。
- 本章数字一律不与 Qwen3 官方公布的 MMLU 分数并列：协议不同（0-shot vs 5-shot、base 口径 vs
  chat/CoT、8 个偏难 subject vs 全部 57 个）。它只是**我们自己管线的基线**。
- 第一版 `max_new_tokens=64` 的全部 generative 数据作废重跑，不保留在结果表里。

## 验证

- 默认全套：`7,624 / 7,624`（Episode 07 新增 Chapter 37 的 `79`）。
- Chapter 37 离线测试：冻结任务集加载与七类损坏输入的 fail-closed（含 `nan`/`inf`/`0x10`）、
  两种协议的 prompt 逐字符冻结、MMLU/GSM8K 抽取规则表（含冠词 `A`、集合 `A`、`options A-D`
  三类反例）、accuracy/Wilson/subject 报告。
- opt-in `LIFEAI_QWEN3_MODEL_DIR`：与 HF fp32 参照协议对齐（`path=:general`）的 50 题逐题对照
  （CPU，跨 8 个 subject 按 stride 取样，约 10 分钟，实测一致 `49/50`），断言 `prompt_token_count`
  逐题相等且一致率 `>= 0.9`；全量 200 题的对照数字冻结在 Q2。按前 60 题取前缀只覆盖两个最难的
  subject，实测只有 `55/60`，离门槛三题——这一点也记下来，子集取样方式会改变结论。
- 冻结产物：`eval_tasks.json`、`eval_tasks_provenance.json`、
  `mmlu_loglikelihood_reference_qwen3_0_6b.json`（sha256 `67da2278…`）。

## 遗留

- 8B/14B/32B 未测：8B BF16 在 16 GiB 上放不下，需要量化或换机器。
- GSM8K 只跑了 4B；0.6B/1.7B 的 GSM8K 与三个尺寸的 5-shot loglikelihood 都没做。
- generative 口径的截断上界没有真正关闭——要关闭需要每题给到 1024+ token 预算，成本约为当前的 4 倍。
- 没有跨 seed/采样的方差估计，本章全部是 greedy 单次。
- CPU 与 CUDA 的 BF16 结果不同（`74` vs `73`）这一点只做了观察与解释，没有定位到具体哪个算子
  的归约顺序造成差异。
- 1.7B/4B 的 fp32 参照没做，因此协议对齐的 dtype 对照只有 0.6B 一个尺寸的证据。
- generative 的抽取规则仍是启发式。当前规则要求「无 `answer` 锚点时必须有 `.`/`)`/`:`/`,` 标记
  或位于文末」，这会把「模型确实答了但写法古怪」的情况判成未解析；宁可漏判也不误判，但代价
  是 1.7B 的未解析从 `52` 升到 `64`。
- HF 逐题参照只覆盖 0.6B；1.7B/4B 的 fp32 参照在 CPU 上太慢，未做。
