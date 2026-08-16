# Chapter 36 — Qwen3 tools chat template HF parity 与 4B 工具调用闭环

> 所属 Episode：Episode 07 — 智能体闭环
>
> 状态：Closed
>
> 日期：2026-08-16
>
> 真实资产：`Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c`
>
> 设备：NVIDIA GeForce RTX 5080（16 GiB），本地 NVMe，Julia 8 threads

## 本章问题

分成两个独立可证伪的问题，前者是硬门，后者只是描述性测量：

- **Q1**：LifeAI 的 `apply_qwen3_chat_template` 能否在官方 Qwen3 chat template 的**全部**分支上——`# Tools` 头、assistant `tool_calls`、`tool` 角色的 `<tool_response>`、think 块拆分、`last_query_index` 反向扫描——与 HuggingFace 渲染出逐字节相同的 prompt？
- **Q2**：在一份跑之前就冻结的 20 题任务集上，Qwen3-4B BF16 greedy 发出**合法** tool_call 的比率是多少？

Q2 允许为负，不作为 Close 的门槛，也不构成能力声称。

## 起因：Chapter 08 的 parity 是局部的

Chapter 08 冻结了 4 个 chat template case（`scripts/export_qwen3_text_reference.py`），全部是纯文本单轮或简单多轮。用真实 Qwen3-4B tokenizer 与官方模板对照，发现两条已经在发散的分支：

| case | LifeAI（修复前） | HuggingFace 官方 |
| --- | --- | --- |
| 历史 assistant 含 `<think>` | `…assistant\n<think>\nR\n</think>\n\nA<\|im_end\|>` | `…assistant\nA<\|im_end\|>` |
| 消息中没有任何 user | `…assistant\n<think>\n\n</think>\n\nA<\|im_end\|>` | `…assistant\nA<\|im_end\|>` |

根因是两处：

- `src/data/hf_tokenizer.jl` 的 assistant 非 think 分支打印**原始** `content`，而官方打印的是被 `</think>` 拆分后的 `content`。`_hf_assistant_content` 已经算出了可见部分，只是没有被这条分支使用。
- `last_user === nothing && (last_user = 0)`。官方语义是 `ns.last_query_index = messages|length - 1`，即**没有真实 user 时索引停在末尾之后**，于是任何 assistant 都走不到 think 分支；置 0 的效果恰好相反。

第一条不是纸面问题。`scripts/run_qwen3_cuda_chat.jl` 与 `scripts/run_qwen3_xla_chat.jl` 都有 `--thinking`，而 `<think>` 在 tokenizer_config 中 `special=false`，`skip_special_tokens=true` 不会移除它；两个脚本都把 `result.completion` 原样 push 进 history。因此开启 thinking 时，**从第 2 轮起** LifeAI 的 prompt 就与 HuggingFace 不同。

## 参照物：不需要 transformers 的 Python oracle

本章的参照物是外部独立实现，而不是 LifeAI 的另一条代码路径。Transformers 编译 chat template 的环境可以完整复现，且只依赖 jinja2：

```python
env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                    extensions=[jinja2.ext.loopcontrols])
env.filters["tojson"] = lambda v, indent=None: json.dumps(v, ensure_ascii=False, indent=indent)
```

`scripts/export_qwen3_chat_template_reference.py` 断言 `chat_template` 的 sha256 等于
`a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8`，渲染冻结的 case 列表，并输出带 `template_sha256` / `cases_sha256` / `jinja2_version` 的 reference JSON。依赖钉死在
`requirements/chapter36-template-oracle.txt`（`jinja2==3.1.6`、`markupsafe==3.0.3`）；本机 PEP 668 禁止全局 `pip install`，可行路径是 `uv venv` + `uv pip install`，离线时解包 wheel 后用 `PYTHONPATH` 亦得到逐字节相同的 reference。

## 实现

**渲染**：`apply_qwen3_chat_template` 按官方模板逐分支重写，新增 `tools` kwarg，并支持 `tool` 角色与 assistant `tool_calls`。归一化阶段保留 `content_is_string` 标志，因为官方的 `message.content is string` 守卫会影响 `last_query_index` 的反向扫描：一个 `content` 缺失的 user 不是查询。

**有序 JSON**：官方 `tojson` 是 CPython `json.dumps(..., ensure_ascii=False)`，分隔符 `", "` 与 `": "`、按插入顺序、不排序、中文不转义。`_python_json` 复刻这一语义，只接受保序容器（`NamedTuple`、`Vector{<:Pair}`、`OrderedJSONObject`）。数字是最难的部分，见下方「对抗式复核抓到的问题」：JSON 文本必须由 `parse_qwen3_json` 解析以保留整数/浮点的词法区分，浮点由 `_python_float_repr` 按 CPython `repr` 规则输出，整数保持任意精度；无序 `Dict` 与 `JSON3.Object` 一律拒绝。

**fail closed 边界**：传入非空 `tools`、`tool` 角色或非空 `tool_calls` 时，要求 tokenizer 的 `chat_template` sha256 等于官方常量。tiny fixture 模板（`07bdce04…`）与 embedding 模板都会被拒绝。`tools=[]` 不改变任何字节，因此仍然放行。`src/data/hf_tokenizer.jl` 既有的三选一模板 hash 校验一字未改。

**闭环**：`src/agent/tools.jl` 提供 `AgentTool` / `ToolRegistry` / `parse_qwen3_tool_calls` / `invoke_agent_tool`，以及三个内置工具——`add_integers`、`list_directory`、`read_text_file`；后两个被限制在一个冻结的 sandbox root 内，`../`、绝对路径与 `normpath` 后越界全部拒绝。`src/agent/tool_loop.jl` 的 `run_qwen3_tool_loop` 每一步都重新渲染完整消息列表，因此第 n 轮的 prompt 就是 HuggingFace 从同一段历史会构造的 prompt。

一个有意的取舍：官方模板把可见内容全部放在 `<tool_call>` 之前，所以夹在多个 tool_call 之间或之后的正文**无法**经模板往返。`_qwen3_visible_assistant_content` 因此只保留第一个 `<tool_call>` 之前的前缀；被丢弃的部分完整保留在 JSONL 记录的 raw completion 里。

## 测量方法

**Q1**：30 个 case 覆盖 think 拆分、`reasoning_content` 显式字段、`last_query_index` 的四种形态（普通、无 user、`<tool_response>` 包裹的 user、tool 之后的真实 user）、tools 头有无 system、两个函数与中文 description、boolean/null/空数组 schema、tool_call 的空内容与非空内容换行规则、字符串 arguments、扁平 tool_call、多个 tool_call、连续 tool 消息合并，以及 `add_generation_prompt` × `enable_thinking` 组合。

比较分两层：离线默认测试用冻结的官方模板原文构造 tiny-vocabulary tokenizer 做**字符串**逐字节比较；opt-in 的 `LIFEAI_QWEN3_4B_MODEL_DIR` 用真实 151,669 词表再比较一次 **token id 序列**。

**Q2**：`test/.../fixtures/tool_loop_tasks.json` 的 20 题在第一次测量前冻结（8 题算术、6 题目录列举、6 题文件读取，含 1 题读取不存在的文件），system prompt 一并冻结。判定是机械的：`valid` = 至少一个 `<tool_call>` 块，且每个块 JSON 可解析、函数名已注册、必需参数齐全。同时报告 Wilson 95% 区间与「未发生字符串→整数强制转换」的严格计数。

复现命令：

```bash
# reference（oracle）
python3 scripts/export_qwen3_chat_template_reference.py \
  --model-dir /home/yj/models/huggingface/Qwen/Qwen3-4B/1cfa9a7208912126459214e8b04321603b3df60c \
  --out  test/episodes/episode07_agent_closed_loop/chapter36_qwen3_tools_chat_template/fixtures/chat_template_reference.json \
  --template-out test/episodes/episode07_agent_closed_loop/chapter36_qwen3_tools_chat_template/fixtures/official_chat_template.jinja

# 闭环
julia --project=. --startup-file=no scripts/run_qwen3_tool_loop.jl \
  /home/yj/models/huggingface/Qwen/Qwen3-4B/1cfa9a7208912126459214e8b04321603b3df60c \
  --variant qwen3_4b --revision 1cfa9a7208912126459214e8b04321603b3df60c \
  --out benchmark_results/chapter36/tool_loop_trace_run1.jsonl \
  --summary benchmark_results/chapter36/tool_loop_summary_run1.json
```

## 真实结果

### Q1 — 模板 parity

30 / 30 case 与官方 Jinja 渲染逐字节相同，真实 4B tokenizer 下 token id 序列也逐位相同。两条 Chapter 08 分歧已按 before/after 冻结进测试。

值得记录的一点：30 个 case 只产生 **27 个互不相同的字符串**——`tool_call_empty_content`、`tool_call_string_arguments`、`tool_call_flat_shape` 三者渲染结果相同（模板对字符串 arguments 原样输出，且 `.function` 存在时解包），`last_assistant_think` 与 `reasoning_content_field` 相同。这是模板的正确行为，但意味着 oracle 单独无法区分这三种输入；区分能力来自 Julia 侧接受了三种不同的输入结构。

实现期间由 fixture 抓到一个真实 bug：`_python_json` 判定「Pair 向量即对象」时用了 `all(...)`，而 `all` 对空集合返回 true，于是**空数组被渲染成 `{}`**。`tools_boolean_and_null_schema`（含 `"examples": []` 与 `"required": []`）是唯一暴露它的 case。同一个 bug 也会影响 `required=String[]` 的工具声明。

另一个顺带确认的事实：本机六个 dense checkpoint（0.6B / 1.7B / 4B / 8B / 14B / 32B）的
`chat_template` sha256 全部等于 `a55ee1b1…74d8`，因此本章的 parity 与 fail-closed 门禁对整个
dense family 成立，不只对 4B。

### Q2 — 20 题闭环（描述性，非能力声称）

任务集 sha256 `2973f840dded3fa3b21a2bab74a0d0be1cd4652887039a0a0b774234a0dc10bf`，
Qwen3-4B BF16、CUDA eager、greedy、4,096 context、`max_new_tokens=256`、`max_steps=4`、
non-thinking，sandbox root 为仓库根。

| 指标 | 值 |
| --- | --- |
| `valid_tool_call_rate` | `18 / 20` = `0.900` |
| Wilson 95% 区间 | `0.699 – 0.972` |
| 严格计数（无字符串→整数强制转换） | `18 / 20`（本次没有发生任何强制转换） |
| `invalid`（发出了块但不合法） | `0 / 20` |
| `none`（没有发出任何块） | `2 / 20` |
| 工具执行成功 | `17 / 18` |

`N = 20` 下区间宽度约 `0.27`。这个数字只能判断「协议是否完全不可用」，不构成能力声称。

两个 `none` 都值得如实记录，它们的性质不同：

- `add_worded`（“我有 7 个苹果，朋友又给了 35 个”）直接回答 `You have exactly 42 apples.`——答案正确，但没有走工具。
- `read_project_version` 回答 `I cannot directly access or read the contents of a file unless I have the necessary tools or functions to do so.`——`read_text_file` 明明已在 `# Tools` 里声明，这是真实的模型失败。

唯一一次工具执行失败是 `read_missing`，属于设计内的负例：模型正确发出 `read_text_file("does_not_exist.txt")`，工具 fail closed 返回 `no such path: does_not_exist.txt`，模型在第 2 轮如实报告文件不存在。

**确定性**：两次**独立进程**运行，20 个任务共 38 个 model turn，全部 `prompt_sha256` 与
`generated_ids` 逐位相同，两份 summary 的所有计数一致。

**资源**：加载约 `31 s`；首个 turn 的 prefill 约 `16.7 s`（冷启动，含 kernel warmup），
其后稳定在约 `0.7 s`。加载后剩余显存 `7,233,470,464` bytes，运行期间最低剩余
`50,724,864` bytes——4B BF16 + 4K KV 在 16 GiB 卡上配合 CUDA allocator 的池化确实贴到了
边界，这是本机的真实约束，不应外推到 24 GiB 机器。

上表与轨迹是**复核修复之后**用最终代码重跑的。复核前那一版的沙箱在文件不存在时报
`not a file`，修复后改为 `no such path`，会改变第 2 轮的 prompt，因此旧轨迹不再代表当前
代码，已整体作废重跑；两版的合法率计数恰好相同。

**实现期间抓到的第二个真实 bug**：第一次测量时全部 12 个文件系统工具调用执行失败。原因是
`normpath(joinpath(@__DIR__, ".."))` 产生带尾部斜杠的 sandbox root，而 `_sandboxed_path` 用
`root * "/"` 做前缀比较，得到 `…LifeAI.jl//` 而永远匹配不上。**发出的 tool_call 全部合法，
失败发生在执行侧**——这恰好说明为什么 validity 与 execution 必须分开计数。修复后重跑全部
20 题，并补了带尾部斜杠 root 的回归测试。上面表格是修复后的重跑结果，带 bug 的那次不作数。

## 对抗式复核抓到的问题

Close 前对整份改动做了一轮多视角对抗复核（每条主张都要求给出可复现的失败输入）。结果值得单独记录，因为其中一条**直接击穿了本章的核心主张**：

1. **`JSON3.read` 把整值浮点窄化成 `Int64`（high）**。`JSON3.read("{\"a\": 1.0}").a === 1`，`1e2`、`-0.0` 同理；而 `JSON3.Object` 恰恰是当时文档里推荐的 tools 入口。JSON Schema 的 `"default": 1.0` / `"maximum": 100.0` / `"multipleOf": 2.0` 因此会渲染成 `1` / `100` / `2`，与官方相差 6 字节、token id 分叉，**并且不抛任何异常**。原本的「浮点 fail closed」保护根本走不到。反向也错：超过 `Int64` 的整数被 JSON3 变成 `Float64`（`99999999999999999999999` → `1.0e23`）后反而触发了保护，而 CPython 是精确回显的。28 个 fixture 里一个浮点都没有，测试完全看不见。
2. **模型输出的浮点参数会让整轮运行崩溃（high）**。`_python_json_text(call.arguments)` 在记录轨迹时无保护地抛出，越过 `invoke_agent_tool` 已经做好的优雅降级，直接结束整个脚本。
3. **沙箱只做词法检查，符号链接可越界（high）**。`normpath` 不解析软链，而随后的 `isfile` / `open` / `readdir` 都会跟随，因此沙箱根内任意指向外部的软链都是完整读原语。原测试只覆盖了 `..` 与绝对路径。
4. **`_truncate_to_valid_utf8` 是 O(n²) 且语义错误（high）**。它每轮重建整个 `String` 做 `isvalid`，且因为检查的是整体前缀有效性，实际会从**第一个**非法字节处截断并丢弃其后全部内容，与 docstring 声称的「去掉尾部残缺序列」不符。
5. 单个工具误传成裸 `NamedTuple` 时会按字段迭代，静默渲染出若干无意义声明（low）。
6. `fit_qwen3_chat_context` 的裁剪不认识 `tool` 角色，丢掉 assistant 后会留下无人发起的孤立 `<tool_response>`（low）。
7. 脚本的 `parse(Int, …)` 与空任务集缺少校验（low）；新测试缺 `LIFEAI_REPO_ROOT` 的 `isdefined` 守卫（low）。

修复方式：

- 新增 `parse_qwen3_json` 与 `OrderedJSONObject`——一个保留 JSON 数字词法形态的解析器（有小数点或指数即为浮点，否则是任意精度整数），tools 声明与模型 `<tool_call>` 参数都改走它；
- 新增 `_python_float_repr`，按 CPython `repr(float)` 规则输出（小数点落在 `-3:16` 用定点，否则 `e±NN` 且指数至少两位），因此浮点不再 fail closed 而是**真正正确**；
- `_python_json` 对 `JSON3.Object` 直接抛错并指明改用 `parse_qwen3_json`，杜绝任何静默路径；
- 沙箱改为 `realpath` 解析后再做包含判断，并显式拒绝穿越软链；
- `_truncate_to_valid_utf8` 改为只回退最多 3 个续字节判断尾部序列是否完整（线性），非 UTF-8 文件由调用方 fail closed；`max_bytes` 上限 1 MiB；
- 其余按上表逐条修复，并各自补了回归测试。

fixture 相应新增 `tools_float_schema` 与 `tool_call_float_arguments` 两个 case（共 30 个），把 `1.0` / `0.5` / `1e-05` / `0.0001` / `1e+16` / `1000000000000000.0` / `1.2345678901234567e+20` / `-0.0` 与 `99999999999999999999999` 全部钉死在官方 reference 上。

**这次复核的教训**：一个「fail closed」保护如果建立在错误的类型假设上，它给出的安全感比没有保护更危险——`_python_json` 明明写了浮点保护，却因为上游解析器已经把浮点变成了整数而形同虚设。fixture 覆盖不到的分支，等于没有保护。

## 决策

- 模板 parity 是硬门，工具调用率是描述性测量，两者在文档中严格分开；README 只能写「具备 tools chat template 与单轮工具调用闭环机制」，不得写「具备 agent tool loop」。
- 无序 `Dict` 在 tool payload 中 fail closed；浮点则**正确实现**（CPython `repr` 规则）而不是拒绝，因为拒绝在 `JSON3` 已经窄化的前提下根本触发不到。
- `JSON3.Object` 在渲染路径上被直接拒绝并指向 `parse_qwen3_json`：宁可让调用方改一行，也不留任何静默偏离的入口。
- tools / tool 角色 / tool_calls 全部以官方模板 sha256 为准入条件；tiny fixture 与 embedding 模板保持拒绝。
- 夹在 tool_call 之间的正文按「保留前缀、raw completion 兜底」处理，并显式记录该限制，而不是拼接出一个无法往返的历史。
- 任务集与 system prompt 冻结后不因结果不好看而回调重跑。

## 验证

- 默认全套：`7,545 / 7,545`。Episode 01—06 计数与改动前完全一致（`3862 / 294 / 496 / 543 / 468 / 1602` = `7,265`），Episode 07 新增 `280`；opt-in 真实 tokenizer 打开后 Chapter 36 为 `345`。
- Chapter 36 离线测试：官方模板资产冻结（含 `cases_sha256` 与 reference 的耦合校验）、30 case 逐字节 parity、两条分歧回归、非官方模板下的 fail-closed、CPython JSON 语义、工具注册/解析/沙箱（含尾部斜杠 root）/Wilson 区间。
- opt-in `LIFEAI_QWEN3_4B_MODEL_DIR`：真实 tokenizer 的 30 case 字符串与 token id 双重 parity，四个 tool 标记均为单 token。
- replay 测试不加载模型、不使用 GPU，从冻结轨迹重建每一轮消息并复算 prompt sha256。
- `scripts/verify_qwen3_text_parity.jl` 未改动。

## 遗留

- 20 题只能判断「协议是否可用」，不能判断「任务是否答对」。真正的质量测量需要独立的评测章节。
- 闭环仍是单请求内的多 step，没有跨请求持久状态、没有记忆写回、没有检索。
- 未接入 XLA 常驻 session 与 HTTP 服务；`/api/generate` 仍不支持 tools。
- `arguments` 的字符串→整数强制转换按 case 记录，但没有对模型输出 schema 做类型校验。
