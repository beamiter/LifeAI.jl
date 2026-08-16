# Chapter 40 — 确定性 GridWorld observation/action 与真实反馈闭环

> 所属 Episode：Episode 08 — 环境与行动闭环
>
> 状态：Closed
>
> 日期：2026-08-16

## 本章问题

LifeAI 已经能让 Qwen3 调工具和检索跨请求记忆，但还没有一个由环境终态判定成功的 action loop。
本章回答：**模型能否只通过受限动作改变一个确定性环境，并根据每一步新 observation 调整后续动作，
最终完成任务；整条环境与模型轨迹能否不加载权重地重放？**

## 范围

- 与模型无关的 `Observation` / `Action` / `Transition` / `Environment` 最小类型。
- top-left origin、`x` 向 east、`y` 向 south 的 hidden-wall GridWorld。
- 唯一副作用面 `move(direction)`；方向 enum、精确参数集、边界、墙体和动作预算均 fail closed。
- 8 个冻结 synthetic tasks，BFS scripted oracle、full-feedback 与 feedback-withheld 两臂。
- Qwen3-4B CUDA BF16 真实任务成功率，以及 tokenizer-only + recorded completion 的第二进程 replay。
- 不做连续控制、动力学、视觉输入、并行动作、真实设备 adapter、急停或长期环境事件写回。

## 关键设计

### 1. 成功由环境状态决定

模型说“完成了”不会得分。只有位置等于 goal 且环境产生 `terminal=true, success=true` 才算成功；
提前自然语言回答、撞墙、未知方向、额外参数、动作耗尽和 terminal 后继续动作都会留下机械证据。

每个 spec、state、observation 和 transition 都有稳定 SHA256。state digest 包含 spec digest，因此两个
几何不同但当前位置相同的环境不会被当成同一状态。

### 2. 墙体隐藏，最新 observation 是唯一事实来源

initial observation 和每次 full tool result 只给当前位置、目标、当前 `legal_actions`、剩余预算和终止
状态，不暴露完整墙体。system prompt 要求一次只移动一步并等待反馈。这样任务不能退化为读完整地图后
一次性输出固定 path；下一步决定必须以新 observation 为准。

### 3. 对照证明反馈链路，而不是工具格式

两臂使用相同 model/session、system、user、tool schema、greedy 参数和初始 observation，8 / 8 对
首轮完整 prompt digest 与 token 数相同。withheld 臂仍真实执行动作，但后续 tool result 隐去位置、
合法动作、是否撞墙和原因，只保留 terminal/success。

这个控制证明的是**完整环境反馈链路是否有用**。第一轮之后两臂 prompt 内容与长度都会不同，因此它
不是 Chapter 39 那种 token-matched 单因素实验；不能把差异进一步归因为某个 observation 字段。

## 已实现

- `GridWorldSpec` / `GridWorldEnvironment` 与通用 environment observation/action/transition 类型。
- BFS shortest-path oracle、严格 fixture loader、task success/Wilson/非法动作/路径效率报告。
- `gridworld_move_tool`：唯一 state mutation surface；非法 payload 不改变环境。
- `run_qwen3_environment_loop` 与 `AgentEnvironmentTrace`：同时保存模型 steps 和环境 transitions。
- `run_qwen3_gridworld_eval.jl`：逐题交错运行 full / withheld，两臂首 prompt 不同就无法通过汇总门禁。
- `replay_qwen3_gridworld_eval.jl`：fresh environment 重新执行 recorded completions，独立重算 prompts、
  tool outcomes、observations、transitions 与 final states。

## 默认验证

Chapter 40 离线专项 `122 / 122`：覆盖 8 个 BFS oracle、fixture corruption、阻挡/预算/terminal、
未知 action 与额外参数无副作用、full/withheld feedback，以及两臂的模型无关完整 replay。

fixture SHA256：
`037ec76c95aa24d1429de11bf6d9cd988c0b056cd177f9c5354bc37d37a5377f`。

## 真实 Qwen3-4B 结果

2026-08-16 在 NVIDIA GeForce RTX 5080 16 GiB、宿主机 driver `595.71.05` 上运行冻结
Qwen3-4B revision `1cfa9a7208912126459214e8b04321603b3df60c`，CUDA native BF16、4,096-token
context、每 turn 96 个新 token、non-thinking greedy。模型加载 `30.873 s`，16 条 episode 共
`243.479 s`。

| 实验臂 | 成功 | 动作 | 非法/失败动作 | 相对 BFS 多余动作 |
| --- | ---: | ---: | ---: | ---: |
| BFS scripted oracle | `8/8` | `44` | `0` | `0` |
| Qwen3-4B full-feedback | `8/8` | `44` | `0` | `0` |
| Qwen3-4B feedback-withheld | `0/8` | `88` | `67` | 不适用 |

full-feedback 每题都与 BFS 最短步数相同（`4/4/4/4/6/6/8/8`），不是只“最终碰巧到达”。
withheld → full 的 8 个配对全部是 variant-only，准确率差 `+1.000`，双侧精确 McNemar
p=`0.0078125`；小样本 Wilson 95% 区间仍分别只有 `[.676, 1]` 与 `[0, .324]`，不外推到任意
地图或实体机器人。

第二进程 replay 16 / 16 episode：148 / 148 prompt digest、148 / 148 prompt token count、
148 / 148 recorded generation、140 / 140 tool outcome、132 / 132 transition digest 和 16 / 16
final state 全部一致。

本地 `benchmark_results/chapter40/` 中 summary / trace / replay SHA256 分别为：

- `21106196532b403ae42212c99792f8f2be82bf4d7ce193204d6eb67a07dae31b`
- `65fdfa594d2926bdeedb48e8bdb2b5536f3815d340b9ed2a536eb27bf5d5401a`
- `ccf4e3048200ce374df631acc57561ec60ded948b3bffcfc571c85dabdcae3ca`

benchmark 产物按仓库约定留在 Git ignore 的本地目录，代码、fixture、摘要数字与复现命令进入 Git。

## 复现命令

```bash
julia --project=. --startup-file=no scripts/run_qwen3_gridworld_eval.jl \
  /path/to/Qwen3-4B/<revision> \
  --out benchmark_results/chapter40 \
  --revision <revision>

julia --project=. --startup-file=no scripts/replay_qwen3_gridworld_eval.jl \
  test/episodes/episode08_environment_action_loop/chapter40_deterministic_gridworld/fixtures/gridworld_tasks.json \
  benchmark_results/chapter40/qwen3-4b_gridworld_trace.jsonl \
  /path/to/Qwen3-4B/<revision> \
  --out benchmark_results/chapter40/replay.json
```

replay 只读取 tokenizer，不加载 generation weights。

## 实验过程中的修正

首次 1-task smoke 虽然 full 成功、withheld 失败，但暴露了两个协议问题：tool result 把 transition
哈希重复塞进 prompt，且动作预算只比最短路多 2 步。正式冻结前把模型可见 feedback 收窄为下一条
observation，同时保留完整哈希在 trace；预算改为最短路约 2 倍但仍硬限制。重新运行后 full 的 8 题
全部从“成功”提升为“零非法动作且最短路成功”。smoke 结果不混入正式统计。

## Close 条件

- [x] observation/action/transition/environment 类型与确定性 GridWorld，所有 action 有界且 fail closed。
- [x] 冻结任务经 BFS oracle 证明可解，成功由环境终态机械判定。
- [x] 真实 Qwen3-4B full-feedback 完成多 step action loop，并报告成功率、非法动作和路径效率。
- [x] identical-first-prompt feedback-withheld 对照完成，任务收益不以工具格式合法率代替。
- [x] 第二进程不加载 generation model，重算完整 prompt/tool/environment 轨迹。

本章所有 Close 条件满足，于 2026-08-16 Closed。Episode 08 继续处理环境事件记忆写回和跨 adapter
安全语义。
