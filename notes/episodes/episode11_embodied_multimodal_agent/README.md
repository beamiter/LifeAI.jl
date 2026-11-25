# Episode 11 — 具身多模态视觉-语言-行动安全闭环

> 状态：Open
>
> 收录章节：Chapter 153–

## 本卷主题

Episode 07 建立了工具协议与语义记忆，Episode 08 闭合了确定性环境、事件记忆与跨 adapter 动作安全契约（Chapter 150），Episode 09/10 建立了 Qwen3-VL 图像、多图与视频感知生成能力。Episode 11 将这三条主线连接为统一、可审计的**具身视觉-语言-行动（VLA）安全闭环**：让视觉/视频模型直接观察环境相机画面，通过受超时、失败分类、急停联锁和幂等保护的安全工具执行行动，并仅在干净成功的回合写回长期记忆。

## 章节目录

1. [`Chapter 153 — 具身多模态 VLA 安全闭环`](chapter153_embodied_vla_loop.md)（Closed；离线 CPU 专项 `39/39` 通过，真实 `Qwen3-VL-2B-Instruct` Blackwell GPU 视觉感知/工具调用/安全审计 smoke 通过）

## Episode Close 条件

- 打通不依赖 GridWorld 的视觉/视频观测协议与连续空间具身环境，所有动作强制经过 Chapter 150 `SafeEnvironmentAdapter` 并可离线审计重放。（Chapter 153 已满足）
- 在真实 VLM checkpoint 与目标 GPU 上完成端到端多步视觉观察 → 结构化工具调用 → 安全执行与急停/幂等保护验收。（Chapter 153 已完成 2B smoke，并记录零样本 2B 模型的连续控制上限）
- 扩展到带真实物理动力学/机械臂或机器狗仿真器（如 MuJoCo/Isaac 外部 IPC adapter）与多任务视觉成功率评测，再评估关闭本卷。
