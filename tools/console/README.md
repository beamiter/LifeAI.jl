# LifeAI 终端聊天

保留 `qwen` 命令，可在同一界面使用 Qwen3-8B 和 Flint。客户端仅使用 Python 标准库及 OpenSSH；权重与推理留在 DSW。

```sh
python3 qwen.py --install
qwen                            # 进入 TUI，默认 Qwen
qwen --model flint               # 进入 Flint TUI
qwen --model flint '请介绍自己'  # 单次提问
qwen --model qwen --status
```

TUI 快捷键：`Ctrl+T` / `F2` 切换模型，`Ctrl+L` 清空当前模型历史，`Ctrl+R` 重连，`PgUp/PgDn` 滚动，`Ctrl+C` 退出。也可输入 `/model qwen`、`/model flint`、`/clear`、`/help`、`/quit`。两个模型的历史分别保存在当前进程内；切换或失败不会丢弃另一模型的历史。首次加载可能需要数分钟，界面会显示连接及加载状态。

长会话的换行结果会复用，长提示词编辑只测量光标附近的窗口；空闲刷新和滚动不重复扫描全部文本。完成状态同时支持 Qwen 与 Flint 的速率指标。

默认通过 SSH 转发到 DSW 的 `127.0.0.1:11435`，可以通过 `--ssh-host`、`--ssh-user`、`--ssh-port`、`--ssh-key` 覆盖；已有连接可用 `--url http://127.0.0.1:11435 --no-tunnel`。也保留 `QWEN_URL`、`QWEN_SSH_HOST`、`QWEN_SSH_KEY` 环境变量，默认模型可设置 `LIFEAI_MODEL`。

`-n / --max-tokens` 默认 256；`--temperature` 默认 0（贪心）；`--timeout` 控制生成等待时间，默认 300 秒；`--load-timeout` 控制模型切换等待时间，默认 1800 秒。模型限制取自服务，超出时提示调整参数。

新版服务使用 `/healthz`、`/api/select`、`/api/chat`，聊天模板在服务端生成。旧版单模型 Qwen 服务会自动使用 `/api/generate`；旧服务无法切换 Flint。

```sh
python3 -m unittest -v test_qwen.py
```

测试仅使用本地 HTTP 和内存 mock，不连接 SSH 或 DSW。

## 驻留服务与部署

DSW 部署目录为 `/jfs/application/user/bgl/deployments/lifeai_console`。
服务监听 `127.0.0.1:11435`；Qwen 与 Flint worker 分别使用 11436、11437，
均仅监听回环地址。客户端自动建立 SSH 转发。

```sh
python3 /jfs/application/user/bgl/deployments/lifeai_console/control.py status
python3 /jfs/application/user/bgl/deployments/lifeai_console/control.py start --model flint
python3 /jfs/application/user/bgl/deployments/lifeai_console/control.py stop
```

`models.json` 保存两个 worker 的命令、环境、日志和限制。Flint 使用
LifeAI NVFP4 与原生 cuBLASLt FP4 GEMM；首次加载在 GPU 上量化，并保留驻留
权重供后续请求复用；就绪前会预热推理。Qwen 保留原有 BF16/XLA 推理服务。一次驻留一个模型，
切换时等待旧 worker 完全退出后再加载，避免两份权重争用显存。生成期间
拒绝切换；客户端退出只关闭它创建的 SSH 通道，远端模型服务继续运行。

代码、模型和日志使用 JuiceFS 工作区。已有 Julia 运行时与依赖暂复用
`/mnt/public_data/bgl/envs/lifeai`，无需在本次任务中整体搬迁旧环境。

接口：`GET /api/models` 列模型；`POST /api/select` 异步选择模型；
`GET /healthz` 返回加载阶段与限制；`POST /api/chat` 接受 role/content 消息，
支持逐 token NDJSON。提示词模板在服务端生成，客户端不用混用两个模型的模板。

```sh
python3 -m unittest -v test_qwen.py test_server.py test_control.py
```

`control.py` 校验进程 PID、启动时间与脚本路径，并通过 pidfd 停止所属进程；
启动/停止持目录锁。它不会按端口或进程名结束其他程序。
