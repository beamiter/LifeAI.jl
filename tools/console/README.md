# LifeAI 推理路由与驻留服务

交互客户端已重构为仓库内的 Rust TUI **[Lumen（流明）](../tui/README.md)**。
使用 `lifeai chat --model flint` 或 `lumen --model qwen`；旧 `qwen` 命令及
Python TUI 已移除。本目录保留推理路由、worker 控制与部署配置。

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

代码、模型、Julia 环境和日志使用 JuiceFS 工作区 `/jfs/application/user/bgl`。
旧 `/mnt/public_data/bgl` 已成为兼容符号链接；新任务直接使用 JuiceFS 路径。

接口：`GET /api/models` 列模型；`POST /api/select` 异步选择模型；
`GET /healthz` 返回加载阶段与限制；`POST /api/chat` 接受 role/content 消息，
支持逐 token NDJSON。提示词模板在服务端生成，客户端不用混用两个模型的模板。

```sh
python3 -m unittest -v test_server.py test_control.py
```

`control.py` 校验进程 PID、启动时间与脚本路径，并通过 pidfd 停止所属进程；
启动/停止持目录锁。它不会按端口或进程名结束其他程序。
