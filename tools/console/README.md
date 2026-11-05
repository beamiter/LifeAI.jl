# LifeAI 推理路由与驻留服务

交互客户端已重构为仓库内的 Rust TUI **[Lumen（流明）](../tui/README.md)**。
使用 `lifeai chat --model flint` 或 `lumen --model ripple`；旧 `qwen` 命令及
Python TUI 已移除。本目录保留推理路由、worker 控制与部署配置。

## 驻留服务与部署

DSW 部署目录为 `/jfs/application/user/bgl/deployments/lifeai_console`。
服务监听 `127.0.0.1:11435`；Qwen、Flint worker 分别使用 11436、11437，Ripple 适配器使用 11438，
均仅监听回环地址。客户端自动建立 SSH 转发。

```sh
python3 /jfs/application/user/bgl/deployments/lifeai_console/control.py status
python3 /jfs/application/user/bgl/deployments/lifeai_console/control.py start --model ripple
python3 /jfs/application/user/bgl/deployments/lifeai_console/control.py stop
```

`models.json` 保存各 worker 的命令、环境、日志和限制。Flint 使用
LifeAI NVFP4 与原生 cuBLASLt FP4 GEMM；首次加载在 GPU 上量化，并保留驻留
权重供后续请求复用；就绪前会预热推理。Qwen 保留原有 BF16/XLA 推理服务。一次驻留一个模型，
切换时等待旧 worker 完全退出后再加载，避免两份权重争用显存。生成期间
拒绝切换；客户端退出只关闭它创建的 SSH 通道，远端模型服务继续运行。

代码、模型、Julia 环境和日志使用 JuiceFS 工作区 `/jfs/application/user/bgl`。
旧 `/mnt/public_data/bgl` 已成为兼容符号链接；新任务直接使用 JuiceFS 路径。

接口：`GET /api/models` 列模型；`POST /api/select` 异步选择模型；
`GET /healthz` 返回加载阶段与限制；`POST /api/chat` 接受 role/content 消息，
支持逐 token NDJSON。提示词模板在服务端生成，客户端不用混用两个模型的模板。

Flint 部署示例的总上下文为 262144 tokens，输出上限 4096；总预算包含提示词模板、
历史和预留输出。worker 用 `--context-tokens` 与 `--prefill-chunk-size` 配置，
默认每批预填充 4096 tokens，并通过 NDJSON 的 `phase=prefill` 发布进度。
原生 FP4 服务使用 `--activation-scaling token` 避免跨 token 共用激活缩放；
`--max-new-tokens` 配置输出上限，worker 命令本身的默认值为 512。
路由和 worker 接受最多 16 MiB JSON 请求、64 条消息及合计 4 Mi 字符文本；
实际 token 预算由 worker 编码后校验。部署示例的请求超时为 7200 秒。
长上下文内存和整模型验收见 [256K 说明](../../notes/flint_context256k.md)。

## Ripple DLC 接入

`ripple_worker.py` 提供现有 LifeAI worker 协议，将消息转为 DLC 的
`/v1/chat/completions` 请求。`models.dsw.example.json` 含 `ripple` 条目；
适配器读取 `configs/deployment/ripple_tp8/deployment.json` 的运行目录，再读取
`results/ready.json` 与 `credentials/api_key`，检查远端模型身份并使用 Bearer 鉴权。
也可将 `--deployment` 直接指向运行目录。密钥不出现在模型列表、健康响应或客户端。

启动/停止适配器不提交、停止或重新加载 DLC 任务。路由只拥有适配器子进程；
Qwen/Flint 的本地显存互斥与生命周期规则继续适用。Ripple 的 8192/512-token
限制来自实际部署；完整历史消息由 DLC 服务生成模板并检查 token 预算。

DLC 服务当前非流式。适配器对普通请求返回一个完整 JSON，对原生流式请求返回
一个 `done=true` 的 NDJSON 记录，不模拟逐 token 推送；发布端到端计时，并仅在上游提供有效解码速率时转发该速率。
上游错误或不完整回答不生成成功事件，路由释放请求占用，客户端可重试。

```sh
# 安装/编译更新后的 Lumen 后，通过已有 SSH 路由使用
lumen --model ripple
lumen --model ripple -n 8 'Reply with exactly OK.'
lumen --model ripple --no-stream '请用一句话解释二分查找。'

# 在本目录运行协议与生命周期测试
python3 -m unittest -v test_server.py test_control.py test_ripple_worker.py
```

`control.py` 校验进程 PID、启动时间与脚本路径，并通过 pidfd 停止所属进程；
启动/停止持目录锁。它不会按端口或进程名结束其他程序。
