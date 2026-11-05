# Lumen（流明） · LifeAI 终端

Lumen 寓意把模型能力化为可交互的光。客户端使用 Rust、Ratatui 与 Crossterm，
集成在 LifeAI 仓库内；通过 LifeAI 路由使用 Qwen3-8B、Flint 和 Ripple-65B。
Qwen/Flint 在 DSW 推理，Ripple 使用已部署的 DLC 八卡服务；客户端不需要 Julia、Python、本地 GPU 或 DLC 密钥。

## 编译与启动

需要 Rust 1.88+ 和 OpenSSH 客户端。依赖锁定在 `Cargo.lock` 中。

```sh
# 在 LifeAI 仓库根目录
cargo build --release --locked --manifest-path tools/tui/Cargo.toml
./scripts/lifeai chat --model ripple

# 安装到 ~/.local/bin，提供 lifeai 和 lumen 两个入口
./scripts/lifeai --install
lifeai chat --model flint
lumen --model ripple

# 单次提问、管道输入、只读状态查询
lifeai chat --model flint -n 256 '1+1 等于几？'
lifeai chat --model flint -n 2048 --file long-document.txt
cat question.txt | lifeai chat --model qwen
lifeai --status
```

默认模型为 Qwen，可用 `LIFEAI_MODEL=ripple` 更改。`lifeai chat` 与 `lumen`
都进入同一界面；旧 `qwen` 命令已废弃，不提供兼容别名。
安装不会修改 shell 配置；请确保 `~/.local/bin` 在 `PATH` 中。
`--prefix /path` 可更改安装位置，已有同名文件会备份为 `.before-lumen.N`。

## 对话与编辑

| 操作 | 快捷键 / 命令 |
| --- | --- |
| 切换 Qwen / Flint / Ripple | `Ctrl+T`、`F2`、`/model qwen`、`/model flint`、`/model ripple` |
| 发送 / 换行 | `Enter` / `Alt+Enter`（支持的终端也可用 `Shift+Enter`） |
| 清空当前模型会话 | `Ctrl+L`、`/clear` |
| 重试连接或加载 | `Ctrl+R` |
| 滚动对话 | `PgUp` / `PgDn` |
| 行首 / 行尾 | `Home` / `End`、`Ctrl+A` / `Ctrl+E` |
| 清空草稿 | `Ctrl+U` |
| 从 UTF-8 文件导入当前草稿 | `/load 文件路径`（最多 64 MiB） |
| 帮助 / 退出 | `/help` / `Ctrl+C`、`Ctrl+D`、`/quit` |

三个模型分别保留本次进程内的会话、草稿和滚动位置。失败或不完整回答
不会写入发送给模型的历史。Flint 与 Ripple 已完成的 `<think>` 内容显示在终端中，
下一轮只回传最终答案。退出后不保存会话。

支持中文、组合字符和 emoji 的字素编辑、按显示宽度换行与多行粘贴。
转义控制符不会被作为终端指令执行。界面显示加载阶段、等待时间、
上下文限制和生成速率；网络请求在后台线程执行，加载期间仍能编辑、滚动和退出。
长会话复用换行缓存，空闲刷新不会重复排版全部对话。
长文件草稿只排版光标附近的内容；Flint 长输入会显示预填充 token 数与百分比，
完成后开始显示生成内容。单次问题、文件和标准输入最多 64 MiB UTF-8；全部对话合计最多 64 Mi 字符，
序列化 JSON 请求最多 128 MiB。生成响应仍最多 16 MiB。传输上限与已验收的 token 上限分别检查。

Qwen/Flint 共用 DSW 本地 GPU，切换会释放旧 worker 并加载新模型，可能耗时数分钟。
Ripple 切换仅连接 DSW 上的轻量适配器；DLC 模型独立常驻，切换或退出不会停止 DLC 任务。
加载或生成期间禁止再次切换。退出客户端会关闭它创建的 SSH 进程，
远端推理服务继续驻留。SSH 断线后可按 `Ctrl+R` 重建连接，会话保留。

## 连接与参数

默认连接 `http://127.0.0.1:11435`。无法直连时，客户端通过 OpenSSH 将此本地
端口转发到 DSW 的 `127.0.0.1:11435`，使用本机已授权密钥。可指定：

```sh
lifeai chat --model flint --ssh-key ~/.ssh/dsw_key
lifeai chat --ssh-host HOST --ssh-user root --ssh-port 22
lifeai chat --url http://127.0.0.1:11435 --no-tunnel
```

环境变量：`LIFEAI_MODEL`、`LIFEAI_URL`、`LIFEAI_SSH_HOST`、`LIFEAI_SSH_KEY`。
命令行参数优先。`--known-hosts` 可指定 SSH 主机记录文件。

`-n / --max-tokens` 默认 256；`--temperature` 默认 0，上限 2（Qwen 服务仅支持 0）；
`--timeout` 默认 21600 秒（6 小时，最多 86400 秒），`--load-timeout` 默认 1800 秒；`--system` 设置系统提示词；
`--no-stream` 等待完整回答后输出。上下文和输出上限由服务发布，超出时会报错。
Flint 的 262144-token 上限包含模板、全部历史和预留输出；文件大小不能代替
token 预算。需要为长文件腾出预算时，可先用 `/clear` 开启新会话。
`--status` 只读取当前驻留状态，不切换或加载默认模型。

客户端使用 `/healthz`、`/api/select`、`/api/chat`；模板在服务端生成。
Ripple 同样连接路由的 11435 端口；DLC Bearer key 只由 DSW 适配器读取，不传到客户端。
Ripple 当前完整回答后一次显示，`--no-stream` 也可使用；支持 system/user/assistant 多轮消息。
上游支持异步任务时，Lumen 可显示真实预填充进度；中间心跳保持长请求连接，退出时取消本次任务。
实际上下文上限由当前部署发布，只有通过对应阶段验收后才开放；最大输出仍为 512 tokens，超限会显示服务错误；
TUI 显示输出 tokens、总耗时与服务发布的真实解码速率；上游未提供 decode 计时时不显示解码速度。
也可连接旧版单模型 Qwen `/api/generate` 服务，但该服务无法切换到 Flint。
服务部署与 FP4 worker 说明见 [console](../console/README.md)。

## 验证

```sh
cargo fmt --manifest-path tools/tui/Cargo.toml --check
cargo test --locked --manifest-path tools/tui/Cargo.toml
cargo clippy --locked --manifest-path tools/tui/Cargo.toml --all-targets -- -D warnings
```

测试覆盖 Ripple 非流式上游到完整 NDJSON/JSON 的适配、鉴权与错误恢复、三模型会话，以及真实本地 HTTP 流、UTF-8 分段、失败与不完整事件、模型加载与限制，
以及 Ratatui TestBackend 下的会话隔离、编辑和终端布局；无需连接 DSW。

可生成无需目标机 Rust 环境的 Linux 静态二进制：

```sh
rustup target add x86_64-unknown-linux-musl
cargo build --release --locked --target x86_64-unknown-linux-musl --manifest-path tools/tui/Cargo.toml
```

框架 API 参考：[Ratatui](https://ratatui.rs/installation/)、
[Reqwest](https://docs.rs/reqwest/latest/reqwest/blocking/)。
