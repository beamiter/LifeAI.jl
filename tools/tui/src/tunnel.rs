//! Own exactly the OpenSSH child we spawn; never stop a remote inference worker.
use crate::{Args, api::Api};
use anyhow::{Context, Result, bail};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub type Shutdown = Arc<Mutex<Option<Child>>>;

pub struct Tunnel {
    child: Shutdown,
}

pub fn shutdown(child: &Shutdown) {
    if let Ok(mut guard) = child.lock()
        && let Some(mut process) = guard.take()
    {
        let _ = process.kill();
        let _ = process.wait();
    }
}

impl Tunnel {
    pub fn shutdown_handle(&self) -> Shutdown {
        self.child.clone()
    }
}

impl Drop for Tunnel {
    fn drop(&mut self) {
        shutdown(&self.child);
    }
}

fn expand(path: &std::path::Path) -> PathBuf {
    if let Ok(rest) = path.strip_prefix("~")
        && let Some(home) = std::env::var_os("HOME")
    {
        return PathBuf::from(home).join(rest);
    }
    path.to_owned()
}

pub fn connect(args: &Args, api: &Api) -> Result<Tunnel> {
    let tunnel = Tunnel {
        child: Arc::new(Mutex::new(None)),
    };
    let on_interrupt = tunnel.child.clone();
    ctrlc::set_handler(move || {
        shutdown(&on_interrupt);
        // SIGINT outside raw mode must also restore a partially initialized TUI.
        if std::io::IsTerminal::is_terminal(&std::io::stdout()) {
            ratatui::restore();
            let _ = crossterm::execute!(std::io::stdout(), crossterm::event::DisableBracketedPaste);
        }
        std::process::exit(130);
    })
    .context("注册退出处理失败")?;
    ensure_connection(args, api, &tunnel.child)?;
    Ok(tunnel)
}

/// Retry the transport in the network worker without discarding TUI sessions.
pub fn ensure_connection(args: &Args, api: &Api, child: &Shutdown) -> Result<()> {
    if api.probe(Duration::from_secs(3)).is_ok() {
        return Ok(());
    }
    if args.no_tunnel {
        bail!("服务无法访问；请确认 --url 和远端服务状态");
    }
    shutdown(child);
    let result = open_tunnel(args, api, child);
    if result.is_err() {
        shutdown(child);
    }
    result
}

fn open_tunnel(args: &Args, api: &Api, child: &Shutdown) -> Result<()> {
    let url = reqwest::Url::parse(&args.url)?;
    if url.scheme() != "http" || !matches!(url.host_str(), Some("127.0.0.1" | "localhost")) {
        bail!("服务无法访问；自动 SSH 转发需要 http://127.0.0.1:端口");
    }
    let port = url.port_or_known_default().context("URL 缺少端口")?;
    // A bound port is evidence to inspect the existing service, not to start SSH over it.
    let probe = std::net::TcpListener::bind(("127.0.0.1", port)).with_context(|| {
        format!("本地端口 {port} 已占用，但不是可用的 LifeAI 服务；请检查服务或更换 --url 端口")
    })?;
    drop(probe);
    if args.ssh_host.is_empty()
        || args.ssh_host.starts_with('-')
        || !args
            .ssh_host
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || "._-:[]".contains(ch))
        || args.ssh_user.is_empty()
        || args.ssh_user.starts_with('-')
        || !args
            .ssh_user
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || "._-".contains(ch))
    {
        bail!("SSH 主机名或用户名无效");
    }
    let mut command = Command::new("ssh");
    command.args([
        "-N",
        "-T",
        "-p",
        &args.ssh_port.to_string(),
        "-o",
        "BatchMode=yes",
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=3",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-L",
        &format!("127.0.0.1:{port}:127.0.0.1:11435"),
    ]);
    if let Some(path) = &args.ssh_key {
        let path = expand(path);
        if !path.is_file() {
            bail!("SSH 私钥不存在: {}", path.display());
        }
        command
            .arg("-i")
            .arg(path)
            .args(["-o", "IdentitiesOnly=yes"]);
    }
    if let Some(path) = &args.known_hosts {
        command
            .arg("-o")
            .arg(format!("UserKnownHostsFile={}", expand(path).display()));
    }
    let mut log: File = tempfile::tempfile()?;
    command
        .arg(format!("{}@{}", args.ssh_user, args.ssh_host))
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(log.try_clone()?);
    *child.lock().unwrap() = Some(
        command
            .spawn()
            .context("无法启动 ssh，请安装 OpenSSH 客户端")?,
    );
    let deadline = Instant::now() + Duration::from_secs(30);
    while Instant::now() < deadline {
        let exited = child
            .lock()
            .unwrap()
            .as_mut()
            .unwrap()
            .try_wait()?
            .is_some();
        if exited {
            log.seek(SeekFrom::Start(0))?;
            let mut detail = String::new();
            log.take(4096).read_to_string(&mut detail)?;
            bail!(
                "SSH 连接失败；请检查 DSW 授权的密钥（--ssh-key）。{}",
                crate::clean(&detail)
            );
        }
        if api.probe(Duration::from_secs(1)).is_ok() {
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(150));
    }
    bail!("SSH 已启动，但模型服务不可访问；请确认 DSW 的 11435 服务正在运行")
}
