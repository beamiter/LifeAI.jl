//! Lumen (流明), LifeAI's native terminal client.
mod api;
mod tunnel;
mod ui;

use anyhow::{Context, Result, bail};
use api::{Api, Message, Model};
use clap::Parser;
use std::io::{self, IsTerminal, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

const DSW_HOST: &str = "dsw-notebook-dsw-l3wna78bngwh3pfple-22.vpc-0jlbz3pri2042fd5xw2ov.instance-forward.dsw.cn-wulanchabu.aliyuncs.com";

#[derive(Clone, Debug, Parser)]
#[command(
    name = "lifeai",
    version,
    about = "Lumen（流明） · LifeAI 的 Qwen / Flint 终端",
    after_help = "lifeai chat --model flint       打开 Lumen\nlifeai chat --model qwen '你好'  单次提问\ncat question.txt | lifeai chat   从标准输入提问\nlifeai --install                安装 lifeai 与 lumen 命令\n\nTUI: Ctrl+T / F2 切换 · Ctrl+L 清空 · Ctrl+R 重连 · Ctrl+C 退出"
)]
pub struct Args {
    /// 问题；省略时进入 TUI，也可通过标准输入提供
    #[arg(trailing_var_arg = true)]
    prompt: Vec<String>,
    /// qwen 或 flint
    #[arg(long, env = "LIFEAI_MODEL", default_value = "qwen")]
    model: Model,
    /// 服务根地址
    #[arg(long, env = "LIFEAI_URL", default_value = "http://127.0.0.1:11435")]
    url: String,
    /// 只使用已有连接，不自动建立 SSH 转发
    #[arg(long)]
    no_tunnel: bool,
    #[arg(long, env = "LIFEAI_SSH_HOST", default_value = DSW_HOST)]
    ssh_host: String,
    #[arg(long, default_value = "root")]
    ssh_user: String,
    #[arg(long, default_value = "22", value_parser = clap::value_parser!(u16).range(1..))]
    ssh_port: u16,
    #[arg(long, env = "LIFEAI_SSH_KEY")]
    ssh_key: Option<PathBuf>,
    #[arg(long)]
    known_hosts: Option<PathBuf>,
    /// 查看当前驻留服务状态，不加载或切换模型
    #[arg(long)]
    status: bool,
    #[arg(short = 'n', long, default_value = "256", value_parser = clap::value_parser!(u32).range(1..))]
    max_tokens: u32,
    /// 生成超时秒数
    #[arg(long, default_value = "300", value_parser = positive_seconds)]
    timeout: f64,
    /// 模型加载超时秒数
    #[arg(long, default_value = "1800", value_parser = positive_seconds)]
    load_timeout: f64,
    #[arg(long, default_value = "0", value_parser = temperature)]
    temperature: f64,
    #[arg(long, default_value = "")]
    system: String,
    #[arg(long)]
    no_stream: bool,
    /// 将本机二进制安装到 PREFIX/bin，提供 lifeai 和 lumen
    #[arg(long)]
    install: bool,
    #[arg(long)]
    prefix: Option<PathBuf>,
}

fn positive_seconds(value: &str) -> std::result::Result<f64, String> {
    let n = value.parse::<f64>().map_err(|_| "请输入秒数")?;
    if n.is_finite() && n > 0.0 && n <= 86_400.0 {
        Ok(n)
    } else {
        Err("秒数必须大于 0 且不超过 86400".into())
    }
}

fn temperature(value: &str) -> std::result::Result<f64, String> {
    let n = value.parse::<f64>().map_err(|_| "请输入温度")?;
    if n.is_finite() && (0.0..=2.0).contains(&n) {
        Ok(n)
    } else {
        Err("温度必须在 0～2 之间".into())
    }
}

// `chat` is an optional namespace, leaving the original prompt interface usable.
// `lifeai -- chat` sends the literal word "chat".
fn parse_args() -> Args {
    let mut argv: Vec<_> = std::env::args_os().collect();
    if argv.get(1).is_some_and(|arg| arg == "chat") {
        argv.remove(1);
    }
    Args::parse_from(argv)
}

fn clean(text: &str) -> String {
    text.chars()
        .filter_map(|ch| match ch {
            '\n' => Some(ch),
            '\t' => Some(' '),
            ch if !ch.is_control() => Some(ch),
            _ => None,
        })
        .collect()
}

fn main() {
    if let Err(error) = run(parse_args()) {
        eprintln!("Lumen: {}", clean(&format!("{error:#}")));
        std::process::exit(1);
    }
}

fn run(args: Args) -> Result<()> {
    if args.install {
        return install(args.prefix.as_deref());
    }
    let mut prompt = args.prompt.join(" ");
    if prompt.is_empty() && !args.status && !io::stdin().is_terminal() {
        io::stdin().take(1_048_577).read_to_string(&mut prompt)?;
        if prompt.len() > 1_048_576 {
            bail!("标准输入超过 1 MiB，请缩短问题");
        }
        prompt = prompt.trim().to_owned();
        if prompt.is_empty() {
            bail!("请输入问题，或在交互终端执行 lifeai chat");
        }
    }
    let is_tui = prompt.is_empty() && !args.status;
    if is_tui
        && (!io::stdin().is_terminal()
            || !io::stdout().is_terminal()
            || std::env::var("TERM").is_ok_and(|term| term == "dumb"))
    {
        bail!("Lumen 需要交互终端；也可执行 lifeai chat '你的问题'");
    }
    let api = Api::new(
        &args.url,
        Duration::from_secs_f64(args.timeout),
        args.max_tokens,
        args.temperature,
        args.no_stream,
    )?;
    let tunnel = tunnel::connect(&args, &api)?;
    if args.status {
        println!("{}", serde_json::to_string_pretty(&api.health()?)?);
    } else if !is_tui {
        api.select(
            args.model,
            Duration::from_secs_f64(args.load_timeout),
            |phase| {
                eprintln!("{}", clean(&phase));
            },
        )?;
        let mut messages = Vec::new();
        if !args.system.is_empty() {
            messages.push(Message {
                role: "system".into(),
                content: args.system,
            });
        }
        messages.push(Message {
            role: "user".into(),
            content: prompt,
        });
        let mut write_error = None;
        let result = api.generate(args.model, &messages, |piece| {
            let mut stdout = io::stdout().lock();
            if let Err(error) = write!(stdout, "{}", clean(&piece)).and_then(|_| stdout.flush()) {
                write_error = Some(error);
            }
        })?;
        if let Some(error) = write_error {
            return Err(error.into());
        }
        println!();
        if result.metadata["done_reason"] == "length" {
            eprintln!("达到输出上限；可用 -n 增加输出长度（以服务限制为准）。");
        }
    } else {
        let retry_api = api.clone();
        let retry_args = args.clone();
        let transport = tunnel.shutdown_handle();
        ui::run_with_connector(
            api,
            args.model,
            Duration::from_secs_f64(args.load_timeout),
            args.system,
            Box::new(move || tunnel::ensure_connection(&retry_args, &retry_api, &transport)),
        )?;
    }
    drop(tunnel);
    Ok(())
}

fn backup(path: &Path) -> Result<()> {
    if path.symlink_metadata().is_err() {
        return Ok(());
    }
    for index in 0..10_000 {
        let dest = path.with_file_name(format!(
            "{}.before-lumen.{index}",
            path.file_name().unwrap().to_string_lossy()
        ));
        if dest.symlink_metadata().is_err() {
            std::fs::rename(path, &dest)?;
            eprintln!("已备份: {}", dest.display());
            return Ok(());
        }
    }
    bail!("备份文件过多: {}", path.display())
}

fn install(prefix: Option<&Path>) -> Result<()> {
    let prefix = match prefix {
        Some(prefix) => prefix.to_owned(),
        None => std::env::var_os("HOME")
            .map(PathBuf::from)
            .context("HOME 未设置，请使用 --prefix")?
            .join(".local"),
    };
    let bin = prefix.join("bin");
    std::fs::create_dir_all(&bin)?;
    let source = std::env::current_exe()?;
    let target = bin.join("lifeai");
    if target.canonicalize().ok().as_ref() != Some(&source) {
        let temporary = tempfile::NamedTempFile::new_in(&bin)?;
        std::fs::copy(&source, temporary.path())?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(temporary.path(), std::fs::Permissions::from_mode(0o755))?;
        }
        backup(&target)?;
        temporary.persist(&target).map_err(|error| error.error)?;
    }
    for name in ["lumen"] {
        let alias = bin.join(name);
        if std::fs::read_link(&alias).ok().as_deref() == Some(Path::new("lifeai")) {
            continue;
        }
        backup(&alias)?;
        #[cfg(unix)]
        std::os::unix::fs::symlink("lifeai", &alias)?;
        #[cfg(not(unix))]
        std::fs::copy(&target, &alias)?;
    }
    println!(
        "已安装 Lumen: {}\n启动: lifeai chat --model flint，或 lumen --model flint",
        target.display()
    );
    if !std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default()).any(|path| path == bin)
    {
        println!("请将 {} 加入 PATH。", bin.display());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_invalid_numeric_options() {
        for value in ["0", "-1", "inf", "NaN", "86401"] {
            assert!(positive_seconds(value).is_err());
        }
        for value in ["-1", "inf", "NaN", "2.1"] {
            assert!(temperature(value).is_err());
        }
        assert_eq!(positive_seconds("0.5").unwrap(), 0.5);
    }
    #[test]
    fn never_emits_terminal_control_codes() {
        assert_eq!(clean("你好\x1b[2J\r\t\n"), "你好[2J \n");
    }
}
