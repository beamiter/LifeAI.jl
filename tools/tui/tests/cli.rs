//! Executable-level contracts against a local HTTP service; no DSW or SSH required.
use serde_json::{Value, json};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::{Command, Output, Stdio};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};
use std::thread::{self, JoinHandle};
use std::time::Duration;

const QWEN: &str = "Qwen/Qwen3-8B";
const FLINT: &str = "RecursiveSI/flint";

#[derive(Clone, Debug)]
struct Request {
    method: String,
    path: String,
    body: Value,
}

#[derive(Clone, Copy)]
enum Reply {
    Stream,
    Json,
    Incomplete,
    Error,
}

struct Service {
    url: String,
    requests: Arc<Mutex<Vec<Request>>>,
    stop: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
}

impl Service {
    fn new(selected: &'static str, reply: Reply, cap: u32) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&requests);
        let stop = Arc::new(AtomicBool::new(false));
        let stopping = Arc::clone(&stop);
        let worker = thread::spawn(move || {
            let mut model = selected.to_string();
            while !stopping.load(Ordering::Relaxed) {
                let (mut stream, _) = match listener.accept() {
                    Ok(connection) => connection,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(2));
                        continue;
                    }
                    Err(error) => panic!("mock accept failed: {error}"),
                };
                stream
                    .set_read_timeout(Some(Duration::from_secs(3)))
                    .unwrap();
                let request = read_request(&mut stream);
                let (status, body, fragmented) = match request.path.as_str() {
                    "/healthz" => (200, json!({
                        "ready": true, "model": model, "selected_model": model,
                        "state": "ready", "phase": "ready", "context_tokens": 4096,
                        "max_new_tokens": cap,
                        "models": [
                            {"key":"qwen", "id":QWEN, "label":"Qwen3 8B", "context_tokens":4096,"max_new_tokens":cap},
                            {"key":"flint", "id":FLINT, "label":"Flint NVFP4", "context_tokens":4096,"max_new_tokens":cap}
                        ]
                    }).to_string(), false),
                    "/api/models" => (200, json!({"models":[
                        {"key":"qwen", "id":QWEN}, {"key":"flint", "id":FLINT}
                    ]}).to_string(), false),
                    "/api/select" => {
                        model = request.body["model"].as_str().unwrap().to_string();
                        (200, json!({"ready":true,"model":model}).to_string(), false)
                    }
                    "/api/chat" => match reply {
                        Reply::Stream => (200, format!("{}\n{}\n{}\n",
                            json!({"model":model,"response":"你","done":false}),
                            json!({"model":model,"response":"好🌟","done":false}),
                            json!({"model":model,"response":"","done":true,"done_reason":"stop","eval_count":3,"eval_duration":1000000000u64})), true),
                        Reply::Json => (200, json!({"model":model,"response":"你好🌟","done":true,"done_reason":"stop"}).to_string(), false),
                        Reply::Incomplete => (200, format!("{}\n", json!({"model":model,"response":"部分回答","done":false})), true),
                        Reply::Error => (400, json!({"error":"prompt tokens exceed context 4096"}).to_string(), false),
                    },
                    _ => (404, json!({"error":"unknown endpoint"}).to_string(), false),
                };
                captured.lock().unwrap().push(request);
                let content_type = if fragmented {
                    "application/x-ndjson"
                } else {
                    "application/json"
                };
                let head = format!(
                    "HTTP/1.1 {status} OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                if stream.write_all(head.as_bytes()).is_err() {
                    continue;
                }
                if fragmented {
                    // Split inside Chinese and emoji UTF-8 encodings, not just between records.
                    for byte in body.as_bytes() {
                        if stream.write_all(&[*byte]).is_err() {
                            break;
                        }
                    }
                } else {
                    let _ = stream.write_all(body.as_bytes());
                }
            }
        });
        Self {
            url,
            requests,
            stop,
            worker: Some(worker),
        }
    }

    fn command(&self) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_lifeai"));
        command.args([
            "--url",
            &self.url,
            "--no-tunnel",
            "--timeout",
            "3",
            "--load-timeout",
            "3",
        ]);
        command
            .env_remove("LIFEAI_MODEL")
            .env_remove("LIFEAI_URL")
            .env_remove("QWEN_URL")
            .stdin(Stdio::null());
        command
    }

    fn captured(&self) -> Vec<Request> {
        self.requests.lock().unwrap().clone()
    }
}

impl Drop for Service {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        self.worker.take().unwrap().join().unwrap();
    }
}

fn read_request(stream: &mut TcpStream) -> Request {
    let mut bytes = Vec::new();
    let header_end = loop {
        let mut byte = [0];
        stream.read_exact(&mut byte).unwrap();
        bytes.push(byte[0]);
        if bytes.ends_with(b"\r\n\r\n") {
            break bytes.len();
        }
        assert!(bytes.len() < 65536, "oversize mock header");
    };
    let headers = String::from_utf8(bytes).unwrap();
    let mut first = headers.lines().next().unwrap().split_whitespace();
    let method = first.next().unwrap().to_string();
    let path = first.next().unwrap().to_string();
    let length = headers
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().unwrap())
        })
        .unwrap_or(0);
    assert!(length <= 16 * 1024 * 1024);
    let mut body = vec![0; length];
    stream.read_exact(&mut body).unwrap();
    assert!(header_end > 0);
    Request {
        method,
        path,
        body: if body.is_empty() {
            Value::Null
        } else {
            serde_json::from_slice(&body).unwrap()
        },
    }
}

fn success(output: &Output) {
    assert!(
        output.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn help_uses_lumen_brand_and_explains_model_option() {
    let output = Command::new(env!("CARGO_BIN_EXE_lifeai"))
        .arg("--help")
        .output()
        .unwrap();
    success(&output);
    let text = String::from_utf8(output.stdout).unwrap();
    assert!(text.contains("Lumen"), "{text}");
    assert!(text.contains("--model"), "{text}");
    assert!(text.contains("--file"), "{text}");
}

#[test]
fn file_prompt_transports_over_one_mib_of_utf8() {
    let service = Service::new(FLINT, Reply::Stream, 512);
    let file = tempfile::NamedTempFile::new().unwrap();
    let content = "长上下文🌟".repeat(100_000);
    assert!(content.len() > 1024 * 1024);
    std::fs::write(file.path(), &content).unwrap();
    let output = service
        .command()
        .args(["--model", "flint", "--file"])
        .arg(file.path())
        .output()
        .unwrap();
    success(&output);
    let requests = service.captured();
    let request = requests.iter().find(|r| r.path == "/api/chat").unwrap();
    assert_eq!(request.body["messages"][0]["content"], content);
}

#[test]
fn oversized_or_non_utf8_file_fails_before_connecting() {
    let file = tempfile::NamedTempFile::new().unwrap();
    for bytes in [vec![b'x'; 4 * 1024 * 1024 + 1], vec![0xff, 0xfe]] {
        std::fs::write(file.path(), bytes).unwrap();
        let output = Command::new(env!("CARGO_BIN_EXE_lifeai"))
            .args(["--no-tunnel", "--url", "http://127.0.0.1:1", "--file"])
            .arg(file.path())
            .output()
            .unwrap();
        assert!(!output.status.success());
        let error = String::from_utf8_lossy(&output.stderr);
        assert!(
            error.contains("4 MiB") || error.contains("UTF-8"),
            "{error}"
        );
        assert!(!error.contains("无法连接"), "{error}");
    }
}

#[test]
fn status_reports_resident_model_without_selecting_default_model() {
    let service = Service::new(FLINT, Reply::Stream, 512);
    let output = service.command().arg("--status").output().unwrap();
    success(&output);
    let health: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(health["model"], FLINT);
    assert!(service.captured().iter().all(|r| r.method == "GET"));
}

#[test]
fn unicode_streaming_survives_fragmented_utf8_and_selects_flint() {
    let service = Service::new(QWEN, Reply::Stream, 512);
    let output = service
        .command()
        .args(["--model", "flint", "请用中文问候我"])
        .output()
        .unwrap();
    success(&output);
    assert!(String::from_utf8(output.stdout).unwrap().contains("你好🌟"));
    let captured = service.captured();
    let selection = captured
        .iter()
        .position(|r| r.path == "/api/select")
        .unwrap();
    let chat = captured.iter().position(|r| r.path == "/api/chat").unwrap();
    assert!(selection < chat);
    assert_eq!(captured[selection].body["model"], FLINT);
    assert_eq!(captured[chat].body["model"], FLINT);
    assert_eq!(
        captured[chat].body["messages"][0]["content"],
        "请用中文问候我"
    );
    assert_eq!(captured[chat].body["stream"], true);
}

#[test]
fn chat_entry_supports_no_stream_and_system_message() {
    let service = Service::new(QWEN, Reply::Json, 512);
    // Put the optional `chat` entry before flags, as documented for LifeAI.
    let mut command = Command::new(env!("CARGO_BIN_EXE_lifeai"));
    let output = command
        .args([
            "chat",
            "--url",
            &service.url,
            "--no-tunnel",
            "--no-stream",
            "--system",
            "你是中文助手",
            "-n",
            "32",
            "你好",
        ])
        .env_remove("LIFEAI_MODEL")
        .stdin(Stdio::null())
        .output()
        .unwrap();
    success(&output);
    assert!(String::from_utf8(output.stdout).unwrap().contains("你好🌟"));
    let requests = service.captured();
    let body = &requests
        .iter()
        .find(|r| r.path == "/api/chat")
        .unwrap()
        .body;
    assert_eq!(body["stream"], false);
    assert_eq!(
        body["messages"],
        json!([
            {"role":"system", "content":"你是中文助手"},
            {"role":"user", "content":"你好"}
        ])
    );
    assert_eq!(body["options"]["num_predict"], 32);
}

#[test]
fn incomplete_stream_returns_failed_exit() {
    let service = Service::new(QWEN, Reply::Incomplete, 512);
    let output = service.command().arg("hello").output().unwrap();
    assert!(!output.status.success());
    assert!(!output.stderr.is_empty());
}

#[test]
fn server_context_error_is_reported_with_failed_exit() {
    let service = Service::new(QWEN, Reply::Error, 512);
    let output = service.command().arg("hello").output().unwrap();
    assert!(!output.status.success());
    let detail = String::from_utf8(output.stderr).unwrap();
    assert!(detail.contains("context 4096"), "{detail}");
}

#[test]
fn output_above_service_limit_never_starts_generation() {
    let service = Service::new(QWEN, Reply::Stream, 16);
    let output = service
        .command()
        .args(["-n", "32", "hello"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(!service.captured().iter().any(|r| r.path == "/api/chat"));
}

#[test]
fn unsupported_model_is_rejected_before_any_connection() {
    let output = Command::new(env!("CARGO_BIN_EXE_lifeai"))
        .args(["--model", "unknown", "hello"])
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("unknown"));
}

#[test]
fn installer_provides_lifeai_and_lumen_and_backs_up_existing_launcher() {
    let directory = tempfile::tempdir().unwrap();
    let bin = directory.path().join("bin");
    std::fs::create_dir(&bin).unwrap();
    let previous = b"#!/bin/sh\necho previous-lifeai\n";
    std::fs::write(bin.join("lifeai"), previous).unwrap();
    let install = || {
        Command::new(env!("CARGO_BIN_EXE_lifeai"))
            .arg("--install")
            .arg("--prefix")
            .arg(directory.path())
            .stdin(Stdio::null())
            .output()
            .unwrap()
    };
    success(&install());
    assert_eq!(
        std::fs::read(bin.join("lifeai.before-lumen.0")).unwrap(),
        previous
    );
    assert!(bin.join("lifeai").is_file());
    assert!(bin.join("lumen").is_file());
    assert!(!bin.join("qwen").exists());
    let installed = std::fs::read(bin.join("lifeai")).unwrap();
    success(&install());
    assert_eq!(
        std::fs::read(bin.join("lifeai.before-lumen.1")).unwrap(),
        installed
    );
    assert!(!bin.join("qwen").exists());
    let output = Command::new(bin.join("lumen"))
        .arg("--help")
        .output()
        .unwrap();
    success(&output);
    assert!(String::from_utf8_lossy(&output.stdout).contains("Lumen"));
}

#[test]
fn retired_qwen_alias_option_is_not_accepted() {
    let directory = tempfile::tempdir().unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_lifeai"))
        .args(["--install", "--qwen-alias", "--prefix"])
        .arg(directory.path())
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("--qwen-alias"));
}

#[test]
fn explicit_install_prefix_does_not_require_home_environment() {
    let directory = tempfile::tempdir().unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_lifeai"))
        .arg("--install")
        .arg("--prefix")
        .arg(directory.path())
        .env_remove("HOME")
        .stdin(Stdio::null())
        .output()
        .unwrap();
    success(&output);
    assert!(directory.path().join("bin/lifeai").is_file());
    assert!(directory.path().join("bin/lumen").is_file());
    assert!(!directory.path().join("bin/qwen").exists());
}
