//! LifeAI router protocol shared by the interactive and one-shot clients.
//!
//! Networking is blocking by design: callers run operations on their worker
//! thread and forward progress and text events to the terminal thread.

use anyhow::{Context, Result, bail, ensure};
use reqwest::blocking::{Client, Response};
use reqwest::{StatusCode, Url};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::fmt;
use std::io::{BufRead, BufReader, Read};
use std::str::FromStr;
use std::thread;
use std::time::{Duration, Instant};

const MAX_JSON: usize = 1024 * 1024;
const MAX_REPLY: usize = 16 * MAX_JSON;
const HEALTH_TIMEOUT: Duration = Duration::from_secs(3);
const POLL_INTERVAL: Duration = Duration::from_millis(500);

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Model {
    Qwen,
    Flint,
}

impl Model {
    pub fn key(self) -> &'static str {
        match self {
            Self::Qwen => "qwen",
            Self::Flint => "flint",
        }
    }

    pub fn id(self) -> &'static str {
        match self {
            Self::Qwen => "Qwen/Qwen3-8B",
            Self::Flint => "RecursiveSI/flint",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Qwen => "Qwen3-8B",
            Self::Flint => "Flint NVFP4",
        }
    }

    pub fn next(self) -> Self {
        match self {
            Self::Qwen => Self::Flint,
            Self::Flint => Self::Qwen,
        }
    }
}

impl fmt::Display for Model {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.key())
    }
}

impl FromStr for Model {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value {
            "qwen" | "Qwen/Qwen3-8B" => Ok(Self::Qwen),
            "flint" | "RecursiveSI/flint" => Ok(Self::Flint),
            _ => Err("模型可选 qwen 或 flint。".into()),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Debug)]
pub struct Generation {
    pub text: String,
    pub metadata: Value,
}

#[derive(Clone)]
pub struct Api {
    url: String,
    client: Client,
    timeout: Duration,
    max_tokens: u32,
    temperature: f64,
    no_stream: bool,
}

#[derive(Debug)]
struct HttpError {
    code: StatusCode,
    detail: String,
}

impl fmt::Display for HttpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.detail.to_ascii_lowercase().contains("context")
            || self.detail.to_ascii_lowercase().contains("prompt")
        {
            write!(
                f,
                "提示词与输出可能超过上下文；请缩短对话、降低 -n，或按 Ctrl+L 清空。HTTP {}: {}",
                self.code.as_u16(),
                self.detail
            )
        } else {
            write!(f, "服务返回 HTTP {}: {}", self.code.as_u16(), self.detail)
        }
    }
}

impl std::error::Error for HttpError {}

impl Api {
    pub fn new(
        url: &str,
        timeout: Duration,
        max_tokens: u32,
        temperature: f64,
        no_stream: bool,
    ) -> Result<Self> {
        let parsed = Url::parse(url).context("--url 需要有效的 HTTP 服务根地址。")?;
        ensure!(
            matches!(parsed.scheme(), "http" | "https")
                && parsed.host_str().is_some()
                && parsed.username().is_empty()
                && parsed.password().is_none()
                && parsed.path() == "/"
                && parsed.query().is_none()
                && parsed.fragment().is_none(),
            "--url 请填写服务根地址，例如 http://127.0.0.1:11435"
        );
        ensure!(!timeout.is_zero(), "--timeout 必须大于 0。");
        ensure!(max_tokens > 0, "--max-tokens 必须大于 0。");
        ensure!(
            temperature.is_finite() && (0.0..=2.0).contains(&temperature),
            "--temperature 必须在 0 到 2 之间。"
        );
        let client = Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .connect_timeout(Duration::from_secs(10))
            .timeout(timeout)
            .build()
            .context("无法创建 HTTP 客户端。")?;
        Ok(Self {
            url: parsed.as_str().trim_end_matches('/').to_owned(),
            client,
            timeout,
            max_tokens,
            temperature,
            no_stream,
        })
    }

    /// Short health check used before deciding whether to open an SSH tunnel.
    /// It validates the service identity, without selecting or loading a model.
    pub fn probe(&self, timeout: Duration) -> Result<Value> {
        let value = self.get_json("/healthz", timeout)?;
        validate_health(&value)?;
        Ok(value)
    }

    pub fn health(&self) -> Result<Value> {
        self.probe(HEALTH_TIMEOUT)
    }

    pub fn select(
        &self,
        model: Model,
        load_timeout: Duration,
        mut progress: impl FnMut(String),
    ) -> Result<Value> {
        ensure!(!load_timeout.is_zero(), "--load-timeout 必须大于 0。");
        let started = Instant::now();
        let mut health = self.probe(HEALTH_TIMEOUT.min(load_timeout))?;
        let mut requested = false;
        let mut last_progress = String::new();
        loop {
            let router = is_router(&health);
            ensure!(
                router || model == Model::Qwen,
                "当前是旧版 Qwen 服务，不支持 Flint；请启动 LifeAI 多模型服务。"
            );
            if is_ready(&health, model) {
                self.validate_limits(&health)?;
                return Ok(health);
            }
            if requested
                && selected(&health) == Some(model.id())
                && health["state"].as_str() == Some("error")
            {
                bail!(
                    "{} 加载失败：{}",
                    model.label(),
                    detail(&health["last_error"])
                );
            }
            let remaining = load_timeout.saturating_sub(started.elapsed());
            ensure!(
                !remaining.is_zero(),
                "等待 {} 加载超时；可增大 --load-timeout，或按 Ctrl+R 重试。",
                model.label()
            );
            let loading = health["state"].as_str() == Some("loading")
                || health["loading"].as_bool() == Some(true);
            let status = if router && !loading {
                match self.post_json(
                    "/api/select",
                    &json!({"model": model.id()}),
                    HEALTH_TIMEOUT.min(remaining),
                ) {
                    Ok(_) => {
                        requested = true;
                        format!("正在准备 {}，首次加载可能需要数分钟…", model.label())
                    }
                    Err(error) if is_busy(&error) => "服务正在处理其他请求，等待后切换模型…".into(),
                    Err(error) => return Err(error),
                }
            } else if loading {
                let target = health["loading_model"]
                    .as_str()
                    .or_else(|| selected(&health));
                requested |= target == Some(model.id());
                if target.is_some_and(|value| value != model.id()) {
                    format!(
                        "服务正在加载 {}；完成后切换到 {}…",
                        target.unwrap_or("模型"),
                        model.label()
                    )
                } else {
                    let phase = match health["phase"].as_str() {
                        Some("quantizing") => "GPU 量化",
                        Some("preparing_native") => "准备原生 FP4",
                        Some("warming") => "预热推理",
                        Some("loading_artifact") => "读取量化权重",
                        _ => "加载权重",
                    };
                    let elapsed = health["load_seconds"]
                        .as_f64()
                        .filter(|x| x.is_finite() && *x >= 0.0);
                    format!(
                        "{} 加载中 · {}{}",
                        model.label(),
                        phase,
                        elapsed.map(|x| format!(" · {:.0}s", x)).unwrap_or_default()
                    )
                }
            } else {
                format!("等待 {} 服务就绪…", model.label())
            };
            emit_progress(&mut progress, &mut last_progress, status);
            thread::sleep(POLL_INTERVAL.min(remaining));
            let remaining = load_timeout.saturating_sub(started.elapsed());
            ensure!(
                !remaining.is_zero(),
                "等待 {} 加载超时；可增大 --load-timeout。",
                model.label()
            );
            loop {
                let remaining = load_timeout.saturating_sub(started.elapsed());
                ensure!(
                    !remaining.is_zero(),
                    "等待 {} 加载超时；可增大 --load-timeout。",
                    model.label()
                );
                match self.probe(HEALTH_TIMEOUT.min(remaining)) {
                    Ok(value) => {
                        health = value;
                        break;
                    }
                    Err(error) if is_transport(&error) => {
                        emit_progress(
                            &mut progress,
                            &mut last_progress,
                            "等待模型服务恢复连接…".into(),
                        );
                        // Never select a model based on stale health after a
                        // transport failure; wait for a fresh response first.
                        thread::sleep(POLL_INTERVAL.min(remaining));
                    }
                    Err(error) => return Err(error),
                }
            }
        }
    }

    pub fn generate(
        &self,
        model: Model,
        messages: &[Message],
        mut emit: impl FnMut(String),
    ) -> Result<Generation> {
        let health = self.health()?;
        ensure!(
            is_ready(&health, model),
            "{} 尚未就绪；按 Ctrl+R 连接或重新选择模型。",
            model.label()
        );
        let (context, _) = self.validate_limits(&health)?;
        ensure!(
            !messages.is_empty() && messages.len() <= 128,
            "对话需包含 1 到 128 条消息；按 Ctrl+L 开始新对话。"
        );
        ensure!(
            messages
                .iter()
                .all(|m| matches!(m.role.as_str(), "system" | "user" | "assistant")),
            "消息角色必须为 system、user 或 assistant。"
        );
        if model == Model::Qwen {
            ensure!(
                self.temperature == 0.0,
                "当前 Qwen 服务只支持 --temperature 0。"
            );
        }
        let (endpoint, payload) = if is_router(&health) {
            (
                "/api/chat",
                json!({
                    "model":model.id(), "messages":messages, "stream":!self.no_stream,
                    "options":{"num_predict":self.max_tokens,"temperature":self.temperature}
                }),
            )
        } else {
            ensure!(model == Model::Qwen, "旧版服务不支持 Flint。");
            (
                "/api/generate",
                json!({
                    "model":model.id(),"prompt":qwen_prompt(messages),"raw":true,"think":false,
                    "stream":!self.no_stream,
                    "options":{"num_predict":self.max_tokens,"temperature":0,
                        "repeat_penalty":1,"repeat_last_n":0,"num_ctx":context}
                }),
            )
        };
        ensure!(
            serde_json::to_vec(&payload)?.len() <= MAX_JSON,
            "对话超过服务 1 MiB 限制；请缩短内容或按 Ctrl+L 清空。"
        );
        let response = self
            .client
            .post(format!("{}{endpoint}", self.url))
            .json(&payload)
            .timeout(self.timeout)
            .send()
            .context("生成请求失败。")?;
        let response = check_status(response)?;
        let mut text = String::new();
        if self.no_stream {
            let event = parse_json(response, MAX_REPLY)?;
            consume_event(&event, model, &mut text, &mut emit)?;
            ensure!(
                event["done"].as_bool() == Some(true),
                "连接中断，服务没有返回完整回答；本轮未写入对话历史。"
            );
            return Ok(Generation {
                text,
                metadata: event,
            });
        }
        let mut reader = BufReader::new(response);
        let mut total = 0usize;
        while let Some(record) = read_record(&mut reader, MAX_JSON)? {
            total = total.saturating_add(record.len());
            ensure!(total <= MAX_REPLY, "服务生成响应超过 16 MiB 限制。");
            if record.iter().all(u8::is_ascii_whitespace) {
                continue;
            }
            let event: Value =
                serde_json::from_slice(&record).context("服务返回了无效的 NDJSON 生成事件。")?;
            consume_event(&event, model, &mut text, &mut emit)?;
            if event["done"].as_bool() == Some(true) {
                return Ok(Generation {
                    text,
                    metadata: event,
                });
            }
        }
        bail!("连接中断，服务没有返回完整回答；本轮未写入对话历史。")
    }

    fn validate_limits(&self, health: &Value) -> Result<(u64, u64)> {
        let limits = health.get("limits").unwrap_or(health);
        let context = limits["context_tokens"]
            .as_u64()
            .filter(|x| *x > 0)
            .context("服务缺少有效 context_tokens，无法确认生成限制。")?;
        let maximum = limits["max_new_tokens"]
            .as_u64()
            .filter(|x| *x > 0)
            .context("服务缺少有效 max_new_tokens，无法确认生成限制。")?;
        ensure!(maximum < context, "服务输出上限必须小于上下文限制。");
        ensure!(
            u64::from(self.max_tokens) <= maximum,
            "--max-tokens={} 超过服务输出上限 {}（上下文 {} tokens）。",
            self.max_tokens,
            maximum,
            context
        );
        ensure!(
            u64::from(self.max_tokens) < context,
            "请求输出没有给提示词留下空间；请降低 --max-tokens。"
        );
        Ok((context, maximum))
    }

    fn get_json(&self, path: &str, timeout: Duration) -> Result<Value> {
        let response = self
            .client
            .get(format!("{}{path}", self.url))
            .timeout(timeout)
            .send()
            .context("无法连接 LifeAI 服务。")?;
        parse_json(check_status(response)?, MAX_JSON)
    }

    fn post_json(&self, path: &str, value: &Value, timeout: Duration) -> Result<Value> {
        let response = self
            .client
            .post(format!("{}{path}", self.url))
            .json(value)
            .timeout(timeout)
            .send()
            .context("无法连接 LifeAI 服务。")?;
        let result = parse_json(check_status(response)?, MAX_JSON)?;
        if let Some(error) = result.get("error") {
            bail!("{}", detail(error));
        }
        Ok(result)
    }
}

fn is_busy(error: &anyhow::Error) -> bool {
    error
        .downcast_ref::<HttpError>()
        .is_some_and(|e| e.code == StatusCode::CONFLICT)
}

fn is_transport(error: &anyhow::Error) -> bool {
    error
        .downcast_ref::<reqwest::Error>()
        .is_some_and(|e| e.is_connect() || e.is_timeout() || e.is_request())
}

fn emit_progress(emit: &mut impl FnMut(String), previous: &mut String, current: String) {
    if current != *previous {
        *previous = current.clone();
        emit(current);
    }
}

fn is_router(health: &Value) -> bool {
    health.get("models").is_some()
}

fn selected(health: &Value) -> Option<&str> {
    let value = health
        .get("model")
        .filter(|v| !v.is_null())
        .or_else(|| health.get("selected_model"))
        .or_else(|| health.get("selected"))?;
    let name = value
        .as_str()
        .or_else(|| value["id"].as_str())
        .or_else(|| value["model"].as_str())?;
    match name {
        "qwen" => Some(Model::Qwen.id()),
        "flint" => Some(Model::Flint.id()),
        name => Some(name),
    }
}

fn is_ready(health: &Value, model: Model) -> bool {
    health["ready"].as_bool() == Some(true) && selected(health) == Some(model.id())
}

fn validate_health(health: &Value) -> Result<()> {
    ensure!(health.is_object(), "服务 /healthz 响应必须为 JSON 对象。");
    if let Some(model) = selected(health) {
        ensure!(
            matches!(model, "Qwen/Qwen3-8B" | "RecursiveSI/flint"),
            "该端口不是兼容的 LifeAI 模型服务；请检查 --url。"
        );
    } else {
        ensure!(
            health["model"].is_null()
                && health["selected_model"].is_null()
                && health["selected"].is_null(),
            "服务 /healthz 模型标识无效。"
        );
        ensure!(
            health["ready"].as_bool() != Some(true),
            "服务已就绪但没有发布模型标识。"
        );
    }
    if is_router(health) {
        let models = health["models"]
            .as_array()
            .context("服务模型列表格式无效。")?;
        ensure!(
            models.iter().any(|v| matches!(
                v["id"].as_str(),
                Some("Qwen/Qwen3-8B" | "RecursiveSI/flint")
            )),
            "服务没有发布兼容的 LifeAI 模型。"
        );
    } else {
        ensure!(
            selected(health) == Some(Model::Qwen.id()),
            "该端口不是兼容的 LifeAI Qwen 服务；请检查 --url。"
        );
    }
    Ok(())
}

fn check_status(mut response: Response) -> Result<Response> {
    let code = response.status();
    if code.is_success() {
        return Ok(response);
    }
    let mut body = Vec::new();
    response
        .by_ref()
        .take(MAX_JSON as u64)
        .read_to_end(&mut body)?;
    let description = serde_json::from_slice::<Value>(&body)
        .ok()
        .map(|value| detail(value.get("error").unwrap_or(&value)))
        .unwrap_or_else(|| String::from_utf8_lossy(&body).trim().to_owned());
    Err(HttpError {
        code,
        detail: if description.is_empty() {
            code.to_string()
        } else {
            description
        },
    }
    .into())
}

fn detail(value: &Value) -> String {
    value
        .as_str()
        .map(str::to_owned)
        .or_else(|| {
            value
                .get("message")
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
        .unwrap_or_else(|| value.to_string())
}

fn parse_json(mut response: Response, limit: usize) -> Result<Value> {
    let mut body = Vec::new();
    response
        .by_ref()
        .take(limit as u64 + 1)
        .read_to_end(&mut body)
        .context("读取服务响应失败。")?;
    ensure!(body.len() <= limit, "服务 JSON 响应超过允许大小。");
    let value: Value = serde_json::from_slice(&body).context("服务返回了无效 JSON。")?;
    ensure!(value.is_object(), "服务返回的 JSON 必须为对象。");
    Ok(value)
}

/// Read one NDJSON record while bounding allocation even without a newline.
fn read_record(reader: &mut impl BufRead, limit: usize) -> Result<Option<Vec<u8>>> {
    let mut result = Vec::new();
    loop {
        let buffer = reader.fill_buf().context("读取流式响应失败。")?;
        if buffer.is_empty() {
            return Ok((!result.is_empty()).then_some(result));
        }
        let end = buffer.iter().position(|b| *b == b'\n').map(|x| x + 1);
        let used = end.unwrap_or(buffer.len());
        ensure!(
            result.len().saturating_add(used) <= limit,
            "服务生成事件超过 1 MiB 限制。"
        );
        result.extend_from_slice(&buffer[..used]);
        reader.consume(used);
        if end.is_some() {
            return Ok(Some(result));
        }
    }
}

fn consume_event(
    event: &Value,
    model: Model,
    text: &mut String,
    emit: &mut impl FnMut(String),
) -> Result<()> {
    ensure!(event.is_object(), "服务返回了无效的生成事件。");
    if let Some(error) = event.get("error") {
        bail!("{}", detail(error));
    }
    if let Some(id) = event.get("model") {
        ensure!(
            id.as_str() == Some(model.id()) || id.as_str() == Some(model.key()),
            "服务响应来自其他模型；本轮未写入对话历史。"
        );
    }
    if let Some(done) = event.get("done") {
        ensure!(done.is_boolean(), "生成事件的 done 字段必须为布尔值。");
    }
    if let Some(piece) = event.get("response") {
        let piece = piece.as_str().context("服务返回了无效的文本片段。")?;
        ensure!(
            text.len().saturating_add(piece.len()) <= MAX_REPLY,
            "服务生成文本超过 16 MiB 限制。"
        );
        if !piece.is_empty() {
            text.push_str(piece);
            emit(piece.to_owned());
        }
    }
    Ok(())
}

fn qwen_prompt(messages: &[Message]) -> String {
    let mut result = String::new();
    for message in messages {
        result.push_str("<|im_start|>");
        result.push_str(&message.role);
        result.push('\n');
        result.push_str(&message.content);
        result.push_str("<|im_end|>\n");
    }
    result.push_str("<|im_start|>assistant\n<think>\n\n</think>\n\n");
    result
}

pub fn assistant_history(model: Model, text: &str) -> String {
    if model == Model::Flint
        && text.starts_with("<think>")
        && let Some((_, answer)) = text.split_once("</think>")
    {
        return answer.trim_start().to_owned();
    }
    text.to_owned()
}

pub fn generation_rate(metadata: &Value) -> Option<f64> {
    metadata["lifeai"]["decode_tokens_per_second"]
        .as_f64()
        .filter(|x| x.is_finite() && *x >= 0.0)
        .or_else(|| {
            let count = metadata["eval_count"].as_u64()? as f64;
            let duration = metadata["eval_duration"]
                .as_f64()
                .filter(|x| x.is_finite() && *x > 0.0)?;
            Some(count * 1e9 / duration)
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Cursor, Write};
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};

    struct Mock {
        url: String,
        requests: Arc<Mutex<Vec<(String, Value)>>>,
        thread: Option<thread::JoinHandle<()>>,
    }

    impl Mock {
        fn new(responses: Vec<(u16, String, usize)>) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let url = format!("http://{}", listener.local_addr().unwrap());
            let requests = Arc::new(Mutex::new(Vec::new()));
            let saved = Arc::clone(&requests);
            let thread = thread::spawn(move || {
                for (status, body, chunk_size) in responses {
                    let (mut socket, _) = listener.accept().unwrap();
                    socket
                        .set_read_timeout(Some(Duration::from_secs(5)))
                        .unwrap();
                    let mut reader = BufReader::new(socket.try_clone().unwrap());
                    let mut first = String::new();
                    reader.read_line(&mut first).unwrap();
                    let mut len = 0usize;
                    loop {
                        let mut line = String::new();
                        reader.read_line(&mut line).unwrap();
                        if line == "\r\n" {
                            break;
                        }
                        if let Some((_, value)) =
                            line.to_ascii_lowercase().split_once("content-length:")
                        {
                            len = value.trim().parse().unwrap();
                        }
                    }
                    let mut request_body = vec![0; len];
                    reader.read_exact(&mut request_body).unwrap();
                    saved.lock().unwrap().push((
                        first.trim().to_owned(),
                        serde_json::from_slice(&request_body).unwrap_or(Value::Null),
                    ));
                    write!(socket, "HTTP/1.1 {status} test\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len()).unwrap();
                    for chunk in body.as_bytes().chunks(chunk_size.max(1)) {
                        if socket.write_all(chunk).is_err() {
                            break;
                        }
                        socket.flush().ok();
                    }
                }
            });
            Self {
                url,
                requests,
                thread: Some(thread),
            }
        }

        fn api(&self, no_stream: bool, max: u32) -> Api {
            Api::new(&self.url, Duration::from_secs(5), max, 0.0, no_stream).unwrap()
        }

        fn finish(mut self) -> Vec<(String, Value)> {
            self.thread.take().unwrap().join().unwrap();
            Arc::try_unwrap(self.requests)
                .unwrap()
                .into_inner()
                .unwrap()
        }
    }

    fn health(model: Model) -> Value {
        json!({"ready":true,"model":model.id(),"state":"ready","context_tokens":2048,
            "max_new_tokens":512,"models":[{"key":"qwen","id":Model::Qwen.id()},{"key":"flint","id":Model::Flint.id()}]})
    }

    fn response(value: Value) -> (u16, String, usize) {
        (200, value.to_string(), 4096)
    }

    fn messages() -> Vec<Message> {
        vec![Message {
            role: "user".into(),
            content: "你好，世界 👋".into(),
        }]
    }

    #[test]
    fn validates_url_and_generation_options() {
        for url in [
            "ftp://example.com",
            "http://user@example.com",
            "http://localhost/path",
            "http://localhost/?x=1",
            "http://localhost/#x",
            "xyz",
        ] {
            assert!(
                Api::new(url, Duration::from_secs(1), 1, 0.0, false).is_err(),
                "{url}"
            );
        }
        assert!(
            Api::new(
                "http://localhost:11435/",
                Duration::from_secs(1),
                1,
                0.0,
                false
            )
            .is_ok()
        );
        assert!(Api::new("http://localhost", Duration::ZERO, 1, 0.0, false).is_err());
        assert!(Api::new("http://localhost", Duration::from_secs(1), 0, 0.0, false).is_err());
        for value in [f64::NAN, f64::INFINITY, -0.1, 2.1] {
            assert!(Api::new("http://localhost", Duration::from_secs(1), 1, value, false).is_err());
        }
    }

    #[test]
    fn real_http_stream_reassembles_split_utf8_and_sends_chat_messages() {
        let body = [
            json!({"model":Model::Flint.id(),"response":"你好","done":false}),
            json!({"response":"，世界 👋","done":false}),
            json!({"response":"！","done":true,"eval_count":3,"eval_duration":1_000_000_000u64}),
        ]
        .iter()
        .map(Value::to_string)
        .collect::<Vec<_>>()
        .join("\n");
        let mock = Mock::new(vec![response(health(Model::Flint)), (200, body, 1)]);
        let mut emitted = String::new();
        let generated = mock
            .api(false, 256)
            .generate(Model::Flint, &messages(), |piece| emitted.push_str(&piece))
            .unwrap();
        assert_eq!(generated.text, "你好，世界 👋！");
        assert_eq!(generated.text, emitted);
        assert_eq!(generation_rate(&generated.metadata), Some(3.0));
        let requests = mock.finish();
        assert_eq!(requests[1].0, "POST /api/chat HTTP/1.1");
        assert_eq!(requests[1].1["messages"][0]["content"], "你好，世界 👋");
        assert_eq!(requests[1].1["options"]["num_predict"], 256);
        assert!(requests[1].1.get("prompt").is_none());
    }

    #[test]
    fn incomplete_stream_errors_after_partial_text() {
        let mock = Mock::new(vec![
            response(health(Model::Flint)),
            response(json!({"response":"partial","done":false})),
        ]);
        let mut emitted = String::new();
        let error = mock
            .api(false, 16)
            .generate(Model::Flint, &messages(), |piece| emitted.push_str(&piece))
            .unwrap_err();
        assert_eq!(emitted, "partial");
        assert!(error.to_string().contains("未写入"));
        mock.finish();
    }

    #[test]
    fn rejects_stream_error_invalid_text_wrong_model_and_bad_done() {
        for event in [
            json!({"error":{"message":"GPU failed"}}),
            json!({"response":12,"done":true}),
            json!({"model":Model::Qwen.id(),"response":"wrong","done":true}),
            json!({"done":"true"}),
        ] {
            let mock = Mock::new(vec![response(health(Model::Flint)), response(event)]);
            assert!(
                mock.api(false, 16)
                    .generate(Model::Flint, &messages(), |_| {})
                    .is_err()
            );
            mock.finish();
        }
    }

    #[test]
    fn handles_nonstream_and_requires_done() {
        for done in [true, false] {
            let mock = Mock::new(vec![
                response(health(Model::Flint)),
                response(json!({"response":"回答","done":done})),
            ]);
            let result = mock
                .api(true, 16)
                .generate(Model::Flint, &messages(), |_| {});
            assert_eq!(result.is_ok(), done);
            let requests = mock.finish();
            assert_eq!(requests[1].1["stream"], false);
        }
    }

    #[test]
    fn no_request_when_limits_or_selected_model_are_wrong() {
        for value in [
            health(Model::Qwen),
            {
                let mut h = health(Model::Flint);
                h["max_new_tokens"] = json!(8);
                h
            },
            {
                let mut h = health(Model::Flint);
                h["context_tokens"] = json!(true);
                h
            },
            {
                let mut h = health(Model::Flint);
                h["ready"] = json!(false);
                h
            },
        ] {
            let mock = Mock::new(vec![response(value)]);
            assert!(
                mock.api(false, 16)
                    .generate(Model::Flint, &messages(), |_| {})
                    .is_err()
            );
            assert_eq!(mock.finish().len(), 1);
        }
    }

    #[test]
    fn rejects_unrelated_service_and_malformed_health() {
        for value in [
            json!([]),
            json!({"ready":true,"model":"other"}),
            json!({"ready":true,"model":42}),
            json!({"ready":true,"model":null,"models":[]}),
            json!({"ready":true,"model":Model::Flint.id()}),
        ] {
            let mock = Mock::new(vec![response(value)]);
            assert!(mock.api(false, 16).health().is_err());
            mock.finish();
        }
    }

    #[test]
    fn selecting_ready_model_never_reloads_worker() {
        let mock = Mock::new(vec![response(health(Model::Flint))]);
        let ready = mock
            .api(false, 16)
            .select(Model::Flint, Duration::from_secs(2), |_| {})
            .unwrap();
        assert_eq!(ready["ready"], true);
        assert_eq!(mock.finish().len(), 1);
    }

    #[test]
    fn selection_waits_on_busy_then_loading_and_validates_model() {
        let mut loading = health(Model::Flint);
        loading["ready"] = json!(false);
        loading["state"] = json!("loading");
        loading["phase"] = json!("warming");
        loading["loading_model"] = json!(Model::Flint.id());
        loading["load_seconds"] = json!(15.0);
        let mock = Mock::new(vec![
            response(health(Model::Qwen)),
            (409, json!({"error":"busy"}).to_string(), 4096),
            response(health(Model::Qwen)),
            (202, loading.to_string(), 4096),
            response(loading),
            response(health(Model::Flint)),
        ]);
        let mut events = Vec::new();
        mock.api(false, 16)
            .select(Model::Flint, Duration::from_secs(6), |s| events.push(s))
            .unwrap();
        assert!(events.iter().any(|s| s.contains("其他请求")));
        assert!(events.iter().any(|s| s.contains("预热")));
        let requests = mock.finish();
        assert_eq!(
            requests
                .iter()
                .filter(|r| r.0.starts_with("POST /api/select"))
                .count(),
            2
        );
        assert_eq!(requests[1].1["model"], Model::Flint.id());
    }

    #[test]
    fn reports_load_failure_without_restart_loop() {
        let mut failed = health(Model::Flint);
        failed["ready"] = json!(false);
        failed["state"] = json!("error");
        failed["last_error"] = json!("weights missing");
        let mock = Mock::new(vec![
            response(health(Model::Qwen)),
            (202, "{}".into(), 4096),
            response(failed),
        ]);
        let error = mock
            .api(false, 16)
            .select(Model::Flint, Duration::from_secs(3), |_| {})
            .unwrap_err();
        assert!(error.to_string().contains("weights missing"));
        assert_eq!(mock.finish().len(), 3);
    }

    #[test]
    fn legacy_qwen_template_and_flint_rejection() {
        let mut legacy = health(Model::Qwen);
        legacy.as_object_mut().unwrap().remove("models");
        let mock = Mock::new(vec![
            response(legacy.clone()),
            response(json!({"response":"hi","done":true})),
        ]);
        assert_eq!(
            mock.api(false, 16)
                .generate(Model::Qwen, &messages(), |_| {})
                .unwrap()
                .text,
            "hi"
        );
        let requests = mock.finish();
        assert_eq!(requests[1].0, "POST /api/generate HTTP/1.1");
        assert_eq!(
            requests[1].1["prompt"],
            "<|im_start|>user\n你好，世界 👋<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
        let mock = Mock::new(vec![response(legacy)]);
        assert!(
            mock.api(false, 16)
                .select(Model::Flint, Duration::from_secs(2), |_| {})
                .unwrap_err()
                .to_string()
                .contains("旧版")
        );
        mock.finish();
    }

    #[test]
    fn http_errors_keep_service_detail() {
        let mock = Mock::new(vec![
            response(health(Model::Flint)),
            (
                400,
                json!({"error":{"message":"context too long"}}).to_string(),
                4096,
            ),
        ]);
        let error = mock
            .api(false, 16)
            .generate(Model::Flint, &messages(), |_| {})
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("HTTP 400")
                && error.contains("context too long")
                && error.contains("Ctrl+L")
        );
        mock.finish();
    }

    #[test]
    fn bounded_record_reader_and_json_validation() {
        assert!(read_record(&mut Cursor::new(vec![b'x'; 100]), 32).is_err());
        assert_eq!(
            read_record(&mut Cursor::new(b"hello\nworld"), 32)
                .unwrap()
                .unwrap(),
            b"hello\n"
        );
        let mut reader = Cursor::new(b"last");
        assert_eq!(read_record(&mut reader, 32).unwrap().unwrap(), b"last");
        assert!(read_record(&mut reader, 32).unwrap().is_none());
    }

    #[test]
    fn reasoning_history_and_speed_metrics() {
        assert_eq!(
            assistant_history(Model::Flint, "<think>推理</think>\n答案"),
            "答案"
        );
        assert_eq!(
            assistant_history(Model::Flint, "<think>unfinished"),
            "<think>unfinished"
        );
        assert_eq!(
            assistant_history(Model::Qwen, "<think>x</think>answer"),
            "<think>x</think>answer"
        );
        assert_eq!(
            generation_rate(
                &json!({"lifeai":{"decode_tokens_per_second":7.5},"eval_count":5,"eval_duration":1e9})
            ),
            Some(7.5)
        );
        assert_eq!(
            generation_rate(&json!({"eval_count":5,"eval_duration":0})),
            None
        );
        assert_eq!(
            generation_rate(&json!({"eval_count":5,"eval_duration":2e9})),
            Some(2.5)
        );
        assert_eq!("flint".parse::<Model>().unwrap().next(), Model::Qwen);
        assert!("invalid".parse::<Model>().is_err());
    }
}
