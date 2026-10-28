//! Lumen's terminal interface. Network I/O lives on one worker; rendering and
//! editing never wait for a model to load or a token to arrive.
use std::{
    io::{self, Stdout},
    sync::mpsc::{self, Receiver, Sender},
    thread,
    time::{Duration, Instant},
};

use anyhow::{Context, Result};
use crossterm::{
    event::{
        self, DisableBracketedPaste, EnableBracketedPaste, Event, KeyCode, KeyEvent, KeyEventKind,
        KeyModifiers,
    },
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{
    Frame, Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Tabs},
};
use serde_json::Value;
use unicode_segmentation::{GraphemeCursor, UnicodeSegmentation};
use unicode_width::UnicodeWidthStr;

use crate::api::{
    Api, Generation, MAX_INPUT_BYTES, Message, Model, assistant_history, generation_rate,
};

const BACKGROUND: Color = Color::Rgb(11, 18, 25);
const PANEL: Color = Color::Rgb(16, 26, 35);
const TEXT: Color = Color::Rgb(220, 230, 237);
const MUTED: Color = Color::Rgb(130, 151, 165);
const ACCENT: Color = Color::Rgb(83, 213, 206);
const ERROR: Color = Color::Rgb(250, 134, 128);

fn index(model: Model) -> usize {
    match model {
        Model::Qwen => 0,
        Model::Flint => 1,
    }
}

/// Remove terminal controls, preserving plain text, newlines and graphemes.
pub fn clean_text(text: &str) -> String {
    text.chars()
        .filter_map(|c| match c {
            '\n' => Some(c),
            '\t' => Some(' '),
            c if c.is_control() => None,
            c => Some(c),
        })
        .collect()
}

#[derive(Default)]
struct Editor {
    text: String,
    cursor: usize, // UTF-8 byte offset, always at a grapheme boundary.
    revision: u64,
    cache: EditorCache,
}

#[derive(Default)]
struct EditorCache {
    revision: u64,
    width: usize,
    rows: Vec<String>,
    cursor_row: usize,
    cursor_column: usize,
}

impl Editor {
    fn changed(&mut self) {
        self.revision = self.revision.wrapping_add(1);
    }

    fn insert(&mut self, value: &str) -> bool {
        let value = clean_text(value);
        if self.text.len().saturating_add(value.len()) > MAX_INPUT_BYTES {
            return false;
        }
        self.text.insert_str(self.cursor, &value);
        self.cursor += value.len();
        // Pasting before an existing combining mark can merge graphemes.
        if self.cursor < self.text.len() {
            let mut boundary = GraphemeCursor::new(self.cursor, self.text.len(), true);
            if !boundary
                .is_boundary(&self.text, 0)
                .expect("complete UTF-8 draft")
            {
                self.cursor = boundary
                    .next_boundary(&self.text, 0)
                    .expect("complete UTF-8 draft")
                    .unwrap_or(self.text.len());
            }
        }
        self.changed();
        true
    }

    fn left(&mut self) {
        self.cursor = self.text[..self.cursor]
            .grapheme_indices(true)
            .next_back()
            .map_or(0, |(i, _)| i);
        self.changed();
    }

    fn right(&mut self) {
        self.cursor += self.text[self.cursor..]
            .graphemes(true)
            .next()
            .map_or(0, str::len);
        self.changed();
    }

    fn backspace(&mut self) {
        if self.cursor == 0 {
            return;
        }
        let end = self.cursor;
        self.left();
        self.text.replace_range(self.cursor..end, "");
        self.changed();
    }

    fn delete(&mut self) {
        let end = self.cursor
            + self.text[self.cursor..]
                .graphemes(true)
                .next()
                .map_or(0, str::len);
        self.text.replace_range(self.cursor..end, "");
        self.changed();
    }

    fn home(&mut self) {
        self.cursor = self.text[..self.cursor].rfind('\n').map_or(0, |i| i + 1);
        self.changed();
    }

    fn end(&mut self) {
        self.cursor += self.text[self.cursor..]
            .find('\n')
            .unwrap_or(self.text.len() - self.cursor);
        self.changed();
    }

    fn clear(&mut self) {
        self.text.clear();
        self.cursor = 0;
        self.changed();
    }

    fn take(&mut self) -> String {
        let text = std::mem::take(&mut self.text);
        self.cursor = 0;
        self.changed();
        text
    }

    fn layout(&mut self, width: usize) -> &EditorCache {
        let width = width.max(1);
        if self.cache.width == width
            && self.cache.revision == self.revision
            && !self.cache.rows.is_empty()
        {
            return &self.cache;
        }
        // Only lay out the cursor neighbourhood of large drafts. The composer
        // displays a few lines; wrapping a multi-megabyte file on each keypress
        // would otherwise block terminal input.
        let (start, end) = if self.text.len() > 64 * 1024 {
            let budget = width.saturating_mul(8);
            let mut cells = 0usize;
            let mut start = self.cursor;
            for (offset, grapheme) in self.text[..self.cursor].grapheme_indices(true).rev() {
                cells = cells.saturating_add(if grapheme == "\n" {
                    width
                } else {
                    grapheme.width().max(1)
                });
                start = offset;
                if cells >= budget {
                    break;
                }
            }
            cells = 0;
            let mut end = self.cursor;
            for (offset, grapheme) in self.text[self.cursor..].grapheme_indices(true) {
                cells = cells.saturating_add(if grapheme == "\n" {
                    width
                } else {
                    grapheme.width().max(1)
                });
                end = self.cursor + offset + grapheme.len();
                if cells >= budget {
                    break;
                }
            }
            (start, end)
        } else {
            (0, self.text.len())
        };
        let mut rows = vec![String::new()];
        let (mut column, mut cursor_row, mut cursor_column) = (0, 0, 0);
        for (offset, grapheme) in self.text[start..end].grapheme_indices(true) {
            if grapheme != "\n" {
                let size = grapheme.width();
                if column + size > width && column > 0 {
                    rows.push(String::new());
                    column = 0;
                }
            }
            if start + offset == self.cursor {
                cursor_row = rows.len() - 1;
                cursor_column = column;
            }
            if grapheme == "\n" {
                rows.push(String::new());
                column = 0;
            } else {
                rows.last_mut().unwrap().push_str(grapheme);
                column += grapheme.width();
            }
        }
        if self.cursor == end {
            if column >= width {
                rows.push(String::new());
                column = 0;
            }
            cursor_row = rows.len() - 1;
            cursor_column = column;
        }
        self.cache = EditorCache {
            revision: self.revision,
            width,
            rows,
            cursor_row,
            cursor_column,
        };
        &self.cache
    }
}

#[derive(Clone, Copy)]
enum Speaker {
    User,
    Assistant,
    Error,
}

struct Entry {
    speaker: Speaker,
    text: String,
    cached_width: usize,
    cached_rows: Vec<Line<'static>>,
    dirty: bool,
}

impl Entry {
    fn new(speaker: Speaker, text: String) -> Self {
        Self {
            speaker,
            text,
            cached_width: 0,
            cached_rows: Vec::new(),
            dirty: true,
        }
    }

    fn rows(&mut self, model: Model, width: usize) -> &[Line<'static>] {
        if !self.dirty && self.cached_width == width {
            return &self.cached_rows;
        }
        let (label, color) = match self.speaker {
            Speaker::User => ("你", ACCENT),
            Speaker::Assistant => (model.label(), TEXT),
            Speaker::Error => ("错误", ERROR),
        };
        let mut rows = vec![Line::styled(
            format!("{label}  ›"),
            Style::default().fg(color).add_modifier(Modifier::BOLD),
        )];
        for row in wrap_text(&clean_text(&self.text), width.max(1)) {
            rows.push(Line::styled(
                row,
                Style::default().fg(if matches!(self.speaker, Speaker::Error) {
                    ERROR
                } else {
                    TEXT
                }),
            ));
        }
        rows.push(Line::default());
        self.cached_rows = rows;
        self.cached_width = width;
        self.dirty = false;
        &self.cached_rows
    }
}

fn wrap_text(text: &str, width: usize) -> Vec<String> {
    let width = width.max(1);
    let mut rows = vec![String::new()];
    let mut column = 0;
    for grapheme in text.graphemes(true) {
        if grapheme == "\n" {
            rows.push(String::new());
            column = 0;
            continue;
        }
        let size = grapheme.width();
        if column + size > width && column > 0 {
            rows.push(String::new());
            column = 0;
        }
        rows.last_mut().unwrap().push_str(grapheme);
        column += size;
    }
    rows
}

struct Session {
    history: Vec<Message>,
    entries: Vec<Entry>,
    editor: Editor,
    scroll: usize, // rows above the end; zero follows streaming output.
    revision: u64,
    cache_revision: u64,
    cache_width: usize,
    dirty_from: usize,
    cached_ends: Vec<usize>,
    rows: Vec<Line<'static>>,
}

impl Session {
    fn new(system: &str) -> Self {
        let history = if system.is_empty() {
            Vec::new()
        } else {
            vec![Message {
                role: "system".into(),
                content: system.into(),
            }]
        };
        Self {
            history,
            entries: Vec::new(),
            editor: Editor::default(),
            scroll: 0,
            revision: 0,
            cache_revision: 0,
            cache_width: 0,
            dirty_from: 0,
            cached_ends: Vec::new(),
            rows: Vec::new(),
        }
    }

    fn append(&mut self, speaker: Speaker, text: String) {
        self.dirty_from = self.dirty_from.min(self.entries.len());
        self.entries.push(Entry::new(speaker, text));
        self.revision = self.revision.wrapping_add(1);
    }

    fn piece(&mut self, piece: &str) {
        self.dirty_from = self.dirty_from.min(self.entries.len().saturating_sub(1));
        if let Some(entry) = self.entries.last_mut() {
            entry.text.push_str(piece);
            entry.dirty = true;
            self.revision = self.revision.wrapping_add(1);
        }
    }

    fn finish_text(&mut self, text: String) {
        self.dirty_from = self.dirty_from.min(self.entries.len().saturating_sub(1));
        if let Some(entry) = self.entries.last_mut() {
            entry.text = text;
            entry.dirty = true;
            self.revision = self.revision.wrapping_add(1);
        }
    }

    fn rows(&mut self, model: Model, width: usize) -> &[Line<'static>] {
        if self.cache_width == width && self.cache_revision == self.revision {
            return &self.rows;
        }
        // Only the streaming tail changes during generation. Keep all previous
        // rows in place instead of cloning the entire conversation per token.
        let start = if self.cache_width == width {
            self.dirty_from.min(self.cached_ends.len())
        } else {
            0
        };
        self.rows.truncate(if start == 0 {
            0
        } else {
            self.cached_ends[start - 1]
        });
        self.cached_ends.truncate(start);
        for entry in self.entries.iter_mut().skip(start) {
            self.rows.extend_from_slice(entry.rows(model, width));
            self.cached_ends.push(self.rows.len());
        }
        if self.rows.is_empty() {
            self.rows = [
                "欢迎使用 Lumen。选择一个模型，开始对话。",
                "Qwen 与 Flint 分别保留会话和草稿。",
                "Ctrl+T / F2 切换模型，/help 查看快捷键。",
                "首次加载需要数分钟，等待时仍可编辑与滚动。",
            ]
            .into_iter()
            .flat_map(|text| wrap_text(text, width))
            .map(|text| Line::styled(text, Style::default().fg(MUTED)))
            .collect();
        }
        self.cache_width = width;
        self.cache_revision = self.revision;
        self.dirty_from = self.entries.len();
        &self.rows
    }
}

enum Job {
    Select(Model),
    Generate(Model, Vec<Message>),
}

enum Update {
    Status(String),
    Selected(Model, Value),
    Piece(Model, String),
    Done(Model, Vec<Message>, Generation),
    Failed(Model, bool, String),
}

#[derive(Clone, Copy)]
enum Busy {
    Loading(Model),
    Generating(Model),
}

struct App {
    active: Model,
    sessions: [Session; 2],
    system: String,
    ready: bool,
    busy: Option<Busy>,
    status: String,
    started: Instant,
    health: Option<Value>,
    jobs: Sender<Job>,
}

impl App {
    fn new(model: Model, system: String, jobs: Sender<Job>) -> Self {
        Self {
            active: model,
            sessions: [Session::new(&system), Session::new(&system)],
            system,
            ready: false,
            busy: None,
            status: "正在连接 LifeAI 服务…".into(),
            started: Instant::now(),
            health: None,
            jobs,
        }
    }

    fn session(&mut self) -> &mut Session {
        &mut self.sessions[index(self.active)]
    }

    fn insert_input(&mut self, text: &str) {
        if !self.session().editor.insert(text) {
            self.status = "草稿超过 4 MiB，请缩短输入。".into();
        }
    }

    fn select(&mut self, model: Model) {
        if self.busy.is_some() {
            self.status = "请等当前加载或回答完成后切换模型。".into();
            return;
        }
        self.busy = Some(Busy::Loading(model));
        self.started = Instant::now();
        self.status = format!("正在准备 {}，首次加载可能需要数分钟…", model.label());
        if self.jobs.send(Job::Select(model)).is_err() {
            self.busy = None;
            self.ready = false;
            self.status = "网络线程已退出，请重新启动 Lumen。".into();
        }
    }

    fn clear(&mut self) {
        if self.busy.is_some() {
            self.status = "请等当前加载或回答完成后清空历史。".into();
            return;
        }
        self.sessions[index(self.active)] = Session::new(&self.system);
        self.status = format!(
            "已清空 {} 的历史；另一模型的会话已保留。",
            self.active.label()
        );
    }

    /// Returns true only when the user requests to exit.
    fn submit(&mut self) -> bool {
        let command = self.session().editor.text.trim().to_string();
        if matches!(command.as_str(), "/quit" | "/exit") {
            return true;
        }
        if command.is_empty() {
            return false;
        }
        if self.busy.is_some() {
            self.status = "正在加载或回答，草稿已保留；就绪后按 Enter 发送。".into();
            return false;
        }
        match command.as_str() {
            "/clear" => {
                self.clear();
                return false;
            }
            "/help" | "/model" | "/models" => {
                self.session().editor.clear();
                self.status = "Ctrl+T/F2 切换 · Ctrl+L 清空 · Ctrl+R 重试 · Alt+Enter 换行 · PgUp/PgDn 滚动 · /load 路径 · /model qwen|flint · /quit".into();
                return false;
            }
            _ => {}
        }
        if command == "/load" || command.starts_with("/load ") {
            let path = command.strip_prefix("/load").unwrap().trim();
            if path.is_empty() {
                self.status = "用法：/load UTF-8文件路径（最多 4 MiB）。".into();
            } else {
                match crate::read_prompt_file(std::path::Path::new(path)) {
                    Ok(text) => {
                        let bytes = text.len();
                        self.session().editor.clear();
                        self.session().editor.insert(&text);
                        self.status =
                            format!("已导入 {bytes} 字节到草稿；Enter 发送，Alt+Enter 继续编辑。");
                    }
                    Err(error) => self.status = clean_text(&format!("{error:#}；草稿已保留。")),
                }
            }
            return false;
        }
        if let Some(value) = command.strip_prefix("/model ").or_else(|| {
            command
                .strip_prefix('/')
                .filter(|v| matches!(*v, "qwen" | "flint"))
        }) {
            match value.trim() {
                "qwen" => {
                    self.session().editor.clear();
                    self.select(Model::Qwen);
                }
                "flint" => {
                    self.session().editor.clear();
                    self.select(Model::Flint);
                }
                _ => self.status = "用法：/model qwen 或 /model flint。".into(),
            }
            return false;
        }
        if !self.ready {
            self.status = "模型尚未就绪；Ctrl+R 重试连接。草稿已保留。".into();
            return false;
        }
        let model = self.active;
        let session = self.session();
        let prompt = session.editor.take();
        let mut pending = session.history.clone();
        pending.push(Message {
            role: "user".into(),
            content: prompt.clone(),
        });
        session.append(Speaker::User, prompt);
        session.append(Speaker::Assistant, String::new());
        session.scroll = 0;
        self.busy = Some(Busy::Generating(model));
        self.started = Instant::now();
        self.status = format!("{} 正在生成…", model.label());
        if self.jobs.send(Job::Generate(model, pending)).is_err() {
            self.busy = None;
            self.ready = false;
            self.status = "网络线程已退出，请重新启动 Lumen。".into();
        }
        false
    }

    fn update(&mut self, update: Update) {
        match update {
            Update::Status(text) => self.status = clean_text(&text),
            Update::Selected(model, health) => {
                self.active = model;
                self.busy = None;
                self.ready = true;
                self.status = format!(
                    "{} 已就绪 · 上下文 {} tokens · 输出上限 {} · 历史 {} 条",
                    model.label(),
                    health["context_tokens"],
                    health["max_new_tokens"],
                    self.sessions[index(model)].history.len()
                );
                self.health = Some(health);
            }
            Update::Piece(model, piece) => {
                self.sessions[index(model)].piece(&piece);
                self.status = format!("{} 正在生成…", model.label());
            }
            Update::Done(model, mut pending, generation) => {
                pending.push(Message {
                    role: "assistant".into(),
                    content: assistant_history(model, &generation.text),
                });
                let session = &mut self.sessions[index(model)];
                session.history = pending;
                // The server can return the entire answer in one non-streaming record.
                session.finish_text(generation.text);
                self.busy = None;
                let capped = generation.metadata["done_reason"] == "length";
                let rate = generation_rate(&generation.metadata)
                    .map_or(String::new(), |rate| format!(" · {rate:.1} tok/s"));
                self.status = format!(
                    "{} · {} tokens{rate} · {:.1}s{}",
                    if capped {
                        "达到输出上限"
                    } else {
                        "回答完成"
                    },
                    generation.metadata["eval_count"].as_u64().unwrap_or(0),
                    self.started.elapsed().as_secs_f64(),
                    if capped { " · 可增大 -n" } else { "" }
                );
            }
            Update::Failed(model, selecting, text) => {
                self.busy = None;
                self.ready = false;
                self.sessions[index(if selecting { self.active } else { model })]
                    .append(Speaker::Error, clean_text(&text));
                self.status = if selecting {
                    "连接或切换失败，历史已保留；Ctrl+R 重试。"
                } else {
                    "请求失败，本轮未写入历史；Ctrl+R 重试连接，Ctrl+L 清空对话。"
                }
                .into();
            }
        }
    }

    fn key(&mut self, key: KeyEvent) -> bool {
        if key.kind == KeyEventKind::Release {
            return false;
        }
        let control = key.modifiers.contains(KeyModifiers::CONTROL);
        if control {
            match key.code {
                KeyCode::Char('c' | 'd') => return true,
                KeyCode::Char('t') => {
                    self.select(self.active.next());
                    return false;
                }
                KeyCode::Char('l') => {
                    self.clear();
                    return false;
                }
                KeyCode::Char('r') => {
                    self.select(self.active);
                    return false;
                }
                KeyCode::Char('a') => {
                    self.session().editor.home();
                    return false;
                }
                KeyCode::Char('e') => {
                    self.session().editor.end();
                    return false;
                }
                KeyCode::Char('b') => {
                    self.session().editor.left();
                    return false;
                }
                KeyCode::Char('f') => {
                    self.session().editor.right();
                    return false;
                }
                KeyCode::Char('u') => {
                    self.session().editor.clear();
                    return false;
                }
                _ => return false,
            }
        }
        match key.code {
            KeyCode::F(2) => self.select(self.active.next()),
            KeyCode::PageUp => {
                self.session().scroll = self.session().scroll.saturating_add(8);
            }
            KeyCode::PageDown => {
                self.session().scroll = self.session().scroll.saturating_sub(8);
            }
            KeyCode::Enter
                if key
                    .modifiers
                    .intersects(KeyModifiers::ALT | KeyModifiers::SHIFT) =>
            {
                self.insert_input("\n");
            }
            KeyCode::Enter => return self.submit(),
            KeyCode::Backspace => self.session().editor.backspace(),
            KeyCode::Delete => self.session().editor.delete(),
            KeyCode::Left => self.session().editor.left(),
            KeyCode::Right => self.session().editor.right(),
            KeyCode::Home => self.session().editor.home(),
            KeyCode::End => self.session().editor.end(),
            KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::ALT) => {
                self.insert_input(&c.to_string())
            }
            _ => {}
        }
        false
    }

    fn render(&mut self, frame: &mut Frame) {
        let area = frame.area();
        frame.render_widget(
            Block::default().style(Style::default().bg(BACKGROUND).fg(TEXT)),
            area,
        );
        if area.width < 44 || area.height < 14 {
            frame.render_widget(Paragraph::new("请放大终端至至少 44 列、14 行。\nCtrl+C 退出，草稿与会话会保留到本次进程结束。").style(Style::default().fg(ACCENT)), area);
            return;
        }
        let margin = if area.width > 124 {
            (area.width - 120) / 2
        } else {
            2
        };
        let inner = Rect {
            x: area.x + margin,
            y: area.y + 1,
            width: area.width - 2 * margin,
            height: area.height - 2,
        };
        let chunks = Layout::vertical([
            Constraint::Length(1),
            Constraint::Length(2),
            Constraint::Min(3),
            Constraint::Length(1),
            Constraint::Length(5),
            Constraint::Length(1),
        ])
        .split(inner);
        let header = Line::from(vec![
            Span::styled(
                "Lumen",
                Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
            ),
            Span::styled("  ·  LifeAI", Style::default().fg(MUTED)),
        ]);
        frame.render_widget(Paragraph::new(header), chunks[0]);
        let labels =
            [Model::Qwen, Model::Flint].map(|model| Line::from(format!(" {} ", model.label())));
        let tabs = Tabs::new(labels)
            .select(index(self.active))
            .divider("  ")
            .style(Style::default().fg(MUTED))
            .highlight_style(
                Style::default()
                    .bg(ACCENT)
                    .fg(BACKGROUND)
                    .add_modifier(Modifier::BOLD),
            );
        frame.render_widget(tabs, chunks[1]);
        let model = self.active;
        let session = self.session();
        let visible_height = chunks[2].height as usize;
        session.rows(model, chunks[2].width as usize);
        session.scroll = session
            .scroll
            .min(session.rows.len().saturating_sub(visible_height));
        let end = session.rows.len() - session.scroll;
        let start = end.saturating_sub(visible_height);
        let visible = session.rows[start..end].to_vec();
        frame.render_widget(Paragraph::new(visible), chunks[2]);
        let state = match self.busy {
            Some(Busy::Loading(model)) => format!("{} 加载中", model.label()),
            Some(Busy::Generating(model)) => format!("{} 生成中", model.label()),
            None if self.ready => "就绪".into(),
            None => "未连接".into(),
        };
        let spinner = if self.busy.is_some() {
            ["◐", "◓", "◑", "◒"][(self.started.elapsed().as_millis() / 160 % 4) as usize]
        } else {
            "●"
        };
        let elapsed = if self.busy.is_some() {
            format!(" · {}s", self.started.elapsed().as_secs())
        } else {
            String::new()
        };
        let status = format!("{spinner} {state}{elapsed} · {}", self.status);
        frame.render_widget(
            Paragraph::new(status).style(Style::default().fg(
                if self.ready || self.busy.is_some() {
                    MUTED
                } else {
                    ERROR
                },
            )),
            chunks[3],
        );
        let composer = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(if self.busy.is_some() { MUTED } else { ACCENT }))
            .style(Style::default().bg(PANEL))
            .title(" 输入 ");
        let composer_inner = composer.inner(chunks[4]);
        frame.render_widget(composer, chunks[4]);
        let editor = &mut self.session().editor;
        let layout = editor.layout(composer_inner.width as usize);
        let first = layout
            .cursor_row
            .saturating_sub(composer_inner.height.saturating_sub(1) as usize);
        let visible: Vec<Line<'static>> = layout
            .rows
            .iter()
            .skip(first)
            .take(composer_inner.height as usize)
            .cloned()
            .map(Line::from)
            .collect();
        frame.render_widget(
            Paragraph::new(visible).style(Style::default().fg(TEXT)),
            composer_inner,
        );
        frame.set_cursor_position((
            composer_inner.x
                + layout
                    .cursor_column
                    .min(composer_inner.width.saturating_sub(1) as usize) as u16,
            composer_inner.y + (layout.cursor_row - first) as u16,
        ));
        frame.render_widget(
            Paragraph::new(
                "Enter 发送 · Alt+Enter 换行 · Ctrl+T 切换 · PgUp/PgDn 滚动 · /help · Ctrl+C 退出",
            )
            .style(Style::default().fg(MUTED)),
            chunks[5],
        );
    }
}

type Connector = Box<dyn Fn() -> Result<()> + Send>;

fn start_worker(
    api: Api,
    load_timeout: Duration,
    connector: Connector,
) -> (Sender<Job>, Receiver<Update>) {
    let (jobs, receiver) = mpsc::channel();
    let (updates, events) = mpsc::channel();
    thread::spawn(move || {
        while let Ok(job) = receiver.recv() {
            let status = |message| {
                let _ = updates.send(Update::Status(message));
            };
            match job {
                Job::Select(model) => {
                    match connector().and_then(|_| api.select(model, load_timeout, status)) {
                        Ok(health) => {
                            if updates.send(Update::Selected(model, health)).is_err() {
                                break;
                            }
                        }
                        Err(error) => {
                            if updates
                                .send(Update::Failed(model, true, format!("{error:#}")))
                                .is_err()
                            {
                                break;
                            }
                        }
                    }
                }
                Job::Generate(model, messages) => {
                    // Another client may have switched the GPU worker since our last turn.
                    let result = connector()
                        .and_then(|_| api.select(model, load_timeout, status))
                        .and_then(|_| {
                            api.generate_with_progress(
                                model,
                                &messages,
                                |text| {
                                    let _ = updates.send(Update::Piece(model, text));
                                },
                                |phase| {
                                    let _ = updates.send(Update::Status(phase));
                                },
                            )
                        });
                    let update = match result {
                        Ok(generation) => Update::Done(model, messages, generation),
                        Err(error) => Update::Failed(model, false, format!("{error:#}")),
                    };
                    if updates.send(update).is_err() {
                        break;
                    }
                }
            }
        }
    });
    (jobs, events)
}

struct TerminalGuard;

fn restore_terminal() {
    let _ = disable_raw_mode();
    let _ = execute!(
        io::stdout(),
        DisableBracketedPaste,
        LeaveAlternateScreen,
        crossterm::cursor::Show
    );
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        restore_terminal();
    }
}

/// Run with no system prompt. Kept as a simple embedding entry point.
#[allow(dead_code)]
pub fn run(api: Api, initial_model: Model, load_timeout: Duration) -> Result<()> {
    run_with_system(api, initial_model, load_timeout, String::new())
}

pub fn run_with_system(
    api: Api,
    initial_model: Model,
    load_timeout: Duration,
    system: String,
) -> Result<()> {
    run_with_connector(
        api,
        initial_model,
        load_timeout,
        system,
        Box::new(|| Ok(())),
    )
}

/// The connector runs on the network worker before every operation. Embedders
/// can restore an owned SSH tunnel here without blocking input or losing state.
pub fn run_with_connector(
    api: Api,
    initial_model: Model,
    load_timeout: Duration,
    system: String,
    connector: Connector,
) -> Result<()> {
    let previous_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        restore_terminal();
        previous_hook(info);
    }));
    enable_raw_mode().context("无法进入终端 raw 模式")?;
    let _guard = TerminalGuard;
    execute!(io::stdout(), EnterAlternateScreen, EnableBracketedPaste)
        .context("无法进入终端全屏模式")?;
    let mut terminal: Terminal<CrosstermBackend<Stdout>> =
        Terminal::new(CrosstermBackend::new(io::stdout()))?;
    terminal.clear()?;
    let (jobs, updates) = start_worker(api, load_timeout, connector);
    let mut app = App::new(initial_model, system, jobs);
    app.select(initial_model);
    loop {
        while let Ok(update) = updates.try_recv() {
            app.update(update);
        }
        terminal.draw(|frame| app.render(frame))?;
        if event::poll(Duration::from_millis(50))? {
            match event::read()? {
                Event::Key(key) if app.key(key) => break,
                Event::Paste(text) => app.insert_input(&text),
                _ => {}
            }
        }
    }
    // Dropping the sender lets an idle worker finish. Do not join a worker that
    // may be loading a model: quitting must restore the terminal immediately.
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;
    use serde_json::json;

    fn app() -> (App, Receiver<Job>) {
        let (tx, rx) = mpsc::channel();
        (App::new(Model::Qwen, "系统提示".into(), tx), rx)
    }

    #[test]
    fn editor_deletes_whole_chinese_emoji_and_combining_graphemes() {
        let mut editor = Editor::default();
        assert!(editor.insert("中👩‍💻e\u{301}"));
        editor.backspace();
        assert_eq!(editor.text, "中👩‍💻");
        editor.left();
        assert_eq!(editor.cursor, "中".len());
        editor.delete();
        assert_eq!(editor.text, "中");
        editor.backspace();
        assert!(editor.text.is_empty());
    }

    #[test]
    fn editor_wrap_cursor_and_newlines_use_display_cells() {
        let mut editor = Editor::default();
        editor.insert("你好ab\nc");
        let layout = editor.layout(4);
        assert_eq!(layout.rows, ["你好", "ab", "c"]);
        assert_eq!((layout.cursor_row, layout.cursor_column), (2, 1));
        editor.home();
        editor.insert("中");
        assert_eq!(editor.text, "你好ab\n中c");
        editor.left();
        assert_eq!(editor.layout(4).cursor_column, 0);
    }

    #[test]
    fn editor_exact_width_cursor_gets_next_row() {
        let mut editor = Editor::default();
        editor.insert("你好");
        let layout = editor.layout(4);
        assert_eq!((layout.cursor_row, layout.cursor_column), (1, 0));
    }

    #[test]
    fn paste_does_not_execute_controls_and_is_bounded() {
        let mut editor = Editor::default();
        assert!(editor.insert("hello\x1b[31m\r\n\t世界\u{85}"));
        assert_eq!(editor.text, "hello[31m\n 世界");
        let before = editor.text.clone();
        assert!(!editor.insert(&"x".repeat(MAX_INPUT_BYTES)));
        assert_eq!(editor.text, before);
    }

    #[test]
    fn long_draft_layout_keeps_only_cursor_neighbourhood() {
        let mut editor = Editor::default();
        let text = "长上下文👩‍💻\n".repeat(60000);
        assert!(text.len() > 1024 * 1024);
        assert!(editor.insert(&text));
        let layout = editor.layout(80);
        assert!(layout.rows.len() < 32);
        assert!(layout.cursor_row < layout.rows.len());
        editor.cursor = text.len() / 2;
        editor.changed();
        let layout = editor.layout(80);
        assert!(layout.rows.len() < 32);
        assert_eq!(editor.text, text);
        editor.end();
        assert!(editor.insert("追加"));
        assert!(editor.layout(80).rows.len() < 32);
    }

    #[test]
    fn load_command_imports_utf8_draft_and_preserves_it_on_error() {
        let (mut app, jobs) = app();
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("长文.txt");
        std::fs::write(&path, "文件中的上下文\n问题").unwrap();
        app.session()
            .editor
            .insert(&format!("/load {}", path.display()));
        assert!(!app.submit());
        assert_eq!(app.session().editor.text, "文件中的上下文\n问题");
        assert!(app.status.contains("已导入"));
        assert!(jobs.try_recv().is_err());
        app.session().editor.clear();
        let command = format!("/load {}", path.with_extension("missing").display());
        app.session().editor.insert(&command);
        app.submit();
        assert_eq!(app.session().editor.text, command);
        assert!(app.status.contains("草稿已保留"));
    }

    #[test]
    fn incomplete_turn_never_reaches_history() {
        let (mut app, jobs) = app();
        app.ready = true;
        app.session().editor.insert("问题");
        assert!(!app.submit());
        assert!(matches!(
            jobs.try_recv().unwrap(),
            Job::Generate(Model::Qwen, _)
        ));
        app.update(Update::Piece(Model::Qwen, "半个回答".into()));
        app.update(Update::Failed(Model::Qwen, false, "断流".into()));
        assert_eq!(app.session().history.len(), 1);
        assert_eq!(app.session().entries.len(), 3);
        assert!(!app.ready);
    }

    #[test]
    fn model_sessions_and_drafts_survive_switching_and_clear() {
        let (mut app, jobs) = app();
        app.session().editor.insert("Qwen 草稿");
        app.session().append(Speaker::User, "Qwen 对话".into());
        app.select(Model::Flint);
        assert!(matches!(
            jobs.try_recv().unwrap(),
            Job::Select(Model::Flint)
        ));
        app.update(Update::Selected(
            Model::Flint,
            json!({"context_tokens":2048,"max_new_tokens":512}),
        ));
        app.session().editor.insert("Flint 草稿");
        app.clear();
        app.select(Model::Qwen);
        jobs.try_recv().unwrap();
        app.update(Update::Selected(
            Model::Qwen,
            json!({"context_tokens":40960,"max_new_tokens":512}),
        ));
        assert_eq!(app.session().editor.text, "Qwen 草稿");
        assert_eq!(app.session().entries.len(), 1);
        assert_eq!(app.session().history[0].content, "系统提示");
    }

    #[test]
    fn only_successful_final_answer_is_committed() {
        let (mut app, _) = app();
        let pending = vec![Message {
            role: "user".into(),
            content: "1+1".into(),
        }];
        app.active = Model::Flint;
        app.session().append(Speaker::Assistant, String::new());
        app.update(Update::Done(
            Model::Flint,
            pending,
            Generation {
                text: "<think>计算</think>2".into(),
                metadata: json!({"done":true,"eval_count":10,"eval_duration":1000000000_u64}),
            },
        ));
        assert_eq!(app.session().history[1].content, "2");
        assert!(app.session().entries[0].text.contains("<think>"));
        assert!(app.status.contains("回答完成"));
    }

    #[test]
    fn busy_state_blocks_switch_and_send_but_allows_quit() {
        let (mut app, jobs) = app();
        app.busy = Some(Busy::Generating(Model::Qwen));
        app.session().editor.insert("未发送草稿");
        app.select(Model::Flint);
        app.submit();
        assert!(jobs.try_recv().is_err());
        assert_eq!(app.session().editor.text, "未发送草稿");
        assert!(app.key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL)));
    }

    #[test]
    fn reconnect_failure_is_reported_before_any_api_request() {
        use std::net::TcpListener;
        use std::sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        };

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let api = Api::new(
            &format!("http://{}", listener.local_addr().unwrap()),
            Duration::from_secs(1),
            16,
            0.0,
            false,
        )
        .unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let attempts = Arc::clone(&calls);
        let (jobs, updates) = start_worker(
            api,
            Duration::from_secs(1),
            Box::new(move || {
                attempts.fetch_add(1, Ordering::SeqCst);
                anyhow::bail!("SSH 重连失败")
            }),
        );
        jobs.send(Job::Select(Model::Flint)).unwrap();
        assert!(
            matches!(updates.recv_timeout(Duration::from_secs(2)).unwrap(), Update::Failed(Model::Flint, true, text) if text.contains("SSH 重连失败"))
        );
        jobs.send(Job::Generate(Model::Qwen, Vec::new())).unwrap();
        assert!(
            matches!(updates.recv_timeout(Duration::from_secs(2)).unwrap(), Update::Failed(Model::Qwen, false, text) if text.contains("SSH 重连失败"))
        );
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert_eq!(
            listener.accept().unwrap_err().kind(),
            io::ErrorKind::WouldBlock
        );
    }

    #[test]
    fn transcript_wrap_cache_is_reused_during_idle_render() {
        let mut session = Session::new("");
        session.append(Speaker::User, "你好世界".into());
        session.rows(Model::Qwen, 4);
        let allocation = session.rows.as_ptr();
        assert_eq!(session.rows(Model::Qwen, 4).as_ptr(), allocation);
        assert_eq!(session.rows[1].to_string(), "你好");
        assert_eq!(session.rows[2].to_string(), "世界");
    }

    #[test]
    fn streaming_updates_only_rebuild_the_changed_tail() {
        let mut session = Session::new("");
        for turn in 0..100 {
            session.append(Speaker::User, format!("历史第 {turn} 轮：你好世界"));
        }
        session.append(Speaker::Assistant, String::new());
        session.rows(Model::Flint, 40);
        let first_text_allocation = session.rows[1].spans[0].content.as_ptr();
        let first_entry_rows = session.cached_ends[0];
        session.piece("新回答");
        session.rows(Model::Flint, 40);
        assert_eq!(
            session.rows[1].spans[0].content.as_ptr(),
            first_text_allocation
        );
        assert_eq!(session.cached_ends[0], first_entry_rows);
        assert!(session.rows.iter().any(|row| row.to_string() == "新回答"));
        session.finish_text("完整回答".into());
        session.rows(Model::Flint, 40);
        assert!(session.rows.iter().any(|row| row.to_string() == "完整回答"));
        assert!(!session.rows.iter().any(|row| row.to_string() == "新回答"));
        session.rows(Model::Flint, 8); // A terminal resize reflows all messages.
        assert_eq!(session.cache_width, 8);
        assert_eq!(session.cached_ends.len(), 101);
    }

    #[test]
    fn renders_real_widgets_at_normal_and_small_sizes() {
        let (mut app, _) = app();
        app.ready = true;
        app.session()
            .append(Speaker::Assistant, "中文流式回答\x1b[31m".into());
        app.session().editor.insert("你好 👩‍💻");
        let mut terminal = Terminal::new(TestBackend::new(100, 26)).unwrap();
        terminal.draw(|frame| app.render(frame)).unwrap();
        let buffer = terminal.backend().buffer();
        let rendered = buffer
            .content()
            .iter()
            .map(|cell| cell.symbol())
            .collect::<String>();
        assert!(rendered.contains("Lumen"));
        assert!(rendered.contains("LifeAI"));
        assert!(rendered.contains("Qwen"));
        assert!(rendered.contains("Flint"));
        assert!(!rendered.contains('\x1b'));
        let mut small = Terminal::new(TestBackend::new(30, 8)).unwrap();
        small.draw(|frame| app.render(frame)).unwrap();
    }
}
