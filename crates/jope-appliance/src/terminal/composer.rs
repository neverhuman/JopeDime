//! Multiline editing component (spec §6.4). The client supplies the canonical
//! history path and completion context; no transport event can mutate this model.
use super::complete::{self, CompletionData, Suggestion};
use super::sanitize_user_string;
use ratatui::crossterm::event::{Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::layout::{Position, Rect};
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

const MAX_DRAFT: usize = 8 * 1024 * 1024;
const MAX_HISTORY: usize = 16 * 1024 * 1024;
const HISTORY_ENTRIES: usize = 1000;
/// Client help/dispatch wiring belongs to TUI-09.
pub const HISTORY_HELP: &str = "History keeps up to 1,000 entries in $JOPEDIME_HOME/history (0600). Inline JSON objectives over 1,024 bytes are omitted. To clear it, close the client and remove that history file.";
static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, PartialEq, Eq)]
pub enum ComposerAction {
    None,
    Submit(String),
    ExitRequested,
    Interrupt,
    Redraw,
}

struct CompletionSession {
    original: String,
    range: std::ops::Range<usize>,
    choices: Vec<Suggestion>,
    selected: usize,
}
struct HistoryNavigation {
    index: usize,
    saved: String,
    cursor: usize,
}
struct ReverseSearch {
    saved: String,
    cursor: usize,
    query: String,
    found: Option<usize>,
}

#[derive(Default)]
pub struct Composer {
    text: String,
    cursor: usize,
    menu: Option<CompletionSession>,
    history: Option<HistoryFile>,
    navigation: Option<HistoryNavigation>,
    search: Option<ReverseSearch>,
}

/// Byte boundaries from ratatui's existing unicode-segmentation backend. Its Span
/// iterator excludes LF, so handle each LF explicitly rather than losing it.
fn boundaries(text: &str) -> Vec<usize> {
    let mut result = vec![0];
    let mut offset = 0;
    for line in text.split_inclusive('\n') {
        let body = line.strip_suffix('\n').unwrap_or(line);
        let span = Span::raw(body);
        for g in span.styled_graphemes(Style::default()) {
            offset += g.symbol.len();
            result.push(offset);
        }
        if line.ends_with('\n') {
            offset += 1;
            result.push(offset);
        }
    }
    result
}
fn clean(text: &str) -> String {
    text.replace("\r\n", "\n")
        .replace('\r', "\n")
        .split('\n')
        .map(|line| {
            line.split('\t')
                .map(sanitize_user_string)
                .collect::<Vec<_>>()
                .join("\t")
        })
        .collect::<Vec<_>>()
        .join("\n")
}
fn too_large() -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidInput,
        "composer input exceeds 8 MiB; use a file argument",
    )
}

impl Composer {
    pub fn with_history(path: impl AsRef<Path>) -> io::Result<Self> {
        Ok(Self {
            history: Some(HistoryFile::open(path.as_ref())?),
            ..Self::default()
        })
    }
    pub fn text(&self) -> &str {
        &self.text
    }
    pub fn cursor_byte(&self) -> usize {
        self.cursor
    }
    pub fn completion(&self) -> Option<(&[Suggestion], usize)> {
        self.menu
            .as_ref()
            .map(|m| (m.choices.as_slice(), m.selected))
    }
    pub fn set_text(&mut self, text: &str) -> io::Result<()> {
        let text = clean(text);
        if text.len() > MAX_DRAFT {
            return Err(too_large());
        }
        self.cursor = text.len();
        self.text = text;
        self.menu = None;
        self.navigation = None;
        self.search = None;
        Ok(())
    }
    fn edit(&mut self, range: std::ops::Range<usize>, insert: &str) -> io::Result<()> {
        if self.text.len() - range.len() + insert.len() > MAX_DRAFT {
            return Err(too_large());
        }
        let cursor = range.start + insert.len();
        self.text.replace_range(range, insert);
        // Inserting a combining mark/ZWJ can merge clusters across the old
        // cursor. Snap forward to the end of the resulting whole grapheme.
        self.cursor = boundaries(&self.text)
            .into_iter()
            .find(|&b| b >= cursor)
            .unwrap_or(self.text.len());
        self.menu = None;
        self.navigation = None;
        Ok(())
    }
    fn insert(&mut self, text: &str) -> io::Result<()> {
        self.edit(self.cursor..self.cursor, text)
    }
    fn previous(&self) -> usize {
        boundaries(&self.text)
            .into_iter()
            .take_while(|&b| b < self.cursor)
            .last()
            .unwrap_or(0)
    }
    fn next(&self) -> usize {
        boundaries(&self.text)
            .into_iter()
            .find(|&b| b > self.cursor)
            .unwrap_or(self.text.len())
    }
    fn home(&self) -> usize {
        self.text[..self.cursor].rfind('\n').map_or(0, |p| p + 1)
    }
    fn end(&self) -> usize {
        self.text[self.cursor..]
            .find('\n')
            .map_or(self.text.len(), |p| self.cursor + p)
    }
    fn word_back(&self, whitespace_only: bool) -> usize {
        let points = boundaries(&self.text);
        let mut p = points.partition_point(|&b| b < self.cursor);
        let word = |s: &str| {
            if whitespace_only {
                !s.chars().all(char::is_whitespace)
            } else {
                s.chars().any(|c| c.is_alphanumeric() || c == '_')
            }
        };
        while p > 0 && !word(&self.text[points[p - 1]..points[p]]) {
            p -= 1;
        }
        while p > 0 && word(&self.text[points[p - 1]..points[p]]) {
            p -= 1;
        }
        points[p]
    }
    fn word_forward(&self) -> usize {
        let points = boundaries(&self.text);
        let mut p = points.partition_point(|&b| b < self.cursor);
        let word = |s: &str| s.chars().any(|c| c.is_alphanumeric() || c == '_');
        while p + 1 < points.len() && !word(&self.text[points[p]..points[p + 1]]) {
            p += 1;
        }
        while p + 1 < points.len() && word(&self.text[points[p]..points[p + 1]]) {
            p += 1;
        }
        points[p]
    }
    fn history_move(&mut self, older: bool) {
        let Some(history) = &self.history else {
            return;
        };
        if history.entries.is_empty() {
            return;
        }
        let next = match self.navigation.take() {
            None if older => HistoryNavigation {
                index: history.entries.len() - 1,
                saved: self.text.clone(),
                cursor: self.cursor,
            },
            Some(mut nav) if older => {
                nav.index = nav.index.saturating_sub(1);
                nav
            }
            Some(mut nav) if nav.index + 1 < history.entries.len() => {
                nav.index += 1;
                nav
            }
            Some(nav) => {
                self.text = nav.saved;
                self.cursor = nav.cursor;
                return;
            }
            None => return,
        };
        self.text = history.entries[next.index].clone();
        self.cursor = self.text.len();
        self.navigation = Some(next);
        self.menu = None;
    }
    fn search_match(&mut self, older: bool) {
        let Some(search) = &mut self.search else {
            return;
        };
        let entries = self
            .history
            .as_ref()
            .map(|h| h.entries.as_slice())
            .unwrap_or(&[]);
        let limit = if older {
            search.found.unwrap_or(entries.len())
        } else {
            entries.len()
        };
        search.found = (0..limit)
            .rev()
            .find(|&i| entries[i].contains(&search.query));
        if let Some(i) = search.found {
            self.text = entries[i].clone();
            self.cursor = self.text.len();
        } else {
            self.text = search.saved.clone();
            self.cursor = search.cursor;
        }
    }
    fn search_key(&mut self, key: KeyEvent) -> io::Result<Option<ComposerAction>> {
        if self.search.is_none() {
            return Ok(None);
        }
        match key.code {
            KeyCode::Esc | KeyCode::Char('g')
                if key.code == KeyCode::Esc || key.modifiers.contains(KeyModifiers::CONTROL) =>
            {
                let s = self.search.take().unwrap();
                self.text = s.saved;
                self.cursor = s.cursor;
            }
            KeyCode::Enter => {
                self.search = None;
                return Ok(None); // Enter accepts the match and submits normally.
            }
            KeyCode::Char('r') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.search_match(true)
            }
            KeyCode::Backspace => {
                let s = self.search.as_mut().unwrap();
                let b = boundaries(&s.query);
                let end = b.get(b.len().saturating_sub(2)).copied().unwrap_or(0);
                s.query.truncate(end);
                self.search_match(false);
            }
            KeyCode::Char(c)
                if !key
                    .modifiers
                    .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT)
                    && !c.is_control() =>
            {
                let s = self.search.as_mut().unwrap();
                if s.query.len() + c.len_utf8() > MAX_DRAFT {
                    return Err(too_large());
                }
                s.query.push(c);
                self.search_match(false);
            }
            KeyCode::Char('l') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                return Ok(Some(ComposerAction::Redraw))
            }
            _ => {
                self.search = None;
                return Ok(None);
            }
        }
        Ok(Some(ComposerAction::None))
    }
    fn tab(&mut self, data: &CompletionData<'_>, backwards: bool) -> io::Result<()> {
        let menu = if let Some(mut menu) = self.menu.take() {
            menu.selected =
                complete::cycle_index(Some(menu.selected), menu.choices.len(), backwards).unwrap();
            menu
        } else {
            let choices = complete::complete(&self.text, self.cursor, data);
            let Some(selected) = complete::cycle_index(None, choices.len(), backwards) else {
                return Ok(());
            };
            CompletionSession {
                original: self.text.clone(),
                range: token_range(&self.text, self.cursor),
                choices,
                selected,
            }
        };
        let insert = &menu.choices[menu.selected].insert;
        if insert.chars().any(char::is_control) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "completion contains control characters",
            ));
        }
        let mut text = menu.original.clone();
        if text.len() - menu.range.len() + insert.len() > MAX_DRAFT {
            return Err(too_large());
        }
        text.replace_range(menu.range.clone(), insert);
        self.cursor = menu.range.start + insert.len();
        self.text = text;
        // Keep the original token for reversible cycling, rather than completing
        // an already completed token into a duplicated suffix.
        self.navigation = None;
        self.menu = Some(menu);
        Ok(())
    }
    pub fn handle(
        &mut self,
        event: Event,
        data: &CompletionData<'_>,
    ) -> io::Result<ComposerAction> {
        let key = match event {
            Event::Paste(text) => {
                self.search = None;
                self.insert(&clean(&text))?;
                return Ok(ComposerAction::None);
            }
            Event::Resize(..) => return Ok(ComposerAction::Redraw),
            Event::Key(key) if key.kind != KeyEventKind::Release => key,
            _ => return Ok(ComposerAction::None),
        };
        if let Some(action) = self.search_key(key)? {
            return Ok(action);
        }
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let alt = key.modifiers.contains(KeyModifiers::ALT);
        match key.code {
            KeyCode::Enter
                if key
                    .modifiers
                    .intersects(KeyModifiers::ALT | KeyModifiers::SHIFT) =>
            {
                self.insert("\n")?
            }
            KeyCode::Enter if key.modifiers.is_empty() => {
                if let Some(history) = &mut self.history {
                    history.push(&self.text)?;
                }
                let text = std::mem::take(&mut self.text);
                self.cursor = 0;
                self.menu = None;
                self.navigation = None;
                return Ok(ComposerAction::Submit(text));
            }
            KeyCode::Left => {
                self.cursor = self.previous();
                self.menu = None;
            }
            KeyCode::Right => {
                self.cursor = self.next();
                self.menu = None;
            }
            KeyCode::Home | KeyCode::Char('a') if key.code == KeyCode::Home || ctrl => {
                self.cursor = self.home();
                self.menu = None;
            }
            KeyCode::End | KeyCode::Char('e') if key.code == KeyCode::End || ctrl => {
                self.cursor = self.end();
                self.menu = None;
            }
            KeyCode::Char('b') if alt => {
                self.cursor = self.word_back(false);
                self.menu = None;
            }
            KeyCode::Char('f') if alt => {
                self.cursor = self.word_forward();
                self.menu = None;
            }
            KeyCode::Backspace => self.edit(self.previous()..self.cursor, "")?,
            KeyCode::Delete => self.edit(self.cursor..self.next(), "")?,
            KeyCode::Char('d') if ctrl => {
                if self.text.is_empty() {
                    return Ok(ComposerAction::ExitRequested);
                }
                self.edit(self.cursor..self.next(), "")?;
            }
            KeyCode::Char('c') if ctrl => {
                if self.text.is_empty() {
                    return Ok(ComposerAction::Interrupt);
                }
                self.set_text("")?;
            }
            KeyCode::Char('w') if ctrl => self.edit(self.word_back(true)..self.cursor, "")?,
            KeyCode::Char('u') if ctrl => self.edit(self.home()..self.cursor, "")?,
            KeyCode::Char('k') if ctrl => {
                let end = self.end();
                let end = if end == self.cursor { self.next() } else { end };
                self.edit(self.cursor..end, "")?;
            }
            KeyCode::Char('l') if ctrl => return Ok(ComposerAction::Redraw),
            KeyCode::Char('r') if ctrl => {
                self.menu = None;
                self.navigation = None;
                self.search = Some(ReverseSearch {
                    saved: self.text.clone(),
                    cursor: self.cursor,
                    query: String::new(),
                    found: None,
                });
                self.search_match(false);
            }
            KeyCode::Up => self.history_move(true),
            KeyCode::Down => self.history_move(false),
            KeyCode::Tab => self.tab(data, key.modifiers.contains(KeyModifiers::SHIFT))?,
            KeyCode::BackTab => self.tab(data, true)?,
            KeyCode::Esc => {
                self.menu = None;
            }
            KeyCode::Char(c) if !ctrl && !alt && !c.is_control() => self.insert(&c.to_string())?,
            _ => {}
        }
        Ok(ComposerAction::None)
    }

    pub fn view(&self, width: u16, height: u16) -> ComposerView {
        if width == 0 || height == 0 {
            return ComposerView {
                lines: Vec::new(),
                cursor: (0, 0),
            };
        }
        let width = usize::from(width);
        let height = usize::from(height);
        let mut menu = Vec::new();
        if let Some(search) = &self.search {
            menu.push(format!("reverse search: {}", search.query));
        } else if let Some(m) = &self.menu {
            let count = m.choices.len().min(2).min(height.saturating_sub(1));
            let start = m.selected.saturating_sub(count.saturating_sub(1));
            for i in start..start + count {
                menu.push(format!(
                    "{}{}",
                    if i == m.selected { "> " } else { "  " },
                    sanitize_user_string(&m.choices[i].insert)
                ));
            }
        }
        menu.truncate(height.saturating_sub(1));
        let room = (height - menu.len()).clamp(1, 4);
        let prefix = if width >= 3 { "> " } else { "" };
        let continuation = if width >= 3 { "  " } else { "" };
        let mut lines = Vec::new();
        let mut line = prefix.to_owned();
        let mut col = prefix.len();
        let mut caret = (col, 0);
        let points = boundaries(&self.text);
        for pair in points.windows(2) {
            let g = &self.text[pair[0]..pair[1]];
            if g != "\n" {
                let cells = if g == "\t" {
                    4 - col % 4
                } else {
                    Span::raw(g).width()
                };
                if col + cells > width && col > continuation.len() {
                    lines.push(std::mem::replace(&mut line, continuation.into()));
                    col = continuation.len();
                }
            }
            if pair[0] == self.cursor {
                caret = if g == "\n" && col == width {
                    (continuation.len(), lines.len() + 1)
                } else {
                    (col, lines.len())
                };
            }
            if g == "\n" {
                lines.push(std::mem::replace(&mut line, continuation.into()));
                col = continuation.len();
            } else {
                let cells = if g == "\t" {
                    4 - col % 4
                } else {
                    Span::raw(g).width()
                };
                if cells > width - col {
                    line.push('?');
                    col += 1;
                } else if g == "\t" {
                    line.push_str(&" ".repeat(cells));
                    col += cells;
                } else {
                    line.push_str(g);
                    col += cells;
                }
            }
        }
        if self.cursor == self.text.len() {
            if col == width {
                lines.push(std::mem::replace(&mut line, continuation.into()));
                col = continuation.len();
            }
            caret = (col, lines.len());
        }
        lines.push(line);
        let top = caret.1.saturating_sub(room - 1);
        let mut result: Vec<_> = menu.into_iter().map(|s| clip(&s, width)).collect();
        let menu_rows = result.len();
        result.extend(lines.into_iter().skip(top).take(room));
        ComposerView {
            lines: result,
            cursor: (
                caret.0.min(width - 1) as u16,
                (menu_rows + caret.1 - top).min(height - 1) as u16,
            ),
        }
    }
    pub fn draw(&self, frame: &mut ratatui::Frame<'_>, area: Rect) {
        if area.width == 0 || area.height == 0 {
            return;
        }
        let view = self.view(area.width, area.height);
        frame.render_widget(
            Paragraph::new(view.lines.into_iter().map(Line::from).collect::<Vec<_>>()),
            area,
        );
        frame.set_cursor_position(Position::new(
            area.x + view.cursor.0,
            area.y + view.cursor.1,
        ));
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ComposerView {
    pub lines: Vec<String>,
    pub cursor: (u16, u16),
}
fn clip(text: &str, width: usize) -> String {
    let span = Span::raw(text);
    let mut result = String::new();
    let mut cells = 0;
    for g in span.styled_graphemes(Style::default()) {
        let n = Span::raw(g.symbol).width();
        if cells + n > width {
            break;
        }
        result.push_str(g.symbol);
        cells += n;
    }
    result
}
fn token_range(text: &str, cursor: usize) -> std::ops::Range<usize> {
    let mut start = 0;
    let mut quote = None;
    let mut escaped = false;
    let mut end = text.len();
    for (i, c) in text.char_indices() {
        if i >= cursor && quote.is_none() && !escaped && c.is_whitespace() {
            end = i;
            break;
        }
        if escaped {
            escaped = false;
            continue;
        }
        if c == '\\'
            && text[i + 1..]
                .chars()
                .next()
                .is_some_and(|n| n.is_whitespace() || n == '\'' || n == '"')
        {
            escaped = true;
            continue;
        }
        if let Some(q) = quote {
            if c == q {
                quote = None;
            }
        } else if c == '\'' || c == '"' {
            quote = Some(c);
        } else if c.is_whitespace() && i < cursor {
            start = i + c.len_utf8();
        }
    }
    start..end
}

/// Take the history writer lock, failing if another writer holds it. A child that another
/// thread forks can briefly hold a duplicate of the lock descriptor until it execs, which
/// `flock` reports as contention; retry for a bounded time so that is not mistaken for a
/// concurrent writer.
fn acquire_history_lock(lock: &File) -> io::Result<()> {
    use std::time::{Duration, Instant};
    let deadline = Instant::now() + Duration::from_millis(250);
    loop {
        match lock.try_lock() {
            Ok(()) => return Ok(()),
            Err(fs::TryLockError::WouldBlock) if Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(5));
            }
            Err(error) => return Err(io::Error::other(error)),
        }
    }
}

/// JSONL strings retain multiline entries. The client supplies the canonical
/// home path; this component does not guess a second home-resolution policy.
pub struct HistoryFile {
    path: PathBuf,
    entries: Vec<String>,
}
impl HistoryFile {
    pub fn open(path: &Path) -> io::Result<Self> {
        let mut history = Self {
            path: path.into(),
            entries: Vec::new(),
        };
        history.push_inner(None)?;
        Ok(history)
    }
    pub fn entries(&self) -> &[String] {
        &self.entries
    }
    pub fn push(&mut self, text: &str) -> io::Result<()> {
        if !remember(text) {
            return Ok(());
        }
        self.push_inner(Some(text))
    }
    fn push_inner(&mut self, text: Option<&str>) -> io::Result<()> {
        let parent = self
            .path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        fs::create_dir_all(parent)?;
        let mut lock_name = self.path.as_os_str().to_os_string();
        lock_name.push(".lock");
        let mut options = regular_options();
        options.write(true).create(true);
        let lock = options.open(PathBuf::from(lock_name))?;
        regular(&lock)?;
        acquire_history_lock(&lock)?;
        let mut entries = match regular_options().open(&self.path) {
            Ok(file) => {
                regular(&file)?;
                if file.metadata()?.len() > MAX_HISTORY as u64 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "history exceeds 16 MiB",
                    ));
                }
                let mut bytes = Vec::new();
                file.take(MAX_HISTORY as u64 + 1).read_to_end(&mut bytes)?;
                if bytes.len() > MAX_HISTORY {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "history exceeds 16 MiB",
                    ));
                }
                let content = std::str::from_utf8(&bytes).map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidData, "history is not UTF-8")
                })?;
                let mut entries = Vec::new();
                for line in content.lines() {
                    let entry: String = serde_json::from_str(line).map_err(|_| {
                        io::Error::new(io::ErrorKind::InvalidData, "invalid history record")
                    })?;
                    if remember(&entry) {
                        entries.push(clean(&entry));
                    }
                }
                entries
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => Vec::new(),
            Err(e) => return Err(e),
        };
        if let Some(text) = text {
            let text = clean(text);
            if entries.last() != Some(&text) {
                entries.push(text);
            }
        }
        if entries.len() > HISTORY_ENTRIES {
            entries.drain(..entries.len() - HISTORY_ENTRIES);
        }
        let mut records: Vec<_> = entries
            .iter()
            .map(|s| serde_json::to_string(s).expect("string serialization") + "\n")
            .collect();
        let mut size: usize = records.iter().map(String::len).sum();
        let mut drop_count = 0;
        while size > MAX_HISTORY {
            size -= records[drop_count].len();
            drop_count += 1;
        }
        entries.drain(..drop_count);
        records.drain(..drop_count);
        let id = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let name = format!(".history-{}-{id}.tmp", std::process::id());
        let temporary = parent.join(name);
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        owner_mode(&mut options);
        let mut file = options.open(&temporary)?;
        let published: io::Result<()> = (|| {
            for line in records {
                file.write_all(line.as_bytes())?;
            }
            file.sync_all()?;
            drop(file);
            fs::rename(&temporary, &self.path)?;
            #[cfg(unix)]
            File::open(parent)?.sync_all()?;
            Ok(())
        })();
        if published.is_err() {
            let _ = fs::remove_file(&temporary);
        }
        published?;
        self.entries = entries;
        Ok(())
    }
}
fn remember(text: &str) -> bool {
    let text = text.trim();
    if text.is_empty() || text.len() > MAX_DRAFT {
        return false;
    }
    let command = text.trim_start_matches(['\\', '/']);
    let name = command.split_whitespace().next().unwrap_or("");
    if name.eq_ignore_ascii_case("objective")
        || name.eq_ignore_ascii_case("fitness")
        || text.starts_with('{')
    {
        if let Some(start) = text.find('{') {
            if text[start..].len() > 1024 {
                return false;
            }
        }
    }
    true
}
fn owner_mode(options: &mut OpenOptions) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    #[cfg(not(unix))]
    let _ = options;
}
fn regular_options() -> OpenOptions {
    let mut options = OpenOptions::new();
    options.read(true);
    owner_mode(&mut options);
    // OS ABI flags: refuse symlinks/reparse points and avoid blocking on FIFOs.
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(0x20000 | 0x800);
    }
    #[cfg(target_os = "macos")]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(0x100 | 0x4);
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        options.custom_flags(0x00200000);
    }
    options
}
fn regular(file: &File) -> io::Result<()> {
    let metadata = file.metadata()?;
    if !metadata.is_file() || metadata.file_type().is_symlink() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "history must be a regular file",
        ));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if metadata.permissions().mode() & 0o777 != 0o600 {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "history must have mode 0600",
            ));
        }
    }
    Ok(())
}

#[cfg(all(test, target_os = "linux"))]
#[path = "../../tests/support/pty.rs"]
#[allow(clippy::duplicate_mod)] // Reuse TUI-10; app.rs keeps its own inclusion private.
mod pty;

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::crossterm::event::KeyEvent;
    #[cfg(target_os = "linux")]
    use std::time::{Duration, Instant};

    fn key(c: &mut Composer, code: KeyCode, mods: KeyModifiers) -> ComposerAction {
        c.handle(
            Event::Key(KeyEvent::new(code, mods)),
            &CompletionData::default(),
        )
        .unwrap()
    }
    #[test]
    fn grapheme_editing_handles_combining_zwj_flags_and_cjk() {
        let clusters = ["a", "e\u{301}", "👨‍👩‍👧‍👦", "🇨🇦", "中"];
        let mut c = Composer::default();
        c.set_text(&clusters.concat()).unwrap();
        key(&mut c, KeyCode::Home, KeyModifiers::NONE);
        let mut expected = 0;
        for g in clusters {
            key(&mut c, KeyCode::Right, KeyModifiers::NONE);
            expected += g.len();
            assert_eq!(c.cursor_byte(), expected);
        }
        for end in (0..clusters.len()).rev() {
            key(&mut c, KeyCode::Backspace, KeyModifiers::NONE);
            assert_eq!(c.text(), clusters[..end].concat());
        }
        c.set_text("ab").unwrap();
        key(&mut c, KeyCode::Left, KeyModifiers::NONE);
        key(&mut c, KeyCode::Char('\u{301}'), KeyModifiers::NONE);
        assert_eq!(c.text(), "a\u{301}b");
        assert_eq!(c.cursor_byte(), 3);
        key(&mut c, KeyCode::Left, KeyModifiers::NONE);
        assert_eq!(c.cursor_byte(), 0);
        key(&mut c, KeyCode::Delete, KeyModifiers::NONE);
        assert_eq!(c.text(), "b");
        c.set_text("").unwrap();
        for ch in "👩‍💻".chars() {
            key(&mut c, KeyCode::Char(ch), KeyModifiers::NONE);
        }
        key(&mut c, KeyCode::Left, KeyModifiers::NONE);
        assert_eq!(c.cursor_byte(), 0);
        key(&mut c, KeyCode::Delete, KeyModifiers::NONE);
        assert_eq!(c.text(), "");
    }
    #[test]
    fn multiline_bindings_paste_submission_and_controls_are_separate() {
        let mut c = Composer::default();
        c.set_text("one two").unwrap();
        key(&mut c, KeyCode::Char('w'), KeyModifiers::CONTROL);
        assert_eq!(c.text(), "one ");
        key(&mut c, KeyCode::Char('u'), KeyModifiers::CONTROL);
        assert_eq!(c.text(), "");
        c.set_text("one two\nthree four").unwrap();
        key(&mut c, KeyCode::Char('a'), KeyModifiers::CONTROL);
        assert_eq!(c.cursor_byte(), 8);
        key(&mut c, KeyCode::Char('k'), KeyModifiers::CONTROL);
        assert_eq!(c.text(), "one two\n");
        key(&mut c, KeyCode::Char('e'), KeyModifiers::CONTROL);
        assert_eq!(c.cursor_byte(), c.text().len());
        assert_eq!(
            key(&mut c, KeyCode::Enter, KeyModifiers::ALT),
            ComposerAction::None
        );
        assert_eq!(
            key(&mut c, KeyCode::Enter, KeyModifiers::SHIFT),
            ComposerAction::None
        );
        assert_eq!(
            c.handle(
                Event::Paste("pasted\r\n\ttext\x1b[31m!\r".into()),
                &CompletionData::default()
            )
            .unwrap(),
            ComposerAction::None
        );
        assert!(c.text().ends_with("pasted\n\ttext!\n"));
        assert!(!c.text().contains('\x1b'));
        let text = c.text().to_owned();
        assert_eq!(
            key(&mut c, KeyCode::Enter, KeyModifiers::CONTROL),
            ComposerAction::None
        );
        assert_eq!(c.text(), text);
        assert_eq!(
            key(&mut c, KeyCode::Char('l'), KeyModifiers::CONTROL),
            ComposerAction::Redraw
        );
        assert_eq!(c.text(), text);
        assert_eq!(
            key(&mut c, KeyCode::Enter, KeyModifiers::NONE),
            ComposerAction::Submit(text)
        );
        assert_eq!(c.text(), "");
        assert_eq!(
            key(&mut c, KeyCode::Char('d'), KeyModifiers::CONTROL),
            ComposerAction::ExitRequested
        );
        assert_eq!(
            key(&mut c, KeyCode::Char('c'), KeyModifiers::CONTROL),
            ComposerAction::Interrupt
        );
        c.set_text("e\u{301}x").unwrap();
        key(&mut c, KeyCode::Home, KeyModifiers::NONE);
        key(&mut c, KeyCode::Char('d'), KeyModifiers::CONTROL);
        assert_eq!(c.text(), "x");
        key(&mut c, KeyCode::Char('c'), KeyModifiers::CONTROL);
        assert_eq!(c.text(), "");
        c.set_text("one two three").unwrap();
        key(&mut c, KeyCode::Char('b'), KeyModifiers::ALT);
        assert_eq!(&c.text()[c.cursor_byte()..], "three");
        key(&mut c, KeyCode::Char('b'), KeyModifiers::ALT);
        assert_eq!(&c.text()[c.cursor_byte()..], "two three");
        key(&mut c, KeyCode::Char('f'), KeyModifiers::ALT);
        assert_eq!(&c.text()[c.cursor_byte()..], " three");
    }
    #[test]
    fn completion_cycles_original_token_and_replaces_quoted_suffix() {
        let mut c = Composer::default();
        c.set_text("\\o").unwrap();
        key(&mut c, KeyCode::Tab, KeyModifiers::NONE);
        let first = c.text().to_owned();
        assert!(c.completion().unwrap().0.len() >= 2);
        key(&mut c, KeyCode::BackTab, KeyModifiers::SHIFT);
        assert_ne!(c.text(), first);
        key(&mut c, KeyCode::Tab, KeyModifiers::NONE);
        assert_eq!(c.text(), first);
        let before = c.view(60, 6);
        assert_eq!(
            c.handle(Event::Resize(60, 24), &CompletionData::default())
                .unwrap(),
            ComposerAction::Redraw
        );
        assert_eq!(c.view(60, 6), before);
        c.set_text("\\opeBAD other").unwrap();
        key(&mut c, KeyCode::Home, KeyModifiers::NONE);
        for _ in 0..4 {
            key(&mut c, KeyCode::Right, KeyModifiers::NONE);
        }
        key(&mut c, KeyCode::Tab, KeyModifiers::NONE);
        assert_eq!(c.text(), "\\open other");
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("folder/name with space.csv");
        let paths = [path];
        let data = CompletionData {
            paths: &paths,
            working_directory: Some(directory.path()),
            ..Default::default()
        };
        c.set_text("\\open \"folder/naBAD\" --name x").unwrap();
        c.cursor = "\\open \"folder/na".len();
        c.handle(
            Event::Key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE)),
            &data,
        )
        .unwrap();
        assert!(c.text().contains("name with space.csv"));
        assert!(c.text().ends_with(" --name x"));
        assert!(!c.text().contains("BAD"));
    }
    #[test]
    fn view_tracks_cells_scrolls_and_never_changes_edit_state() {
        let mut c = Composer::default();
        c.set_text("e\u{301}中👩‍💻").unwrap();
        assert_eq!(c.view(40, 4).cursor, (7, 0));
        c.set_text("first\nsecond\nthird\nfourth\nfifth").unwrap();
        let text = c.text().to_owned();
        let cursor = c.cursor_byte();
        let view = c.view(40, 4);
        assert_eq!(view.cursor.1, 3);
        assert!(view.lines.last().unwrap().ends_with("fifth"));
        for width in [0, 1, 2, 3, 10, 80] {
            let v = c.view(width, 4);
            if width > 0 {
                assert!(v.cursor.0 < width);
            }
            assert_eq!(c.text(), text);
            assert_eq!(c.cursor_byte(), cursor);
        }
        c.set_text("abcd").unwrap();
        assert_eq!(c.view(6, 4).cursor, (2, 1));
        c.set_text("abcd\nz").unwrap();
        c.cursor = 4;
        assert_eq!(c.view(6, 4).cursor, (2, 1));
    }
    fn create_history(path: &Path, entries: &[String]) {
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        owner_mode(&mut options);
        let mut file = options.open(path).unwrap();
        for e in entries {
            writeln!(file, "{}", serde_json::to_string(e).unwrap()).unwrap();
        }
    }
    #[test]
    fn history_retention_permissions_multiline_and_json_privacy_roundtrip() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("history");
        create_history(
            &path,
            &(0..1005).map(|i| format!("entry{i}")).collect::<Vec<_>>(),
        );
        let mut h = HistoryFile::open(&path).unwrap();
        assert_eq!(h.entries().len(), 1000);
        assert_eq!(h.entries()[0], "entry5");
        h.push("entry1004").unwrap();
        assert_eq!(h.entries().len(), 1000);
        h.push("\\open \"line one\nline two\"").unwrap();
        let small = format!("{{\"x\":\"{}\"}}", "a".repeat(1016));
        assert_eq!(small.len(), 1024);
        h.push(&format!("\\objective {small}")).unwrap();
        let count = h.entries().len();
        let last = h.entries().last().unwrap().clone();
        let big = format!("{{\"x\":\"{}\"}}", "a".repeat(1017));
        for command in ["\\objective", "/objective", "\\FITNESS"] {
            h.push(&format!("{command} {big}")).unwrap();
            assert_eq!(h.entries().len(), count);
            assert_eq!(h.entries().last().unwrap(), &last);
        }
        let reopened = HistoryFile::open(&path).unwrap();
        assert_eq!(reopened.entries(), h.entries());
        assert!(reopened
            .entries()
            .iter()
            .any(|s| s.contains("line one\nline two")));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                fs::metadata(&path).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
    }
    #[test]
    fn history_navigation_and_reverse_search_restore_the_original_draft() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("history");
        let mut h = HistoryFile::open(&path).unwrap();
        for entry in ["\\open alpha.csv", "\\open beta.csv", "\\run"] {
            h.push(entry).unwrap();
        }
        let mut c = Composer::with_history(&path).unwrap();
        c.set_text("my draft").unwrap();
        key(&mut c, KeyCode::Left, KeyModifiers::NONE);
        let cursor = c.cursor_byte();
        key(&mut c, KeyCode::Up, KeyModifiers::NONE);
        assert_eq!(c.text(), "\\run");
        key(&mut c, KeyCode::Down, KeyModifiers::NONE);
        assert_eq!(c.text(), "my draft");
        assert_eq!(c.cursor_byte(), cursor);
        key(&mut c, KeyCode::Char('r'), KeyModifiers::CONTROL);
        for ch in "open".chars() {
            key(&mut c, KeyCode::Char(ch), KeyModifiers::NONE);
        }
        assert_eq!(c.text(), "\\open beta.csv");
        key(&mut c, KeyCode::Char('r'), KeyModifiers::CONTROL);
        assert_eq!(c.text(), "\\open alpha.csv");
        key(&mut c, KeyCode::Esc, KeyModifiers::NONE);
        assert_eq!(c.text(), "my draft");
        assert_eq!(c.cursor_byte(), cursor);
        key(&mut c, KeyCode::Char('r'), KeyModifiers::CONTROL);
        assert_eq!(
            key(&mut c, KeyCode::Enter, KeyModifiers::NONE),
            ComposerAction::Submit("\\run".into())
        );
        assert_eq!(c.text(), "");
    }
    #[cfg(unix)]
    #[test]
    fn insecure_nonregular_symlink_and_busy_history_refuse_without_overwriting() {
        use std::os::unix::fs::{symlink, PermissionsExt};
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("history");
        fs::write(&path, "private").unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).unwrap();
        assert!(HistoryFile::open(&path).is_err());
        assert_eq!(fs::read_to_string(&path).unwrap(), "private");
        fs::remove_file(&path).unwrap();
        let target = directory.path().join("target");
        create_history(&target, &["outside".into()]);
        symlink(&target, &path).unwrap();
        assert!(HistoryFile::open(&path).is_err());
        assert!(fs::read_to_string(&target).unwrap().contains("outside"));
        fs::remove_file(&path).unwrap();
        let mut h = HistoryFile::open(&path).unwrap();
        h.push("before").unwrap();
        let lock = regular_options()
            .write(true)
            .open(path.with_extension("lock"))
            .unwrap();
        lock.try_lock().unwrap();
        assert!(h.push("blocked").is_err());
        drop(lock);
        assert_eq!(HistoryFile::open(&path).unwrap().entries(), ["before"]);
        #[cfg(target_os = "linux")]
        {
            let fifo = directory.path().join("fifo");
            assert!(std::process::Command::new("mkfifo")
                .arg(&fifo)
                .status()
                .unwrap()
                .success());
            let start = Instant::now();
            assert!(HistoryFile::open(&fifo).is_err());
            assert!(start.elapsed() < Duration::from_secs(1));
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    #[ignore = "supervised real-PTY child; explicitly executed by the U1 parent"]
    fn pty_composer_probe() {
        use super::super::restore::TerminalGuard;
        use ratatui::crossterm::event;
        use ratatui::widgets::Widget;
        use ratatui::{backend::CrosstermBackend, Terminal, TerminalOptions, Viewport};
        let history = std::env::var_os("JOPE_COMPOSER_HISTORY").unwrap();
        let report = PathBuf::from(std::env::var_os("JOPE_COMPOSER_REPORT").unwrap());
        let mut composer = Composer::with_history(Path::new(&history)).unwrap();
        println!("COMPOSER_START");
        io::stdout().flush().unwrap();
        let mut go = String::new();
        io::stdin().read_line(&mut go).unwrap();
        assert_eq!(go.trim(), "go");
        let guard = TerminalGuard::acquire(true).unwrap();
        println!("COMPOSER_ENTER");
        io::stdout().flush().unwrap();
        let mut terminal = Terminal::with_options(
            CrosstermBackend::new(io::stdout()),
            TerminalOptions {
                viewport: Viewport::Inline(8),
            },
        )
        .unwrap();
        let (send, receive) = std::sync::mpsc::sync_channel(256);
        let started = Instant::now();
        let producer = std::thread::spawn(move || {
            let start = Instant::now();
            for i in 0..150_u64 {
                let deadline = start + Duration::from_millis(i * 20);
                std::thread::sleep(deadline.saturating_duration_since(Instant::now()));
                if send.send(i).is_err() {
                    break;
                }
            }
        });
        let mut events = 0;
        let mut submissions = 0;
        let mut snapshots = Vec::new();
        let mut clock = super::super::app::RedrawClock::new();
        loop {
            while let Ok(sequence) = receive.try_recv() {
                events += 1;
                terminal
                    .insert_before(1, |buffer| {
                        Paragraph::new(format!("EVENT {sequence}")).render(buffer.area, buffer)
                    })
                    .unwrap();
            }
            if event::poll(Duration::from_millis(5)).unwrap() {
                let input = event::read().unwrap();
                if matches!(&input,Event::Key(k) if k.code==KeyCode::F(12)) {
                    let view = composer.view(80, 7);
                    let menu=composer.completion().map(|(s,i)|serde_json::json!({"selected":i,"choices":s.iter().map(|s|&s.insert).collect::<Vec<_>>()}));
                    snapshots.push(serde_json::json!({"text":composer.text(),"cursor":composer.cursor_byte(),"view_lines":view.lines,"view_cursor":view.cursor,"menu":menu,"events":events,"submissions":submissions,"elapsed_ms":started.elapsed().as_millis()}));
                    fs::write(&report, serde_json::to_vec(&snapshots).unwrap()).unwrap();
                } else {
                    match composer.handle(input, &CompletionData::default()).unwrap() {
                        ComposerAction::ExitRequested => break,
                        ComposerAction::Submit(_) => submissions += 1,
                        _ => {}
                    }
                }
            }
            if clock.remaining(Instant::now(), true).is_zero() {
                terminal
                    .draw(|frame| {
                        let area = frame.area();
                        frame.render_widget(
                            Paragraph::new(format!("COMPOSER_READY events={events}")),
                            Rect::new(area.x, area.y, area.width, 1),
                        );
                        composer.draw(
                            frame,
                            Rect::new(
                                area.x,
                                area.y + 1,
                                area.width,
                                area.height.saturating_sub(1),
                            ),
                        );
                    })
                    .unwrap();
                clock.drawn(Instant::now());
            }
        }
        drop(receive);
        producer.join().unwrap();
        drop(terminal);
        drop(guard);
        println!("COMPOSER_RESTORED");
        io::stdout().flush().unwrap();
    }
    #[cfg(target_os = "linux")]
    #[test]
    fn linux_pty_typing_paste_cursor_and_menu_survive_fifty_hz_events() {
        let directory = tempfile::tempdir().unwrap();
        let history = directory.path().join("history");
        let report = directory.path().join("report.json");
        let executable = std::env::current_exe().unwrap();
        let h = format!("JOPE_COMPOSER_HISTORY={}", history.display());
        let r = format!("JOPE_COMPOSER_REPORT={}", report.display());
        let mut session = pty::PtySession::spawn(
            "/usr/bin/env",
            &[
                &h,
                &r,
                "RUST_BACKTRACE=0",
                executable.to_str().unwrap(),
                "--exact",
                "terminal::composer::tests::pty_composer_probe",
                "--ignored",
                "--nocapture",
                "--color=never",
                "--test-threads=1",
            ],
        )
        .unwrap();
        session
            .wait_for_text("COMPOSER_START", Duration::from_secs(10))
            .unwrap();
        session.type_text("go\n").unwrap();
        session
            .wait_for_text("COMPOSER_ENTER", Duration::from_secs(10))
            .unwrap();
        session.type_text("\x1b[1;1R").unwrap();
        session
            .wait_for_text("COMPOSER_READY", Duration::from_secs(10))
            .unwrap();
        let snapshot = |session: &mut pty::PtySession, count: usize| {
            session.type_text("\x1b[24~").unwrap();
            let deadline = Instant::now() + Duration::from_secs(5);
            loop {
                if let Ok(bytes) = fs::read(&report) {
                    if let Ok(rows) = serde_json::from_slice::<Vec<serde_json::Value>>(&bytes) {
                        if rows.len() >= count {
                            return rows[count - 1].clone();
                        }
                    }
                }
                assert!(
                    Instant::now() < deadline,
                    "missing PTY snapshot: {}",
                    session.read_screen_as_text()
                );
                std::thread::sleep(Duration::from_millis(10));
            }
        };
        session.type_text("e\u{301}中").unwrap();
        session.type_text("\x1b[200~pasted\ntext\x1b[201~").unwrap();
        session.press_key(pty::Key::Left).unwrap();
        let typed = snapshot(&mut session, 1);
        assert_eq!(typed["text"], "e\u{301}中pasted\ntext");
        assert_eq!(
            typed["cursor"].as_u64().unwrap() as usize,
            "e\u{301}中pasted\ntex".len()
        );
        std::thread::sleep(Duration::from_millis(440));
        let after = snapshot(&mut session, 2);
        for field in ["text", "cursor", "view_lines", "view_cursor", "menu"] {
            assert_eq!(typed[field], after[field], "{field}");
        }
        assert!(after["events"].as_u64().unwrap() >= typed["events"].as_u64().unwrap() + 18);
        session.type_text("\x03\\o\t").unwrap();
        let menu = snapshot(&mut session, 3);
        assert!(!menu["menu"].is_null());
        std::thread::sleep(Duration::from_millis(440));
        let menu_after = snapshot(&mut session, 4);
        for field in ["text", "cursor", "view_lines", "view_cursor", "menu"] {
            assert_eq!(menu[field], menu_after[field], "{field}");
        }
        assert!(menu_after["events"].as_u64().unwrap() >= menu["events"].as_u64().unwrap() + 18);
        session.type_text("\x03\x04").unwrap();
        let _ = session
            .wait_for_text("COMPOSER_RESTORED", Duration::from_secs(10))
            .unwrap();
        let screen = session
            .wait_for_text("1 passed; 0 failed", Duration::from_secs(10))
            .unwrap();
        for point in [&typed, &after, &menu, &menu_after] {
            assert_eq!(point["submissions"], 0);
        }
        if let Some(path) = std::env::var_os("JOPE_COMPOSER_EVIDENCE_DIR") {
            fs::write(
                Path::new(&path).join("u1-snapshots.json"),
                fs::read(&report).unwrap(),
            )
            .unwrap();
            fs::write(Path::new(&path).join("u1-screen.txt"), screen).unwrap();
        }
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            fs::metadata(&history).unwrap().permissions().mode() & 0o777,
            0o600
        );
    }
}
