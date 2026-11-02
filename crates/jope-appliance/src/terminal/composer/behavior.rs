use super::*;

/// Byte boundaries from ratatui's existing unicode-segmentation backend. Its Span
/// iterator excludes LF, so handle each LF explicitly rather than losing it.

pub(super) fn boundaries(text: &str) -> Vec<usize> {
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
pub(super) fn clean(text: &str) -> String {
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
        // Exit takes precedence over completion, history search, and unsent text.
        if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
            return Ok(ComposerAction::ExitRequested);
        }
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
