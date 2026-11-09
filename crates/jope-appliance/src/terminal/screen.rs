//! Full-screen landing, bounded transcript, and a composer independent of scrolling.
use std::collections::VecDeque;

use ratatui::{
    crossterm::event::{Event, KeyCode, KeyEventKind, MouseEventKind},
    layout::{Alignment, Constraint, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{
        Block, Clear, List, ListItem, ListState, Paragraph, Scrollbar, ScrollbarOrientation,
        ScrollbarState,
    },
    Frame,
};

use super::{composer::Composer, links, sanitize_user_string, ColorLevel, Theme, ThemeMode};

const MAX_LINES: usize = 2000;
const MAX_BYTES: usize = 1024 * 1024;
const TAGLINE: &str = "Your data. Your models. Your next discovery.";

/// What a transcript line is, which decides how it is drawn.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Kind {
    /// Output from jAIn: command results, run cards and notices.
    #[default]
    Output,
    /// A command the user submitted.
    Input,
    /// The search's reasoning ("Thinking" cards).
    Thinking,
    /// A completed step (`✓ ...`).
    Success,
    /// A failed command (`error: ...`).
    Error,
}

/// Columns every transcript row reserves for its gutter (`❯ `, `⎿ `, `┊ `, ...).
const GUTTER: u16 = 2;

#[derive(Default)]
pub struct Screen {
    pub browser: Option<super::runs::Browser>,
    pub cloud_label: String,
    /// A modal dialog (first run) shown instead of the landing or transcript.
    pub panel: Option<Vec<String>>,
    /// Trusted URLs drawn in the last frame, for the OSC 8 overlay (`links::overlay`).
    pub links: Vec<links::Spot>,
    /// Whether the mouse is captured (wheel scrolling) rather than left to the terminal's own
    /// text selection; shown in the footer.
    pub mouse: bool,
    lines: VecDeque<(Kind, String)>,
    bytes: usize,
    wrapped: Vec<Wrapped>,
    width: u16,
    dirty: bool,
    top: Option<usize>,
    unread: usize,
    page_height: usize,
    started: Option<std::time::Instant>,
}

impl Screen {
    /// A command the user submitted, kept as `> text` in the plain transcript.
    pub fn push_input(&mut self, text: &str) {
        self.push_kind(Some(Kind::Input), &format!("> {text}"));
    }

    /// Output from jAIn; thinking cards, `✓` steps and `error:` lines are recognised.
    pub fn push(&mut self, text: &str) {
        self.push_kind(None, text);
    }

    fn classify(&self, line: &str) -> Kind {
        let previous = self.lines.back().map(|(kind, _)| *kind);
        if line.starts_with("  Thinking ") || line == "  Thinking" {
            Kind::Thinking
        } else if previous == Some(Kind::Thinking) && line.starts_with("   ") {
            // A wrapped thinking card continues with its hanging indent.
            Kind::Thinking
        } else if line.starts_with("error:") {
            Kind::Error
        } else if line.starts_with('✓') {
            Kind::Success
        } else {
            Kind::Output
        }
    }

    fn push_kind(&mut self, kind: Option<Kind>, text: &str) {
        let clean = sanitize_user_string(text);
        for line in clean.lines() {
            // Transcript is a bounded view, not the durable run/event store.
            let line: String = line.chars().take(8192).collect();
            let kind = kind.unwrap_or_else(|| self.classify(&line));
            self.bytes += line.len();
            self.lines.push_back((kind, line));
            if self.top.is_some() {
                self.unread += 1;
            }
            while self.lines.len() > MAX_LINES || self.bytes > MAX_BYTES {
                if let Some((_, old)) = self.lines.pop_front() {
                    self.bytes -= old.len();
                    if let Some(top) = self.top.as_mut() {
                        *top = top.saturating_sub(wrap(&old, self.width.max(1)).len());
                    }
                }
            }
        }
        self.dirty = true;
    }

    fn reflow(&mut self, width: u16) {
        if self.dirty || self.width != width {
            self.width = width;
            self.wrapped = rows(&self.lines, width);
            self.dirty = false;
        }
    }

    /// The landing shows until the first transcript line or the runs panel.
    pub fn is_landing(&self) -> bool {
        self.browser.is_none() && self.panel.is_none() && self.lines.is_empty()
    }

    /// Whether the logo is on screen: the landing, or above the first-run panel.
    fn shows_logo(&self) -> bool {
        self.browser.is_none() && self.lines.is_empty()
    }

    /// Whether the landing logo needs frequent redraws.
    pub fn animating(&self) -> bool {
        self.shows_logo()
            && super::splash::animated()
            && self
                .started
                .is_none_or(|started| super::splash::running(started.elapsed()))
    }

    pub fn transcript(&self) -> Vec<String> {
        self.lines.iter().map(|(_, line)| line.clone()).collect()
    }

    pub fn follow(&mut self) {
        self.top = None;
        self.unread = 0;
    }

    /// Page and wheel keys scroll results without entering the composer/history editor.
    pub fn handle_scroll(&mut self, event: &Event) -> bool {
        let page = self.page_height.saturating_sub(1).max(1) as isize;
        let delta = match event {
            Event::Mouse(mouse) => match mouse.kind {
                MouseEventKind::ScrollUp => -3,
                MouseEventKind::ScrollDown => 3,
                _ => return false,
            },
            Event::Key(key) if key.kind != KeyEventKind::Release => match key.code {
                KeyCode::PageUp => -page,
                KeyCode::PageDown => page,
                KeyCode::End if self.top.is_some() => {
                    self.follow();
                    return true;
                }
                _ => return false,
            },
            _ => return false,
        };
        let max = self.wrapped.len().saturating_sub(self.page_height);
        let next = self
            .top
            .unwrap_or(max)
            .saturating_add_signed(delta)
            .min(max);
        if next == max {
            self.follow();
        } else {
            self.top = Some(next);
        }
        true
    }

    pub fn draw(
        &mut self,
        frame: &mut Frame<'_>,
        composer: &Composer,
        version: &str,
        status: &str,
        activity: &str,
    ) {
        let area = frame.area();
        let theme = Theme::from_env(ThemeMode::Auto);
        let light = theme.mode() == ThemeMode::Light;
        let accent = Style::default().fg(if light { Color::Blue } else { Color::Cyan });
        let muted = Style::default().fg(Color::DarkGray);
        let (accent, muted) = if theme.color_level() == ColorLevel::None {
            (Style::default(), Style::default())
        } else {
            (accent, muted)
        };
        let input_height = composer
            .view_with(area.width.saturating_sub(4).max(1), 6, false)
            .lines
            .len() as u16
            + 2;
        let rows = Layout::vertical([
            Constraint::Length(1),
            Constraint::Min(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(input_height),
            Constraint::Length(1),
        ])
        .split(area);
        self.links.clear();
        frame.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled(" jAIn by VEOX ", accent.add_modifier(Modifier::BOLD)),
                Span::styled(format!("· CLI {version}"), muted),
            ])),
            rows[0],
        );
        let body = Rect {
            x: rows[1].x + 1,
            width: rows[1].width.saturating_sub(2),
            ..rows[1]
        };
        self.reflow(body.width.max(1));
        self.page_height = usize::from(body.height);
        if let Some(panel) = &self.panel {
            // Onboarding: the jAIn logo above the first-run panel when there is room for both.
            let panel_height = (panel.len() as u16 + 2).min(body.height);
            let logo_height = body.height.saturating_sub(panel_height + 1);
            let started = *self.started.get_or_insert_with(std::time::Instant::now);
            let elapsed = if super::splash::animated() {
                started.elapsed()
            } else {
                super::splash::still()
            };
            let palette = super::splash::Palette::new(theme.color_level(), theme.mode());
            match logo_lines(body.width, logo_height, elapsed, &palette) {
                Some(logo) => {
                    let height = logo.len() as u16;
                    let logo_area = Rect {
                        y: body.y + logo_height.saturating_sub(height),
                        height,
                        ..body
                    };
                    frame.render_widget(
                        Paragraph::new(logo).alignment(Alignment::Center),
                        logo_area,
                    );
                    let panel_area = Rect {
                        y: body.y + logo_height + 1,
                        height: panel_height,
                        ..body
                    };
                    self.links = draw_panel(frame, panel_area, panel, accent);
                }
                None => self.links = draw_panel(frame, body, panel, accent),
            }
        } else if let Some(browser) = &mut self.browser {
            browser.draw(frame, body, accent);
        } else if self.lines.is_empty() {
            let started = *self.started.get_or_insert_with(std::time::Instant::now);
            let elapsed = if super::splash::animated() {
                started.elapsed()
            } else {
                // Reduced motion uses the finished blue logo.
                super::splash::still()
            };
            let palette = super::splash::Palette::new(theme.color_level(), theme.mode());
            landing(frame, body, elapsed, &palette, muted);
        } else {
            let max = self.wrapped.len().saturating_sub(self.page_height);
            let start = self.top.map(|top| top.min(max)).unwrap_or(max);
            self.top = self.top.map(|_| start);
            let visible = self.wrapped.iter().skip(start).take(self.page_height);
            for (row, line) in visible.clone().enumerate() {
                for (column, url) in &line.links {
                    self.links.push(links::Spot {
                        x: body.x + column,
                        y: body.y + row as u16,
                        url: url.clone(),
                    });
                }
            }
            let styles = Styles::new(&theme, accent, muted);
            frame.render_widget(
                Paragraph::new(
                    visible
                        .map(|row| styles.line(row, usize::from(body.width)))
                        .collect::<Vec<_>>(),
                ),
                body,
            );
            if self.wrapped.len() > self.page_height {
                let mut state = ScrollbarState::new(max).position(start);
                frame.render_stateful_widget(
                    Scrollbar::new(ScrollbarOrientation::VerticalRight)
                        .begin_symbol(None)
                        .end_symbol(None)
                        .track_symbol(Some("│"))
                        .thumb_symbol("┃")
                        .style(muted)
                        .thumb_style(accent),
                    Rect {
                        x: rows[1].x,
                        width: rows[1].width,
                        ..body
                    },
                    &mut state,
                );
            }
        }
        let landing_view = (self.is_landing() || self.panel.is_some()) && status == IDLE_STATUS;
        if landing_view {
            let connected = !self.cloud_label.is_empty();
            frame.render_widget(
                Paragraph::new(guidance(usize::from(rows[3].width), connected)).style(muted),
                rows[3],
            );
        } else {
            frame.render_widget(
                Paragraph::new(sanitize_user_string(status)).style(accent),
                rows[2],
            );
            frame.render_widget(
                Paragraph::new(sanitize_user_string(activity)).style(muted),
                rows[3],
            );
        }
        let block = Block::bordered().title(" Command ").border_style(accent);
        let input = block.inner(rows[4]);
        frame.render_widget(block, rows[4]);
        composer.draw_with(frame, input, false);
        let menu_open = draw_menu(frame, composer, rows[1], rows[4], accent);
        let footer = if menu_open {
            " Up/Down choose · Enter opens a folder or runs · Tab next · Esc closes".into()
        } else if self.panel.is_some() {
            " Answer above to continue · Ctrl-C exits".into()
        } else if self.browser.is_some() {
            " Up/Down choose · Enter attach/open · o cloud · d download · r refresh · Esc/right back · Ctrl-C exit".into()
        } else if landing_view {
            // Instructions live in the guidance sentence; keep only account state here.
            if self.cloud_label.is_empty() {
                String::new()
            } else {
                format!(" {}", self.cloud_label)
            }
        } else if self.top.is_some() {
            format!(
                " {} new lines · PgUp/PgDn scroll · End follows live · {} · Ctrl-C exits",
                self.unread,
                mouse_hint(self.mouse)
            )
        } else {
            format!(
                " Left: runs · PgUp/PgDn: scroll · {} · /copy · Ctrl-C: exit · {}",
                mouse_hint(self.mouse),
                if self.cloud_label.is_empty() {
                    "Cloud not connected · /login"
                } else {
                    &self.cloud_label
                }
            )
        };
        frame.render_widget(
            Paragraph::new(sanitize_user_string(&footer)).style(muted),
            rows[5],
        );
    }
}

/// The footer's mouse state: what a drag does now, and the key that changes it.
pub fn mouse_hint(capture: bool) -> &'static str {
    if capture {
        "F2 mouse: scroll (Shift-drag selects)"
    } else {
        "F2 mouse: select text"
    }
}

/// Status text shown before any run or message; the landing replaces it with guidance.
pub const IDLE_STATUS: &str = "ready · no active run";

/// All landing instructions as one sentence, shortened to fit narrow terminals.
fn guidance(width: usize, connected: bool) -> String {
    let login = if connected {
        ""
    } else {
        ", /login for your own data (accounts open from the waitlist: https://app.veox.ai/register)"
    };
    let full = format!(
        " Type /demo to try a bundled example (your own data trains in the veox Cloud), press Left to browse your runs{login}, /help for every command, or Ctrl-C to exit."
    );
    if full.chars().count() <= width {
        return full;
    }
    let short = " Type /demo or /open <file.csv> to start, /help for commands, or Ctrl-C to exit.";
    if short.chars().count() <= width {
        short.into()
    } else {
        " /demo · /open <file.csv> · /help · Ctrl-C".into()
    }
}

/// Most completion choices shown at once; the list scrolls with the selection.
const MENU_ROWS: usize = 8;

/// The completion choices as a list just above the command box, when there is more than one.
/// Returns whether it was drawn.
fn draw_menu(
    frame: &mut Frame<'_>,
    composer: &Composer,
    body: Rect,
    command: Rect,
    accent: Style,
) -> bool {
    let Some((choices, selected)) = composer.completion() else {
        return false;
    };
    let room = command.y.saturating_sub(body.y);
    if choices.len() < 2 || room < 3 {
        return false;
    }
    let labels: Vec<String> = choices
        .iter()
        .map(|choice| sanitize_user_string(&choice.insert))
        .collect();
    let title = format!(" {}/{} ", selected + 1, choices.len());
    let longest = labels
        .iter()
        .map(|label| Span::raw(label.as_str()).width())
        .max()
        .unwrap_or(0)
        .max(title.len()) as u16;
    let height = (choices.len().min(MENU_ROWS) as u16 + 2).min(room);
    let rect = Rect {
        x: command.x,
        y: command.y - height,
        // Full width, so nothing behind it shows through beside the list.
        width: command.width.max((longest + 4).min(command.width)),
        height,
    };
    let list = List::new(labels.into_iter().map(ListItem::new).collect::<Vec<_>>())
        .block(Block::bordered().title(title).border_style(accent))
        .highlight_style(accent.add_modifier(Modifier::REVERSED));
    let mut state = ListState::default().with_selected(Some(selected));
    frame.render_widget(Clear, rect);
    frame.render_stateful_widget(list, rect, &mut state);
    true
}

/// The splash logo for an area, or `None` when it does not fit.
fn logo_lines(
    width: u16,
    height: u16,
    elapsed: std::time::Duration,
    palette: &super::splash::Palette,
) -> Option<Vec<Line<'static>>> {
    super::splash::fit(width, height)
        .map(|(columns, rows_per_px)| super::splash::frame(elapsed, columns, rows_per_px, palette))
}

fn landing(
    frame: &mut Frame<'_>,
    area: Rect,
    elapsed: std::time::Duration,
    palette: &super::splash::Palette,
    muted: Style,
) {
    let mut lines = match super::splash::fit(area.width, area.height) {
        Some((columns, rows_per_px)) => {
            super::splash::frame(elapsed, columns, rows_per_px, palette)
        }
        None => vec![Line::styled(
            "J A I N",
            Style::default().add_modifier(Modifier::BOLD),
        )],
    };
    lines.push(Line::raw(""));
    lines.push(Line::styled(TAGLINE, muted));
    let height = (lines.len() as u16).min(area.height);
    let centered = Rect {
        y: area.y + area.height.saturating_sub(height) / 2,
        height,
        ..area
    };
    frame.render_widget(Paragraph::new(lines).alignment(Alignment::Center), centered);
}

/// One wrapped row and the trusted URLs that lie wholly inside it, as (column, url).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Wrapped {
    text: String,
    links: Vec<(u16, String)>,
    kind: Kind,
    gutter: &'static str,
}

/// Wrap typed transcript lines into drawn rows: a blank row before each command, a `❯`
/// gutter on commands, `⎿` joining the first reply to its command, a `┊` rail beside
/// thinking. Link columns include the gutter.
fn rows(lines: &VecDeque<(Kind, String)>, width: u16) -> Vec<Wrapped> {
    let text_width = width.saturating_sub(GUTTER).max(1);
    let mut out = Vec::new();
    let mut previous: Option<Kind> = None;
    for (kind, line) in lines {
        let kind = *kind;
        if kind == Kind::Input && previous.is_some() {
            out.push(Wrapped::default());
        }
        let (body, first, rest): (&str, &'static str, &'static str) = match kind {
            Kind::Input => (line.strip_prefix("> ").unwrap_or(line), "❯ ", "  "),
            Kind::Thinking if previous != Some(Kind::Thinking) => (line.trim_start(), "✻ ", "┊ "),
            Kind::Thinking => (line.trim_start(), "┊ ", "┊ "),
            Kind::Error => (line.strip_prefix("error: ").unwrap_or(line), "✗ ", "  "),
            _ if previous == Some(Kind::Input) => (line.as_str(), "⎿ ", "  "),
            _ => (line.as_str(), "  ", "  "),
        };
        for (index, row) in wrap_links(body, text_width).into_iter().enumerate() {
            out.push(Wrapped {
                text: row.text,
                links: row
                    .links
                    .into_iter()
                    .map(|(column, url)| (column + GUTTER, url))
                    .collect(),
                kind,
                gutter: if index == 0 { first } else { rest },
            });
        }
        previous = Some(kind);
    }
    out
}

/// Row styles for the transcript, degrading to plain text when colour is off.
struct Styles {
    accent: Style,
    muted: Style,
    input_bar: Style,
    thinking: Style,
    success: Style,
    error: Style,
}

impl Styles {
    fn new(theme: &Theme, accent: Style, muted: Style) -> Self {
        if theme.color_level() == ColorLevel::None {
            let plain = Style::default();
            return Self {
                accent: plain.add_modifier(Modifier::BOLD),
                muted: plain,
                input_bar: plain.add_modifier(Modifier::BOLD),
                thinking: plain.add_modifier(Modifier::ITALIC),
                success: plain,
                error: plain.add_modifier(Modifier::BOLD),
            };
        }
        let light = theme.mode() == ThemeMode::Light;
        Self {
            accent,
            muted,
            input_bar: Style::default()
                .bg(if light {
                    Color::Indexed(254)
                } else {
                    Color::Indexed(236)
                })
                .add_modifier(Modifier::BOLD),
            thinking: Style::default()
                .fg(if light {
                    Color::Indexed(97)
                } else {
                    Color::Indexed(183)
                })
                .add_modifier(Modifier::ITALIC),
            success: Style::default().fg(Color::Green),
            error: Style::default().fg(Color::Red),
        }
    }

    fn line(&self, row: &Wrapped, width: usize) -> Line<'static> {
        let (gutter, text) = match row.kind {
            Kind::Input => (self.accent.add_modifier(Modifier::BOLD), self.input_bar),
            Kind::Thinking => (
                self.thinking.remove_modifier(Modifier::ITALIC),
                self.thinking,
            ),
            Kind::Success => (self.success, Style::default()),
            Kind::Error => (self.error, self.error),
            Kind::Output => (self.muted, Style::default()),
        };
        let mut spans = vec![
            Span::styled(row.gutter, gutter),
            Span::styled(row.text.clone(), text),
        ];
        if row.kind == Kind::Success {
            if let Some(rest) = row.text.strip_prefix('✓') {
                spans.truncate(1);
                spans.push(Span::styled("✓", self.success));
                spans.push(Span::raw(rest.to_owned()));
            }
        }
        if row.kind == Kind::Input {
            // A full-width bar sets each command apart from what jAIn printed.
            let used = Span::raw(row.gutter).width() + Span::raw(row.text.as_str()).width();
            spans.push(Span::styled(
                " ".repeat(width.saturating_sub(used)),
                self.input_bar,
            ));
            return Line::from(spans).style(self.input_bar);
        }
        Line::from(spans)
    }
}

fn wrap(text: &str, width: u16) -> Vec<String> {
    wrap_links(text, width)
        .into_iter()
        .map(|line| line.text)
        .collect()
}

/// Wrap at grapheme boundaries. A trusted URL that fits the width starts a new row rather
/// than being split, so it stays whole and clickable; a URL wider than the row is wrapped
/// and shown as plain text only.
fn wrap_links(text: &str, width: u16) -> Vec<Wrapped> {
    let width = usize::from(width.max(1));
    let text = text.replace('\t', "    ");
    let urls: Vec<(usize, usize)> = links::find_urls(&text)
        .into_iter()
        .filter(|(start, end)| links::trusted(&text[*start..*end]))
        .collect();
    // (row text, starting byte offset of each row in `text`)
    let mut rows: Vec<(String, usize, usize)> = Vec::new();
    let mut line = String::new();
    let mut line_start = 0;
    let mut cells = 0;
    // After the last space on the current row: (byte index in `line`, offset in `text`, cells).
    let mut last_space: Option<(usize, usize, usize)> = None;
    for (offset, symbol) in graphemes(&text) {
        let size = Span::raw(symbol).width();
        let url_fits = urls
            .iter()
            .find(|(start, _)| *start == offset)
            .map(|(start, end)| end - start)
            .filter(|len| *len <= width);
        let breaks = match url_fits {
            Some(len) => cells + len > width,
            None => cells + size > width,
        };
        if breaks && !line.is_empty() && symbol == " " {
            // The break falls on a space: end the row here and drop the space.
            rows.push((std::mem::take(&mut line), line_start, offset));
            line_start = offset + symbol.len();
            cells = 0;
            last_space = None;
            continue;
        }
        if breaks && !line.is_empty() {
            match last_space.filter(|(_, _, at)| *at > 0 && cells - at + size <= width) {
                // Word wrap: carry the partial word after the last space to the next row.
                Some((index, text_offset, at))
                    if !symbol.trim().is_empty() && index < line.len() =>
                {
                    let carry = line.split_off(index);
                    rows.push((std::mem::take(&mut line), line_start, text_offset));
                    line = carry;
                    line_start = text_offset;
                    cells -= at;
                }
                _ => {
                    rows.push((std::mem::take(&mut line), line_start, offset));
                    line_start = offset;
                    cells = 0;
                }
            }
            last_space = None;
        }
        if size > width {
            line.push('?');
            cells += 1;
        } else {
            line.push_str(symbol);
            cells += size;
        }
        if symbol == " " {
            last_space = Some((line.len(), offset + symbol.len(), cells));
        }
    }
    rows.push((line, line_start, text.len()));
    rows.into_iter()
        .map(|(row, from, to)| {
            let links = urls
                .iter()
                .filter(|(start, end)| *start >= from && *end <= to)
                .map(|(start, end)| {
                    let column = Span::raw(&text[from..*start]).width() as u16;
                    (column, text[*start..*end].to_owned())
                })
                .collect();
            Wrapped {
                text: row,
                links,
                ..Wrapped::default()
            }
        })
        .collect()
}

/// Grapheme clusters with their byte offsets, via ratatui's own segmentation. Callers pass
/// single logical lines, so no grapheme is dropped and offsets stay exact.
fn graphemes(text: &str) -> Vec<(usize, &str)> {
    let mut offset = 0;
    Span::raw(text)
        .styled_graphemes(Style::default())
        .map(|grapheme| {
            let at = offset;
            offset += grapheme.symbol.len();
            (at, &text[at..offset])
        })
        .collect()
}

/// A bordered dialog centered in `area`. Returns the trusted URLs drawn inside it.
fn draw_panel(
    frame: &mut Frame<'_>,
    area: Rect,
    lines: &[String],
    accent: Style,
) -> Vec<links::Spot> {
    let clean: Vec<String> = lines
        .iter()
        .map(|line| sanitize_user_string(line))
        .collect();
    let longest = clean
        .iter()
        .map(|line| Span::raw(line.as_str()).width())
        .max()
        .unwrap_or(0) as u16;
    let width = (longest + 4).min(area.width).max(1);
    let height = (clean.len() as u16 + 2).min(area.height).max(1);
    let rect = Rect {
        x: area.x + area.width.saturating_sub(width) / 2,
        y: area.y + area.height.saturating_sub(height) / 2,
        width,
        height,
    };
    let block = Block::bordered().border_style(accent);
    let inner = block.inner(rect);
    let inner = Rect {
        x: inner.x + 1,
        width: inner.width.saturating_sub(1),
        ..inner
    };
    frame.render_widget(block, rect);
    let mut spots = Vec::new();
    for (row, line) in clean.iter().enumerate().take(usize::from(inner.height)) {
        for (column, url) in links::spots_in(line) {
            if column + url.len() as u16 <= inner.width {
                spots.push(links::Spot {
                    x: inner.x + column,
                    y: inner.y + row as u16,
                    url,
                });
            }
        }
    }
    let styled: Vec<Line> = clean
        .into_iter()
        .enumerate()
        .map(|(index, line)| {
            if index == 0 {
                Line::styled(line, accent.add_modifier(Modifier::BOLD))
            } else {
                Line::from(line)
            }
        })
        .collect();
    frame.render_widget(Paragraph::new(styled), inner);
    spots
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::{
        backend::TestBackend,
        crossterm::event::{KeyEvent, KeyModifiers},
        Terminal,
    };

    #[test]
    fn landing_is_fullscreen_and_composer_stays_at_bottom() {
        for (width, height) in [(100, 30), (40, 15)] {
            let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
            let mut screen = Screen::default();
            let mut composer = Composer::default();
            composer.set_text("my pending draft").unwrap();
            terminal
                .draw(|frame| screen.draw(frame, &composer, "test", IDLE_STATUS, ""))
                .unwrap();
            let buffer = terminal.backend().buffer();
            let row = |y| {
                (0..width)
                    .map(|x| buffer[(x, y)].symbol())
                    .collect::<String>()
            };
            let output = (0..height).map(row).collect::<Vec<_>>().join("\n");
            if width >= 60 {
                assert!(output.contains(TAGLINE));
            }
            assert!(!output.contains("Powered by JopeDime"));
            assert!(output.contains("/open <file.csv>"));
            assert!(row(height - 3).contains("my pending draft"));
        }
    }

    #[test]
    fn scroll_resize_and_new_events_preserve_draft_and_history_position() {
        let mut screen = Screen::default();
        for i in 0..100 {
            screen.push(&format!("event {i} 中 e\u{301}"));
        }
        let mut composer = Composer::default();
        composer.set_text("draft that must survive").unwrap();
        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
        terminal
            .draw(|frame| screen.draw(frame, &composer, "test", "ready", ""))
            .unwrap();
        let page = Event::Key(KeyEvent::new(KeyCode::PageUp, KeyModifiers::NONE));
        assert!(screen.handle_scroll(&page));
        let anchor = screen.top.unwrap();
        screen.push("new background event");
        terminal
            .draw(|frame| screen.draw(frame, &composer, "test", "ready", ""))
            .unwrap();
        assert_eq!(screen.top, Some(anchor));
        assert_eq!(screen.unread, 1);
        terminal.backend_mut().resize(40, 15);
        terminal
            .draw(|frame| screen.draw(frame, &composer, "test", "ready", ""))
            .unwrap();
        assert_eq!(composer.text(), "draft that must survive");
        screen.handle_scroll(&Event::Key(KeyEvent::new(KeyCode::End, KeyModifiers::NONE)));
        assert_eq!(screen.top, None);
        assert_eq!(screen.unread, 0);
        assert_eq!(wrap("e\u{301}中x", 3), vec!["e\u{301}中", "x"]);
        assert_eq!(
            wrap("target target (regression) subset of the original", 20),
            vec!["target target ", "(regression) subset ", "of the original"]
        );
    }

    #[test]
    fn header_is_jain_by_veox_and_trusted_urls_stay_whole_and_are_reported_for_links() {
        let mut terminal = Terminal::new(TestBackend::new(60, 20)).unwrap();
        let mut screen = Screen::default();
        let composer = Composer::default();
        screen.push("Open https://app.veox.ai/cli/connect?user_code=ABCD-1234 then wait");
        screen.push("untrusted https://evil.example/x");
        terminal
            .draw(|frame| screen.draw(frame, &composer, "1.2.3", "ready", ""))
            .unwrap();
        let buffer = terminal.backend().buffer().clone();
        let row = |y: u16| (0..60).map(|x| buffer[(x, y)].symbol()).collect::<String>();
        assert!(row(0).contains("jAIn by VEOX"), "{}", row(0));
        assert_eq!(screen.links.len(), 1);
        let spot = &screen.links[0];
        assert_eq!(
            spot.url,
            "https://app.veox.ai/cli/connect?user_code=ABCD-1234"
        );
        // The URL moved to its own row instead of being split at the edge.
        assert!(row(spot.y)[usize::from(spot.x)..].starts_with(&spot.url));
        let mut out = Vec::new();
        links::overlay(&mut out, &screen.links, &buffer).unwrap();
        assert!(String::from_utf8(out)
            .unwrap()
            .contains("\u{1b}]8;;https://app.veox.ai/cli/connect"));
    }

    #[test]
    fn long_urls_wrap_as_plain_text_without_partial_links() {
        let url = format!("https://veox.ai/{}", "a".repeat(40));
        let rows = wrap_links(&format!("x {url}"), 20);
        assert!(rows.iter().all(|row| row.links.is_empty()));
        assert_eq!(
            rows.iter().map(|r| r.text.as_str()).collect::<String>(),
            format!("x {url}")
        );
        let rows = wrap_links("see https://veox.ai/terms now", 22);
        assert_eq!(rows[0].text, "see ");
        assert_eq!(rows[1].links, vec![(0, "https://veox.ai/terms".to_owned())]);
    }

    #[test]
    fn first_run_panel_replaces_landing_and_reports_the_terms_link() {
        let mut terminal = Terminal::new(TestBackend::new(100, 30)).unwrap();
        let mut screen = Screen {
            panel: Some(vec![
                "Welcome".into(),
                String::new(),
                "https://veox.ai/terms".into(),
            ]),
            ..Screen::default()
        };
        let composer = Composer::default();
        terminal
            .draw(|frame| screen.draw(frame, &composer, "t", IDLE_STATUS, ""))
            .unwrap();
        assert!(!screen.is_landing());
        assert_eq!(screen.links.len(), 1);
        assert_eq!(screen.links[0].url, "https://veox.ai/terms");
        let buffer = terminal.backend().buffer();
        let spot = &screen.links[0];
        let drawn: String = (0..21)
            .map(|i| buffer[(spot.x + i, spot.y)].symbol())
            .collect();
        assert_eq!(drawn, "https://veox.ai/terms");
        // The jAIn logo is drawn above the onboarding panel.
        let ink_above = (2..spot.y.saturating_sub(2))
            .any(|y| (0..100).any(|x| !buffer[(x, y)].symbol().trim().is_empty()));
        assert!(ink_above, "no logo above the first-run panel");
        assert!(screen.animating() || !crate::terminal::splash::animated());
    }

    #[test]
    fn transcript_sets_commands_replies_and_thinking_apart() {
        let mut screen = Screen::default();
        screen.push_input("/demo regression");
        screen.push("Demo regression_parkinsons · 1000 rows");
        screen.push("registered regression_parkinsons.csv");
        screen.push("  Thinking Trying a smoother kernel because the residuals");
        screen.push("           curve at high values · rmse 0.41 → 0.36");
        screen.push("✓ First model · CV rmse 0.067");
        screen.push_input("/open nope.csv");
        screen.push("error: no such file");
        // The plain transcript keeps exactly what was said.
        assert_eq!(screen.transcript()[0], "> /demo regression");
        let rows = rows(&screen.lines, 80);
        let gutters: Vec<(&str, Kind)> = rows.iter().map(|r| (r.gutter, r.kind)).collect();
        assert_eq!(
            gutters,
            vec![
                ("❯ ", Kind::Input),
                ("⎿ ", Kind::Output),
                ("  ", Kind::Output),
                ("✻ ", Kind::Thinking),
                ("┊ ", Kind::Thinking),
                ("  ", Kind::Success),
                ("", Kind::Output),
                ("❯ ", Kind::Input),
                ("✗ ", Kind::Error),
            ]
        );
        assert_eq!(rows[0].text, "/demo regression");
        assert_eq!(
            rows[3].text,
            "Thinking Trying a smoother kernel because the residuals"
        );
        assert_eq!(rows[4].text, "curve at high values · rmse 0.41 → 0.36");
        assert_eq!(rows[8].text, "no such file");
        // Drawn: the command bar, the reply connector and a scrollbar once history overflows.
        for i in 0..60 {
            screen.push(&format!("line {i}"));
        }
        let mut terminal = Terminal::new(TestBackend::new(60, 20)).unwrap();
        terminal
            .draw(|frame| screen.draw(frame, &Composer::default(), "t", "ready", ""))
            .unwrap();
        let buffer = terminal.backend().buffer().clone();
        let right: String = (1..14).map(|y| buffer[(59, y)].symbol()).collect();
        assert!(right.contains('┃') || right.contains('│'), "{right}");
        screen.handle_scroll(&Event::Key(KeyEvent::new(
            KeyCode::PageUp,
            KeyModifiers::NONE,
        )));
        screen.handle_scroll(&Event::Key(KeyEvent::new(
            KeyCode::PageUp,
            KeyModifiers::NONE,
        )));
        screen.handle_scroll(&Event::Key(KeyEvent::new(
            KeyCode::PageUp,
            KeyModifiers::NONE,
        )));
        screen.handle_scroll(&Event::Key(KeyEvent::new(
            KeyCode::PageUp,
            KeyModifiers::NONE,
        )));
        screen.handle_scroll(&Event::Key(KeyEvent::new(
            KeyCode::PageUp,
            KeyModifiers::NONE,
        )));
        terminal
            .draw(|frame| screen.draw(frame, &Composer::default(), "t", "ready", ""))
            .unwrap();
        let buffer = terminal.backend().buffer().clone();
        let text: String = (0..20)
            .map(|y| (0..60).map(|x| buffer[(x, y)].symbol()).collect::<String>() + "\n")
            .collect();
        assert!(text.contains("❯ /demo regression"), "{text}");
        assert!(text.contains("⎿ Demo regression_parkinsons"), "{text}");
        assert!(text.contains("✻ Thinking"), "{text}");
    }

    #[test]
    fn transcript_retention_is_bounded_and_escape_sequences_are_removed() {
        let mut screen = Screen::default();
        for _ in 0..3000 {
            screen.push("\x1b[2Jsafe");
        }
        assert_eq!(screen.lines.len(), MAX_LINES);
        assert!(screen.lines.iter().all(|(_, line)| line == "safe"));
        for _ in 0..300 {
            screen.push(&"z".repeat(8192));
        }
        assert!(screen.bytes <= MAX_BYTES);
    }
}
