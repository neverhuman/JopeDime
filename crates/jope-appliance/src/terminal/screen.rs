//! Full-screen landing, bounded transcript, and a composer independent of scrolling.
use std::collections::VecDeque;

use ratatui::{
    crossterm::event::{Event, KeyCode, KeyEventKind, MouseEventKind},
    layout::{Alignment, Constraint, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Paragraph},
    Frame,
};

use super::{composer::Composer, sanitize_user_string, ColorLevel, Theme, ThemeMode};

const MAX_LINES: usize = 2000;
const MAX_BYTES: usize = 1024 * 1024;
// Seven-cell ASCII glyphs; stretch the strokes to the available terminal width.
const LOGO: [&str; 7] = [
    "##   ##  #######   #####   ##   ##",
    "##   ##  ##       ##   ##  ##   ##",
    "##   ##  ##       ##   ##   ## ## ",
    "##   ##  #####    ##   ##    ###  ",
    " ## ##   ##       ##   ##   ## ## ",
    "  ###    ##       ##   ##  ##   ##",
    "   #     #######   #####   ##   ##",
];
const TAGLINE: &str = "your data. your models. your next discover.";

#[derive(Default)]
pub struct Screen {
    pub browser: Option<super::runs::Browser>,
    pub cloud_label: String,
    lines: VecDeque<String>,
    bytes: usize,
    wrapped: Vec<String>,
    width: u16,
    dirty: bool,
    top: Option<usize>,
    unread: usize,
    page_height: usize,
}

impl Screen {
    pub fn push(&mut self, text: &str) {
        let clean = sanitize_user_string(text);
        for line in clean.lines() {
            // Transcript is a bounded view, not the durable run/event store.
            let line: String = line.chars().take(8192).collect();
            self.bytes += line.len();
            self.lines.push_back(line);
            if self.top.is_some() {
                self.unread += 1;
            }
            while self.lines.len() > MAX_LINES || self.bytes > MAX_BYTES {
                if let Some(old) = self.lines.pop_front() {
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
            self.wrapped = self
                .lines
                .iter()
                .flat_map(|line| wrap(line, width))
                .collect();
            self.dirty = false;
        }
    }

    pub fn transcript(&self) -> Vec<String> {
        self.lines.iter().cloned().collect()
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
            .view(area.width.saturating_sub(4).max(1), 6)
            .lines
            .len() as u16
            + 2;
        let is_landing = self.browser.is_none() && self.lines.is_empty();
        let rows = Layout::vertical([
            Constraint::Length(1),
            Constraint::Min(1),
            Constraint::Length(1),
            Constraint::Length(if is_landing && area.width < 55 { 2 } else { 1 }),
            Constraint::Length(input_height),
            Constraint::Length(1),
        ])
        .split(area);
        frame.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled(" JAIN by VEOX ", accent.add_modifier(Modifier::BOLD)),
                Span::styled(format!("/ CLI   ·   JopeDime {version}"), muted),
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
        if let Some(browser) = &mut self.browser {
            browser.draw(frame, body, accent);
        } else if self.lines.is_empty() {
            landing(frame, rows[1], accent);
        } else {
            let max = self.wrapped.len().saturating_sub(self.page_height);
            let start = self.top.map(|top| top.min(max)).unwrap_or(max);
            self.top = self.top.map(|_| start);
            frame.render_widget(
                Paragraph::new(
                    self.wrapped
                        .iter()
                        .skip(start)
                        .take(self.page_height)
                        .cloned()
                        .map(Line::from)
                        .collect::<Vec<_>>(),
                ),
                body,
            );
        }
        frame.render_widget(
            Paragraph::new(sanitize_user_string(status)).style(accent),
            rows[2],
        );
        frame.render_widget(
            Paragraph::new(if is_landing {
                if area.width < 55 {
                    " /help  Commands\n /open <csv>  Start locally".to_owned()
                } else {
                    " /help  Commands     /open <csv>  Start locally".to_owned()
                }
            } else {
                sanitize_user_string(activity)
            })
            .style(muted),
            rows[3],
        );
        let block = Block::bordered().title(" Command ").border_style(accent);
        let input = block.inner(rows[4]);
        frame.render_widget(block, rows[4]);
        composer.draw(frame, input);
        let footer = if self.browser.is_some() {
            " Up/Down choose · Enter attach/open · o cloud · d download · r refresh · Esc/right back · Ctrl-C exit".into()
        } else if self.top.is_some() {
            format!(
                " {} new lines · PgUp/PgDn scroll · End follows live · Ctrl-C exits",
                self.unread
            )
        } else {
            format!(
                " Left: runs · PgUp/PgDn: scroll · Ctrl-C: exit · {}",
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

fn landing(frame: &mut Frame<'_>, area: Rect, accent: Style) {
    let tagline_lines = if usize::from(area.width) >= TAGLINE.len() {
        vec![TAGLINE.to_owned()]
    } else {
        TAGLINE
            .split_inclusive(". ")
            .flat_map(|sentence| wrap(sentence.trim(), area.width))
            .collect()
    };
    let compact = area.height < LOGO.len() as u16 + 1 + tagline_lines.len() as u16;
    let logo_height = if compact {
        1
    } else {
        area.height
            .saturating_sub(tagline_lines.len() as u16 + 2)
            .min(14)
    };
    let crop = 2;
    // The wordmark fills roughly 70% of the width; centering OX shifts V off-screen.
    let logo_width = (area.width.saturating_mul(7) / 10 + crop).max(LOGO[0].len() as u16);
    let height = logo_height + 1 + tagline_lines.len() as u16;
    let top = area.y + area.height.saturating_sub(height) / 2;
    // Render into the full-width body, cropping the V at the actual screen edge.
    // The O/X midpoint lands near the screen center, rather than centering VEOX.
    let mut lines = Vec::new();
    for y in 0..logo_height {
        let logo = LOGO[usize::from(y) * LOGO.len() / usize::from(logo_height)];
        let stretched: String = (crop..logo_width)
            .map(|x| logo.as_bytes()[usize::from(x) * logo.len() / usize::from(logo_width)] as char)
            .collect();
        lines.push(Line::styled(stretched, accent.add_modifier(Modifier::BOLD)));
    }
    let logo = if compact {
        Paragraph::new("VEOX").style(accent.add_modifier(Modifier::BOLD))
    } else {
        Paragraph::new(lines)
    };
    frame.render_widget(
        logo,
        Rect {
            y: top,
            height: logo_height.min(area.height),
            ..area
        },
    );
    let tagline_y = top + logo_height + 1;
    let tagline = Rect {
        y: tagline_y,
        height: tagline_lines.len() as u16,
        ..area
    }
    .intersection(area);
    frame.render_widget(
        Paragraph::new(tagline_lines.into_iter().map(Line::raw).collect::<Vec<_>>())
            .alignment(Alignment::Center),
        tagline,
    );
}

fn wrap(text: &str, width: u16) -> Vec<String> {
    let width = usize::from(width.max(1));
    let text = text.replace('\t', "    ");
    let span = Span::raw(text);
    let mut lines = Vec::new();
    let mut line = String::new();
    let mut cells = 0;
    for grapheme in span.styled_graphemes(Style::default()) {
        let size = Span::raw(grapheme.symbol).width();
        if cells + size > width && !line.is_empty() {
            lines.push(std::mem::take(&mut line));
            cells = 0;
        }
        if size > width {
            line.push('?');
            cells += 1;
        } else {
            line.push_str(grapheme.symbol);
            cells += size;
        }
    }
    lines.push(line);
    lines
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
        for (width, height) in [(120, 40), (100, 30), (80, 24), (40, 15)] {
            let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
            let mut screen = Screen::default();
            let mut composer = Composer::default();
            composer.set_text("my pending draft").unwrap();
            terminal
                .draw(|frame| screen.draw(frame, &composer, "test", "ready", ""))
                .unwrap();
            let buffer = terminal.backend().buffer();
            let row = |y| {
                (0..width)
                    .map(|x| buffer[(x, y)].symbol())
                    .collect::<String>()
            };
            let output = (0..height).map(row).collect::<Vec<_>>().join("\n");
            assert!(output.contains("JAIN by VEOX"));
            assert!(output.contains("your data."));
            assert!(output.contains("your models."));
            assert!(output.contains("your next"));
            assert!(output.contains("discover."));
            assert!(row(height - 5).contains("/open <csv>"));
            assert!(row(height - 3).contains("my pending draft"));
            if width >= 80 {
                let tagline_y = (1..height - 5).find(|&y| row(y).contains(TAGLINE)).unwrap();
                assert_eq!(
                    row(tagline_y).find(TAGLINE),
                    Some((usize::from(width) - TAGLINE.len()).div_ceil(2))
                );
                let logo_top = (1..tagline_y).find(|&y| row(y).contains('#')).unwrap();
                assert_eq!(
                    buffer[(0, logo_top)].symbol(),
                    "#",
                    "V must be cropped at left edge"
                );
                assert!(
                    (1..tagline_y).all(|y| !row(y).contains("/help") && !row(y).contains("/open"))
                );
            }
            if let Some(directory) = std::env::var_os("VEOX_SPLASH_PREVIEW_DIR") {
                std::fs::create_dir_all(&directory).unwrap();
                std::fs::write(
                    std::path::Path::new(&directory).join(format!("{width}x{height}.txt")),
                    output,
                )
                .unwrap();
            }
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
    }

    #[test]
    fn transcript_retention_is_bounded_and_escape_sequences_are_removed() {
        let mut screen = Screen::default();
        for _ in 0..3000 {
            screen.push("\x1b[2Jsafe");
        }
        assert_eq!(screen.lines.len(), MAX_LINES);
        assert!(screen.lines.iter().all(|line| line == "safe"));
        for _ in 0..300 {
            screen.push(&"z".repeat(8192));
        }
        assert!(screen.bytes <= MAX_BYTES);
    }
}
