use super::behavior::boundaries;
use super::*;

impl Composer {
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
