use super::*;
pub(super) enum Screen {
    Inline(Box<Terminal<CrosstermBackend<io::Stdout>>>),
    Plain(LineMode<io::Stdout>),
}

impl Screen {
    pub(super) fn new(inline: bool, height: u16) -> io::Result<Self> {
        if inline {
            Ok(Self::Inline(Box::new(Terminal::with_options(
                CrosstermBackend::new(io::stdout()),
                TerminalOptions {
                    viewport: Viewport::Inline(height.clamp(6, 10)),
                },
            )?)))
        } else {
            Ok(Self::Plain(LineMode::new(io::stdout())))
        }
    }

    pub(super) fn transcript(&mut self, lines: &[String], width: u16) -> io::Result<()> {
        match self {
            Self::Plain(output) => output.transcript(lines),
            Self::Inline(terminal) => {
                // Explicit cell-width wrapping preserves long/Unicode lines in native scrollback.
                let lines: Vec<_> = lines
                    .iter()
                    .flat_map(|line| line.split('\n'))
                    .flat_map(|line| wrap_line(&sanitize_user_string(line), width))
                    .collect();
                for lines in lines.chunks(1000) {
                    terminal.insert_before(lines.len() as u16, |buffer| {
                        let text: Vec<_> = lines.iter().cloned().map(Line::from).collect();
                        Paragraph::new(text).render(buffer.area, buffer);
                    })?;
                }
                Ok(())
            }
        }
    }

    pub(super) fn draw(&mut self, state: &State) -> io::Result<()> {
        if let Self::Inline(terminal) = self {
            terminal.draw(|frame| {
                let area = frame.area();
                let lines = vec![
                    Line::from(state.status.as_str()),
                    Line::from(state.activity.as_str()),
                    Line::from(format!("> {}", state.draft)),
                ];
                frame.render_widget(Paragraph::new(lines), area);
                if area.width > 0 && area.height > 0 {
                    let x = Line::from(format!("> {}", state.draft)).width();
                    frame.set_cursor_position(Position::new(
                        area.x + x.min(usize::from(area.width - 1)) as u16,
                        area.y + 2.min(area.height - 1),
                    ));
                }
            })?;
        }
        Ok(())
    }

    pub(super) fn downgrade(&mut self, guard: &TerminalGuard) -> io::Result<()> {
        if let Self::Inline(terminal) = self {
            terminal.clear()?;
            // Drop the old terminal (including its cursor cleanup) before plain output starts.
            let old = std::mem::replace(self, Self::Plain(LineMode::new(io::stdout())));
            drop(old);
            guard.leave_inline()?;
        }
        Ok(())
    }

    pub(super) fn reconcile_width(
        &mut self,
        guard: &TerminalGuard,
        width: &mut u16,
    ) -> io::Result<()> {
        if matches!(self, Self::Inline(_)) {
            // The OS size can change before crossterm delivers its queued resize event.
            *width = ratatui::crossterm::terminal::size()?.0;
            if *width < 40 {
                self.downgrade(guard)?;
            }
        }
        Ok(())
    }
}

pub(super) fn wrap_line(text: &str, width: u16) -> Vec<String> {
    let width = usize::from(width.max(1));
    let mut lines = Vec::new();
    let mut line = String::new();
    let mut cells = 0;
    for character in text.chars() {
        let size = Line::from(character.to_string()).width();
        if cells + size > width && !line.is_empty() {
            lines.push(std::mem::take(&mut line));
            cells = 0;
        }
        line.push(character);
        cells += size;
    }
    lines.push(line);
    lines
}
