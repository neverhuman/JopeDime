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

#[path = "tests/editing_cases.rs"]
mod editing_cases;
#[path = "tests/history_cases.rs"]
mod history_cases;
#[path = "tests/terminal_cases.rs"]
mod terminal_cases;
#[path = "tests/view_cases.rs"]
mod view_cases;
