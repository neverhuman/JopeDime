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
#[cfg(all(
    target_os = "linux",
    not(any(target_arch = "x86_64", target_arch = "aarch64"))
))]
compile_error!("Linux history-file flags are only verified for x86_64 and aarch64");

/// Client help/dispatch wiring belongs to TUI-09.
pub const HISTORY_HELP: &str = "History keeps up to 1,000 entries in $JOPEDIME_HOME/history (0600). Login commands and inline JSON objectives over 1,024 bytes are omitted. To clear it, close the client and remove that history file.";
static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, PartialEq, Eq)]
pub enum ComposerAction {
    None,
    Submit(String),
    ExitRequested,
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ComposerView {
    pub lines: Vec<String>,
    pub cursor: (u16, u16),
}

mod behavior;
mod history;
mod view;

pub(super) use behavior::clean;
pub use history::HistoryFile;
pub(super) use history::{owner_mode, regular_options, remember};

#[cfg(all(test, target_os = "linux"))]
#[path = "../../tests/support/pty.rs"]
#[allow(clippy::duplicate_mod)] // Reuse TUI-10; app.rs keeps its own inclusion private.
mod pty;

#[cfg(test)]
#[path = "composer/tests.rs"]
mod tests;
