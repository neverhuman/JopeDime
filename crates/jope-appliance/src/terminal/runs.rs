//! Run availability is independent of execution state and model download state.
use jope_api::ids::RunId;
use ratatui::{
    crossterm::event::{Event, KeyCode, KeyEventKind, MouseEventKind},
    layout::Rect,
    style::{Modifier, Style},
    text::Line,
    widgets::{Block, List, ListItem, ListState},
    Frame,
};

#[derive(Clone, Debug)]
pub struct Entry {
    pub id: String,
    pub status: String,
    pub local: Option<RunId>,
    pub cloud_url: Option<String>,
    pub model_here: bool,
    pub model_cloud: bool,
    pub local_record: bool,
}
impl Entry {
    pub fn availability(&self) -> &'static str {
        match (
            self.local.is_some() || self.local_record,
            self.cloud_url.is_some(),
        ) {
            (true, true) => "Local + cloud",
            (true, false) => "Local only",
            (false, _) => "Cloud only",
        }
    }
}

pub enum Action {
    None,
    Close,
    Refresh,
    Attach(RunId),
    Open(String),
    Fetch(String),
}
#[derive(Default)]
pub struct Browser {
    pub entries: Vec<Entry>,
    pub notice: String,
    selected: ListState,
}
impl Browser {
    pub fn replace(&mut self, mut entries: Vec<Entry>, notice: String) {
        let current = self
            .selected
            .selected()
            .and_then(|i| self.entries.get(i))
            .map(|row| row.id.clone());
        entries.reverse();
        self.entries = entries;
        self.notice = notice;
        let index = current
            .and_then(|id| self.entries.iter().position(|row| row.id == id))
            .unwrap_or(0);
        self.selected.select(if self.entries.is_empty() {
            None
        } else {
            Some(index)
        });
    }
    pub fn handle(&mut self, event: &Event) -> Action {
        let key = match event {
            Event::Key(key) if key.kind != KeyEventKind::Release => Some(key.code),
            Event::Mouse(mouse) => match mouse.kind {
                MouseEventKind::ScrollUp => Some(KeyCode::Up),
                MouseEventKind::ScrollDown => Some(KeyCode::Down),
                _ => None,
            },
            _ => None,
        };
        match key {
            Some(KeyCode::Esc | KeyCode::Right) => return Action::Close,
            Some(KeyCode::Char('r')) => return Action::Refresh,
            Some(KeyCode::Up | KeyCode::Down) => {
                let current = self.selected.selected().unwrap_or(0);
                let next = if key == Some(KeyCode::Up) {
                    current.saturating_sub(1)
                } else {
                    (current + 1).min(self.entries.len().saturating_sub(1))
                };
                self.selected.select(if self.entries.is_empty() {
                    None
                } else {
                    Some(next)
                });
            }
            Some(KeyCode::Char('d')) => {
                if let Some(url) = self
                    .selected
                    .selected()
                    .and_then(|i| self.entries.get(i))
                    .and_then(|row| row.cloud_url.as_deref())
                {
                    if let Some(id) = url.rsplit('/').next() {
                        return Action::Fetch(id.to_owned());
                    }
                }
            }
            Some(KeyCode::Enter | KeyCode::Char('o')) => {
                if let Some(row) = self.selected.selected().and_then(|n| self.entries.get(n)) {
                    if key == Some(KeyCode::Enter) {
                        if let Some(local) = &row.local {
                            return Action::Attach(local.clone());
                        }
                    }
                    if let Some(url) = &row.cloud_url {
                        return Action::Open(url.clone());
                    }
                }
            }
            _ => {}
        }
        Action::None
    }
    pub fn draw(&mut self, frame: &mut Frame<'_>, area: Rect, accent: Style) {
        let title = format!(
            " Runs ({}) · {} ",
            self.entries.len(),
            super::sanitize_user_string(&self.notice)
        );
        let rows: Vec<ListItem<'static>> = if self.entries.is_empty() {
            vec![ListItem::new(
                "No runs loaded. Refresh or Esc to return to the composer.",
            )]
        } else {
            self.entries
                .iter()
                .map(|row| {
                    let model = match (row.model_here, row.model_cloud) {
                        (true, true) => "model here + cloud",
                        (true, false) => "model here",
                        (false, true) => "model in cloud · not downloaded",
                        (false, false) => "model not available here",
                    };
                    ListItem::new(vec![
                        Line::raw(super::sanitize_user_string(&format!(
                            "{}  ·  {}  ·  {}",
                            row.id,
                            row.status,
                            row.availability()
                        ))),
                        Line::raw(format!("  {model}")),
                        Line::raw(super::sanitize_user_string(
                            row.cloud_url
                                .as_deref()
                                .unwrap_or("  No cloud URL · not synced"),
                        )),
                        Line::raw(""),
                    ])
                })
                .collect()
        };
        frame.render_stateful_widget(
            List::new(rows)
                .block(Block::bordered().title(title).border_style(accent))
                .highlight_style(accent.add_modifier(Modifier::BOLD))
                .highlight_symbol("> "),
            area,
            &mut self.selected,
        );
    }
}

#[cfg(feature = "client")]
pub fn local_runs<T: jope_client::Transport>(
    client: &T,
) -> anyhow::Result<Vec<jope_api::RunSnapshot>> {
    use jope_client::transport::Request;
    let mut runs = Vec::new();
    let mut after = None::<String>;
    let mut cursors = std::collections::HashSet::new();
    loop {
        let request = Request::new(7, 0, &[])?.query(&[("limit", "256")]);
        let request = if let Some(cursor) = &after {
            request.query(&[("after", cursor)])
        } else {
            request
        };
        let page: jope_api::registry::RunPage = client.execute(request)?.json()?;
        runs.extend(page.runs);
        match page.next_after {
            None => return Ok(runs),
            Some(cursor) => {
                anyhow::ensure!(
                    cursors.insert(cursor.clone()),
                    "engine repeated a run-list cursor"
                );
                after = Some(cursor);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::crossterm::event::{KeyEvent, KeyModifiers};
    fn key(code: KeyCode) -> Event {
        Event::Key(KeyEvent::new(code, KeyModifiers::NONE))
    }
    #[test]
    fn location_is_independent_of_status_and_cloud_only_opens_real_url() {
        let mut browser = Browser::default();
        let row = Entry {
            id: "cloud-id".into(),
            status: "completed".into(),
            local: None,
            cloud_url: Some("https://dev.example.test/runs/cloud-id".into()),
            model_here: false,
            model_cloud: true,
            local_record: false,
        };
        assert_eq!(row.availability(), "Cloud only");
        browser.replace(vec![row], "Connected".into());
        assert!(
            matches!(browser.handle(&key(KeyCode::Enter)), Action::Open(url) if url.ends_with("/runs/cloud-id"))
        );
        assert!(matches!(browser.handle(&key(KeyCode::Esc)), Action::Close));
    }
    #[test]
    fn empty_browser_navigation_is_safe_and_refresh_is_explicit() {
        let mut browser = Browser::default();
        browser.replace(Vec::new(), "Local".into());
        assert!(matches!(browser.handle(&key(KeyCode::Down)), Action::None));
        assert!(matches!(browser.handle(&key(KeyCode::Enter)), Action::None));
        assert!(matches!(
            browser.handle(&key(KeyCode::Char('r'))),
            Action::Refresh
        ));
    }
}
