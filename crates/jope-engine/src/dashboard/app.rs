use super::evidence::{campaign_with_manifest, number, state, tail};
use super::layout::{compact_layout, compact_lines, DashboardData, History};
use super::prelude::*;
use super::render::render;

pub fn run(root: &Path) -> Result<()> {
    run_with_manifest(root, None)
}

pub fn run_with_manifest(root: &Path, experiment_manifest: Option<&Path>) -> Result<()> {
    use crossterm::{
        event::{self, Event, KeyCode},
        execute,
        terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    };
    enable_raw_mode()?;
    execute!(std::io::stdout(), EnterAlternateScreen)?;
    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            let _ = disable_raw_mode();
            let _ = crossterm::execute!(std::io::stdout(), LeaveAlternateScreen);
        }
    }
    let _restore = Restore;
    let mut terminal = Terminal::new(CrosstermBackend::new(std::io::stdout()))?;
    let mut view = 0;
    let mut history = History::default();
    let learning = crate::learning_history::Watcher::start(root.to_path_buf());
    let mut run = campaign_with_manifest(root, experiment_manifest)?;
    let mut campaign_updated = Instant::now();
    loop {
        if let Some(snapshot) = learning.latest() {
            history.learning = snapshot;
        }
        let h: Hardware = read_json(&root.join("runs/hardware.json")).unwrap_or_default();
        history.update(&h);
        let gp =
            crate::telemetry::observed_status(&state(&root.join("runs/gp-service.json")), now());
        let mut neural = state(&root.join("runs/neural-service.json"));
        let encoding = state(&root.join("runs/encoding-service.json"));
        if number(&encoding, "timestamp") > number(&neural, "timestamp") {
            neural = encoding;
            neural["encoding"] = Value::Bool(true);
        }
        if campaign_updated.elapsed() >= Duration::from_secs(5) {
            run = campaign_with_manifest(root, experiment_manifest)?;
            campaign_updated = Instant::now();
        }
        let events = tail(&root.join("runs/events.log"), 64 * 1024);
        terminal.draw(|f| {
            render(
                f,
                view,
                &h,
                crate::dashboard::DashboardData {
                    gp: &gp,
                    neural: &neural,
                    run: &run,
                    events: &events,
                    history: &history,
                },
            )
        })?;
        if event::poll(Duration::from_millis(250))? {
            if let Event::Key(key) = event::read()? {
                let size = terminal.size()?;
                if compact_layout(Rect::new(0, 0, size.width, size.height)) {
                    let visible = size.height.saturating_sub(3) as usize;
                    let maximum = compact_lines(
                        view,
                        &h,
                        crate::dashboard::DashboardData {
                            gp: &gp,
                            neural: &neural,
                            run: &run,
                            events: &events,
                            history: &history,
                        },
                        size.width,
                    )
                    .len()
                    .saturating_sub(visible)
                    .min(u16::MAX as usize) as u16;
                    let current = history.compact_offsets[view].min(maximum);
                    let selected = match key.code {
                        KeyCode::Down => Some(current.saturating_add(1)),
                        KeyCode::Up => Some(current.saturating_sub(1)),
                        KeyCode::PageDown => Some(current.saturating_add(visible as u16)),
                        KeyCode::PageUp => Some(current.saturating_sub(visible as u16)),
                        KeyCode::Home => Some(0),
                        KeyCode::End => Some(maximum),
                        _ => None,
                    };
                    if let Some(selected) = selected {
                        history.compact_offsets[view] = selected.min(maximum);
                        continue;
                    }
                }
                match key.code {
                    KeyCode::Char('q') => break,
                    KeyCode::Char(c @ '1'..='5') => view = c as usize - '1' as usize,
                    KeyCode::Char('d') if view == 2 => {
                        history.learning_details = !history.learning_details
                    }
                    KeyCode::Char('a') if view == 2 => {
                        history.learning_details = false;
                        history.learning_navigation.selected = None;
                    }
                    KeyCode::Down if view == 2 => {
                        history.learning_navigation.move_by(&history.learning, 1)
                    }
                    KeyCode::Up if view == 2 => {
                        history.learning_navigation.move_by(&history.learning, -1)
                    }
                    KeyCode::PageDown if view == 2 => {
                        history.learning_navigation.move_by(&history.learning, 10)
                    }
                    KeyCode::PageUp if view == 2 => {
                        history.learning_navigation.move_by(&history.learning, -10)
                    }
                    KeyCode::Home if view == 2 => {
                        history.learning_navigation.edge(&history.learning, false)
                    }
                    KeyCode::End if view == 2 => {
                        history.learning_navigation.edge(&history.learning, true)
                    }
                    KeyCode::Down if view == 3 => {
                        history.worker_offset = (history.worker_offset + 1)
                            .min(number(&gp, "active").saturating_sub(1) as usize)
                    }
                    KeyCode::Up if view == 3 => {
                        history.worker_offset = history.worker_offset.saturating_sub(1)
                    }
                    KeyCode::PageDown if view == 3 => {
                        history.worker_offset = (history.worker_offset + 20)
                            .min(number(&gp, "active").saturating_sub(1) as usize)
                    }
                    KeyCode::PageUp if view == 3 => {
                        history.worker_offset = history.worker_offset.saturating_sub(20)
                    }
                    KeyCode::Right => view = (view + 1) % 5,
                    KeyCode::Left => view = (view + 4) % 5,
                    _ => {}
                }
            }
        }
    }
    Ok(())
}
