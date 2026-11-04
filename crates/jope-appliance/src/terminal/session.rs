//! Real-client terminal entry point and §6.5 session policy.

use crate::draft::Draft;
use jope_api::{
    decimal::DecimalU64, ids::RunId, EventBody, Plan, PublicEvent, RunSnapshot, RunState,
};
use jope_client::{
    discover::{self, Engine, Mode},
    sse::StreamEvent,
    transport::{Request, RunResult, Transport},
    Http,
};
use ratatui::{
    backend::CrosstermBackend,
    crossterm::{
        event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers},
        terminal,
    },
    Terminal,
};
use std::{
    io::{self, IsTerminal},
    path::PathBuf,
    str::FromStr,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc,
    },
    thread,
    time::{Duration, Instant},
};

const INPUT_LATENCY: Duration = Duration::from_millis(20);

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Launch {
    New,
    Attach(Option<String>),
}

enum BackgroundResult {
    Sync(anyhow::Result<Vec<String>>),
    Publish(anyhow::Result<String>),
}

impl BackgroundResult {
    fn into_messages(self) -> Vec<String> {
        match self {
            Self::Sync(Ok(messages)) => messages,
            Self::Sync(Err(error)) => vec![format!("Sync: {error:#}")],
            Self::Publish(Ok(message)) => vec![message],
            Self::Publish(Err(error)) => vec![format!("Publish: {error:#}")],
        }
    }
}

pub fn requested(args: &[String]) -> Option<Launch> {
    requested_with_tty(args, io::stdin().is_terminal(), io::stdout().is_terminal())
}

fn requested_with_tty(args: &[String], input: bool, output: bool) -> Option<Launch> {
    if !input || !output {
        return None;
    }
    match args {
        [] => Some(Launch::New),
        [command] if command == "attach" => Some(Launch::Attach(None)),
        [command, run] if command == "attach" && !run.starts_with('-') => {
            Some(Launch::Attach(Some(run.clone())))
        }
        _ => None,
    }
}

pub fn run(launch: Launch) -> anyhow::Result<()> {
    let home = jope_paths::jopedime_home();
    let executable = std::env::current_exe()?;
    let engine = discover::discover(&discover::Options {
        home: home.clone(),
        executable,
        endpoint: None,
        mode: Mode::Interactive,
        engine_available: cfg!(feature = "engine"),
    })?;
    let (client, mode) = match engine {
        Engine::Background(background) => (background.client, "background"),
        Engine::Endpoint(_) => anyhow::bail!(
            "interactive terminal sessions require the authenticated local background engine"
        ),
        Engine::InProcess => anyhow::bail!("in-process terminal transport is unavailable"),
    };
    let system = client.system_info()?;
    let mut session = Session::new(client, home, mode, system.version)?;
    if let Launch::Attach(run) = launch {
        let run = match run {
            Some(run) => RunId::from_str(&run)?,
            None => session.latest_active_run()?,
        };
        session.attach(run, Some(DecimalU64(0)))?;
    }
    session.run()
}

struct Session {
    client: Http,
    engine_mode: &'static str,
    version: String,
    draft: Draft,
    composer: crate::terminal::composer::Composer,
    snapshot: Option<RunSnapshot>,
    events: mpsc::Receiver<PublicEvent>,
    event_sender: mpsc::Sender<PublicEvent>,
    activity: String,
    message: Option<String>,
    exit_requested: bool,
    screen: crate::terminal::screen::Screen,
    cloud_events: mpsc::Receiver<Vec<String>>,
    cloud_sender: mpsc::Sender<Vec<String>>,
    background_events: mpsc::Receiver<BackgroundResult>,
    background_sender: mpsc::Sender<BackgroundResult>,
    login_cancel: Option<Arc<AtomicBool>>,
    run_lists: mpsc::Receiver<anyhow::Result<(Vec<crate::terminal::runs::Entry>, String)>>,
    run_list_sender: mpsc::Sender<anyhow::Result<(Vec<crate::terminal::runs::Entry>, String)>>,
    cloud_busy: Arc<AtomicBool>,
    auto_sync: Option<(RunId, (String, String))>,
    last_sync: Instant,
    synced_terminal: mpsc::Receiver<RunId>,
    synced_terminal_sender: mpsc::Sender<RunId>,
    home: PathBuf,
    first_run: Option<crate::terminal::onboarding::FirstRun>,
    links: bool,
}

impl Session {
    fn new(
        client: Http,
        home: PathBuf,
        engine_mode: &'static str,
        version: String,
    ) -> anyhow::Result<Self> {
        let (event_sender, events) = mpsc::channel();
        let (cloud_sender, cloud_events) = mpsc::channel();
        let (background_sender, background_events) = mpsc::channel();
        let (run_list_sender, run_lists) = mpsc::channel();
        let (synced_terminal_sender, synced_terminal) = mpsc::channel();
        let history = home.join("history");
        let first_run = first_run_dialog(&home);
        let mut screen = crate::terminal::screen::Screen::default();
        screen.cloud_label = crate::cloud::account_label();
        let composer = crate::terminal::composer::Composer::with_history(history)?;
        Ok(Self {
            client,
            engine_mode,
            version,
            draft: Draft::default(),
            composer,
            snapshot: None,
            events,
            event_sender,
            activity: "Type \\help for commands".into(),
            message: None,
            exit_requested: false,
            screen,
            cloud_events,
            cloud_sender,
            background_events,
            background_sender,
            login_cancel: None,
            run_lists,
            run_list_sender,
            cloud_busy: Arc::new(AtomicBool::new(false)),
            auto_sync: None,
            last_sync: Instant::now(),
            synced_terminal,
            synced_terminal_sender,
            home,
            first_run,
            links: crate::terminal::links::supported_from_env(),
        })
    }
}

#[path = "session/activity.rs"]
mod activity;
#[path = "session/commands.rs"]
mod commands;
#[path = "session/events.rs"]
mod events;
#[path = "session/first_run.rs"]
mod first_run;
#[path = "session/ui.rs"]
mod ui;

fn pending_composer_lines(
    composer: &crate::terminal::composer::Composer,
    width: u16,
) -> Vec<String> {
    if composer.text().is_empty() {
        return Vec::new();
    }
    let view = composer.view(width.max(1), 6);
    let mut lines = vec![
        "interactive draft retained; press Enter to submit it or type a replacement command:"
            .into(),
    ];
    lines.extend(view.lines);
    lines.push(format!(
        "cursor at row {}, column {}",
        view.cursor.1 + 1,
        view.cursor.0 + 1
    ));
    lines
}

fn accept_line(
    composer: &mut crate::terminal::composer::Composer,
    line: &str,
) -> io::Result<Option<String>> {
    if line.is_empty() && composer.text().is_empty() {
        return Ok(None);
    }
    if !line.is_empty() {
        composer.set_text(line)?;
    }
    match composer.handle(
        Event::Key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
        &crate::terminal::complete::CompletionData::default(),
    )? {
        crate::terminal::composer::ComposerAction::Submit(text) => Ok(Some(text)),
        crate::terminal::composer::ComposerAction::ExitRequested
        | crate::terminal::composer::ComposerAction::Redraw
        | crate::terminal::composer::ComposerAction::None => {
            unreachable!("plain line Enter submits the current composer text")
        }
    }
}

fn update_snapshot_from_event(snapshot: &mut Option<RunSnapshot>, event: &PublicEvent) {
    let Some(snapshot) = snapshot.as_mut() else {
        return;
    };
    if event.run_id != snapshot.run_id {
        return;
    }
    snapshot.event_watermark = event.sequence;
    if let EventBody::RunState(body) = &event.event {
        snapshot.state = body.state;
        snapshot.phase = body.phase;
        snapshot.pending = body.pending.clone();
    }
}

fn plan_lines(plan: Plan) -> Vec<String> {
    vec![format!(
        "plan validated: {:?} · {} rows · {} · digest {}",
        plan.task.value, plan.rows.total.0, plan.objective.name, plan.plan_digest
    )]
}

fn open_cloud_url(url: &str) -> anyhow::Result<()> {
    anyhow::ensure!(
        url.starts_with("https://") && !url.chars().any(char::is_control),
        "invalid cloud URL"
    );
    crate::cloud::open_browser(url)
}

/// The first-run dialog to show, if any. Terms accepted non-interactively (`--accept-terms`
/// or `JAIN_ACCEPT_TERMS=1`) skip the dialog entirely: agents and scripts never see a prompt.
fn first_run_dialog(home: &std::path::Path) -> Option<crate::terminal::onboarding::FirstRun> {
    use crate::terminal::onboarding;
    if onboarding::accepted_noninteractively() {
        return None;
    }
    if onboarding::env_accepts_terms() {
        let _ = onboarding::accept_noninteractive(home, "env");
        return None;
    }
    onboarding::FirstRun::new(
        &onboarding::load(home),
        &onboarding::required_terms_version_now(),
        onboarding::telemetry_forced_off_from_env(),
        crate::cloud::connection().is_ok(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interactive_and_attach_launch_only_for_ttys() {
        assert_eq!(requested_with_tty(&[], true, true), Some(Launch::New));
        assert_eq!(
            requested_with_tty(&["attach".into()], true, true),
            Some(Launch::Attach(None))
        );
        assert_eq!(
            requested_with_tty(
                &["attach".into(), "run_01J00000000000000000000000".into()],
                true,
                true
            ),
            Some(Launch::Attach(Some(
                "run_01J00000000000000000000000".into()
            )))
        );
        assert_eq!(requested_with_tty(&[], false, true), None);
        assert_eq!(requested_with_tty(&["--help".into()], true, true), None);
    }

    #[test]
    fn control_requests_refresh_the_snapshot_before_using_its_revision() {
        let request = Request::new(8, 0, &[("id", "run_01J00000000000000000000000")])
            .unwrap()
            .key(Some("tui_test"))
            .unwrap()
            .if_match(DecimalU64(7))
            .json(&serde_json::json!({"action":"pause"}))
            .unwrap();
        assert_eq!(request.method(), jope_api::registry::HttpMethod::Post);
        assert_eq!(
            request.target(),
            "/v1/runs/run_01J00000000000000000000000/control"
        );
    }

    #[test]
    fn matching_run_state_event_advances_replay_watermark_and_snapshot() {
        let mut snapshot = Some(
            serde_json::from_value(serde_json::json!({
                "run_id": "run_01JBD8Q6N4V2T5Y8K3M7P9R1S0",
                "revision": "1",
                "state": "running",
                "phase": "searching",
                "pending": null,
                "effort": "fast",
                "budget_used": {"evaluations": "0", "wall_seconds": "0.0"},
                "budget_total": {"evaluations": "10", "wall_seconds": "60"},
                "best": null,
                "event_watermark": "1",
                "warnings": [],
                "completion_reason": null
            }))
            .unwrap(),
        );
        let event: PublicEvent = serde_json::from_value(serde_json::json!({
            "run_id": "run_01JBD8Q6N4V2T5Y8K3M7P9R1S0",
            "sequence": "4",
            "time": "2026-10-30T10:03:02.000Z",
            "kind": "run.state",
            "body": {
                "state": "paused",
                "phase": "searching",
                "pending": null,
                "reason": "paused by user"
            }
        }))
        .unwrap();

        update_snapshot_from_event(&mut snapshot, &event);

        let snapshot = snapshot.unwrap();
        assert_eq!(snapshot.event_watermark, DecimalU64(4));
        assert_eq!(snapshot.state, RunState::Paused);
        assert_eq!(snapshot.phase, jope_api::RunPhase::Searching);
        assert_eq!(snapshot.pending, None);
    }

    #[test]
    fn event_from_another_run_does_not_advance_attached_replay_cursor() {
        let mut snapshot = Some(
            serde_json::from_value(serde_json::json!({
                "run_id": "run_01JBD8Q6N4V2T5Y8K3M7P9R1S0",
                "revision": "1",
                "state": "running",
                "phase": "searching",
                "pending": null,
                "effort": "fast",
                "budget_used": {"evaluations": "0", "wall_seconds": "0.0"},
                "budget_total": {"evaluations": "10", "wall_seconds": "60"},
                "best": null,
                "event_watermark": "1",
                "warnings": [],
                "completion_reason": null
            }))
            .unwrap(),
        );
        let event: PublicEvent = serde_json::from_value(serde_json::json!({
            "run_id": "run_01JBD8Q6N4V2T5Y8K3M7P9R1S1",
            "sequence": "9",
            "time": "2026-10-30T10:03:02.000Z",
            "kind": "run.state",
            "body": {
                "state": "cancelled",
                "phase": "finalizing",
                "pending": null,
                "reason": "cancelled"
            }
        }))
        .unwrap();

        update_snapshot_from_event(&mut snapshot, &event);

        let snapshot = snapshot.unwrap();
        assert_eq!(snapshot.event_watermark, DecimalU64(1));
        assert_eq!(snapshot.state, RunState::Running);
    }

    #[test]
    fn narrow_line_mode_keeps_pending_composer_state_until_submit_or_replace() {
        let mut composer = crate::terminal::composer::Composer::default();
        composer.set_text("\\he xyz").unwrap();
        for _ in 0..4 {
            composer
                .handle(
                    Event::Key(KeyEvent::new(KeyCode::Left, KeyModifiers::NONE)),
                    &crate::terminal::complete::CompletionData::default(),
                )
                .unwrap();
        }
        composer
            .handle(
                Event::Key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE)),
                &crate::terminal::complete::CompletionData::default(),
            )
            .unwrap();
        let text = composer.text().to_owned();
        let cursor = composer.cursor_byte();
        let completion = composer.completion().map(|(_, selected)| selected);
        assert!(completion.is_some());

        let notice = pending_composer_lines(&composer, 24);
        assert!(notice[0].contains("press Enter to submit"));
        assert!(notice.iter().any(|line| line.contains("help")));
        assert_eq!(composer.text(), text);
        assert_eq!(composer.cursor_byte(), cursor);
        assert_eq!(
            composer.completion().map(|(_, selected)| selected),
            completion
        );

        assert_eq!(accept_line(&mut composer, "").unwrap(), Some(text));
        assert!(composer.text().is_empty());
        assert!(composer.completion().is_none());
    }

    #[test]
    fn narrow_line_mode_replaces_pending_composer_only_for_nonempty_input() {
        let mut composer = crate::terminal::composer::Composer::default();
        composer.set_text("unfinished draft").unwrap();
        assert_eq!(
            accept_line(&mut composer, "\\help history").unwrap(),
            Some("\\help history".into())
        );
        assert!(composer.text().is_empty());
        assert_eq!(accept_line(&mut composer, "").unwrap(), None);
    }
}
