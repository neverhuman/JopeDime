//! Inline terminal skeleton. Real-client launch and full composer policy belong to TUI-09/02.

use std::io::{self, IsTerminal};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, SyncSender};
use std::time::{Duration, Instant};

use jope_api::{EventBody, PublicEvent, RunState};
use ratatui::backend::CrosstermBackend;
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::layout::Position;
use ratatui::text::Line;
use ratatui::widgets::{Paragraph, Widget};
use ratatui::{Terminal, TerminalOptions, Viewport};

use super::line_mode::{self, LineMode};
use super::restore::{ExitSignal, TerminalGuard};
use super::{cards, sanitize_user_string, symbols_from_env, ColorLevel, Theme, ThemeMode};

const INPUT_LATENCY: Duration = Duration::from_millis(20);

#[cfg(all(test, target_os = "linux"))]
#[path = "../../tests/support/pty.rs"]
mod pty;

/// All transport producers share this bounded queue; they must handle receiver closure.
pub fn channel() -> (SyncSender<AppEvent>, Receiver<AppEvent>) {
    mpsc::sync_channel(256)
}

/// Client adapters may provide public events or already statefully rendered transcript lines.
#[derive(Debug)]
pub enum AppEvent {
    Public(Box<PublicEvent>),
    Transcript(Vec<String>),
    Status {
        status: String,
        activity: String,
        active: bool,
    },
    TransportClosed,
    TransportError(String),
    Quit,
}

#[derive(Clone, Debug)]
pub struct Options {
    pub plain: bool,
    pub term: Option<String>,
    pub no_color: bool,
    pub viewport_height: u16,
    /// Useful for recorded replay; interactive sessions remain open after transport EOF.
    pub exit_on_stream_end: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            plain: false,
            term: std::env::var("TERM").ok(),
            no_color: std::env::var_os("NO_COLOR").is_some(),
            viewport_height: 6,
            exit_on_stream_end: false,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Exit {
    Quit,
    StreamEnded,
    Signal(ExitSignal),
}

/// No catch-up bursts after delayed frames; each frame establishes its own next deadline.
#[derive(Debug)]
pub struct RedrawClock {
    last: Option<Instant>,
}

impl RedrawClock {
    pub fn new() -> Self {
        Self { last: None }
    }

    pub fn remaining(&self, now: Instant, active: bool) -> Duration {
        let period = if active {
            Duration::from_millis(50)
        } else {
            Duration::from_secs(1)
        };
        self.last
            .map(|last| period.saturating_sub(now.saturating_duration_since(last)))
            .unwrap_or(Duration::ZERO)
    }

    pub fn drawn(&mut self, now: Instant) {
        self.last = Some(now);
    }
}

impl Default for RedrawClock {
    fn default() -> Self {
        Self::new()
    }
}

enum Screen {
    Inline(Box<Terminal<CrosstermBackend<io::Stdout>>>),
    Plain(LineMode<io::Stdout>),
}

impl Screen {
    fn new(inline: bool, height: u16) -> io::Result<Self> {
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

    fn transcript(&mut self, lines: &[String], width: u16) -> io::Result<()> {
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

    fn draw(&mut self, state: &State) -> io::Result<()> {
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

    fn downgrade(&mut self, guard: &TerminalGuard) -> io::Result<()> {
        if let Self::Inline(terminal) = self {
            terminal.clear()?;
            // Drop the old terminal (including its cursor cleanup) before plain output starts.
            let old = std::mem::replace(self, Self::Plain(LineMode::new(io::stdout())));
            drop(old);
            guard.leave_inline()?;
        }
        Ok(())
    }

    fn reconcile_width(&mut self, guard: &TerminalGuard, width: &mut u16) -> io::Result<()> {
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

fn wrap_line(text: &str, width: u16) -> Vec<String> {
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

struct State {
    draft: String,
    status: String,
    activity: String,
    active: bool,
}

impl Default for State {
    fn default() -> Self {
        Self {
            draft: String::new(),
            status: "JopeDime · ready".into(),
            activity: "Type \\help for commands".into(),
            active: false,
        }
    }
}

/// A bounded presentation context; the terminal does not retain the replay's full history.
#[derive(Default)]
struct Presenter {
    plan: Option<PublicEvent>,
}

impl Presenter {
    fn lines(&mut self, event: &PublicEvent, width: u16) -> Vec<String> {
        let render = |events: &[PublicEvent]| {
            cards::render_transcript(
                events,
                usize::from(width),
                Theme::new(ThemeMode::Auto, ColorLevel::None, None),
                symbols_from_env(),
            )
        };
        if matches!(event.event, EventBody::Plan(_)) {
            self.plan = Some(event.clone());
        } else if matches!(event.event, EventBody::AlgorithmAccepted(_)) {
            if let Some(plan) = &self.plan {
                let prefix = render(std::slice::from_ref(plan)).len();
                return render(&[plan.clone(), event.clone()])
                    .into_iter()
                    .skip(prefix)
                    .collect();
            }
        }
        render(std::slice::from_ref(event))
    }
}

/// Run the terminal against a client-owned event receiver and command callback.
///
/// Input reads, cursor queries and draws stay on this thread. Queue waits are bounded to20ms
/// for input/signal responsiveness; only redraw deadlines can draw the live viewport.
/// Stateful transcript aggregation and full command/composer policy can be supplied by TUI-09.
pub fn run(
    options: Options,
    receiver: Receiver<AppEvent>,
    mut command: impl FnMut(&str) -> io::Result<()>,
) -> io::Result<Exit> {
    let input = io::stdin().is_terminal();
    let mut width = ratatui::crossterm::terminal::size()
        .map(|size| size.0)
        .unwrap_or(80);
    let inline = input
        && io::stdout().is_terminal()
        && !line_mode::required(
            options.plain,
            options.term.as_deref(),
            options.no_color,
            width,
        );
    let guard = TerminalGuard::acquire(inline)?;
    let mut screen = Screen::new(inline, options.viewport_height)?;
    let mut state = State::default();
    let mut presenter = Presenter::default();
    let mut clock = RedrawClock::new();
    let mut connected = true;
    let mut recent_activity = None;
    if matches!(screen, Screen::Plain(_)) {
        screen.transcript(&[state.status.clone(), state.activity.clone()], width)?;
    }
    loop {
        if let Some(signal) = guard.signal() {
            return Ok(Exit::Signal(signal));
        }
        screen.reconcile_width(&guard, &mut width)?;
        if input && event::poll(Duration::ZERO)? {
            recent_activity = Some(Instant::now());
            match event::read()? {
                Event::Resize(columns, _) => {
                    width = columns;
                    if columns < 40 {
                        screen.downgrade(&guard)?;
                    }
                }
                Event::Paste(text) => state.draft.push_str(&sanitize_user_string(&text)),
                Event::Key(key) if key.kind != KeyEventKind::Release => match key.code {
                    KeyCode::Char('d') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                        if state.draft.is_empty() {
                            return Ok(Exit::Quit);
                        }
                        state.draft.pop();
                    }
                    KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                        state.draft.clear();
                        screen.transcript(&["Use \\quit or Ctrl-D to exit.".into()], width)?;
                    }
                    KeyCode::Enter => {
                        let draft = std::mem::take(&mut state.draft);
                        if matches!(draft.trim(), "\\quit" | "/quit") {
                            return Ok(Exit::Quit);
                        }
                        command(&draft)?;
                    }
                    KeyCode::Backspace => {
                        state.draft.pop();
                    }
                    KeyCode::Char(character) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                        state.draft.push(character);
                    }
                    _ => {}
                },
                _ => {}
            }
        }
        let now = Instant::now();
        let active = state.active
            || recent_activity
                .is_some_and(|last| now.duration_since(last) < Duration::from_secs(1));
        if clock.remaining(now, active).is_zero() {
            screen.reconcile_width(&guard, &mut width)?;
            screen.draw(&state)?;
            clock.drawn(Instant::now());
        }
        let wait = clock.remaining(Instant::now(), active).min(INPUT_LATENCY);
        if !connected {
            std::thread::sleep(wait);
            continue;
        }
        let event = match receiver.recv_timeout(wait) {
            Ok(event) => event,
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => {
                if options.exit_on_stream_end {
                    return Ok(Exit::StreamEnded);
                }
                connected = false;
                state.active = false;
                continue;
            }
        };
        recent_activity = Some(Instant::now());
        match event {
            AppEvent::Public(event) => {
                let previous_status = state.status.clone();
                match &event.event {
                    EventBody::RunState(body) => {
                        state.active = matches!(body.state, RunState::Running | RunState::Pausing);
                        state.status = format!("JopeDime · {:?}", body.state);
                        state.activity = format!("phase: {:?}", body.phase);
                    }
                    EventBody::RunFinished(_) => {
                        state.active = false;
                        state.status = "JopeDime · finished".into();
                    }
                    EventBody::ExperimentStarted(_) => state.active = true,
                    _ => {}
                }
                if state.status != previous_status && matches!(screen, Screen::Plain(_)) {
                    screen.transcript(&[state.status.clone()], width)?;
                }
                screen.transcript(&presenter.lines(&event, width), width)?;
            }
            AppEvent::Transcript(lines) => screen.transcript(&lines, width)?,
            AppEvent::Status {
                status,
                activity,
                active,
            } => {
                state.status = sanitize_user_string(&status);
                state.activity = sanitize_user_string(&activity);
                state.active = active;
            }
            AppEvent::TransportError(message) => return Err(io::Error::other(message)),
            AppEvent::TransportClosed => {
                state.active = false;
                if options.exit_on_stream_end {
                    return Ok(Exit::StreamEnded);
                }
            }
            AppEvent::Quit => return Ok(Exit::Quit),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redraw_caps_cover_bursts_idle_and_activity_transitions() {
        let now = Instant::now();
        let mut clock = RedrawClock::new();
        assert!(clock.remaining(now, true).is_zero());
        clock.drawn(now);
        assert_eq!(clock.remaining(now, true), Duration::from_millis(50));
        assert_eq!(clock.remaining(now, false), Duration::from_secs(1));
        assert_eq!(
            clock.remaining(now + Duration::from_millis(49), true),
            Duration::from_millis(1)
        );
        assert!(clock
            .remaining(now + Duration::from_millis(50), true)
            .is_zero());
        let delayed = now + Duration::from_secs(5);
        clock.drawn(delayed);
        assert_eq!(clock.remaining(delayed, true), Duration::from_millis(50));
        assert_eq!(clock.remaining(delayed, false), Duration::from_secs(1));
    }

    #[test]
    fn wrapping_preserves_wide_combining_and_long_transcript_text() {
        for text in ["hello this is a very long transcript", "界界e\u{301}界界"] {
            let lines = wrap_line(text, 4);
            assert_eq!(lines.concat(), text);
            assert!(lines
                .iter()
                .all(|line| Line::from(line.as_str()).width() <= 4));
        }
    }

    #[cfg(target_os = "linux")]
    const FIXTURES: &[(&str, &str)] = &[
        (
            "success",
            include_str!("../../tests/fixtures/events/success.jsonl"),
        ),
        (
            "warnings",
            include_str!("../../tests/fixtures/events/warnings.jsonl"),
        ),
        (
            "failed",
            include_str!("../../tests/fixtures/events/failed-candidate-burst.jsonl"),
        ),
        (
            "paused",
            include_str!("../../tests/fixtures/events/paused-continued.jsonl"),
        ),
        (
            "cancelled",
            include_str!("../../tests/fixtures/events/cancelled.jsonl"),
        ),
    ];

    #[cfg(target_os = "linux")]
    fn replay(sender: SyncSender<AppEvent>, fixture: &str) -> std::thread::JoinHandle<()> {
        let replay =
            jope_client::fake::FakeTransport::from_jsonl(fixture, Duration::from_millis(25))
                .expect("recorded TUI-11 fixture");
        std::thread::spawn(move || {
            for event in replay {
                std::thread::sleep(event.delay_before);
                if sender
                    .send(AppEvent::Public(Box::new(event.event)))
                    .is_err()
                {
                    return;
                }
            }
            let _ = sender.send(AppEvent::TransportClosed);
        })
    }

    #[cfg(target_os = "linux")]
    fn termios() -> String {
        let output = std::process::Command::new("stty")
            .arg("-g")
            .stdin(std::process::Stdio::inherit())
            .output()
            .expect("Linux stty");
        assert!(output.status.success());
        String::from_utf8(output.stdout).unwrap()
    }

    /// One supervised helper, executed in separate real PTYs by the acceptance tests below.
    #[cfg(target_os = "linux")]
    #[test]
    #[ignore = "PTY child helper; executed explicitly by the Linux U2 parent tests"]
    fn pty_probe() {
        use std::io::Write;
        let scenario = std::env::var("JOPE_TUI_SCENARIO").expect("supervised PTY scenario");
        let (sender, receiver) = channel();
        if let Some(fixture) = scenario.strip_prefix("pipe-") {
            let jsonl = FIXTURES
                .iter()
                .find(|(name, _)| *name == fixture)
                .unwrap()
                .1;
            let producer = replay(sender, jsonl);
            let result = run(
                Options {
                    exit_on_stream_end: true,
                    ..Options::default()
                },
                receiver,
                |_| Ok(()),
            )
            .unwrap();
            producer.join().unwrap();
            assert_eq!(result, Exit::StreamEnded);
            return;
        }
        println!("PROBE_START");
        io::stdout().flush().unwrap();
        let mut barrier = String::new();
        io::stdin().read_line(&mut barrier).unwrap();
        assert_eq!(barrier.trim(), "go");
        let before = termios();
        println!("PROBE_ENTER");
        io::stdout().flush().unwrap();
        let mut producers = Vec::new();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run(
                Options {
                    plain: scenario == "plain",
                    exit_on_stream_end: scenario == "normal",
                    ..Options::default()
                },
                receiver,
                |command| {
                    match command {
                        "check" => {
                            let raw = ratatui::crossterm::terminal::is_raw_mode_enabled()?;
                            let changed = termios() != before;
                            sender
                                .try_send(AppEvent::Transcript(vec![format!(
                                    "PROBE_RAW raw={raw} changed={changed}"
                                )]))
                                .unwrap();
                        }
                        "replay" => producers.push(replay(sender.clone(), FIXTURES[0].1)),
                        "error" => return Err(io::Error::other("scripted command failure")),
                        "transport-error" => sender
                            .try_send(AppEvent::TransportError(
                                "scripted transport failure".into(),
                            ))
                            .unwrap(),
                        "panic" => panic!("scripted terminal panic"),
                        "int" => signal_hook::low_level::raise(signal_hook::consts::SIGINT)?,
                        "term" => signal_hook::low_level::raise(signal_hook::consts::SIGTERM)?,
                        "hup" => signal_hook::low_level::raise(signal_hook::consts::SIGHUP)?,
                        other => sender
                            .try_send(AppEvent::Transcript(vec![format!("COMMAND {other}")]))
                            .unwrap(),
                    }
                    Ok(())
                },
            )
        }));
        for producer in producers {
            producer.join().unwrap();
        }
        match scenario.as_str() {
            "panic" => assert!(result.is_err()),
            "error" | "transport-error" => assert!(result.unwrap().is_err()),
            "int" => assert_eq!(
                result.unwrap().unwrap(),
                Exit::Signal(ExitSignal::Interrupt)
            ),
            "term" => assert_eq!(
                result.unwrap().unwrap(),
                Exit::Signal(ExitSignal::Terminate)
            ),
            "hup" => assert_eq!(result.unwrap().unwrap(), Exit::Signal(ExitSignal::Hangup)),
            "normal" => assert_eq!(result.unwrap().unwrap(), Exit::StreamEnded),
            _ => assert_eq!(result.unwrap().unwrap(), Exit::Quit),
        }
        let raw = ratatui::crossterm::terminal::is_raw_mode_enabled().unwrap();
        let restored = termios() == before;
        println!("PROBE_RESTORED raw={raw} termios={restored}");
        assert!(!raw);
        assert!(
            restored,
            "exact pre-session Linux terminal flags must be restored"
        );
    }

    #[cfg(target_os = "linux")]
    fn spawn_probe(scenario: &str, inline: bool) -> pty::PtySession {
        let executable = std::env::current_exe().unwrap();
        let scenario_env = format!("JOPE_TUI_SCENARIO={scenario}");
        let color = if scenario == "no-color" {
            "NO_COLOR=1"
        } else {
            "COLORTERM=truecolor"
        };
        let term = if scenario == "dumb" {
            "TERM=dumb"
        } else {
            "TERM=xterm-256color"
        };
        let args = [
            "-u",
            "NO_COLOR",
            &scenario_env,
            color,
            term,
            "RUST_BACKTRACE=0",
            executable.to_str().unwrap(),
            "--exact",
            "terminal::app::tests::pty_probe",
            "--ignored",
            "--nocapture",
            "--color=never",
            "--test-threads=1",
        ];
        let mut session = pty::PtySession::spawn("/usr/bin/env", &args).unwrap();
        session
            .wait_for_text("PROBE_START", Duration::from_secs(10))
            .unwrap();
        if scenario == "narrow" {
            session.resize(39, 24).unwrap();
        }
        session.type_text("go\n").unwrap();
        session
            .wait_for_text("PROBE_ENTER", Duration::from_secs(10))
            .unwrap();
        if inline {
            // TUI-10 supplies a raw PTY, not a terminal emulator: answer the initial CPR query.
            session.type_text("\x1b[1;1R").unwrap();
        }
        session
            .wait_for_text("JopeDime", Duration::from_secs(10))
            .unwrap();
        session.type_text("check\r").unwrap();
        let marker = if inline {
            "PROBE_RAW raw=true changed=true"
        } else {
            "PROBE_RAW raw=false changed=false"
        };
        session
            .wait_for_text(marker, Duration::from_secs(10))
            .unwrap();
        session
    }

    #[cfg(target_os = "linux")]
    fn retain_screen(scenario: &str, screen: &str) {
        if let Some(directory) = std::env::var_os("JOPE_TUI_EVIDENCE_DIR") {
            std::fs::write(
                std::path::Path::new(&directory).join(format!("{scenario}.txt")),
                screen,
            )
            .unwrap();
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn linux_u2_restores_normal_error_panic_and_three_real_signals() {
        for (scenario, command) in [
            ("normal", "replay"),
            ("error", "error"),
            ("transport-error", "transport-error"),
            ("panic", "panic"),
            ("int", "int"),
            ("term", "term"),
            ("hup", "hup"),
        ] {
            let mut session = spawn_probe(scenario, true);
            session.type_text(&format!("{command}\r")).unwrap();
            let screen = session
                .wait_for_text(
                    "PROBE_RESTORED raw=false termios=true",
                    Duration::from_secs(10),
                )
                .unwrap();
            if scenario == "normal" {
                assert!(
                    screen.contains("First model"),
                    "recorded fixture did not reach transcript: {screen}"
                );
                assert!(
                    screen.contains("Run finished"),
                    "recorded fixture did not finish: {screen}"
                );
            }
            retain_screen(scenario, &screen);
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn linux_fallbacks_never_change_raw_mode_and_resize_preserves_draft() {
        for scenario in ["plain", "no-color", "dumb", "narrow"] {
            let mut session = spawn_probe(scenario, false);
            session.type_text("\\quit\r").unwrap();
            let screen = session
                .wait_for_text(
                    "PROBE_RESTORED raw=false termios=true",
                    Duration::from_secs(10),
                )
                .unwrap();
            retain_screen(scenario, &screen);
        }
        let mut session = spawn_probe("resize", true);
        session.type_text("draft").unwrap();
        session
            .wait_for_text("draft", Duration::from_secs(10))
            .unwrap();
        session.resize(39, 24).unwrap();
        // Resize delivery is asynchronous. Wait for raw-mode restoration before sending Enter,
        // so this checks draft submission after fallback instead of racing the PTY resize signal.
        session
            .wait_for_output_bytes(b"\x1b[?2004l", Duration::from_secs(10))
            .unwrap();
        session.press_key(pty::Key::Enter).unwrap();
        session
            .wait_for_text("COMMAND draft", Duration::from_secs(10))
            .unwrap();
        session.type_text("\\quit\r").unwrap();
        let screen = session
            .wait_for_text(
                "PROBE_RESTORED raw=false termios=true",
                Duration::from_secs(10),
            )
            .unwrap();
        retain_screen("resize", &screen);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn no_color_replays_all_five_real_recorded_fixtures_without_escape_bytes() {
        for (name, _) in FIXTURES {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "terminal::app::tests::pty_probe",
                    "--ignored",
                    "--nocapture",
                    "--color=never",
                    "--test-threads=1",
                ])
                .env("JOPE_TUI_SCENARIO", format!("pipe-{name}"))
                .env("NO_COLOR", "1")
                .env("TERM", "xterm-256color")
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(
                !output.stdout.contains(&0x1b),
                "NO_COLOR output contains escapes"
            );
            let screen = String::from_utf8(output.stdout).unwrap();
            assert!(screen.contains("JopeDime"));
            if *name == "success" {
                assert!(screen.contains("First model"));
                assert!(screen.contains("Run finished"));
            }
            retain_screen(&format!("no-color-{name}"), &screen);
        }
    }
}
