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

#[cfg(target_os = "linux")]
use std::io::Read;
#[cfg(target_os = "linux")]
use std::os::unix::fs::OpenOptionsExt;

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

struct State {
    draft: String,
    status: String,
    activity: String,
    active: bool,
}

/// Canonical input must be drained from the tty itself after the inline backend is left.
/// Crossterm's poller can consume a queued resize notification while leaving a following
/// newline readable in the kernel, then miss that byte when its readiness state is stale.
#[cfg(target_os = "linux")]
struct CanonicalInput {
    input: std::fs::File,
    pending: Vec<u8>,
}

#[cfg(target_os = "linux")]
impl CanonicalInput {
    fn new() -> io::Result<Self> {
        let input = std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(0o4000) // O_NONBLOCK
            .open("/dev/stdin")?;
        Ok(Self {
            input,
            pending: Vec::new(),
        })
    }

    fn lines(&mut self) -> io::Result<Vec<String>> {
        let mut bytes = [0; 1024];
        loop {
            match self.input.read(&mut bytes) {
                Ok(0) => break,
                Ok(count) => self.pending.extend_from_slice(&bytes[..count]),
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => break,
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(error),
            }
        }
        let mut lines = Vec::new();
        while let Some(end) = self
            .pending
            .iter()
            .position(|byte| matches!(byte, b'\r' | b'\n'))
        {
            let mut line: Vec<_> = self.pending.drain(..=end).collect();
            let delimiter = line.pop();
            if delimiter == Some(b'\r') && self.pending.first() == Some(&b'\n') {
                self.pending.remove(0);
            }
            lines.push(String::from_utf8_lossy(&line).into_owned());
        }
        Ok(lines)
    }
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
    #[cfg(target_os = "linux")]
    let mut canonical_input = None;
    if matches!(screen, Screen::Plain(_)) {
        screen.transcript(&[state.status.clone(), state.activity.clone()], width)?;
    }
    loop {
        if let Some(signal) = guard.signal() {
            return Ok(Exit::Signal(signal));
        }
        if let Some(exit) =
            reconcile_width_and_drain(&mut screen, &guard, &mut width, &mut state, &mut command)?
        {
            return Ok(exit);
        }
        #[cfg(all(test, target_os = "linux"))]
        stop_before_event_poll_for_test()?;
        #[cfg(target_os = "linux")]
        if input && matches!(screen, Screen::Plain(_)) {
            if canonical_input.is_none() {
                canonical_input = Some(CanonicalInput::new()?);
            }
            for line in canonical_input.as_mut().unwrap().lines()? {
                recent_activity = Some(Instant::now());
                state.draft.push_str(&sanitize_user_string(&line));
                if let Some(exit) = submit_draft(&mut state, &mut command)? {
                    return Ok(exit);
                }
            }
        }
        #[cfg(target_os = "linux")]
        let use_crossterm_input = matches!(screen, Screen::Inline(_));
        #[cfg(not(target_os = "linux"))]
        let use_crossterm_input = true;
        if input && use_crossterm_input && event::poll(Duration::ZERO)? {
            recent_activity = Some(Instant::now());
            let was_inline = matches!(screen, Screen::Inline(_));
            if let Some(exit) = handle_input_event(
                event::read()?,
                &mut screen,
                &guard,
                &mut width,
                &mut state,
                &mut command,
            )? {
                return Ok(exit);
            }
            if was_inline && matches!(screen, Screen::Plain(_)) {
                if let Some(exit) =
                    drain_queued_input(&mut screen, &guard, &mut width, &mut state, &mut command)?
                {
                    return Ok(exit);
                }
            }
        }
        let now = Instant::now();
        let active = state.active
            || recent_activity
                .is_some_and(|last| now.duration_since(last) < Duration::from_secs(1));
        if clock.remaining(now, active).is_zero() {
            if let Some(exit) = reconcile_width_and_drain(
                &mut screen,
                &guard,
                &mut width,
                &mut state,
                &mut command,
            )? {
                return Ok(exit);
            }
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
mod tests;

mod input;
mod screen;
use input::*;
use screen::Screen;
