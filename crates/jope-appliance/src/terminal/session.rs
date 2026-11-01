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
        event::{self, Event, KeyCode, KeyEvent, KeyModifiers},
        terminal,
    },
    layout::{Constraint, Layout},
    text::Line,
    widgets::{Paragraph, Widget},
    Terminal, TerminalOptions, Viewport,
};
use std::{
    io::{self, IsTerminal},
    path::PathBuf,
    str::FromStr,
    sync::mpsc,
    thread,
    time::{Duration, Instant},
};

const INPUT_LATENCY: Duration = Duration::from_millis(20);

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Launch {
    New,
    Attach(Option<String>),
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
}

impl Session {
    fn new(
        client: Http,
        home: PathBuf,
        engine_mode: &'static str,
        version: String,
    ) -> anyhow::Result<Self> {
        let (event_sender, events) = mpsc::channel();
        let history = home.join("history");
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
        })
    }

    fn run(&mut self) -> anyhow::Result<()> {
        let width = terminal::size().map(|size| size.0).unwrap_or(80);
        let inline = !crate::terminal::line_mode::required(
            false,
            std::env::var("TERM").ok().as_deref(),
            std::env::var_os("NO_COLOR").is_some(),
            width,
        );
        let guard = crate::terminal::restore::TerminalGuard::acquire(inline)?;
        if !inline {
            return self.line_mode(&guard);
        }
        let mut terminal = Terminal::with_options(
            CrosstermBackend::new(io::stdout()),
            TerminalOptions {
                viewport: Viewport::Inline(8),
            },
        )?;
        let mut last_draw = None;
        loop {
            if guard.signal().is_some() {
                self.leave();
                return Ok(());
            }
            let width = terminal::size().map(|size| size.0).unwrap_or(80);
            if width < 40 {
                terminal.clear()?;
                drop(terminal);
                guard.leave_inline()?;
                return self.line_mode(&guard);
            }
            while let Ok(event) = self.events.try_recv() {
                self.on_event(event, width, &mut terminal)?;
            }
            if event::poll(INPUT_LATENCY)? {
                let input = event::read()?;
                let columns = self.draft.columns.clone();
                let run_ids = self
                    .snapshot
                    .as_ref()
                    .map(|snapshot| vec![snapshot.run_id.to_string()])
                    .unwrap_or_default();
                let working_directory = std::env::current_dir().ok();
                let completion = crate::terminal::complete::CompletionData {
                    columns: &columns,
                    run_ids: &run_ids,
                    algorithm_ids: &[],
                    paths: &[],
                    working_directory: working_directory.as_deref(),
                };
                match self.composer.handle(input, &completion)? {
                    crate::terminal::composer::ComposerAction::Submit(text) => {
                        for line in self.execute(&text) {
                            terminal.insert_before(1, |buffer| {
                                Paragraph::new(crate::terminal::sanitize_user_string(&line))
                                    .render(buffer.area, buffer);
                            })?;
                        }
                        if self.exit_requested {
                            return Ok(());
                        }
                    }
                    crate::terminal::composer::ComposerAction::ExitRequested => {
                        self.leave();
                        return Ok(());
                    }
                    crate::terminal::composer::ComposerAction::Redraw
                    | crate::terminal::composer::ComposerAction::None => {}
                }
            }
            let now = Instant::now();
            let period = if self
                .snapshot
                .as_ref()
                .is_some_and(|snapshot| snapshot.state == RunState::Running)
            {
                Duration::from_millis(50)
            } else {
                Duration::from_secs(1)
            };
            if last_draw.is_none_or(|last| now.duration_since(last) >= period) {
                self.draw(&mut terminal)?;
                last_draw = Some(Instant::now());
            }
        }
    }

    fn line_mode(&mut self, guard: &crate::terminal::restore::TerminalGuard) -> anyhow::Result<()> {
        use std::io::BufRead;
        enum Input {
            Line(String),
            Eof,
            Error(io::Error),
        }
        let (sender, receiver) = mpsc::channel();
        thread::Builder::new()
            .name("jope-terminal-line-input".into())
            .spawn(move || {
                let stdin = io::stdin();
                let mut input = stdin.lock();
                loop {
                    let mut line = String::new();
                    let message = match input.read_line(&mut line) {
                        Ok(0) => Input::Eof,
                        Ok(_) => Input::Line(line),
                        Err(error) => Input::Error(error),
                    };
                    let done = matches!(message, Input::Eof | Input::Error(_));
                    if sender.send(message).is_err() || done {
                        break;
                    }
                }
            })?;
        let mut output = crate::terminal::line_mode::LineMode::new(io::stdout());
        output.transcript(&[format!(
            "JopeDime {} · engine {}",
            self.version, self.engine_mode
        )])?;
        let width = terminal::size().map(|size| size.0).unwrap_or(80);
        output.transcript(&pending_composer_lines(&self.composer, width))?;
        loop {
            if guard.signal().is_some() {
                self.leave();
                return Ok(());
            }
            let width = usize::from(terminal::size().map(|size| size.0).unwrap_or(80));
            while let Ok(event) = self.events.try_recv() {
                output.transcript(&self.handle_event(event, width))?;
            }
            match receiver.recv_timeout(INPUT_LATENCY) {
                Ok(Input::Line(line)) => {
                    let line = line.trim_end_matches(['\r', '\n']);
                    if let Some(input) = accept_line(&mut self.composer, line)? {
                        output.transcript(&self.execute(&input))?;
                    }
                }
                Ok(Input::Eof) => {
                    self.leave();
                    return Ok(());
                }
                Ok(Input::Error(error)) => return Err(error.into()),
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    self.leave();
                    return Ok(());
                }
            }
            if self.exit_requested {
                return Ok(());
            }
        }
    }

    fn draw(&mut self, terminal: &mut Terminal<CrosstermBackend<io::Stdout>>) -> io::Result<()> {
        let header = format!("JopeDime {} · engine {}", self.version, self.engine_mode);
        let width = usize::from(terminal.size()?.width);
        let (status, activity) = self.status_lines(width);
        terminal.draw(|frame| {
            let area = frame.area();
            let rows = Layout::vertical([
                Constraint::Length(1),
                Constraint::Length(1),
                Constraint::Length(1),
                Constraint::Min(3),
            ])
            .split(area);
            frame.render_widget(Paragraph::new(header), rows[0]);
            frame.render_widget(Paragraph::new(status), rows[1]);
            frame.render_widget(Paragraph::new(activity), rows[2]);
            let input = rows[3];
            self.composer.draw(frame, input);
        })?;
        Ok(())
    }

    fn status_lines(&self, width: usize) -> (String, String) {
        if let Some(snapshot) = &self.snapshot {
            crate::terminal::status::render(snapshot, &[], width)
        } else {
            (
                self.message
                    .clone()
                    .unwrap_or_else(|| "ready · no active run".into()),
                self.activity.clone(),
            )
        }
    }

    fn submit(&mut self, input: &str) -> anyhow::Result<Vec<String>> {
        let input = input.trim();
        if input.is_empty() {
            return Ok(Vec::new());
        }
        let parsed = crate::terminal::commands::parse(input)?;
        let args = parsed.arguments;
        let messages = match parsed.command.name {
            "\\help" => vec![if args.first().is_some_and(|arg| arg == "history") {
                crate::terminal::composer::HISTORY_HELP.to_owned()
            } else {
                "\\open PATH · \\target COLUMN · \\plan · \\run · \\attach RUN · \\quit".into()
            }],
            "\\quit" => {
                if args.iter().any(|arg| arg == "--pause") {
                    self.control("pause")?;
                }
                self.leave();
                self.exit_requested = true;
                vec!["session closed; run continues in the background".into()]
            }
            "\\open" => self.open_dataset(&args)?,
            "\\target" | "\\objective" | "\\effort" | "\\landmarks" | "\\lang" | "\\language" => {
                self.update_draft(input)?
            }
            "\\plan" => self.validate_draft()?.map(plan_lines).unwrap_or_else(|| {
                vec![
                    "draft needs a registered dataset and target before it can be validated".into(),
                ]
            }),
            "\\run" => self.start_run(&args)?,
            "\\export" => self.export(&args)?,
            "\\attach" => {
                let run = match args.first() {
                    Some(run) => RunId::from_str(run)?,
                    None => self.latest_active_run()?,
                };
                self.attach(run, Some(DecimalU64(0)))?;
                vec![format!(
                    "attached to {} from event 0",
                    self.snapshot.as_ref().unwrap().run_id
                )]
            }
            "\\status" => {
                let run = match args.first() {
                    Some(run) => RunId::from_str(run)?,
                    None => self
                        .snapshot
                        .as_ref()
                        .map(|snapshot| snapshot.run_id.clone())
                        .ok_or_else(|| anyhow::anyhow!("no active run"))?,
                };
                self.snapshot = Some(self.client.run_status(&run)?);
                vec![format!(
                    "{} · {:?}",
                    run,
                    self.snapshot.as_ref().unwrap().state
                )]
            }
            "\\pause" | "\\continue" | "\\cancel" => {
                let action = &parsed.command.name[1..];
                self.control(action)?;
                vec![format!("run {} acknowledged", action)]
            }
            _ => vec![format!(
                "{} is not available in the interactive client yet",
                parsed.command.name
            )],
        };
        let mut output = Vec::new();
        for message in messages {
            let clean = crate::terminal::sanitize_user_string(&message);
            output.extend(clean.split('\n').map(str::to_owned));
        }
        Ok(output)
    }

    fn execute(&mut self, input: &str) -> Vec<String> {
        self.submit(input)
            .unwrap_or_else(|error| vec![format!("error: {error:#}")])
    }

    fn open_dataset(&mut self, args: &[String]) -> anyhow::Result<Vec<String>> {
        let path = args
            .first()
            .ok_or_else(|| anyhow::anyhow!("usage: \\open PATH"))?;
        let mut name = PathBuf::from(path)
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("dataset.csv")
            .to_owned();
        if let Some(index) = args.iter().position(|arg| arg == "--name") {
            name = args
                .get(index + 1)
                .ok_or_else(|| anyhow::anyhow!("--name needs a value"))?
                .clone();
        }
        let file = std::fs::File::open(path)?;
        let key = format!("tui_{}", ulid::Ulid::new());
        let profile: jope_api::DatasetProfile =
            jope_client::files::upload_csv(&self.client, &name, file, &key)?.json()?;
        let target = profile
            .target_suggestions
            .first()
            .map(|item| item.column.clone());
        let dataset = profile.dataset_id.to_string();
        self.draft.set_dataset(profile);
        let mut lines = vec![format!("registered {name} as {dataset}")];
        if let Some(target) = target {
            lines.push(format!("suggested target: {target}"));
        }
        if let Some(plan) = self.validate_draft()? {
            lines.extend(plan_lines(plan));
        }
        Ok(lines)
    }

    fn update_draft(&mut self, input: &str) -> anyhow::Result<Vec<String>> {
        let previous = self.draft.clone();
        if let Err(error) = self.draft.apply(input) {
            self.draft = previous;
            return Err(anyhow::Error::msg(error));
        }
        match self.validate_draft() {
            Ok(Some(plan)) => Ok(plan_lines(plan)),
            Ok(None) => Ok(vec![
                "draft updated; dataset and target are needed for dry-run validation".into(),
            ]),
            Err(error) => {
                self.draft = previous;
                Err(error)
            }
        }
    }

    fn validate_draft(&self) -> anyhow::Result<Option<Plan>> {
        let Some(mut spec) = self.draft.run_spec(true) else {
            return Ok(None);
        };
        spec.dry_run = true;
        match self
            .client
            .run_start(&spec, Some(&format!("tui_plan_{}", ulid::Ulid::new())))?
        {
            RunResult::Plan(plan) => Ok(Some(*plan)),
            RunResult::Started(_) => anyhow::bail!("dry-run unexpectedly started a run"),
        }
    }

    fn export(&mut self, args: &[String]) -> anyhow::Result<Vec<String>> {
        let run = self
            .snapshot
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("no run selected"))?
            .run_id
            .clone();
        let snapshot = self.client.run_status(&run)?;
        let best = snapshot
            .best
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("run has no champion"))?;
        let mut request = jope_api::ExportRequest {
            algorithm_id: best.algorithm_id.clone(),
            kind: jope_api::ExportKind::Bundle,
            languages: self.draft.languages(),
            require_all_languages: true,
            refit: jope_api::Refit::FitHoldout,
        };
        let mut out = None;
        let mut force = false;
        let mut index = 0;
        while index < args.len() {
            let flag = &args[index];
            if flag == "--force" {
                force = true;
                index += 1;
                continue;
            }
            let value = args
                .get(index + 1)
                .ok_or_else(|| anyhow::anyhow!("{flag} needs a value"))?;
            match flag.as_str() {
                "--out" => out = Some(PathBuf::from(value)),
                "--alg" => request.algorithm_id = value.parse()?,
                "--kind" => request.kind = serde_json::from_value(serde_json::json!(value))?,
                "--refit" => request.refit = serde_json::from_value(serde_json::json!(value))?,
                "--lang" => {
                    request.languages = value
                        .split(',')
                        .map(|name| serde_json::from_value(serde_json::json!(name)))
                        .collect::<Result<_, _>>()?
                }
                _ => anyhow::bail!("unknown export option {flag}"),
            }
            index += 2;
        }
        anyhow::ensure!(
            !request.languages.is_empty(),
            "export needs at least one language"
        );
        let out = out.ok_or_else(|| anyhow::anyhow!("export requires --out DIR"))?;
        let key = format!("tui_export_{}", ulid::Ulid::new());
        let manifest = self.client.model_export(&run, &request, Some(&key))?;
        jope_client::files::download_export_artifact(
            &self.client,
            &manifest.artifact_id,
            &out,
            force,
        )?;
        Ok(vec![format!(
            "Export {} · {} · parity {:?}",
            manifest.artifact_id,
            out.display(),
            manifest.parity_status
        )])
    }

    fn start_run(&mut self, args: &[String]) -> anyhow::Result<Vec<String>> {
        anyhow::ensure!(
            args.len().is_multiple_of(2),
            "usage: \\run [--seed S] [--idempotency-key KEY]"
        );
        let mut key = None;
        let mut seeded = false;
        for pair in args.chunks_exact(2) {
            match pair[0].as_str() {
                "--seed" if !seeded => {
                    let seed = pair[1]
                        .parse::<u64>()
                        .map_err(|_| anyhow::anyhow!("--seed must be a nonnegative integer"))?;
                    self.draft.set_seed(DecimalU64(seed));
                    seeded = true;
                }
                "--idempotency-key" if key.is_none() => key = Some(pair[1].clone()),
                _ => anyhow::bail!("usage: \\run [--seed S] [--idempotency-key KEY]"),
            }
        }
        let key = key.unwrap_or_else(|| format!("tui_run_{}", ulid::Ulid::new()));
        let plan = self
            .validate_draft()?
            .ok_or_else(|| anyhow::anyhow!("draft needs a registered dataset and target"))?;
        let mut spec = self
            .draft
            .run_spec(false)
            .expect("validated draft is complete");
        spec.plan_digest = Some(plan.plan_digest);
        match self.client.run_start(&spec, Some(&key))? {
            RunResult::Started(snapshot) => {
                let run = snapshot.run_id.clone();
                self.attach(run.clone(), Some(snapshot.event_watermark))?;
                Ok(vec![format!("run {run} started")])
            }
            RunResult::Plan(_) => anyhow::bail!("run request returned a dry-run plan"),
        }
    }

    fn attach(&mut self, run: RunId, after: Option<DecimalU64>) -> anyhow::Result<()> {
        self.snapshot = Some(self.client.run_status(&run)?);
        let client = self.client.clone();
        let sender = self.event_sender.clone();
        thread::Builder::new()
            .name("jope-terminal-events".into())
            .spawn(move || {
                let mut cursor = after;
                loop {
                    if let Ok(mut stream) = client.follow_events(run.clone(), cursor) {
                        loop {
                            match stream.next() {
                                Some(Ok(StreamEvent::Durable(event))) => {
                                    cursor = Some(event.sequence);
                                    if sender.send(*event).is_err() {
                                        return;
                                    }
                                }
                                Some(Ok(StreamEvent::Telemetry(_))) => {}
                                Some(Err(_)) | None => {
                                    cursor = stream.last_event_id().or(cursor);
                                    break;
                                }
                            }
                        }
                    }
                    thread::sleep(Duration::from_millis(100));
                }
            })?;
        Ok(())
    }

    fn latest_active_run(&self) -> anyhow::Result<RunId> {
        let value = serde_json::to_value(self.client.runs()?)?;
        let runs = value
            .get("runs")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| anyhow::anyhow!("engine did not return a run list"))?;
        for run in runs.iter().rev() {
            let Some(id) = run
                .get("run_id")
                .or_else(|| run.get("id"))
                .and_then(serde_json::Value::as_str)
            else {
                continue;
            };
            let Ok(id) = RunId::from_str(id) else {
                continue;
            };
            if let Ok(snapshot) = self.client.run_status(&id) {
                if matches!(
                    snapshot.state,
                    RunState::Running | RunState::Pausing | RunState::Paused
                ) {
                    return Ok(id);
                }
            }
        }
        anyhow::bail!("there is no active run to attach")
    }

    fn control(&mut self, action: &str) -> anyhow::Result<()> {
        let current = self
            .snapshot
            .as_ref()
            .map(|snapshot| snapshot.run_id.clone())
            .ok_or_else(|| anyhow::anyhow!("there is no active run"))?;
        let snapshot = self.client.run_status(&current)?;
        let key = format!("tui_{}", ulid::Ulid::new());
        let request = Request::new(8, 0, &[("id", &current.to_string())])?
            .key(Some(&key))?
            .if_match(snapshot.revision)
            .json(&serde_json::json!({"action": action}))?;
        self.snapshot = Some(self.client.execute(request)?.json()?);
        Ok(())
    }

    fn handle_event(&mut self, event: PublicEvent, width: usize) -> Vec<String> {
        update_snapshot_from_event(&mut self.snapshot, &event);
        crate::terminal::cards::render_transcript(
            &[event],
            width,
            crate::terminal::Theme::from_env(crate::terminal::ThemeMode::Auto),
            crate::terminal::symbols_from_env(),
        )
    }

    fn on_event(
        &mut self,
        event: PublicEvent,
        width: u16,
        terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    ) -> io::Result<()> {
        let lines = self.handle_event(event, usize::from(width));
        terminal.insert_before(lines.len().max(1) as u16, |buffer| {
            let lines = if lines.is_empty() {
                vec![Line::from("event")]
            } else {
                lines.into_iter().map(Line::from).collect()
            };
            Paragraph::new(lines).render(buffer.area, buffer);
        })?;
        Ok(())
    }

    fn leave(&self) {
        if let Some(snapshot) = &self.snapshot {
            eprintln!(
                "run continues in the background; reconnect with `jopedime attach {}`",
                snapshot.run_id
            );
        }
    }
}

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
