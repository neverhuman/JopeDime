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
    login_cancel: Option<Arc<AtomicBool>>,
    run_lists: mpsc::Receiver<anyhow::Result<(Vec<crate::terminal::runs::Entry>, String)>>,
    run_list_sender: mpsc::Sender<anyhow::Result<(Vec<crate::terminal::runs::Entry>, String)>>,
    cloud_busy: Arc<AtomicBool>,
    auto_sync: Option<(RunId, (String, String))>,
    last_sync: Instant,
    synced_terminal: mpsc::Receiver<RunId>,
    synced_terminal_sender: mpsc::Sender<RunId>,
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
        let (run_list_sender, run_lists) = mpsc::channel();
        let (synced_terminal_sender, synced_terminal) = mpsc::channel();
        let history = home.join("history");
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
            login_cancel: None,
            run_lists,
            run_list_sender,
            cloud_busy: Arc::new(AtomicBool::new(false)),
            auto_sync: None,
            last_sync: Instant::now(),
            synced_terminal,
            synced_terminal_sender,
        })
    }

    fn run(&mut self) -> anyhow::Result<()> {
        let width = terminal::size().map(|size| size.0).unwrap_or(80);
        let fullscreen = !crate::terminal::line_mode::required(
            false,
            std::env::var("TERM").ok().as_deref(),
            false, // NO_COLOR controls styling; it does not disable the full-screen UI.
            width,
        );
        let guard = if fullscreen {
            crate::terminal::restore::TerminalGuard::acquire_fullscreen()?
        } else {
            crate::terminal::restore::TerminalGuard::acquire(false)?
        };
        if !fullscreen {
            return self.line_mode(&guard);
        }
        let mut terminal = Terminal::new(CrosstermBackend::new(io::stdout()))?;
        let mut last_draw = None;
        loop {
            if guard.signal().is_some() {
                guard.leave_inline()?;
                self.leave();
                return Ok(());
            }
            let width = terminal::size().map(|size| size.0).unwrap_or(80);
            if width < 40 || terminal::size().is_ok_and(|(_, rows)| rows < 12) {
                terminal.clear()?;
                drop(terminal);
                guard.leave_inline()?;
                let retained = self.screen_lines();
                crate::terminal::line_mode::LineMode::new(io::stdout()).transcript(&retained)?;
                return self.line_mode(&guard);
            }
            while let Ok(result) = self.run_lists.try_recv() {
                if let Some(browser) = &mut self.screen.browser {
                    match result {
                        Ok((entries, notice)) => browser.replace(entries, notice),
                        Err(error) => browser.notice = format!("Refresh failed: {error:#}"),
                    }
                }
                last_draw = None;
            }
            self.tick_sync();
            while let Ok(lines) = self.cloud_events.try_recv() {
                for line in lines {
                    self.screen.push(&line);
                }
                self.screen.cloud_label = crate::cloud::account_label();
                last_draw = None;
            }
            while let Ok(event) = self.events.try_recv() {
                for line in self.handle_event(event, usize::from(width)) {
                    self.screen.push(&line);
                }
                last_draw = None;
            }
            if event::poll(INPUT_LATENCY)? {
                let input = event::read()?;
                last_draw = None;
                if matches!(&input, Event::Key(key) if key.kind != KeyEventKind::Release && key.code == KeyCode::Left)
                    && self.composer.text().is_empty()
                    && self.screen.browser.is_none()
                {
                    self.refresh_runs();
                    continue;
                }
                let exit_key = matches!(&input, Event::Key(key) if key.modifiers.contains(KeyModifiers::CONTROL) && (key.code == KeyCode::Char('c') || (key.code == KeyCode::Char('d') && self.composer.text().is_empty())));
                if self.screen.browser.is_some() && !exit_key {
                    let action = self.screen.browser.as_mut().unwrap().handle(&input);
                    match action {
                        crate::terminal::runs::Action::None => {}
                        crate::terminal::runs::Action::Close => self.screen.browser = None,
                        crate::terminal::runs::Action::Refresh => self.refresh_runs(),
                        crate::terminal::runs::Action::Attach(run) => {
                            match self.attach(run, Some(DecimalU64(0))) {
                                Ok(()) => self.screen.browser = None,
                                Err(error) => {
                                    self.screen.browser.as_mut().unwrap().notice = error.to_string()
                                }
                            }
                        }
                        crate::terminal::runs::Action::Fetch(id) => {
                            let sender = self.cloud_sender.clone();
                            thread::spawn(move || {
                                let _ = sender.send(vec![crate::cloud::fetch_model(&id)
                                    .unwrap_or_else(|error| format!("Download: {error:#}"))]);
                            });
                            self.screen.browser.as_mut().unwrap().notice =
                                "Downloading model; r refreshes availability".into();
                        }
                        crate::terminal::runs::Action::Open(url) => {
                            if let Err(error) = open_cloud_url(&url) {
                                self.screen.browser.as_mut().unwrap().notice = error.to_string();
                            }
                        }
                    }
                    self.draw(&mut terminal)?;
                    continue;
                }
                if self.screen.handle_scroll(&input) {
                    self.draw(&mut terminal)?;
                    last_draw = Some(Instant::now());
                    continue;
                }
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
                        self.screen.follow();
                        let first = text
                            .split_whitespace()
                            .next()
                            .unwrap_or("")
                            .trim_start_matches(['/', '\\']);
                        self.screen.push(&if first.eq_ignore_ascii_case("login") {
                            "> /login".into()
                        } else {
                            format!("> {text}")
                        });
                        for line in self.execute(&text) {
                            self.screen.push(&line);
                        }
                        if self.exit_requested {
                            guard.leave_inline()?;
                            self.leave();
                            return Ok(());
                        }
                    }
                    crate::terminal::composer::ComposerAction::ExitRequested => {
                        guard.leave_inline()?;
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
            "JAIN by VEOX · Powered by JopeDime {} · engine {}",
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
            self.tick_sync();
            while let Ok(lines) = self.cloud_events.try_recv() {
                output.transcript(&lines)?;
            }
            while let Ok(result) = self.run_lists.try_recv() {
                match result {
                    Ok((entries, notice)) => {
                        let mut lines = vec![notice];
                        lines.extend(entries.into_iter().map(|row| {
                            format!(
                                "{} · {} · {} · {}",
                                row.id,
                                row.status,
                                row.availability(),
                                row.cloud_url.as_deref().unwrap_or("local only")
                            )
                        }));
                        output.transcript(&lines)?;
                    }
                    Err(error) => output.transcript(&[format!("Run refresh: {error:#}")])?,
                }
            }
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
                self.leave();
                return Ok(());
            }
        }
    }

    fn draw(&mut self, terminal: &mut Terminal<CrosstermBackend<io::Stdout>>) -> io::Result<()> {
        let width = usize::from(terminal.size()?.width);
        let (status, activity) = self.status_lines(width);
        terminal.draw(|frame| {
            self.screen
                .draw(frame, &self.composer, &self.version, &status, &activity);
        })?;
        Ok(())
    }

    fn refresh_runs(&mut self) {
        self.screen
            .browser
            .get_or_insert_with(Default::default)
            .notice = "Loading runs...".into();
        let client = self.client.clone();
        let sender = self.run_list_sender.clone();
        thread::spawn(move || {
            let _ = sender.send(crate::cloud::run_entries(&client));
        });
    }
    fn background_sync(&mut self, runs: Vec<RunId>, owner: Option<(String, String)>) {
        if self.cloud_busy.swap(true, Ordering::SeqCst) {
            return;
        }
        let busy = Arc::clone(&self.cloud_busy);
        let done = self.synced_terminal_sender.clone();
        let client = self.client.clone();
        let sender = self.cloud_sender.clone();
        thread::spawn(move || {
            let result = (|| -> anyhow::Result<Vec<String>> {
                let version = client.system_info()?.version;
                let mut lines = Vec::new();
                for run in runs {
                    let snapshot = client.run_status(&run)?;
                    lines.push(crate::cloud::sync_snapshot_for(
                        &snapshot,
                        &version,
                        owner.as_ref(),
                    )?);
                    if matches!(
                        snapshot.state,
                        RunState::Completed | RunState::Cancelled | RunState::Failed
                    ) {
                        let _ = done.send(run);
                    }
                }
                Ok(lines)
            })();
            busy.store(false, Ordering::SeqCst);
            let _ = sender.send(result.unwrap_or_else(|error| vec![format!("Sync: {error:#}")]));
        });
    }

    fn tick_sync(&mut self) {
        while let Ok(run) = self.synced_terminal.try_recv() {
            if self
                .auto_sync
                .as_ref()
                .is_some_and(|(current, _)| current == &run)
            {
                self.auto_sync = None;
            }
        }
        if self.last_sync.elapsed() >= Duration::from_secs(5) {
            if let Some((run, owner)) = self.auto_sync.clone() {
                let same =
                    crate::cloud::connection().is_ok_and(|c| (c.origin, c.account_id) == owner);
                if !same {
                    self.auto_sync = None;
                } else if !self.cloud_busy.load(Ordering::SeqCst) {
                    self.background_sync(vec![run], Some(owner));
                }
            }
            self.last_sync = Instant::now();
        }
    }

    fn screen_lines(&self) -> Vec<String> {
        self.screen.transcript()
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
        let mut words = input.split_whitespace();
        match words.next() {
            Some("/login" | "\\login") => {
                if let Some(cancel) = self.login_cancel.take() {
                    cancel.store(true, Ordering::SeqCst);
                }
                let workspace = words
                    .next()
                    .map(str::to_owned)
                    .unwrap_or_else(crate::cloud::default_workspace);
                anyhow::ensure!(
                    words.next().is_none(),
                    "usage: /login [https://workspace-origin]"
                );
                crate::cloud::origin(&workspace)?;
                let cancel = Arc::new(AtomicBool::new(false));
                crate::cloud::spawn_login(
                    workspace,
                    self.cloud_sender.clone(),
                    Arc::clone(&cancel),
                );
                self.login_cancel = Some(cancel);
                return Ok(vec![
                    "Starting browser login; Ctrl-C exits without stopping background runs.".into(),
                ]);
            }
            Some("/logout" | "\\logout") => {
                self.auto_sync = None;
                if let Some(cancel) = self.login_cancel.take() {
                    cancel.store(true, Ordering::SeqCst);
                }
                let sender = self.cloud_sender.clone();
                thread::spawn(move || {
                    let message = crate::cloud::logout()
                        .unwrap_or_else(|error| format!("Sign out: {error:#}"));
                    let _ = sender.send(vec![message]);
                });
                return Ok(vec!["Signing this CLI out...".into()]);
            }
            Some("/runs" | "\\runs") => {
                self.refresh_runs();
                return Ok(vec!["Loading runs...".into()]);
            }
            Some("/sync" | "\\sync") => {
                let c = crate::cloud::connection()?;
                anyhow::ensure!(c.may_publish, "this account token is read-only");
                let owner = (c.origin, c.account_id);
                let runs = if let Some(run) = words.next() {
                    vec![RunId::from_str(run)?]
                } else if let Some(snapshot) = &self.snapshot {
                    vec![snapshot.run_id.clone()]
                } else {
                    crate::terminal::runs::local_runs(&self.client)?
                        .into_iter()
                        .map(|r| r.run_id)
                        .collect()
                };
                anyhow::ensure!(words.next().is_none(), "usage: /sync [RUN]");
                if let Some(run) = runs.first().filter(|_| runs.len() == 1) {
                    self.auto_sync = Some((run.clone(), owner.clone()));
                }
                self.background_sync(runs, Some(owner));
                return Ok(vec![
                    "Synchronizing run metadata. Dataset rows and model bytes stay local.".into(),
                ]);
            }
            Some("/publish" | "\\publish") => {
                let args = crate::terminal::commands::tokenize(
                    input
                        .split_once(char::is_whitespace)
                        .map(|(_, rest)| rest)
                        .unwrap_or(""),
                    false,
                )?;
                let mut index = 0;
                let run = if args.first().is_some_and(|a| a != "--model") {
                    index += 1;
                    RunId::from_str(&args[0])?
                } else {
                    self.snapshot
                        .as_ref()
                        .ok_or_else(|| anyhow::anyhow!("attach a run or specify its ID"))?
                        .run_id
                        .clone()
                };
                let model = if args.len() == index {
                    None
                } else {
                    anyhow::ensure!(
                        args.len() == index + 2 && args[index] == "--model",
                        "usage: /publish [RUN] [--model PATH]"
                    );
                    Some(PathBuf::from(&args[index + 1]))
                };
                if self.cloud_busy.swap(true, Ordering::SeqCst) {
                    return Ok(vec!["A cloud operation is already in progress.".into()]);
                }
                let sender = self.cloud_sender.clone();
                let busy = Arc::clone(&self.cloud_busy);
                thread::spawn(move || {
                    let result = if let Some(model) = model {
                        crate::cloud::publish_model(&run.to_string(), &model)
                    } else {
                        crate::cloud::publish_best(&run.to_string())
                    };
                    busy.store(false, Ordering::SeqCst);
                    let _ = sender.send(vec![result.unwrap_or_else(|e| format!("Publish: {e:#}"))]);
                });
                return Ok(vec!["Publishing the selected fitted model explicitly. Fitted state can include training rows.".into()]);
            }
            Some("/fetch" | "\\fetch") => {
                let run = words
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("usage: /fetch CLOUD_UUID"))?
                    .to_owned();
                anyhow::ensure!(words.next().is_none(), "usage: /fetch CLOUD_UUID");
                let sender = self.cloud_sender.clone();
                thread::spawn(move || {
                    let _ = sender.send(vec![crate::cloud::fetch_model(&run)
                        .unwrap_or_else(|error| format!("Download: {error:#}"))]);
                });
                return Ok(vec![
                    "Downloading and verifying the selected cloud model...".into(),
                ]);
            }
            _ => {}
        }
        let parsed = crate::terminal::commands::parse(input)?;
        let args = parsed.arguments;
        let messages = match parsed.command.name {
            "\\help" => vec![if args.first().is_some_and(|arg| arg == "history") {
                crate::terminal::composer::HISTORY_HELP.to_owned()
            } else {
                "/login [URL] · /logout · /sync [RUN] · /publish [RUN] · /fetch UUID · Left: runs · \\open PATH · \\target COLUMN · \\plan · \\run · \\attach RUN · \\quit".into()
            }],
            "\\quit" => {
                if args.iter().any(|arg| arg == "--pause") {
                    self.control("pause")?;
                }
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
                if let Ok(c) = crate::cloud::connection() {
                    if c.may_publish {
                        self.auto_sync = Some((run.clone(), (c.origin, c.account_id)));
                        self.last_sync = Instant::now() - Duration::from_secs(5);
                    }
                }
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

fn open_cloud_url(url: &str) -> anyhow::Result<()> {
    anyhow::ensure!(
        url.starts_with("https://") && !url.chars().any(char::is_control),
        "invalid cloud URL"
    );
    #[cfg(target_os = "macos")]
    let command = "open";
    #[cfg(not(target_os = "macos"))]
    let command = "xdg-open";
    std::process::Command::new(command)
        .arg(url)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()?;
    Ok(())
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
