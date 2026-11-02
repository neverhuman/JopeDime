use super::*;

impl super::Session {
    pub(super) fn run(&mut self) -> anyhow::Result<()> {
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
            while let Ok(result) = self.background_events.try_recv() {
                let lines = result.into_messages();
                for line in lines {
                    self.screen.push(&line);
                }
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

    pub(super) fn line_mode(
        &mut self,
        guard: &crate::terminal::restore::TerminalGuard,
    ) -> anyhow::Result<()> {
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
            while let Ok(result) = self.background_events.try_recv() {
                let lines = result.into_messages();
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
}
