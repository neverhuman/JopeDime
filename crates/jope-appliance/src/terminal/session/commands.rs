use super::routing;
use super::*;

impl super::Session {
    pub(super) fn submit(&mut self, input: &str) -> anyhow::Result<Vec<String>> {
        let input = input.trim();
        if let Some(result) = self.answer(input) {
            return result.map(|lines| clean_lines(&lines));
        }
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
                    Some(self.login_done_sender.clone()),
                );
                self.login_cancel = Some(cancel);
                return Ok(vec![
                    "Starting browser login to your veox Cloud Workspace; Ctrl-C exits without stopping background runs.".into(),
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
            Some("/feedback" | "\\feedback") => {
                let rest = input
                    .split_once(char::is_whitespace)
                    .map(|(_, rest)| rest.trim())
                    .unwrap_or("");
                anyhow::ensure!(
                    !rest.is_empty(),
                    "usage: /feedback [issue|idea|good] WHAT HAPPENED, e.g. /feedback issue the upload stalls at 90%"
                );
                // A leading kind word is optional: `/feedback the export button is great` is
                // still feedback, filed as `other`.
                let (kind, text) = match rest.split_once(char::is_whitespace) {
                    Some((word, tail))
                        if crate::cloud::feedback::kind_from_word(word).is_some()
                            && !tail.trim().is_empty() =>
                    {
                        (word, tail.trim())
                    }
                    _ => ("other", rest),
                };
                let (summary, details) = crate::cloud::feedback::split_text(text);
                let feedback = crate::cloud::feedback::Feedback::new(
                    kind,
                    "user",
                    &summary,
                    details.as_deref(),
                )?;
                let sender = self.cloud_sender.clone();
                thread::spawn(move || {
                    let message = crate::cloud::feedback::send(&feedback)
                        .unwrap_or_else(|error| format!("Feedback: {error:#}"));
                    let _ = sender.send(vec![message]);
                });
                return Ok(vec!["Sending your feedback to the VEOX team...".into()]);
            }
            Some("/usage" | "\\usage") => {
                anyhow::ensure!(words.next().is_none(), "usage: /usage");
                return Ok(self.usage());
            }
            Some("/runs" | "\\runs") => {
                self.refresh_runs();
                return Ok(vec!["Loading runs...".into()]);
            }
            Some("/sync" | "\\sync")
                if matches!(input.split_whitespace().nth(1), Some("on" | "off")) =>
            {
                let setting = words.next().map(str::to_owned);
                anyhow::ensure!(words.next().is_none(), "usage: /sync on|off");
                return Ok(vec![crate::cloud::results_command(setting)?]);
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
                self.background_sync(runs, Some(owner), false);
                return Ok(vec![
                    "Syncing run status (progress and scores) to your veox Cloud Workspace. Dataset rows and model bytes stay local.".into(),
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
                let sender = self.background_sender.clone();
                let busy = Arc::clone(&self.cloud_busy);
                thread::spawn(move || {
                    let result = if let Some(model) = model {
                        crate::cloud::publish_model(&run.to_string(), &model)
                    } else {
                        crate::cloud::publish_best(&run.to_string())
                    };
                    busy.store(false, Ordering::SeqCst);
                    let _ = sender.send(BackgroundResult::Publish(result));
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
            "\\help" => {
                if args.first().is_some_and(|arg| arg == "history") {
                    vec![crate::terminal::composer::HISTORY_HELP.to_owned()]
                } else {
                    help_lines()
                }
            }
            "\\demo" => self.demo(&args)?,
            "\\auto" => {
                anyhow::ensure!(args.is_empty(), "usage: /auto");
                if self.draft.run_spec(true).is_none() {
                    vec!["/auto runs the default search on an opened dataset: /open FILE.csv (or /demo NAME) first.".into()]
                } else {
                    // The default search: the opened table, its suggested target, the default
                    // effort and objective, routed exactly as /run routes it.
                    let mut lines =
                        vec!["Running the default search on the opened dataset.".to_owned()];
                    lines.extend(self.route_run(&[])?);
                    lines
                }
            }
            "\\copy" => self.copy(&args)?,
            "\\transcript" => {
                anyhow::ensure!(args.is_empty(), "usage: /transcript");
                if self.fullscreen {
                    self.transcript_requested = true;
                    Vec::new()
                } else {
                    vec!["This plain view already writes everything to your terminal: scroll back and select it there.".into()]
                }
            }
            "\\mouse" => {
                let capture = match args.first().map(String::as_str) {
                    None => !self.mouse,
                    Some("on") => true,
                    Some("off") => false,
                    Some(_) => anyhow::bail!("usage: /mouse [on|off]"),
                };
                self.set_mouse(capture);
                vec![mouse_line(capture)]
            }
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
            "\\run" => self.route_run(&args)?,
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
        Ok(clean_lines(&messages))
    }

    pub(super) fn execute(&mut self, input: &str) -> Vec<String> {
        self.submit(input)
            .unwrap_or_else(|error| vec![format!("error: {error:#}")])
    }

    pub(super) fn open_dataset(&mut self, args: &[String]) -> anyhow::Result<Vec<String>> {
        use crate::terminal::{demos, onboarding};
        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let Some(path) = args.first() else {
            return Ok(crate::terminal::complete::open_listing(&cwd));
        };
        let expanded = expand_home(path);
        let path = &expanded;
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
        // A real file takes precedence; `example/<name>.csv` otherwise opens a bundled demo.
        let on_disk = std::path::Path::new(path).exists();
        // A file holding exactly a bundled demo's bytes (such as the copy `jain run
        // example/NAME.csv` writes) is that demo, not the user's own data.
        let bundled_file = on_disk && demos::is_bundled(std::path::Path::new(path));
        let mut notice = Vec::new();
        let reader: Box<dyn io::Read + Send> = match demos::from_path(path).filter(|_| !on_disk) {
            Some(demo) => {
                notice = demo.license_notice();
                Box::new(io::Cursor::new(demo.bytes))
            }
            None => {
                if !on_disk
                    && path
                        .replace('\\', "/")
                        .trim_start_matches("./")
                        .starts_with(demos::PREFIX)
                {
                    let mut lines = vec![format!("There is no bundled demo {path}.")];
                    lines.extend(demos::listing());
                    return Ok(lines);
                }
                if !bundled_file {
                    if let Err(message) = onboarding::own_file_gate() {
                        return Ok(vec![message]);
                    }
                }
                let full = cwd.join(path);
                if full.is_dir() {
                    return Ok(crate::terminal::complete::open_listing(&full));
                }
                if !on_disk {
                    let mut lines = vec![format!(
                        "No such file: {path} (looked in {})",
                        cwd.display()
                    )];
                    lines.extend(crate::terminal::complete::open_listing(&cwd));
                    return Ok(lines);
                }
                Box::new(std::fs::File::open(path)?)
            }
        };
        let demo = !on_disk || bundled_file;
        let own = !demo && crate::cloud::route::cloud_first();
        notice.extend(self.register(&name, reader)?);
        if self.draft.run_spec(true).is_none() {
            notice.push("Choose the column to predict with /target COLUMN, then type /run.".into());
        } else if own {
            notice.push(routing::OPENED_OWN.into());
        } else {
            notice.push(
                "Type /run to start (or /auto for the defaults; /target COLUMN predicts another column)."
                    .into(),
            );
        }
        if let Some(source) = self.routing.source.as_mut() {
            source.demo = demo;
            if !demo {
                source.path = std::fs::canonicalize(path).ok();
            }
        }
        Ok(notice)
    }

    pub(super) fn register(
        &mut self,
        name: &str,
        reader: Box<dyn io::Read + Send>,
    ) -> anyhow::Result<Vec<String>> {
        let key = format!("tui_{}", ulid::Ulid::new());
        let profile: jope_api::DatasetProfile =
            jope_client::files::upload_csv(&self.client, name, reader, &key)?.json()?;
        let target = profile
            .target_suggestions
            .first()
            .map(|item| item.column.clone());
        let dataset = profile.dataset_id.to_string();
        self.routing.source = Some(routing::Source {
            path: None,
            demo: false,
            rows: profile.rows.0,
            columns: profile.columns.len() as u64,
            name: name.to_owned(),
            profile: Some(profile.clone()),
        });
        self.draft.set_dataset(profile);
        let mut lines = vec![format!("registered {name} as {dataset}")];
        if let Some(target) = target {
            lines.push(format!("suggested target column: `{target}`"));
        }
        if let Some(plan) = self.validate_draft()? {
            lines.extend(plan_lines(plan));
        }
        Ok(lines)
    }

    pub(super) fn update_draft(&mut self, input: &str) -> anyhow::Result<Vec<String>> {
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

    pub(super) fn validate_draft(&self) -> anyhow::Result<Option<Plan>> {
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

    pub(super) fn export(&mut self, args: &[String]) -> anyhow::Result<Vec<String>> {
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
            "Export {} · {} · parity {}",
            manifest.artifact_id,
            out.display(),
            manifest.parity_status.explanation()
        )])
    }

    pub(super) fn start_run(&mut self, args: &[String]) -> anyhow::Result<Vec<String>> {
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
        let started = self.client.run_start(&spec, Some(&key)).map_err(|error| {
            let text = error.to_string();
            if crate::terminal::runs::is_busy(&text) {
                let run = crate::terminal::runs::active_run(&self.client);
                anyhow::anyhow!(crate::terminal::runs::busy_message(run.as_ref(), true))
            } else {
                anyhow::Error::new(error)
            }
        })?;
        match started {
            RunResult::Started(snapshot) => {
                let run = snapshot.run_id.clone();
                let source = self.routing.source.clone().unwrap_or_default();
                crate::cloud::remember_run(
                    &run,
                    &crate::cloud_events::RunContext::from_spec(
                        &spec,
                        &source.name,
                        source.profile,
                    ),
                );
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
}

/// Split every message into lines first, then strip terminal controls from each, so a
/// multi-line message never runs its lines together (the old "\\quitDocs"; #2148).
pub(super) fn clean_lines(messages: &[String]) -> Vec<String> {
    messages
        .iter()
        .flat_map(|message| message.split('\n'))
        .map(crate::terminal::sanitize_user_string)
        .collect()
}

/// `/help`: one command per line, all with the `/` prefix (`\\` still works as an alias).
pub(super) fn help_lines() -> Vec<String> {
    const ROWS: &[(&str, &str)] = &[
        (
            "/demo [NAME]",
            "list the bundled demos, or run one (no account needed)",
        ),
        (
            "/open PATH",
            "open a bundled example/NAME.csv, or your CSV once signed in (~ works)",
        ),
        ("/target COLUMN", "choose the column to predict"),
        (
            "/run [--local|--cloud]",
            "start a run: demos here, your own data in the veox Cloud (asks first)",
        ),
        ("/auto", "run the default search on the opened dataset"),
        ("/plan", "check the run plan without starting it"),
        (
            "/effort fast|medium|high|ultra",
            "how long to search (--evals N, --time 60s)",
        ),
        ("/status · /pause · /continue · /cancel", "the current run"),
        ("/attach RUN", "follow a run (Left opens the runs panel)"),
        ("/export --out DIR", "write the champion's code and model"),
        (
            "/copy [link|run|result|all]",
            "copy to the clipboard (default: the last output)",
        ),
        (
            "/transcript",
            "show everything as plain text to select and copy",
        ),
        (
            "/mouse [on|off]",
            "F2: mouse wheel scrolling, or text selection (default)",
        ),
        (
            "/login [URL] · /logout",
            "sign in to your veox account (your own data trains in the cloud)",
        ),
        ("/usage", "account, plan and this machine's runs"),
        (
            "/feedback [issue|idea|good] TEXT",
            "tell the VEOX team what works, what broke or what you need",
        ),
        (
            "/sync [RUN] · /sync on|off",
            "send run status, or turn results sync on or off",
        ),
        (
            "/publish [RUN] · /fetch UUID",
            "publish a model, or download one",
        ),
        ("/help history", "history and editing keys"),
        (
            "/quit or /exit [--pause]",
            "leave; runs continue in the background",
        ),
    ];
    let mut lines = vec!["Commands:".to_owned()];
    lines.extend(
        ROWS.iter()
            .map(|(command, what)| format!("  {command:<40} {what}")),
    );
    lines.push("Keys: Left runs · PgUp/PgDn scroll · F2 mouse · Ctrl-C exit".into());
    lines.push("Docs https://veox.ai/docs".into());
    lines.push("Terms https://veox.ai/terms".into());
    lines
}

pub(super) fn mouse_line(capture: bool) -> String {
    if capture {
        "Mouse: the wheel scrolls; hold Shift while dragging to select text (F2 or /mouse off for plain selection).".into()
    } else {
        "Mouse: select and copy text with your terminal as usual; PgUp/PgDn scroll (F2 or /mouse on for wheel scrolling).".into()
    }
}

/// `~` and `~/…` name the home directory, as in a shell (#2148).
pub(super) fn expand_home(path: &str) -> String {
    let home = || std::env::var_os("HOME").filter(|home| !home.is_empty());
    if path == "~" {
        if let Some(home) = home() {
            return home.to_string_lossy().into_owned();
        }
    }
    if let Some(rest) = path.strip_prefix("~/").or_else(|| path.strip_prefix("~\\")) {
        if let Some(home) = home() {
            return std::path::Path::new(&home)
                .join(rest)
                .to_string_lossy()
                .into_owned();
        }
    }
    path.to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn help_is_one_slash_command_per_line_and_never_runs_together() {
        let lines = help_lines();
        assert!(lines.len() > 10);
        assert!(lines.iter().all(|line| !line.contains('\\')), "{lines:?}");
        assert!(lines.iter().any(|line| line.contains("/quit or /exit")));
        assert!(lines
            .iter()
            .any(|line| line.trim_start().starts_with("/auto")));
        assert!(lines
            .iter()
            .any(|line| line.trim_start().starts_with("/copy")));
        assert_eq!(lines.last().unwrap(), "Terms https://veox.ai/terms");
        assert!(!lines.iter().any(|line| line.contains("quitDocs")));
        let split = clean_lines(&["a\nb\u{1b}[31m".into()]);
        assert_eq!(split, ["a", "b"]);
    }

    #[test]
    fn a_leading_tilde_names_the_home_directory() {
        let home = std::env::var("HOME").unwrap_or_default();
        if home.is_empty() {
            return;
        }
        assert_eq!(
            expand_home("~/my-sales.csv"),
            format!("{home}/my-sales.csv")
        );
        assert_eq!(expand_home("~"), home);
        assert_eq!(expand_home("data/~x.csv"), "data/~x.csv");
        assert_eq!(expand_home("~other/x.csv"), "~other/x.csv");
    }
}
