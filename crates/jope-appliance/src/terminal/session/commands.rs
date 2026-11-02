use super::*;

impl super::Session {
    pub(super) fn submit(&mut self, input: &str) -> anyhow::Result<Vec<String>> {
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

    pub(super) fn execute(&mut self, input: &str) -> Vec<String> {
        self.submit(input)
            .unwrap_or_else(|error| vec![format!("error: {error:#}")])
    }

    pub(super) fn open_dataset(&mut self, args: &[String]) -> anyhow::Result<Vec<String>> {
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
            "Export {} · {} · parity {:?}",
            manifest.artifact_id,
            out.display(),
            manifest.parity_status
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
}
