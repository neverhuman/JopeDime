use super::*;

pub fn auth_command(command: AuthCommand) -> Result<()> {
    match command {
        AuthCommand::Status => println!("{}", account_label()),
        AuthCommand::Logout => println!("{}", logout()?),
        AuthCommand::Login {
            workspace,
            token_file,
            token_stdin,
            gateway_auth_file,
            no_browser,
        } => {
            let workspace = workspace.unwrap_or_else(default_workspace);
            let client = Client::new(
                &workspace,
                None,
                gateway(gateway_auth_file.as_deref(), &workspace)?,
            )?;
            let c = if token_file.is_some() || token_stdin {
                let mut token = String::new();
                if let Some(path) = token_file {
                    token = String::from_utf8(read_private(&path)?)?;
                } else {
                    anyhow::ensure!(!io::stdin().is_terminal(),"use browser login interactively; API tokens require a private file or redirected stdin");
                    io::stdin().take(513).read_to_string(&mut token)?;
                }
                client.save_token(token.trim().to_owned())?
            } else {
                let device = client.start_login()?;
                println!(
                    "Open {}\nPairing code: {}\nSign in and approve the CLI in your browser.",
                    device.verification_uri_complete, device.user_code
                );
                io::stdout().flush()?;
                if !no_browser {
                    let _ = open_browser(&device.verification_uri_complete);
                }
                client.finish_login(&device, &AtomicBool::new(false))?
            };
            println!("Signed in to {} as {}", c.origin, c.email);
        }
    }
    Ok(())
}
pub fn cloud_command(command: CloudCommand) -> Result<()> {
    use jope_client::Transport;
    match command {
        CloudCommand::Runs {
            json,
            local,
            cloud,
            unsynced,
            limit,
        } => {
            let (entries, notice) = run_entries(&super::sync::local_client()?)?;
            let rows = filter_runs(entries, local, cloud, unsynced, limit);
            if json {
                for row in rows {
                    println!(
                        "{}",
                        json!({"id":row.id,"status":row.status,"availability":row.availability(),
                               "local":row.local.as_ref().map(ToString::to_string),"cloudUrl":row.cloud_url,
                               "modelHere":row.model_here,"modelCloud":row.model_cloud})
                    );
                }
            } else {
                println!("{notice}");
                for row in rows {
                    println!(
                        "{} · {} · {}{}",
                        row.id,
                        row.status,
                        row.availability(),
                        row.cloud_url
                            .map(|url| format!(" · {url}"))
                            .unwrap_or_default()
                    );
                }
            }
        }
        CloudCommand::Sync { run } => {
            let local = super::sync::local_client()?;
            let version = local.system_info()?.version;
            let runs = if let Some(run) = run {
                vec![local.run_status(&run.parse()?)?]
            } else {
                crate::terminal::runs::local_runs(&local)?
            };
            for snapshot in runs {
                println!("{}", sync_snapshot(&snapshot, &version)?);
                // Reported, not fatal: a workspace without the events endpoint still syncs runs.
                println!(
                    "{}",
                    sync_events(&local, &snapshot)
                        .unwrap_or_else(|error| format!("Run events: {error:#}"))
                );
            }
            println!("{RUNS_HINT}");
        }
        CloudCommand::Publish {
            run,
            model,
            report,
            code,
        } => {
            if report.is_some() || code.is_some() {
                if let Some(model) = model {
                    println!("{}", publish_model(&run, &model)?);
                }
                if let Some(report) = report {
                    println!("{}", publish_file(&run, Upload::Report, &report)?);
                }
                if let Some(code) = code {
                    println!("{}", publish_file(&run, Upload::Code, &code)?);
                }
            } else if let Some(model) = model {
                println!("{}", publish_model(&run, &model)?);
            } else {
                println!("{}", publish_best(&run)?);
            }
            println!("{RUNS_HINT}");
        }
        CloudCommand::Results { setting } => println!("{}", results_command(setting)?),
        CloudCommand::Telemetry { setting } => println!("{}", sync::telemetry_command(setting)?),
        CloudCommand::Fetch { run } => println!("{}", fetch_model(&run)?),
        CloudCommand::Feedback {
            kind,
            summary,
            details,
            source,
            tools,
        } => {
            let mut feedback =
                super::feedback::Feedback::new(&kind, &source, &summary, details.as_deref())?;
            feedback.tools_used = tools;
            println!("{}", super::feedback::send(&feedback)?);
        }
    }
    Ok(())
}

/// Printed after a sync or publication.
pub const RUNS_HINT: &str = "See all your runs: jain cloud runs";

/// `cloud runs` filters: where a run exists, and how many.
pub(crate) fn filter_runs(
    entries: Vec<crate::terminal::runs::Entry>,
    local: bool,
    cloud: bool,
    unsynced: bool,
    limit: Option<usize>,
) -> Vec<crate::terminal::runs::Entry> {
    entries
        .into_iter()
        .filter(|e| {
            let here = e.local.is_some() || e.local_record;
            let there = e.cloud_url.is_some();
            (!local || here) && (!cloud || there) && (!unsynced || (here && !there))
        })
        .take(limit.unwrap_or(usize::MAX))
        .collect()
}

/// `cloud results on|off`, and the terminal's `/sync on|off`: one stored answer.
pub fn results_command(setting: Option<String>) -> Result<String> {
    use crate::terminal::onboarding;
    let home = jope_paths::jopedime_home();
    match setting.as_deref() {
        Some("on") => {
            onboarding::update(&home, |s| s.results_sync = Some(true))?;
        }
        Some("off") => {
            onboarding::update(&home, |s| s.results_sync = Some(false))?;
        }
        Some(_) => anyhow::bail!("use `on` or `off`"),
        None => {}
    }
    Ok(match onboarding::load(&home).results_sync {
        Some(true) => format!("Results sync on: {}", RESULTS_NOTE),
        Some(false) => "Results sync off: finished local runs keep their model and code on this machine. Turn on with /sync on.".into(),
        None => "Results sync not decided yet: you are asked after your first finished local run while logged in.".into(),
    })
}

/// What a results sync sends, for the consent question and the setting.
pub const RESULTS_NOTE: &str = "after a local run finishes, its fitted model and generated code go to your workspace. No training rows are uploaded, except that kernel, smoother and Gaussian-process models keep the support rows they predict from (rows of your training data) inside the fitted model.";

/// The question asked once, the first time a logged-in local run finishes.
pub const RESULTS_QUESTION: &str = "Sync results (model + code, not your data) to your workspace? No training rows are uploaded, except that kernel, smoother and Gaussian-process models keep the support rows they predict from inside the fitted model. [Y/n]";

pub fn spawn_login(
    workspace: String,
    sender: std::sync::mpsc::Sender<Vec<String>>,
    cancel: Arc<AtomicBool>,
    done: Option<std::sync::mpsc::Sender<bool>>,
) {
    std::thread::spawn(move || {
        let result = super::flow::login_blocking(&workspace, &cancel, &mut |line| {
            let _ = sender.send(vec![line]);
        })
        .map(|c| {
            format!(
                "Signed in to {} as {}. Local datasets stay local; model publication is explicit.",
                c.origin, c.email
            )
        });
        let ok = result.is_ok();
        let _ = sender.send(vec![
            result.unwrap_or_else(|error| format!("Login: {error:#}"))
        ]);
        if let Some(done) = done {
            let _ = done.send(ok);
        }
    });
}
pub fn default_workspace() -> String {
    std::env::var("JAIN_WORKSPACE")
        .ok()
        .or_else(|| connection().ok().map(|c| c.origin))
        .or_else(|| {
            cloud_root()
                .ok()
                .and_then(|root| read_private(&root.join("workspace-origin")).ok())
                .and_then(|bytes| String::from_utf8(bytes).ok())
                .and_then(|value| origin(&value).ok())
        })
        .unwrap_or_else(|| "https://app.veox.ai".into())
}
