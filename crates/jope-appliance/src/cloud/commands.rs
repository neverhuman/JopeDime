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
                let links =
                    io::stdout().is_terminal() && crate::terminal::links::supported_from_env();
                println!(
                    "Open {}\nPairing code: {}\nLog in to your free veox Cloud Workspace (or create one) and approve the jAIn CLI in your browser.",
                    crate::terminal::links::linkify(&device.verification_uri_complete, links),
                    device.user_code
                );
                io::stdout().flush()?;
                if !no_browser && open_browser(&device.verification_uri_complete).is_err() {
                    println!("Could not open a browser here; open the link above on any device.");
                }
                client.finish_login(&device, &AtomicBool::new(false))?
            };
            println!("Logged in to {} as {}", c.origin, c.signed_in_as());
        }
    }
    Ok(())
}
pub fn cloud_command(command: CloudCommand) -> Result<()> {
    use jope_client::Transport;
    match command {
        CloudCommand::Runs => {
            let (entries, notice) = run_entries(&super::sync::local_client()?)?;
            println!("{notice}");
            for row in entries {
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
        }
        CloudCommand::Telemetry { setting } => println!("{}", sync::telemetry_command(setting)?),
        CloudCommand::Fetch { run } => println!("{}", fetch_model(&run)?),
    }
    Ok(())
}

pub fn spawn_login(
    workspace: String,
    sender: std::sync::mpsc::Sender<Vec<String>>,
    cancel: Arc<AtomicBool>,
) {
    std::thread::spawn(move || {
        let result = (|| -> Result<String> {
            let client = Client::new(&workspace, None, gateway(None, &workspace)?)?;
            let device = client.start_login()?;
            let _ = sender.send(vec![
                format!("Open {}", device.verification_uri_complete),
                format!(
                    "Pairing code: {} · log in to your free veox Cloud Workspace and approve this jAIn CLI.",
                    device.user_code
                ),
            ]);
            if open_browser(&device.verification_uri_complete).is_err() {
                let _ = sender.send(vec![
                    "Could not open a browser here; open the link above on any device.".into(),
                ]);
            }
            let c = client.finish_login(&device, &cancel)?;

            Ok(format!(
                "Logged in as {} on {}. Local datasets stay local; model publication is explicit.",
                c.signed_in_as(),
                c.origin
            ))
        })();
        let _ = sender.send(vec![
            result.unwrap_or_else(|error| format!("Login: {error:#}"))
        ]);
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
