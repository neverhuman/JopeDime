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
                    "Pairing code: {} · sign in and approve this CLI in your browser.",
                    device.user_code
                ),
            ]);
            let _ = open_browser(&device.verification_uri_complete);
            let c = client.finish_login(&device, &cancel)?;

            Ok(format!(
                "Signed in to {} as {}. Local datasets stay local; model publication is explicit.",
                c.origin, c.email
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
