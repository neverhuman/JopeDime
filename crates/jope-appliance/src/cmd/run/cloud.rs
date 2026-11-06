//! `jopedime run` routing. Scripts and agents get the same rule as the terminal, with one
//! difference: nothing is uploaded unless `--cloud` is passed. When the rule would send a run to
//! the cloud (a Pro deep search, a large table) the run stays local and one stderr line says how
//! to send it. With `--cloud`: a just-in-time device login if needed, the per-dataset upload
//! consent (remembered, asked on a terminal, or given with `--consent-upload`), then progress on
//! stderr and a `cloud_run` JSON line on stdout.
use crate::cloud::{
    cloud_run::{Refusal, Update},
    flow::{self, Prepared},
    route::{self, Force},
};
use std::io::{BufRead, IsTerminal, Write};
use std::sync::atomic::AtomicBool;

pub(super) enum Routed {
    Local,
    Cloud,
}

fn ask(question: &str) -> Option<bool> {
    if !std::io::stdin().is_terminal() || !std::io::stderr().is_terminal() {
        return None;
    }
    eprint!("{question} ");
    let _ = std::io::stderr().flush();
    let mut line = String::new();
    std::io::stdin().lock().read_line(&mut line).ok()?;
    route::answer(&line)
}

pub(super) fn route_run(
    args: &super::RunArgs,
    path: &std::path::Path,
    profile: &jope_api::DatasetProfile,
    effort: jope_api::Effort,
) -> anyhow::Result<Routed> {
    let force = match (args.cloud, args.local) {
        (true, _) => Some(Force::Cloud),
        (_, true) => Some(Force::Local),
        _ => None,
    };
    let rows = profile.rows.0;
    let connection = crate::cloud::connection().ok();
    let needs_tier = force != Some(Force::Local)
        && (effort != jope_api::Effort::Fast || rows > route::LOCAL_ROW_THRESHOLD);
    let tier = match (&connection, needs_tier) {
        (Some(c), true) => flow::tier(c).ok().flatten(),
        _ => None,
    };
    let decision = route::decide(route::Input {
        effort,
        rows: Some(rows),
        demo: false,
        force,
        tier,
    });
    match (decision.target, force) {
        (route::Target::Cloud, Some(Force::Cloud)) => {}
        (route::Target::Cloud, _) => {
            eprintln!(
                "{} Scripts upload only with --cloud, so it runs locally on this machine.",
                decision.line
            );
            return Ok(Routed::Local);
        }
        (route::Target::Local, _) if decision.offer_cloud => {
            eprintln!(
                "{} Pass --cloud to run it in your veox Cloud Workspace.",
                decision.line
            );
            return Ok(Routed::Local);
        }
        (route::Target::Local, Some(_)) | (route::Target::Local, None)
            if decision.reason != route::Reason::Default =>
        {
            eprintln!("{}", decision.line);
            return Ok(Routed::Local);
        }
        (route::Target::Local, _) => return Ok(Routed::Local),
    }
    eprintln!("{}", decision.line);
    let cancel = AtomicBool::new(false);
    let c = match connection {
        Some(c) => c,
        None => {
            eprintln!("{}", flow::NEEDS_LOGIN);
            let c = flow::login_blocking(&crate::cloud::default_workspace(), &cancel, &mut |l| {
                eprintln!("{l}")
            })?;
            eprintln!("Signed in to {} as {}.", c.origin, c.email);
            c
        }
    };
    let prepared = Prepared::read(path, rows, profile.columns.len() as u64)?;
    let home = jope_paths::jopedime_home();
    let key = flow::consent_key(&c.origin, &c.account_id, &prepared.sha256);
    if !flow::consented(&crate::terminal::onboarding::load(&home), &key, &prepared) {
        let agreed = if args.consent_upload {
            eprintln!("{} yes (--consent-upload)", prepared.question());
            true
        } else {
            match ask(&prepared.question()) {
                Some(answer) => answer,
                None => anyhow::bail!(
                    "uploading {} needs the user's consent: ask them, then pass --consent-upload (or run without --cloud)",
                    prepared.file_name
                ),
            }
        };
        if !agreed {
            eprintln!("Nothing was uploaded.");
            return Ok(match ask(route::Limit::local_question()) {
                Some(true) => Routed::Local,
                _ => anyhow::bail!("not started"),
            });
        }
        flow::remember_consent(&home, &key, &prepared)?;
    }
    let mut started = None;
    let mut finished = Vec::new();
    let mut errors = Vec::new();
    let outcome = flow::submit_and_follow(
        &c,
        &prepared,
        route::profile(effort),
        decision.cloud_fallback(),
        Some(&args.target),
        std::time::Duration::from_millis(
            std::env::var("JAIN_CLOUD_POLL_MS")
                .ok()
                .and_then(|v| v.parse().ok())
                .filter(|ms| (100..=10_000).contains(ms))
                .unwrap_or(2000),
        ),
        &cancel,
        &mut |update| match update {
            Update::Started { id, url } => {
                eprintln!("Cloud run started · {url}");
                started = Some((id, url));
            }
            Update::Thinking(text) => eprintln!("  Thinking {text}"),
            Update::Finished(lines) => {
                for line in &lines {
                    eprintln!("{line}");
                }
                finished = lines;
            }
            Update::Error(line) => {
                eprintln!("{line}");
                errors.push(line);
            }
        },
    )?;
    match outcome {
        Ok(()) => {
            let (id, url) = started.unwrap_or_default();
            let state = if finished.first().is_some_and(|l| l.starts_with('✓')) {
                "completed"
            } else if finished.is_empty() {
                "unknown"
            } else {
                "failed"
            };
            println!(
                "{}",
                serde_json::json!({"kind":"cloud_run","id":id,"url":url,"state":state,"lines":finished})
            );
            anyhow::ensure!(
                errors.is_empty() && state == "completed",
                "the cloud run did not complete"
            );
            Ok(Routed::Cloud)
        }
        Err(Refusal::Limit(limit)) => {
            eprintln!("{}", limit.render());
            match ask("") {
                Some(true) => Ok(Routed::Local),
                _ => anyhow::bail!("{} Upgrade: {}", limit.sentence(), limit.upgrade_url),
            }
        }
        Err(Refusal::Other(message)) => anyhow::bail!(message),
    }
}

/// After a local run completes: with results sync on and a signed-in account, send its model
/// and report. Never fails the run.
pub(crate) fn sync_after<T: jope_client::Transport>(transport: &T, run: &jope_api::ids::RunId) {
    let home = jope_paths::jopedime_home();
    if crate::terminal::onboarding::load(&home).results_sync != Some(true) {
        return;
    }
    let Ok(c) = crate::cloud::connection() else {
        return;
    };
    if !c.may_publish {
        return;
    }
    match crate::cloud::sync_results(
        transport,
        run,
        Some(&(c.origin.clone(), c.account_id.clone())),
    ) {
        Ok(lines) => lines.iter().for_each(|l| eprintln!("{l}")),
        Err(error) => eprintln!("Results sync: {error:#}"),
    }
}
