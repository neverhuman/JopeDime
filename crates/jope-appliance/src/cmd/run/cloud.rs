//! `jain run` routing (owner rule 2026-11-06). A bundled `example/NAME.csv` runs on this
//! machine; the user's own CSV trains in the veox Cloud. Before anything is registered or
//! uploaded, own data needs a login on an account admitted from the waitlist ([`preflight`]);
//! then the per-table upload consent (remembered, asked on a terminal, or given with
//! `--upload-consent` by an agent that asked the user), then progress on stderr and a
//! `cloud_run` JSON line on stdout.
use crate::cloud::{
    cloud_run::{CloudRefusal, Update},
    flow::{self, Prepared},
    route::{self, AccountTier, Force},
    Connection,
};
use std::io::{BufRead, IsTerminal, Write};
use std::sync::atomic::AtomicBool;

/// Where `jain run` sends a CSV, decided before it is registered.
pub(super) enum Routed {
    Local,
    /// Own data for the cloud, from a signed-in, admitted account.
    Cloud {
        decision: route::Decision,
        connection: Box<Connection>,
        tier: Option<AccountTier>,
    },
}

fn force(args: &super::RunArgs) -> Option<Force> {
    match (args.cloud, args.local) {
        (true, _) => Some(Force::Cloud),
        (_, true) => Some(Force::Local),
        _ => None,
    }
}

/// The routing decision and the own-data gate, before the CSV is registered or read.
pub(super) fn preflight(
    args: &super::RunArgs,
    bundled: bool,
    effort: jope_api::Effort,
) -> anyhow::Result<Routed> {
    let decision = route::decide(route::Input {
        effort,
        rows: None,
        demo: bundled,
        force: force(args),
        cloud_first: route::cloud_first(),
    });
    match decision.target {
        route::Target::Refused => Err(jope_runtime::Failure::input(decision.line)),
        route::Target::Local => Ok(Routed::Local),
        route::Target::Cloud => {
            // --cloud already conflicts with these; for implied cloud runs, say why.
            if args.detach || args.out.is_some() || args.events.is_some() || args.dry_run {
                return Err(jope_runtime::Failure::input(
                    "your own data trains in the veox Cloud, where --dry-run, --detach, --out and --events do not apply yet: leave them out and jain follows the run to its result (and its workspace link)",
                ));
            }
            let (connection, tier) = flow::admitted()
                .map_err(|gate| jope_runtime::Failure::input(gate.message(false)))?;
            Ok(Routed::Cloud {
                decision,
                connection: Box::new(connection),
                tier,
            })
        }
    }
}

/// The upload question on a terminal. `None` without one.
fn ask(question: &str) -> Option<bool> {
    if !std::io::stdin().is_terminal() || !std::io::stderr().is_terminal() {
        return None;
    }
    eprint!("{question} ");
    let _ = std::io::stderr().flush();
    let mut line = String::new();
    std::io::stdin().lock().read_line(&mut line).ok()?;
    Some(route::consent_answer(&line).unwrap_or(false))
}

/// The cloud run of a registered own-data CSV: the consent, the upload, and following the run.
pub(super) fn run_in_cloud(
    args: &super::RunArgs,
    path: &std::path::Path,
    profile: &jope_api::DatasetProfile,
    effort: jope_api::Effort,
    decision: &route::Decision,
    c: &Connection,
    tier: Option<AccountTier>,
) -> anyhow::Result<()> {
    let rows = profile.rows.0;
    let line = route::decide(route::Input {
        effort,
        rows: Some(rows),
        demo: false,
        force: force(args),
        cloud_first: route::cloud_first(),
    })
    .line;
    eprintln!("{line}");
    // Free's row ceiling is known: say so before uploading anything.
    if rows > route::FREE_MAX_DATA_ROWS && tier == Some(AccountTier::Free) {
        let limit = route::Limit {
            status: 400,
            limit: "size".into(),
            used: Some(rows),
            allowed: Some(route::FREE_MAX_DATA_ROWS),
            resets_at: None,
            upgrade_url: route::UPGRADE_URL.into(),
            message: String::new(),
        };
        anyhow::bail!(limit.render());
    }
    let (search, note) = route::profile_for(effort, tier);
    if let Some(note) = note {
        eprintln!("{note}");
    }
    let prepared = Prepared::read(path, rows, profile.columns.len() as u64)?;
    let home = jope_paths::jopedime_home();
    let key = flow::consent_key(&c.origin, &c.account_id, &prepared.sha256);
    if !flow::consented(&crate::terminal::onboarding::load(&home), &key, &prepared) {
        let agreed = if args.upload_consent {
            eprintln!("{} yes (--upload-consent)", prepared.question());
            true
        } else {
            match ask(&prepared.question()) {
                Some(answer) => answer,
                None => {
                    return Err(jope_runtime::Failure::input(format!(
                        "uploading {} to train in the veox Cloud needs the user's consent: ask them first, then pass --upload-consent (nothing was uploaded)",
                        prepared.file_name
                    )))
                }
            }
        };
        if !agreed {
            anyhow::bail!("Nothing was uploaded, and no run started.");
        }
        flow::remember_consent(&home, &key, &prepared)?;
    }
    eprintln!(
        "Uploading {} ({}) to your veox Cloud Workspace…",
        prepared.file_name,
        route::size(prepared.bytes.len() as u64)
    );
    let cancel = AtomicBool::new(false);
    let mut started = None;
    let mut finished = Vec::new();
    let mut errors = Vec::new();
    let outcome = flow::submit_and_follow(
        c,
        &prepared,
        search,
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
            if state == "completed" {
                eprintln!("{}", crate::cloud::cloud_run::next_line(&url, false));
            }
            println!(
                "{}",
                serde_json::json!({"kind":"cloud_run","id":id,"url":url,"state":state,"lines":finished})
            );
            anyhow::ensure!(
                errors.is_empty() && state == "completed",
                "the cloud run did not complete"
            );
            Ok(())
        }
        Err(CloudRefusal::Limit(limit)) => anyhow::bail!(limit.render()),
        Err(CloudRefusal::Other(message)) => anyhow::bail!("{message} No run started."),
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
