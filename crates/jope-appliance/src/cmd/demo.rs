//! `jain demo list` and `jain demo run NAME`: the TUI's bundled demos without a terminal, for
//! agents and scripts (#2142). A demo runs exactly as `/demo NAME` does in the TUI: the same
//! bundled CSV, target, task and objective presets and the same draft defaults, on the shared
//! background engine.
use crate::terminal::demos::{self, Demo};
use clap::Subcommand;
use jope_api::{RunSnapshot, RunState};
use jope_client::{RunResult, Transport};
use serde_json::{json, Value};

#[derive(Subcommand, Debug)]
pub enum DemoCommand {
    /// List the bundled demos (no account needed).
    List {
        /// One JSON document with every demo.
        #[arg(long)]
        json: bool,
    },
    /// Run one bundled demo to completion and print the result.
    Run {
        /// A demo name from `jain demo list` (or a unique prefix, or a task such as `regression`).
        name: String,
        /// Print one JSON document (`demo` and the final `run` snapshot) instead of text.
        #[arg(long)]
        json: bool,
        /// Start the run and print its identifier without waiting for it to finish.
        #[arg(long)]
        detach: bool,
    },
}

pub fn run(command: DemoCommand) -> anyhow::Result<()> {
    match command {
        DemoCommand::List { json } => {
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&json!({
                        "demos": demos::DEMOS.iter().map(describe).collect::<Vec<_>>()
                    }))?
                );
            } else {
                for line in list_lines() {
                    println!("{line}");
                }
            }
            Ok(())
        }
        DemoCommand::Run { name, json, detach } => {
            let Some(demo) = demos::find(&name) else {
                let names: Vec<&str> = demos::DEMOS.iter().map(|demo| demo.name).collect();
                anyhow::bail!(
                    "there is no bundled demo named {name}; choose one of: {} (jain demo list)",
                    names.join(", ")
                );
            };
            let client = super::client()?;
            run_demo(&client, demo, json, detach)
        }
    }
}

/// The demo listing for a command line: the TUI's lines with `jain` commands.
fn list_lines() -> Vec<String> {
    let mut lines =
        vec!["Bundled demos (no account needed). Run one with: jain demo run NAME".to_owned()];
    for demo in demos::DEMOS {
        lines.push(format!(
            "  {:<26} {:<10} {:>5} rows · {}",
            demo.name,
            demo.task,
            demo.rows(),
            demo.title
        ));
    }
    lines.push(
        "Or use the file directly: jain run example/<name>.csv --target COLUMN (also jain fit --train)."
            .into(),
    );
    lines.push("Each demo's data licence and credit are shown when you run it.".into());
    lines
}

fn describe(demo: &Demo) -> Value {
    json!({
        "name": demo.name,
        "title": demo.title,
        "task": demo.task,
        "target": demo.target,
        "objective": demo.objective,
        "rows": demo.rows(),
        "path": demo.path(),
        "license": demo.license,
        "license_url": demo.license_url,
        "credit": demo.credit,
    })
}

/// The draft the TUI's `/demo NAME` builds: presets first, then the registered dataset.
fn demo_draft(
    demo: &Demo,
    profile: jope_api::DatasetProfile,
) -> anyhow::Result<crate::draft::Draft> {
    let mut draft = crate::draft::Draft::default();
    let presets = std::iter::once(format!("\\target {} --task {}", demo.target, demo.task)).chain(
        demo.objective
            .map(|objective| format!("\\objective {objective}")),
    );
    for preset in presets {
        draft.apply(&preset).map_err(anyhow::Error::msg)?;
    }
    draft.set_dataset(profile);
    Ok(draft)
}

fn run_demo<T: Transport>(
    client: &T,
    demo: &'static Demo,
    json_out: bool,
    detach: bool,
) -> anyhow::Result<()> {
    let say = |line: &str| {
        if !json_out {
            println!("{line}");
        }
    };
    say(&format!(
        "Demo {} · {} · {} rows · predicts column `{}` ({})",
        demo.name,
        demo.title,
        demo.rows(),
        demo.target,
        demo.task
    ));
    for line in demo.license_notice() {
        say(&line);
    }
    let file = format!("{}.csv", demo.name);
    let key = format!("demo_{}", ulid::Ulid::new());
    let profile: jope_api::DatasetProfile =
        jope_client::files::upload_csv(client, &file, std::io::Cursor::new(demo.bytes), &key)?
            .json()?;
    let draft = demo_draft(demo, profile.clone())?;
    let mut spec = draft
        .run_spec(true)
        .ok_or_else(|| anyhow::anyhow!("the demo draft is incomplete"))?;
    let plan = match client.run_start(&spec, Some(&format!("demo_plan_{}", ulid::Ulid::new())))? {
        RunResult::Plan(plan) => plan,
        RunResult::Started(_) => anyhow::bail!("dry-run unexpectedly started a run"),
    };
    spec = draft.run_spec(false).expect("validated draft is complete");
    spec.plan_digest = Some(plan.plan_digest);
    let started = client
        .run_start(&spec, Some(&format!("demo_run_{}", ulid::Ulid::new())))
        .map_err(|error| {
            if crate::terminal::runs::is_busy(&error.to_string()) {
                let run = crate::terminal::runs::active_run(client);
                anyhow::anyhow!(crate::terminal::runs::busy_message(run.as_ref(), false))
            } else {
                anyhow::Error::new(error)
            }
        })?;
    let snapshot = match started {
        RunResult::Started(snapshot) => *snapshot,
        RunResult::Plan(_) => anyhow::bail!("run request returned a dry-run plan"),
    };
    crate::cloud::remember_run(
        &snapshot.run_id,
        &crate::cloud_events::RunContext::from_spec(&spec, &file, Some(profile)),
    );
    let run = snapshot.run_id.clone();
    if detach {
        if json_out {
            println!(
                "{}",
                serde_json::to_string(&json!({"demo": describe(demo), "run": snapshot}))?
            );
        } else {
            println!("Run {run} started in the background engine.");
            println!("Follow it: jain status {run} --json · jain events {run} --follow");
        }
        return Ok(());
    }
    say(&format!(
        "Run {run} started ({}, up to {} s); waiting for it to finish…",
        crate::terminal::status::effort_name(snapshot.effort),
        snapshot.budget_total.wall_seconds.0
    ));
    let current = wait_terminal(client, snapshot)?;
    if current.state == RunState::Completed {
        super::run::sync_after(client, &current.run_id);
    }
    if json_out {
        println!(
            "{}",
            serde_json::to_string(&json!({"demo": describe(demo), "run": current}))?
        );
    } else {
        for line in result_lines(&current) {
            println!("{line}");
        }
    }
    anyhow::ensure!(
        current.state == RunState::Completed,
        "the demo run {} ended {}",
        current.run_id,
        crate::terminal::status::state_name(current.state)
    );
    Ok(())
}

pub(crate) fn wait_terminal<T: Transport>(
    client: &T,
    mut current: RunSnapshot,
) -> anyhow::Result<RunSnapshot> {
    while !matches!(
        current.state,
        RunState::Completed | RunState::Cancelled | RunState::Failed
    ) {
        current = super::run::wait_status(client, &current, 30)?;
    }
    Ok(current)
}

/// What a finished demo prints: the outcome, how the search stopped, and what to do next.
pub(crate) fn result_lines(snapshot: &RunSnapshot) -> Vec<String> {
    let mut lines = vec![crate::terminal::status::finished_line(snapshot)];
    lines.extend(crate::terminal::status::budget_stop_line(snapshot));
    if snapshot.state == RunState::Completed {
        lines.push(crate::terminal::status::next_steps(
            &snapshot.run_id.to_string(),
            crate::cloud::connection().is_ok(),
            false,
        ));
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn listing_names_every_demo_and_the_cli_commands() {
        let lines = list_lines();
        for demo in demos::DEMOS {
            assert!(lines.iter().any(|line| line.contains(demo.name)));
        }
        assert!(lines[0].contains("jain demo run NAME"));
        assert!(!lines.iter().any(|line| line.contains("/demo")));
        let value = describe(&demos::DEMOS[0]);
        assert_eq!(value["path"], "example/regression_parkinsons.csv");
        assert_eq!(value["target"], "target");
    }

    #[test]
    fn the_demo_draft_matches_the_tui_presets() {
        let demo = demos::find("regression_parkinsons").unwrap();
        let profile: jope_api::DatasetProfile = serde_json::from_value(json!({
            "dataset_id": "ds_21fbfe8b53dbcd605d79486b",
            "name": "regression_parkinsons.csv",
            "bytes": "10", "rows": "30",
            "columns": [
                {"name": "a", "data_type": "numeric", "missing": "0", "distinct": "30"},
                {"name": "target", "data_type": "numeric", "missing": "0", "distinct": "30"}
            ],
            "target_suggestions": [], "warnings": []
        }))
        .unwrap();
        let spec = demo_draft(demo, profile).unwrap().run_spec(false).unwrap();
        assert_eq!(spec.target, "target");
        assert_eq!(spec.task, jope_api::Task::Regression);
        assert_eq!(spec.effort, jope_api::Effort::Fast);
        assert_eq!(spec.comparators, jope_api::ComparatorPolicy::Off);
        assert_eq!(
            spec.languages,
            [jope_api::Language::Rust, jope_api::Language::Python]
        );
    }
}
