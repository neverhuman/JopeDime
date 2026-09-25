//! Shared helpers for benchmark plan/run/report front doors.
use super::campaign::{CampaignPlan, Method, Runtime, PLAN_SCHEMA};
use anyhow::{ensure, Context, Result};
use clap::Args;
use jope_datasets::catalog::Task;
use jope_pipeline::{
    execution::{Artifact, ExecutionPool},
    experiment::{Experiment, Resources, EXPERIMENT_SCHEMA},
};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

#[derive(Args)]
pub struct RunArgs {
    #[arg(long)]
    pub plan: PathBuf,
    #[arg(long)]
    pub plan_sha256: String,
    #[arg(long)]
    pub manifest: PathBuf,
    #[arg(long)]
    pub datasets: PathBuf,
    #[arg(long)]
    pub journal: PathBuf,
    /// Run and retain every fit, but leave development hold scoring to `report`.
    /// Applies only to a single-dataset origin proof plan.
    #[arg(long)]
    pub defer_report: bool,
    #[arg(long, hide = true)]
    pub supervisor_unit: Option<String>,
}

#[derive(Args)]
pub struct ReportArgs {
    #[arg(long)]
    pub plan: PathBuf,
    #[arg(long)]
    pub plan_sha256: String,
    #[arg(long)]
    pub manifest: PathBuf,
    #[arg(long)]
    pub datasets: PathBuf,
    #[arg(long)]
    pub journal: PathBuf,
}

pub fn prepare(
    manifest: &Path,
    datasets: &Path,
    source_sha: String,
    runtimes: &Path,
    output: &Path,
) -> Result<Value> {
    let (panel, panel_sha256) = super::cli::read_panel(manifest)?;
    let verification = super::cli::verify_inputs(&panel, datasets);
    ensure!(
        verification["ok"] == true,
        "development input verification failed: {verification}"
    );
    let runtimes: BTreeMap<String, Runtime> = serde_json::from_slice(&fs::read(runtimes)?)?;
    let mut methods = BTreeMap::new();
    for task in [Task::Regression, Task::Binary, Task::Multiclass] {
        let mut list: Vec<_> = super::methods::rust_methods(task)?
            .into_iter()
            .map(|recipe| Method::Rust { recipe })
            .collect();
        for engine in ["xgboost", "lightgbm", "catboost", "sklearn"] {
            list.push(Method::External {
                id: format!("external.{engine}.default"),
                engine: engine.into(),
                runtime: engine.into(),
            });
        }
        methods.insert(task.as_str().to_owned(), list);
    }
    let experiment = Experiment {
        schema: EXPERIMENT_SCHEMA.into(),
        panel_sha256,
        methods_sha256: jope_pipeline::sha256_hex(&serde_json::to_vec(&(&methods, &runtimes))?),
        source_sha,
        checkpoints: BTreeMap::new(),
        seed: panel.seed,
        resources: Resources {
            workers: 20,
            cpus: 20,
            memory_bytes: 64 << 30,
            swap_bytes: 0,
            gpu_tasks: 1,
            threads_per_worker: 1,
            total_seconds: 7200,
            reporting_seconds: 300,
            per_fit_seconds: 30,
        },
        expected_artifacts: vec![
            "finished".into(),
            "observations".into(),
            "frozen".into(),
            "reports".into(),
        ],
        protocol: super::PROTOCOL.into(),
        numerical_policy: super::score::NUMERICAL_POLICY.into(),
    };
    let plan = CampaignPlan {
        schema: PLAN_SCHEMA.into(),
        experiment,
        panel,
        methods,
        runtimes,
        scope: "comparator-qualification".into(),
    };
    let id = plan.identity()?;
    super::worker::write_private(output, &plan)?;
    Ok(
        json!({"ok":true,"plan":output,"plan_sha256":id,"scope":plan.scope,
        "execution_budget_started":false,"fits":0,
        "note":"Comparator qualification only; this is not the complete ASM panel demonstration."}),
    )
}

fn read_plan(path: &Path, expected: &str) -> Result<CampaignPlan> {
    ensure!(
        fs::metadata(path)?.len() <= 32 * 1024 * 1024,
        "plan exceeds 32 MiB"
    );
    let plan: CampaignPlan = serde_json::from_slice(&fs::read(path)?)?;
    ensure!(plan.identity()? == expected, "plan checksum changed");
    Ok(plan)
}

fn journal(path: &Path) -> Result<PathBuf> {
    ensure!(
        path.is_absolute() && !path.starts_with("/tmp"),
        "journal requires durable absolute path"
    );
    fs::create_dir_all(path)?;
    let path = fs::canonicalize(path)?;
    ensure!(
        !path.starts_with("/tmp") && !path.ancestors().any(|p| p.join(".git").exists()),
        "private campaign data must remain outside Git working trees and /tmp"
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&path, fs::Permissions::from_mode(0o700))?;
    }
    Ok(path)
}

pub fn run(args: RunArgs) -> Result<Value> {
    let plan = read_plan(&args.plan, &args.plan_sha256)?;
    let defer = args.defer_report;
    // Literal scope: this module also compiles without the asm feature.
    ensure!(
        !defer || plan.scope == "asm-origin-proof",
        "--defer-report applies only to a single-dataset origin proof plan"
    );
    let journal = journal(&args.journal)?;
    bind_deferral(&journal, &args.plan_sha256, defer)?;
    let manifest = fs::canonicalize(&args.manifest)?;
    let datasets = fs::canonicalize(&args.datasets)?;
    if let Some(unit) = args.supervisor_unit {
        let mut pool = ExecutionPool::open(
            &plan.experiment,
            &journal,
            jope_pipeline::supervisor::for_unit(&unit)?,
        )?;
        let cancellation = jope_pipeline::supervisor::cancellation()?;
        return super::campaign::run_with(
            &plan,
            &manifest,
            &datasets,
            &journal,
            &mut pool,
            &cancellation,
            defer,
        );
    }
    let program_path = fs::canonicalize(std::env::current_exe()?)?;
    let program = Artifact {
        sha256: jope_pipeline::sha256_hex(&fs::read(&program_path)?),
        path: program_path,
    };
    // An installed binary must be the exact worker binary pinned by the plan.
    let expected = &plan
        .runtimes
        .get("rust")
        .context("native runtime missing")?
        .program;
    ensure!(
        program.path == expected.path && program.sha256 == expected.sha256,
        "running binary differs from pinned native runtime"
    );
    let mut arguments = plan.runtimes["rust"].argument_prefix.clone();
    arguments.extend([
        "run".into(),
        "--plan".into(),
        fs::canonicalize(&args.plan)?.display().to_string(),
        "--plan-sha256".into(),
        args.plan_sha256,
        "--manifest".into(),
        manifest.display().to_string(),
        "--datasets".into(),
        datasets.display().to_string(),
        "--journal".into(),
        journal.display().to_string(),
    ]);
    if defer {
        arguments.push("--defer-report".into());
    }
    let launched =
        jope_pipeline::supervisor::launch(&plan.experiment, &journal, &program, &arguments)?;
    if defer {
        return Ok(deferred_run(
            serde_json::to_value(&launched)?,
            launched.exit_code,
        ));
    }
    let report = if journal.join("campaign.json").try_exists()? {
        Some(super::campaign::report(
            &plan, &manifest, &datasets, &journal,
        )?)
    } else {
        None
    };
    Ok(json!({"ok":launched.exit_code==Some(0),"launch":launched,"report":report}))
}

/// Parent output for a deferred run: launch evidence only; `report` is the reveal.
fn deferred_run(launch: Value, exit_code: Option<i32>) -> Value {
    let complete = exit_code == Some(0);
    json!({"ok":complete,"state":if complete {"complete_report_deferred"} else {"incomplete_report_deferred"},"launch":launch})
}

pub(crate) const DEFERRAL_MARKER: &str = "report-deferred.json";

/// Deferral belongs to the journal, not to one invocation. Without this, a
/// resume that omits the flag would score the development hold rows.
fn bind_deferral(journal: &Path, plan_sha256: &str, defer: bool) -> Result<()> {
    let marker = journal.join(DEFERRAL_MARKER);
    if defer {
        ensure!(
            marker.try_exists()? || !journal.join("campaign.json").try_exists()?,
            "journal was started without --defer-report; its hold rows may already be scored"
        );
        super::campaign::immutable(
            &marker,
            &json!({"schema":"jope.report-deferred/v1","plan_sha256":plan_sha256}),
        )
    } else {
        ensure!(
            !marker.try_exists()?,
            "journal was started with --defer-report; resume with the flag or run optimize report"
        );
        Ok(())
    }
}

pub fn report(args: ReportArgs) -> Result<Value> {
    let plan = read_plan(&args.plan, &args.plan_sha256)?;
    super::campaign::report(
        &plan,
        &args.manifest,
        &args.datasets,
        &fs::canonicalize(args.journal)?,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    struct Wrapper {
        #[command(flatten)]
        run: RunArgs,
    }

    #[test]
    fn defer_report_flag_and_parent_output() {
        let required = [
            "wrapper",
            "--plan",
            "/private/plan.json",
            "--plan-sha256",
            "synthetic",
            "--manifest",
            "/private/panel.json",
            "--datasets",
            "/private/prepared",
            "--journal",
            "/private/journal",
        ];
        assert!(!Wrapper::try_parse_from(required).unwrap().run.defer_report);
        let deferred = required.iter().copied().chain(["--defer-report"]);
        assert!(Wrapper::try_parse_from(deferred).unwrap().run.defer_report);
        let complete = deferred_run(json!({}), Some(0));
        assert_eq!(complete["ok"], true);
        assert_eq!(complete["state"], "complete_report_deferred");
        assert!(complete.get("report").is_none());
        let incomplete = deferred_run(json!({}), None);
        assert_eq!(incomplete["ok"], false);
        assert_eq!(incomplete["state"], "incomplete_report_deferred");
        assert!(incomplete.get("report").is_none());
    }

    fn scratch() -> Result<tempfile::TempDir> {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/tmp");
        fs::create_dir_all(&root)?;
        Ok(tempfile::Builder::new()
            .prefix("campaign-cli-")
            .tempdir_in(fs::canonicalize(root)?)?)
    }

    #[test]
    fn deferral_is_bound_to_the_journal() -> Result<()> {
        let plan = "a".repeat(64);
        let deferred = scratch()?;
        bind_deferral(deferred.path(), &plan, true)?;
        assert!(deferred.path().join("report-deferred.json").try_exists()?);
        bind_deferral(deferred.path(), &plan, true)?;
        // A resume that omits the flag is refused.
        let error = bind_deferral(deferred.path(), &plan, false).unwrap_err();
        assert!(format!("{error:#}").contains("started with --defer-report"));
        // Another plan cannot reuse the deferred journal.
        let error = bind_deferral(deferred.path(), &"b".repeat(64), true).unwrap_err();
        assert!(format!("{error:#}").contains("stale evidence"));
        // A journal started without deferral cannot become deferred.
        let started = scratch()?;
        fs::write(started.path().join("campaign.json"), b"{}")?;
        let error = bind_deferral(started.path(), &plan, true).unwrap_err();
        assert!(format!("{error:#}").contains("started without --defer-report"));
        assert!(!started.path().join("report-deferred.json").try_exists()?);
        bind_deferral(started.path(), &plan, false)?;
        Ok(())
    }
}
