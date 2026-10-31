//! A warm invocation of the existing authenticated origin proof. No alternate
//! encoder, proposal sampler, fit scheduler, reduced candidate bank or clock.
use super::{
    campaign::{self, CampaignPlan},
    demo::{self, DemoCli},
    optimize::{Configuration, GENERIC_METHOD, NEURAL_METHOD},
    score::{self, Predictions},
    synthesized::{FittedSynthesizedArtifact, SynthesizedWorkerOutput},
};
use anyhow::{ensure, Context, Result};
use jope_datasets::catalog::Task;
use jope_pipeline::{
    execution::{ArmBudget, ExecutionPool, WorkerOutcome},
    experiment::{EvaluationStage, EvaluationStatus, Resources},
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{fs, path::Path, time::Instant};

fn warm_limits(resources: &Resources) -> Result<()> {
    resources.validate()?;
    ensure!(
        resources.reporting_seconds >= 5
            && resources.per_fit_seconds <= 30
            && resources.workers <= 20
            && resources.cpus <= 20
            && resources.memory_bytes <= 64u64 << 30
            && resources.swap_bytes == 0
            && resources.gpu_tasks == 0
            && resources.threads_per_worker == 1,
        "warm ASM plan requires a declared bounded total, >=5s reporting, <=30s per fit, <=20 CPU/workers, <=64GiB, no swap/GPU"
    );
    Ok(())
}
// Warm plans preregister their own equal allowances. This does not shorten the
// panel/proof default or reinterpret a partially available arm as matched.
fn latency_target_met(execution_ok: bool, elapsed_seconds: f64) -> bool {
    execution_ok && elapsed_seconds.is_finite() && (0. ..60.).contains(&elapsed_seconds)
}

fn warm_pair_limits(resources: &Resources, seconds_per_arm: u64) -> Result<()> {
    warm_limits(resources)?;
    ensure!(
        (1..=120).contains(&seconds_per_arm)
            && seconds_per_arm.checked_mul(2).is_some_and(|seconds| {
                seconds <= resources.total_seconds - resources.reporting_seconds
            }),
        "warm ASM requires two equal declared arm budgets within its original fit window"
    );
    Ok(())
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ArmClock {
    id_sha256: String,
    seconds: u64,
    started_unix_ms: u64,
    deadline_unix_ms: u64,
}
#[derive(Serialize)]
struct AvailableArmBudget {
    role: &'static str,
    id_sha256: String,
    declared_seconds: u64,
    started_unix_ms: u64,
    declared_deadline_unix_ms: u64,
    effective_deadline_unix_ms: u64,
    available_milliseconds: u64,
    truncated: bool,
}
fn observed_arm_budget(
    role: &'static str,
    arm: &ArmBudget,
    clock: &ArmClock,
    effective_deadline: u64,
) -> Result<AvailableArmBudget> {
    let allowance = arm
        .seconds
        .checked_mul(1000)
        .context("arm allowance overflow")?;
    ensure!(
        clock.id_sha256 == arm.id_sha256
            && clock.seconds == arm.seconds
            && clock.started_unix_ms.checked_add(allowance) == Some(clock.deadline_unix_ms)
            && effective_deadline <= clock.deadline_unix_ms,
        "warm arm clock differs from its original declared budget"
    );
    let available_milliseconds = effective_deadline.saturating_sub(clock.started_unix_ms);
    Ok(AvailableArmBudget {
        role,
        id_sha256: arm.id_sha256.clone(),
        declared_seconds: arm.seconds,
        started_unix_ms: clock.started_unix_ms,
        declared_deadline_unix_ms: clock.deadline_unix_ms,
        effective_deadline_unix_ms: effective_deadline,
        available_milliseconds,
        truncated: available_milliseconds != allowance,
    })
}
fn matched_full_allowances(arms: &[AvailableArmBudget]) -> bool {
    arms.len() == 2
        && arms[0].declared_seconds == arms[1].declared_seconds
        && arms.iter().all(|arm| !arm.truncated)
}
fn pair_availability(plan: &CampaignPlan, journal: &Path, proof: &Value) -> Result<Value> {
    let id = proof["dataset"].as_str().context("proof dataset absent")?;
    let search: super::optimize::DatasetSearch =
        serde_json::from_value(proof["outcomes"][id]["search"].clone())
            .context("authenticated pair outcome unavailable")?;
    let config = Configuration::from_campaign(plan)?;
    ensure!(
        search.pair.seconds == config.seconds,
        "warm pair allowance changed"
    );
    warm_pair_limits(&plan.experiment.resources, config.seconds)?;
    let experiment_id = plan.experiment.identity()?;
    let read = |role, neural| -> Result<AvailableArmBudget> {
        let arm = search.pair.arm(neural)?;
        let clock: ArmClock = serde_json::from_slice(&demo::read_bounded(
            &journal.join("arms").join(format!("{}.json", arm.id_sha256)),
        )?)?;
        // This shared reader verifies the original experiment/window, start,
        // declared allowance and reporting cutoff. It never starts a new arm.
        let deadline = super::receipt_timing::arm_deadline(journal, &arm, &experiment_id)?;
        observed_arm_budget(role, &arm, &clock, deadline)
    };
    let mut arms = Vec::new();
    let mut observed = Vec::new();
    for (role, neural) in [("neural", true), ("generic", false)] {
        match read(role, neural) {
            Ok(arm) => {
                observed.push(serde_json::to_value(&arm)?);
                arms.push(arm);
            }
            Err(error) => observed.push(json!({"role":role,"available_milliseconds":null,
                "status":"unknown","reason":format!("{error:#}")})),
        }
    }
    Ok(json!({
        "schema":"jope.warm-paired-budgets/v1",
        "matched":matched_full_allowances(&arms),
        "execution_order":["neural","generic"],
        "declared_seconds_per_arm":config.seconds,
        "required":"both arms received their entire equal preregistered allowance",
        "arms":observed
    }))
}

/// The warm demo presents and exports the accepted system only. A native challenger
/// proof plan is a preregistered comparison arm; it has no presentation or export route.
pub(super) fn refuse_native_challenger(config: &Configuration) -> Result<()> {
    ensure!(
        config.native_challenger.is_none(),
        "a native challenger proof plan is a comparison arm, not a demo route"
    );
    Ok(())
}

fn read_plan(cli: &DemoCli) -> Result<CampaignPlan> {
    let path = cli.asm_plan.as_ref().context("ASM plan required")?;
    let expected = cli
        .asm_plan_sha256
        .as_ref()
        .context("ASM plan identity required")?;
    let plan: CampaignPlan = serde_json::from_slice(&demo::read_bounded(path)?)?;
    ensure!(plan.identity()? == *expected, "ASM plan identity changed");
    ensure!(
        plan.scope == super::origin_proof::SCOPE,
        "warm ASM requires the one-dataset origin-proof scope"
    );
    warm_limits(&plan.experiment.resources)?;
    if let Some(source) = &cli.source_sha {
        ensure!(
            source == &plan.experiment.source_sha,
            "source differs from authenticated ASM plan"
        );
    }
    let config = Configuration::from_campaign(&plan)?;
    refuse_native_challenger(&config)?;
    warm_pair_limits(&plan.experiment.resources, config.seconds)?;
    config.verify_worker_builds()?;
    for dependency in config.dependencies() {
        dependency.verify()?;
    }
    // Positive numerical admission remains in the shared context verifier; a
    // missing qualification cannot turn this route into a generic success.
    ensure!(
        config.qualification.is_some(),
        "qualified reference cases unavailable; no warm neural demo admitted"
    );
    let current = demo::program()?;
    let expected_program = &plan
        .runtimes
        .get("rust")
        .context("native runtime absent")?
        .program;
    ensure!(
        current.path == expected_program.path && current.sha256 == expected_program.sha256,
        "installed CLI differs from pinned native fit runtime"
    );
    Ok(plan)
}

fn fitted_results(
    plan: &CampaignPlan,
    cli: &DemoCli,
    output: &Path,
    proof: &Value,
) -> Result<Value> {
    let manifest = cli.manifest.as_ref().context("manifest required")?;
    let datasets = cli.datasets.as_ref().context("datasets required")?;
    let (panel, manifest_sha) = super::cli::read_panel(manifest)?;
    ensure!(
        manifest_sha == plan.experiment.panel_sha256
            && campaign::digest(&panel)? == campaign::digest(&plan.panel)?,
        "frozen manifest changed"
    );
    let id = proof["dataset"]
        .as_str()
        .context("completed proof dataset absent")?;
    let spec = panel
        .datasets
        .iter()
        .find(|d| d.id == id)
        .context("proof dataset absent")?;
    ensure!(
        spec.task == Task::Regression,
        "accepted warm neural route is regression-only"
    );
    let dataset = super::data::load_development(datasets, spec)?;
    let cut = dataset.cut(0)?;
    let journal = output.join("journal");
    // Re-authenticate row-bound context, proposal, original arms and all selected
    // OOF evidence through the same report recovery used by benchmark/optimize.
    let recovered = super::asm_campaign::recover(plan, std::slice::from_ref(&dataset), &journal)?;
    let methods = recovered
        .methods
        .get(id)
        .context("selected recipes unavailable")?;
    let mean = cut.train.targets.iter().sum::<f64>() / cut.train.targets.len() as f64;
    ensure!(mean.is_finite(), "nonfinite train-mean comparator");
    let baseline_predictions = Predictions::Values(vec![mean; cut.hold.labels.len()]);
    let baseline_score = score::score_predictions(
        spec.task,
        &spec.classes,
        &cut.hold.labels,
        &baseline_predictions,
    )?;
    super::worker::write_private(
        &output.join("baseline.json"),
        &json!({
            "state":{"kind":"train_mean","value":mean},"predictions":baseline_predictions,"score":baseline_score
        }),
    )?;
    let mut entries = Vec::new();
    for (role, selected) in [("asm", NEURAL_METHOD), ("generic", GENERIC_METHOD)] {
        let matches: Vec<_> = methods
            .iter()
            .filter(|method| method.id() == selected)
            .collect();
        ensure!(
            matches.len() == 1,
            "missing or duplicated selected {role} recipe"
        );
        let fit = campaign::prepare_fit_impl(
            plan,
            &dataset,
            &cut,
            matches[0],
            EvaluationStage::Screen,
            &journal,
            false,
        )?;
        let outcome_path = journal
            .join("finished")
            .join(format!("{}.json", fit.job.identity()?));
        let outcome: WorkerOutcome = serde_json::from_slice(&demo::read_bounded(&outcome_path)?)?;
        ensure!(
            outcome.command_status == EvaluationStatus::Completed,
            "{role} outer refit unavailable: {:?}",
            outcome.reason
        );
        let (predictions, artifact_sha256) = campaign::checked_predictions(&fit, &outcome)?;
        let worker: SynthesizedWorkerOutput =
            serde_json::from_slice(&demo::read_bounded(&fit.job.directory.join("output.json"))?)?;
        let bytes = worker.artifact.bytes()?;
        ensure!(
            super::methods::digest(&bytes) == artifact_sha256,
            "{role} artifact digest differs"
        );
        let fitted_path = output.join(format!("{role}-fitted.json"));
        demo::exact_private(&fitted_path, &bytes)?;
        let reloaded = FittedSynthesizedArtifact::from_bytes(
            &demo::read_bounded(&fitted_path)?,
            &artifact_sha256,
        )?;
        let replayed = reloaded.predict(&cut.hold.table)?;
        ensure!(
            serde_json::to_vec(&replayed)? == serde_json::to_vec(&predictions)?,
            "{role} fitted reload changed predictions"
        );
        let prediction_path = output.join(format!("{role}-predictions.json"));
        super::worker::write_private(&prediction_path, &replayed)?;
        let scored =
            score::score_predictions(spec.task, &spec.classes, &cut.hold.labels, &replayed)?;
        let mut exports = Vec::new();
        for target in &cli.export {
            let backend = target.backend();
            match super::inference::export_synthesized(&reloaded, backend)? {
                super::inference::ExportOutcome::Unsupported { reason, .. } => {
                    exports.push(json!({"backend":backend,"status":"unsupported","reason":reason}))
                }
                super::inference::ExportOutcome::Emitted { bundle } => {
                    let folder = output
                        .join("exports")
                        .join(role)
                        .join(format!("{backend:?}").to_ascii_lowercase());
                    fs::create_dir_all(&folder)?;
                    for (name, source) in &bundle.files {
                        demo::exact_private(&folder.join(name), source.as_bytes())?;
                    }
                    super::worker::write_private(&folder.join("manifest.json"), &bundle.manifest)?;
                    exports.push(json!({"backend":backend,"status":"emitted-unqualified","bundle_sha256":bundle.identity()?,"directory":folder}));
                }
            }
        }
        entries.push(json!({"role":role,"method":selected,"score":scored,"loss_improvement_over_train_mean":baseline_score.loss-scored.loss,
            "fitted":{"path":fitted_path,"sha256":artifact_sha256,"reload_verified":true},
            "predictions":{"path":prediction_path,"sha256":super::methods::digest(&demo::read_bounded(&prediction_path)?)},
            "original_worker_output":outcome.outputs,"original_worker_outcome":{"path":outcome_path,"sha256":super::methods::digest(&demo::read_bounded(&outcome_path)?)},
            "exports":exports}));
    }
    let asm_loss = entries[0]["score"]["loss"]
        .as_f64()
        .context("ASM score absent")?;
    let generic_loss = entries[1]["score"]["loss"]
        .as_f64()
        .context("generic score absent")?;
    Ok(
        json!({"dataset":id,"task":spec.task,"classes":spec.classes,"fitting_rows_sha256":cut.train.rows_sha256,
        "scoring_rows_sha256":cut.hold.rows_sha256,"manifest_sha256":manifest_sha,"baseline_score":baseline_score,"predictors":entries,
        "asm_loss_improvement_over_generic":generic_loss-asm_loss,"neural_consumed":true,"fitted_artifacts":2,
        "selection":"unchanged train-side OOF winners; development scores never select or replace recipes"}),
    )
}

fn supervised(cli: &DemoCli, plan: &CampaignPlan, output: &Path, unit: &str) -> Result<Value> {
    let started = Instant::now();
    let (kn, kg) = Configuration::from_campaign(plan)?
        .incumbents
        .as_ref()
        .map_or((0, 0), |i| (i.neural.len(), i.generic.len()));
    let mut pool = ExecutionPool::open(
        &plan.experiment,
        &output.join("journal"),
        jope_pipeline::supervisor::for_unit(unit)?,
    )?;
    let cancel = jope_pipeline::supervisor::cancellation()?;
    let proof = super::origin_proof::run(
        plan,
        cli.manifest.as_ref().context("manifest required")?,
        cli.datasets.as_ref().context("datasets required")?,
        &output.join("journal"),
        &mut pool,
        &cancel,
    )?;
    super::worker::write_private(&output.join("origin-proof.json"), &proof)?;
    let budgets = pair_availability(plan, &output.join("journal"), &proof)
        .unwrap_or_else(|error| json!({"matched":false,"reason":format!("{error:#}")}));
    let completed = proof["generation_zero_and_evolution_proved"] == true;
    let matched = budgets["matched"] == true;
    let (ok, result, reason) = if !matched {
        (false, None, Some("warm comparison unavailable: both original arms must receive their entire equal preregistered allowance".to_owned()))
    } else if completed {
        match fitted_results(plan, cli, output, &proof) {
            Ok(value) => (true, Some(value), None),
            Err(error) => (false, None, Some(format!("{error:#}"))),
        }
    } else {
        (false, None, Some("authenticated neural generation-zero and evolved fits were not both completed in the original warm window".to_owned()))
    };
    let report = json!({"schema":"jope.warm-asm-demo/v1","ok":ok,"reason":reason,"plan_sha256":plan.identity()?,
        "source_sha":plan.experiment.source_sha,"neural_consumed":proof["genuine_neural_origin_fit_proved"] == true,"matched_comparison":matched,"paired_budget":budgets,"result":result,"origin_proof":proof,
        "limits":plan.experiment.resources,"supervised_seconds":started.elapsed().as_secs_f64(),
        "slots":{"neural":64,"exploration":16 - kn,"generic":80 - kg,"incumbent":{"neural":kn,"generic":kg}},"sealed_inputs_opened":false,"complete_panel_demonstration":false});
    super::worker::write_private(&output.join("asm-demo-result.json"), &report)?;
    Ok(report)
}

pub fn run(cli: DemoCli) -> Result<Value> {
    let started = Instant::now();
    ensure!(cli.synthetic.is_none() && cli.dataset.is_none() && cli.method.is_none() && cli.plan.is_none(),
        "ASM mode does not accept a generic recipe, synthetic fixture, dataset override or private generic plan");
    let targets: Vec<_> = cli.export.iter().map(|t| t.backend()).collect();
    for (index, target) in targets.iter().enumerate() {
        ensure!(
            !targets[..index].contains(target),
            "duplicate export backend"
        );
    }
    let plan = read_plan(&cli)?;
    if let Some(unit) = &cli.supervisor_unit {
        let output = fs::canonicalize(&cli.output)?;
        ensure!(
            cli.asm_plan.as_ref() == Some(&output.join("asm-plan.json")),
            "unexpected supervised ASM plan path"
        );
        let result = supervised(&cli, &plan, &output, unit);
        if let Err(error) = &result {
            let _ = super::worker::write_private(
                &output.join("asm-demo-failure.json"),
                &json!({"ok":false,"error":format!("{error:#}")}),
            );
        }
        return result;
    }
    let output = demo::new_output(&cli.output)?;
    let plan_path = output.join("asm-plan.json");
    super::worker::write_private(&plan_path, &plan)?;
    let mut args = vec![
        "demo".into(),
        "--asm-plan".into(),
        plan_path.display().to_string(),
        "--asm-plan-sha256".into(),
        plan.identity()?,
        "--manifest".into(),
        fs::canonicalize(cli.manifest.as_ref().context("manifest required")?)?
            .display()
            .to_string(),
        "--datasets".into(),
        fs::canonicalize(cli.datasets.as_ref().context("datasets required")?)?
            .display()
            .to_string(),
        "--output".into(),
        output.display().to_string(),
    ];
    for target in &cli.export {
        use clap::ValueEnum;
        args.extend([
            "--export".into(),
            target
                .to_possible_value()
                .context("export name unavailable")?
                .get_name()
                .to_owned(),
        ]);
    }
    let program = demo::program()?;
    let launched = jope_pipeline::supervisor::launch(
        &plan.experiment,
        &output.join("journal"),
        &program,
        &args,
    )?;
    let result_path = output.join("asm-demo-result.json");
    let child = if result_path.is_file() {
        Some(serde_json::from_slice::<Value>(&demo::read_bounded(
            &result_path,
        )?)?)
    } else {
        None
    };
    let ok = launched.exit_code == Some(0) && child.as_ref().is_some_and(|r| r["ok"] == true);
    let elapsed = started.elapsed().as_secs_f64();
    let result_sha256 = if result_path.is_file() {
        Some(super::methods::digest(&demo::read_bounded(&result_path)?))
    } else {
        None
    };
    let report = json!({"schema":"jope.warm-asm-demo-invocation/v1","ok":ok,"plan_sha256":plan.identity()?,
        "plan_file_sha256":super::methods::digest(&demo::read_bounded(&plan_path)?),"program":program,"source_sha":plan.experiment.source_sha,
        "warm_elapsed_seconds":elapsed,"warm_target_seconds":60,"target_met":latency_target_met(ok,elapsed),"under_60_seconds":latency_target_met(ok,elapsed),"cold_install_included":false,
        "execution_complete":ok,"latency_target_is_admission_cap":false,
        "launch":launched,"result_file_sha256":result_sha256,"result":child});
    super::worker::write_private(&output.join("invocation-receipt.json"), &report)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn warm_wrapper_does_not_extend_global_deadline_or_reporting_budget() {
        let mut resources = Resources {
            workers: 20,
            cpus: 20,
            memory_bytes: 64u64 << 30,
            swap_bytes: 0,
            gpu_tasks: 0,
            threads_per_worker: 1,
            total_seconds: 60,
            reporting_seconds: 5,
            per_fit_seconds: 30,
        };
        warm_limits(&resources).unwrap();
        warm_pair_limits(&resources, 20).unwrap();
        assert!(warm_pair_limits(&resources, 28).is_err());
        assert!(warm_pair_limits(&resources, 120).is_err());
        assert!(warm_pair_limits(&resources, 0).is_err());
        resources.total_seconds = 300;
        warm_limits(&resources).unwrap();
        warm_pair_limits(&resources, 120).unwrap();
        resources.total_seconds = 240;
        assert!(warm_pair_limits(&resources, 120).is_err());
        resources.total_seconds = 7201;
        assert!(warm_limits(&resources).is_err());
        resources.total_seconds = 60;
        resources.reporting_seconds = 4;
        assert!(warm_limits(&resources).is_err());
        resources.reporting_seconds = 5;
        resources.gpu_tasks = 1;
        assert!(warm_limits(&resources).is_err());
    }

    #[test]
    fn warm_latency_target_is_measured_separately_from_completed_execution() {
        assert!(latency_target_met(true, 59.9));
        for elapsed in [60., 61., 230.13, f64::NAN, -1.] {
            assert!(!latency_target_met(true, elapsed));
        }
        assert!(!latency_target_met(false, 20.));
    }

    fn allowance(role: &'static str, start: u64, cutoff: u64) -> AvailableArmBudget {
        let arm = ArmBudget {
            id_sha256: "a".repeat(64),
            seconds: 20,
        };
        let clock = ArmClock {
            id_sha256: arm.id_sha256.clone(),
            seconds: arm.seconds,
            started_unix_ms: start,
            deadline_unix_ms: start + 20000,
        };
        observed_arm_budget(role, &arm, &clock, cutoff.min(clock.deadline_unix_ms)).unwrap()
    }
    #[test]
    fn warm_comparison_rejects_unequal_and_equally_clipped_allowances() {
        let complete = [
            allowance("neural", 1000, 56000),
            allowance("generic", 24000, 56000),
        ];
        assert!(matched_full_allowances(&complete));
        assert!(!matched_full_allowances(&[]));
        assert!(!matched_full_allowances(&complete[..1]));
        let mut unequal = [
            allowance("neural", 1000, 56000),
            allowance("generic", 24000, 56000),
        ];
        unequal[1].declared_seconds = 21;
        assert!(!matched_full_allowances(&unequal));
        assert_eq!(complete[0].available_milliseconds, 20000);
        assert_eq!(complete[1].available_milliseconds, 20000);
        let clipped = [
            allowance("neural", 1000, 56000),
            allowance("generic", 40000, 56000),
        ];
        assert!(!matched_full_allowances(&clipped));
        assert_eq!(clipped[1].available_milliseconds, 16000);
        assert!(clipped[1].truncated);
        let both_clipped = [
            allowance("neural", 40000, 56000),
            allowance("generic", 40000, 56000),
        ];
        assert!(!matched_full_allowances(&both_clipped));
        assert!(!matched_full_allowances(&[
            allowance("neural", 1000, 56000),
            allowance("generic", 57000, 56000)
        ]));
    }
    #[test]
    fn warm_comparison_rejects_changed_original_arm_identity_or_clock() {
        let arm = ArmBudget {
            id_sha256: "a".repeat(64),
            seconds: 20,
        };
        let mut clock = ArmClock {
            id_sha256: arm.id_sha256.clone(),
            seconds: 20,
            started_unix_ms: 1000,
            deadline_unix_ms: 21000,
        };
        assert!(observed_arm_budget("neural", &arm, &clock, 21001).is_err());
        clock.deadline_unix_ms += 1;
        assert!(observed_arm_budget("neural", &arm, &clock, 21000).is_err());
        clock.deadline_unix_ms = 21000;
        clock.id_sha256 = "b".repeat(64);
        assert!(observed_arm_budget("neural", &arm, &clock, 21000).is_err());
    }
}
