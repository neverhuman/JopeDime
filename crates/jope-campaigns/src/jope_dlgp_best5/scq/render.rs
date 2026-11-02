use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::process::Command;

use super::*;

fn toml_quote(value: &str) -> String {
    serde_json::to_string(value).unwrap()
}

/// Renders the common immutable SCQ envelope. Command-specific immutable
/// inputs and outputs are supplied by the caller; this function owns the
/// frozen resource, preemption, timeout, and retry policy.
pub fn render_scq_spec(job: &ScqJobV1, args: &[String], workdir: &str) -> String {
    render_scq_spec_with_io(job, args, workdir, &[], &[])
}

pub fn render_scq_spec_with_io(
    job: &ScqJobV1,
    args: &[String],
    workdir: &str,
    inputs: &[ScqInputV1],
    outputs: &[ScqOutputV1],
) -> String {
    let policy = super::ScqPolicyV1::frozen();
    let args = args
        .iter()
        .map(|value| toml_quote(value))
        .collect::<Vec<_>>()
        .join(", ");
    let mut rendered = format!(
        "[job]\nid = {}\nproject = {}\ncommand = \"/usr/bin/env\"\nargs = [{}]\nworkdir = {}\ntier = \"normal\"\n\n[resource]\ngpus_per_task = {}\ncpus_reserved = 4\nomp_threads = 1\nram_mb = 8192\ngang_size = 1\nplacement = \"single_node\"\n\n[env]\nRAYON_NUM_THREADS = \"4\"\nOMP_NUM_THREADS = \"1\"\nOPENBLAS_NUM_THREADS = \"1\"\nMKL_NUM_THREADS = \"1\"\nJOPE_DLGP_CANDIDATE_TIMEOUT_SECONDS = \"30\"\nJOPE_EXPECTED_PROJECT_PRIORITY_WEIGHT = \"{}\"\nJOPE_MIN_WORKER_DISK_FREE_FRACTION_MICROS = \"{}\"\n\n[preempt]\npreemptible = true\n\n[limits]\nmax_runtime_hours = 2\nmax_restarts = 2\n",
        toml_quote(&job.identity),
        toml_quote(PROJECT_ID),
        args,
        toml_quote(workdir),
        job.gpus,
        policy.priority_weight,
        policy.min_worker_disk_free_fraction_micros,
    );
    for input in inputs {
        rendered.push_str(&format!(
            "\n[[input]]\nname = {}\nsource = {}\ntarget = {}\nsha256 = {}\n",
            toml_quote(&input.name),
            toml_quote(&input.source),
            toml_quote(&input.target),
            toml_quote(&input.sha256),
        ));
        if input.executable {
            rendered.push_str("executable = true\n");
        }
    }
    for output in outputs {
        rendered.push_str(&format!(
            "\n[[output]]\nname = {}\nsource = {}\ndirectory = {}\n",
            toml_quote(&output.name),
            toml_quote(&output.source),
            output.directory,
        ));
    }
    rendered
}

fn write_spec_once(path: &Path, bytes: &[u8]) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("create SCQ spec directory: {error}"))?;
    }
    match OpenOptions::new().write(true).create_new(true).open(path) {
        Ok(mut file) => {
            file.write_all(bytes)
                .map_err(|error| format!("write SCQ spec: {error}"))?;
            file.sync_all()
                .map_err(|error| format!("sync SCQ spec: {error}"))
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            let present = fs::read(path).map_err(|read| format!("read SCQ spec: {read}"))?;
            if present == bytes {
                Ok(())
            } else {
                Err(format!(
                    "immutable SCQ spec collision at {}",
                    path.display()
                ))
            }
        }
        Err(error) => Err(format!("create SCQ spec: {error}")),
    }
}

fn worker_args(job: &ScqJobV1) -> Result<Vec<String>, String> {
    let mut args = vec!["{input:worker}".into()];
    match job.stage {
        ScqStageV1::ProposalAll42 => args.extend([
            "propose-all".into(),
            "--campaign".into(),
            "{input:campaign}".into(),
            "--artifact-root".into(),
            "worker-output/artifacts".into(),
            "--work-root".into(),
            "worker-output/work".into(),
            "--output".into(),
            "worker-output/proposal-batch.json".into(),
        ]),
        ScqStageV1::ControlProof
        | ScqStageV1::ProtectedPilot
        | ScqStageV1::RemainingPilot
        | ScqStageV1::ConfirmationSearch => args.extend([
            "search-one".into(),
            "--campaign".into(),
            "{input:campaign}".into(),
            "--proposal-batch".into(),
            "{input:proposal_batch}".into(),
            "--dataset-id".into(),
            job.dataset_id
                .clone()
                .ok_or_else(|| "search SCQ job lacks dataset ID".to_string())?,
            "--arm".into(),
            arm_name(
                job.arm
                    .ok_or_else(|| "search SCQ job lacks arm".to_string())?,
            )
            .into(),
            "--master-seed".into(),
            job.master_seed
                .ok_or_else(|| "search SCQ job lacks seed".to_string())?
                .to_string(),
            "--artifact-root".into(),
            "worker-output/artifacts".into(),
            "--work-root".into(),
            "worker-output/work".into(),
            "--output".into(),
            "worker-output/search-bundle.json".into(),
        ]),
        ScqStageV1::PilotSealedScore | ScqStageV1::ConfirmationSealedScore => args.extend([
            "sealed-one".into(),
            "--campaign".into(),
            "{input:campaign}".into(),
            "--selection-set".into(),
            "{input:selection_set}".into(),
            "--artifact-root".into(),
            "worker-output/artifacts".into(),
            "--work-root".into(),
            "worker-output/work".into(),
            "--output".into(),
            "worker-output/sealed-bundle.json".into(),
        ]),
    }
    Ok(args)
}

/// Renders every immutable job envelope and invokes `scq check`; it has no
/// submission path. Stage-specific dependent inputs must already exist and be
/// supplied by name before their specs can pass checking.
pub fn emit_and_check_plan<F>(
    plan: &ScqPlanV1,
    worker: ScqInputV1,
    campaign: ScqInputV1,
    spec_root: &Path,
    workdir: &str,
    scq_binary: &Path,
    mut dependent_input: F,
) -> Result<Vec<String>, String>
where
    F: FnMut(&ScqJobV1) -> Result<Vec<ScqInputV1>, String>,
{
    plan.validate()?;
    let mut checked = Vec::with_capacity(plan.jobs.len());
    for job in &plan.jobs {
        let mut inputs = vec![worker.clone(), campaign.clone()];
        inputs.extend(dependent_input(job)?);
        let outputs = vec![ScqOutputV1 {
            name: "worker_artifacts".into(),
            source: "worker-output".into(),
            directory: true,
        }];
        let spec = render_scq_spec_with_io(job, &worker_args(job)?, workdir, &inputs, &outputs);
        let path = spec_root.join(format!("{}.toml", job.identity));
        write_spec_once(&path, spec.as_bytes())?;
        let output = Command::new(scq_binary)
            .arg("check")
            .arg(&path)
            .output()
            .map_err(|error| format!("launch scq check for {}: {error}", job.identity))?;
        if !output.status.success() {
            return Err(format!(
                "scq check failed for {}: {}",
                job.identity,
                String::from_utf8_lossy(&output.stderr).trim()
            ));
        }
        checked.push(path.to_string_lossy().into_owned());
    }
    Ok(checked)
}
