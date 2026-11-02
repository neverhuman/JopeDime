use super::*;

pub(crate) fn scq_spec(manifest: &ShardManifest) -> Result<String> {
    let bundle_root = remote_bundle_root(&manifest.node, &manifest.release_binary_sha256)?;
    let binary = format!("{bundle_root}/jope-dime-regressor");
    let manifest_sha256 = sha256_bytes(&canonical_json(manifest)?);
    let remote_manifest = format!("{bundle_root}/jobs/manifests/{manifest_sha256}.json");
    let remote_output = format!("{bundle_root}/jobs/{}/output", manifest.job_id);
    let args = vec![
        "OMP_NUM_THREADS=1".to_string(),
        "OPENBLAS_NUM_THREADS=1".to_string(),
        "MKL_NUM_THREADS=1".to_string(),
        "NUMEXPR_NUM_THREADS=1".to_string(),
        binary,
        "binary-test-adaptive-batch-worker".to_string(),
        "--manifest".to_string(),
        remote_manifest.clone(),
        "--output-dir".to_string(),
        remote_output,
    ];
    let rendered_args = args
        .iter()
        .map(serde_json::to_string)
        .collect::<std::result::Result<Vec<_>, _>>()?
        .join(", ");
    Ok(format!(
        "[job]\nid = {id:?}\nproject = {PROJECT:?}\ncommand = \"/usr/bin/env\"\nargs = [{rendered_args}]\nworkdir = \"/tmp\"\ntier = \"normal\"\n\n[resource]\ngpus_per_task = 0\ncpus_reserved = {cpus}\nomp_threads = 1\nram_gb = 0\ngang_size = 1\nplacement = \"single_node\"\n\n[[dependencies.smoke]]\nname = \"Binary42 immutable worker manifest staged\"\ncommand = \"/usr/bin/test\"\nargs = [\"-f\", {remote_manifest:?}]\ntimeout_secs = 5\n\n[preempt]\npreemptible = false\nsignal = \"SIGTERM\"\ngrace_seconds = 60\n\n[limits]\nmax_runtime_hours = 24\nmax_restarts = 0\n",
        id = manifest.job_id,
        cpus = manifest.cpus_reserved,
    ))
}

pub(crate) fn copy_remote_output(node: &str, remote: &str, local: &Path) -> Result<()> {
    if local.exists() {
        fs::remove_dir_all(local)?;
    }
    fs::create_dir_all(local)?;
    if is_controller_node(node)? {
        copy_directory_contents(Path::new(remote), local)?;
    } else {
        scq_batch::command_output(
            {
                let mut command = Command::new("scp");
                command
                    .arg("-r")
                    .arg("--")
                    .arg(format!("{node}:{remote}/."))
                    .arg(local);
                command
            },
            "pull Binary42 SCQ output",
        )?;
    }
    Ok(())
}

pub(crate) fn copy_directory_contents(source: &Path, destination: &Path) -> Result<()> {
    for entry in fs::read_dir(source)
        .with_context(|| format!("read local SCQ output {}", source.display()))?
    {
        let entry = entry?;
        let source_path = entry.path();
        let destination_path = destination.join(entry.file_name());
        let file_type = entry.file_type()?;
        if file_type.is_symlink() {
            bail!(
                "local SCQ output contains a symbolic link: {}",
                source_path.display()
            );
        }
        if file_type.is_dir() {
            fs::create_dir_all(&destination_path)?;
            copy_directory_contents(&source_path, &destination_path)?;
        } else if file_type.is_file() {
            fs::copy(&source_path, &destination_path)?;
        } else {
            bail!(
                "local SCQ output contains a non-file entry: {}",
                source_path.display()
            );
        }
    }
    Ok(())
}

pub(crate) fn scq_json_at(binary: &str, args: &[&str]) -> Result<Value> {
    let output = scq_batch::command_output(
        {
            let mut command = Command::new(binary);
            command.args(args).arg("--json");
            command
        },
        "Binary42 SCQ JSON command",
    )?;
    serde_json::from_slice(&output.stdout).context("parse Binary42 SCQ JSON output")
}

pub(crate) fn wait_for_job_at(binary: &str, id: &str) -> Result<Value> {
    loop {
        let show = scq_json_at(binary, &["show", id])?;
        let state = scq_batch::scq_job(&show, id)?
            .get("state")
            .and_then(Value::as_str)
            .context("Binary42 SCQ job has no state")?;
        if matches!(state, "completed" | "failed" | "cancelled") {
            return Ok(show);
        }
        thread::sleep(std::time::Duration::from_secs(1));
    }
}

pub(crate) fn submit_or_recover_job(manifest: &ShardManifest, spec_path: &Path) -> Result<Value> {
    let scq_binary = manifest
        .scq_binary_path
        .as_deref()
        .unwrap_or(scq_binary_str());
    let exists = scq_json_at(scq_binary, &["show", &manifest.job_id])
        .ok()
        .and_then(|show| scq_batch::scq_job(&show, &manifest.job_id).ok().map(|_| ()))
        .is_some();
    if !exists {
        scq_batch::command_output(
            {
                let mut command = Command::new(scq_binary);
                command.arg("submit").arg(spec_path);
                command
            },
            "submit Binary42 SCQ shard",
        )?;
    }
    wait_for_job_at(scq_binary, &manifest.job_id)
}

pub(crate) fn completed_shard_outputs(
    root: &Path,
    manifest: &ShardManifest,
    manifest_sha256: &str,
) -> Result<Option<Vec<FreshOutput>>> {
    let job_receipt_path = root
        .join("jobs")
        .join("receipts")
        .join(format!("{}.v1.json", manifest.job_id));
    if !job_receipt_path.exists() {
        return Ok(None);
    }
    let job_receipt_bytes = fs::read(&job_receipt_path)?;
    let job_receipt: JobReceipt = serde_json::from_slice(&job_receipt_bytes)?;
    if job_receipt_bytes != canonical_json(&job_receipt)?
        || job_receipt.schema != "JopeBinary42TestAdaptiveScqJob.v1"
        || !job_receipt.test_adaptive
        || job_receipt.benchmark_valid
        || job_receipt.job_id != manifest.job_id
        || job_receipt.node != manifest.node
        || job_receipt.attempt != manifest.attempt
        || job_receipt.manifest_sha256 != manifest_sha256
        || job_receipt.state != "completed"
        || job_receipt.exit_code != 0
    {
        bail!("Binary42 completed shard receipt changed or is invalid");
    }
    let raw = root.join("jobs").join("raw").join(&manifest.job_id);
    let worker_bytes = fs::read(raw.join("batch-worker-receipt.v1.json"))?;
    let worker: WorkerReceipt = serde_json::from_slice(&worker_bytes)?;
    if worker_bytes != canonical_json(&worker)?
        || sha256_bytes(&worker_bytes) != job_receipt.worker_receipt_sha256
        || worker.schema != WORKER_SCHEMA
        || !worker.test_adaptive
        || worker.benchmark_valid
        || worker.job_id != manifest.job_id
        || worker.node != manifest.node
        || worker.hostname != physical_hostname(&manifest.node)?
        || worker.manifest_sha256 != manifest_sha256
        || worker.logical_slots != manifest.slots.len()
        || worker.fresh_candidates != manifest.slots.iter().filter(|slot| slot.fresh).count()
    {
        bail!("Binary42 completed worker receipt changed or is invalid");
    }
    let mut outputs = Vec::with_capacity(worker.candidates.len());
    for candidate in &worker.candidates {
        let receipt_bytes = fs::read(raw.join(&candidate.receipt_file))?;
        let prediction_bytes = fs::read(raw.join(&candidate.predictions_file))?;
        if sha256_bytes(&receipt_bytes) != candidate.receipt_sha256
            || sha256_bytes(&prediction_bytes) != candidate.predictions_sha256
        {
            bail!("Binary42 completed candidate output changed");
        }
        let receipt: CandidateReceipt = serde_json::from_slice(&receipt_bytes)?;
        if receipt_bytes != canonical_json(&receipt)?
            || receipt.schema != CANDIDATE_SCHEMA
            || !receipt.test_adaptive
            || receipt.benchmark_valid
            || receipt.scq_job_id != manifest.job_id
            || receipt.node != manifest.node
            || receipt.hostname != physical_hostname(&manifest.node)?
            || receipt.predictions_sha256 != candidate.predictions_sha256
            || receipt.semantic_sha256 != candidate.semantic_sha256
        {
            bail!("Binary42 completed candidate receipt is invalid");
        }
        validate_candidate_backend(&receipt, manifest.cohort_mode.as_deref() == Some("all"))?;
        outputs.push(FreshOutput {
            receipt,
            receipt_bytes,
            prediction_bytes,
        });
    }
    Ok(Some(outputs))
}

pub(crate) fn execute_shard(root: &Path, manifest: &ShardManifest) -> Result<Vec<FreshOutput>> {
    let manifest_bytes = canonical_json(manifest)?;
    let manifest_sha256 = sha256_bytes(&manifest_bytes);
    let local_manifest = root
        .join("jobs")
        .join("manifests")
        .join(format!("{manifest_sha256}.json"));
    immutable_or_equal(&local_manifest, &manifest_bytes)?;
    if let Some(outputs) = completed_shard_outputs(root, manifest, &manifest_sha256)? {
        return Ok(outputs);
    }
    let bundle_root = remote_bundle_root(&manifest.node, &manifest.release_binary_sha256)?;
    let remote_manifest = format!("{bundle_root}/jobs/manifests/{manifest_sha256}.json");
    let remote_output = format!("{bundle_root}/jobs/{}/output", manifest.job_id);
    if is_controller_node(&manifest.node)? {
        copy_local_atomic(&local_manifest, Path::new(&remote_manifest), false)?;
        fs::create_dir_all(&remote_output)?;
    } else {
        scq_batch::copy_atomic(&manifest.node, &local_manifest, &remote_manifest, false)?;
        scq_batch::ssh_output(
            &manifest.node,
            &["mkdir", "-p", &remote_output],
            "create Binary42 remote output directory",
        )?;
    }
    let spec = scq_spec(manifest)?;
    let spec_path = root
        .join("jobs")
        .join("specs")
        .join(format!("{}.toml", manifest.job_id));
    immutable_or_equal(&spec_path, spec.as_bytes())?;
    let show = submit_or_recover_job(manifest, &spec_path)?;
    let job = scq_batch::scq_job(&show, &manifest.job_id)?;
    let state = job
        .get("state")
        .and_then(Value::as_str)
        .context("Binary42 SCQ job has no state")?;
    let exit_code = job
        .get("exit_code")
        .and_then(Value::as_i64)
        .context("Binary42 SCQ job has no exit code")?;
    if state != "completed"
        || exit_code != 0
        || job.get("node").and_then(Value::as_str) != Some(manifest.node.as_str())
        || job.get("cpus").and_then(Value::as_u64) != Some(manifest.cpus_reserved as u64)
    {
        bail!(
            "Binary42 SCQ job {} failed: state={state}, exit={exit_code}",
            manifest.job_id
        );
    }
    let temporary = std::env::temp_dir().join(format!(
        "jope-b42-pull-{}-{}",
        std::process::id(),
        manifest.job_id
    ));
    copy_remote_output(&manifest.node, &remote_output, &temporary)?;
    let raw = root.join("jobs").join("raw").join(&manifest.job_id);
    let worker_bytes = fs::read(temporary.join("batch-worker-receipt.v1.json"))?;
    let worker: WorkerReceipt = serde_json::from_slice(&worker_bytes)?;
    if worker.schema != WORKER_SCHEMA
        || !worker.test_adaptive
        || worker.benchmark_valid
        || worker.job_id != manifest.job_id
        || worker.node != manifest.node
        || worker.hostname != physical_hostname(&manifest.node)?
        || worker.manifest_sha256 != manifest_sha256
        || worker.logical_slots != manifest.slots.len()
        || worker.fresh_candidates != manifest.slots.iter().filter(|slot| slot.fresh).count()
    {
        bail!("Binary42 worker receipt binding mismatch");
    }
    immutable_or_equal(&raw.join("batch-worker-receipt.v1.json"), &worker_bytes)?;
    let mut outputs = Vec::new();
    for candidate in &worker.candidates {
        let receipt_bytes = fs::read(temporary.join(&candidate.receipt_file))?;
        let prediction_bytes = fs::read(temporary.join(&candidate.predictions_file))?;
        if sha256_bytes(&receipt_bytes) != candidate.receipt_sha256
            || sha256_bytes(&prediction_bytes) != candidate.predictions_sha256
        {
            bail!("Binary42 worker candidate output hash mismatch");
        }
        let receipt: CandidateReceipt = serde_json::from_slice(&receipt_bytes)?;
        if receipt.schema != CANDIDATE_SCHEMA
            || !receipt.test_adaptive
            || receipt.benchmark_valid
            || receipt.scq_job_id != manifest.job_id
            || receipt.node != manifest.node
            || receipt.hostname != physical_hostname(&manifest.node)?
            || receipt.predictions_sha256 != candidate.predictions_sha256
            || receipt.semantic_sha256 != candidate.semantic_sha256
        {
            bail!("Binary42 candidate receipt binding mismatch");
        }
        validate_candidate_backend(&receipt, manifest.cohort_mode.as_deref() == Some("all"))?;
        immutable_or_equal(&raw.join(&candidate.receipt_file), &receipt_bytes)?;
        immutable_or_equal(&raw.join(&candidate.predictions_file), &prediction_bytes)?;
        outputs.push(FreshOutput {
            receipt,
            receipt_bytes,
            prediction_bytes,
        });
    }
    fs::remove_dir_all(&temporary)?;
    let job_receipt = JobReceipt {
        schema: "JopeBinary42TestAdaptiveScqJob.v1".to_string(),
        test_adaptive: true,
        benchmark_valid: false,
        job_id: manifest.job_id.clone(),
        node: manifest.node.clone(),
        attempt: manifest.attempt,
        manifest_sha256,
        scq_show_sha256: sha256_bytes(&canonical_json(&show)?),
        worker_receipt_sha256: sha256_bytes(&worker_bytes),
        state: state.to_string(),
        exit_code,
    };
    write_json(
        &root
            .join("jobs")
            .join("receipts")
            .join(format!("{}.v1.json", manifest.job_id)),
        &job_receipt,
    )?;
    Ok(outputs)
}
