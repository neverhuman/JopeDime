//! Admit GP work only after its exact training-cut contexts are durable.
use crate::{encoding::Encoding, manifest::*, prepare::Prepared, service::GpJobs};
use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};
use std::{collections::BTreeMap, fs, os::fd::AsRawFd, path::Path};

pub fn enqueue(root: &Path, jobs_path: &Path, contexts_path: &Path) -> Result<usize> {
    let mut jobs: GpJobs = read_json(jobs_path)?;
    let index: Value = read_json(contexts_path)?;
    ensure!(jobs.schema == "parallel.gp_jobs.v1" && !jobs.jobs.is_empty(), "invalid GP queue manifest");
    ensure!(index["schema"] == "parallel.context_index.v1" && index["dimensions"] == 2048,
        "saved full context index required before GP admission");
    ensure!(jobs.qualification || index["qualification"] == false,
        "qualification encodings cannot enter a clean GP campaign");
    let contexts = index["encodings"].as_array().context("context entries absent")?;
    let mut lookup = BTreeMap::new();
    for row in contexts {
        let key = (row["dataset"].as_str().context("dataset absent")?, row["cut"].as_u64().context("cut absent")?);
        ensure!(lookup.insert(key, row).is_none(), "duplicate context entry");
    }
    let mut prepared = BTreeMap::new();
    for job in &jobs.jobs {
        relative(&job.output)?;
        if !prepared.contains_key(&job.prepared) {
            let path = resolve(root, &job.prepared)?.join("prepared.json");
            ensure!(file_hash(&path)? == job.prepared_sha256, "prepared input changed");
            prepared.insert(job.prepared.clone(), (job.prepared_sha256.clone(), read_json::<Prepared>(&path)?));
        }
        let (prepared_sha, p) = &prepared[&job.prepared];
        ensure!(*prepared_sha == job.prepared_sha256, "inconsistent prepared identity within queue");
        let cut = p.cuts.iter().find(|c| c.index == job.cut).context("queued cut missing")?;
        let row = lookup.get(&(p.dataset.as_str(), job.cut as u64)).context("queued cut has no precalculated context")?;
        ensure!(row["prepared_sha256"] == job.prepared_sha256 && row["family"] == p.family
            && row["partition"] == p.meta_partition, "context provenance mismatch");
        let payload = resolve(root, Path::new(row["path"].as_str().context("context path absent")?))?;
        let bytes = fs::read(&payload)?;
        let metadata: Encoding = read_json(&payload.with_extension("json"))?;
        ensure!(bytes.len() == 8192 && hash(&bytes) == row["sha256"]
            && metadata.output_sha256 == row["sha256"]
            && metadata.encoder_spec_sha256 == index["assembly_sha256"]
            && metadata.prepared_sha256 == job.prepared_sha256
            && metadata.training_sha256 == cut.training_sha256
            && metadata.preprocessing_sha256 == cut.preprocessing_sha256
            && metadata.selected_training_rows_sha256 == hash(&serde_json::to_vec(&cut.train_indices)?)
            && metadata.dataset == p.dataset && metadata.family == p.family
            && metadata.partition == p.meta_partition && metadata.cut == job.cut
            && metadata.dataset_dimensions == 1536 && metadata.target_dimensions == 512,
            "queued context bytes or training-cut identity changed");
        ensure!(bytes.chunks_exact(4).all(|b| f32::from_le_bytes(b.try_into().unwrap()).is_finite()),
            "queued context contains nonfinite values");
        match (&job.seeds, &job.seeds_sha256) {
            (Some(path), Some(sha)) => ensure!(file_hash(&resolve(root, path)?)? == *sha, "seed population changed"),
            (None, None) => (),
            _ => anyhow::bail!("seed population receipt incomplete"),
        }
    }
    // Process the next cut across datasets before taking another hundred cuts
    // from one dataset. This makes early teaching snapshots cover the cohort.
    jobs.jobs.sort_by(|a,b| a.cut.cmp(&b.cut)
        .then(prepared[&a.prepared].1.family.cmp(&prepared[&b.prepared].1.family))
        .then(a.prepared.cmp(&b.prepared)));
    let added = merge(root, &jobs)?;
    let receipt = root.join("runs/queue-receipts").join(format!("{}.json", file_hash(jobs_path)?));
    write_json(&receipt, &json!({"schema":"parallel.encoded_gp_admission.v1", "timestamp":crate::telemetry::now(),
        "jobs_sha256":file_hash(jobs_path)?, "contexts_sha256":file_hash(contexts_path)?,
        "verified_contexts":jobs.jobs.len(), "new_jobs":added, "qualification":jobs.qualification,
        "copied_dataset_bytes":0, "training_contexts_precalculated":true, "final_test_targets_opened":false}))?;
    Ok(added)
}

/// Atomic, idempotent priority merge shared by campaign and data controllers.
pub fn merge(root: &Path, incoming: &GpJobs) -> Result<usize> {
    fs::create_dir_all(root.join("runs"))?;
    let lock = fs::OpenOptions::new().create(true).truncate(false).write(true)
        .open(root.join("runs/gp-priority-jobs.lock"))?;
    ensure!(unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX) } == 0, "queue lock failed");
    let path = root.join("runs/gp-priority-jobs.json");
    let mut combined = incoming.clone();
    let mut added = incoming.jobs.len();
    if path.exists() {
        let old: GpJobs = read_json(&path)?;
        ensure!(old.schema == incoming.schema && old.qualification == incoming.qualification, "queue eligibility changed");
        let vocab_hash = |j: &GpJobs| -> Result<Option<String>> {
            j.vocabulary.as_ref().map(|p| file_hash(&resolve(root,p)?)).transpose()
        };
        ensure!(vocab_hash(&old)? == vocab_hash(incoming)?, "queue vocabulary changed");
        let mut new = BTreeMap::new();
        for job in &incoming.jobs {
            ensure!(new.insert(&job.output, job).is_none(), "duplicate output in incoming queue");
        }
        for job in old.jobs {
            if let Some(replacement) = new.get(&job.output) {
                ensure!(serde_json::to_value(replacement)? == serde_json::to_value(&job)?, "existing GP output identity changed");
                added -= 1;
            } else { combined.jobs.push(job); }
        }
    }
    write_json(&path, &combined)?;
    Ok(added)
}
