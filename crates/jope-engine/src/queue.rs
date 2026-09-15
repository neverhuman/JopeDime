//! Admit GP work only after its exact training-cut contexts are durable.
use crate::{encoding::Encoding, manifest::*, prepare::Prepared, service::GpJobs};
use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    os::fd::AsRawFd,
    path::{Path, PathBuf},
};

fn input_bytes(path: &Path, inputs: &mut BTreeMap<PathBuf, String>) -> Result<Vec<u8>> {
    let bytes =
        fs::read(path).with_context(|| format!("read admission input {}", path.display()))?;
    let sha = hash(&bytes);
    if let Some(previous) = inputs.insert(path.to_owned(), sha.clone()) {
        ensure!(
            previous == sha,
            "admission input changed while validating: {}",
            path.display()
        );
    }
    Ok(bytes)
}

fn recheck_inputs(inputs: &BTreeMap<PathBuf, String>) -> Result<()> {
    for (path, expected) in inputs {
        ensure!(
            file_hash(path)? == *expected,
            "admission input changed before publication: {}",
            path.display()
        );
    }
    Ok(())
}

fn is_sha(value: &Value) -> bool {
    value.as_str().is_some_and(|s| {
        s.len() == 64
            && s.bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
    })
}

pub fn enqueue(root: &Path, jobs_path: &Path, contexts_path: &Path) -> Result<usize> {
    enqueue_bound(root, jobs_path, contexts_path, None)
}

/// Campaign teaching admissions must hit the frozen campaign index. Research GP
/// keeps calling `enqueue` with a broader index and is unchanged.
pub fn enqueue_bound(
    root: &Path,
    jobs_path: &Path,
    contexts_path: &Path,
    campaign_contexts: Option<&Path>,
) -> Result<usize> {
    let mut inputs = BTreeMap::new();
    let jobs_bytes = input_bytes(jobs_path, &mut inputs)?;
    let jobs_sha = hash(&jobs_bytes);
    let mut jobs: GpJobs = serde_json::from_slice(&jobs_bytes)?;
    let index_bytes = input_bytes(contexts_path, &mut inputs)?;
    let contexts_sha = hash(&index_bytes);
    let index: Value = serde_json::from_slice(&index_bytes)?;
    ensure!(
        jobs.schema == "parallel.gp_jobs.v1" && !jobs.jobs.is_empty(),
        "invalid GP queue manifest"
    );
    ensure!(
        index["schema"] == "parallel.context_index.v1" && index["dimensions"] == 2048,
        "saved full context index required before GP admission"
    );
    ensure!(
        jobs.qualification || index["qualification"] == false,
        "qualification encodings cannot enter a clean GP campaign"
    );
    let contexts = index["encodings"]
        .as_array()
        .context("context entries absent")?;
    let mut lookup = BTreeMap::new();
    for row in contexts {
        let key = (
            row["dataset"].as_str().context("dataset absent")?,
            row["cut"].as_u64().context("cut absent")?,
        );
        ensure!(lookup.insert(key, row).is_none(), "duplicate context entry");
    }
    let campaign = campaign_contexts
        .map(|path| {
            let bytes = input_bytes(path, &mut inputs)?;
            let campaign: Value = serde_json::from_slice(&bytes)?;
            ensure!(
                campaign["schema"] == "parallel.context_index.v1"
                    && campaign["dimensions"] == 2048
                    && campaign["qualification"].is_boolean()
                    && campaign["qualification"] == index["qualification"]
                    && is_sha(&campaign["assembly_sha256"])
                    && campaign["assembly_sha256"] == index["assembly_sha256"],
                "campaign context index policy or encoder differs from supplied contexts"
            );
            let mut keys = BTreeMap::<(String, u64), Value>::new();
            for row in campaign["encodings"]
                .as_array()
                .context("campaign encodings absent")?
            {
                let dataset = row["dataset"]
                    .as_str()
                    .filter(|s| !s.is_empty())
                    .context("campaign dataset absent")?;
                let key = (
                    dataset.to_string(),
                    row["cut"].as_u64().context("campaign cut absent")?,
                );
                for field in [
                    "prepared",
                    "prepared_sha256",
                    "family",
                    "partition",
                    "path",
                    "sha256",
                ] {
                    ensure!(
                        row[field].as_str().is_some_and(|s| !s.is_empty()),
                        "campaign context has missing or invalid {field}"
                    );
                }
                for field in ["prepared", "path"] {
                    relative(Path::new(row[field].as_str().unwrap()))?;
                }
                for field in [
                    "prepared_sha256",
                    "sha256",
                    "receipt_sha256",
                    "components_sha256",
                ] {
                    if !row[field].is_null() {
                        ensure!(is_sha(&row[field]), "campaign context has invalid {field}");
                    }
                }
                ensure!(
                    matches!(
                        row["partition"].as_str(),
                        Some("meta_train" | "meta_validation" | "meta_test")
                    ),
                    "campaign context has unsupported partition"
                );
                ensure!(
                    keys.insert(key, row.clone()).is_none(),
                    "duplicate campaign context"
                );
            }
            Ok((hash(&bytes), keys))
        })
        .transpose()?;
    let mut prepared = BTreeMap::new();
    for job in &jobs.jobs {
        crate::service::validate_generation(root, job)?;
        relative(&job.output)?;
        if !prepared.contains_key(&job.prepared) {
            let path = resolve(root, &job.prepared)?.join("prepared.json");
            let bytes = input_bytes(&path, &mut inputs)?;
            ensure!(
                hash(&bytes) == job.prepared_sha256,
                "prepared input changed"
            );
            prepared.insert(
                job.prepared.clone(),
                (
                    job.prepared_sha256.clone(),
                    serde_json::from_slice::<Prepared>(&bytes)?,
                ),
            );
        }
        let (prepared_sha, p) = &prepared[&job.prepared];
        ensure!(
            *prepared_sha == job.prepared_sha256,
            "inconsistent prepared identity within queue"
        );
        // Match the search worker's prepared-data contract before publishing jobs.
        // A saved encoding alone does not make a new preparation status executable.
        crate::task_contract::require_task(&p.task)?;
        ensure!(
            p.schema == "parallel.prepared.v1"
                && (p.meta_partition == "meta_train" || jobs.qualification)
                && (p.family_partition_frozen || jobs.qualification)
                && (p.clean_evaluation
                    || (jobs.qualification
                        && p.qualification_only
                        && p.clean_status == "raw_training_reconstructed_qualification_only")),
            "queued preparation is incompatible with GP workers: {} (task={}, status={})",
            job.prepared.display(),
            p.task,
            p.clean_status
        );
        let cut = p
            .cuts
            .iter()
            .find(|c| c.index == job.cut)
            .context("queued cut missing")?;
        let row = lookup
            .get(&(p.dataset.as_str(), job.cut as u64))
            .context("queued cut has no precalculated context")?;
        if let Some((_, campaign_keys)) = &campaign {
            ensure!(
                p.meta_partition != "meta_test",
                "sealed-test jobs cannot enter a teaching campaign"
            );
            let bound = campaign_keys
                .get(&(p.dataset.clone(), job.cut as u64))
                .with_context(|| {
                    format!(
                        "job is outside the bound campaign context index: {} cut {}",
                        p.dataset, job.cut
                    )
                })?;
            for field in [
                "dataset",
                "cut",
                "prepared",
                "prepared_sha256",
                "family",
                "partition",
                "path",
                "sha256",
                "receipt_sha256",
                "components_sha256",
            ] {
                ensure!(
                    bound[field] == row[field],
                    "campaign context {field} binding differs: {} cut {}",
                    p.dataset,
                    job.cut
                );
            }
            ensure!(
                bound["prepared"].as_str() == job.prepared.to_str(),
                "campaign prepared path differs from queued input"
            );
        }
        ensure!(
            row["prepared_sha256"] == job.prepared_sha256
                && row["family"] == p.family
                && row["partition"] == p.meta_partition,
            "context provenance mismatch"
        );
        let payload = resolve(
            root,
            Path::new(row["path"].as_str().context("context path absent")?),
        )?;
        let bytes = input_bytes(&payload, &mut inputs)?;
        let metadata_bytes = input_bytes(&payload.with_extension("json"), &mut inputs)?;
        let metadata: Encoding = serde_json::from_slice(&metadata_bytes)?;
        let metadata_value: Value = serde_json::from_slice(&metadata_bytes)?;
        if !row["receipt_sha256"].is_null() {
            ensure!(
                row["receipt_sha256"] == hash(&metadata_bytes),
                "context receipt changed"
            );
        }
        if !row["components_sha256"].is_null() {
            ensure!(
                row["components_sha256"] == metadata_value["components_sha256"],
                "context component binding changed"
            );
        }
        ensure!(
            metadata.schema == "parallel.encoding.v1"
                && bytes.len() == 8192
                && hash(&bytes) == row["sha256"]
                && metadata.output_sha256 == row["sha256"]
                && metadata.encoder_spec_sha256 == index["assembly_sha256"]
                && metadata.prepared_sha256 == job.prepared_sha256
                && metadata.training_sha256 == cut.training_sha256
                && metadata.preprocessing_sha256 == cut.preprocessing_sha256
                && metadata.selected_training_rows_sha256
                    == hash(&serde_json::to_vec(&cut.train_indices)?)
                && metadata.dataset == p.dataset
                && metadata.family == p.family
                && metadata.partition == p.meta_partition
                && metadata.cut == job.cut
                && metadata.dataset_dimensions == 1536
                && metadata.target_dimensions == 512,
            "queued context bytes or training-cut identity changed"
        );
        ensure!(
            bytes
                .chunks_exact(4)
                .all(|b| f32::from_le_bytes(b.try_into().unwrap()).is_finite()),
            "queued context contains nonfinite values"
        );
        match (&job.seeds, &job.seeds_sha256) {
            (Some(path), Some(sha)) => ensure!(
                hash(&input_bytes(&resolve(root, path)?, &mut inputs)?) == *sha,
                "seed population changed"
            ),
            (None, None) => (),
            _ => anyhow::bail!("seed population receipt incomplete"),
        }
    }
    // Process the next cut across datasets before taking another hundred cuts
    // from one dataset. This makes early teaching snapshots cover the cohort.
    jobs.jobs.sort_by(|a, b| {
        a.cut
            .cmp(&b.cut)
            .then(
                prepared[&a.prepared]
                    .1
                    .family
                    .cmp(&prepared[&b.prepared].1.family),
            )
            .then(a.prepared.cmp(&b.prepared))
    });
    recheck_inputs(&inputs)?;
    let added = merge(root, &jobs)?;
    let receipt_path = root
        .join("runs/queue-receipts")
        .join(format!("{jobs_sha}.json"));
    let mut receipt = json!({"schema":"parallel.encoded_gp_admission.v1", "timestamp":crate::telemetry::now(),
        "jobs_sha256":jobs_sha, "contexts_sha256":contexts_sha,
        "verified_contexts":jobs.jobs.len(), "new_jobs":added, "qualification":jobs.qualification,
        "copied_dataset_bytes":0, "training_contexts_precalculated":true, "final_test_targets_opened":false,
        "campaign_bound":campaign.is_some(), "inputs_rechecked_before_merge":true});
    if let Some((sha, _)) = campaign {
        receipt["campaign_contexts_sha256"] = sha.into();
    }
    write_json(&receipt_path, &receipt)?;
    Ok(added)
}

fn encoding_keys(index: &Value) -> Result<BTreeMap<(String, u64), &Value>> {
    let mut keys = BTreeMap::new();
    for row in index["encodings"].as_array().context("encodings absent")? {
        keys.insert(
            (
                row["dataset"]
                    .as_str()
                    .context("dataset absent")?
                    .to_string(),
                row["cut"].as_u64().context("cut absent")?,
            ),
            row,
        );
    }
    Ok(keys)
}

fn gp_cut_keys(roots: &[&Path]) -> BTreeSet<(String, u64)> {
    let mut keys = BTreeSet::new();
    for root in roots {
        let Ok(datasets) = fs::read_dir(root) else {
            continue;
        };
        for dataset in datasets.flatten() {
            let Ok(cuts) = fs::read_dir(dataset.path()) else {
                continue;
            };
            for cut in cuts.flatten() {
                let name = cut.file_name();
                let Some(n) = name
                    .to_str()
                    .and_then(|s| s.strip_prefix("cut-")?.parse().ok())
                else {
                    continue;
                };
                keys.insert((dataset.file_name().to_string_lossy().into_owned(), n));
            }
        }
    }
    keys
}

/// Read-only coverage of union encodings vs a frozen campaign index vs GP cut dirs.
pub fn coverage_report(campaign: &Value, union: &Value, gp_roots: &[&Path]) -> Result<Value> {
    ensure!(
        campaign["schema"] == "parallel.context_index.v1"
            && union["schema"] == "parallel.context_index.v1",
        "coverage requires context indexes"
    );
    let campaign_keys = encoding_keys(campaign)?;
    let union_keys = encoding_keys(union)?;
    let gp = gp_cut_keys(gp_roots);
    let in_cohort = gp.iter().filter(|k| campaign_keys.contains_key(*k)).count();
    let wasted = gp
        .iter()
        .filter(|k| !campaign_keys.contains_key(*k))
        .count();
    let unused: Vec<_> = campaign_keys
        .keys()
        .filter(|k| !gp.contains(*k))
        .map(|(d, c)| json!({"dataset":d,"cut":c,"reason":"campaign encoding has no GP cut directory"}))
        .collect();
    Ok(
        json!({"schema":"parallel.campaign_coverage.v1","timestamp":crate::telemetry::now(),
        "union_encodings":union_keys.len(),"campaign_encodings":campaign_keys.len(),
        "gp_cuts":gp.len(),"in_cohort_gp":in_cohort,"wasted_gp":wasted,
        "unused_campaign_cuts":unused.len(),"unused":unused,
        "scope":"Directory inventory only; out-of-cohort GP is unavailable to this selected campaign, not inherently wasted. Counts do not prove completed fits or row identity.",
        "final_test_targets_opened":false}),
    )
}

pub fn write_coverage(
    root: &Path,
    campaign_path: &Path,
    union_path: &Path,
    gp_roots: &[&Path],
    output: &Path,
) -> Result<Value> {
    let report = coverage_report(
        &read_json(campaign_path)?,
        &read_json(union_path)?,
        gp_roots,
    )?;
    write_json(output, &report)?;
    let _ = write_json(&root.join("runs/campaign-coverage.json"), &report);
    Ok(report)
}

#[cfg(test)]
mod admission_snapshot_tests {
    use super::*;

    #[test]
    fn input_snapshots_reject_changes_and_deletions_before_publication() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("campaign.json");
        fs::write(&path, b"old").unwrap();
        let mut inputs = BTreeMap::new();
        assert_eq!(input_bytes(&path, &mut inputs).unwrap(), b"old");
        recheck_inputs(&inputs).unwrap();
        fs::write(&path, b"new").unwrap();
        assert!(recheck_inputs(&inputs).is_err());
        assert!(input_bytes(&path, &mut inputs).is_err());
        fs::remove_file(path).unwrap();
        assert!(recheck_inputs(&inputs).is_err());
    }
}

/// Atomic, idempotent priority merge shared by campaign and data controllers.
pub fn merge(root: &Path, incoming: &GpJobs) -> Result<usize> {
    fs::create_dir_all(root.join("runs"))?;
    let lock = fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(root.join("runs/gp-priority-jobs.lock"))?;
    ensure!(
        unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX) } == 0,
        "queue lock failed"
    );
    let path = root.join("runs/gp-priority-jobs.json");
    let mut combined = incoming.clone();
    let mut added = incoming.jobs.len();
    if path.exists() {
        let old: GpJobs = read_json(&path)?;
        ensure!(
            old.schema == incoming.schema && old.qualification == incoming.qualification,
            "queue eligibility changed"
        );
        let vocab_hash = |j: &GpJobs| -> Result<Option<String>> {
            j.vocabulary
                .as_ref()
                .map(|p| file_hash(&resolve(root, p)?))
                .transpose()
        };
        ensure!(
            vocab_hash(&old)? == vocab_hash(incoming)?,
            "queue vocabulary changed"
        );
        let mut new = BTreeMap::new();
        for job in &incoming.jobs {
            ensure!(
                new.insert(&job.output, job).is_none(),
                "duplicate output in incoming queue"
            );
        }
        for job in old.jobs {
            if let Some(replacement) = new.get(&job.output) {
                ensure!(
                    serde_json::to_value(replacement)? == serde_json::to_value(&job)?,
                    "existing GP output identity changed"
                );
                added -= 1;
            } else {
                combined.jobs.push(job);
            }
        }
    }
    write_json(&path, &combined)?;
    Ok(added)
}
