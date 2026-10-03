//! Publish and re-derive the coordinator's complete measurement/trial evidence.

use anyhow::{ensure, Context, Result};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeSet,
    path::{Component, Path},
};

use super::{
    receipt::{self, Receipt, Search},
    records::Measurement,
    search::CutSearch,
    TrainingRefinement,
};
use crate::{receipt::Seeding, sha256_hex};

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RefinementRecord {
    pub parent_sha256: String,
    pub refinement: TrainingRefinement,
    /// Index in observations.json, not a second unaccounted measurement.
    pub measurement: Option<usize>,
    pub measurement_not_started_reason: Option<String>,
}

pub fn publish(
    cut: &crate::prepare::Cut,
    searched: &CutSearch,
    seeding: Seeding,
    root: &Path,
) -> Result<(Search, String)> {
    let measurements: Vec<_> = searched
        .observations
        .iter()
        .map(Measurement::record)
        .collect::<Result<_>>()?;
    let refinements = searched
        .refinements
        .iter()
        .map(|r| {
            let measurement = r
                .measurement
                .as_ref()
                .map(|m| {
                    searched
                        .observations
                        .iter()
                        .position(|o| o == m)
                        .context("refined measurement missing from observations")
                })
                .transpose()?;
            let reason = (measurement.is_none() && r.refinement.failure.is_none())
                .then(|| searched.stopped_by.to_owned());
            Ok(RefinementRecord {
                parent_sha256: r.parent_sha256.clone(),
                refinement: r.refinement.clone(),
                measurement,
                measurement_not_started_reason: reason,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let dir = root.join("search/cut-0");
    std::fs::create_dir_all(&dir)?;
    let write = |name: &str, value: &serde_json::Value| -> Result<String> {
        let bytes = serde_json::to_vec_pretty(value)?;
        std::fs::write(dir.join(name), &bytes)?;
        Ok(sha256_hex(&bytes))
    };
    let observations_sha256 = write("observations.json", &serde_json::to_value(&measurements)?)?;
    let refinements_sha256 = write("refinements.json", &serde_json::to_value(&refinements)?)?;
    let record = Search {
        engine: receipt::ENGINE.into(),
        profile_id: searched.profile_id.into(),
        profile_sha256: searched.profile_sha256.clone(),
        fault_policy: searched.fault_policy.clone(),
        rng_seed: searched.rng_seed,
        evaluations: searched.budget_evaluations,
        seconds: searched.budget_seconds,
        refinement: searched.refinement,
        cut: cut.index,
        train_rows: cut.train_rows(),
        hold_rows: cut.hold_rows(),
        generations: searched.generations,
        completed_evaluations: searched.completed_evaluations,
        candidate_measurements: searched.candidate_measurements,
        objective_trials: searched.objective_trials,
        rejected_trials: searched.rejected_trials,
        successful_measurements: searched.successful_measurements,
        stopped_by: searched.stopped_by.into(),
        active_seconds: searched.seconds,
        deadline_overrun: searched.seconds > searched.budget_seconds as f64,
        arm_counts: searched
            .arm_counts
            .iter()
            .map(|(name, count)| ((*name).into(), *count))
            .collect(),
        seeding,
        selection: receipt::SELECTION.into(),
        best: Measurement::record(&searched.best)?,
        observations_sha256,
        refinements_sha256,
    };
    let digest = write("search.json", &serde_json::to_value(&record)?)?;
    Ok((record, digest))
}

fn read_evidence(root: &Path, relative: &Path) -> Result<Vec<u8>> {
    let mut path = root.to_path_buf();
    ensure!(!relative.as_os_str().is_empty(), "empty evidence path");
    for component in relative.components() {
        let Component::Normal(name) = component else {
            anyhow::bail!("unsafe evidence path");
        };
        path.push(name);
        ensure!(
            !std::fs::symlink_metadata(&path)?.file_type().is_symlink(),
            "symlink evidence path"
        );
    }
    std::fs::read(&path).with_context(|| format!("read {}", path.display()))
}

fn read_file(root: &Path, relative: &Path, digest: &str) -> Result<Vec<u8>> {
    let bytes = read_evidence(root, relative)?;
    ensure!(
        sha256_hex(&bytes) == digest,
        "{}: evidence digest mismatch",
        root.join(relative).display()
    );
    Ok(bytes)
}

/// Re-derive scores, fault rates, refinement accounting and the selected measurement.
/// This checks retained evidence; it is not a fit replay or an authenticated run attestation.
pub fn verify_run(path: &Path) -> Result<Receipt> {
    ensure!(
        !std::fs::symlink_metadata(path)?.file_type().is_symlink(),
        "symlink receipt path"
    );
    let r = receipt::read(path)?;
    let root = path.parent().context("receipt has no run directory")?;
    let s = &r.search;
    let prepared: serde_json::Value = serde_json::from_slice(&read_file(
        root,
        Path::new("prepared/prepared.json"),
        &r.prepared.sha256,
    )?)?;
    ensure!(
        prepared["schema"] == crate::prepare::PREPARED_SCHEMA_ID
            && prepared["dataset"] == r.dataset.dataset_hash
            && prepared["seed"] == r.prepared.seed
            && prepared["rows"] == r.prepared.rows
            && prepared["folds"] == r.prepared.folds
            && prepared["columns"] == r.dataset.columns
            && prepared["cuts"].as_array().is_some_and(|c| c.len() == 1)
            && prepared["cuts"][0]["cut"] == s.cut
            && prepared["cuts"][0]["train_rows"] == s.train_rows
            && prepared["cuts"][0]["hold_rows"] == s.hold_rows,
        "prepared evidence does not re-derive"
    );
    let published: Search = serde_json::from_slice(&read_file(
        root,
        Path::new("search/cut-0/search.json"),
        &r.search_sha256,
    )?)?;
    ensure!(published == *s, "search summary disagrees with receipt");
    let records: Vec<Measurement> = serde_json::from_slice(&read_file(
        root,
        Path::new("search/cut-0/observations.json"),
        &s.observations_sha256,
    )?)?;
    let refinements: Vec<RefinementRecord> = serde_json::from_slice(&read_file(
        root,
        Path::new("search/cut-0/refinements.json"),
        &s.refinements_sha256,
    )?)?;
    let evaluator = s.evaluator()?;
    let observations = records
        .iter()
        .map(|r| r.derive(evaluator.fault_policy()))
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        observations.len() == s.candidate_measurements
            && observations
                .iter()
                .filter(|o| o.observation.measured())
                .count()
                == s.successful_measurements
            && observations
                .iter()
                .all(|o| o.observation.cut == s.cut && o.observation.generation < s.generations),
        "measurements do not account for the cut"
    );
    let ranked = evaluator.rank(&observations);
    let best = ranked.first().context("no measured champion")?;
    ensure!(
        Measurement::record(best)? == s.best,
        "selected measurement does not re-derive"
    );
    feat_math::kernel::search::admit_child(
        &best.observation.genome,
        evaluator.profile(),
        r.dataset
            .columns
            .checked_sub(1)
            .context("no feature columns")? as usize,
    )?;
    check_refinements(s, &records, &refinements)?;
    check_seeding(s, &records)?;
    let manifest: serde_json::Value = serde_json::from_slice(&read_file(
        root,
        Path::new("emit/manifest.json"),
        &r.emit_manifest_sha256,
    )?)?;
    ensure!(
        manifest["dsl"] == best.observation.genome
            && manifest["genome_sha256"] == best.observation.program_sha256,
        "emitted manifest does not identify the selected program"
    );
    let mut languages = BTreeSet::new();
    for a in &r.artifacts {
        let language = check_artifact_manifest(&manifest, a)?;
        ensure!(languages.insert(language), "duplicate emitted language");
        let bytes = read_file(root, &Path::new("emit").join(&a.path), &a.sha256)?;
        ensure!(bytes.len() as u64 == a.bytes, "artifact length mismatch");
    }
    check_artifact_targets(root, &languages)?;
    Ok(r)
}

fn check_artifact_manifest(
    manifest: &serde_json::Value,
    artifact: &crate::receipt::Artifact,
) -> Result<&'static str> {
    use crate::emit::Language;
    let (expected_path, language) = match Language::parse(&artifact.language) {
        Some(Language::Rust) => ("model.rs", "rust"),
        Some(Language::Python) => ("model.py", "python"),
        Some(Language::JavaScript) => ("model.js", "javascript"),
        Some(language @ (Language::Cpp17 | Language::Cpp20)) => ("model.cpp", language.as_str()),
        Some(Language::Cpp17Fitted) => ("cpp17-fitted/model.cpp", "cpp17-fitted"),
        None if artifact.language == "cpp17-fitted-codec" => {
            ("cpp17-fitted/model.codec.json", "cpp17-fitted-codec")
        }
        None if artifact.language == "cpp17-fitted-standalone" => {
            ("cpp17-fitted/standalone.cpp", "cpp17-fitted-standalone")
        }
        None => anyhow::bail!("unknown emitted language"),
    };
    ensure!(
        artifact.path == expected_path,
        "emitted artifact path disagrees"
    );
    let digest_field = match expected_path {
        "model.rs" => Some("rs_sha256"),
        "model.py" => Some("py_sha256"),
        _ => None,
    };
    if let Some(field) = digest_field {
        ensure!(
            manifest[field].as_str() == Some(artifact.sha256.as_str()),
            "emitted artifact digest disagrees with selected manifest"
        );
    }
    Ok(language)
}

fn check_artifact_targets(root: &Path, languages: &BTreeSet<&str>) -> Result<()> {
    let retained = String::from_utf8(read_evidence(root, Path::new("emit/targets.txt"))?)?;
    let mut targets = BTreeSet::new();
    for target in retained.split_whitespace() {
        ensure!(
            ["py", "rs", "cpp", "js", "cpp17-fitted"].contains(&target) && targets.insert(target),
            "unknown or duplicate emitted target"
        );
    }
    let actual: BTreeSet<_> = languages
        .iter()
        .map(|language| match *language {
            "python" => "py",
            "rust" => "rs",
            "cpp17" | "cpp20" => "cpp",
            "javascript" => "js",
            _ => "cpp17-fitted",
        })
        .collect();
    ensure!(
        targets == actual,
        "artifact languages disagree with emitted targets"
    );
    for language in [
        "cpp17-fitted",
        "cpp17-fitted-codec",
        "cpp17-fitted-standalone",
    ] {
        ensure!(
            languages.contains(language) == targets.contains("cpp17-fitted"),
            "incomplete fitted artifact set"
        );
    }
    Ok(())
}

fn check_refinements(
    s: &Search,
    measurements: &[Measurement],
    records: &[RefinementRecord],
) -> Result<()> {
    let mut trials = 0usize;
    let mut rejected = 0usize;
    let mut measured = BTreeSet::new();
    for record in records {
        let refinement = &record.refinement;
        ensure!(
            refinement.evaluations <= s.refinement.trials_per_program
                && refinement.rejected_trials <= refinement.evaluations
                && s.refinement.top_k > 0,
            "invalid refinement allowance"
        );
        trials = trials
            .checked_add(refinement.evaluations)
            .context("trial count overflow")?;
        rejected = rejected
            .checked_add(refinement.rejected_trials)
            .context("rejected count overflow")?;
        let parents: Vec<_> = measurements
            .iter()
            .filter(|m| {
                m.observation.program_sha256 == record.parent_sha256
                    && m.observation.status == crate::search::MEASURED
            })
            .collect();
        ensure!(!parents.is_empty(), "refinement parent was not measured");
        ensure!(
            refinement.changed
                == parents
                    .iter()
                    .all(|p| p.observation.genome != refinement.program),
            "refinement change flag disagrees"
        );
        if let Some(index) = record.measurement {
            let child = measurements
                .get(index)
                .context("refinement measurement index out of range")?;
            ensure!(
                measured.insert(index)
                    && child.observation.origin == super::REFINE
                    && child.observation.genome == refinement.program
                    && refinement.evaluations > 0
                    && refinement.failure.is_none()
                    && record.measurement_not_started_reason.is_none()
                    && measurements[..index]
                        .iter()
                        .any(|p| p.observation.program_sha256 == record.parent_sha256
                            && p.observation.status == crate::search::MEASURED
                            && p.observation.generation <= child.observation.generation),
                "refinement child does not re-derive"
            );
        } else {
            ensure!(
                (refinement.failure.is_some() && record.measurement_not_started_reason.is_none())
                    || (record.measurement_not_started_reason.as_deref()
                        == Some(crate::search::STOPPED_BY_SECONDS)
                        && s.stopped_by == crate::search::STOPPED_BY_SECONDS),
                "missing child measurement lacks a stop/failure reason"
            );
        }
    }
    ensure!(
        trials == s.objective_trials
            && rejected == s.rejected_trials
            && measured
                == measurements
                    .iter()
                    .enumerate()
                    .filter(|(_, m)| m.observation.origin == super::REFINE)
                    .map(|(i, _)| i)
                    .collect(),
        "refinement accounting does not re-derive"
    );
    Ok(())
}

fn check_seeding(s: &Search, measurements: &[Measurement]) -> Result<()> {
    let seeding = &s.seeding;
    let mut distinct = BTreeSet::new();
    ensure!(
        seeding.seeds.len() <= crate::search::ELITES
            && seeding.seeds.len() as u64 <= seeding.offered
            && seeding.parent_receipt_sha256.is_none(),
        "unsupported or invalid seeding record"
    );
    for seed in &seeding.seeds {
        ensure!(
            seed.source == crate::seeds::FROM_SEEDS_FILE
                && seeding.seeds_file_sha256.is_some()
                && distinct.insert(&seed.genome)
                && seed.genome != crate::search::BOOTSTRAP_GENOME
                && seed.program_sha256 == sha256_hex(seed.genome.as_bytes())
                && measurements
                    .iter()
                    .any(|m| m.observation.genome == seed.genome
                        && m.observation.origin == crate::search::INCUMBENT
                        && m.observation.generation == 0),
            "seed was not re-measured in generation zero"
        );
    }
    ensure!(
        measurements
            .first()
            .is_some_and(|m| m.observation.genome == crate::search::BOOTSTRAP_GENOME
                && m.observation.origin == crate::search::BOOTSTRAP
                && m.observation.generation == 0),
        "missing bootstrap measurement"
    );
    Ok(())
}
