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
    Kernel2Observation, TrainingRefinement,
};
use crate::{receipt::Seeding, sha256_hex};
use feat_math::kernel::pareto::Archive;

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
    // Two parents can refine to one program in a round, giving equal child
    // measurements. Match by occurrence: the k-th child equal to a measurement
    // takes the k-th equal observation, in the order the search recorded them.
    let mut claimed = vec![false; searched.observations.len()];
    let refinements = searched
        .refinements
        .iter()
        .map(|r| {
            let measurement = r
                .measurement
                .as_ref()
                .map(|m| -> Result<usize> {
                    let index = searched
                        .observations
                        .iter()
                        .zip(&claimed)
                        .position(|(o, &taken)| !taken && o == m)
                        .context("refined measurement missing from observations")?;
                    claimed[index] = true;
                    Ok(index)
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
    // laneD:B18-pareto-archive begin
    let engine_selection = super::archive::publish(searched, root)?;
    // laneD:B18-pareto-archive end
    // laneD:B20-rewrite-simplify begin
    let engine_selection = super::search::moves::publish(searched, root, engine_selection)?;
    // laneD:B20-rewrite-simplify end
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
        engine_selection,
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
    // laneD:B18-pareto-archive begin
    check_archive(root, s, &observations)?;
    // laneD:B18-pareto-archive end
    // laneD:B20-rewrite-simplify begin
    check_moves(root, &r, &records, &observations, &evaluator)?;
    // laneD:B20-rewrite-simplify end
    let manifest: serde_json::Value = serde_json::from_slice(&read_file(
        root,
        Path::new("emit/manifest.json"),
        &r.emit_manifest_sha256,
    )?)?;
    if manifest["envelope_schema"] == feat_math::artifact_manifest::ENVELOPE {
        use feat_math::artifact_manifest::{ExpectedArtifactV3, Target};
        let requested: Vec<Target> =
            serde_json::from_value(manifest["payload"]["requested"].clone())?;
        let expected = ExpectedArtifactV3 {
            manifest_sha256: r.emit_manifest_sha256.clone(),
            program_sha256: best.observation.program_sha256.clone(),
            requested,
        };
        check_v3_artifacts(root, &r, &expected)?;
    } else {
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
    }
    Ok(r)
}

/// Stronger opt-in consumer: the selected manifest/request pin comes from the
/// caller, independently of a mutable run receipt. Search evidence still follows
/// verify_run's retained-evidence contract, not authenticated execution.
pub fn verify_run_v3(
    path: &Path,
    expected: &feat_math::artifact_manifest::ExpectedArtifactV3,
) -> Result<Receipt> {
    let r = verify_run(path)?;
    ensure!(
        r.emit_manifest_sha256 == expected.manifest_sha256
            && r.search.best.observation.program_sha256 == expected.program_sha256,
        "run receipt disagrees with pinned selected manifest"
    );
    check_v3_artifacts(
        path.parent().context("receipt has no run directory")?,
        &r,
        expected,
    )?;
    Ok(r)
}

fn check_v3_artifacts(
    root: &Path,
    receipt: &Receipt,
    expected: &feat_math::artifact_manifest::ExpectedArtifactV3,
) -> Result<()> {
    use feat_math::artifact_manifest::TargetState;
    let manifest = crate::emit::read_identity_v3(&root.join("emit"), expected)?;
    let expected_exclusions: std::collections::BTreeMap<_, _> = manifest
        .targets
        .iter()
        .filter_map(|r| match &r.state {
            TargetState::Unsupported { reason, .. } => Some((r.target.as_str(), reason.as_str())),
            _ => None,
        })
        .collect();
    let actual_exclusions: std::collections::BTreeMap<_, _> = receipt
        .emit_exclusions
        .iter()
        .map(|r| (r.language.as_str(), r.reason.as_str()))
        .collect();
    ensure!(
        receipt.emit_exclusions.len() == actual_exclusions.len()
            && actual_exclusions == expected_exclusions,
        "receipt exclusions disagree with requested unsupported targets"
    );
    let files: Vec<_> = manifest
        .targets
        .iter()
        .flat_map(|r| match &r.state {
            TargetState::Supported { files, .. } => {
                files.iter().map(move |f| (r.target, f)).collect::<Vec<_>>()
            }
            TargetState::Unsupported { .. } => vec![],
        })
        .collect();
    ensure!(
        receipt.artifacts.len() == files.len(),
        "receipt omits or adds a requested artifact"
    );
    let mut seen = BTreeSet::new();
    for artifact in &receipt.artifacts {
        ensure!(
            seen.insert((&artifact.language, &artifact.path)),
            "duplicate receipt artifact"
        );
        let (target, file) = files
            .iter()
            .find(|(t, f)| f.path == artifact.path && artifact.language == v3_language(*t, &f.path))
            .context("receipt artifact is not in selected target manifest")?;
        ensure!(
            artifact.sha256 == file.sha256 && artifact.bytes == file.bytes,
            "receipt artifact disagrees with selected manifest"
        );
        let record = manifest
            .targets
            .iter()
            .find(|r| r.target == *target)
            .context("missing target record")?;
        if let TargetState::Supported { build, .. } = &record.state {
            if let Some(command) = &artifact.build {
                let declared = std::iter::once(build.toolchain.as_str())
                    .chain(build.flags.iter().map(String::as_str))
                    .collect::<Vec<_>>()
                    .join(" ");
                ensure!(
                    command == &declared,
                    "receipt build disagrees with selected manifest"
                );
            }
        }
    }
    Ok(())
}
fn v3_language(target: feat_math::artifact_manifest::Target, path: &str) -> &'static str {
    use feat_math::artifact_manifest::Target;
    match target {
        Target::Python => "python",
        Target::Rust => "rust",
        Target::Cpp17 => "cpp17",
        Target::Cpp20 => "cpp20",
        Target::JavaScript => "javascript",
        Target::Go if path.ends_with(".kernel.rs") => "go-kernel",
        Target::Go if path == "go.mod" => "go-module",
        Target::Go if path == "go.sum" => "go-checksums",
        Target::Go => "go",
        Target::Cpp17RecipeV2 => "cpp17-recipe-v2",
        Target::Cpp17Fitted if path.ends_with(".codec.json") => "cpp17-fitted-codec",
        Target::Cpp17Fitted if path.ends_with("standalone.cpp") => "cpp17-fitted-standalone",
        Target::Cpp17Fitted => "cpp17-fitted",
    }
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
        Some(Language::Go) => ("model.go", "go"),
        None if artifact.language == "go-module" => ("go.mod", "go-module"),
        None if artifact.language == "go-checksums" => ("go.sum", "go-checksums"),
        None if artifact.language == "go-kernel" => ("model.go.kernel.rs", "go-kernel"),
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
            ["py", "rs", "cpp", "js", "golang", "cpp17-fitted"].contains(&target)
                && targets.insert(target),
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
            "go" | "go-kernel" | "go-module" | "go-checksums" => "golang",
            _ => "cpp17-fitted",
        })
        .collect();
    ensure!(
        targets == actual,
        "artifact languages disagree with emitted targets"
    );
    for language in ["go", "go-kernel", "go-module", "go-checksums"] {
        ensure!(
            languages.contains(language) == targets.contains("golang"),
            "incomplete Go artifact set"
        );
    }
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
        // laneD:B17-lm-refinement begin
        check_trace(s, refinement, &parents[0].observation.genome)?;
        // laneD:B17-lm-refinement end
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
            // Publication assigns children by occurrence in the order the search
            // measured them, so the recorded child indices strictly increase.
            ensure!(
                measured.last().is_none_or(|&last| index > last)
                    && measured.insert(index)
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

// laneD:B17-lm-refinement begin
/// A refinement carries a trace exactly when the damped least-squares method was selected and it
/// spent a trial. The trace's stages, each checked against its own counts and bound to the
/// `parent` program's stages and the refinement allowance, account for every charged trial and
/// refusal.
fn check_trace(s: &Search, refinement: &TrainingRefinement, parent: &str) -> Result<()> {
    let selected = s
        .engine_selection
        .as_ref()
        .and_then(super::EngineRecord::damped_ls);
    let expected = selected.is_some() && refinement.evaluations > 0;
    ensure!(
        refinement.trace.is_some() == expected,
        "refinement trace presence disagrees with the selected method"
    );
    if let (Some(trace), Some(selected)) = (&refinement.trace, selected) {
        let (evaluations, rejected) =
            trace.check(parent, &selected.jacobian, s.refinement.trials_per_program)?;
        ensure!(
            evaluations == refinement.evaluations && rejected == refinement.rejected_trials,
            "refinement trace does not account for its trials"
        );
    }
    Ok(())
}
// laneD:B17-lm-refinement end

// laneD:B18-pareto-archive begin
/// The archive is published exactly when it was selected, reads under the archive reader's own
/// refusals, and is byte for byte the archive rebuilt from the re-derived observations.
fn check_archive(root: &Path, s: &Search, observations: &[Kernel2Observation]) -> Result<()> {
    let path = Path::new(super::archive::PATH);
    let Some(record) = s.engine_selection.as_ref().and_then(|e| e.archive.as_ref()) else {
        ensure!(
            std::fs::symlink_metadata(root.join(path)).is_err(),
            "an archive is published without its selection"
        );
        return Ok(());
    };
    let digest = record
        .sha256
        .as_deref()
        .context("the archive record has no digest")?;
    ensure!(
        std::fs::symlink_metadata(root.join(path)).is_ok(),
        "a selected archive is not published"
    );
    let bytes = read_file(root, path, digest)?;
    let text = std::str::from_utf8(&bytes).context("the archive is not UTF-8")?;
    let published = Archive::from_json(text).map_err(anyhow::Error::msg)?;
    ensure!(
        published.cost_version() == record.cost_version,
        "the archive's cost version disagrees with its record"
    );
    ensure!(
        super::archive::build(observations)?.to_json().as_bytes() == bytes.as_slice(),
        "the archive does not re-derive from the observations"
    );
    Ok(())
}
// laneD:B18-pareto-archive end

// laneD:B20-rewrite-simplify begin
/// The move records are published exactly when a move was selected, match their digest, and
/// re-derive (`moves::check`): which parents each round's passes read, each proposal from its
/// parent, each outcome from the measurements, and the pool refusal count. Without a selected
/// move no measurement has a move's origin.
fn check_moves(
    root: &Path,
    r: &Receipt,
    measurements: &[Measurement],
    observations: &[Kernel2Observation],
    evaluator: &super::Kernel2Evaluator,
) -> Result<()> {
    use super::search::moves;
    let s = &r.search;
    let path = Path::new(moves::PATH);
    let Some(record) = s.engine_selection.as_ref().and_then(|e| e.moves.as_ref()) else {
        ensure!(
            std::fs::symlink_metadata(root.join(path)).is_err()
                && measurements.iter().all(|m| {
                    // laneD:B19-minimax-linearize: the linearization pass's origin too.
                    m.observation.origin != super::REWRITE
                        && m.observation.origin != super::LINEARIZE
                }),
            "move evidence is published without its selection"
        );
        return Ok(());
    };
    let digest = record
        .sha256
        .as_deref()
        .context("the moves record has no digest")?;
    let published: Vec<moves::MoveRecord> =
        serde_json::from_slice(&read_file(root, path, digest)?)?;
    let width = r
        .dataset
        .columns
        .checked_sub(1)
        .context("no feature columns")? as usize;
    moves::check(&published, s, measurements, observations, evaluator, width)
}
// laneD:B20-rewrite-simplify end

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
