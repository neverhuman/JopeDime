use super::*;

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
        let mut members = BTreeSet::new();
        for a in &r.artifacts {
            let language = check_artifact_manifest(&manifest, a)?;
            ensure!(
                members.insert((language, a.path.as_str())),
                "duplicate emitted artifact"
            );
            ensure!(
                matches!(language, "c17" | "cuda") || languages.insert(language),
                "duplicate emitted language"
            );
            languages.insert(language);
            let bytes = read_file(root, &Path::new("emit").join(&a.path), &a.sha256)?;
            ensure!(bytes.len() as u64 == a.bytes, "artifact length mismatch");
        }
        if languages.contains("c17") {
            for path in [
                "c17/model.h",
                "c17/model_core.rs",
                "c17/model_ffi.rs",
                "c17/manifest.json",
            ] {
                ensure!(
                    members.contains(&("c17", path)),
                    "incomplete C17 artifact set"
                );
            }
        }
        if languages.contains("cuda") {
            for path in [
                "cuda/CMakeLists.txt",
                "cuda/manifest.json",
                "cuda/model.cu",
                "cuda/model.hpp",
                "cuda/model_core.rs",
                "cuda/model_native.rs",
            ] {
                ensure!(
                    members.contains(&("cuda", path)),
                    "incomplete CUDA artifact set"
                );
            }
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
