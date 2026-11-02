use super::*;

/// Explicit opt-in publication result. `expected` must be retained in the selected
/// program/run receipt; reacquiring it from mutable files loses the trust anchor.
pub struct EmittedV3 {
    pub expected: feat_math::artifact_manifest::ExpectedArtifactV3,
    pub manifest: feat_math::artifact_manifest::ArtifactManifestV3,
}

/// New target-complete writer; old `emit`/`emit_with_training` remain diagnostic
/// legacy routes. The independent requests are bound before disk publication.
pub fn emit_v3(
    genome: &str,
    requested: &[feat_math::artifact_manifest::Target],
    directory: &Path,
    rows: Option<&[Vec<f64>]>,
) -> Result<EmittedV3> {
    use feat_math::artifact_manifest::Target;
    let parsed = GenomeIr::parse(genome.trim()).context("parse v3 selected program")?;
    ensure!(
        rows.is_none() || requested.contains(&Target::Cpp17Fitted),
        "training rows require fitted target"
    );
    let mut emission = transpile::emit_targets_v3(&parsed, &EmitOptions::default(), requested)?;
    if requested.contains(&Target::Cpp17Fitted) {
        fitted::add_v3(
            &mut emission,
            &parsed,
            rows.context("fitted target requires explicit training rows")?,
        )?;
    }
    publish_v3(emission, directory)
}

/// Publish already rendered recipe or bound-model sources and verify disk bytes.
pub fn publish_v3(
    emission: feat_math::artifact_manifest::TargetEmission,
    directory: &Path,
) -> Result<EmittedV3> {
    use feat_math::artifact_manifest::{self, Domain, ExpectedArtifactV3, TargetState};
    emission.manifest.validate()?;
    let bound: std::collections::BTreeSet<_> = emission
        .manifest
        .targets
        .iter()
        .flat_map(|record| match &record.state {
            TargetState::Supported { files, .. } => {
                files.iter().map(|f| f.path.clone()).collect::<Vec<_>>()
            }
            TargetState::Unsupported { .. } => vec![],
        })
        .collect();
    ensure!(
        bound == emission.sources.keys().cloned().collect(),
        "source set disagrees with expected target closure"
    );
    let manifest_bytes = emission.manifest.envelope_bytes(Domain::Native)?;
    let expected = ExpectedArtifactV3::from_emission(&manifest_bytes, &emission.manifest);
    std::fs::create_dir_all(directory)?;
    ensure!(
        !std::fs::symlink_metadata(directory)?
            .file_type()
            .is_symlink(),
        "symlink output directory"
    );
    for (path, source) in &emission.sources {
        let destination = directory.join(path);
        if let Some(parent) = destination.parent() {
            std::fs::create_dir_all(parent)?;
            ensure!(
                !std::fs::symlink_metadata(parent)?.file_type().is_symlink(),
                "symlink artifact parent"
            );
        }
        write_v3_file(&destination, source.as_bytes())?;
    }
    write_v3_file(&directory.join("manifest.json"), &manifest_bytes)?;
    let manifest = artifact_manifest::verify_directory(directory, "manifest.json", &expected)?;
    Ok(EmittedV3 { expected, manifest })
}
fn write_v3_file(path: &Path, bytes: &[u8]) -> Result<()> {
    // create_new prevents following/replacing a prior file or symlink. A failed
    // publication is retained for diagnosis and requires a fresh output directory.
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| format!("create v3 artifact {}", path.display()))?;
    file.write_all(bytes)?;
    Ok(())
}

/// Consumption always requires the external root and independent request set.
pub fn read_identity_v3(
    directory: &Path,
    expected: &feat_math::artifact_manifest::ExpectedArtifactV3,
) -> Result<feat_math::artifact_manifest::ArtifactManifestV3> {
    feat_math::artifact_manifest::verify_directory(directory, "manifest.json", expected)
}

/// Task-local I09 adapter: fitted predictor with explicit requested target and profile.
/// Writes exact source, fitted codec and manifest; no native policy export is implied.
pub fn emit_numeric_v2(
    genome: &str,
    profile: feat_math::numeric_v2::NumericProfileV2,
    x: &feat_math::Rows,
    y: &[f64],
    options: feat_math::numeric_v2_emit::OptionsV2,
    directory: &Path,
) -> Result<feat_math::numeric_v2_emit::FittedArtifactV2> {
    use feat_math::{
        numeric_v2::NumericModelV2,
        numeric_v2_emit::{self, TargetV2},
    };
    let parsed = GenomeIr::parse(genome)?;
    let model = NumericModelV2::fit(profile, &parsed, x, y)?;
    let artifact = numeric_v2_emit::emit(&model, options)?;
    let filename = match artifact.options.target {
        TargetV2::Python => "model.py",
        TargetV2::Rust => "model.rs",
    };
    std::fs::create_dir_all(directory)?;
    std::fs::write(directory.join(filename), &artifact.source)?;
    std::fs::write(directory.join("model.codec.json"), &artifact.codec)?;
    std::fs::write(
        directory.join("numeric-manifest.json"),
        serde_json::to_vec_pretty(&artifact)?,
    )?;
    Ok(artifact)
}

/// Reload and verify exactly the persisted source and fitted codec before use.
pub fn reload_numeric_v2(
    directory: &Path,
    expected: &feat_math::numeric_v2_emit::OptionsV2,
    model_digest: &str,
) -> Result<feat_math::numeric_v2::NumericModelV2> {
    use feat_math::numeric_v2_emit::{FittedArtifactV2, TargetV2};
    let artifact: FittedArtifactV2 =
        serde_json::from_slice(&std::fs::read(directory.join("numeric-manifest.json"))?)?;
    let filename = match expected.target {
        TargetV2::Python => "model.py",
        TargetV2::Rust => "model.rs",
    };
    ensure!(
        std::fs::read(directory.join(filename))? == artifact.source.as_bytes(),
        "shipped source mismatch"
    );
    ensure!(
        std::fs::read(directory.join("model.codec.json"))? == artifact.codec.as_bytes(),
        "shipped codec mismatch"
    );
    artifact.reload(expected, model_digest)
}
