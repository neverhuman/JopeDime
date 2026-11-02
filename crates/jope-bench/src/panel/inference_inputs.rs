//! Validate already-projected fitting artifacts without opening development or sealed tables.
use super::*;
use feat_math::advanced_v3::{ColumnSpec, DataSchema, RowSemantics};
use jope_pipeline::{
    experiment::{ColumnKind, Panel},
    prepared_data::{DevelopmentData, PreparedRow},
};
use jope_policy::context_producer::Feature;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FittingManifest {
    schema: String,
    panel_sha256: String,
    configuration_sha256: String,
    projection_source_sha: String,
    seed: u64,
    assignments: BTreeMap<String, Artifact>,
    inputs: Vec<Dataset>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Assignments {
    protocol: String,
    seed: u64,
    source_sha256: String,
    rows: Vec<Assignment>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Assignment {
    id: String,
    group: String,
    source_index: usize,
    fold: Option<u8>,
}
fn fitting_ids(assignments: &Assignments) -> Vec<&str> {
    assignments
        .rows
        .iter()
        .filter(|r| matches!(r.fold, Some(1..=4)))
        .map(|r| r.id.as_str())
        .collect()
}
/// W1 follow-up 2d and W1-T07. The pinned manifest is read once and its bytes
/// must be the pin. The committed fitting-only staging protocol's replay
/// verifier then admits the closure before any split membership or projected
/// input is opened. It refuses the historical closure, its projection source
/// and every object that the historical staging path produced. The closure is
/// admitted only with the producer's corrected-run attestation for exactly
/// those bytes (`verify_corrected_run`): a run for other bytes, under another
/// protocol, in the historical window or without a fitting-only declaration is
/// refused. Returns the consumed bytes and the closure receipt.
fn admit_closure(pin: &Artifact, run: Option<&Artifact>) -> Result<(Vec<u8>, Value)> {
    use jope_datasets::staging_disposition::{committed, verify_closure, verify_corrected_run};
    ensure!(
        fs::symlink_metadata(&pin.path)?.file_type().is_file(),
        "fitting input manifest must be a regular file"
    );
    let bytes = fs::read(&pin.path)?;
    ensure!(
        digest(&bytes) == pin.sha256,
        "fitting input manifest changed"
    );
    let (disposition, protocol) = committed()?;
    let mut receipt = verify_closure(&protocol, &bytes)?;
    let run = run.context(
        "the fitting-input closure lacks the producer's corrected-run attestation \
         (jope-datasets staging-disposition attest)",
    )?;
    ensure!(
        fs::symlink_metadata(&run.path)?.file_type().is_file(),
        "corrected-run attestation must be a regular file"
    );
    let run_bytes = fs::read(&run.path)?;
    ensure!(
        digest(&run_bytes) == run.sha256,
        "corrected-run attestation changed"
    );
    receipt["corrected_run"] = verify_corrected_run(&disposition, &protocol, &run_bytes, &bytes)?;
    receipt["staging_protocol_id"] = json!(protocol.protocol_id);
    Ok((bytes, receipt))
}

/// W1 follow-up 2d: whichever planner built a plan, a plan pinning any object
/// of the historical staging closure is never admitted.
pub fn refuse_historical_plan(fitting_manifest: &Artifact, inputs: &[Dataset]) -> Result<usize> {
    let (_, protocol) = jope_datasets::staging_disposition::committed()?;
    let mut objects = vec![(
        "fitting-input manifest".to_string(),
        fitting_manifest.sha256.as_str(),
    )];
    for d in inputs {
        objects.push((
            format!("{}: projected fitting input", d.id),
            d.raw.sha256.as_str(),
        ));
        for (k, (fold, _)) in d.gp.iter().flat_map(|g| g.folds.iter()).enumerate() {
            objects.push((
                format!("{}: inner fold {k} input", d.id),
                fold.sha256.as_str(),
            ));
        }
    }
    jope_datasets::staging_disposition::refuse_historical_objects(&protocol, objects)
}

pub fn load(
    pin: &Artifact,
    corrected_run: &Artifact,
    panel: &Panel,
    panel_sha: &str,
    config_sha: &str,
    config: &super::super::optimize::Configuration,
) -> Result<(Vec<Dataset>, Value)> {
    let (bytes, closure) = admit_closure(pin, Some(corrected_run))?;
    // W1-T01: the ten fitting inputs are the committed regression release's
    // members, and each context-producer input names its member's identities.
    let release = jope_datasets::regression_release::committed()?;
    ensure!(
        release.derived_from.panel_sha256 == panel_sha,
        "fitting inputs' panel is not the committed regression release's panel"
    );
    let manifest: FittingManifest = serde_json::from_slice(&bytes)?;
    ensure!(
        manifest.schema == "jope.inference-fitting-inputs/v1"
            && manifest.panel_sha256 == panel_sha
            && manifest.configuration_sha256 == config_sha
            && manifest.seed == 0
            && manifest.projection_source_sha.len() == 40
            && manifest
                .projection_source_sha
                .bytes()
                .all(|b| b.is_ascii_hexdigit()),
        "fitting manifest identity differs"
    );
    let expected = release
        .members
        .iter()
        .map(|m| m.dataset.as_str())
        .collect::<std::collections::BTreeSet<_>>();
    ensure!(
        panel
            .datasets
            .iter()
            .filter(|d| d.task == jope_datasets::catalog::Task::Regression)
            .map(|d| d.id.as_str())
            .collect::<std::collections::BTreeSet<_>>()
            == expected,
        "panel regression datasets differ from the committed regression release"
    );
    ensure!(
        expected.len() == 10
            && manifest.inputs.len() == 10
            && manifest.assignments.len() == 10
            && manifest
                .inputs
                .iter()
                .map(|d| d.id.as_str())
                .collect::<std::collections::BTreeSet<_>>()
                == expected,
        "fitting dataset inventory differs"
    );
    for entry in &manifest.inputs {
        let dataset = panel
            .datasets
            .iter()
            .find(|d| d.id == entry.id)
            .context("panel dataset")?;
        let assignment_pin = manifest
            .assignments
            .get(&entry.id)
            .context("missing split membership")?;
        assignment_pin.verify()?;
        ensure!(
            assignment_pin.sha256 == dataset.split.sha256,
            "split membership differs from panel"
        );
        let assignments: Assignments = serde_json::from_slice(&fs::read(&assignment_pin.path)?)?;
        ensure!(
            assignments.protocol == dataset.split.protocol
                && assignments.source_sha256 == dataset.source.sha256
                && assignments.seed == panel.seed,
            "split provenance differs"
        );
        entry.raw.verify()?;
        let raw: TrainingInput = serde_json::from_slice(&fs::read(&entry.raw.path)?)?;
        let prepared = raw.prepare()?;
        let ids = raw.rows.iter().map(|r| r.id.as_str()).collect::<Vec<_>>();
        ensure!(
            ids == fitting_ids(&assignments),
            "fitting membership includes missing, reordered, outer or sealed rows"
        );
        ensure!(
            raw.identity.cut == 0
                && raw.identity.dataset_sha256 == dataset.source.sha256
                && raw.identity.prepared_sha256 == dataset.split.development_sha256,
            "fitting source identity differs"
        );
        // W1-T01: the member's identities in the committed release.
        let member = release.member(&entry.id)?;
        ensure!(
            raw.identity.dataset_sha256 == member.raw.sha256
                && raw.identity.prepared_sha256 == member.prepared.development_sha256
                && assignment_pin.sha256 == member.prepared.split_sha256,
            "{}: fitting input differs from its member of regression release {}",
            entry.id,
            release.release_id
        );
        let features = dataset
            .features
            .iter()
            .map(|f| Feature {
                name: f.name.clone(),
                kind: match f.kind {
                    ColumnKind::Numeric => FeatureKind::Numeric,
                    ColumnKind::Categorical => FeatureKind::Categorical,
                },
            })
            .collect::<Vec<_>>();
        ensure!(
            serde_json::to_value(&features)? == serde_json::to_value(&raw.features)?,
            "fitting features differ"
        );
        let needs_gp =
            ["230_machine_cpu", "auto_insurance_price", "197_cpu_act"].contains(&entry.id.as_str());
        ensure!(
            entry.gp.is_some() == needs_gp,
            "paired GP inventory differs"
        );
        if let Some(gp) = &entry.gp {
            gp.pair.arm(true)?;
            let request = gp.pair.proposal.accepted()?;
            let seed =
                super::super::optimize::pair_seed(0, &dataset.source.sha256, config.search_round)?;
            ensure!(
                gp.pair
                    .selection
                    .fitting_rows()
                    .iter()
                    .map(String::as_str)
                    .collect::<Vec<_>>()
                    == ids
                    && gp.pair.training_sha256 == raw.identity.training_sha256
                    && gp.pair.prepared_sha256 == raw.identity.prepared_sha256
                    && gp.pair.preprocessing_sha256
                        == digest(&serde_json::to_vec(&prepared.preprocessing)?)
                    && gp.pair.cut == 0
                    && gp.pair.seconds == 120
                    && gp.pair.max_evaluations == 96
                    && gp.pair.incumbents.is_none()
                    && gp.pair.random_seed == seed
                    && gp.folds.len() == 5,
                "paired input differs from fitting subject/budget"
            );
            ensure!(
                request.challenger.is_none()
                    && request.checkpoint_sha256 == jope_policy::ACCEPTED_STATE_SHA256
                    && request.policy_safetensors_sha256 == config.policy_weights.sha256
                    && request.vocabulary_sha256 == config.vocabulary.sha256
                    && request.sampling_seed == seed
                    && request.fallback_seed == seed.wrapping_add(1)
                    && request.exploration_seed == seed.wrapping_add(2)
                    && request.sampling_rounds == config.sampling_rounds
                    && request.sampling_batch == config.sampling_batch
                    && request.sampling_temperature == config.sampling_temperature,
                "paired producer configuration differs"
            );
            let assignment_map = assignments
                .rows
                .iter()
                .map(|r| (r.id.as_str(), r))
                .collect::<BTreeMap<_, _>>();
            let rows = raw
                .rows
                .iter()
                .map(|r| {
                    let a = assignment_map[r.id.as_str()];
                    PreparedRow {
                        id: r.id.clone(),
                        source_index: a.source_index,
                        group: a.group.clone(),
                        fold: a.fold.unwrap(),
                        values: r.values.clone(),
                        target: r.label.clone(),
                    }
                })
                .collect::<Vec<_>>();
            let data = DevelopmentData {
                dataset: dataset.clone(),
                schema: DataSchema {
                    columns: dataset
                        .features
                        .iter()
                        .map(|f| match f.kind {
                            ColumnKind::Numeric => ColumnSpec::numeric(&f.name),
                            ColumnKind::Categorical => ColumnSpec::categorical(&f.name),
                        })
                        .collect(),
                    row_semantics: RowSemantics::Iid,
                },
                rows,
                // Projected fitting rows, not a development partition: the
                // release admission is the member check above.
                release: None,
            };
            let grouped = jope_pipeline::search_plan::grouped_inner_folds(
                &data
                    .rows
                    .iter()
                    .map(|r| (r.id.clone(), r.group.clone()))
                    .collect::<Vec<_>>(),
            )?;
            for (fold, (artifact, labels)) in gp.folds.iter().enumerate() {
                let (fit_ids, predict_ids) = gp.pair.selection.inner_rows(fold)?;
                ensure!(
                    predict_ids == grouped[fold],
                    "inner grouped membership differs"
                );
                let fit = data
                    .rows
                    .iter()
                    .filter(|r| fit_ids.contains(&r.id))
                    .collect::<Vec<_>>();
                let predict = data
                    .rows
                    .iter()
                    .filter(|r| predict_ids.contains(&r.id))
                    .collect::<Vec<_>>();
                let expected = super::super::optimize_fit::fold_input(&data, &fit, &predict)?;
                artifact.verify()?;
                let actual: super::super::synthesized::SynthesizedFoldInput =
                    serde_json::from_slice(&fs::read(&artifact.path)?)?;
                ensure!(
                    serde_json::to_value(&actual)? == serde_json::to_value(&expected)?
                        && *labels == predict.iter().map(|r| r.target.clone()).collect::<Vec<_>>(),
                    "inner fold data or labels differ from fitting-only projection"
                );
            }
        }
    }
    let mut inputs = manifest.inputs;
    inputs.sort_by(|a, b| a.id.cmp(&b.id));
    Ok((inputs, closure))
}
#[cfg(test)]
mod tests;
