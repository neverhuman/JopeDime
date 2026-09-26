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
/// W1 follow-up 2d. The pinned manifest is read once and its bytes must be the
/// pin. The committed fitting-only staging protocol's replay verifier then
/// admits the closure before any split membership or projected input is
/// opened. It refuses the historical closure, its projection source and every
/// object that the historical staging path produced. Returns the consumed bytes
/// and the closure receipt.
fn admit_closure(pin: &Artifact) -> Result<(Vec<u8>, Value)> {
    ensure!(
        fs::symlink_metadata(&pin.path)?.file_type().is_file(),
        "fitting input manifest must be a regular file"
    );
    let bytes = fs::read(&pin.path)?;
    ensure!(
        digest(&bytes) == pin.sha256,
        "fitting input manifest changed"
    );
    let (_, protocol) = jope_datasets::staging_disposition::committed()?;
    let mut receipt = jope_datasets::staging_disposition::verify_closure(&protocol, &bytes)?;
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
    panel: &Panel,
    panel_sha: &str,
    config_sha: &str,
    config: &super::super::optimize::Configuration,
) -> Result<(Vec<Dataset>, Value)> {
    let (bytes, closure) = admit_closure(pin)?;
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
    let expected = panel
        .datasets
        .iter()
        .filter(|d| d.task == jope_datasets::catalog::Task::Regression)
        .map(|d| d.id.as_str())
        .collect::<std::collections::BTreeSet<_>>();
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
mod tests {
    use super::*;
    #[test]
    fn membership_excludes_outer_and_sealed_without_target_fields() {
        let value = serde_json::json!({"protocol":"p","seed":0,"source_sha256":"s","rows":[{"id":"outer","group":"a","source_index":0,"fold":0},{"id":"fit","group":"b","source_index":1,"fold":1},{"id":"sealed","group":"c","source_index":2,"fold":null}]});
        let assignments: Assignments = serde_json::from_value(value.clone()).unwrap();
        assert_eq!(fitting_ids(&assignments), vec!["fit"]);
        let mut injected = value;
        injected["rows"][0]["target"] = json!(123);
        assert!(serde_json::from_value::<Assignments>(injected).is_err());
    }

    use jope_datasets::staging_disposition::committed;

    fn pinned(dir: &Path, name: &str, bytes: &[u8]) -> Artifact {
        let path = dir.join(name);
        fs::write(&path, bytes).unwrap();
        Artifact {
            path,
            sha256: digest(bytes),
        }
    }

    /// A fitting-input manifest naming `source`, one projected input per raw
    /// digest, and the inner folds on the first. No named object exists.
    fn closure(source: &str, raw: &[&str], folds: &[&str]) -> Vec<u8> {
        let inputs = raw
            .iter()
            .enumerate()
            .map(|(i, sha)| {
                let gp = if i == 0 {
                    json!({"folds":folds.iter().map(|f| json!([{"path":"/absent/fold.json","sha256":f},[]])).collect::<Vec<_>>()})
                } else {
                    Value::Null
                };
                json!({"id":format!("d{i}"),"raw":{"path":format!("/absent/raw-{i}.json"),"sha256":sha},"gp":gp})
            })
            .collect::<Vec<_>>();
        serde_json::to_vec(&json!({"schema":"jope.inference-fitting-inputs/v1",
            "projection_source_sha":source,"inputs":inputs}))
        .unwrap()
    }

    /// W1 follow-up 2d at the planner's receiving path.
    #[test]
    fn the_planner_admits_only_closures_outside_the_historical_staging_path() {
        let (d, p) = committed().unwrap();
        let temp = tempfile::tempdir().unwrap();
        let dir = temp.path();
        let source = "0123456789abcdef0123456789abcdef01234567";
        let (r0, r1, f0) = (
            digest(b"fresh raw 0"),
            digest(b"fresh raw 1"),
            digest(b"fresh fold 0"),
        );
        // The historical member with inner folds (197_cpu_act).
        let member = &d.input_closure.members[0];
        assert!(!member.gp_fold_sha256.is_empty());

        // Positive control: a fresh closure is admitted, from the pinned bytes.
        let bytes = closure(source, &[&r0, &r1], &[&f0]);
        let pin = pinned(dir, "fresh.json", &bytes);
        let (consumed, receipt) = admit_closure(&pin).unwrap();
        assert_eq!(consumed, bytes);
        assert_eq!(receipt["objects_checked"], 3);
        assert_eq!(receipt["projection_source_sha"], source);
        assert_eq!(receipt["staging_protocol_id"], p.protocol_id.as_str());

        // The historical projection source, one historical projected input or
        // one historical inner-fold input: each is refused before any named
        // object is opened (none exists).
        for (name, bytes, expected) in [
            (
                "source",
                closure(&d.study.source_sha, &[&r0, &r1], &[&f0]),
                "historical staging source",
            ),
            (
                "raw",
                closure(source, &[&r0, &member.raw_sha256], &[&f0]),
                "d1: projected fitting input",
            ),
            (
                "fold",
                closure(source, &[&r0, &r1], &[&f0, &member.gp_fold_sha256[2]]),
                "d0: inner fold 1 input",
            ),
        ] {
            let err = admit_closure(&pinned(dir, &format!("{name}.json"), &bytes))
                .unwrap_err()
                .to_string();
            assert!(
                err.contains(expected) && err.contains("historical"),
                "{name}: {err}"
            );
        }

        // Bytes replaced after pinning are never consumed.
        fs::write(&pin.path, closure(source, &[&r1, &r0], &[&f0])).unwrap();
        let err = admit_closure(&pin).unwrap_err().to_string();
        assert!(err.contains("manifest changed"), "{err}");
        // A symlink is refused even when it resolves to the pinned bytes.
        let target = pinned(dir, "target.json", &bytes);
        let link = dir.join("link.json");
        std::os::unix::fs::symlink(&target.path, &link).unwrap();
        let err = admit_closure(&Artifact {
            path: link,
            sha256: target.sha256,
        })
        .unwrap_err()
        .to_string();
        assert!(err.contains("regular file"), "{err}");
    }

    /// W1 follow-up 2d at plan admission (`read`: run, cold and v3 report).
    #[test]
    fn a_plan_pinning_any_historical_object_is_never_admitted() {
        let (d, _) = committed().unwrap();
        let fresh = |s: &str| Artifact {
            path: PathBuf::from(format!("/absent/{s}")),
            sha256: digest(s.as_bytes()),
        };
        let inputs = ["a", "b"]
            .map(|id| Dataset {
                id: id.into(),
                raw: fresh(&format!("raw-{id}")),
                gp: None,
            })
            .to_vec();
        // Positive control.
        assert_eq!(
            refuse_historical_plan(&fresh("manifest"), &inputs).unwrap(),
            3
        );
        // The retained historical manifest (the local v3 plan built on it).
        let retained = Artifact {
            path: PathBuf::from("/absent/retained-fitting-inputs.json"),
            sha256: d.input_closure.manifest_sha256.clone(),
        };
        let err = refuse_historical_plan(&retained, &inputs)
            .unwrap_err()
            .to_string();
        assert!(err.contains("fitting-input manifest"), "{err}");
        // A fresh manifest re-pinning one historical projected input.
        let mut reused = inputs.clone();
        reused[1].raw.sha256 = d.input_closure.members[3].raw_sha256.clone();
        let err = refuse_historical_plan(&fresh("manifest"), &reused)
            .unwrap_err()
            .to_string();
        assert!(err.contains("b: projected fitting input"), "{err}");
    }
}
