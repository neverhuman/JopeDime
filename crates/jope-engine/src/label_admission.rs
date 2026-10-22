//! Current native evaluation identity, checked before any program deduplication.
use crate::{
    corpus::ContextRow,
    manifest::*,
    prepare::{Prepared, Preprocessing},
};
use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};
use std::path::Path;

pub const PROTOCOL: &str = "featmath_training_fold_normalized_v1";
pub const RESOURCE: &str = "single_candidate_30s_8gib_v1";
pub const POLICY_SCHEMA: &str = "parallel.label_admission.v2";
pub const LABEL_POLICY: &str = "strict_midrank_v4";
pub const COMPLETION_SCHEMA: &str = "parallel.clean_completion.v3";

pub fn is_sha256(value: &Value) -> bool {
    value.as_str().is_some_and(|s| {
        s.len() == 64
            && s.bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    })
}

/// References are checked against original bytes, never against caller-provided
/// presence flags. Missing/evicted evidence remains unavailable for admission.
pub fn artifact_bytes(root: &Path, reference: &Value) -> Result<Vec<u8>> {
    ensure!(
        is_sha256(&reference["sha256"]),
        "admission malformed artifact digest"
    );
    let path = resolve(
        root,
        Path::new(
            reference["path"]
                .as_str()
                .context("admission artifact path absent")?,
        ),
    )?;
    let bytes = std::fs::read(path)?;
    ensure!(
        hash(&bytes) == reference["sha256"],
        "admission artifact digest changed"
    );
    Ok(bytes)
}

pub fn artifact(root: &Path, reference: &Value) -> Result<Value> {
    Ok(serde_json::from_slice(&artifact_bytes(root, reference)?)?)
}

/// The supervisor retains these original artifacts. A stale result by itself
/// cannot turn an interrupted, failed, or unobserved execution into a label.
pub fn validate_completion(root: &Path, row: &Value) -> Result<()> {
    let completion = artifact(root, &row["clean_completion"])?;
    ensure!(
        completion["schema"] == COMPLETION_SCHEMA,
        "admission original completion absent"
    );
    let attempt = artifact(root, &completion["attempt"])?;
    for key in [
        "evaluation_identity_sha256",
        "program_sha256",
        "evaluator_sha256",
    ] {
        ensure!(
            is_sha256(&row[key]) && completion[key] == row[key] && attempt[key] == row[key],
            "admission attempt/completion identity conflict: {key}"
        );
    }
    let deadline = completion["deadline_seconds"]
        .as_f64()
        .context("admission completion deadline absent")?;
    ensure!(
        deadline.is_finite() && deadline > 0. && deadline <= 30.,
        "admission invalid deadline"
    );
    let exceeded = completion["deadline_exceeded"]
        .as_bool()
        .context("admission deadline outcome absent")?;
    let observed = completion["exit_observed"]
        .as_bool()
        .context("admission observed exit absent")?;
    ensure!(
        completion.get("exit_code").is_some()
            && completion.get("signal").is_some()
            && completion.get("runtime_seconds").is_some(),
        "admission incomplete process outcome"
    );
    let runtime = completion["runtime_seconds"].as_f64();
    ensure!(
        completion["runtime_seconds"].is_null()
            || runtime.is_some_and(|r| r.is_finite() && r >= 0.),
        "admission invalid measured runtime"
    );
    ensure!(
        row.get("runtime_seconds").is_some()
            && row["runtime_seconds"] == completion["runtime_seconds"],
        "admission original runtime conflict"
    );
    let outcome = completion["outcome"]
        .as_str()
        .context("admission execution outcome absent")?;
    let native_bytes = artifact_bytes(root, &completion["fit_completion"])?;
    crate::eval_registry::verify_completion_receipt(&native_bytes)?;
    let native: Value = serde_json::from_slice(&native_bytes)?;
    ensure!(
        row.get("numerical_runtime_seconds").is_some()
            && row["numerical_runtime_seconds"] == native["elapsed_seconds"],
        "admission original numerical runtime conflict"
    );
    for (native_key, row_key) in [
        ("program_sha256", "program_sha256"),
        ("executable_sha256", "evaluator_sha256"),
    ] {
        ensure!(
            is_sha256(&row[row_key]) && native[native_key] == row[row_key],
            "admission native execution identity conflict: {native_key}"
        );
    }
    if row.get("failure").is_some_and(Value::is_null) {
        ensure!(
            outcome == "success"
                && observed
                && completion["exit_code"] == 0
                && completion["signal"].is_null()
                && !exceeded
                && runtime.is_some_and(|r| r <= deadline),
            "admission execution did not complete cleanly"
        );
        ensure!(
            native["outcome"] == "success",
            "admission original model did not fit successfully"
        );
        ensure!(
            is_sha256(&native["model_sha256"])
                && native["model_sha256"] == completion["model"]["sha256"],
            "admission original model identity conflict"
        );
        artifact_bytes(root, &completion["model"])?;
        ensure!(
            artifact(root, &completion["output"])? == row["metrics"],
            "admission original output/metrics conflict"
        );
    } else {
        ensure!(
            row["failure"]
                .as_str()
                .is_some_and(|s| !s.trim().is_empty())
                && row["metrics"].is_null(),
            "admission failed execution carries quality or lacks explicit failure"
        );
        match outcome {
            "fit_failure" => ensure!(
                observed
                    && !exceeded
                    && completion["exit_code"].as_i64().is_some_and(|c| c != 0)
                    && completion["signal"].is_null()
                    && native["outcome"] == "deterministic_failure",
                "admission invalid deterministic failure"
            ),
            "timeout" => ensure!(exceeded, "admission timeout lacks censoring threshold"),
            "infrastructure_failure" | "unknown" => {}
            _ => anyhow::bail!("admission invalid failed outcome"),
        }
    }
    Ok(())
}

pub fn identity(row: &Value) -> Value {
    let keys = [
        "program_sha256",
        "family",
        "partition",
        "prepared_sha256",
        "cut",
        "training_sha256",
        "validation_sha256",
        "evaluator_sha256",
        "evaluator_protocol",
        "preprocessing_sha256",
        "target_scale",
        "metric_units",
        "vocabulary_sha256",
        "feature_view_sha256",
        "evaluation_seed",
        "resource_protocol",
    ];
    let mut value = serde_json::Map::new();
    for key in keys {
        value.insert(key.into(), row[key].clone());
    }
    Value::Object(value)
}

pub fn raw_view(training: &str, preprocessing: &str) -> String {
    hash(
        &serde_json::to_vec(&json!({"protocol":"raw_training_fold_v1",
        "training_sha256":training,"preprocessing_sha256":preprocessing}))
        .unwrap(),
    )
}

pub fn validate_receipt(search: &Value, row: &Value, rows: usize, expected: &Value) -> Result<()> {
    ensure!(
        expected["schema"] == POLICY_SCHEMA && is_sha256(&expected["evaluator_sha256"]),
        "admission frozen policy absent"
    );
    for key in [
        "evaluator_sha256",
        "prepared_sha256",
        "vocabulary_sha256",
        "observations_sha256",
    ] {
        ensure!(
            is_sha256(&search[key]),
            "admission malformed search identity: {key}"
        );
    }
    ensure!(
        search["schema"] == "parallel.search.v1"
            && search["completed_evaluations"] == rows
            && search["completed_at"].as_u64().is_some_and(|t| t > 0)
            && search["final_tests_opened"] == false,
        "admission incomplete or invalid search receipt"
    );
    ensure!(
        search["evaluator_sha256"] == expected["evaluator_sha256"]
            && search["evaluator_protocol"] == expected["evaluator_protocol"]
            && search["prepared_sha256"] == row["prepared_sha256"]
            && search["cut"] == row["cut"]
            && search["vocabulary_sha256"] == row["vocabulary_sha256"],
        "admission search/observation identity conflict"
    );
    Ok(())
}

/// `expected` is an experiment-frozen evaluator manifest, not a value supplied
/// by the observation. Historical observations missing fields are quarantined.
pub fn verify_inputs(root: &Path, context: &ContextRow, prepared: &Prepared) -> Result<String> {
    ensure!(
        matches!(context.partition.as_str(), "meta_train" | "meta_validation")
            && prepared.task == "regression",
        "admission sealed or unsupported task inputs"
    );
    ensure!(
        prepared.dataset == context.dataset
            && prepared.family == context.family
            && prepared.meta_partition == context.partition
            && prepared.clean_evaluation
            && !context.family.trim().is_empty(),
        "admission prepared/context provenance conflict"
    );
    let path = resolve(root, &context.prepared)?;
    // Route R8: the preparation is read once and bound by its digest. The
    // caller's parse (checked above) must be exactly that preparation, and the
    // views are resolved from it; `prepared.json` is never reopened.
    let bytes = read_verified(
        &path.join("prepared.json"),
        &context.prepared_sha256,
        "admission prepared artifact",
    )?;
    let bound: Prepared =
        serde_json::from_slice(&bytes).context("parse admission prepared artifact")?;
    ensure!(
        serde_json::to_value(&bound)? == serde_json::to_value(prepared)?,
        "admission prepared parse differs from the bound preparation bytes"
    );
    let mut training =
        crate::prepare::load_bound_view_in_root(root, &path, &bound, context.cut, true)?;
    let targets: Vec<_> = training
        .iter_mut()
        .map(|row| row.pop().context("admission target absent"))
        .collect::<Result<_>>()?;
    let digest = crate::eval_registry::ordered_training_sha256(&training, &targets)?;
    drop(training);
    crate::prepare::load_bound_view_in_root(root, &path, &bound, context.cut, false)?;
    Ok(digest)
}

fn approved_witness_path(
    root: &Path,
    requested: &Path,
    native: Option<&Path>,
) -> Result<std::path::PathBuf> {
    match native {
        None => resolve(root, requested),
        Some(approved) => {
            let requested = if requested.is_absolute() {
                requested.to_owned()
            } else {
                resolve(root, requested)?
            };
            ensure!(
                requested == approved
                    && std::fs::canonicalize(approved)? == approved
                    && std::fs::symlink_metadata(approved)?.is_file(),
                "external witness must be exact original physical path in verified native grant"
            );
            for ancestor in approved.ancestors() {
                ensure!(
                    !std::fs::symlink_metadata(ancestor)?
                        .file_type()
                        .is_symlink(),
                    "external witness symlink ancestor"
                );
            }
            Ok(approved.to_owned())
        }
    }
}

pub fn approved_authority(
    root: &Path,
    expected: &Value,
) -> Result<crate::eval_registry::Authority> {
    ensure!(
        expected["schema"] == POLICY_SCHEMA && expected["evaluator_protocol"] == PROTOCOL,
        "unknown admission policy"
    );
    for key in [
        "authority_instance",
        "runtime_sha256",
        "numerical_contract_sha256",
        "evaluator_sha256",
    ] {
        ensure!(
            is_sha256(&expected[key]),
            "admission approved numerical/authority identity absent: {key}"
        );
    }
    let evaluator = resolve(
        root,
        Path::new(
            expected["evaluator"]
                .as_str()
                .context("approved evaluator path absent")?,
        ),
    )?;
    ensure!(
        file_hash(&evaluator)? == expected["evaluator_sha256"],
        "admission approved evaluator changed"
    );
    let authority_root = resolve(
        root,
        Path::new(
            expected["authority_root"]
                .as_str()
                .context("approved authority root absent")?,
        ),
    )?;
    let witness_path = Path::new(
        expected["authority_witness"]
            .as_str()
            .context("approved authority witness absent")?,
    );
    let native = if let Some(binding) = expected.get("native_fit_policy").filter(|v| !v.is_null()) {
        let path = Path::new(
            binding["path"]
                .as_str()
                .context("native label policy path absent")?,
        );
        let original = crate::eval_registry::native_scq::policy::protected(path)?;
        ensure!(
            crate::manifest::hash(&original) == binding["sha256"],
            "native label policy bytes changed"
        );
        let installed = crate::eval_registry::native_scq::policy::InstalledPolicy::load(path)?;
        ensure!(
            crate::eval_registry::native_scq::policy::protected(path)? == original,
            "native label policy replaced during verification"
        );
        let g = &installed.signed.body;
        ensure!(
            g.root == root
                && g.authority_root == authority_root
                && g.authority_instance == expected["authority_instance"]
                && g.evaluator_executable_sha256 == expected["evaluator_sha256"]
                && g.numerical_contract_sha256 == expected["numerical_contract_sha256"]
                && g.allowed_runtime_sha256.contains(
                    expected["runtime_sha256"]
                        .as_str()
                        .context("label runtime absent")?
                )
                && file_hash(&authority_root.join("authority.json"))?
                    == g.authority_manifest_sha256,
            "native label authority/runtime differs from original signed allocation policy"
        );
        Some(installed)
    } else {
        None
    };
    let witness = approved_witness_path(
        root,
        witness_path,
        native.as_ref().map(|p| p.signed.body.witness.as_path()),
    )?;
    let authority = crate::eval_registry::Authority::open(&authority_root, &witness)?;
    ensure!(
        authority.instance() == expected["authority_instance"],
        "admission authority instance changed"
    );
    Ok(authority)
}

pub fn validate(
    root: &Path,
    row: &Value,
    context: &ContextRow,
    prepared: &Prepared,
    vocabulary_sha: &str,
    expected: &Value,
) -> Result<()> {
    let training = verify_inputs(root, context, prepared)?;
    let mut authority = approved_authority(root, expected)?;
    validate_preverified(
        root,
        row,
        context,
        prepared,
        crate::label_admission::AdmissionInputs {
            vocabulary_sha,
            expected,
            fit_training: &training,
        },
        &mut authority,
    )
}

/// Frozen admission policy and preverified training/vocabulary identities.
#[derive(Clone, Copy)]
pub struct AdmissionInputs<'a> {
    pub vocabulary_sha: &'a str,
    pub expected: &'a Value,
    pub fit_training: &'a str,
}

/// The publication owner verifies each immutable cut once, outside label scans.
pub(crate) fn validate_preverified(
    root: &Path,
    row: &Value,
    context: &ContextRow,
    prepared: &Prepared,
    inputs: AdmissionInputs<'_>,
    authority: &mut crate::eval_registry::Authority,
) -> Result<()> {
    let AdmissionInputs {
        vocabulary_sha,
        expected,
        fit_training,
    } = inputs;

    ensure!(
        expected["schema"] == POLICY_SCHEMA && expected["evaluator_protocol"] == PROTOCOL,
        "unknown admission policy"
    );
    let cut = prepared
        .cuts
        .iter()
        .find(|cut| cut.index == context.cut)
        .context("admission cut absent")?;
    ensure!(
        prepared.task == "regression" && row["task"] == "regression",
        "admission non-regression protocol unavailable"
    );
    for key in [
        "program_sha256",
        "prepared_sha256",
        "training_sha256",
        "validation_sha256",
        "evaluator_sha256",
        "preprocessing_sha256",
        "vocabulary_sha256",
        "feature_view_sha256",
        "evaluation_identity_sha256",
    ] {
        ensure!(is_sha256(&row[key]), "admission malformed identity: {key}");
    }
    ensure!(
        context.partition != "meta_test"
            && row["family"] == context.family
            && row["partition"] == context.partition
            && row["dataset"] == context.dataset
            && row["prepared_sha256"] == context.prepared_sha256
            && row["cut"] == context.cut
            && row["training_sha256"] == cut.training_sha256
            && row["validation_sha256"] == cut.validation_sha256,
        "admission family/prepared/cut conflict"
    );
    ensure!(
        row["evaluator_sha256"] == expected["evaluator_sha256"]
            && is_sha256(&expected["evaluator_sha256"])
            && row["evaluator_protocol"] == PROTOCOL
            && row["resource_protocol"] == RESOURCE
            && row["vocabulary_sha256"] == vocabulary_sha
            && row["metric_units"] == "cut_normalized_target"
            && row["evaluation_seed"] == 0,
        "admission evaluator/units/vocabulary/resource conflict"
    );
    let path =
        resolve(root, &context.prepared)?.join(format!("cut-{}/preprocessing.json", context.cut));
    // Route R8: the scale is read from exactly the verified preparation state.
    let preprocessing: Preprocessing = serde_json::from_slice(&read_verified(
        &path,
        &cut.preprocessing_sha256,
        "admission preprocessing",
    )?)
    .context("parse admission preprocessing")?;
    ensure!(
        row["preprocessing_sha256"] == cut.preprocessing_sha256
            && row["target_scale"].as_f64() == Some(preprocessing.target_scale)
            && preprocessing.target_scale.is_finite()
            && preprocessing.target_scale > 0.
            && row["feature_view_sha256"]
                == raw_view(&cut.training_sha256, &cut.preprocessing_sha256),
        "admission preprocessing/scale/feature-view conflict"
    );
    let genome = row["genome"].as_str().context("admission program absent")?;
    ensure!(
        feat_math::GenomeIr::parse(genome)?.canonical() == genome
            && row["program_sha256"] == hash(genome.as_bytes()),
        "admission canonical program conflict"
    );
    ensure!(
        row["evaluation_identity_sha256"] == hash(&serde_json::to_vec(&identity(row))?),
        "admission evaluation identity changed"
    );
    validate_completion(root, row)?;
    let completion = artifact(root, &row["clean_completion"])?;
    let committed = authority.verify_committed_completion(
        completion["fit_completion"]["sha256"]
            .as_str()
            .context("original authority completion digest absent")?,
    )?;
    let native = artifact(root, &completion["fit_completion"])?;
    ensure!(
        serde_json::to_value(&committed)? == native,
        "admission copied receipt differs from original authority"
    );
    let fit = &native["fit_identity"];
    ensure!(
        fit["training_sha256"] == fit_training
            && fit["preprocessing_sha256"] == row["preprocessing_sha256"]
            && fit["feature_view_sha256"] == row["feature_view_sha256"]
            && fit["fit_seed"] == row["evaluation_seed"]
            && fit["runtime_sha256"] == expected["runtime_sha256"]
            && fit["numerical_contract_sha256"] == expected["numerical_contract_sha256"],
        "admission original numerical input/preprocessing/view/seed conflict"
    );
    let outcome = row
        .get("failure")
        .context("admission explicit outcome absent")?;
    if outcome.is_null() {
        ensure!(
            matches!(
                row["evaluation_origin"].as_str(),
                Some("execution" | "reuse")
            ) && row["evaluation_origin"] == row["metrics"]["evaluation_origin"],
            "admission evaluation origin absent or inconsistent"
        );
        ensure!(
            row["metrics"]["units"] == row["metric_units"],
            "admission metric units/outcome conflict"
        );
        let task = row["task"].as_str().unwrap_or("regression");
        crate::task_contract::require_task(task)?;
        let primary = crate::task_contract::observation_primary(&row["metrics"])?;
        if task == "regression" {
            ensure!(
                row["metrics"]["rmse"].as_f64() == Some(primary)
                    && row["metric_units"] == "cut_normalized_target",
                "admission metric units/outcome conflict"
            );
        } else {
            ensure!(
                row["metrics"]["primary_metric"] == crate::task_contract::primary_metric(task)?
                    && row["metric_units"] == crate::task_contract::cut_units(task)?,
                "admission metric units/outcome conflict"
            );
        }
    } else {
        ensure!(
            outcome
                .as_str()
                .is_some_and(|reason| !reason.trim().is_empty()),
            "invalid failed outcome"
        );
    }
    Ok(())
}

#[cfg(test)]
mod native_path_tests {
    use super::*;
    #[test]
    fn external_witness_requires_exact_verified_native_binding() {
        let t = tempfile::tempdir().unwrap();
        let root = t.path().join("data");
        std::fs::create_dir(&root).unwrap();
        let witness = t.path().join("original-witness");
        std::fs::write(&witness, b"original witness").unwrap();
        assert!(approved_witness_path(&root, &witness, None).is_err());
        assert_eq!(
            approved_witness_path(&root, &witness, Some(&witness)).unwrap(),
            witness
        );
        let other = t.path().join("other");
        std::fs::write(&other, b"original witness").unwrap();
        assert!(approved_witness_path(&root, &other, Some(&witness)).is_err());
        let link = t.path().join("link");
        std::os::unix::fs::symlink(&witness, &link).unwrap();
        assert!(approved_witness_path(&root, &link, Some(&link)).is_err());
        let local = root.join("local");
        std::fs::write(&local, b"old").unwrap();
        assert_eq!(
            approved_witness_path(&root, Path::new("local"), None).unwrap(),
            local
        );
    }
}

/// Pure preflight of a frozen `parallel.label_admission.v2` policy. It inspects only the
/// policy fields, before any evaluator, witness or authority file is opened.
pub fn preflight_policy(expected: &Value) -> Result<()> {
    ensure!(
        expected["schema"] == POLICY_SCHEMA,
        "unknown admission policy schema"
    );
    ensure!(
        expected["evaluator_protocol"] == PROTOCOL,
        "admission evaluator protocol differs"
    );
    for key in [
        "authority_instance",
        "runtime_sha256",
        "numerical_contract_sha256",
        "evaluator_sha256",
    ] {
        ensure!(
            is_sha256(&expected[key]),
            "admission identity malformed: {key}"
        );
    }
    for key in ["evaluator", "authority_root", "authority_witness"] {
        ensure!(
            expected[key].as_str().is_some_and(|s| !s.trim().is_empty()),
            "admission path absent: {key}"
        );
    }
    if let Some(binding) = expected.get("native_fit_policy").filter(|v| !v.is_null()) {
        ensure!(
            binding["path"]
                .as_str()
                .is_some_and(|s| !s.trim().is_empty()),
            "native fit policy path absent"
        );
        ensure!(
            is_sha256(&binding["sha256"]),
            "native fit policy sha256 malformed"
        );
    }
    Ok(())
}

/// Key and length closure of the historical shard census. It deliberately does not claim
/// payload hashes: a census with `all_7877_payload_hashes_verified: false` stays unverified.
pub fn census_closure(census: &Value) -> Result<()> {
    ensure!(
        census["directories_declared"] == 7877,
        "shard directory count differs"
    );
    ensure!(
        census["expected_member_keys"] == 23631,
        "member key count differs"
    );
    ensure!(
        census["located_member_keys"] == census["expected_member_keys"],
        "member keys are missing"
    );
    ensure!(
        census["missing_member_keys"] == 0,
        "missing member keys present"
    );
    Ok(())
}

/// Byte-level check of one historical shard directory. Each member's SHA-256 must equal its
/// expected digest. A missing member or any mismatch refuses. This verifies payload bytes
/// only; it infers no label or training semantics.
pub fn verify_shard_directory_payload_hashes(
    shard_dir: &Path,
    expected_shard_sha256: &str,
    expected_records_sha256: &str,
    expected_labels_sha256: &str,
) -> Result<()> {
    for (member, expected) in [
        ("shard.json", expected_shard_sha256),
        ("records.bin", expected_records_sha256),
        ("labels.jsonl", expected_labels_sha256),
    ] {
        let actual = file_hash(&shard_dir.join(member))
            .with_context(|| format!("shard member absent or unreadable: {member}"))?;
        ensure!(actual == expected, "shard member payload changed: {member}");
    }
    Ok(())
}

/// The census sample payload checks: each sampled member matched its index digest and its
/// metadata matched the parent shard.
pub fn verify_census_payload_samples(census: &Value) -> Result<()> {
    let samples = census["sample_exact_version_payload_verification"]
        .as_array()
        .context("census payload samples absent")?;
    ensure!(!samples.is_empty(), "census payload samples empty");
    for sample in samples {
        ensure!(
            sample["hash_matches"] == true,
            "census sample hash mismatch"
        );
        ensure!(
            sample["manifest_metadata_matches_parent"] == true,
            "census sample metadata mismatch"
        );
    }
    Ok(())
}

/// Promotion gate: refuse unless the census states that every 7,877 shard payload was verified.
pub fn require_full_payload_closure(census: &Value) -> Result<()> {
    ensure!(
        census["all_7877_payload_hashes_verified"] == true,
        "full 7877 payload closure not verified"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sha(c: char) -> String {
        c.to_string().repeat(64)
    }

    fn policy() -> Value {
        json!({
            "schema": POLICY_SCHEMA,
            "evaluator_protocol": PROTOCOL,
            "authority_instance": sha('a'),
            "runtime_sha256": sha('b'),
            "numerical_contract_sha256": sha('c'),
            "evaluator_sha256": sha('d'),
            "evaluator": "evaluator/bin",
            "authority_root": "authority",
            "authority_witness": "witness.json",
            "native_fit_policy": {"path": "native.json", "sha256": sha('e')}
        })
    }

    #[test]
    fn strict_v2_and_native_fit_binding_preflight() {
        preflight_policy(&policy()).unwrap();
        let mut without_native = policy();
        without_native["native_fit_policy"] = Value::Null;
        preflight_policy(&without_native).unwrap();

        type Mutation = (&'static str, fn(&mut Value));
        let negatives: &[Mutation] = &[
            ("schema", |p| {
                p["schema"] = json!("parallel.label_admission.v1")
            }),
            ("protocol", |p| p["evaluator_protocol"] = json!("other")),
            ("instance", |p| p["authority_instance"] = json!("xyz")),
            ("evaluator_sha", |p| {
                p["evaluator_sha256"] = json!(sha('D').to_uppercase())
            }),
            ("witness", |p| p["authority_witness"] = json!("  ")),
            ("native_path", |p| {
                p["native_fit_policy"]["path"] = json!("")
            }),
            ("native_sha", |p| {
                p["native_fit_policy"]["sha256"] = json!("short")
            }),
        ];
        for (name, mutate) in negatives {
            let mut p = policy();
            mutate(&mut p);
            assert!(
                preflight_policy(&p).is_err(),
                "negative case {name} accepted"
            );
        }
    }

    #[test]
    fn historical_census_closure_verified() {
        let closed = json!({
            "directories_declared": 7877,
            "expected_member_keys": 23631,
            "located_member_keys": 23631,
            "missing_member_keys": 0,
            "all_7877_payload_hashes_verified": false
        });
        census_closure(&closed).unwrap();
        let mut missing = closed.clone();
        missing["located_member_keys"] = json!(23630);
        missing["missing_member_keys"] = json!(1);
        assert!(census_closure(&missing).is_err());
        let mut short = closed.clone();
        short["directories_declared"] = json!(7876);
        assert!(census_closure(&short).is_err());
    }

    #[test]
    fn shard_directory_payload_hashes_verified() {
        let dir = tempfile::tempdir().unwrap();
        let shard = dir.path();
        std::fs::write(shard.join("shard.json"), b"{}").unwrap();
        std::fs::write(shard.join("records.bin"), [1u8, 2, 3]).unwrap();
        std::fs::write(shard.join("labels.jsonl"), b"{\"y\":1}\n").unwrap();
        let (s, r, l) = (hash(b"{}"), hash(&[1u8, 2, 3]), hash(b"{\"y\":1}\n"));
        verify_shard_directory_payload_hashes(shard, &s, &r, &l).unwrap();
        std::fs::write(shard.join("records.bin"), [1u8, 2, 4]).unwrap();
        assert!(
            verify_shard_directory_payload_hashes(shard, &s, &r, &l).is_err(),
            "tampered byte accepted"
        );
        std::fs::remove_file(shard.join("labels.jsonl")).unwrap();
        assert!(
            verify_shard_directory_payload_hashes(shard, &s, &r, &l).is_err(),
            "missing member accepted"
        );
    }

    #[test]
    fn census_payload_samples_and_promotion_gate() {
        let good = json!({
            "all_7877_payload_hashes_verified": false,
            "sample_exact_version_payload_verification": [
                {"hash_matches": true, "manifest_metadata_matches_parent": true}
            ]
        });
        verify_census_payload_samples(&good).unwrap();
        let mut bad = good.clone();
        bad["sample_exact_version_payload_verification"][0]["hash_matches"] = json!(false);
        assert!(verify_census_payload_samples(&bad).is_err());
        assert!(
            require_full_payload_closure(&good).is_err(),
            "unverified closure promoted"
        );
    }
}
