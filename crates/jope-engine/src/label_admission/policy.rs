use super::*;

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
