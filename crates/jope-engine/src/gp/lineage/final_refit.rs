use super::*;

/// The explicit selected-to-final-refit transformation.
#[derive(Clone, Debug, Serialize)]
pub struct FinalRefit {
    pub transformation: &'static str,
    pub predictor: PathBuf,
    pub predictor_sha256: String,
    pub fit_key: String,
    pub model_sha256: String,
}

fn native_receipt(
    root: &Path,
    reference: &Value,
) -> Result<crate::eval_registry::CompletionReceipt> {
    let bytes = crate::label_admission::artifact_bytes(root, reference)?;
    crate::eval_registry::verify_completion_receipt(&bytes)
}

/// Link an exported predictor to the selected subject. The only final-fit scope
/// the fitted-view export implements is the selected cut's training rows, so a
/// valid final refit carries the selected fold fit's exact fitted identity and
/// serves exactly the model that fit retained.
/// Anything else is a different subject requiring its own qualification.
pub fn verify_final_refit(
    root: &Path,
    selected: &SelectedLineage,
    predictor: &Path,
) -> Result<FinalRefit> {
    verify_final_refit_bytes(root, selected, predictor, &std::fs::read(predictor)?)
}

/// [`verify_final_refit`] of the exact predictor bytes a caller already read
/// and bound (W1 route R8): the verified predictor is the recorded one.
pub fn verify_final_refit_bytes(
    root: &Path,
    selected: &SelectedLineage,
    predictor: &Path,
    predictor_bytes: &[u8],
) -> Result<FinalRefit> {
    let root = root.canonicalize()?;
    let chosen = selected
        .selected
        .as_ref()
        .context("no selected subject to refit")?;
    let winner = selected
        .selection
        .replay
        .winner
        .as_ref()
        .context("no selected subject to refit")?;
    let exported: Predictor = serde_json::from_slice(predictor_bytes)?;
    ensure!(
        exported.schema == "parallel.predictor.v1"
            && exported.task == "regression"
            && hash(exported.canonical.as_bytes()) == chosen.chain.executable_sha256
            && exported.canonical == feat_math::GenomeIr::parse(&winner.genome)?.canonical(),
        "final refit is not the selected executable recipe"
    );
    ensure!(
        exported.training_sha256 == chosen.training_sha256,
        "final refit declares another training scope"
    );
    let fold = native_receipt(&root, &chosen.fold_fit_completion)?;
    let reference = exported
        .fit_completion
        .as_ref()
        .context("final refit lacks its original fit completion")?;
    let path = Path::new(reference["path"].as_str().context("final fit path")?);
    // Route R8: the completion verified is the one whose digest the predictor records.
    let bytes =
        crate::manifest::read_bound(path, &reference["sha256"], "final refit completion changed")?;
    let refit = crate::eval_registry::verify_completion_receipt(&bytes)?;
    ensure!(
        refit.fit_key == fold.fit_key
            && refit.model_sha256 == fold.model_sha256
            && refit.program_sha256 == fold.program_sha256
            && refit.training_sha256 == fold.training_sha256,
        "final refit fitted state is not the selected fold fit"
    );
    let model_sha256 = refit.model_sha256.context("final refit model absent")?;
    // The model parameters the predictor serves, taken from its verified
    // bytes, are the model that completion retained (review of #111).
    ensure!(
        retained_model_sha256(predictor_bytes)? == model_sha256,
        "final refit model parameters are not the verified completion's model"
    );
    Ok(FinalRefit {
        transformation: "selected_cut_training_refit_reuses_fold_fit",
        predictor: predictor.to_path_buf(),
        predictor_sha256: hash(predictor_bytes),
        fit_key: refit.fit_key,
        model_sha256,
    })
}

/// The fit authority's identity of the model carried in `bytes`, by its own
/// definition of a completion's `model_sha256` (`eval_registry::service`): the
/// SHA-256 of the `serde_json::to_vec` encoding of the `model` member of the
/// bytes parsed as a JSON value. Never a re-serialization of the typed model.
fn retained_model_sha256(bytes: &[u8]) -> Result<String> {
    let carrier: Value = serde_json::from_slice(bytes)?;
    Ok(hash(&serde_json::to_vec(&carrier["model"])?))
}
