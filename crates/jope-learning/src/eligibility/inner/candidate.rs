use super::*;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `InnerEvidence` is a record for the inner evidence contract in this module: Explicit inner-OOF partition and execution binding. Original development folds remain a different protocol; no old receipt is retroactively upgraded.
pub struct InnerEvidence {
    /// SHA-256 digest identifying the associated selection.
    pub selection_sha256: String,
    /// SHA-256 digest identifying the associated frozen recipe.
    pub frozen_recipe_sha256: String,
    /// SHA-256 digest identifying the associated search receipt.
    pub search_receipt_sha256: String,
    /// SHA-256 digest identifying the associated runtime.
    pub runtime_sha256: String,
    /// SHA-256 digest identifying the associated arm window.
    pub arm_window_sha256: String,
    /// SHA-256 digest identifying the associated memory policy.
    pub memory_policy_sha256: String,
    /// SHA-256 digest identifying the associated candidate freeze.
    pub candidate_freeze_sha256: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct CandidateFreeze {
    pub(super) schema: String,
    pub(super) comparison_sha256: String,
    pub(super) frozen_recipe_sha256: String,
    pub(super) search_receipt_sha256: String,
    pub(super) frozen_at_unix_ms: u64,
    pub(super) selection_source: String,
}
/// Called after measured inner selection and before the controller schedules the
/// candidate's outer prediction job. It records ordering; no outcome is invented.
pub fn register_inner_candidate(
    state: &State,
    freeze: &FreezeReceipt,
    frozen_recipe: &str,
    search_receipt: &str,
) -> Result<String> {
    let policy = load_freeze(state, freeze)?;
    ensure!(
        policy.inner_partition.is_some(),
        "not an inner teaching freeze"
    );
    require_blob(state, frozen_recipe)?;
    require_blob(state, search_receipt)?;
    let path = state
        .root()
        .join("eligibility/candidate-freezes")
        .join(&freeze.comparison_sha256)
        .join(format!("{frozen_recipe}.json"));
    let _lock = state.lock("eligibility")?;
    let ledger = Ledger::open(state.root(), "eligibility");
    ledger.require_intact()?;
    if path.exists() {
        let bytes = fs::read(path)?;
        let value: CandidateFreeze = serde_json::from_slice(&bytes)?;
        ensure!(
            value.comparison_sha256 == freeze.comparison_sha256
                && value.frozen_recipe_sha256 == frozen_recipe
                && value.search_receipt_sha256 == search_receipt,
            "candidate freeze changed"
        );
        return state.put_blob(&bytes);
    }
    let value = CandidateFreeze {
        schema: "jope.inner-candidate-freeze/v1".into(),
        comparison_sha256: freeze.comparison_sha256.clone(),
        frozen_recipe_sha256: frozen_recipe.into(),
        search_receipt_sha256: search_receipt.into(),
        frozen_at_unix_ms: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_millis()
            .try_into()?,
        selection_source: "completed-train-inner-oof-before-candidate-outer-scoring".into(),
    };
    let bytes = canonical_json(&value)?;
    let sha = state.put_blob(&bytes)?;
    immutable_write(&path, &bytes)?;
    ledger.append(&[serde_json::json!({"outcome":"candidate_frozen","comparison_sha256":freeze.comparison_sha256,"candidate_freeze_sha256":sha})],serde_json::json!({"source":"bounded-inner-selection-controller"}))?;
    Ok(sha)
}
