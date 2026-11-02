use super::*;

/// Two-stage proposal funnel (`K -> B`, e.g. `K=64 -> B=8`) that pre-screens candidate proposals
/// using a `Selector` before running expensive 2-fold CV evaluation and Rust parity compilation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FunnelAllocation {
    /// Identifier for the associated context.
    pub context_id: String,
    /// Selector name recorded for this funnel allocation.
    pub selector_name: String,
    /// Proposal pool k recorded for this funnel allocation.
    pub proposal_pool_k: usize,
    /// Eval budget b recorded for this funnel allocation.
    pub eval_budget_b: usize,
    /// SHA-256 digest identifying the associated selected program.
    pub selected_program_sha256: Vec<String>,
    /// Collection of selected scores values carried by this funnel allocation.
    pub selected_scores: Vec<f64>,
}

/// Performs the pre screen funnel operation and returns an error if it cannot be completed.
pub fn pre_screen_funnel(
    view: &PoolView<'_>,
    selector: &dyn Selector,
    proposal_pool_k: usize,
    eval_budget_b: usize,
) -> Result<FunnelAllocation> {
    ensure!(
        eval_budget_b >= 1 && proposal_pool_k >= eval_budget_b,
        "funnel requires 1 <= eval_budget_b <= proposal_pool_k"
    );
    ensure!(
        !view.programs.is_empty(),
        "cannot pre-screen an empty candidate view"
    );
    let scores = selector.scores(view)?;
    ensure!(
        scores.len() == view.programs.len() && scores.iter().all(|s| s.is_finite()),
        "selector must return one finite score per view program"
    );
    let mut seen = BTreeSet::new();
    let mut ranked: Vec<(f64, &str)> = scores
        .into_iter()
        .zip(view.programs.iter().copied())
        .filter(|(_, prog_sha)| seen.insert(*prog_sha))
        .take(proposal_pool_k)
        .collect();
    let capped_k = ranked.len();
    ranked.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(b.1)));
    ranked.truncate(eval_budget_b);
    Ok(FunnelAllocation {
        context_id: view.context_id.to_owned(),
        selector_name: selector.name().to_owned(),
        proposal_pool_k: capped_k,
        eval_budget_b: ranked.len(),
        selected_program_sha256: ranked.iter().map(|(_, p)| (*p).to_owned()).collect(),
        selected_scores: ranked.iter().map(|(s, _)| *s).collect(),
    })
}
