/// Timing-study entry point. It reuses the production population and selection
/// algorithm but is NOT a producer-admission proof. Callers must retain their
/// distinct benchmark envelope; this cannot establish teaching or promotion.
#[cfg(feature = "timing-study")]
pub fn timing_arm<B: Backend, V: Variation>(
    backend: &mut B,
    plan: &PairInputPlan,
    variation: &mut V,
    proposal: Option<&jope_policy::seeds::Proposal>,
) -> Result<SearchReceipt> {
    plan.validate()?;
    ensure!(
        plan.incumbents.is_none(),
        "timing baseline excludes incumbents"
    );
    if let Some(proposal) = proposal {
        let arm = plan.arm(true)?;
        let deadline = backend.begin_arm(&arm)?;
        let candidates = proposal
            .population()
            .candidates()
            .iter()
            .enumerate()
            .map(|(slot, c)| {
                let origin = match c.origin() {
                    ORIGIN_NEURAL => Origin::NeuralPolicy,
                    ORIGIN_NEURAL_FALLBACK => Origin::NeuralRejectionFallback,
                    ORIGIN_RANDOM => Origin::RandomExploration,
                    _ => anyhow::bail!("unknown proposal origin"),
                };
                Candidate::new(
                    &arm,
                    c.genome().to_owned(),
                    0,
                    slot,
                    origin,
                    vec![],
                    c.rank_log_weight(),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        run_population(backend, plan, variation, arm, deadline, candidates, None)
    } else {
        run_generic(backend, plan, variation)
    }
}
