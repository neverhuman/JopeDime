/// W4-T07: the artifact handed to the controller carries every candidate's original tokens,
/// origin and draw, predicted scores of exactly that program, parser identity and explicit
/// adaptation edges, plus reconciled denominators. Pre-fix reproduction (retained evidence
/// w2b-w4-repro-prefix-20260926T071411Z): the envelope carried none of these, and the
/// grammar-uniform fallback and exploration draws left no terminal records.
#[test]
fn producer_artifact_carries_a_reconciled_proposal_ledger() -> Result<()> {
    let (request, proposal) = unbound_proposal();
    check_reconciled_ledger(&request, &proposal)?;
    let d = &proposal.ledger().denominators;
    // First fixture: every neural draw completed, the fallback fill topped the pool up and
    // refused at least one draw.
    assert_eq!((d.scheduled_neural_draws, d.completed), (8, 8));
    assert!(d.fallback_draws > d.fallback_admitted && d.fallback_admitted == 56);
    // Second fixture: the pool filled during sampling, so later valid draws are surplus or
    // duplicates, and no fallback draw ran.
    let (request, proposal) = filled_proposal();
    check_reconciled_ledger(&request, &proposal)?;
    let d = &proposal.ledger().denominators;
    assert_eq!((d.completed, d.fallback_draws), (64, 0));
    assert!(d.surplus > 0 && d.duplicate > 0);
    assert_eq!(d.unique_programs, d.completed + d.surplus);
    Ok(())
}

fn check_reconciled_ledger(request: &Request, proposal: &seeds::Proposal) -> Result<()> {
    use crate::decode::{DrawOutcome, UniformOutcome, EXPLORATION_SIZE, POOL_SIZE};
    use crate::proposal_ledger::{SlotSource, EDGE_DECODE, EDGE_SCORE_ENCODING};
    let bytes = artifact_bytes(request, proposal)?;
    let file = read_claims(&bytes, request)?;
    let ledger = file.ledger().expect("the producer writes a ledger");
    assert_eq!(ledger, proposal.ledger());
    let pool = proposal.pool();
    let d = &ledger.denominators;
    // Every scheduled attempt terminates exactly once and the denominators reconcile.
    assert_eq!(
        d.scheduled_neural_draws,
        ledger.sampling.rounds_run * request.sampling_batch
    );
    assert_eq!(
        d.completed + d.surplus + d.duplicate + d.parser_rejected + d.incomplete,
        d.scheduled_neural_draws
    );
    assert_eq!(
        d.complete_sequences,
        d.scheduled_neural_draws - d.incomplete
    );
    assert_eq!(d.valid_programs, pool.valid);
    assert_eq!(d.completed, pool.unique_neural);
    assert_eq!(d.fallback_draws as u64, pool.fallback_attempts);
    assert_eq!(d.completed + d.fallback_admitted, POOL_SIZE);
    assert_eq!(d.exploration_admitted, EXPLORATION_SIZE);
    assert!(d.exploration_draws >= EXPLORATION_SIZE);
    assert_eq!((d.scored, d.rank_survivors, d.handed_off), (64, 64, 80));
    // No fallback inflation: neural slots come from completed neural draws only.
    let neural = ledger
        .candidates
        .iter()
        .filter(|c| c.origin == seeds::ORIGIN_NEURAL)
        .count();
    assert_eq!(neural, d.completed);
    for c in &ledger.candidates {
        match &c.source {
            SlotSource::NeuralDraw { round, row } => {
                let draw = &ledger.neural_draws[round * request.sampling_batch + row];
                assert_eq!(draw.outcome, DrawOutcome::Completed);
                assert_eq!(draw.tokens, c.tokens);
            }
            SlotSource::FallbackDraw { seed } | SlotSource::ExplorationDraw { seed } => {
                let draws = if c.slot < POOL_SIZE {
                    &ledger.fallback_draws
                } else {
                    &ledger.exploration_draws
                };
                let draw = draws.iter().find(|x| x.seed == *seed).unwrap();
                assert_eq!(draw.outcome, UniformOutcome::Admitted);
                assert_eq!(draw.tokens, c.tokens);
            }
        }
        assert_eq!(c.adaptations[0].kind, EDGE_DECODE);
        if let Some(p) = &c.prediction {
            assert_eq!(c.adaptations[1].kind, EDGE_SCORE_ENCODING);
            assert_eq!(p.scored_program_sha256, c.program_sha256);
            let ranked = &proposal.ranking().selected()[c.slot];
            assert_eq!(ranked.program, c.program);
            assert_eq!(ranked.predicted_quality, p.predicted_quality);
        } else {
            assert!(c.slot >= POOL_SIZE && c.adaptations.len() == 1);
        }
        assert_eq!(c.canonical_recipe_sha256, recipe_hash(&c.program)?);
    }
    // Rejected pre-gate draws are kept for shadow studies, never handed off.
    let shadow = ledger.shadow();
    assert_eq!(shadow.len(), d.shadow);
    assert!(shadow.iter().all(|s| !s.fitting_authority));
    let handed = ledger
        .candidates
        .iter()
        .map(|c| c.tokens.clone())
        .collect::<std::collections::BTreeSet<_>>();
    assert!(shadow
        .iter()
        .filter(|s| !s.outcome.contains("Duplicate") && !s.outcome.contains("Surplus"))
        .all(|s| !handed.contains(s.tokens)));
    // Canonical duplicates are visible rather than counted as independent proposals.
    assert_eq!(
        ledger.canonical_groups().len(),
        d.unique_canonical_handed_off
    );
    // Slot records have stable, distinct identities for the GP lineage join.
    assert_eq!(
        ledger.slot_record_sha256(0)?,
        proposal.ledger().slot_record_sha256(0)?
    );
    assert_ne!(ledger.slot_record_sha256(0)?, ledger.slot_record_sha256(1)?);
    assert!(ledger.slot_record_sha256(80).is_err());
    Ok(())
}

/// Serialize `changed` into the producer artifact of `proposal` and require the reader to
/// refuse it with `needle`.
fn assert_ledger_refused(
    request: &Request,
    proposal: &seeds::Proposal,
    changed: &crate::proposal_ledger::ProposalLedger,
    needle: &str,
) {
    let mut envelope: serde_json::Value =
        serde_json::from_slice(&artifact_bytes(request, proposal).unwrap()).unwrap();
    envelope["proposal_ledger"] = serde_json::to_value(changed).unwrap();
    let error = match read_claims(&serde_json::to_vec(&envelope).unwrap(), request) {
        Ok(_) => panic!("{needle}: tampered ledger admitted"),
        Err(error) => format!("{error:#}"),
    };
    assert!(error.contains(needle), "{needle}: {error}");
}

/// W4-T07 negative cases: each tampered ledger is refused by the reader, so no producer
/// file (and no candidate) is handed on.
#[test]
fn tampered_proposal_ledgers_are_refused() -> Result<()> {
    use crate::decode::{DrawOutcome, POOL_SIZE};
    use crate::proposal_ledger::{AdaptationEdge, ProposalLedger, EDGE_DECODE};
    let (request, proposal) = unbound_proposal();
    let ledger = proposal.ledger().clone();
    let refused = |changed: ProposalLedger, needle: &str| {
        assert_ledger_refused(&request, &proposal, &changed, needle)
    };
    // Score attached to a different subject.
    let mut changed = ledger.clone();
    let (a, b) = (
        changed.candidates[0].prediction.clone(),
        changed.candidates[1].prediction.clone(),
    );
    changed.candidates[0].prediction = b;
    changed.candidates[1].prediction = a;
    refused(changed, "own program");
    // A fallback relabelled as a neural proposal (fallback inflation).
    let mut changed = ledger.clone();
    let fallback = changed
        .candidates
        .iter()
        .position(|c| c.origin == seeds::ORIGIN_NEURAL_FALLBACK)
        .expect("fixture has fallback slots");
    changed.candidates[fallback].origin = seeds::ORIGIN_NEURAL.into();
    refused(changed, "origin, source and slot range disagree");
    // A dropped failure in the fallback fill breaks its consecutive streams.
    let mut changed = ledger.clone();
    let failed = changed
        .fallback_draws
        .iter()
        .position(|d| d.outcome != crate::decode::UniformOutcome::Admitted)
        .expect("fixture has a failed fallback draw");
    changed.fallback_draws.remove(failed);
    refused(changed, "consecutive fresh streams");
    // An unterminated or relabelled attempt.
    let mut changed = ledger.clone();
    changed.neural_draws[0].outcome = DrawOutcome::Completed;
    changed.neural_draws[0].program_sha256 = None;
    refused(changed, "disagree");
    // Duplicate inflation: one draw handed off twice.
    let mut changed = ledger.clone();
    let mut twin = changed.candidates[0].clone();
    twin.slot = 1;
    twin.prediction = changed.candidates[1].prediction.clone();
    changed.candidates[1] = twin;
    refused(changed, "more than once");
    // Double adaptation.
    let mut changed = ledger.clone();
    let edge = AdaptationEdge {
        kind: EDGE_DECODE.into(),
        ..changed.candidates[POOL_SIZE].adaptations[0].clone()
    };
    changed.candidates[POOL_SIZE].adaptations.push(edge);
    refused(changed, "adaptations");
    // Inflated denominators.
    let mut changed = ledger.clone();
    changed.denominators.valid_programs += 1;
    refused(changed, "denominators differ");
    // A ledger recorded for another request.
    let mut changed = ledger.clone();
    changed.sampling.sampling_seed += 1;
    refused(changed, "another request");
    // Dropped exploration failures break the exploration streams the same way.
    let mut changed = ledger.clone();
    let failed = changed
        .exploration_draws
        .iter()
        .position(|d| d.outcome != crate::decode::UniformOutcome::Admitted)
        .expect("fixture has a refused exploration draw");
    changed.exploration_draws.remove(failed);
    refused(changed, "consecutive fresh streams");
    // Unknown ledger schema.
    let mut changed = ledger;
    changed.schema = "jope.proposal-ledger/v0".into();
    refused(changed, "unknown proposal ledger schema");

    // Second fixture (surplus and duplicate draws): a dropped non-completed neural draw
    // leaves a scheduled attempt without its terminal record.
    let (request, proposal) = filled_proposal();
    let ledger = proposal.ledger().clone();
    for outcome in [DrawOutcome::Surplus, DrawOutcome::Duplicate] {
        let mut changed = ledger.clone();
        let failed = changed
            .neural_draws
            .iter()
            .position(|d| d.outcome == outcome)
            .expect("fixture has surplus and duplicate draws");
        changed.neural_draws.remove(failed);
        assert_ledger_refused(&request, &proposal, &changed, "lack terminal records");
    }
    // A duplicate relabelled as a new program (duplicate inflation) is refused.
    let mut changed = ledger.clone();
    let duplicate = changed
        .neural_draws
        .iter()
        .position(|d| d.outcome == DrawOutcome::Duplicate)
        .unwrap();
    changed.neural_draws[duplicate].outcome = DrawOutcome::Surplus;
    assert_ledger_refused(&request, &proposal, &changed, "not a new program");
    Ok(())
}
