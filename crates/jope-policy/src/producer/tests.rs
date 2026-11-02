use super::*;
pub(super) fn request() -> Request {
    Request {
        worker_source_sha256: sha256_hex(b"synthetic producer source"),
        prepared_sha256: sha256_hex(b"synthetic prepared"),
        training_sha256: sha256_hex(b"synthetic training"),
        cut: 0,
        context_sha256: sha256_hex(b"synthetic context"),
        checkpoint_sha256: crate::ACCEPTED_STATE_SHA256.into(),
        policy_safetensors_sha256:
            "4b4b5a40f03c9226b4b15856fd35b90127c8b6a4d1435b815484cdd050744f1e".into(),
        vocabulary_sha256: crate::CHECKPOINT_VOCABULARY_SHA256.into(),
        sampling_seed: 1,
        fallback_seed: 2,
        exploration_seed: 3,
        sampling_rounds: 8,
        sampling_batch: 128,
        sampling_temperature: 1.,
        challenger: None,
        context_admission_sha256: None,
    }
}
// Deliberately forgeable file claims, never a Proposal or execution proof.
fn claims() -> Envelope {
    claims_for(request())
}
fn claims_for(request: Request) -> Envelope {
    let genomes = (0..80)
        .map(|i| format!("(pipe X (ridge lambda={}))", i + 1))
        .collect::<Vec<_>>();
    let population = serde_json::json!({"schema":seeds::SEED_SCHEMA,"prepared_sha256":request.prepared_sha256,"training_sha256":request.training_sha256,"cut":request.cut,"context_sha256":request.context_sha256,"checkpoint_sha256":request.checkpoint_sha256,"candidates":genomes.iter().enumerate().map(|(i,g)|serde_json::json!({"genome":g,"origin":if i<64 { seeds::ORIGIN_NEURAL } else { seeds::ORIGIN_RANDOM },"rank_log_weight":null})).collect::<Vec<_>>()});
    let bytes = serde_json::to_vec(&population).unwrap();
    Envelope {
        schema: SCHEMA.into(),
        request_sha256: request.identity().unwrap(),
        request,
        population_json: String::from_utf8(bytes.clone()).unwrap(),
        population_sha256: sha256_hex(&bytes),
        canonical_recipe_sha256: genomes.iter().map(|g| recipe_hash(g).unwrap()).collect(),
        ranking_backend: crate::ranking::BACKEND_CPU.into(),
        proposal_ledger: None,
    }
}
#[test]
fn producer_reader_preserves_80_typed_recipes_but_is_only_claims() -> Result<()> {
    let file = read_claims(&serde_json::to_vec(&claims())?, &request())?;
    assert_eq!(file.population().candidates().len(), 80);
    assert_eq!(
        file.population().candidates()[0].genome(),
        "(pipe X (ridge lambda=1))"
    );
    Ok(())
}
#[test]
fn producer_transport_rejects_seed_request_recipe_and_context_tampering() -> Result<()> {
    let expected = request();
    let mut changed = claims();
    changed.request.sampling_seed += 1;
    assert!(read_claims(&serde_json::to_vec(&changed)?, &expected).is_err());
    let mut changed = claims();
    changed.canonical_recipe_sha256[0] = sha256_hex(b"different recipe");
    assert!(read_claims(&serde_json::to_vec(&changed)?, &expected).is_err());
    let mut changed = claims();
    changed.population_json.push(' ');
    assert!(read_claims(&serde_json::to_vec(&changed)?, &expected).is_err());
    let mut different = expected.clone();
    different.context_sha256 = sha256_hex(b"another context");
    assert!(read_claims(&serde_json::to_vec(&claims())?, &different).is_err());
    Ok(())
}

/// A real (untrained, unbound) proposal and a challenger-shaped request naming its test
/// state, so the producer transport can be exercised end to end in a unit test.
pub(super) fn unbound_proposal() -> (Request, crate::seeds::Proposal) {
    unbound_proposal_with(1, 8, 1.0)
}
/// A second fixture whose pool fills during sampling: surplus and duplicate neural draws,
/// no fallback draws (the first fixture has only completed neural draws, many fallback
/// draws and at least one refused fallback draw).
pub(super) fn filled_proposal() -> (Request, crate::seeds::Proposal) {
    unbound_proposal_with(2, 64, 25.0)
}
pub(super) fn unbound_proposal_with(
    rounds: usize,
    batch: usize,
    temperature: f32,
) -> (Request, crate::seeds::Proposal) {
    let config = tiny_config(128);
    let model = crate::model::PolicyModel::random_weights(config, 5).unwrap();
    fixture_proposal(&model, &accepted_vocabulary(), rounds, batch, temperature).unwrap()
}

fn accepted_vocabulary() -> crate::grammar::BoundVocabulary {
    crate::grammar::bind_vocabulary(
        include_bytes!("../../../jope-engine/assets/grammar/featmath.json"),
        crate::CHECKPOINT_VOCABULARY_SHA256,
    )
    .unwrap()
}

fn tiny_config(max_seq: usize) -> crate::config::PolicyConfig {
    let mut config = crate::config::PolicyConfig::regression(401);
    config.width = 16;
    config.heads = 4;
    config.ffn = 32;
    config.layers = 1;
    config.dropout = 0.0;
    config.max_seq = max_seq;
    config
}

/// A real proposal by untrained `model` (the test state, a challenger-shaped request naming
/// it), so the producer transport can be exercised end to end in a unit test. The accepted
/// vocabulary goes through `propose_unbound`; any other only through the unit-test entry.
fn fixture_proposal(
    model: &crate::model::PolicyModel,
    vocabulary: &crate::grammar::BoundVocabulary,
    rounds: usize,
    batch: usize,
    temperature: f32,
) -> Result<(Request, crate::seeds::Proposal)> {
    let checkpoint = crate::config::CheckpointBinding::unbound_for_testing(
        model.config().clone(),
        crate::CHECKPOINT_VOCABULARY_SHA256,
    );
    let context = PolicyContext::from_frozen_values(
        (0..2048).map(|i| (i % 13) as f32 / 13.0 - 0.4).collect(),
    )?;
    let state = crate::testing::UNBOUND_TEST_STATE_SHA256.to_string();
    let weights = sha256_hex(b"unbound test weights");
    let request = Request {
        checkpoint_sha256: state.clone(),
        policy_safetensors_sha256: weights.clone(),
        context_sha256: crate::context_sha256(&context),
        vocabulary_sha256: vocabulary.sha256().into(),
        sampling_rounds: rounds,
        sampling_batch: batch,
        sampling_temperature: temperature,
        challenger: Some(crate::config::ChallengerIdentity {
            state_sha256: state,
            parameter_sha256: sha256_hex(b"unbound test parameters"),
            config_sha256: crate::ACCEPTED_CONFIG_SHA256.into(),
            policy_safetensors_sha256: weights,
        }),
        ..request()
    };
    let binding = seeds::SeedBinding::new(
        &request.prepared_sha256,
        request.cut,
        &request.training_sha256,
        &request.context_sha256,
    )?;
    let proposal_seeds = seeds::ProposalSeeds {
        sampling: request.sampling_seed,
        fallback: request.fallback_seed,
        exploration: request.exploration_seed,
    };
    let sampling = crate::decode::SamplingOptions {
        rounds: request.sampling_rounds,
        batch: request.sampling_batch,
        temperature: request.sampling_temperature,
    };
    let proposal = if vocabulary.sha256() == crate::CHECKPOINT_VOCABULARY_SHA256 {
        seeds::propose_unbound(
            model,
            &checkpoint,
            vocabulary,
            &context,
            &binding,
            &proposal_seeds,
            &sampling,
        )?
    } else {
        seeds::propose_with_vocabulary_for_test(
            model,
            vocabulary,
            &context,
            &binding,
            &proposal_seeds,
            &sampling,
        )?
    };
    Ok((request, proposal))
}

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

/// W4-T07: with the vocabulary, every recorded outcome is re-derived from its tokens, so a
/// valid program recorded as a refusal (or the reverse) is caught even when the counts are
/// made consistent.
#[test]
fn recorded_outcomes_are_rederived_from_tokens() -> Result<()> {
    use crate::decode::{DrawOutcome, UniformOutcome};
    let vocabulary = crate::grammar::bind_vocabulary(
        include_bytes!("../../../jope-engine/assets/grammar/featmath.json"),
        crate::CHECKPOINT_VOCABULARY_SHA256,
    )?;
    let (_, proposal) = filled_proposal();
    let ledger = proposal.ledger();
    ledger.reconcile_with_vocabulary(vocabulary.vocab())?;
    // A valid duplicate relabelled as a parser refusal, with consistent denominators.
    let mut changed = ledger.clone();
    let duplicate = changed
        .neural_draws
        .iter()
        .position(|d| d.outcome == DrawOutcome::Duplicate)
        .unwrap();
    changed.neural_draws[duplicate].outcome = DrawOutcome::ParserRejected;
    changed.neural_draws[duplicate].program_sha256 = None;
    changed.denominators.duplicate -= 1;
    changed.denominators.parser_rejected += 1;
    changed.denominators.valid_programs -= 1;
    changed.reconcile()?;
    let error = format!(
        "{:#}",
        changed
            .reconcile_with_vocabulary(vocabulary.vocab())
            .unwrap_err()
    );
    assert!(error.contains("outcome differs from its tokens"), "{error}");
    // A refused grammar-uniform draw relabelled with another refusal, keeping the claims
    // consistent, is caught the same way.
    let (_, proposal) = unbound_proposal();
    let mut changed = proposal.ledger().clone();
    let outcomes = changed
        .exploration_draws
        .iter()
        .map(|d| d.outcome)
        .collect::<Vec<_>>();
    let refused = changed
        .exploration_draws
        .iter()
        .position(|d| {
            !matches!(
                d.outcome,
                UniformOutcome::Admitted | UniformOutcome::Incomplete
            )
        })
        .unwrap_or_else(|| panic!("no relabelable refusal among {outcomes:?}"));
    let draw = &mut changed.exploration_draws[refused];
    (draw.outcome, draw.program_sha256) = match draw.outcome {
        UniformOutcome::ReencodingMismatch => (UniformOutcome::ParserRejected, None),
        UniformOutcome::ParserRejected => (
            UniformOutcome::ReencodingMismatch,
            Some(sha256_hex(b"program")),
        ),
        _ => (
            UniformOutcome::ReencodingMismatch,
            draw.program_sha256.clone(),
        ),
    };
    changed.reconcile()?;
    let error = format!(
        "{:#}",
        changed
            .reconcile_with_vocabulary(vocabulary.vocab())
            .unwrap_err()
    );
    assert!(error.contains("outcome differs from its tokens"), "{error}");
    Ok(())
}

/// W4-T07: an artifact written before the ledger existed stays readable, but carries no
/// candidate-identity or denominator evidence.
#[test]
fn historical_producer_artifact_has_no_ledger_evidence() -> Result<()> {
    let file = read_claims(&serde_json::to_vec(&claims())?, &request())?;
    assert!(file.ledger().is_none());
    Ok(())
}

/// Rewrite the seed population inside a producer artifact (and its pinned digest).
fn with_population(bytes: &[u8], edit: impl FnOnce(&mut serde_json::Value)) -> Vec<u8> {
    let mut envelope: serde_json::Value = serde_json::from_slice(bytes).unwrap();
    let mut population: serde_json::Value =
        serde_json::from_str(envelope["population_json"].as_str().unwrap()).unwrap();
    edit(&mut population);
    let json = serde_json::to_string_pretty(&population).unwrap();
    envelope["population_sha256"] = serde_json::json!(sha256_hex(json.as_bytes()));
    envelope["population_json"] = serde_json::json!(json);
    serde_json::to_vec(&envelope).unwrap()
}

/// W4-T07 join (the W5-T01 side): every seed slot names its proposal-ledger record, so a
/// candidate ledger seeded by the population joins the draw, tokens, scores and parser result
/// of each candidate. Reproduction: before the join the population carried only genome,
/// origin and rank weight.
#[test]
fn seed_slots_name_their_proposal_records() -> Result<()> {
    let (fixture, proposal) = unbound_proposal();
    let bytes = artifact_bytes(&fixture, &proposal)?;
    let file = read_claims(&bytes, &fixture)?;
    let ledger = file.ledger().expect("the producer writes a ledger");
    let population: serde_json::Value =
        serde_json::from_str(&serde_json::from_slice::<Envelope>(&bytes)?.population_json)?;
    for slot in 0..seeds::SEED_COUNT {
        assert_eq!(
            population["candidates"][slot]["proposal_record_sha256"],
            serde_json::json!(ledger.slot_record_sha256(slot)?),
            "seed slot {slot} does not name its proposal record"
        );
    }
    let refused = |bytes: Vec<u8>, request: &Request, needle: &str| {
        let error = match read_claims(&bytes, request) {
            Ok(_) => panic!("{needle}: admitted"),
            Err(error) => format!("{error:#}"),
        };
        assert!(error.contains(needle), "{needle}: {error}");
    };
    // A slot naming another slot's record.
    refused(
        with_population(&bytes, |p| {
            let first = p["candidates"][0]["proposal_record_sha256"].take();
            p["candidates"][0]["proposal_record_sha256"] =
                p["candidates"][1]["proposal_record_sha256"].clone();
            p["candidates"][1]["proposal_record_sha256"] = first;
        }),
        &fixture,
        "names another proposal record",
    );
    // A population that joins only some of its slots.
    refused(
        with_population(&bytes, |p| {
            p["candidates"][3]
                .as_object_mut()
                .unwrap()
                .remove("proposal_record_sha256");
        }),
        &fixture,
        "joins only some slots",
    );
    // A historical artifact (no ledger) cannot name records it has nothing to join to.
    let historical = serde_json::to_vec(&claims())?;
    refused(
        with_population(&historical, |p| {
            for c in p["candidates"].as_array_mut().unwrap() {
                c["proposal_record_sha256"] = serde_json::json!(sha256_hex(b"record"));
            }
        }),
        &request(),
        "without a proposal ledger",
    );
    // Positive control: a population without any record keeps its old meaning.
    read_claims(&with_population(&historical, |_| {}), &request())?;
    Ok(())
}

/// W4-T03: every draw's terminal outcome, as the controller's candidate ledger records it.
#[test]
fn draw_outcomes_record_every_draw_once() -> Result<()> {
    use crate::proposal_ledger::SlotSource;
    for (request, proposal) in [unbound_proposal(), filled_proposal()] {
        let file = read_claims(&artifact_bytes(&request, &proposal)?, &request)?;
        let ledger = file.ledger().expect("the producer writes a ledger");
        let outcomes = file.draw_outcomes()?.expect("a ledger has draw outcomes");
        assert_eq!(outcomes.ledger_sha256, ledger.sha256()?);
        assert_eq!(outcomes.denominators, ledger.denominators);
        assert_eq!(
            outcomes.draws.len(),
            ledger.neural_draws.len()
                + ledger.fallback_draws.len()
                + ledger.exploration_draws.len()
        );
        let d = &ledger.denominators;
        let neural = |label: &str| {
            outcomes
                .draws
                .iter()
                .filter(|o| matches!(o.draw, SlotSource::NeuralDraw { .. }) && o.outcome == label)
                .count()
        };
        assert_eq!(
            [
                neural("completed"),
                neural("surplus"),
                neural("duplicate"),
                neural("parser_rejected"),
                neural("incomplete")
            ],
            [
                d.completed,
                d.surplus,
                d.duplicate,
                d.parser_rejected,
                d.incomplete
            ]
        );
        // Each handed-off draw names its slot and that slot's record, which the seed slot
        // names too; no other draw names one.
        let mut handed = 0;
        for o in &outcomes.draws {
            match o.slot {
                Some(slot) => {
                    handed += 1;
                    let record = ledger.slot_record_sha256(slot)?;
                    assert_eq!(o.slot_record_sha256.as_deref(), Some(record.as_str()));
                    assert_eq!(
                        proposal.population().candidates()[slot].proposal_record_sha256(),
                        Some(record.as_str())
                    );
                    assert_eq!(ledger.candidates[slot].source, o.draw);
                    let admitted = match o.draw {
                        SlotSource::NeuralDraw { .. } => "completed",
                        _ => "admitted",
                    };
                    assert_eq!(o.outcome, admitted);
                }
                None => assert!(o.slot_record_sha256.is_none()),
            }
        }
        assert_eq!(
            (handed, outcomes.draws.len() - handed),
            (seeds::SEED_COUNT, d.shadow)
        );
    }
    // A historical artifact records no draws.
    let file = read_claims(&serde_json::to_vec(&claims())?, &request())?;
    assert!(file.draw_outcomes()?.is_none());
    Ok(())
}

/// A policy whose next-token logit for token `t` is exactly `logit(t)` at every step (the
/// `tests/decode_outcomes.rs` fixture).
fn fixed_logit_model(
    config: crate::config::PolicyConfig,
    logit: impl Fn(usize) -> f32,
) -> crate::model::PolicyModel {
    use candle_core::{Device, Tensor};
    let width = config.width;
    let tensors = crate::model::expected_inventory(&config)
        .into_iter()
        .enumerate()
        .map(|(i, (name, shape))| {
            let numel = shape.iter().product::<usize>();
            let values = match name.as_str() {
                "out_norm.weight" => vec![0.0; numel],
                "out_norm.bias" => vec![1.0; numel],
                "token_emb.weight" => (0..numel)
                    .map(|j| {
                        if j % width == 0 {
                            logit(j / width)
                        } else {
                            0.0
                        }
                    })
                    .collect(),
                _ => (0..numel)
                    .map(|j| ((i * 131 + j) as f32 * 0.37).sin() * 0.2)
                    .collect::<Vec<f32>>(),
            };
            let tensor = if shape.is_empty() {
                Tensor::new(values[0], &Device::Cpu)
            } else {
                Tensor::from_vec(values, shape, &Device::Cpu)
            }
            .unwrap();
            (name, tensor)
        })
        .collect::<Vec<_>>();
    crate::model::PolicyModel::from_tensors(config, tensors, Device::Cpu).unwrap()
}

/// W4-T07: real parser refusals (not relabelled records) through a whole proposal and its
/// ledger. Under the frozen grammar every accepting walk parses, so the product path never
/// refuses one; a vocabulary whose `stack` head renders as `stack)` makes every neural draw's
/// preferred walk `(stack)` a refusal, and grammar-uniform walks through that head refusals
/// in the fills. The product path and the producer transport refuse that vocabulary ("unbound
/// producer vocabulary"), so the ledger is checked directly.
#[test]
fn real_parser_refusals_reconcile_and_forgeries_are_refused() -> Result<()> {
    use crate::decode::{DrawOutcome, UniformOutcome};
    use crate::grammar::EOS;
    const STACK: usize = 372;
    let mut vocab = accepted_vocabulary().vocab().clone();
    vocab.ops.get_mut(&STACK).unwrap().name = "stack)".into();
    let vocabulary = crate::grammar::BoundVocabulary::for_unit_test(vocab.clone());
    // Every token preferred by descending id: the walk is `(stack)`.
    let model = fixed_logit_model(tiny_config(128), |t| -1000.0 * (401 - t.min(401)) as f32);
    let (request, proposal) = fixture_proposal(&model, &vocabulary, 1, 4, 1.0)?;
    let refused = artifact_bytes(&request, &proposal).map(drop).unwrap_err();
    assert!(format!("{refused:#}").contains("unbound producer vocabulary"));
    let ledger = proposal.ledger();
    ledger.reconcile()?;
    ledger.reconcile_with_vocabulary(&vocab)?;
    ledger.matches_population(proposal.population().candidates())?;
    let d = &ledger.denominators;
    assert_eq!(
        (
            d.scheduled_neural_draws,
            d.parser_rejected,
            d.complete_sequences,
            d.valid_programs
        ),
        (4, 4, 4, 0)
    );
    assert_eq!(d.fallback_admitted, crate::decode::POOL_SIZE);
    for draw in &ledger.neural_draws {
        assert_eq!(draw.outcome, DrawOutcome::ParserRejected);
        assert_eq!(draw.tokens, [1, 3, 397, STACK, EOS]);
    }
    let uniform = ledger
        .fallback_draws
        .iter()
        .chain(&ledger.exploration_draws)
        .position(|d| d.outcome == UniformOutcome::ParserRejected)
        .expect("a grammar-uniform walk through the refused head");
    // The candidate ledger records each refusal as a draw without a slot.
    let outcomes = ledger.draw_outcomes()?;
    assert_eq!(
        outcomes
            .draws
            .iter()
            .filter(|o| o.outcome == "parser_rejected" && o.slot.is_none())
            .count(),
        4 + ledger
            .fallback_draws
            .iter()
            .chain(&ledger.exploration_draws)
            .filter(|d| d.outcome == UniformOutcome::ParserRejected)
            .count()
    );
    // Under the frozen vocabulary the same tokens parse: the refusal is not re-derivable.
    assert!(ledger
        .reconcile_with_vocabulary(accepted_vocabulary().vocab())
        .is_err());
    // A refusal relabelled incomplete (its EOS dropped), with consistent denominators.
    let mut forged = ledger.clone();
    forged.neural_draws[0].tokens.pop();
    forged.neural_draws[0].outcome = DrawOutcome::Incomplete;
    forged.denominators.parser_rejected -= 1;
    forged.denominators.incomplete += 1;
    forged.denominators.complete_sequences -= 1;
    let error = format!("{:#}", forged.reconcile().unwrap_err());
    assert!(
        error.contains("did not exhaust its token budget"),
        "{error}"
    );
    // The same for a grammar-uniform refusal.
    let mut forged = ledger.clone();
    let fallback = uniform < forged.fallback_draws.len();
    let draw = if fallback {
        &mut forged.fallback_draws[uniform]
    } else {
        &mut forged.exploration_draws[uniform - ledger.fallback_draws.len()]
    };
    draw.tokens.pop();
    draw.outcome = UniformOutcome::Incomplete;
    let error = format!("{:#}", forged.reconcile().unwrap_err());
    assert!(
        error.contains("did not exhaust its token budget"),
        "{error}"
    );
    Ok(())
}

fn plan_of(r: &Request) -> RequestPlan {
    RequestPlan {
        worker_source_sha256: r.worker_source_sha256.clone(),
        prepared_sha256: r.prepared_sha256.clone(),
        training_sha256: r.training_sha256.clone(),
        cut: r.cut,
        checkpoint_sha256: r.checkpoint_sha256.clone(),
        policy_safetensors_sha256: r.policy_safetensors_sha256.clone(),
        vocabulary_sha256: r.vocabulary_sha256.clone(),
        sampling_seed: r.sampling_seed,
        fallback_seed: r.fallback_seed,
        exploration_seed: r.exploration_seed,
        sampling_rounds: r.sampling_rounds,
        sampling_batch: r.sampling_batch,
        sampling_temperature: r.sampling_temperature,
        challenger: r.challenger.clone(),
    }
}
/// W2-T02 G-03: the controller-issued admission digest is optional (native request bytes and
/// identities are unchanged), strictly formatted, and part of the request identity.
#[test]
fn context_admission_pin_is_optional_and_bound() -> Result<()> {
    let native = request();
    let bytes = serde_json::to_vec(&native)?;
    assert!(!String::from_utf8(bytes.clone())?.contains("context_admission_sha256"));
    let parsed: Request = serde_json::from_slice(&bytes)?;
    assert_eq!(parsed.context_admission_sha256, None);
    assert_eq!(parsed.identity()?, native.identity()?);
    let issued = Request {
        context_admission_sha256: Some(sha256_hex(b"controller pins")),
        ..request()
    };
    issued.validate()?;
    assert_ne!(issued.identity()?, native.identity()?);
    let parsed: Request = serde_json::from_slice(&serde_json::to_vec(&issued)?)?;
    assert_eq!(parsed, issued);
    for bad in [
        "not hex".to_string(),
        sha256_hex(b"controller pins").to_ascii_uppercase(),
    ] {
        assert!(Request {
            context_admission_sha256: Some(bad),
            ..request()
        }
        .validate()
        .is_err());
    }
    // The plan attaches exactly the encoding outputs.
    let plan = plan_of(&native);
    assert_eq!(plan.request(native.context_sha256.clone()), native);
    assert_eq!(
        plan.qualified_request(
            native.context_sha256.clone(),
            sha256_hex(b"controller pins")
        ),
        issued
    );
    // Claims written for one request never satisfy the other.
    assert!(read_claims(&serde_json::to_vec(&claims())?, &issued).is_err());
    read_claims(&serde_json::to_vec(&claims_for(issued.clone()))?, &issued)?;
    Ok(())
}

fn challenger() -> crate::config::ChallengerIdentity {
    crate::config::ChallengerIdentity {
        state_sha256: sha256_hex(b"challenger state"),
        parameter_sha256: sha256_hex(b"challenger parameters"),
        config_sha256: crate::ACCEPTED_CONFIG_SHA256.into(),
        policy_safetensors_sha256: sha256_hex(b"challenger weights"),
    }
}
fn challenger_request() -> Request {
    let c = challenger();
    Request {
        checkpoint_sha256: c.state_sha256.clone(),
        policy_safetensors_sha256: c.policy_safetensors_sha256.clone(),
        challenger: Some(c),
        ..request()
    }
}
fn refused(request: Request, needle: &str) {
    let error = request.validate().unwrap_err();
    assert!(format!("{error:#}").contains(needle), "{error:#}");
}
#[test]
fn challenger_request_is_optional_and_bound() -> Result<()> {
    // Absent: accepted request bytes, parse and identity are unchanged.
    let accepted = request();
    let bytes = serde_json::to_vec(&accepted)?;
    assert!(!String::from_utf8(bytes.clone())?.contains("\"challenger\""));
    let parsed: Request = serde_json::from_slice(&bytes)?;
    assert_eq!(parsed.challenger, None);
    assert_eq!(serde_json::to_vec(&parsed)?, bytes);
    assert_eq!(parsed.identity()?, accepted.identity()?);
    // Present: bound to its own state and weights, and a different request identity.
    let c = challenger();
    let bound = challenger_request();
    bound.validate()?;
    assert_ne!(bound.identity()?, accepted.identity()?);
    refused(
        Request {
            checkpoint_sha256: sha256_hex(b"other state"),
            ..challenger_request()
        },
        "differs from its challenger identity",
    );
    refused(
        Request {
            policy_safetensors_sha256: sha256_hex(b"other weights"),
            ..challenger_request()
        },
        "differs from its challenger identity",
    );
    let mut named_accepted = c.clone();
    named_accepted.state_sha256 = crate::ACCEPTED_STATE_SHA256.into();
    refused(
        Request {
            checkpoint_sha256: crate::ACCEPTED_STATE_SHA256.into(),
            challenger: Some(named_accepted),
            ..challenger_request()
        },
        "names the accepted identity",
    );
    // An accepted-shaped request cannot carry a challenger.
    assert!(Request {
        challenger: Some(c.clone()),
        ..request()
    }
    .validate()
    .is_err());
    // The plan carries the identity into the request.
    let r = challenger_request();
    let plan = plan_of(&r);
    plan.validate()?;
    assert_eq!(plan.request(r.context_sha256.clone()), r);
    assert_eq!(plan.request(r.context_sha256.clone()).challenger, Some(c));
    // The reader binds the population to the challenger state, never the accepted one.
    let file = read_claims(
        &serde_json::to_vec(&claims_for(challenger_request()))?,
        &challenger_request(),
    )?;
    assert_eq!(
        file.population().checkpoint_sha256(),
        challenger_request().checkpoint_sha256
    );
    let mut relabelled = claims_for(challenger_request());
    let mut population: serde_json::Value = serde_json::from_str(&relabelled.population_json)?;
    population["checkpoint_sha256"] = serde_json::json!(crate::ACCEPTED_STATE_SHA256);
    let population = serde_json::to_vec(&population)?;
    relabelled.population_sha256 = sha256_hex(&population);
    relabelled.population_json = String::from_utf8(population)?;
    assert!(read_claims(&serde_json::to_vec(&relabelled)?, &challenger_request()).is_err());
    Ok(())
}
