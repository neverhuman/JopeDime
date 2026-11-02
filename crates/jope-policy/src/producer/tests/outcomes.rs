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
