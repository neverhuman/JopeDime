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
