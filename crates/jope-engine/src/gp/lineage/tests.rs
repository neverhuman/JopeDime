use super::*;

const SEED: u64 = 7;

fn context<'a>() -> LineageContext<'a> {
    LineageContext {
        rng_seed: SEED,
        vocabulary: None,
        vocabulary_sha256: Value::Null,
        seeds: None,
        seeds_sha256: Value::Null,
        native: None,
        inherited: None,
    }
}

fn part(program: &str) -> Value {
    json!({"program":program,"program_sha256":hash(program.as_bytes())})
}

/// An unadapted (no vocabulary) ledger row carrying its envelope.
fn row(
    original: &str,
    source: Value,
    slot: Option<u64>,
    attempts: u64,
    rmse: Option<f64>,
) -> Value {
    let origin = if slot.is_some() {
        "generic_bootstrap"
    } else {
        "gp_evolution"
    };
    json!({"genome":original,"program_sha256":hash(original.as_bytes()),
        "proposal_before_projection":null,"initial_seed_slot":slot,"origin":origin,
        "proposal_attempts":attempts,"target_scale":1.0,
        "failure":if rmse.is_some() {Value::Null} else {json!("timeout")},
        "metrics":rmse.map(|r| json!({"rmse":r,"semantic_sha256":hash(original.as_bytes())})),
        "lineage":{"schema":LINEAGE_SCHEMA,"original":part(original),"source":source,
            "adaptation":{"kind":"none","applications":0},"executable":part(original)}})
}

/// Slot `slot`'s variation accepted at `draw`, after a counter of `before`.
fn variation(slot: u64, draw: u64, before: u64, operator: &str, parents: &[(i64, &str)]) -> Value {
    json!({"kind":"variation","operator":operator,"slot":slot,"slot_draw":draw,
        "proposal_attempt":before + draw + 1,
        "variation_seed":random_word(SEED, slot, draw, "variation"),
        "operator_word":random_word(SEED, slot, draw, "operator") % 16,
        "parents":parents.iter().map(|(at, program)| json!({"position":at,
            "program_sha256":hash(program.as_bytes())})).collect::<Vec<_>>()})
}

/// First draw within `slot` whose operator word satisfies `accept` and
/// whose operator output exists.
fn draw_where(
    slot: u64,
    accept: impl Fn(u64) -> bool,
    child: impl Fn(u64, u64) -> Option<String>,
) -> (u64, String) {
    (0..10_000)
        .find_map(|d| {
            let word = random_word(SEED, slot, d, "operator") % 16;
            accept(word)
                .then(|| child(d, random_word(SEED, slot, d, "variation")))
                .flatten()
                .map(|c| (d, c))
        })
        .expect("a matching operator draw")
}

/// Reference, one bootstrap draw, then a uniform mutation, a uniform
/// crossover and a fresh draw, all re-derivable from the stream words.
fn ledger() -> Vec<Value> {
    let reference = FIXED_REFERENCE.to_string();
    let bootstrap = random_canonical_pipeline(SEED, 6);
    let mut rows = vec![
        row(
            &reference,
            json!({"kind":"fixed_reference","slot":0}),
            Some(0),
            1,
            Some(0.5),
        ),
        row(
            &bootstrap,
            json!({"kind":"generic_bootstrap","slot":1,"generator":GENERATOR,
                "generator_seed":SEED}),
            Some(1),
            2,
            Some(0.4),
        ),
    ];
    // Pool order by objective: bootstrap (0.4), reference (0.5).
    let pool2 = [(1i64, bootstrap.clone()), (0, reference.clone())];
    let (a, child) = draw_where(
        2,
        |w| (2..9).contains(&w),
        |d, seed| {
            let left = &pool2[(random_word(SEED, 2, d, "left_parent") % 2) as usize].1;
            mutate_canonical_pipeline(left, seed, 6).ok()
        },
    );
    let left = pool2[(random_word(SEED, 2, a, "left_parent") % 2) as usize].clone();
    let after_mutation = 2 + a + 1;
    rows.push(row(
        &child,
        variation(2, a, 2, "mutate", &[(left.0, left.1.as_str())]),
        None,
        after_mutation,
        Some(0.3),
    ));
    let pool3 = [
        (2i64, child.clone()),
        (1, bootstrap.clone()),
        (0, reference.clone()),
    ];
    let (b, cross) = draw_where(
        3,
        |w| w >= 9,
        |d, seed| {
            let l = &pool3[(random_word(SEED, 3, d, "left_parent") % 3) as usize].1;
            let r = &pool3[(random_word(SEED, 3, d, "right_parent") % 3) as usize].1;
            crossover_canonical_pipelines(l, r, seed, 6)
                .ok()
                .filter(|c| c != &child && c != &bootstrap && c != &reference)
        },
    );
    let l = pool3[(random_word(SEED, 3, b, "left_parent") % 3) as usize].clone();
    let r = pool3[(random_word(SEED, 3, b, "right_parent") % 3) as usize].clone();
    let after_crossover = after_mutation + b + 1;
    rows.push(row(
        &cross,
        variation(
            3,
            b,
            after_mutation,
            "crossover",
            &[(l.0, l.1.as_str()), (r.0, r.1.as_str())],
        ),
        None,
        after_crossover,
        None,
    ));
    let (c, fresh) = draw_where(
        4,
        |w| w < 2,
        |_, seed| Some(random_canonical_pipeline(seed, 6)),
    );
    rows.push(row(
        &fresh,
        variation(4, c, after_crossover, "fresh_random", &[]),
        None,
        after_crossover + c + 1,
        Some(0.9),
    ));
    rows
}

fn refused(rows: &[Value], expected: &str) {
    let error = format!("{:#}", verify_rows(&context(), rows).unwrap_err());
    assert!(error.contains(expected), "expected {expected:?}: {error}");
}

/// W4-T07 (the W5-T01 join): a seed slot's lineage source carries the
/// proposal-ledger slot record digest (`ProposalLedger::slot_record_sha256`)
/// its seed population binds. A source that omits, changes or invents it is
/// refused; a seed population without records keeps the old meaning.
#[test]
fn a_seed_slot_joins_its_proposal_record_and_refuses_any_other() {
    let program = random_canonical_pipeline(SEED, 6);
    let record = hash(b"proposal ledger slot record 0");
    let population = |candidate: Value| -> crate::gp::SeedPopulation {
        serde_json::from_value(json!({"schema":"parallel.seed_population.v1",
            "prepared_sha256":"0".repeat(64),"cut":0,"training_sha256":"1".repeat(64),
            "context_sha256":"2".repeat(64),"checkpoint_sha256":"3".repeat(64),
            "candidates":[candidate]}))
        .unwrap()
    };
    let joined = population(json!({"genome":program,"origin":"neural_policy",
        "rank_log_weight":-0.5,"proposal_record_sha256":record}));
    let unjoined =
        population(json!({"genome":program,"origin":"neural_policy","rank_log_weight":-0.5}));
    let seeds_sha256 = json!(hash(b"seed file"));
    let seed_row = |source: Value| {
        let mut r = row(&program, source, Some(0), 1, Some(0.5));
        r["origin"] = json!("neural_policy");
        vec![r]
    };
    let source = |record: Option<&str>| {
        let mut s = json!({"kind":"seed_population_slot","slot":0,"seeds_sha256":seeds_sha256,
            "origin":"neural_policy","rank_log_weight":-0.5,"score_subject":"original_proposal"});
        if let Some(record) = record {
            s["proposal_record_sha256"] = json!(record);
        }
        s
    };
    let verify = |seeds: &crate::gp::SeedPopulation, rows: &[Value]| {
        verify_rows(
            &LineageContext {
                seeds: Some(seeds),
                seeds_sha256: seeds_sha256.clone(),
                ..context()
            },
            rows,
        )
        .map(drop)
        .map_err(|e| format!("{e:#}"))
    };
    // Positive controls: the joined record, and an unjoined (older) population.
    verify(&joined, &seed_row(source(Some(&record)))).unwrap();
    verify(&unjoined, &seed_row(source(None))).unwrap();
    // The population binds a record: a source omitting or changing it is refused.
    for bad in [source(None), source(Some(&hash(b"another slot record")))] {
        let refused = verify(&joined, &seed_row(bad));
        assert!(
            matches!(&refused, Err(e) if e.contains("seed slot source differs")),
            "a seed slot source that does not join its proposal record was accepted: {refused:?}"
        );
    }
    // A source may not invent a record the population does not bind.
    let refused = verify(&unjoined, &seed_row(source(Some(&record))));
    assert!(
        matches!(&refused, Err(e) if e.contains("seed slot source differs")),
        "a seed slot source invented a proposal record: {refused:?}"
    );
}

#[test]
fn recorded_sources_operators_and_parents_re_derive_every_original() {
    let rows = ledger();
    let lineage = verify_rows(&context(), &rows).unwrap();
    let sources: Vec<&str> = lineage.chains.iter().map(|c| c.source.as_str()).collect();
    assert_eq!(
        sources,
        [
            "fixed_reference",
            "generic_bootstrap",
            "variation",
            "variation",
            "variation"
        ]
    );
    assert_eq!(lineage.variations, 3);
    assert_eq!(lineage.adapted, 0);
    assert_eq!(lineage.chains[2].parents.len(), 1);
    assert_eq!(lineage.chains[3].parents.len(), 2);
    assert!(lineage.chains[4].parents.is_empty());
}

#[test]
fn parent_swaps_and_ineligible_parents_are_refused() {
    let rows = ledger();
    let parent = rows[2]["lineage"]["source"]["parents"][0]["position"]
        .as_i64()
        .unwrap();
    let (other, program) = if parent == 0 {
        (1, rows[1]["genome"].clone())
    } else {
        (0, rows[0]["genome"].clone())
    };
    let mut swapped = rows.clone();
    swapped[2]["lineage"]["source"]["parents"][0] = json!({"position":other,
        "program_sha256":hash(program.as_str().unwrap().as_bytes())});
    refused(&swapped, "uniform stream draw");
    // A parent whose identity does not match the ledger row it names.
    let mut relabelled = rows.clone();
    relabelled[2]["lineage"]["source"]["parents"][0]["program_sha256"] =
        json!(hash(b"another parent"));
    refused(&relabelled, "parent identity differs");
    // A failed row was never an eligible parent.
    let mut failed = rows.clone();
    let at = parent as usize;
    failed[at]["failure"] = json!("timeout");
    failed[at]["metrics"] = Value::Null;
    refused(&failed, "not an eligible measured entry");
    // A later row cannot parent an earlier one.
    let mut future = rows.clone();
    future[2]["lineage"]["source"]["parents"][0] = json!({"position":4,
        "program_sha256":rows[4]["program_sha256"]});
    refused(&future, "not an eligible measured entry");
}

#[test]
fn omitted_or_contradictory_variation_events_are_refused() {
    let rows = ledger();
    let mut omitted = rows.clone();
    omitted[2]["lineage"]["source"] = Value::Null;
    refused(&omitted, "source omitted");
    let mut orphan = rows.clone();
    orphan[3]["lineage"]["source"]["parents"] = json!([]);
    refused(&orphan, "contradicts the operator stream");
    let mut relabelled = rows.clone();
    relabelled[2]["lineage"]["source"] = json!({"kind":"generic_bootstrap","slot":2,
        "generator":GENERATOR,"generator_seed":SEED + 1});
    refused(&relabelled, "non-initial row");
    let mut operator = rows.clone();
    operator[2]["lineage"]["source"]["operator"] = json!("crossover");
    refused(&operator, "contradicts the operator stream");
    let mut streams = rows.clone();
    streams[4]["lineage"]["source"]["variation_seed"] = json!(1);
    refused(&streams, "random streams");
    // Streams belong to one slot and one draw the ledger counters confirm:
    // an event cannot claim another slot, another draw, or reuse another
    // slot's words (W5-T02).
    let mut slot = rows.clone();
    slot[4]["lineage"]["source"]["slot"] = json!(3);
    refused(&slot, "slot or draw differs");
    let mut draw = rows.clone();
    let recorded = draw[4]["lineage"]["source"]["slot_draw"].as_u64().unwrap();
    draw[4]["lineage"]["source"]["slot_draw"] = json!(recorded + 1);
    refused(&draw, "slot or draw differs");
    draw[4]["lineage"]["source"]["slot_draw"] = json!(u64::MAX);
    refused(&draw, "slot or draw differs");
    let mut borrowed = rows.clone();
    let from_slot_three = rows[3]["lineage"]["source"].clone();
    for key in ["variation_seed", "operator_word"] {
        borrowed[4]["lineage"]["source"][key] = from_slot_three[key].clone();
    }
    refused(&borrowed, "random streams");
    let mut missing = rows.clone();
    missing[0]["lineage"] = Value::Null;
    refused(&missing, "envelope absent");
    // A bootstrap slot must be the recorded generator draw.
    let mut draw = rows.clone();
    draw[1]["lineage"]["source"]["generator_seed"] = json!(SEED + 1);
    refused(&draw, "generator draw");
}

#[test]
fn recipe_substitution_and_double_adaptation_are_refused() {
    let rows = ledger();
    let substitute = random_canonical_pipeline(99, 6);
    let mut executable = rows.clone();
    executable[1]["lineage"]["executable"] = part(&substitute);
    refused(&executable, "executable recipe differs");
    let mut consistent = rows.clone();
    consistent[1]["genome"] = json!(substitute);
    consistent[1]["program_sha256"] = json!(hash(substitute.as_bytes()));
    consistent[1]["lineage"]["executable"] = part(&substitute);
    refused(&consistent, "unadapted executable differs");
    let mut declared = rows.clone();
    declared[1]["lineage"]["adaptation"] = json!({"kind":"vocabulary_projection",
        "applications":1});
    refused(&declared, "adaptation differs from the frozen request");

    // With the frozen vocabulary: one projection of the original only.
    let vocabulary: crate::grammar::Vocab =
        read_json(&Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/grammar/featmath.json"))
            .unwrap();
    let (seed, raw, adapted) = (SEED..SEED + 1000)
        .find_map(|seed| {
            let raw = random_canonical_pipeline(seed, 6);
            let adapted = vocabulary.project_proposal(&raw).ok()?;
            (adapted != raw).then_some((seed, raw, adapted))
        })
        .unwrap();
    let mut adapted_row = row(
        &raw,
        json!({"kind":"generic_bootstrap","slot":1,"generator":GENERATOR,
            "generator_seed":seed}),
        Some(1),
        1,
        Some(0.4),
    );
    adapted_row["genome"] = json!(adapted);
    adapted_row["program_sha256"] = json!(hash(adapted.as_bytes()));
    adapted_row["proposal_before_projection"] = json!(raw);
    adapted_row["lineage"]["executable"] = part(&adapted);
    adapted_row["lineage"]["adaptation"] = json!({"kind":"vocabulary_projection",
        "applications":1,"vocabulary_sha256":"v","input_sha256":hash(raw.as_bytes()),
        "output_sha256":hash(adapted.as_bytes()),"changed":true});
    let mut context = context();
    context.vocabulary = Some(&vocabulary);
    context.vocabulary_sha256 = json!("v");
    let check = |row: &Value| verify_rows(&context, std::slice::from_ref(row));
    check(&adapted_row).unwrap();
    // The adapted program posing as the original: a second adaptation.
    let mut twice = adapted_row.clone();
    twice["lineage"]["original"] = part(&adapted);
    twice["lineage"]["adaptation"]["input_sha256"] = json!(hash(adapted.as_bytes()));
    twice["lineage"]["adaptation"]["changed"] = json!(false);
    twice["proposal_before_projection"] = Value::Null;
    let error = format!("{:#}", check(&twice).unwrap_err());
    assert!(error.contains("generator draw"), "{error}");
    let mut repeated = adapted_row.clone();
    repeated["lineage"]["adaptation"]["applications"] = json!(2);
    assert!(check(&repeated).is_err());
    let mut unrecorded = adapted_row.clone();
    unrecorded["proposal_before_projection"] = Value::Null;
    let error = format!("{:#}", check(&unrecorded).unwrap_err());
    assert!(error.contains("ledger original proposal"), "{error}");
}
