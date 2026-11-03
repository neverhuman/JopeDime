use super::*;

// Fix round 02: the last round of a run stopped on its evaluations bound, the last-round rule on
// its own, and the not-run count.

/// `engine` with the digest and both counts of `moves` (serialized as `bytes`), as publication
/// writes them, so that only the record checks can refuse the records.
pub(super) fn recounted(engine: &EngineRecord, moves: &[MoveRecord], bytes: &[u8]) -> EngineRecord {
    let mut engine = digested(engine, bytes);
    let record = engine.moves.as_mut().unwrap();
    record.refused_by_pool = Some(refused_by_pool(moves));
    record.not_run = Some(not_run(moves));
    engine
}

/// Publishes `moves` in place of the run's records, under a recounted selection record.
pub(super) fn publish_recounted(
    output: &Path,
    receipt: &Receipt,
    engine: &EngineRecord,
    moves: &[MoveRecord],
) -> PathBuf {
    let bytes = serde_json::to_vec_pretty(moves).unwrap();
    rewrite(
        output,
        receipt,
        Some(bytes.as_slice()),
        Some(recounted(engine, moves, &bytes)),
    )
}

/// The files a records edit rewrites, read back so they can be restored.
pub(super) fn originals(output: &Path) -> Vec<(PathBuf, Vec<u8>)> {
    ["receipt.json", "search/cut-0/search.json", PATH]
        .iter()
        .map(|r| (output.join(r), std::fs::read(output.join(r)).unwrap()))
        .collect()
}

pub(super) fn restore(originals: &[(PathBuf, Vec<u8>)]) {
    for (file, bytes) in originals {
        std::fs::write(file, bytes).unwrap();
    }
}

#[test]
fn under_an_evaluations_stop_the_last_round_stops_where_the_allowance_ran_out() {
    let temp = tempfile::tempdir().unwrap();
    // 24 evaluations: one round, whose rewrite pass measures one child with the last evaluation
    // and stops short of its other three ranked parents.
    let opt = run_options(temp.path(), 24, four_parents(), simplify());
    let receipt = run::run(&opt).unwrap();
    let s = &receipt.search;
    assert_eq!(s.stopped_by, crate::search::STOPPED_BY_EVALUATIONS);
    assert_eq!(s.generations, 1);
    let output = &opt.run.output;
    let saved = originals(output);
    let engine = s.engine_selection.clone().unwrap();
    let moves = published_moves(&opt);
    let measurements = published_measurements(&opt);
    assert_eq!(moves.len(), 1);
    assert_eq!(moves[0].outcome, MEASURED);
    assert_eq!(moves[0].measurement, Some(measurements.len() - 1));
    publication::verify_run(&output.join("receipt.json")).unwrap();
    // The round's ranked parents, as the pass read them.
    let observations: Vec<Kernel2Observation> = measurements
        .iter()
        .map(|m| m.derive(FaultPolicy::RejectHard).unwrap())
        .collect();
    let start = observations.len() - 1;
    let parents =
        owned(ranked_parents(&evaluator(), &observations[..start], REWRITE_SATURATE, 4).unwrap());
    assert_eq!(parents.len(), 4);
    assert_eq!(moves[0].parent_sha256, parents[0].1);
    let mut errors = Vec::new();
    // A record for the next ranked parent appended after the measurement that spent the
    // allowance. Its program was already measured, so as a duplicate it re-derives on its own,
    // and only the allowance refuses it.
    let width = (receipt.dataset.columns - 1) as usize;
    let proposal = propose(&evaluator(), &parents[1].0, width).unwrap();
    assert!(proposal.admitted);
    let program = proposal.program.clone().unwrap();
    assert!(measurements.iter().any(|m| m.observation.genome == program));
    let mut appended = moves.clone();
    appended.push(MoveRecord {
        id: REWRITE_SATURATE.into(),
        round: 0,
        parent_sha256: parents[1].1.clone(),
        program: proposal.program,
        outcome: DUPLICATE.into(),
        reason: proposal.reason,
        measurement: None,
        rewrites: proposal.rewrites,
        linearization: None,
    });
    let path = publish_recounted(output, &receipt, &engine, &appended);
    errors.push(refusal_of(
        &path,
        "a record after the allowance was spent",
        &["a move record starts after the allowance was spent"],
    ));
    restore(&saved);
    // The round's measured record removed: the pass now stops short with an evaluation of
    // allowance left, the one its child spent.
    let path = publish_recounted(output, &receipt, &engine, &[]);
    errors.push(refusal_of(
        &path,
        "the last round's measured record removed",
        &["a move pass stopped short of its ranked parents with allowance left"],
    ));
    restore(&saved);
    // The not-run count changed or dropped, the records unchanged.
    for (label, count) in [("not-run count", Some(1)), ("not-run count dropped", None)] {
        let mut record = engine.clone();
        assert_eq!(record.moves.as_ref().unwrap().not_run, Some(0));
        record.moves.as_mut().unwrap().not_run = count;
        let path = rewrite(output, &receipt, None, Some(record));
        errors.push(refusal_of(
            &path,
            label,
            &["the not-run count does not re-derive from the move records"],
        ));
        restore(&saved);
    }
    publication::verify_run(&output.join("receipt.json")).unwrap();
    let root = output.display().to_string();
    for error in errors {
        lint(&error.replace(&root, "<run>"));
    }
}

#[test]
fn only_the_last_round_may_stop_short() {
    let temp = tempfile::tempdir().unwrap();
    // 36 evaluations: two rounds. Round 0's pass reads four parents (measured, duplicate,
    // unchanged, unchanged); round 1's reads none, the allowance spent before it.
    let opt = run_options(temp.path(), 36, four_parents(), simplify());
    let receipt = run::run(&opt).unwrap();
    let s = &receipt.search;
    assert_eq!(s.stopped_by, crate::search::STOPPED_BY_EVALUATIONS);
    assert_eq!(s.generations, 2);
    let output = &opt.run.output;
    let saved = originals(output);
    let engine = s.engine_selection.clone().unwrap();
    let moves = published_moves(&opt);
    let outcomes: Vec<&str> = moves.iter().map(|m| m.outcome.as_str()).collect();
    assert_eq!(outcomes, [MEASURED, DUPLICATE, UNCHANGED, UNCHANGED]);
    assert!(moves.iter().all(|m| m.round == 0));
    // Round 0's records after its measured one removed. No measurement is left unaccounted, no
    // record follows, and the counts are recounted, so only the rule that a pass may stop short
    // only in the run's last round refuses it.
    let kept = &moves[..1];
    let path = publish_recounted(output, &receipt, &engine, kept);
    let error = refusal_of(
        &path,
        "an earlier round's tail removed",
        &["a move pass stopped short of its ranked parents before the run's last round"],
    );
    restore(&saved);
    publication::verify_run(&output.join("receipt.json")).unwrap();
    lint(&error.replace(&output.display().to_string(), "<run>"));
}

// Fix round 03: the owner's D9b ruling on what the pool refusal count counts.

#[test]
fn a_parent_ranked_top_for_n_rounds_adds_n_pool_refusals() {
    let temp = tempfile::tempdir().unwrap();
    // [`NEGATED`] as the only seed, one parent per round and no refinement trials: the seed stays
    // the top ranked parent with an expression stage in every round whose pass runs. Its only
    // rewrite output, the negation of `c0`, is refused by the profile, so no child is measured
    // and nothing outranks it. 48 evaluations: six rounds, the last with its allowance spent
    // before its pass.
    let mut opt = run_options(
        temp.path(),
        48,
        Refinement {
            top_k: 1,
            trials_per_program: 0,
        },
        simplify(),
    );
    let seeds = temp.path().join("negated-seed.json");
    let file = crate::seeds::SeedsFile::new(
        None,
        vec![crate::seeds::SeedGenome {
            genome: NEGATED.into(),
            program_sha256: crate::sha256_hex(NEGATED.as_bytes()),
            bank_entry_sha256: None,
            receipt_sha256: None,
        }],
    );
    std::fs::write(&seeds, serde_json::to_vec(&file).unwrap()).unwrap();
    opt.run.seed_genomes = Some(seeds);
    let receipt = run::run(&opt).unwrap();
    let s = &receipt.search;
    let moves = published_moves(&opt);
    let n = moves.len();
    assert!(n >= 3, "{n} rounds read the parent");
    assert_eq!(s.stopped_by, crate::search::STOPPED_BY_EVALUATIONS);
    assert_eq!(s.generations, n + 1);
    // One record per round, each for the same parent, each the same refusal by the pool.
    let parent = crate::sha256_hex(NEGATED.as_bytes());
    for (round, m) in moves.iter().enumerate() {
        assert_eq!(
            (m.round, m.parent_sha256.as_str(), m.outcome.as_str()),
            (round, parent.as_str(), REFUSED)
        );
        assert!(m.program.is_none() && m.measurement.is_none());
        assert_eq!(m.rewrites.len(), 1);
        assert!(m.rewrites[0].refused_by_pool(), "{:?}", m.rewrites[0]);
        assert_eq!(m.rewrites, moves[0].rewrites);
    }
    // The published count is N: one per round the parent is read, not one per distinct
    // expression.
    let record = s.engine_selection.as_ref().unwrap().moves.as_ref().unwrap();
    assert_eq!(record.refused_by_pool, Some(n));
    assert_eq!(refused_by_pool(&moves), n);
    assert_eq!(pool_count(&record.ids, &moves), Some(n));
    // `verify_run` accepts the run, re-deriving N from the records (and, since it re-derives the
    // set and order of records against the ranked measurements, that the parent was the top
    // ranked one in each of those rounds).
    let output = &opt.run.output;
    assert_eq!(
        publication::verify_run(&output.join("receipt.json")).unwrap(),
        receipt
    );
    // N - 1 and N + 1, search.json and the receipt rehashed: refused.
    let saved = originals(output);
    let engine = s.engine_selection.clone().unwrap();
    let mut errors = Vec::new();
    for count in [n - 1, n + 1] {
        let mut edited = engine.clone();
        edited.moves.as_mut().unwrap().refused_by_pool = Some(count);
        let path = rewrite(output, &receipt, None, Some(edited));
        errors.push(refusal_of(
            &path,
            &format!("a pool refusal count of {count} for {n} rounds"),
            &["the pool refusal count does not re-derive from the move records"],
        ));
        restore(&saved);
    }
    publication::verify_run(&output.join("receipt.json")).unwrap();
    let root = output.display().to_string();
    for error in errors {
        lint(&error.replace(&root, "<run>"));
    }
}

// Fix round 04: a record past the run's last round, which only the final check that every
// record was read refuses (mutants-01, M-E2).

#[test]
fn verify_refuses_a_move_record_past_the_runs_last_round() {
    let temp = tempfile::tempdir().unwrap();
    let opt = run_options(temp.path(), 40, four_parents(), simplify());
    let receipt = run::run(&opt).unwrap();
    let s = &receipt.search;
    let output = &opt.run.output;
    let saved = originals(output);
    let engine = s.engine_selection.clone().unwrap();
    let moves = published_moves(&opt);
    publication::verify_run(&output.join("receipt.json")).unwrap();
    // A copy of a record that measures nothing, for the round after the run's last, appended
    // with the counts recounted. The check reads records round by round up to the run's last
    // round, so it never compares this one with a parent, a proposal or a measurement; only the
    // final check that every record was read refuses it.
    let mut extra = moves
        .iter()
        .find(|m| m.outcome != MEASURED)
        .expect("a record that measures nothing")
        .clone();
    extra.round = s.generations;
    let mut appended = moves.clone();
    appended.push(extra);
    let path = publish_recounted(output, &receipt, &engine, &appended);
    let error = publication::verify_run(&path)
        .map(|_| ())
        .expect_err("a record past the run's last round");
    assert_eq!(
        error.root_cause().to_string(),
        "move records do not follow the ranked parents",
        "{error:#}"
    );
    lint(&format!("{error:#}").replace(&output.display().to_string(), "<run>"));
    restore(&saved);
    publication::verify_run(&output.join("receipt.json")).unwrap();
}

/// [`check`] reads the pass's parents back from evidence, which is untrusted, so
/// `has_expression_stage` and `propose` parse them within `parse_untrusted_pipeline`'s bounds (the
/// class of B17's review 02): the pipeline form only, and each expression's nesting bytes counted
/// before it is parsed. Unary plus and redundant parentheses never reach the parsed tree, so a
/// parent at the limit is read and rewritten as its plain form is; one byte more is not readable,
/// and `propose` refuses it unparsed.
#[test]
fn rewrite_parents_are_bounded_before_they_are_parsed() {
    use crate::kernel2::refinement::MAX_PARENT_EXPRESSION_NESTING;
    let opt = options(4, 64, FaultPolicy::RejectHard);
    let body = "1*c0";
    // `*`.
    let room = MAX_PARENT_EXPRESSION_NESTING - 1;
    let parent = |source: String| ONE_TIMES.replace(body, &source);
    let plus = |n: usize| parent(format!("{}{body}", "+".repeat(n)));
    let grouped = |n: usize| parent(format!("{}{body}{}", "(".repeat(n), ")".repeat(n)));
    let plain = propose(&opt.evaluator, ONE_TIMES, 1).unwrap();
    assert!(plain.admitted);
    for program in [plus(room), grouped(room)] {
        assert!(has_expression_stage(&program), "{program}");
        let padded = propose(&opt.evaluator, &program, 1).unwrap();
        assert_eq!(
            (&padded.program, padded.admitted),
            (&plain.program, plain.admitted),
            "{program}"
        );
    }
    for program in [plus(room + 1), grouped(room + 1)] {
        assert!(!has_expression_stage(&program), "{program}");
        let error = propose(&opt.evaluator, &program, 1).unwrap_err();
        assert!(
            format!("{error:#}").contains("nests too deeply"),
            "{program}: {error:#}"
        );
    }
    let closed_form = format!("(sym raw \"{}c0\")", "+".repeat(room + 1));
    assert!(!has_expression_stage(&closed_form));
    let error = propose(&opt.evaluator, &closed_form, 1).unwrap_err();
    // Refused before any parse, by the pre-parse check's own message.
    assert!(
        format!("{error:#}").contains("not in the pipeline form"),
        "{error:#}"
    );
}

/// Independent review 01 of the wiring PR (B20-R01-AST-FIRST): `MathExpr::simplify_v2`, the
/// public entry, checks a decoded tree's structure before it copies or normalizes it. A
/// `MathExpr` decoded from JSON has had no structure check, and the copy and `Search::finish`'s
/// normalization both recurse.
#[test]
fn simplify_v2_checks_a_decoded_tree_before_copying_it() {
    let opt = options(4, 64, FaultPolicy::RejectHard);
    let profile = opt.evaluator.profile();
    let pool = profile.expr.as_ref().unwrap();
    let scope = || ExpressionScope {
        columns: pool.column_upper_exclusive,
        elementwise: false,
        max_nodes: pool.max_nodes,
        max_depth: pool.max_depth,
    };
    // #202 review 04's shallow 65-node tree: one call with 64 literal arguments.
    let leaves = vec![serde_json::json!({ "Const": 1.0 }); 64];
    let wide: MathExpr =
        serde_json::from_value(serde_json::json!({ "root": { "Call": ["Abs", leaves] } })).unwrap();
    let error = wide.simplify_v2(profile, scope()).unwrap_err();
    assert!(
        format!("{error:#}").contains("before it is copied"),
        "{error:#}"
    );
    // An admitted tree passes the check and simplifies as before.
    assert!(MathExpr::parse("1*c0")
        .unwrap()
        .simplify_v2(profile, scope())
        .is_ok());
}

/// Independent review 01 of the wiring PR (B20-R01-ROUND-BOUND): `check` replays every round the
/// receipt declares, so the round count is bounded by the search's stop rule first, at most
/// `BARREN_ROUND_LIMIT` rounds per completed evaluation plus that many. One more is refused, with
/// `search.json` and the receipt rehashed so that only the bound can see it.
#[test]
fn verify_bounds_the_round_count_before_replaying_rounds() {
    let temp = tempfile::tempdir().unwrap();
    let opt = run_options(temp.path(), 40, four_parents(), simplify());
    let receipt = run::run(&opt).unwrap();
    let s = &receipt.search;
    let bound = BARREN_ROUND_LIMIT * (s.completed_evaluations + 1);
    assert!(s.generations <= bound, "{} > {bound}", s.generations);
    let output = &opt.run.output;
    let mut edited = receipt.clone();
    edited.search.generations = bound + 1;
    let summary = serde_json::to_vec_pretty(&edited.search).unwrap();
    std::fs::write(output.join("search/cut-0/search.json"), &summary).unwrap();
    edited.search_sha256 = crate::sha256_hex(&summary);
    let path = output.join("receipt.json");
    std::fs::write(&path, serde_json::to_vec(&edited).unwrap()).unwrap();
    let error = publication::verify_run(&path).unwrap_err();
    assert!(
        format!("{error:#}").contains("round count exceeds"),
        "{error:#}"
    );
}
