use super::*;

#[test]
fn entropy_field_rename_preserves_telemetry_wire_key() {
    let mut summary = test_gen_summary("test", 1, 0.5);
    summary.op_entropy_nat = Some(0.75);

    let value = serde_json::to_value(summary).expect("serialize generation summary");
    assert_eq!(value["op_entropy_nat\u{73}"], 0.75);
    assert!(value.get("op_entropy_nat").is_none());
}

#[test]
fn parse_matches_python_reference_cases() {
    let dsl = "(pipe X (poly_interact) (bin_onehot append=true bins=14) (histgbm depth=6 lr=0.08 rounds=160))";
    let (stages, head) = parse_pipe_dsl(dsl);
    assert_eq!(
        stages,
        vec!["(poly_interact)", "(bin_onehot append=true bins=14)"]
    );
    assert_eq!(head, "(histgbm depth=6 lr=0.08 rounds=160)");
    assert_eq!(parse_pipe_dsl("(ridge)"), (vec![], "(ridge_gcv)"));
    assert_eq!(parse_pipe_dsl("(pipe X"), (vec![], "(ridge_gcv)"));
    assert_eq!(parse_pipe_dsl("(pipe X )"), (vec![], "(ridge_gcv)"));
    assert_eq!(
        parse_pipe_dsl("  (pipe X (a (b)) (c))  "),
        (vec!["(a (b))"], "(c)")
    );
    assert_eq!(clause_op("(winsor_clip q=0.07)"), "winsor_clip");
    assert_eq!(clause_op("(poly_interact)"), "poly_interact");
}

#[test]
fn interner_is_bounded_and_overflow_ids_stay_distinct() {
    let mut i = Interner::with_bounds(2, 1 << 20);
    let a = i.intern("(a)");
    let b = i.intern("(b)");
    let c = i.intern("(c)");
    let d = i.intern("(d)");
    assert_eq!((a, b), (0, 1));
    assert!(c & HASHED_ID_BIT != 0 && d & HASHED_ID_BIT != 0 && c != d);
    assert_eq!(i.intern("(c)"), c, "hashed ids are stable");
    assert_eq!((i.len(), i.overflowed()), (2, 3));
    let mut j = Interner::with_bounds(100, 4);
    j.intern("(ab)");
    assert!(
        j.intern("(cd)") & HASHED_ID_BIT != 0,
        "byte bound applies too"
    );
}

#[test]
fn candidate_genome_and_behavior_bounds_are_enforced() {
    let (mut ci, mut oi) = (Interner::default(), Interner::default());
    let long = format!("(pipe X {} (h))", "(s) ".repeat(100));
    let g = Genome::from_dsl(&long, &mut ci, &mut oi);
    assert!(g.truncated && g.clauses.len() == MAX_CLAUSES);
    assert_eq!(
        *g.clauses.last().unwrap(),
        ci.intern("(h)"),
        "the head is kept"
    );
    let cand = |i: usize| Candidate {
        program: program_sha256(&i.to_string()),
        genome: g.clone(),
        parent: None,
        edit_op: EditOp::Other,
        status: EvalStatus::Ok,
        gate_passed: true,
        fitness: -(i as f64),
        improved: None,
        hwm_winner: false,
        complexity: None,
        proj16: None,
        behavior: vec![0.5; 100],
        provenance: Provenance::Novel,
    };
    let cands: Vec<Candidate> = (0..12).map(cand).collect();
    let cfg = Config {
        max_candidates: 10,
        max_behavior_len: 4,
        ..Default::default()
    };
    let mut t = Tracker::new("c", cfg);
    let s = t.summarize(0, 0, &cands, ProposalCounters::default());
    assert_eq!(
        (s.pop_size, s.candidates_truncated, s.genomes_truncated),
        (10, 2, 10)
    );
    t.summarize(0, 1, &cands, ProposalCounters::default());
    assert!(t.archive.iter().all(|b| b.len() == 4));
    assert_eq!(ci.dictionary(), vec!["(s)", "(h)"]);
}

#[test]
fn levenshtein_and_spearman() {
    assert_eq!(levenshtein(&[1, 2, 3], &[1, 3]), 1);
    assert_eq!(levenshtein(&[], &[4, 5]), 2);
    assert_eq!(levenshtein(&[1, 2], &[2, 1]), 2);
    let r = spearman(&[1.0, 2.0, 3.0, 4.0], &[10.0, 20.0, 30.0, 40.0], 3).unwrap();
    assert!((r - 1.0).abs() < 1e-12);
    let r = spearman(&[1.0, 2.0, 3.0, 4.0], &[4.0, 3.0, 2.0, 1.0], 3).unwrap();
    assert!((r + 1.0).abs() < 1e-12);
    assert!(spearman(&[1.0, f64::NAN, 3.0], &[1.0, 2.0, 3.0], 3).is_none());
    assert!(spearman(&[1.0, 1.0, 1.0], &[1.0, 2.0, 3.0], 3).is_none());
}

#[test]
fn quantiles_match_numpy_linear() {
    let mut v = vec![4.0, 1.0, 3.0, 2.0];
    let q = Quantiles::of(&mut v);
    assert_eq!(q.p50, Some(2.5));
    assert!((q.p10.unwrap() - 1.3).abs() < 1e-12);
    assert!((q.iqr.unwrap() - 1.5).abs() < 1e-12);
}

#[test]
fn sampled_pairs_are_distinct_and_seeded() {
    let mut a = SplitMix64(7);
    let mut b = SplitMix64(7);
    let pa = sample_pairs(&mut a, 100, 50);
    assert_eq!(pa, sample_pairs(&mut b, 100, 50));
    assert_eq!(pa.iter().collect::<HashSet<_>>().len(), 50);
    assert!(pa.iter().all(|&(i, j)| i < j && j < 100));
    assert_eq!(sample_pairs(&mut a, 4, 50).len(), 6);
}
