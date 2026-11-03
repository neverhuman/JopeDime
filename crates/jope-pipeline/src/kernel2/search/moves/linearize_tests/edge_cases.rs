use super::*;

/// A one-feature dataset in `dir` whose feature cycles through `-1e-310`, `0` and `1e-310` (75
/// training and 15 test rows, the target cycling through 0 to 6), with the pmlb-229-pwlinear
/// fixture's metadata rewritten for it. Every line fitted on these subnormal values has a slope
/// past the finite range. The review's fixture.

pub(super) fn subnormal_dataset(dir: &Path) -> String {
    let csv = |rows: std::ops::Range<usize>| -> String {
        rows.map(|i| format!("{},{}\n", ["-1e-310", "0", "1e-310"][i % 3], i % 7))
            .collect()
    };
    let (train, test) = (csv(0..75), csv(75..90));
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join("train.csv"), &train).unwrap();
    std::fs::write(dir.join("test.csv"), &test).unwrap();
    let scaffold = std::fs::read(Path::new(&dataset()).join("metadata.json")).unwrap();
    let mut meta: serde_json::Value = serde_json::from_slice(&scaffold).unwrap();
    let hash = crate::sha256_hex(format!("{train}{test}").as_bytes());
    meta["id"] = hash[..16].into();
    meta["dataset_hash"] = hash[..16].into();
    meta["canonical_sha256"] = hash.into();
    meta["display_name"] = "subnormal linearization fixture".into();
    meta["shape"] = serde_json::json!({
        "rows": 90, "features": 1, "categorical_features": 0, "columns": 2
    });
    for (name, bytes, rows) in [("train", &train, 75), ("test", &test, 15)] {
        meta["files"][name] = serde_json::json!({
            "sha256": crate::sha256_hex(bytes.as_bytes()),
            "bytes": bytes.len(),
            "rows": rows,
            "columns": 2,
            "has_header": false,
        });
    }
    meta["split"] = serde_json::json!({
        "method": "deterministic test fixture", "seed": 0, "train_rows": 75, "test_rows": 15
    });
    meta["provenance"]["dataset_id"] = "subnormal-linearization".into();
    meta["provenance"]["citation"] =
        "synthetic test fixture; metadata scaffold from the PMLB fixture, not PMLB data".into();
    meta["collections"] = serde_json::json!([]);
    meta["ingest"] = serde_json::Value::Null;
    std::fs::write(
        dir.join("metadata.json"),
        serde_json::to_vec_pretty(&meta).unwrap(),
    )
    .unwrap();
    dir.display().to_string()
}

/// A fit refused for a coefficient past the finite range is published without its coefficients
/// and read back as NaN (B19 wiring review 01, B19-R01-NAN-REFUSAL-ROUNDTRIP). The two equal
/// children of `(sin(c0)+c0)*(sin(c0)+c0)` on the subnormal dataset are each refused so, and the
/// untouched run verifies: a repeated subtree's records compare bit for bit. A finite pair given
/// to one of them is still refused.
#[test]
fn repeated_subtrees_refused_for_their_coefficients_publish_and_verify() {
    let temp = tempfile::tempdir().unwrap();
    let data = subnormal_dataset(&temp.path().join("dataset"));
    let opt = run_options_on(
        data,
        temp.path(),
        &[program("(sin(c0)+c0)*(sin(c0)+c0)")],
        40,
        four_parents(),
        linearize(),
    );
    let receipt = run::run(&opt).unwrap();
    let output = &opt.run.output;
    let path = output.join("receipt.json");
    assert_eq!(publication::verify_run(&path).unwrap(), receipt);
    let moves = published_moves(&opt);
    // The children at paths [0] and [1] of one parent's stage, each refused for its coefficients.
    let twin = |at: &[usize]| {
        moves
            .iter()
            .position(|m| m.id == MINIMAX_LINEARIZE && line(m).path == at)
            .unwrap()
    };
    let (left, right) = (twin(&[0]), twin(&[1]));
    assert_eq!(right, left + 1);
    for m in [&moves[left], &moves[right]] {
        let record = line(m);
        assert_eq!(record.subtree, "(sin(c0)+c0)");
        assert_eq!(record.refusal.as_deref(), Some(LINEARIZE_NON_FINITE));
        assert_eq!(m.outcome, REFUSED);
        assert!(record.fit.as_ref().unwrap().coefficients.is_none());
    }
    // A finite pair for one of the two: the twins record other samples.
    let engine = receipt.search.engine_selection.clone().unwrap();
    let edited = published_edit(output, &receipt, &engine, &moves, |m| {
        let record = m[right].linearization.as_mut().unwrap();
        record.fit.as_mut().unwrap().coefficients = Some([0.0, 1.0]);
    });
    let error = refused_with(
        &edited,
        "a finite pair for one twin",
        "a repeated subtree records other samples",
    );
    lint(&error.replace(&output.display().to_string(), "<run>"));
}

// Fix round r05 (B19 wiring review 05): a faulting stage input, and a subtree whose targets are
// zero on any rows.

/// An admitted parent whose first `derive` stage faults on its training rows (`c0/0`, a domain
/// fault the profile admits) gives its second stage no training input, so every subtree of that
/// stage is refused as [`PREFIX_FAULTS`]. With the pass the run completes, publishes and verifies,
/// as it does without it (B19-R05-PREFIX-FAULT-REFUSAL); a stage whose input faults for only some
/// of its subtrees stays refused (`verify_refuses_inflated_and_hidden_fits_and_moved_parent_groups`).
#[test]
fn a_faulting_stage_input_refuses_its_subtrees_and_the_run_publishes_and_verifies() {
    let temp = tempfile::tempdir().unwrap();
    let seed =
        GenomeIr::parse("(pipe X (derive expr=\"c0/0\") (derive expr=\"sin(c0)*3.0\") (ridge))")
            .unwrap()
            .canonical();
    let parents = Refinement {
        top_k: 4,
        trials_per_program: 0,
    };
    let mut runs = Vec::new();
    for (name, engine) in [("off", EngineSelection::default()), ("on", linearize())] {
        let opt = run_options(
            &temp.path().join(name),
            std::slice::from_ref(&seed),
            40,
            parents,
            engine,
        );
        let receipt = run::run(&opt).unwrap();
        let path = opt.run.output.join("receipt.json");
        assert_eq!(publication::verify_run(&path).unwrap(), receipt, "{name}");
        runs.push(opt);
    }
    let sha = crate::sha256_hex(seed.as_bytes());
    let moves = published_moves(&runs[1]);
    let faulted: Vec<&MoveRecord> = moves
        .iter()
        .filter(|m| m.parent_sha256 == sha && m.id == MINIMAX_LINEARIZE)
        .collect();
    assert!(!faulted.is_empty(), "the seed's stage was not read");
    for m in faulted {
        let record = line(m);
        assert_eq!(record.stage, 1);
        assert_eq!(record.refusal.as_deref(), Some(PREFIX_FAULTS));
        assert!(record.fit.is_none() && record.lowered_error.is_none());
        assert_eq!(m.outcome, REFUSED);
    }
}

/// `(c0-c0)*(c0-c0)` is zero wherever it is finite, so its targets are zero on any rows
/// (B19-R05-KNOWN-ZERO-TARGET-ENVELOPE). On the subnormal dataset its live fit is exact with zero
/// targets and coefficients past the finite range, published without them, and the run verifies;
/// the same records edited to a nonzero `max|f|` with a closed converged bracket and its bound,
/// consistent with their stop, are refused by name.
#[test]
fn a_zero_subtree_publishes_zero_targets_and_a_nonzero_record_of_it_is_refused() {
    let temp = tempfile::tempdir().unwrap();
    let data = subnormal_dataset(&temp.path().join("dataset"));
    let opt = run_options_on(
        data,
        temp.path(),
        &[program("(c0-c0)*(c0-c0)")],
        40,
        four_parents(),
        linearize(),
    );
    let receipt = run::run(&opt).unwrap();
    let output = &opt.run.output;
    let path = output.join("receipt.json");
    assert_eq!(publication::verify_run(&path).unwrap(), receipt);
    let moves = published_moves(&opt);
    let zero: Vec<usize> = (0..moves.len())
        .filter(|&i| {
            moves[i].id == MINIMAX_LINEARIZE
                && line(&moves[i]).subtree.contains("(c0-c0)*(c0-c0)")
                && line(&moves[i]).path.is_empty()
        })
        .collect();
    assert!(!zero.is_empty(), "no record of the zero subtree");
    for &i in &zero {
        let fit = line(&moves[i]).fit.as_ref().unwrap();
        assert_eq!((fit.max_abs, fit.stop.as_str()), (0.0, "exact_fit"));
        assert!(fit.coefficients.is_none(), "{fit:?}");
        assert_eq!(
            line(&moves[i]).refusal.as_deref(),
            Some(LINEARIZE_NON_FINITE)
        );
    }
    let engine = receipt.search.engine_selection.clone().unwrap();
    let edited = published_edit(output, &receipt, &engine, &moves, |m| {
        for &i in &zero {
            let record = m[i].linearization.as_mut().unwrap();
            let fit = record.fit.as_mut().unwrap();
            fit.max_abs = 1.0;
            fit.max_error = 0.25;
            fit.levelled = 0.25;
            fit.stop = "converged".into();
            record.bound = Some(bound_of(fit));
        }
    });
    let error = refused_with(
        &edited,
        "the zero subtree's targets made nonzero",
        "zero on every sample a nonzero target",
    );
    lint(&error.replace(&output.display().to_string(), "<run>"));
}

// Fix round r06 (B19 wiring review 06): a subtree with one value on every sample, and a range a
// norm never takes.

/// `(c0-c0)+1.0` takes the value 1 on every sample (B19-R06-KNOWN-TARGET-EVIDENCE). On the
/// subnormal dataset its live fit is the exact constant fit, published without its coefficients
/// (the conversion scales zero by infinity), and the run verifies; its records edited to a larger
/// `max|f|`, or to a converged bracket at the constant's own size, each with its bound, are
/// refused by name.
#[test]
fn a_one_value_subtree_publishes_its_exact_fit_and_edits_of_it_are_refused() {
    let temp = tempfile::tempdir().unwrap();
    let data = subnormal_dataset(&temp.path().join("dataset"));
    let opt = run_options_on(
        data,
        temp.path(),
        &[program("(c0-c0)+1.0")],
        40,
        four_parents(),
        linearize(),
    );
    let receipt = run::run(&opt).unwrap();
    let output = &opt.run.output;
    let path = output.join("receipt.json");
    assert_eq!(publication::verify_run(&path).unwrap(), receipt);
    let moves = published_moves(&opt);
    let one: Vec<usize> = (0..moves.len())
        .filter(|&i| {
            moves[i].id == MINIMAX_LINEARIZE
                && line(&moves[i]).subtree.contains("(c0-c0)+1")
                && line(&moves[i]).path.is_empty()
        })
        .collect();
    assert!(!one.is_empty(), "no record of the constant subtree");
    for &i in &one {
        let fit = line(&moves[i]).fit.as_ref().unwrap();
        assert_eq!(
            (fit.max_abs, fit.stop.as_str(), fit.iterations, fit.steps),
            (1.0, "exact_fit", 1, 1)
        );
    }
    let engine = receipt.search.engine_selection.clone().unwrap();
    let edited = |max_abs: f64| {
        published_edit(output, &receipt, &engine, &moves, |m| {
            for &i in &one {
                let record = m[i].linearization.as_mut().unwrap();
                let fit = record.fit.as_mut().unwrap();
                fit.max_abs = max_abs;
                fit.max_error = 0.25;
                fit.levelled = 0.25;
                fit.stop = "converged".into();
                record.bound = Some(bound_of(fit));
            }
        })
    };
    let larger = refused_with(
        &edited(2.0),
        "the constant's size doubled",
        "max|f| exceeds every value its subtree can take",
    );
    let bracket = refused_with(
        &edited(1.0),
        "a converged bracket at the constant's size",
        "is not that value's exact fit",
    );
    for error in [larger, bracket] {
        lint(&error.replace(&output.display().to_string(), "<run>"));
    }
}

/// The constant `(c1-c1)+1.0` on the default dataset's range (`c1` takes three values; `c0` two,
/// too few for a fit): the conversion's pair is finite there, `[1, 0]`, and the published record
/// keeps it; the same records published with the pair absent (read back as NaN) and the
/// non-finite refusal are refused by name (Lane B's review of #247). The subnormal run above keeps
/// its legitimately absent pair.
#[test]
fn a_constant_subtrees_finite_line_cannot_be_published_as_absent() {
    let temp = tempfile::tempdir().unwrap();
    let opt = run_options(
        temp.path(),
        &[program("(c1-c1)+1.0")],
        40,
        four_parents(),
        linearize(),
    );
    let receipt = run::run(&opt).unwrap();
    let output = &opt.run.output;
    let path = output.join("receipt.json");
    assert_eq!(publication::verify_run(&path).unwrap(), receipt);
    let moves = published_moves(&opt);
    let one: Vec<usize> = (0..moves.len())
        .filter(|&i| {
            moves[i].id == MINIMAX_LINEARIZE
                && line(&moves[i]).subtree.contains("(c1-c1)+1")
                && line(&moves[i]).path.is_empty()
        })
        .collect();
    assert!(!one.is_empty(), "no record of the constant subtree");
    for &i in &one {
        let fit = line(&moves[i]).fit.as_ref().unwrap();
        assert_eq!(fit.coefficients, Some([1.0, 0.0]), "{fit:?}");
    }
    let engine = receipt.search.engine_selection.clone().unwrap();
    let edited = published_edit(output, &receipt, &engine, &moves, |m| {
        for &i in &one {
            let record = m[i].linearization.as_mut().unwrap();
            record.fit.as_mut().unwrap().coefficients = None;
            record.line_nodes = None;
            record.lowered_error = None;
            record.expression = None;
            record.refusal = Some(LINEARIZE_NON_FINITE.into());
        }
    });
    let error = refused_with(
        &edited,
        "a constant's finite line published as absent",
        "is not that value's exact fit",
    );
    lint(&error.replace(&output.display().to_string(), "<run>"));
}

/// A row's L2 norm is never negative (B19-R06-AGGREGATE-INPUT-RANGE): a run seeded with
/// `sin((0.1*row_l2))*3.0` publishes and verifies, and its fits of the norm edited to a range with
/// a negative end are refused by name.
#[test]
fn a_norms_fit_over_a_range_with_a_negative_end_is_refused() {
    let temp = tempfile::tempdir().unwrap();
    let opt = run_options(
        temp.path(),
        &[program("sin((0.1*row_l2))*3.0")],
        40,
        four_parents(),
        linearize(),
    );
    let receipt = run::run(&opt).unwrap();
    let output = &opt.run.output;
    let path = output.join("receipt.json");
    assert_eq!(publication::verify_run(&path).unwrap(), receipt);
    let moves = published_moves(&opt);
    let norm: Vec<usize> = (0..moves.len())
        .filter(|&i| {
            moves[i].id == MINIMAX_LINEARIZE
                && line(&moves[i]).subtree.contains("row_l2")
                && line(&moves[i]).fit.is_some()
        })
        .collect();
    assert!(!norm.is_empty(), "no fit of the norm");
    let engine = receipt.search.engine_selection.clone().unwrap();
    let edited = published_edit(output, &receipt, &engine, &moves, |m| {
        for &i in &norm {
            let fit = m[i].linearization.as_mut().unwrap().fit.as_mut().unwrap();
            assert!(fit.lo >= 0.0);
            fit.lo = -0.5;
        }
    });
    let error = refused_with(
        &edited,
        "a norm's range moved under zero",
        "range holds values its variable never takes",
    );
    lint(&error.replace(&output.display().to_string(), "<run>"));
}
