use super::*;

#[test]
pub(crate) fn ablation_a_equals_a_replays_byte_for_byte_and_reports_no_difference() {
    let config = small_config();
    let (first_dir, second_dir) = (scratch(), scratch());
    let first = run(&config, first_dir.path()).unwrap();
    let second = run(
        &Config {
            jobs: 2,
            ..config.clone()
        },
        second_dir.path(),
    )
    .unwrap();
    assert_eq!(
        String::from_utf8_lossy(&first.body_bytes),
        String::from_utf8_lossy(&second.body_bytes)
    );
    assert_eq!(first.document.body_sha256, second.document.body_sha256);
    assert_eq!(
        first.document.body_sha256,
        sha256_hex(&serde_json::to_vec(&first.document.body).unwrap())
    );

    let body = &first.document.body;
    assert_eq!(body.verdict, NO_DIFFERENCE);
    for (name, gate) in body.gates.all() {
        assert!(
            gate.status == SAME || gate.status == NOT_EVALUATED,
            "{name}: {}",
            gate.status
        );
    }
    assert_eq!(body.gates.g_a.status, SAME);
    assert_eq!(body.datasets.len(), 2);
    assert_eq!(body.runs.len(), 2 * 2 * 2);
    assert_eq!(body.aggregates[0].overall, body.aggregates[1].overall);
    for record in &body.runs {
        let m = &record.metrics;
        assert!(m.m1.rmse.is_finite() && m.m1.rmse >= 0.0, "{record:?}");
        assert!(m.m2.programs >= 1 && m.m2.programs <= TOP_MEDIAN);
        assert!(m
            .m3
            .per_cut
            .iter()
            .all(|v| (1..=SMALL_EVALUATIONS as u64).contains(v)));
        assert!(m.m4.is_none() && m.m5.is_none() && m.m9.is_none());
        assert_eq!(m.m6.attempted, SMALL_EVALUATIONS as u64);
        assert!(m.m7.programs >= 1 && m.m7.mean_stages >= 1.0);
        assert!(m.m8.unsupported.is_none(), "{:?}", m.m8);
        assert!(m.m8.python_tokens.unwrap() > 0 && m.m8.rust_bytes.unwrap() > 0);
        for cut in &record.cuts {
            assert!(
                cut.exploration_stopped_by == search::STOPPED_BY_EVALUATIONS
                    || cut.exploration_stopped_by == search::STOPPED_BY_EXHAUSTED,
                "{cut:?}"
            );
        }
    }
    // Sixteen evaluations reach the first derived candidates.
    assert!(body.runs.iter().any(|r| r.metrics.m11.valid_derived > 0));
    // Wall time lives outside the body.
    let text = String::from_utf8(first.body_bytes).unwrap();
    assert!(!text.contains("seconds_per") && !text.contains("\"M10\""));
    assert_eq!(first.document.timing.runs.len(), body.runs.len());
    assert!(first.document.timing.m10["A"] > 0.0);
}

#[test]
pub(crate) fn ablation_metric_definitions() {
    // M3: 95% of the improvement from the first measured candidate to the final best.
    let curve = [
        None,
        Some(10.0),
        Some(12.0),
        Some(6.0),
        Some(5.2),
        Some(5.0),
    ];
    // gap 5.0; 95% is 4.75 -> the running best must reach 5.25: the fifth evaluation.
    assert_eq!(evaluations_to_fraction(&curve, 0.95), Some(5));
    assert_eq!(evaluations_to_fraction(&curve, 0.5), Some(4));
    assert_eq!(
        evaluations_to_fraction(&[Some(3.0), Some(4.0)], 0.95),
        Some(1)
    );
    assert_eq!(evaluations_to_fraction(&[None, None], 0.95), None);
    assert_eq!(median(&[3.0, 1.0, 2.0]), Some(2.0));
    assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), Some(2.5));
    assert_eq!(median(&[]), None);
    assert_eq!(sample_sd(&[1.0]), None);
    assert_eq!(sample_sd(&[1.0, 3.0]), Some(2f64.sqrt()));
    let expr = MathExpr::parse("sin(c0) + 2 * c1").unwrap();
    assert_eq!(expression_size(expr.ast()), (6, 3));
    let (stages, expressions) = program_shape(
        "(pipe X (derive expr=\"sin(c0) + 2 * c1\" append=true) (standardize) (ridge))",
    )
    .unwrap();
    assert_eq!((stages, expressions), (3, vec![(6, 3)]));
    assert_eq!(program_shape("(pipe X (ridge))").unwrap(), (1, vec![]));
    for origin in DERIVED_ORIGINS {
        assert!(search::ORIGINS.contains(&origin), "{origin}");
    }
}

#[test]
pub(crate) fn ablation_gates_report_a_difference() {
    let a = slot("A", [1.0, 2.0], Some(0.1), 10.0);
    // B is worse on d1 by more than 0.25 sigma: G-A fails.
    let worse = slot("B", [1.0, 2.1], Some(0.1), 10.0);
    let gate = quality_gate(&a, &worse);
    assert_eq!(gate.status, FAIL);
    assert!(gate.per_dataset[1].worse);
    // B is better by at least one sigma on both datasets: G-A passes.
    let better = slot("B", [0.8, 1.8], Some(0.1), 10.0);
    assert_eq!(quality_gate(&a, &better).status, PASS);
    // Equal quality but 20% faster: G-A passes through M3.
    let faster = slot("B", [1.0, 2.0], Some(0.1), 8.0);
    assert_eq!(quality_gate(&a, &faster).status, PASS);
    // One seed leaves sigma undefined.
    let one_seed = slot("B", [0.9, 1.9], None, 10.0);
    assert_eq!(
        quality_gate(&slot("A", [1.0, 2.0], None, 10.0), &one_seed).status,
        NOT_EVALUATED
    );
    // Identical aggregates: no difference.
    assert_eq!(quality_gate(&a, &a.clone()).status, SAME);
    // G-D: a token cost 10% higher fails.
    let mut costly = a.clone();
    costly.overall.m8_tokens = Some(1100.0);
    let g = gates(Profile::Legacy, Profile::Legacy, &a, &costly);
    assert_eq!(g.g_d.status, FAIL);
    assert_eq!(g.g_c.status, SAME);
    assert!(!g.g_c.complete);
    assert_eq!(g.g_e.status, NOT_EVALUATED);
    // G-D: an M8 value some run lacks was never compared, so its clauses are not evaluated,
    // even when both profiles lack it; M7 expression size missing from both is no difference.
    let clauses = |g: &Gates, metric: &str| -> Vec<&'static str> {
        g.g_d
            .clauses
            .iter()
            .filter(|c| c.metric == metric)
            .map(|c| c.status)
            .collect()
    };
    let mut refused = a.clone();
    refused.overall.m8_tokens = None;
    refused.overall.m8_bytes = None;
    let neither = gates(Profile::Legacy, Profile::Legacy, &refused, &refused.clone());
    assert_eq!(clauses(&neither, "M8"), [NOT_EVALUATED; 2]);
    assert_eq!(clauses(&neither, "M7"), [SAME; 2]);
    assert_eq!(neither.g_d.status, SAME);
    assert!(!neither.g_d.complete);
    let one = gates(Profile::Legacy, Profile::Legacy, &a, &refused);
    assert_eq!(clauses(&one, "M8"), [NOT_EVALUATED; 2]);
    assert!(!one.g_d.complete);
    let both = gates(Profile::Legacy, Profile::Legacy, &a, &a.clone());
    assert_eq!(clauses(&both, "M8"), [SAME; 2]);
    assert!(both.g_d.complete);
}

#[test]
pub(crate) fn ablation_refuses_a_phase_the_seconds_bound_ended() {
    let phase = |stopped_by: &str| jope_pipeline::receipt::Phase {
        evaluation_limit: 20,
        seconds_limit: 600.0,
        completed_evaluations: 3,
        successful_measurements: 3,
        failed_measurements: 0,
        seconds: 600.5,
        stopped_by: stopped_by.to_owned(),
    };
    let cut = |exploration: &str, seconds: f64| CutSearch {
        cut: 0,
        output: PathBuf::new(),
        train_rows: 10,
        hold_rows: 3,
        generations: 1,
        completed_evaluations: 3,
        successful_measurements: 3,
        failed_or_unavailable_measurements: 0,
        stopped_by: search::STOPPED_BY_EVALUATIONS,
        seconds,
        exploration: phase(exploration),
        comparison: jope_pipeline::receipt::Phase::not_requested(),
        best_validation_rmse: None,
        best_validation_r2: None,
        best_validation_objective: None,
        best_program: None,
        migrations: Vec::new(),
        observations: Vec::new(),
        search_sha256: String::new(),
        observations_sha256: String::new(),
        rng_seed: 1,
        budget: Budget {
            evaluations: 20,
            seconds: SEARCH_SECONDS,
        },
    };
    check_replayable(&[cut(search::STOPPED_BY_EVALUATIONS, 1.0)]).unwrap();
    let error = check_replayable(&[cut(search::STOPPED_BY_SECONDS, 1.0)]).unwrap_err();
    assert!(error.to_string().contains("replayable"), "{error:#}");
    let error = check_replayable(&[cut(search::STOPPED_BY_EVALUATIONS, 601.0)]).unwrap_err();
    assert!(error.to_string().contains("overran"), "{error:#}");
}

#[test]
pub(crate) fn ablation_profiles_and_inputs_are_checked() {
    assert_eq!(Profile::parse("legacy").unwrap(), Profile::Legacy);
    let error = Profile::parse("other").unwrap_err().to_string();
    assert!(error.contains("known: legacy"), "{error}");
    // Without --scratch the scratch root is TMPDIR, and never the system default.
    assert_eq!(
        default_scratch_root(Some("/data/scratch".into())).unwrap(),
        PathBuf::from("/data/scratch")
    );
    for unset in [None, Some(std::ffi::OsString::new())] {
        let error = default_scratch_root(unset).unwrap_err().to_string();
        assert!(
            error.contains("--scratch") && error.contains("TMPDIR"),
            "{error}"
        );
    }
    let dir = scratch();
    for broken in [
        Config {
            seeds: vec![],
            ..small_config()
        },
        Config {
            seeds: vec![1, 1],
            ..small_config()
        },
        Config {
            evaluations: search::MAX_EVALUATIONS + 1,
            ..small_config()
        },
        Config {
            cuts: prepare::FOLDS + 1,
            ..small_config()
        },
        Config {
            datasets: vec![d5("pmlb-229-pwlinear"), d5("pmlb-229-pwlinear")],
            ..small_config()
        },
    ] {
        assert!(run(&broken, dir.path()).is_err(), "{broken:?}");
    }
}

/// The ten D5 regression datasets, five seeds, A=A twice. Minutes of CPU: run it on purpose
/// with `JOPE_ABLATION_HEAVY=1` and `--run-ignored only`.
#[test]
#[ignore = "heavy: the ten D5 regression datasets x five seeds; set JOPE_ABLATION_HEAVY=1"]
pub(crate) fn ablation_heavy_a_equals_a_on_every_d5_regression_dataset() {
    assert!(
        std::env::var_os("JOPE_ABLATION_HEAVY").is_some_and(|v| v == "1"),
        "the heavy ablation runs only with JOPE_ABLATION_HEAVY=1"
    );
    let config = Config {
        datasets: vec![Path::new(env!("CARGO_MANIFEST_DIR")).join("../../datasets/test")],
        seeds: vec![1, 2, 3, 4, 5],
        evaluations: 64,
        jobs: std::thread::available_parallelism()
            .map_or(1, usize::from)
            .min(8),
        ..small_config()
    };
    let (first_dir, second_dir) = (scratch(), scratch());
    let first = run(&config, first_dir.path()).unwrap();
    let second = run(&config, second_dir.path()).unwrap();
    assert_eq!(first.body_bytes, second.body_bytes);
    assert_eq!(first.document.body.datasets.len(), 10);
    assert_eq!(first.document.body.verdict, NO_DIFFERENCE);
}
