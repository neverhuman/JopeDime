use super::*;

#[test]
pub(super) fn ablation_a_equals_a_replays_byte_for_byte_and_reports_no_difference() {
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
pub(super) fn ablation_metric_definitions() {
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

pub(super) fn summary_with(m1_rmse: f64, m3: f64) -> Summary {
    Summary {
        m1_rmse,
        m1_r2: 0.5,
        m2_rmse: m1_rmse,
        m3,
        m6_rate: 0.0,
        m7_stages: 2.0,
        m7_expression_nodes: None,
        m8_tokens: Some(1000.0),
        m8_bytes: Some(9000.0),
        m11: 0.3,
    }
}

pub(super) fn slot(name: &'static str, rmse: [f64; 2], sd: Option<f64>, m3: f64) -> SlotAggregate {
    SlotAggregate {
        slot: name,
        profile: "legacy",
        datasets: rmse
            .iter()
            .enumerate()
            .map(|(i, r)| DatasetAggregate {
                dataset: format!("d{i}"),
                seeds: 5,
                m1_rmse_seed_sd: sd,
                mean: summary_with(*r, m3),
            })
            .collect(),
        overall: summary_with((rmse[0] + rmse[1]) / 2.0, m3),
    }
}

#[test]
pub(super) fn ablation_gates_report_a_difference() {
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
pub(super) fn ablation_refuses_a_phase_the_seconds_bound_ended() {
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
pub(super) fn ablation_profiles_and_inputs_are_checked() {
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
pub(super) fn ablation_heavy_a_equals_a_on_every_d5_regression_dataset() {
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

/// Evaluations of the PR-tier kernel2 comparison, and its refinement allowance: one refined
/// parent per round with two objective trials.
pub(super) const KERNEL2_EVALUATIONS: usize = 16;
pub(super) const SMALL_REFINEMENT: Refinement = Refinement {
    top_k: 1,
    trials_per_program: 2,
};

/// The PR-tier kernel2 comparison: the legacy comparison's two datasets with kernel2-baseline
/// on both sides and seed 6 only. Seed 5 draws boosted-tree heads of 1000 rounds there, which
/// take 13 to 20 seconds a search in a debug build; seed 6's take about 3 and still spend
/// refinement trials.
pub(super) fn kernel2_config() -> Config {
    Config {
        profile_a: Profile::Kernel2Baseline,
        profile_b: Profile::Kernel2Baseline,
        seeds: vec![6],
        evaluations: KERNEL2_EVALUATIONS,
        refinement: Some(SMALL_REFINEMENT),
        ..small_config()
    }
}

/// kernel2-baseline against itself, twice with different job counts: byte-identical bodies
/// (wall time is in `timing`, never in the body), no difference, the route's own winner as
/// M1, consistent budget accounting, and a paired comparison of nothing but ties.
#[test]
pub(super) fn ablation_kernel2_a_equals_a_replays_byte_for_byte() {
    let config = kernel2_config();
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

    let body = &first.document.body;
    assert_eq!(body.verdict, NO_DIFFERENCE);
    assert_eq!(body.gates.g_a.status, SAME);
    assert_eq!(body.gates.g_b.status, SAME);
    // The profile record names the shipped profile, its digest, the policy and the allowance.
    let shipped = feat_math::kernel::profile::shipped_kernel2_profiles()
        .into_iter()
        .find(|p| p.id == "kernel2-baseline")
        .unwrap();
    let record = &body.protocol.profile_a;
    assert_eq!(record.id, "kernel2-baseline");
    assert_eq!(record.search, kernel2::receipt::ENGINE);
    assert_eq!((record.threads, record.islands), (1, 1));
    let settings = record.kernel2.as_ref().unwrap();
    assert_eq!(settings.profile_sha256, shipped.sha256());
    assert_eq!(settings.revision, shipped.revision);
    assert_eq!(settings.fault_policy, PolicyRecord::RejectHard);
    assert_eq!(settings.refinement, SMALL_REFINEMENT);
    assert_eq!(settings.selection, kernel2::receipt::SELECTION);
    assert_eq!(body.protocol.profile_a, body.protocol.profile_b);

    assert_eq!(body.runs.len(), 2 * 2);
    for searched in &body.runs {
        let m = &searched.metrics;
        assert!(m.m1.rmse.is_finite() && m.m1.rmse >= 0.0, "{searched:?}");
        assert_eq!(m.m1.cuts_measured, 1);
        assert_eq!(m.m1.program_sha256, sha256_hex(m.m1.program.as_bytes()));
        let [cut] = searched.cuts.as_slice() else {
            panic!("one cut: {searched:?}");
        };
        let accounting = cut.kernel2.as_ref().unwrap();
        assert_eq!(
            cut.completed_evaluations as usize,
            accounting.measurements + accounting.objective_trials
        );
        assert!(cut.completed_evaluations as usize <= KERNEL2_EVALUATIONS);
        assert_eq!(m.m6.attempted as usize, accounting.measurements);
        assert!(REPLAYABLE_STOPS.contains(&cut.exploration_stopped_by.as_str()));
        assert_eq!(cut.comparison_stopped_by, search::NOT_REQUESTED);
    }
    // The allowance is spent: some run charged refinement trials.
    assert!(body
        .runs
        .iter()
        .any(|r| r.cuts[0].kernel2.as_ref().unwrap().objective_trials > 0));

    let paired = body.paired.as_ref().unwrap();
    assert_eq!(paired.definitions, PAIRED_DEFINITIONS);
    let all = &paired.overall;
    assert_eq!((all.pairs, all.wins, all.ties, all.losses), (2, 0, 2, 0));
    assert_eq!((all.sign_test_p, all.same_winner), (1.0, 2));
    assert_eq!(all.median_a, all.median_b);
    assert_eq!((all.median_difference, all.mean_difference), (0.0, 0.0));
    assert_eq!(paired.per_dataset.len(), 2);
    for (dataset, record) in paired.per_dataset.iter().zip(&body.datasets) {
        assert_eq!(dataset.dataset, record.dataset_hash);
        assert_eq!((dataset.counts.pairs, dataset.counts.ties), (1, 1));
    }

    let text = String::from_utf8(first.body_bytes.clone()).unwrap();
    assert!(!text.contains("seconds_per") && !text.contains("\"active_seconds\""));
    assert!(!text.contains("\"M10\""));
    let seconds = first.document.timing.paired.as_ref().unwrap();
    assert!(seconds.total_seconds_a > 0.0 && seconds.total_seconds_b > 0.0);
    assert_eq!(first.document.timing.runs.len(), body.runs.len());
    // Each search's evidence is published under the scratch directory.
    let evidence = first_dir
        .path()
        .join("search/A")
        .join(&body.datasets[0].dataset_hash)
        .join("seed-6/search/cut-0/search.json");
    assert!(evidence.is_file(), "{}", evidence.display());
}

pub(super) fn sorted_keys(value: &serde_json::Value) -> Vec<String> {
    let mut keys: Vec<String> = value.as_object().unwrap().keys().cloned().collect();
    keys.sort();
    keys
}

/// The kernel2 arms leave the legacy document as it was: the legacy profile record and a
/// legacy cut record serialize to the bytes they had before, and a legacy-only run's body,
/// run records and timing carry exactly the keys they had before.
#[test]
pub(super) fn ablation_legacy_document_is_unchanged_by_the_kernel2_arms() {
    let budget = Budget {
        evaluations: SMALL_EVALUATIONS,
        seconds: SEARCH_SECONDS,
    };
    assert_eq!(
        serde_json::to_string(&Profile::Legacy.record(budget, None).unwrap()).unwrap(),
        "{\"id\":\"legacy\",\"search\":\"feat-math canonical-pipeline GP (in-process)\",\
             \"candidate_source\":\"feat_math::evolve canonical-pipeline wrappers, v1\",\
             \"candidates_per_round\":12,\"carried_per_round\":4,\"max_transforms\":4,\
             \"threads\":1,\"islands\":1}"
    );
    let cut = CutRecord {
        cut: 0,
        generations: 2,
        completed_evaluations: 16,
        successful_measurements: 15,
        failed_measurements: 1,
        exploration_stopped_by: search::STOPPED_BY_EVALUATIONS.to_owned(),
        comparison_stopped_by: search::NOT_REQUESTED.to_owned(),
        kernel2: None,
    };
    assert_eq!(
        serde_json::to_string(&cut).unwrap(),
        "{\"cut\":0,\"generations\":2,\"completed_evaluations\":16,\
             \"successful_measurements\":15,\"failed_measurements\":1,\
             \"exploration_stopped_by\":\"evaluations\",\
             \"comparison_stopped_by\":\"not_requested\"}"
    );

    let config = Config {
        datasets: vec![d5("pmlb-first-principles-supernovae-zg")],
        seeds: vec![5],
        ..small_config()
    };
    let dir = scratch();
    let report = run(&config, dir.path()).unwrap();
    let body: serde_json::Value = serde_json::from_slice(&report.body_bytes).unwrap();
    assert_eq!(
        sorted_keys(&body),
        [
            "aggregates",
            "datasets",
            "gates",
            "protocol",
            "runs",
            "verdict"
        ]
    );
    assert_eq!(
        sorted_keys(&body["protocol"]["profile_a"]),
        [
            "candidate_source",
            "candidates_per_round",
            "carried_per_round",
            "id",
            "islands",
            "max_transforms",
            "search",
            "threads"
        ]
    );
    assert_eq!(
        sorted_keys(&body["runs"][0]),
        ["cuts", "dataset", "metrics", "profile", "seed", "slot"]
    );
    assert_eq!(
        sorted_keys(&body["runs"][0]["cuts"][0]),
        [
            "comparison_stopped_by",
            "completed_evaluations",
            "cut",
            "exploration_stopped_by",
            "failed_measurements",
            "generations",
            "successful_measurements"
        ]
    );
    let text = String::from_utf8(report.body_bytes.clone()).unwrap();
    assert!(!text.contains("\"paired\"") && !text.contains("\"kernel2\""));
    let timing = serde_json::to_value(&report.document.timing).unwrap();
    assert_eq!(
        sorted_keys(&timing),
        ["G-D-M10", "M10", "runs", "wall_seconds"]
    );
}

/// The kernel2 arms refuse what `jope pipeline run` refuses for a kernel2 profile, before any
/// search starts, and the command line refuses a refinement option given alone.
#[test]
pub(super) fn ablation_kernel2_refuses_what_the_pipeline_refuses() {
    assert_eq!(
        Profile::parse("kernel2-baseline").unwrap(),
        Profile::Kernel2Baseline
    );
    assert_eq!(
        Profile::parse("kernel2-experimental").unwrap(),
        Profile::Kernel2Experimental
    );
    let error = Profile::parse("kernel2").unwrap_err().to_string();
    assert!(
        error.contains("known: legacy, kernel2-baseline, kernel2-experimental"),
        "{error}"
    );
    for profile in Profile::ALL {
        assert_eq!(profile.is_kernel2(), profile.id().starts_with("kernel2-"));
    }

    // Accepted: a kernel2 profile on either side or both, with the allowance.
    kernel2_config().check().unwrap();
    Config {
        profile_a: Profile::Legacy,
        ..kernel2_config()
    }
    .check()
    .unwrap();
    Config {
        profile_b: Profile::Kernel2Experimental,
        ..kernel2_config()
    }
    .check()
    .unwrap();

    let refused = |config: Config, needle: &str| {
        let error = format!("{:#}", config.check().unwrap_err());
        assert!(error.contains(needle), "{needle:?} not in {error:?}");
    };
    // A kernel2 profile without the allowance.
    refused(
        Config {
            refinement: None,
            ..kernel2_config()
        },
        "--refine-top-k",
    );
    refused(
        Config {
            profile_a: Profile::Legacy,
            refinement: None,
            ..kernel2_config()
        },
        "--refine-trials",
    );
    // The allowance without a kernel2 profile.
    refused(
        Config {
            refinement: Some(SMALL_REFINEMENT),
            ..small_config()
        },
        "legacy profile refuses",
    );
    // More than one cut.
    refused(
        Config {
            cuts: 2,
            ..kernel2_config()
        },
        "--cuts 1",
    );
    // More refined parents than the carried set; more trials than the cut's budget.
    for refinement in [
        Refinement {
            top_k: search::ELITES + 1,
            trials_per_program: 2,
        },
        Refinement {
            top_k: 1,
            trials_per_program: KERNEL2_EVALUATIONS + 1,
        },
    ] {
        refused(
            Config {
                refinement: Some(refinement),
                ..kernel2_config()
            },
            "refinement allowance",
        );
    }
    // run refuses before preparing a cut or starting a search.
    let dir = scratch();
    assert!(run(
        &Config {
            cuts: 2,
            ..kernel2_config()
        },
        dir.path()
    )
    .is_err());
    assert!(!dir.path().join("prepared").exists() && !dir.path().join("search").exists());

    // The command line: the two options go together, and only with a kernel2 profile.
    let args = |a: &str, b: &str, top_k: Option<usize>, trials: Option<usize>| AblationArgs {
        profile_a: a.to_owned(),
        profile_b: b.to_owned(),
        refine_top_k: top_k,
        refine_trials: trials,
        datasets: vec![d5("pmlb-first-principles-supernovae-zg")],
        seeds: vec![5],
        evaluations: KERNEL2_EVALUATIONS,
        cuts: 1,
        fold_seed: jope_pipeline::DEFAULT_SEED,
        jobs: 1,
        out: PathBuf::from("unused.json"),
        scratch: None,
    };
    let parsed = config(&args("legacy", "kernel2-baseline", Some(1), Some(2))).unwrap();
    assert_eq!(
        (parsed.profile_a, parsed.profile_b, parsed.refinement),
        (
            Profile::Legacy,
            Profile::Kernel2Baseline,
            Some(SMALL_REFINEMENT)
        )
    );
    assert_eq!(
        config(&args("legacy", "legacy", None, None))
            .unwrap()
            .refinement,
        None
    );
    for (profile, top_k, trials, needle) in [
        ("kernel2-baseline", None, Some(2), "--refine-top-k"),
        ("kernel2-experimental", Some(1), None, "--refine-trials"),
        ("legacy", Some(1), None, "legacy profile refuses"),
        ("legacy", None, Some(2), "legacy profile refuses"),
    ] {
        let error = config(&args(profile, profile, top_k, trials))
            .unwrap_err()
            .to_string();
        assert!(error.contains(needle), "{needle:?} not in {error:?}");
    }
    let both = config(&args("legacy", "legacy", Some(1), Some(2))).unwrap();
    assert!(format!("{:#}", both.check().unwrap_err()).contains("legacy profile refuses"));
}

/// A run record carrying what the paired comparison reads: slot, dataset, seed and M1.
pub(super) fn paired_run(
    slot: &'static str,
    dataset: &str,
    seed: u64,
    rmse: f64,
    program: &str,
) -> RunRecord {
    RunRecord {
        slot,
        profile: "kernel2-baseline",
        dataset: dataset.to_owned(),
        seed,
        metrics: Metrics {
            m1: Best {
                rmse,
                r2: 0.5,
                cuts_measured: 1,
                program_sha256: sha256_hex(program.as_bytes()),
                program: program.to_owned(),
            },
            m2: TopMedian {
                programs: 1,
                rmse,
                r2: 0.5,
            },
            m3: Speed {
                per_cut: vec![1],
                mean: 1.0,
            },
            m4: None,
            m5: None,
            m6: Invalid {
                rate: 0.0,
                invalid: 0,
                attempted: 1,
                by_status: BTreeMap::new(),
            },
            m7: Shape {
                programs: 1,
                mean_stages: 1.0,
                expression_stages: 0,
                mean_expression_nodes: None,
                mean_expression_depth: None,
                max_expression_nodes: None,
                max_expression_depth: None,
            },
            m8: Cost {
                python_bytes: None,
                rust_bytes: None,
                python_tokens: None,
                rust_tokens: None,
                unsupported: None,
            },
            m9: None,
            m11: Yield {
                per_evaluation: 0.0,
                valid_derived: 0,
                evaluations: 1,
            },
        },
        cuts: Vec::new(),
    }
}

/// The paired comparison on a synthetic input whose every value is exact in binary: wins,
/// ties and losses overall and per dataset, both medians, the differences, the largest gain
/// and the mean without it, the sign test, the same-winner count, an all-tie input, the
/// bit-equality rule for a tie, and the refusal of unpaired runs.
#[test]
pub(super) fn ablation_paired_summary_arithmetic() {
    let runs: Vec<RunRecord> = [
        // (dataset, seed, A RMSE, B RMSE, A program, B program)
        ("d0", 1, 1.0, 0.75, "p", "q"), // win, -0.25
        ("d0", 2, 1.0, 1.0, "p", "p"),  // tie, same program
        ("d0", 3, 1.0, 1.5, "p", "q"),  // loss, +0.5
        ("d1", 1, 2.0, 1.5, "p", "q"),  // win, -0.5
        ("d1", 2, 2.0, 1.75, "p", "q"), // win, -0.25
        ("d1", 3, 2.0, 2.0, "p", "p"),  // tie, same program
    ]
    .iter()
    .flat_map(|&(dataset, seed, a, b, program_a, program_b)| {
        [
            paired_run("A", dataset, seed, a, program_a),
            paired_run("B", dataset, seed, b, program_b),
        ]
    })
    .collect();
    let datasets = vec!["d0".to_owned(), "d1".to_owned()];
    let summary = paired(&datasets, &runs).unwrap();
    assert_eq!(
        (summary.definitions, summary.rule),
        (PAIRED_DEFINITIONS, PAIRED_RULE)
    );
    let all = &summary.overall;
    assert_eq!((all.pairs, all.wins, all.ties, all.losses), (6, 3, 2, 1));
    assert_eq!((all.median_a, all.median_b), (1.5, 1.5));
    assert_eq!(all.median_difference, -0.125);
    assert_eq!(all.mean_difference, -0.5 / 6.0);
    assert_eq!(
        all.largest_gain,
        PairDifference {
            dataset: "d1".to_owned(),
            seed: 1,
            difference: -0.5
        }
    );
    assert_eq!(all.mean_difference_without_largest_gain, Some(0.0));
    // 2 * (C(4,0) + C(4,1)) / 2^4.
    assert_eq!(all.sign_test_p, 0.625);
    assert_eq!(all.same_winner, 2);

    let [d0, d1] = summary.per_dataset.as_slice() else {
        panic!("two datasets: {summary:?}");
    };
    assert_eq!((d0.dataset.as_str(), d1.dataset.as_str()), ("d0", "d1"));
    let c = &d0.counts;
    assert_eq!((c.pairs, c.wins, c.ties, c.losses), (3, 1, 1, 1));
    assert_eq!(
        (c.median_a, c.median_b, c.median_difference),
        (1.0, 1.0, 0.0)
    );
    assert_eq!(c.mean_difference, 0.25 / 3.0);
    assert_eq!((c.largest_gain.seed, c.largest_gain.difference), (1, -0.25));
    assert_eq!(c.mean_difference_without_largest_gain, Some(0.25));
    assert_eq!((c.sign_test_p, c.same_winner), (1.0, 1));
    let c = &d1.counts;
    assert_eq!((c.pairs, c.wins, c.ties, c.losses), (3, 2, 1, 0));
    assert_eq!(
        (c.median_a, c.median_b, c.median_difference),
        (2.0, 1.75, -0.25)
    );
    assert_eq!(c.mean_difference, -0.25);
    assert_eq!(c.mean_difference_without_largest_gain, Some(-0.125));
    // 2 * C(2,0) / 2^2.
    assert_eq!((c.sign_test_p, c.same_winner), (0.5, 1));

    // Nothing but ties: no untied pair, so p is 1, and every difference is zero.
    let ties: Vec<RunRecord> = [1, 2, 3]
        .iter()
        .flat_map(|&seed| {
            [
                paired_run("A", "d0", seed, 0.5, "p"),
                paired_run("B", "d0", seed, 0.5, "p"),
            ]
        })
        .collect();
    let tied = paired(&datasets[..1], &ties).unwrap().overall;
    assert_eq!(
        (tied.wins, tied.ties, tied.losses, tied.same_winner),
        (0, 3, 0, 3)
    );
    assert_eq!(tied.sign_test_p, 1.0);
    assert_eq!((tied.median_difference, tied.mean_difference), (0.0, 0.0));
    assert_eq!(tied.largest_gain.seed, 1);
    assert_eq!(tied.mean_difference_without_largest_gain, Some(0.0));

    // A tie is bit-equality: the next RMSE above A's is a loss.
    let near = [
        paired_run("A", "d0", 1, 1.0, "p"),
        paired_run("B", "d0", 1, f64::from_bits(1.0_f64.to_bits() + 1), "q"),
    ];
    let near = paired(&datasets[..1], &near).unwrap().overall;
    assert_eq!((near.wins, near.ties, near.losses), (0, 0, 1));
    assert_eq!(near.mean_difference_without_largest_gain, None);

    // Every A run needs exactly one B run, and every B run an A run.
    assert!(paired(&datasets, &runs[..runs.len() - 1]).is_err());
    for extra in [
        paired_run("B", "d0", 9, 1.0, "p"),
        paired_run("B", "d0", 1, 1.0, "p"),
    ] {
        let mut runs = runs.clone();
        runs.push(extra);
        assert!(paired(&datasets, &runs).is_err());
    }
}
