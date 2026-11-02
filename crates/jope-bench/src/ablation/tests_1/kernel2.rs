use super::*;

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
