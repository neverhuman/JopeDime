use super::*;

#[test]
fn admitted_library_counts_only_matching_compact_gp_outputs_with_a_bounded_scan() {
    use jope_parallel::manifest::file_hash;
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let admission = root.join("runs/library-admission");
    std::fs::create_dir_all(&admission).unwrap();
    let write = |path: std::path::PathBuf, value: serde_json::Value| {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, serde_json::to_vec(&value).unwrap()).unwrap();
    };
    let jobs=(0..300).map(|cut|serde_json::json!({"prepared":"data/prepared/source","prepared_sha256":"prepared-hash","cut":cut,"output":format!("runs/library-gp/cut-{cut}"),"seeds":format!("seeds-{cut}.json"),"seeds_sha256":"seed-hash"})).collect::<Vec<_>>();
    let manifest = admission.join("combined-jobs.json");
    write(
        manifest.clone(),
        serde_json::json!({"schema":"parallel.gp_jobs.v1","qualification":true,"jobs":jobs}),
    );
    let receipt = admission.join("complete.json");
    write(
        receipt.clone(),
        serde_json::json!({"outcome":"admitted","requested_jobs":300,"newly_admitted_jobs":250,"jobs_sha256":file_hash(&manifest).unwrap()}),
    );
    write(
        root.join("runs/proposal-library-admission.json"),
        serde_json::json!({"phase":"completed","complete":true,"receipt":receipt}),
    );
    write(
        root.join("runs/proposal-library.json"),
        serde_json::json!({"phase":"completed","saved_seed_pools":350,"planned_seed_pools":400,"saved_ranked_programs":22400,"gp_jobs_admitted":0,"gp_evaluations":0}),
    );
    for (cut, evaluations, seed, complete) in [
        (0, 5, "seed-hash", true),
        (1, 7, "seed-hash", false),
        (2, 999, "wrong-seed", true),
        (299, 20, "seed-hash", true),
    ] {
        let output = root.join(format!("runs/library-gp/cut-{cut}"));
        write(
            output.join("progress.json"),
            serde_json::json!({"prepared_sha256":"prepared-hash","cut":cut,"seeds_sha256":seed,"vocabulary_sha256":"vocab-hash","evaluations":evaluations}),
        );
        if complete {
            write(
                output.join("search.json"),
                serde_json::json!({"schema":"parallel.search.v1","completed_evaluations":evaluations,"seeds_sha256":seed,"vocabulary_sha256":"vocab-hash"}),
            );
        }
        std::fs::write(output.join("observations.json"), [0xff, 0xff]).unwrap();
    }
    let first = dashboard::campaign(root);
    assert_eq!(first["proposal_library_admission"]["requested_jobs"], 300);
    assert_eq!(first["proposal_library_gp"]["checked_jobs"], 256);
    assert_eq!(first["proposal_library_gp"]["observed_evaluations"], 12);
    assert_eq!(first["proposal_library_gp"]["identity_mismatches"], 1);
    assert_eq!(
        first["proposal_library_gp"]["observation_ledgers_read"],
        false
    );
    let second = dashboard::campaign(root);
    assert_eq!(second["proposal_library_gp"]["checked_jobs"], 300);
    assert_eq!(second["proposal_library_gp"]["observed_evaluations"], 32);
    assert_eq!(second["proposal_library_gp"]["completed_jobs"], 2);
    let mut terminal = Terminal::new(TestBackend::new(160, 55)).unwrap();
    terminal
        .draw(|f| {
            dashboard::render(
                f,
                0,
                &Hardware::default(),
                jope_parallel::dashboard::DashboardData {
                    gp: &serde_json::Value::Null,
                    neural: &serde_json::Value::Null,
                    run: &second,
                    events: "",
                    history: &History::default(),
                },
            )
        })
        .unwrap();
    let text = terminal
        .backend()
        .buffer()
        .content
        .chunks(160)
        .map(|line| line.iter().map(|c| c.symbol()).collect::<String>())
        .find(|line| line.contains("Proposal seed library"))
        .unwrap();
    assert!(text.contains("300 admitted / 32 observed evals"));
    assert!(text.contains("350 / 400 pools"));
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(
            &std::fs::read(root.join("runs/proposal-library.json")).unwrap()
        )
        .unwrap()["gp_jobs_admitted"],
        0
    );
    write(
        receipt,
        serde_json::json!({"outcome":"admitted","requested_jobs":300,"jobs_sha256":"wrong-manifest-hash"}),
    );
    let mismatch = dashboard::campaign(root);
    assert_eq!(mismatch["proposal_library_gp"]["status"], "unavailable");
    assert!(mismatch["proposal_library_gp"]["observed_evaluations"].is_null());
}

#[test]
fn repeated_gp_search_jobs_are_distinct_from_unique_cut_encodings() {
    let run = serde_json::json!({"contexts":{"completed":10992,"total":10992,"families":72,"dimensions":2048},"teacher":{"cuts_with_teacher":42,"timestamp":now()}});
    let gp = serde_json::json!({"completed_views":12000,"active":104,"queued":3896,"failed_views":0,"timestamp":now()});
    let mut terminal = Terminal::new(TestBackend::new(160, 55)).unwrap();
    terminal
        .draw(|f| {
            dashboard::render(
                f,
                0,
                &Hardware::default(),
                jope_parallel::dashboard::DashboardData {
                    gp: &gp,
                    neural: &serde_json::Value::Null,
                    run: &run,
                    events: "",
                    history: &History::default(),
                },
            )
        })
        .unwrap();
    let text = terminal
        .backend()
        .buffer()
        .content
        .iter()
        .map(|cell| cell.symbol())
        .collect::<String>();
    assert!(text.contains("12000 / 16000 search jobs"));
    assert!(text.contains("10992 / 10992 views"));
    assert!(text.contains("42 unique cuts"));
    assert!(text.contains("GP teacher search jobs"));
    terminal
        .draw(|f| {
            dashboard::render(
                f,
                3,
                &Hardware::default(),
                jope_parallel::dashboard::DashboardData {
                    gp: &gp,
                    neural: &serde_json::Value::Null,
                    run: &run,
                    events: "",
                    history: &History::default(),
                },
            )
        })
        .unwrap();
    let text = terminal
        .backend()
        .buffer()
        .content
        .iter()
        .map(|cell| cell.symbol())
        .collect::<String>();
    assert!(text.contains("GP search jobs"));
    // The fixture has historical counts but no process-bound publisher lease.
    assert!(text.contains("stale activity"));
    assert!(!text.contains("104 active jobs"));
}

#[test]
fn pure_predictions_keep_their_scope_separate_from_admitted_library_searches() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    std::fs::create_dir_all(root.join("runs")).unwrap();
    std::fs::write(root.join("runs/pure-proposal-library.json"),serde_json::to_vec(&serde_json::json!({
        "schema":"parallel.pure_proposal_library_progress.v1","timestamp":now(),"phase":"running",
        "reason":"Using saved contexts for teacher-independent predictions","saved_seed_pools":32,
        "planned_seed_pools":3692,"saved_ranked_programs":2048,"gp_jobs_admitted":0,"gp_evaluations":0,
        "pure_prediction_verified":true,"encodings_recomputed":0
    })).unwrap()).unwrap();
    let mut run = dashboard::campaign(root);
    assert_eq!(run["pure_proposal_library"]["saved_seed_pools"], 32);
    run["contexts"] =
        serde_json::json!({"completed":10992,"total":10992,"families":72,"dimensions":2048});
    run["proposal_library"] = serde_json::json!({"phase":"stopped","saved_seed_pools":3204,"planned_seed_pools":7819,"saved_ranked_programs":205056});
    run["proposal_library_admission"] = serde_json::json!({"committed":true,"requested_jobs":3071});
    run["proposal_library_gp"] =
        serde_json::json!({"observed_evaluations":2226,"checked_jobs":256});
    run["cycle"] = serde_json::json!({"phase":"Existing campaign GP search","timestamp":now(),"neural_seeded_gp":{"completed":84,"total":336}});
    let h = Hardware {
        timestamp: now(),
        gpus: (0..8)
            .map(|index| Gpu {
                index,
                ..Default::default()
            })
            .collect(),
        ..Default::default()
    };
    let mut terminal = Terminal::new(TestBackend::new(160, 55)).unwrap();
    let draw = |terminal: &mut Terminal<TestBackend>, run: &serde_json::Value| {
        terminal
            .draw(|f| {
                dashboard::render(
                    f,
                    0,
                    &h,
                    jope_parallel::dashboard::DashboardData {
                        gp: &serde_json::Value::Null,
                        neural: &serde_json::Value::Null,
                        run,
                        events: "",
                        history: &History::default(),
                    },
                )
            })
            .unwrap();
        terminal
            .backend()
            .buffer()
            .content
            .chunks(160)
            .map(|line| line.iter().map(|c| c.symbol()).collect::<String>())
            .collect::<Vec<_>>()
    };
    let rows = draw(&mut terminal, &run);
    let text = rows.join("\n");
    let pure = rows
        .iter()
        .find(|row| row.contains("Pure policy predictions") && row.contains("RUNNING"))
        .unwrap();
    for expected in [
        "32 / 3692 pools",
        "2048 ranked",
        "unadmitted / 0 GP evals",
        "separate pure scope",
    ] {
        assert!(pure.contains(expected), "missing {expected}");
    }
    let old = rows
        .iter()
        .find(|row| row.contains("Proposal seed library"))
        .unwrap();
    assert!(old.contains("3071 admitted / 2226 observed evals"));
    assert!(old.contains("3204 / 7819 pools"));
    assert!(text.contains("10992 / 10992 views"));
    assert!(text.contains("GP 84 / 336"));
    assert!(text.contains("Saving pure policy predictions on eight GPUs"));
    assert!(text.contains("Using saved contexts for teacher-independent predictions"));
    run["pure_proposal_library"]["phase"] = "waiting".into();
    assert!(draw(&mut terminal, &run)
        .iter()
        .find(|row| row.contains("Pure policy predictions") && row.contains("WAITING"))
        .is_some());
    run["pure_proposal_library"]["timestamp"] = (now() - 20).into();
    let text = draw(&mut terminal, &run).join("\n");
    assert!(!text.contains("Saving pure policy predictions on eight GPUs"));
    assert!(text.contains("STALE"));
    assert_eq!(run["proposal_library_admission"]["requested_jobs"], 3071);
}

#[test]
fn selected_replay_counts_remain_visible_without_legacy_progress_fields() {
    let mut run = serde_json::json!({
        "binary_qualification":{"phase":"completed","complete":true},
        "binary_admitted_contexts":80,"binary_verified_contexts":80,
        "binary_verification":{"verified_pairs":80,"independently_computed_raw_components":480}
    });
    let render = |view, run: &serde_json::Value| {
        let mut terminal = Terminal::new(TestBackend::new(160, 55)).unwrap();
        terminal
            .draw(|f| {
                dashboard::render(
                    f,
                    view,
                    &Hardware::default(),
                    jope_parallel::dashboard::DashboardData {
                        gp: &serde_json::Value::Null,
                        neural: &serde_json::Value::Null,
                        run,
                        events: "",
                        history: &History::default(),
                    },
                )
            })
            .unwrap();
        terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|c| c.symbol())
            .collect::<String>()
    };
    assert!(render(0, &run).contains("80 admitted · 80 replayed"));
    let text = render(4, &run);
    for expected in [
        "80 admitted",
        "80 bindings verified",
        "80 exact replay pairs",
        "480 raw components",
    ] {
        assert!(text.contains(expected), "missing {expected}");
    }
    run["binary_verification"] = serde_json::Value::Null;
    let text = render(4, &run);
    assert!(text.contains("— exact replay pairs"));
    assert!(!text.contains("0 exact replay pairs"));
}

#[test]
fn fresh_unrelated_packing_inventory_does_not_certify_a_stale_cycle() {
    let hardware = Hardware {
        timestamp: now(),
        gpus: (0..8)
            .map(|index| Gpu {
                index,
                ..Default::default()
            })
            .collect(),
        ..Default::default()
    };
    let run = serde_json::json!({
        "active_cycle": "runs/current",
        "cycle": {"phase": "Pack newly scored programs into the next teaching corpus", "timestamp": now()-60},
        "corpus_progress": {"timestamp": now(), "experiment": "unrelated", "phase": "complete",
            "observations_seen": 13204374, "skipped_ledger": 1706, "records": 5323499}
    });
    let mut terminal = Terminal::new(TestBackend::new(180, 70)).unwrap();
    terminal
        .draw(|f| {
            dashboard::render(
                f,
                0,
                &hardware,
                jope_parallel::dashboard::DashboardData {
                    gp: &serde_json::Value::Null,
                    neural: &serde_json::Value::Null,
                    run: &run,
                    events: "",
                    history: &History::default(),
                },
            )
        })
        .unwrap();
    let text = terminal
        .backend()
        .buffer()
        .content
        .iter()
        .map(|c| c.symbol())
        .collect::<String>();
    assert!(text.contains("GPU wait reason unconfirmed"), "{text}");
    assert!(text.contains("stale Pack"), "{text}");
    assert!(!text.contains("aux plateaued"), "{text}");
}
