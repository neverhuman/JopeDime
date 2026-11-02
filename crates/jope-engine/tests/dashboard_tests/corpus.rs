use super::*;

#[test]
fn real_corpus_recovery_does_not_displace_a_standalone_policy() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let path = root.join("runs/standalone-control");
    std::fs::create_dir_all(&path).unwrap();
    std::fs::write(
        path.join("latest.json"),
        r#"{"step":3000,"checkpoint":"control/step-3000"}"#,
    )
    .unwrap();
    std::fs::write(path.join("complete.json"), "{}").unwrap();
    std::fs::write(path.join("metrics.jsonl"),concat!(
        "{\"timestamp\":10,\"step\":2999,\"sequence_nll\":0.8,\"corpus_sha256\":\"real\",\"run\":\"runs/standalone-control\"}\n",
        "{\"timestamp\":11,\"step\":3000,\"steps\":3000,\"sequence_nll\":0.7,\"corpus_sha256\":\"real\",\"run\":\"runs/standalone-control\"}\n")).unwrap();
    std::fs::write(root.join("runs/neural-service.json"),r#"{"timestamp":20,"phase":"recovery qualification","step":20,"steps":20,"corpus_sha256":"real","run":"runs/parity/recovery"}"#).unwrap();
    let run = dashboard::campaign(root);
    assert_eq!(run["policy"]["step"], 3000);
    assert_eq!(run["policy"]["checkpoint"], "control/step-3000");
    assert_eq!(run["latest_neural_activity"]["step"], 20);
    assert_eq!(run["learning_curve"].as_array().unwrap().len(), 2);
}

#[test]
fn transition_receipts_and_five_second_teacher_feed_do_not_look_stalled() {
    let mut terminal = Terminal::new(TestBackend::new(160, 55)).unwrap();
    let mut run = serde_json::json!({
        "campaign_service":{"phase":"Managed campaign active","timestamp":now()-600,"current_cycle":"/workspace/runs/campaign/cycle-0"},
        "active_cycle":"/workspace/runs/campaign/cycle-0",
        "cycle":{"phase":"GP teaching","timestamp":now()},
        "teacher":{"timestamp":now()-6,"cuts_with_teacher":100,"new_bests":10}
    });
    let render = |terminal: &mut Terminal<TestBackend>,
                  run: &serde_json::Value,
                  neural: &serde_json::Value| {
        terminal
            .draw(|f| {
                dashboard::render(
                    f,
                    0,
                    &Hardware::default(),
                    jope_parallel::dashboard::DashboardData {
                        gp: &serde_json::Value::Null,
                        neural,
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
    let rows = render(&mut terminal, &run, &serde_json::Value::Null);
    let campaign = rows
        .iter()
        .find(|line| line.contains("Managed campaign") && line.contains("cycles saved"))
        .unwrap();
    assert!(campaign.contains("RUNNING"));
    assert!(!campaign.contains("STALE"));
    assert!(campaign.contains("phase age"));
    assert!(rows
        .iter()
        .find(|line| line.contains("Teacher telemetry"))
        .unwrap()
        .contains("live"));
    run["cycle"]["timestamp"] = (now() - 600).into();
    let neural = serde_json::json!({"run":"runs/campaign/cycle-0/auxiliary/train","timestamp":now(),"phase":"training"});
    let rows = render(&mut terminal, &run, &neural);
    assert!(rows
        .iter()
        .find(|line| line.contains("Managed campaign") && line.contains("cycles saved"))
        .unwrap()
        .contains("RUNNING"));
    let unrelated = serde_json::json!({"run":"runs/another-campaign/cycle-0/train","timestamp":now(),"phase":"training"});
    run["teacher"]["timestamp"] = (now() - 60).into();
    let rows = render(&mut terminal, &run, &unrelated);
    let campaign = rows
        .iter()
        .find(|line| line.contains("Managed campaign") && line.contains("cycles saved"))
        .unwrap();
    assert!(campaign.contains("PHASE"));
    assert!(!campaign.contains("STALE"));
    assert!(!campaign.contains("RUNNING"));
    assert!(rows
        .iter()
        .find(|line| line.contains("Teacher telemetry"))
        .unwrap()
        .contains("stale"));
}

#[test]
fn completed_auxiliary_displays_actual_steps_and_configured_limit_after_plateau() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let output = root.join("runs/cycle-0/auxiliary");
    std::fs::create_dir_all(&output).unwrap();
    std::fs::write(
        output.join("spec.json"),
        r#"{"options":{"max_steps":3000}}"#,
    )
    .unwrap();
    std::fs::write(
        root.join("runs/auxiliary-training.json"),
        serde_json::to_vec(&serde_json::json!({
            "phase":"completed","complete":true,"completed_optimizer_steps":1500,
            "output":output,"reason":"fixed-panel validation NLL plateau","best":{"step":600}
        }))
        .unwrap(),
    )
    .unwrap();
    let mut run = dashboard::campaign(root);
    assert_eq!(run["auxiliary"]["maximum_steps"], 3000);
    let mut terminal = Terminal::new(TestBackend::new(160, 55)).unwrap();
    let row = |terminal: &mut Terminal<TestBackend>, run: &serde_json::Value| {
        terminal
            .draw(|f| {
                dashboard::render(
                    f,
                    0,
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
            .chunks(160)
            .map(|line| line.iter().map(|c| c.symbol()).collect::<String>())
            .find(|line| line.contains("Auxiliary learning"))
            .unwrap()
    };
    let text = row(&mut terminal, &run);
    assert!(text.contains("DONE"));
    assert!(text.contains("1500 / 3000 steps"));
    assert!(text.contains("validation NLL plateau"));
    assert!(!text.contains("600 /"));
    run["auxiliary"] = serde_json::json!({"step":123,"maximum_steps":200,"timestamp":now(),"phase":"active learning"});
    assert!(row(&mut terminal, &run).contains("123 / 200 steps"));
    run["auxiliary"]["maximum_steps"] = serde_json::Value::Null;
    assert!(row(&mut terminal, &run).contains("123 / — steps"));
}

#[test]
fn proposal_library_is_saved_seeds_not_encodings_or_gp_teaching_scores() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    std::fs::create_dir_all(root.join("runs")).unwrap();
    std::fs::write(
        root.join("runs/proposal-library.json"),
        serde_json::to_vec(&serde_json::json!({
            "timestamp":now(),"phase":"waiting","reason":"waiting for shared GPU lease",
            "saved_seed_pools":12,"planned_seed_pools":100,"saved_ranked_programs":768,
            "gp_jobs_admitted":0,"gp_evaluations":0,"gpu_world":8,"encodings_recomputed":0
        }))
        .unwrap(),
    )
    .unwrap();
    let mut run = dashboard::campaign(root);
    run["contexts"] =
        serde_json::json!({"completed":10992,"total":10992,"families":72,"dimensions":2048});
    run["cycle"] = serde_json::json!({"phase":"Existing campaign GP search","timestamp":now(),"neural_seeded_gp":{"completed":61,"total":336}});
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
    assert!(text.contains("10992 / 10992"));
    assert!(text.contains("GP 61 / 336"));
    let library = rows
        .iter()
        .find(|line| line.contains("Proposal seed library"))
        .unwrap();
    for expected in [
        "WAITING",
        "12 / 100 pools",
        "768 ranked programs",
        "0 jobs admitted / 0 evaluations",
        "seeds only",
    ] {
        assert!(library.contains(expected), "missing {expected}");
    }
    assert!(text.contains("waiting for shared GPU lease"));
    run["proposal_library"]["phase"] = "running".into();
    assert!(draw(&mut terminal, &run)
        .join("\n")
        .contains("Building saved proposal seed library on eight GPUs"));
    run["proposal_library"]["phase"] = "completed".into();
    let rows = draw(&mut terminal, &run);
    assert!(rows
        .iter()
        .find(|line| line.contains("Proposal seed library"))
        .unwrap()
        .contains("SAVED"));
    assert_eq!(run["cycle"]["neural_seeded_gp"]["completed"], 61);
}

#[test]
fn report_watcher_waiting_is_live_at_its_fifteen_second_poll_interval() {
    let mut run = serde_json::json!({"report_progress":{"timestamp":now()-30,"phase":"Waiting for first completed campaign cycle","completed":0,"total":0,"complete":false}});
    let mut terminal = Terminal::new(TestBackend::new(160, 55)).unwrap();
    let row = |terminal: &mut Terminal<TestBackend>, run: &serde_json::Value| {
        terminal
            .draw(|f| {
                dashboard::render(
                    f,
                    0,
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
            .chunks(160)
            .map(|line| line.iter().map(|c| c.symbol()).collect::<String>())
            .find(|line| line.contains("Matched-budget reports"))
            .unwrap()
    };
    let text = row(&mut terminal, &run);
    assert!(text.contains("WAITING"));
    assert!(!text.contains("STALE"));
    assert!(text.contains("Waiting for first completed campaign cycle"));
    run["report_progress"]["timestamp"] = (now() - 60).into();
    assert!(row(&mut terminal, &run).contains("STALE"));
}

/// The Python removal P1 (ADR-0006) archived the only writers of three receipts the overview
/// reads: `watch_campaign_reports.py` (`runs/report-progress.json`), `opportunistic_proposals.py`
/// (`runs/proposal-library.json`) and `opportunistic_proposals_v2.py`
/// (`runs/pure-proposal-library.json`). Without a receipt no row waits for one; a receipt left
/// in the state root by a producer that has stopped is history, shown with its saved status.
#[test]
fn receipts_of_retired_producers_show_history_not_pending_work() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    std::fs::create_dir_all(root.join("runs")).unwrap();
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
    let lines = |run: &serde_json::Value| {
        rendered_lines(
            TerminalSize {
                width: 160,
                height: 55,
            },
            0,
            &hardware,
            &serde_json::Value::Null,
            run,
            "",
            &History::default(),
        )
    };
    let titles = [
        "Matched-budget reports",
        "Proposal seed library",
        "Pure policy predictions",
    ];

    // No receipt: no row, and nothing says a receipt is pending.
    let text = lines(&dashboard::campaign(root)).join("\n");
    for title in titles {
        assert!(
            !text.contains(title),
            "{title} shown without a receipt:\n{text}"
        );
    }
    assert!(!text.contains("reporting receipt pending"), "{text}");

    // Receipts as the producers left them a day before: shown, stale, never current work.
    let stopped = now() - 86_400;
    let write = |name: &str, value: serde_json::Value| {
        std::fs::write(
            root.join("runs").join(name),
            serde_json::to_vec(&value).unwrap(),
        )
        .unwrap()
    };
    write(
        "report-progress.json",
        serde_json::json!({"schema":"parallel.report_progress.v1","timestamp":stopped,
            "phase":"Calculating matched-budget policy comparisons","completed":3,"total":4,
            "complete":false}),
    );
    write(
        "proposal-library.json",
        serde_json::json!({"schema":"parallel.proposal_library_progress.v1","timestamp":stopped,
            "phase":"running","saved_seed_pools":3204,"planned_seed_pools":7819,
            "saved_ranked_programs":205056,"gp_jobs_admitted":0,"gp_evaluations":0}),
    );
    write(
        "pure-proposal-library.json",
        serde_json::json!({"schema":"parallel.pure_proposal_library_progress.v1",
            "timestamp":stopped,"phase":"running","saved_seed_pools":446,
            "planned_seed_pools":3692,"saved_ranked_programs":28544,"gp_jobs_admitted":0,
            "gp_evaluations":0}),
    );
    let mut run = dashboard::campaign(root);
    let rows = lines(&run);
    let row = |rows: &[String], title: &str| {
        rows.iter()
            .find(|line| line.contains(title))
            .cloned()
            .unwrap_or_else(|| panic!("no {title} row for a saved receipt:\n{}", rows.join("\n")))
    };
    for (title, progress) in [
        ("Matched-budget reports", "3 / 4"),
        ("Proposal seed library", "3204 / 7819 pools"),
        ("Pure policy predictions", "446 / 3692 pools"),
    ] {
        let line = row(&rows, title);
        assert!(line.contains("STALE"), "{line}");
        assert!(!line.contains("RUNNING"), "{line}");
        assert!(line.contains(progress), "{line}");
    }
    let text = rows.join("\n");
    assert!(
        !text.contains("Saving pure policy predictions on eight GPUs"),
        "{text}"
    );
    assert!(
        !text.contains("Building saved proposal seed library on eight GPUs"),
        "{text}"
    );

    // A finished receipt reads as finished, not as pending or live.
    run["report_progress"]["complete"] = true.into();
    run["proposal_library"]["phase"] = "completed".into();
    run["pure_proposal_library"]["phase"] = "stopped".into();
    let rows = lines(&run);
    assert!(row(&rows, "Matched-budget reports").contains("DONE"));
    assert!(row(&rows, "Proposal seed library").contains("SAVED"));
    assert!(row(&rows, "Pure policy predictions").contains("STOPPED"));
}
