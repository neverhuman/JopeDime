use super::*;

#[test]
fn host_memory_stalls_and_stale_stage_receipts_are_not_reported_as_normal_gpu_wait() {
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
    let render = |run: &serde_json::Value| {
        let mut terminal = Terminal::new(TestBackend::new(180, 60)).unwrap();
        terminal
            .draw(|f| {
                dashboard::render(
                    f,
                    0,
                    &hardware,
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
    let mut run = serde_json::json!({"campaign_resource_evidence":{"available":true,"timestamp":now(),"training_workers":8,"throttled_workers":8,"memory_current":111669149696u64,"memory_high":103079215104u64}});
    let text = render(&run);
    assert!(text.contains("Host-memory throttle"));
    assert!(text.contains("8 / 8 training workers blocked"));
    run["campaign_resource_evidence"]["timestamp"] = (now() - 60).into();
    assert!(render(&run).contains("Training resource state uncertain"));
    run = serde_json::json!({"active_cycle":"runs/current","cycle":{"phase":"Pack newly scored programs","timestamp":now()-60}});
    let text = render(&run);
    assert!(text.contains("GPU wait reason unconfirmed"));
    assert!(text.split("Corpus packing").nth(1).unwrap()[..20].contains("STALE"));
    run["cycle"]["timestamp"] = now().into();
    run["cycle"]["heartbeat_role"] = "controller_liveness".into();
    run["cycle"]["last_progress_at"] = (now() - 120).into();
    run["cycle"]["work"] = serde_json::json!({"operation":"records packed","completed":123});
    let text = render(&run);
    assert!(text.contains("controller live; records packed 123"));
    assert!(text.contains("work age 2m"));
}

#[test]
fn selected_checkpoint_current_corpus_and_isolated_work_are_visible() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let write = |name: &str, value: serde_json::Value| {
        let p = root.join(name);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, serde_json::to_vec(&value).unwrap()).unwrap();
    };
    write(
        "runs/campaign-progress.json",
        serde_json::json!({"current_cycle":"runs/managed/cycle-1","phase":"qualification","timestamp":now()}),
    );
    write("runs/managed/cycle-1/cycle.json", serde_json::json!({}));
    write(
        "runs/managed/cycle-1/train-1/latest.json",
        serde_json::json!({"checkpoint":"runs/selected","step":30}),
    );
    write(
        "runs/managed/cycle-1/auxiliary/train/best.json",
        serde_json::json!({"checkpoint":"runs/selected","step":30}),
    );
    write(
        "runs/selected/state.json",
        serde_json::json!({"corpus_sha256":"current-corpus","model_sha256":"selected-model"}),
    );
    write(
        "runs/neural-service.json",
        serde_json::json!({"phase":"probe","checkpoint":"runs/unselected-latest","step":40}),
    );
    write(
        "runs/corpus-progress.json",
        serde_json::json!({"records":100,"complete":true}),
    );
    write(
        "runs/managed/cycle-0/complete.json",
        serde_json::json!({"teaching_corpus_sha256":"current-corpus"}),
    );
    write(
        "runs/managed/cycle-0/corpus-2/index.json",
        serde_json::json!({"records":2500}),
    );
    write(
        "runs/seed-quality-experiment-v1/progress.json",
        serde_json::json!({"phase":"downstream waiting for production GPU block","completed":16,"total":48,"timestamp":now(),"reason":"pure rankings frozen; GP fits pending"}),
    );
    write(
        "runs/binary-qualification-v1/progress.json",
        serde_json::json!({"phase":"running","completed":20,"total":80,"timestamp":now(),"reason":"binary encoding window"}),
    );
    write(
        "runs/seed-quality-experiment-v1/report.json",
        serde_json::json!({
            "selected_checkpoint":"runs/managed/cycle-0010/train/checkpoints/step-3000",
            "top8_physical_fits":722,"physical_fits":5000,"logical_evaluations":768,"top8_failures":22,"failures":99,
            "downstream":{"completed_views":48,"required_views":48,"direct_neural_selected_winner_improvements":{"current":2,"generic":3}}
        }),
    );
    let run = dashboard::campaign(root);
    assert_eq!(run["selected_checkpoint"]["checkpoint"], "runs/selected");
    assert_eq!(run["corpus_progress"]["records"], 2500);
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
    for expected in [
        "Selected checkpoint",
        "selected",
        "2500",
        "Predictor export",
        "Corpus packing",
        "Seed quality experiment",
        "Binary qualification",
        "20 / 80",
    ] {
        assert!(text.contains(expected), "missing {expected}");
    }
    assert!(text.split("Seed quality experiment").nth(1).unwrap()[..20].contains("WAITING"));
    let mut terminal = Terminal::new(TestBackend::new(160, 70)).unwrap();
    terminal
        .draw(|f| {
            dashboard::render(
                f,
                0,
                &Hardware::default(),
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
    for expected in [
        "FROZEN cycle-0010",
        "Top8: 722 fits",
        "768 logical evals",
        "22 failures",
        "GP 48 / 48",
        "Selected-seed wins 2 vs pure_first64",
    ] {
        assert!(text.contains(expected), "missing {expected}");
    }
}

#[test]
fn explicit_experiment_selection_never_falls_back_to_a_historical_report() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let write = |name: &str, value: serde_json::Value| {
        let path = root.join(name);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, serde_json::to_vec(&value).unwrap()).unwrap();
    };
    write(
        "runs/seed-quality-experiment-v1/progress.json",
        serde_json::json!({"phase":"old experiment"}),
    );
    write(
        "runs/selected/manifest.json",
        serde_json::json!({"schema":"parallel.seed_experiment.v2","checkpoint":"frozen-selected"}),
    );
    write(
        "runs/selected/progress.json",
        serde_json::json!({"phase":"new experiment","timestamp":now()}),
    );
    let default = dashboard::campaign_with_manifest(root, None).unwrap();
    assert_eq!(default["seed_experiment"]["phase"], "old experiment");
    let selected = dashboard::campaign_with_manifest(
        root,
        Some(std::path::Path::new("runs/selected/manifest.json")),
    )
    .unwrap();
    assert_eq!(selected["seed_experiment"]["phase"], "new experiment");
    assert_eq!(selected["experiment_checkpoint"], "frozen-selected");
    assert!(selected["seed_quality"].is_null());
    assert!(selected["binary_qualification"].is_null());
    write(
        "runs/selected/manifest.json",
        serde_json::json!({"schema":"unknown"}),
    );
    assert!(dashboard::campaign_with_manifest(
        root,
        Some(std::path::Path::new("runs/selected/manifest.json"))
    )
    .is_err());
}

#[test]
fn conflicting_live_workers_and_lease_receipts_show_uncertainty() {
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
    let run = serde_json::json!({"experiment_worker_evidence":{"timestamp":now(),"inventory_available":true,"locks_available":true,"gpu_workers":8,"experimental_gpu_owner":false}});
    let mut terminal = Terminal::new(TestBackend::new(160, 55)).unwrap();
    terminal
        .draw(|f| {
            dashboard::render(
                f,
                0,
                &h,
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
    assert!(text.contains("GPU ownership uncertain"));
}
#[test]
fn all_views_resize_and_mark_missing_telemetry() {
    for (w, h) in [(160, 55), (100, 36), (48, 15), (20, 8)] {
        let mut terminal = Terminal::new(TestBackend::new(w, h)).unwrap();
        for view in 0..5 {
            terminal
                .draw(|f| {
                    dashboard::render(
                        f,
                        view,
                        &Hardware::default(),
                        jope_parallel::dashboard::DashboardData {
                            gp: &serde_json::Value::Null,
                            neural: &serde_json::Value::Null,
                            run: &serde_json::Value::Null,
                            events: "",
                            history: &History::default(),
                        },
                    )
                })
                .unwrap();
        }
    }
    let mut terminal = Terminal::new(TestBackend::new(160, 55)).unwrap();
    terminal
        .draw(|f| {
            dashboard::render(
                f,
                1,
                &Hardware::default(),
                jope_parallel::dashboard::DashboardData {
                    gp: &serde_json::Value::Null,
                    neural: &serde_json::Value::Null,
                    run: &serde_json::Value::Null,
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
    assert!(text.contains("TELEMETRY STALE"));
    let mut h = Hardware {
        timestamp: now(),
        cpu: vec![99.; 124],
        ..Default::default()
    };
    h.gpus = (0..8)
        .map(|index| Gpu {
            index,
            utilization: 90.,
            memory_total_mb: 40960.,
            ..Default::default()
        })
        .collect();
    terminal
        .draw(|f| {
            dashboard::render(
                f,
                1,
                &h,
                jope_parallel::dashboard::DashboardData {
                    gp: &serde_json::Value::Null,
                    neural: &serde_json::Value::Null,
                    run: &serde_json::Value::Null,
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
    for gpu in 0..8 {
        assert!(text.contains(&format!("A100 / {gpu}")));
    }
    assert!(text.contains("123: 99%"));
    assert!(!text.contains("TELEMETRY STALE"));
}

#[test]
fn overview_uses_durable_progress_and_marks_invalidated_training() {
    let mut terminal = Terminal::new(TestBackend::new(160, 55)).unwrap();
    let mut h = Hardware {
        timestamp: now(),
        cpu: vec![88.; 124],
        ..Default::default()
    };
    h.gpus = (0..8)
        .map(|index| Gpu {
            index,
            ..Default::default()
        })
        .collect();
    let gp = serde_json::json!({"phase":"running","timestamp":now(),"active":104,"queued":2396,"completed_views":1000,"completed_evaluations":500000});
    let neural = serde_json::json!({"phase":"completed real-corpus training block","timestamp":now()-100,"step":300,"steps":300,"world":8});
    let run = serde_json::json!({"phase":"old encoding phase","contexts":{"completed":3500,"total":3500,"families":31,"dimensions":2048},"pilot_started":false});
    terminal
        .draw(|f| {
            dashboard::render(
                f,
                0,
                &h,
                jope_parallel::dashboard::DashboardData {
                    gp: &gp,
                    neural: &neural,
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
    for expected in [
        "3500 / 3500",
        "1000 / 3500",
        "completed · saved receipt",
        "not started",
        "Encoding queue finished",
    ] {
        assert!(text.contains(expected), "missing {expected}");
    }
    assert!(!text.contains("old encoding phase"));
    let mut invalid = neural;
    invalid["invalidated"] = true.into();
    terminal
        .draw(|f| {
            dashboard::render(
                f,
                0,
                &h,
                jope_parallel::dashboard::DashboardData {
                    gp: &gp,
                    neural: &invalid,
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
    assert!(text.contains("previous run invalidated"));
}
