use jope_parallel::{
    dashboard::{self, History},
    telemetry::{now, Gpu, Hardware},
};
use ratatui::{backend::TestBackend, Terminal};

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

#[test]
fn stale_gp_history_does_not_display_live_slots_or_worker_rows() {
    let mut terminal = Terminal::new(TestBackend::new(160, 55)).unwrap();
    let gp = serde_json::json!({"phase":"running","timestamp":now(),"active":104,
        "queued":12,"completed_views":50801,"completed_evaluations":100000,
        "active_rows":[{"dataset":"fictional-live-cut","pid":1}],"evaluations_per_second":158.1});
    terminal
        .draw(|f| {
            dashboard::render(
                f,
                3,
                &Hardware::default(),
                jope_parallel::dashboard::DashboardData {
                    gp: &gp,
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
    assert!(text.contains("stale activity"));
    assert!(text.contains("50801 completed"));
    assert!(!text.contains("104 active"));
    assert!(!text.contains("fictional-live-cut"));
    assert!(!text.contains("158.1 eval/s"));
}

#[test]
fn campaign_reads_pointer_manifest_and_complete_checkpoint_receipts() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    std::fs::create_dir_all(root.join("data/encodings/contexts-v1")).unwrap();
    std::fs::create_dir_all(root.join("runs/real-training")).unwrap();
    std::fs::write(root.join("data/encodings/contexts-v1/index.json"),r#"{"dimensions":2048,"qualification":true,"encodings":[{"family":"a"},{"family":"a"},{"family":"b"}]}"#).unwrap();
    std::fs::write(
        root.join("runs/real-training/latest.json"),
        r#"{"checkpoint":"runs/real-training/checkpoints/step-10","step":10}"#,
    )
    .unwrap();
    let progress = dashboard::campaign(root);
    assert_eq!(progress["contexts"]["completed"], 3);
    assert_eq!(progress["contexts"]["families"], 2);
    assert_eq!(progress["checkpoint"]["step"], 10);
    assert!(progress["cycle"].is_null());
}

#[test]
#[ignore = "manual capture of live telemetry for review"]
fn capture_live_dashboard_screens() {
    use jope_parallel::manifest::read_json;
    let root = std::path::PathBuf::from(std::env::var("JOPE_PARALLEL_SNAPSHOT_ROOT").unwrap());
    let h: Hardware = read_json(&root.join("runs/hardware.json")).unwrap();
    let gp = read_json(&root.join("runs/gp-service.json")).unwrap();
    let neural = read_json(&root.join("runs/neural-service.json")).unwrap();
    let run = dashboard::campaign(&root);
    let output = std::env::var("JOPE_PARALLEL_CAPTURE_OUTPUT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| root.join("runs/parity/tui-v45"));
    std::fs::create_dir_all(&output).unwrap();
    for view in 0..5 {
        let mut terminal = Terminal::new(TestBackend::new(160, 55)).unwrap();
        terminal
            .draw(|f| {
                dashboard::render(
                    f,
                    view,
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
        let buffer = terminal.backend().buffer();
        let text = buffer
            .content
            .chunks(160)
            .map(|line| {
                line.iter()
                    .map(|c| c.symbol())
                    .collect::<String>()
                    .trim_end()
                    .to_string()
            })
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(output.join(format!("view-{}.txt", view + 1)), text).unwrap();
        use ratatui::style::{Color, Modifier};
        let color = |c: Color, background: bool| match c {
            Color::Rgb(r, g, b) => format!("#{r:02x}{g:02x}{b:02x}"),
            Color::Black => "#000000".into(),
            Color::White => "#ffffff".into(),
            Color::Cyan => "#5ccdff".into(),
            Color::Green => "#63e0ae".into(),
            Color::Yellow => "#fac565".into(),
            Color::Magenta => "#b59fff".into(),
            Color::Red => "#ff7487".into(),
            Color::DarkGray => "#7e95ae".into(),
            _ => if background { "#09111c" } else { "#dae5f2" }.into(),
        };
        let cells=buffer.content.iter().enumerate().map(|(i,c)|serde_json::json!({"x":i%160,"y":i/160,"text":c.symbol(),"fg":color(c.fg,false),"bg":color(c.bg,true),"bold":c.modifier.contains(Modifier::BOLD)})).collect::<Vec<_>>();
        std::fs::write(output.join(format!("view-{}.cells.json",view+1)),serde_json::to_vec(&serde_json::json!({"schema":"parallel.ratatui_snapshot.v1","columns":160,"rows":55,"view":view+1,"cells":cells})).unwrap()).unwrap();
    }
}

#[test]
fn live_encoder_takes_priority_and_pending_dependency_is_visible() {
    let mut terminal = Terminal::new(TestBackend::new(160, 55)).unwrap();
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
    let gp = serde_json::json!({"active":104,"timestamp":now()});
    let mut neural = serde_json::json!({"phase":"Hyperion new cohort", "encoding":true,"timestamp":now(),"completed":250,"total":4000});
    let run = serde_json::json!({"cycle":{"phase":"GP teaching comparison"},"contexts":{"completed":7000,"total":7000},"encoding_pipeline":{"phase":"Qualify new sampler shapes","timestamp":now()-100,"completed":0,"total":4000,"reason":"eight-rank parity before component inference","complete":false}});
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
    let screen = |t: &Terminal<TestBackend>| {
        t.backend()
            .buffer()
            .content
            .iter()
            .map(|c| c.symbol())
            .collect::<String>()
    };
    let text = screen(&terminal);
    assert!(text
        .chars()
        .take(480)
        .collect::<String>()
        .contains("Hyperion new cohort"));
    neural["phase"] = "completed encoding block".into();
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
    let text = screen(&terminal);
    assert!(text.contains("eight-rank parity before component inference"));
    assert!(text.contains("stale"));
    assert!(!text.contains("Encoding queue finished"));
    terminal
        .draw(|f| {
            dashboard::render(
                f,
                2,
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
    assert!(screen(&terminal).contains("Saved checkpoint"));
}

#[test]
fn learning_history_uses_saved_metrics_and_does_not_promote_a_probe() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let cycle = root.join("runs/cycle-real");
    let training = cycle.join("train-1");
    std::fs::create_dir_all(&training).unwrap();
    std::fs::create_dir_all(cycle.join("report-v1")).unwrap();
    std::fs::write(cycle.join("cycle.json"), "{}").unwrap();
    std::fs::write(
        root.join("runs/neural-service.json"),
        r#"{"phase":"numerical qualification probe","step":10,"steps":10}"#,
    )
    .unwrap();
    std::fs::write(
        training.join("latest.json"),
        r#"{"step":20,"checkpoint":"saved/model"}"#,
    )
    .unwrap();
    std::fs::write(training.join("complete.json"), "{}").unwrap();
    std::fs::write(training.join("metrics.jsonl"),concat!(
        "{\"step\":10,\"rank\":0,\"sequence_nll\":1.5,\"corpus_sha256\":\"real\",\"run\":\"runs/cycle-real/train-1\"}\n",
        "{\"step\":20,\"rank\":0,\"sequence_nll\":1.0,\"training_families\":48,\"corpus_sha256\":\"real\",\"run\":\"runs/cycle-real/train-1\"}\n")).unwrap();
    std::fs::write(cycle.join("report-v1/report.json"),r#"{"dl_gp_dl_complete":true,"teaching_records":100,"comparison_searches":6,"partitions":{"meta_validation":{"source_families":3,"evaluation_prefix":{"neural_over_generic":{"family_balanced_rmse_ratio":1.02},"neural_over_shuffled":{"family_balanced_rmse_ratio":0.99}}}}}"#).unwrap();
    let run = dashboard::campaign(root);
    assert_eq!(run["policy"]["training_families"], 48);
    assert_eq!(run["learning_curve"].as_array().unwrap().len(), 2);
    assert_eq!(
        run["reports"][0]["validation"]["evaluation_prefix"]["neural_over_generic"]
            ["family_balanced_rmse_ratio"],
        1.02
    );
    assert_eq!(
        run["latest_neural_activity"]["phase"],
        "numerical qualification probe"
    );
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
    assert!(text.contains("1.0200"));
    assert!(text.contains("Policy advantage is not established"));
}

#[test]
fn managed_cycles_and_raw_scores_are_distinct_from_matched_reports() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let first = root.join("runs/managed/cycle-0");
    let current = root.join("runs/managed/cycle-1");
    std::fs::create_dir_all(&first).unwrap();
    std::fs::create_dir_all(&current).unwrap();
    let write = |path: std::path::PathBuf, v: serde_json::Value| {
        std::fs::write(path, serde_json::to_vec(&v).unwrap()).unwrap()
    };
    write(
        root.join("runs/campaign-progress.json"),
        serde_json::json!({"current_cycle":current,"phase":"GP comparison","timestamp":now()}),
    );
    write(
        first.parent().unwrap().join("completed-cycles.json"),
        serde_json::json!({"cycles":[{"cycle":first}]}),
    );
    for path in [&first, &current] {
        write(
            path.join("comparisons.json"),
            serde_json::json!({"comparisons":[{"family":"observed-family","partition":"meta_validation","cut":0,"neural":{"best_validation_rmse":1.,"completed_evaluations":100},"generic_retained":{"best_validation_rmse":1.2,"completed_evaluations":200},"shuffled_context":{"best_validation_rmse":1.3,"completed_evaluations":80}}]}),
        );
    }
    write(
        first.join("complete.json"),
        serde_json::json!({"dl_gp_dl_complete":true}),
    );
    write(
        root.join("runs/teacher-progress.json"),
        serde_json::json!({"timestamp":now(),"cuts_with_teacher":1,"new_bests":2,"matched_cut_improvements":1,"top_improvements":[{"family":"observed-family","cut":0,"old_rmse":2.,"new_rmse":1.,"evaluations":100,"trajectory":[{"rmse":3.},{"rmse":2.},{"rmse":1.}]}]}),
    );
    let run = dashboard::campaign(root);
    assert_eq!(run["reports"].as_array().unwrap().len(), 2);
    assert_eq!(run["reports"][0]["raw_final_scores_only"], true);
    let mut terminal = Terminal::new(TestBackend::new(160, 55)).unwrap();
    terminal
        .draw(|f| {
            dashboard::render(
                f,
                3,
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
    assert!(text.contains("unequal completed budgets"));
    assert!(text.contains("3 ordered updates"));
    assert!(!text.contains("NN / generic 0.8333"));
}

#[test]
fn newest_managed_report_is_visible_when_its_path_sorts_before_history() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let current = root.join("runs/a-managed/cycle-2");
    std::fs::create_dir_all(&current).unwrap();
    let write = |path: std::path::PathBuf, value: serde_json::Value| {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, serde_json::to_vec(&value).unwrap()).unwrap();
    };
    write(
        root.join("runs/campaign-progress.json"),
        serde_json::json!({"current_cycle":"runs/a-managed/cycle-2","phase":"running","timestamp":now()}),
    );
    for (name, time) in [
        ("z-historical-1", 10),
        ("z-historical-2", 20),
        ("a-managed/cycle-2", 30),
    ] {
        let path = root.join("runs").join(name);
        write(
            path.join("cycle.json"),
            serde_json::json!({"timestamp":time}),
        );
        write(
            path.join("complete.json"),
            serde_json::json!({"completed_at":time,"dl_gp_dl_complete":true}),
        );
        write(
            path.join("report-v1/report.json"),
            serde_json::json!({"dl_gp_dl_complete":true,"partitions":{"meta_validation":{"source_families":8}}}),
        );
    }
    let run = dashboard::campaign(root);
    assert_eq!(run["reports"][2]["cycle"], "cycle-2");
    assert_eq!(run["active_cycle"], current.to_string_lossy().as_ref());
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
    assert!(text.contains("cycle-2"));
    assert!(text.contains("z-historical-2"));
    assert!(!text.contains("z-historical-1"));
}

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

/// Inputs kept together for rendered lines; no serialized contract changes.
#[derive(Clone, Copy)]
struct TerminalSize {
    width: u16,
    height: u16,
}

fn rendered_lines(
    size: TerminalSize,
    view: usize,
    hardware: &Hardware,
    neural: &serde_json::Value,
    run: &serde_json::Value,
    events: &str,
    history: &History,
) -> Vec<String> {
    let TerminalSize { width, height } = size;

    let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
    terminal
        .draw(|frame| {
            dashboard::render(
                frame,
                view,
                hardware,
                jope_parallel::dashboard::DashboardData {
                    gp: &serde_json::Value::Null,
                    neural,
                    run,
                    events,
                    history,
                },
            )
        })
        .unwrap();
    terminal
        .backend()
        .buffer()
        .content
        .chunks(width as usize)
        .map(|row| row.iter().map(|cell| cell.symbol()).collect::<String>())
        .collect()
}

#[test]
fn held_training_receipt_never_reports_policy_running() {
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
    let neural = serde_json::json!({"phase":"held","step":0,"steps":0,"timestamp":now()});
    let run = serde_json::json!({"phase":"private rendered fixture","pilot_started":false});
    let rows = rendered_lines(
        TerminalSize {
            width: 160,
            height: 55,
        },
        0,
        &hardware,
        &neural,
        &run,
        "",
        &History::default(),
    );
    let policy = rows
        .iter()
        .find(|row| row.contains("Policy optimization"))
        .unwrap();
    assert!(policy.contains("HELD"));
    assert!(!policy.contains("RUNNING"));
    assert!(rows.join("\n").contains("Automatic training held"));
    assert!(rows.join("\n").contains("GPU training held"));
    let mut details = History::default();
    details.learning_details = true;
    let rows = rendered_lines(
        TerminalSize {
            width: 160,
            height: 55,
        },
        2,
        &hardware,
        &neural,
        &run,
        "",
        &details,
    );
    assert!(rows.join("\n").contains("Policy optimization: HELD"));
}

#[test]
fn automatic_hold_does_not_hide_stale_receipt_or_claim_existing_worker_stopped() {
    let hardware = Hardware::default();
    let neural =
        serde_json::json!({"phase":"training","step":20,"steps":1000,"timestamp":now()-120});
    let run = serde_json::json!({"continuous":{"phase":"running","timestamp":now(),"automatic_training":false,
        "training_hold_reason":"frozen diagnostics rejected","training_task_reservation":{"identity":"existing-task"}},
        "operations":{"services":{"jope-parallel-campaign":{"ActiveState":"active"}}}});
    let rows = rendered_lines(
        TerminalSize {
            width: 220,
            height: 70,
        },
        0,
        &hardware,
        &neural,
        &run,
        "",
        &History::default(),
    );
    let policy = rows
        .iter()
        .find(|row| row.contains("Policy optimization"))
        .unwrap();
    assert!(policy.contains("HELD"));
    assert!(!policy.contains("RUNNING"));
    assert!(policy.contains("stale"));
    assert!(policy.contains("separate reconciliation"));
    let future = serde_json::json!({"phase":"training","step":1,"timestamp":now()+120});
    let rows = rendered_lines(
        TerminalSize {
            width: 160,
            height: 55,
        },
        0,
        &hardware,
        &future,
        &serde_json::Value::Null,
        "",
        &History::default(),
    );
    assert!(rows
        .iter()
        .find(|row| row.contains("Policy optimization"))
        .unwrap()
        .contains("STALE"));
}

#[test]
fn compact_overview_exposes_hold_and_scrolls_whole_records_without_clipping() {
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
    let reason="UNIQUE-HOLD-REASON long enough to wrap but all of its words must remain readable END-OF-HOLD";
    let neural =
        serde_json::json!({"phase":"held","timestamp":now(),"training_hold_reason":reason});
    let run = serde_json::json!({"contexts":{"completed":23,"total":23,"families":5}});
    let rows = rendered_lines(
        TerminalSize {
            width: 48,
            height: 15,
        },
        0,
        &hardware,
        &neural,
        &run,
        "",
        &History::default(),
    );
    assert!(rows.join("\n").contains("Policy optimization: HELD"));
    assert!(!rows.join("\n").contains("RUNNING"));
    assert!(rows.last().unwrap().contains("scroll"));
    let mut all = String::new();
    for offset in 0..160 {
        let mut history = History::default();
        history.compact_offsets[0] = offset;
        all.push_str(
            &rendered_lines(
                TerminalSize {
                    width: 48,
                    height: 15,
                },
                0,
                &hardware,
                &neural,
                &run,
                "",
                &history,
            )
            .join("\n"),
        );
    }
    let all = all.split_whitespace().collect::<Vec<_>>().join(" ");
    for marker in [
        "UNIQUE-HOLD-REASON",
        "END-OF-HOLD",
        "Saved contexts: DONE",
        "Policy advantage is not established",
    ] {
        assert!(all.contains(marker), "missing compact record {marker}");
    }
    let wide = rendered_lines(
        TerminalSize {
            width: 160,
            height: 55,
        },
        0,
        &hardware,
        &neural,
        &run,
        "",
        &History::default(),
    );
    assert!(wide.join("\n").contains("Task board"));
}

#[test]
fn compact_hardware_and_events_keep_last_records_reachable_after_resize() {
    let hardware = Hardware {
        timestamp: now(),
        gpus: (0..8)
            .map(|index| Gpu {
                index,
                ..Default::default()
            })
            .collect(),
        cpu: vec![10.; 128],
        errors: vec!["FINAL-HARDWARE-ERROR".into()],
        ..Default::default()
    };
    let mut all = String::new();
    for offset in 0..200 {
        let mut history = History::default();
        history.compact_offsets[1] = offset;
        all.push_str(
            &rendered_lines(
                TerminalSize {
                    width: 48,
                    height: 15,
                },
                1,
                &hardware,
                &serde_json::Value::Null,
                &serde_json::Value::Null,
                "",
                &history,
            )
            .join("\n"),
        );
    }
    for marker in ["GPU 7:", "CPU 127:", "FINAL-HARDWARE-ERROR"] {
        assert!(all.contains(marker), "missing {marker}");
    }
    let mut history = History::default();
    history.compact_offsets[4] = u16::MAX;
    let events = (0..30)
        .map(|n| format!("SAVED-EVENT-{n:02}"))
        .collect::<Vec<_>>()
        .join("\n");
    let rows = rendered_lines(
        TerminalSize {
            width: 48,
            height: 15,
        },
        4,
        &hardware,
        &serde_json::Value::Null,
        &serde_json::Value::Null,
        &events,
        &history,
    );
    assert!(rows.join("\n").contains("SAVED-EVENT-00"));
    assert!(rows.join("\n").contains("FINAL-HARDWARE-ERROR"));
    assert!(rows.last().unwrap().contains("scroll"));
    for view in 0..5 {
        let rows = rendered_lines(
            TerminalSize {
                width: 48,
                height: 15,
            },
            view,
            &Hardware::default(),
            &serde_json::Value::Null,
            &serde_json::Value::Null,
            "",
            &History::default(),
        );
        assert!(rows[0].contains("JOPE / PARALLEL"));
        assert!(rows[1].contains("5 Events"));
        assert!(rows[14].contains("q detach"));
    }
}
