use super::*;

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
