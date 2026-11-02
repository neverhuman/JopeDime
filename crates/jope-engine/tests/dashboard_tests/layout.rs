use super::*;

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
