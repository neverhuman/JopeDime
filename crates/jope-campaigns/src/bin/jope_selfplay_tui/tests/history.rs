#[test]
fn metrics_history_reads_and_downsamples() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("train_metrics.jsonl");
    let mut body = String::new();
    for epoch in 1..=1200_u64 {
        body.push_str(&format!(
            "{{\"epoch\":{epoch},\"train_loss\":{},\"val_loss\":{},\
                 \"macro_validation_bad_passes\":0,\"overfit_gap\":1.0,\
                 \"val_pairwise_win_accuracy\":0.6}}\n",
            2.0 - epoch as f64 / 1200.0,
            9.0 - epoch as f64 / 300.0,
        ));
    }
    std::fs::write(&path, body).unwrap();
    let history = read_metrics_history(&path);
    assert!(history.len() <= DL_HISTORY_MAX + 1);
    assert_eq!(history.first().unwrap().epoch, 1);
    assert_eq!(history.last().unwrap().epoch, 1200);
    assert!(history.last().unwrap().val_loss < history.first().unwrap().val_loss);
}

fn run_series(label: &str, live: bool, points: Vec<TrainerMetricPoint>) -> RunSeries {
    RunSeries {
        run_label: format!("001805437-{label}"),
        short_label: label.to_owned(),
        points,
        live,
        updated_at_ms: 0,
    }
}

#[test]
fn loss_and_pairwise_series_extract_by_epoch() {
    let history = DlHistory {
        runs: vec![run_series(
            "a-g9",
            true,
            vec![
                TrainerMetricPoint {
                    epoch: 1,
                    train_loss: 2.0,
                    val_loss: 9.0,
                    val_pairwise: Some(0.55),
                    ..TrainerMetricPoint::default()
                },
                TrainerMetricPoint {
                    epoch: 2,
                    train_loss: 1.5,
                    val_loss: 8.0,
                    val_pairwise: None,
                    ..TrainerMetricPoint::default()
                },
            ],
        )],
        fetched_at_ms: 0,
    };
    let series = continuous_series(&history);
    assert_eq!(series.train, vec![(1.0, 2.0), (2.0, 1.5)]);
    assert_eq!(series.val, vec![(1.0, 9.0), (2.0, 8.0)]);
    assert_eq!(series.pairwise, vec![(1.0, 0.55)]);
    assert_eq!(series.total_epochs, 2.0);
}

/// The defect the operator hit: trainer a-g8 completed and its successor
/// arrived under a new job id, so a chart keyed on the live job blanked.
/// The series must span every generation on one axis and keep rendering
/// with zero live trainers.
#[test]
fn loss_series_is_continuous_across_trainer_generations() {
    let epoch = |n: u64, train: f64, val: f64, pairwise: f64| TrainerMetricPoint {
        epoch: n,
        train_loss: train,
        val_loss: val,
        val_pairwise: Some(pairwise),
        ..TrainerMetricPoint::default()
    };
    let history = DlHistory {
        runs: vec![
            run_series(
                "a-g8",
                false,
                vec![epoch(1, 9.0, 9.02, 0.61), epoch(2, 5.0, 5.10, 0.66)],
            ),
            run_series("a-g9", true, vec![epoch(1, 3.35, 3.68, 0.70)]),
        ],
        fetched_at_ms: 0,
    };
    let series = continuous_series(&history);
    // a-g9 epoch 1 lands after a-g8's two epochs, not back at x=1.
    assert_eq!(series.train, vec![(1.0, 9.0), (2.0, 5.0), (3.0, 3.35)]);
    assert_eq!(series.total_epochs, 3.0);
    assert_eq!(
        series.marks,
        vec![
            (1.0, "a-g8".to_owned(), false),
            (3.0, "a-g9".to_owned(), true),
        ]
    );
    // Prior generations survive a changeover with nothing running.
    let dead = DlHistory {
        runs: history
            .runs
            .iter()
            .map(|run| RunSeries {
                live: false,
                ..run.clone()
            })
            .collect(),
        fetched_at_ms: 0,
    };
    assert_eq!(continuous_series(&dead).train.len(), 3);
}

/// Absent losses parse as 0.0; plotting them would drop the curve to the
/// floor on a partially written row.
#[test]
fn continuous_series_drops_placeholder_zero_losses() {
    let history = DlHistory {
        runs: vec![run_series(
            "a-g9",
            true,
            vec![
                TrainerMetricPoint {
                    epoch: 1,
                    train_loss: 3.0,
                    val_loss: 3.5,
                    ..TrainerMetricPoint::default()
                },
                TrainerMetricPoint {
                    epoch: 2,
                    train_loss: 0.0,
                    val_loss: 0.0,
                    ..TrainerMetricPoint::default()
                },
            ],
        )],
        fetched_at_ms: 0,
    };
    let series = continuous_series(&history);
    assert_eq!(series.train, vec![(1.0, 3.0)]);
    assert_eq!(series.val, vec![(1.0, 3.5)]);
    // The generation still spans both epochs on the axis.
    assert_eq!(series.total_epochs, 2.0);
}

#[test]
fn generation_ruler_marks_every_generation() {
    let series = DlSeries {
        marks: vec![
            (1.0, "a-g7".to_owned(), false),
            (21.0, "a-g8".to_owned(), false),
            (45.0, "a-g9".to_owned(), true),
        ],
        total_epochs: 47.0,
        ..DlSeries::default()
    };
    let text = tui_render::generation_ruler(&series, 80)
        .spans
        .iter()
        .map(|span| span.content.to_string())
        .collect::<String>();
    assert!(text.contains("├a-g7"), "{text}");
    assert!(text.contains("├a-g8"), "{text}");
    assert!(text.contains("├a-g9"), "{text}");
    // Ordered left to right by epoch offset.
    assert!(text.find("├a-g7") < text.find("├a-g8"), "{text}");
    assert!(text.find("├a-g8") < text.find("├a-g9"), "{text}");
    assert!(tui_render::generation_ruler(&DlSeries::default(), 80)
        .spans
        .iter()
        .any(|span| span.content.contains("no trainer generations")),);
}

#[test]
fn generations_panel_derives_from_history() {
    let history = DlHistory {
        runs: vec![
            run_series(
                "a-g8",
                false,
                vec![TrainerMetricPoint {
                    epoch: 24,
                    val_pairwise: Some(0.72),
                    ..TrainerMetricPoint::default()
                }],
            ),
            run_series("a-g9", true, Vec::new()),
        ],
        fetched_at_ms: 0,
    };
    assert_eq!(
        generations_from_history(&history),
        vec![
            ("001805437-a-g8".to_owned(), Some(0.72)),
            ("001805437-a-g9".to_owned(), None),
        ]
    );
}

/// The funnel panel was a fixed 4 rows — two lines per task plus borders
/// means only regression fit, so the binary totals were invisible on tab 4.
/// Now all four task lanes must be visible, including census-only rows for
/// corpora the index has not scanned yet (everything still waiting).
#[test]
fn dl_tab_funnel_shows_both_task_totals() {
    use ratatui::backend::TestBackend;
    let mut dashboard = build_dashboard(&fixture(), &SlowView::default());
    dashboard.funnel = UnlockFunnel {
        rows: vec![
            FunnelRow {
                task: "regression".to_owned(),
                registry: 27_525,
                active: 8_460,
                in_play: 6_404,
                queued: 15_509,
                encoding: 3_353,
                ..FunnelRow::default()
            },
            FunnelRow {
                task: "binary".to_owned(),
                registry: 24_840,
                active: 2_073,
                in_play: 2_028,
                encoding: 22_747,
                ..FunnelRow::default()
            },
            FunnelRow {
                task: "timeseries".to_owned(),
                registry: 42,
                ..FunnelRow::default()
            },
            FunnelRow {
                task: "multiclass".to_owned(),
                registry: 272,
                encoded: 269,
                queued: 269,
                ..FunnelRow::default()
            },
        ],
        source: "index".to_owned(),
        fetched_at_ms: now_ms(),
    };
    let backend = TestBackend::new(160, 60);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal
        .draw(|frame| {
            tui_render::render(
                frame,
                &dashboard,
                250,
                UiState {
                    tab: Tab::Dl,
                    arm: 0,
                },
            )
        })
        .unwrap();
    let rendered = terminal
        .backend()
        .buffer()
        .content()
        .iter()
        .map(|cell| cell.symbol())
        .collect::<String>();
    assert!(rendered.contains("REG 27,525"), "regression totals visible");
    assert!(rendered.contains("BIN 24,840"), "binary totals visible");
    assert!(
        rendered.contains("active 8,460"),
        "regression active visible"
    );
    assert!(rendered.contains("active 2,073"), "binary active visible");
    assert!(rendered.contains("TS 42"), "timeseries census row visible");
    assert!(rendered.contains("MC 272"), "multiclass census row visible");
    assert!(
        rendered.contains("queued +269"),
        "installed multiclass encodings visible as queued"
    );
}

/// Data sources that predate a task use the typed telemetry fallback and
/// rows always land in canonical display order.
#[test]
fn funnel_rows_are_task_complete_and_canonically_ordered() {
    let mut rows = vec![
        FunnelRow {
            task: "binary".to_owned(),
            registry: 10,
            ..FunnelRow::default()
        },
        FunnelRow {
            task: "regression".to_owned(),
            registry: 20,
            ..FunnelRow::default()
        },
    ];
    let fallback = BTreeMap::from([
        (
            "timeseries".to_owned(),
            FunnelRow {
                task: "timeseries".to_owned(),
                registry: 42,
                ..FunnelRow::default()
            },
        ),
        (
            "multiclass".to_owned(),
            FunnelRow {
                task: "multiclass".to_owned(),
                registry: 272,
                encoded: 269,
                queued: 269,
                ..FunnelRow::default()
            },
        ),
    ]);
    complete_funnel_rows(&mut rows, &fallback);
    let order: Vec<&str> = rows.iter().map(|row| row.task.as_str()).collect();
    assert_eq!(
        order,
        vec!["regression", "binary", "timeseries", "multiclass"]
    );
    assert_eq!(rows[2].registry, 42);
    assert_eq!(
        rows[2].locked(),
        42,
        "telemetry rows report the corpus as waiting"
    );
    assert_eq!(rows[3].registry, 272);
    assert_eq!(rows[3].queued, 269, "installed encodings surface as queued");
    assert_eq!(rows[3].encoded, 269);
    assert_eq!(rows[3].locked(), 3, "remainder still awaiting encoding");
    assert_eq!(rows[3].active, 0, "fallback rows are not active yet");
    // Existing rows are never duplicated or overwritten by the fallback.
    let mut again = rows.clone();
    complete_funnel_rows(&mut again, &BTreeMap::new());
    assert_eq!(again, rows);
}

#[test]
fn funnel_tags_cover_all_tasks_and_flag_unknowns() {
    assert_eq!(funnel_tag("regression"), "REG");
    assert_eq!(funnel_tag("binary"), "BIN");
    assert_eq!(funnel_tag("timeseries"), "TS");
    assert_eq!(funnel_tag("multiclass"), "MC");
    assert_eq!(funnel_tag("multi_label"), "MC");
    assert_eq!(funnel_tag("mystery"), "?");
    assert_eq!(funnel_tag_compact("timeseries"), "ts");
    assert_eq!(funnel_tag_compact("multi_label"), "mc");
    assert_eq!(funnel_tag_compact("mystery"), "?");
}

#[test]
fn short_generation_label_strips_watermark() {
    assert_eq!(short_generation_label("001805437-a-g9"), "a-g9");
    assert_eq!(short_generation_label("001805437-c-r000002"), "c-r000002");
    assert_eq!(short_generation_label("legacy"), "legacy");
}

#[test]
fn validation_phase_never_looks_stuck() {
    let trainer = TrainerDetail {
        phase: "validation".to_owned(),
        epoch: 20,
        epochs: 60,
        batch: 96,
        total_batches: 96,
        validation_batch: 87,
        validation_batches: 175,
        ..TrainerDetail::default()
    };
    let label = tui_render::phase_label(&trainer);
    assert!(label.contains("VALIDATION 87/175"), "{label}");
    let training = TrainerDetail {
        phase: "train".to_owned(),
        epoch: 20,
        epochs: 60,
        batch: 42,
        total_batches: 96,
        ..TrainerDetail::default()
    };
    assert!(tui_render::phase_label(&training).contains("batch 42/96"));
}

#[test]
fn trend_arrows_point_the_right_way() {
    assert_eq!(tui_render::trend_arrow(2.0, 1.0), "↓");
    assert_eq!(tui_render::trend_arrow(1.0, 2.0), "↑");
    assert_eq!(tui_render::trend_arrow(1.0, 1.0), "→");
}

#[test]
fn dl_tab_renders_headless_with_and_without_trainer() {
    use ratatui::backend::TestBackend;
    let mut dashboard = build_dashboard(&fixture(), &SlowView::default());
    let backend = TestBackend::new(140, 45);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal
        .draw(|frame| {
            tui_render::render(
                frame,
                &dashboard,
                250,
                UiState {
                    tab: Tab::Dl,
                    arm: 0,
                },
            )
        })
        .unwrap();
    dashboard.trainers.push(TrainerDetail {
        job_id: "jope-selfplay-train-x".to_owned(),
        node: "xbabe2".to_owned(),
        is_trainer: true,
        phase: "validation".to_owned(),
        epoch: 20,
        epochs: 60,
        validation_batch: 87,
        validation_batches: 175,
        patience: 12,
        history: vec![
            TrainerMetricPoint {
                epoch: 19,
                train_loss: 1.6,
                val_loss: 7.6,
                overfit_gap: 6.0,
                val_pairwise: Some(0.64),
                ..TrainerMetricPoint::default()
            },
            TrainerMetricPoint {
                epoch: 20,
                train_loss: 1.5,
                val_loss: 7.4,
                overfit_gap: 5.9,
                val_pairwise: Some(0.65),
                ..TrainerMetricPoint::default()
            },
        ],
        ..TrainerDetail::default()
    });
    dashboard.daemon_ages = vec![("queue-keeper-v3".to_owned(), 30, 450)];
    dashboard.generations = vec![("001805437-c".to_owned(), Some(0.6452))];
    terminal
        .draw(|frame| {
            tui_render::render(
                frame,
                &dashboard,
                250,
                UiState {
                    tab: Tab::Dl,
                    arm: 0,
                },
            )
        })
        .unwrap();
    let rendered = terminal
        .backend()
        .buffer()
        .content()
        .iter()
        .map(|cell| cell.symbol())
        .collect::<String>();
    assert!(
        rendered.contains("VALIDATION 87/175"),
        "validation phase must be explicit"
    );
    assert!(rendered.contains("LOSS"), "loss chart title present");
}
