/// Baseline fixture mirroring ops/seeding-curve.v1.json as the reconcile
/// hook writes it today: one generation (fbcd5459), 5.2% DL win share.
fn seeding_curve_fixture() -> serde_json::Value {
    json!({
        "schema": "JopeSeedingCurve.v1",
        "generated_at_ms": 1_785_300_000_000_u64,
        "curve": [{
            "generation_index": 0,
            "model_checkpoint_sha256":
                "fbcd5459f2c18f6cd5e4c23eef76a508a91d653a82f27c6ea41bcee8cafa2592",
            "strict_wins_total": 1668,
            "strict_wins_dl": 86,
            "dl_strict_win_share": 0.051_558_752_997_601_92,
            "dl_strict_win_share_pct": 5.2,
            "champions_total": 2385,
            "observations": {
                "rows_scanned": 3_513_076_u64,
                "dl_rows": 291_000_u64,
                "dl_observation_share": 0.082_833,
                "cursor_created_at_ms": 1_785_203_238_986_i64,
                "caught_up": true
            }
        }]
    })
}

#[test]
fn seeding_curve_reader_parses_the_ops_file() {
    let temporary = tempfile::TempDir::new().unwrap();
    let root = temporary.path();
    std::fs::create_dir_all(root.join("ops")).unwrap();
    std::fs::write(
        root.join("ops/seeding-curve.v1.json"),
        serde_json::to_vec(&seeding_curve_fixture()).unwrap(),
    )
    .unwrap();
    let view = read_seeding_curve(root).expect("curve file parses");
    assert_eq!(view.points.len(), 1);
    let point = &view.points[0];
    assert_eq!(point.checkpoint_short, "fbcd5459");
    assert_eq!(point.dl_win_share_pct, Some(5.2));
    assert!((point.dl_obs_share_pct.unwrap() - 8.2833).abs() < 1e-9);
    assert_eq!(point.strict_wins_total, 1668);
    assert!(point.observations_caught_up);
    assert!(view.fetched_at_ms > 0);
    assert!(
        read_seeding_curve(&root.join("absent")).is_none(),
        "missing file keeps the prior snapshot"
    );
}

#[test]
fn dl_tab_generations_panel_renders_dl_share_column() {
    use ratatui::backend::TestBackend;
    let mut dashboard = build_dashboard(&fixture(), &SlowView::default());
    dashboard.generations = vec![("001805437-a-g8".to_owned(), Some(0.7263))];
    // The unlock funnel and the DL history each carry their own NO-DATA
    // tag; keep them quiet so the negative assertion below isolates the
    // seeding-curve tag.
    dashboard.funnel.fetched_at_ms = now_ms();
    dashboard.dl_history.fetched_at_ms = now_ms();
    dashboard.seeding = SeedingCurveView {
        points: vec![SeedingPoint {
            checkpoint_short: "fbcd5459".to_owned(),
            dl_win_share_pct: Some(5.2),
            dl_obs_share_pct: Some(8.3),
            strict_wins_total: 1668,
            observations_caught_up: true,
        }],
        fetched_at_ms: now_ms(),
    };
    let backend = TestBackend::new(140, 50);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal
        .draw(|frame| {
            super::tui_dashboard::render(
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
        rendered.contains("DL-win"),
        "DL-share column header present"
    );
    assert!(rendered.contains("5.2%"), "DL strict-win share rendered");
    assert!(rendered.contains("fbcd5459"), "seed checkpoint labeled");
    assert!(
        !rendered.contains("[NO DATA]"),
        "fresh curve must not be tagged NO DATA"
    );
    // A dashboard that never saw the curve file reports NO DATA instead.
    dashboard.seeding = SeedingCurveView::default();
    terminal
        .draw(|frame| {
            super::tui_dashboard::render(
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
        rendered.contains("[NO DATA]"),
        "missing curve is called out"
    );
}
