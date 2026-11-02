#[test]
fn funnel_rows_parse_from_driver_summary() {
    let driver: Value = serde_json::from_str(
        r#"{"stage_summary":{"regression":{"registry":27525,"sidecars_complete":24172,
                "catalog_active":6412,"observed":2491},
                "binary":{"registry":24840,"sidecars_complete":2091,
                "catalog_active":2073,"observed":657}}}"#,
    )
    .unwrap();
    let rows = funnel_rows_from_driver(&driver);
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].task, "regression");
    assert_eq!(rows[0].registry, 27_525);
    assert_eq!(rows[0].active, 6_412);
    assert_eq!(rows[1].task, "binary");
    assert_eq!(rows[1].in_play, 657);
}

fn line_text(line: &Line<'_>) -> String {
    line.spans
        .iter()
        .map(|span| span.content.as_ref())
        .collect()
}

#[test]
fn funnel_rows_query_from_legacy_index_fixture_falls_back() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("dataset-state.sqlite");
    let connection = rusqlite::Connection::open(&path).unwrap();
    connection
        .execute_batch(
            "CREATE TABLE datasets(task TEXT NOT NULL, dataset_hash TEXT NOT NULL,
                   sidecars_complete INTEGER NOT NULL DEFAULT 0,
                   catalog_active INTEGER NOT NULL DEFAULT 0,
                   obs_count INTEGER NOT NULL DEFAULT 0,
                   PRIMARY KEY(task, dataset_hash)) WITHOUT ROWID;
                 INSERT INTO datasets VALUES
                   ('regression','r1',1,1,5),('regression','r2',1,1,0),
                   ('regression','r3',1,0,0),('binary','b1',1,1,2),
                   ('binary','b2',0,0,0);",
        )
        .unwrap();
    drop(connection);
    let rows = funnel_rows_from_index(&path).unwrap();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].task, "regression");
    assert_eq!(rows[0].registry, 3);
    assert_eq!(rows[0].encoded, 3);
    assert_eq!(rows[0].active, 2);
    assert_eq!(rows[0].in_play, 1);
    assert_eq!(rows[0].queued, 0, "legacy schema zeroes the new stages");
    assert_eq!(rows[1].task, "binary");
    assert_eq!(rows[1].registry, 2);
    assert_eq!(rows[1].active, 1);
    assert_eq!(rows[1].in_play, 1);
}

#[test]
fn funnel_rows_query_full_pipeline_stages() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("dataset-state.sqlite");
    let connection = rusqlite::Connection::open(&path).unwrap();
    connection
        .execute_batch(
            "CREATE TABLE datasets(task TEXT NOT NULL, dataset_hash TEXT NOT NULL,
                   sidecars_complete INTEGER NOT NULL DEFAULT 0,
                   catalog_active INTEGER NOT NULL DEFAULT 0,
                   sealed INTEGER NOT NULL DEFAULT 0,
                   encode_job_state TEXT,
                   obs_count INTEGER NOT NULL DEFAULT 0,
                   PRIMARY KEY(task, dataset_hash)) WITHOUT ROWID;
                 INSERT INTO datasets VALUES
                   ('regression','r1',1,1,0,NULL,5),
                   ('regression','r2',1,0,0,NULL,0),
                   ('regression','r3',1,0,1,NULL,0),
                   ('regression','r4',0,0,0,'jope-regexp-c01',0),
                   ('regression','r5',0,0,0,NULL,0),
                   ('binary','b1',1,1,0,NULL,2),
                   ('binary','b2',0,0,0,'jope-binexp-b2c05',0);",
        )
        .unwrap();
    drop(connection);
    let rows = funnel_rows_from_index(&path).unwrap();
    let reg = &rows[0];
    assert_eq!(reg.task, "regression");
    assert_eq!(reg.registry, 5);
    assert_eq!(reg.active, 1);
    assert_eq!(reg.in_play, 1);
    assert_eq!(reg.queued, 1, "encoded, unsealed, not active");
    assert_eq!(reg.encoding, 1, "encode job assigned, not yet encoded");
    assert_eq!(reg.sealed, 1);
    assert_eq!(reg.locked(), 1);
    let bin = &rows[1];
    assert_eq!(bin.encoding, 1);
    assert_eq!(bin.queued, 0);
}

#[test]
fn funnel_line_stage_colors_and_exact_numbers() {
    let row = FunnelRow {
        task: "regression".to_owned(),
        registry: 27_525,
        encoded: 24_172,
        active: 6_412,
        in_play: 2_491,
        queued: 17_504,
        encoding: 3_353,
        sealed: 256,
        ..Default::default()
    };
    let line = funnel_line(&row);
    let text = line_text(&line);
    assert!(text.contains("27,525"), "{text}");
    assert!(text.contains("active 6,412"), "{text}");
    assert!(text.contains("(in-play 2,491)"), "{text}");
    assert!(text.contains("queued +17,504"), "{text}");
    assert!(text.contains("encoding 3,353"), "{text}");
    let style_of = |needle: &str| {
        line.spans
            .iter()
            .find(|span| span.content.contains(needle))
            .map(|span| span.style)
            .unwrap()
    };
    assert_eq!(style_of("active 6,412").fg, Some(STAGE_ACTIVE));
    assert_eq!(style_of("in-play").fg, Some(STAGE_IN_PLAY));
    assert_eq!(style_of("queued").fg, Some(STAGE_QUEUED));
    assert_eq!(style_of("encoding").fg, Some(STAGE_ENCODING));
    assert_eq!(style_of("locked").fg, Some(STAGE_LOCKED));
    let bar = funnel_bar(&row, 40);
    assert!(!bar.spans.is_empty());
    let bar_cells: usize = bar
        .spans
        .iter()
        .map(|span| span.content.chars().count())
        .sum();
    assert!((38..=42).contains(&bar_cells), "bar width {bar_cells}");
}

#[test]
fn cohort_label_shows_rotation_population() {
    let wave = WaveSummary {
        adaptive_regression: 6_412,
        adaptive_binary: 2_073,
        ..Default::default()
    };
    let label = tui_render::cohort_label(8_485, &wave);
    assert!(label.contains("128 datasets/wave"), "{label}");
    assert!(label.contains("rotating over 8,485 active"), "{label}");
    let fallback = tui_render::cohort_label(0, &wave);
    assert!(
        fallback.contains("rotating over 8,485 active"),
        "{fallback}"
    );
}

#[test]
fn thousands_formats_exact_counts() {
    assert_eq!(thousands(0), "0");
    assert_eq!(thousands(999), "999");
    assert_eq!(thousands(24_840), "24,840");
    assert_eq!(thousands(2_060_048), "2,060,048");
}

#[test]
fn funnel_deltas_measure_the_climb() {
    let mut history = VecDeque::new();
    let hour_ago = 1_000_000_u128;
    history.push_back((
        hour_ago,
        vec![("regression".to_owned(), 6_000_u64, 2_000_u64)],
    ));
    let mut rows = vec![FunnelRow {
        task: "regression".to_owned(),
        registry: 27_525,
        encoded: 24_172,
        active: 6_412,
        in_play: 2_491,
        ..Default::default()
    }];
    apply_funnel_deltas(&mut rows, &mut history, hour_ago + 1_800_000);
    assert_eq!(rows[0].delta_active_1h, 412);
    assert_eq!(rows[0].delta_in_play_1h, 491);
    assert_eq!(history.len(), 2);
    let text = line_text(&funnel_line(&rows[0]));
    assert!(text.contains("REG"), "{text}");
    assert!(text.contains("+412"), "{text}");
}
