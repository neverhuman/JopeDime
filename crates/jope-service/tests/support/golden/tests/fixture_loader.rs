#[test]
fn synthetic_fixture_loads_candle_tensor_and_flat_values() {
    let directory = tempdir().unwrap();
    fs::write(
        directory.path().join("synthetic.safetensors"),
        fixture_bytes(),
    )
    .unwrap();
    let schema = [
        TensorSpec::new("input", vec![2, 2]),
        TensorSpec::new("extra", vec![1]),
    ];

    let fixture = load_from_dir(directory.path(), "synthetic", &schema, true)
        .unwrap()
        .unwrap();

    assert_eq!(fixture.tensors["input"].dims(), &[2, 2]);
    assert_eq!(fixture.values_f32["input"], [1.25, -2.0, 3.5, 4.0]);
    assert_eq!(fixture.values_f32["extra"], [0.5]);
}

#[test]
#[ignore = "pre-stage scaffold; unignore only after RES-07 publishes real full-high fixtures and recipe parity is wired"]
fn synthetic_landmark_qualification_harness_uses_res09_loader() {
    let directory = tempdir().unwrap();
    fs::create_dir_all(directory.path().join("rnk05")).unwrap();
    fs::write(
        directory
            .path()
            .join("rnk05/synthetic-full-high.safetensors"),
        landmark_fixture_bytes(),
    )
    .unwrap();

    let schema = [
        TensorSpec::new("input.X", vec![4, 2]),
        TensorSpec::new("input.y", vec![4]),
        TensorSpec::new("input.row_ids", vec![4]),
        TensorSpec::new("report", vec![REPORT_WIDTH]),
        TensorSpec::new("table", vec![TABLE_WIDTH]),
        TensorSpec::new("interactions", vec![INTERACTION_WIDTH]),
        TensorSpec::new("building_blocks", vec![BUILDING_BLOCK_WIDTH]),
        TensorSpec::new("col_full", vec![2, RAW_COLUMN_WIDTH]),
        TensorSpec::new("v256", vec![1, VECTOR_WIDTH]),
        TensorSpec::new("col", vec![1, RANKER_COLUMN_COUNT, RANKER_COLUMN_WIDTH]),
        TensorSpec::new("sampled_rows", vec![1]),
        TensorSpec::new("full_rows", vec![1]),
        TensorSpec::new("latency_ms", vec![3]),
    ];
    let fixture = load_from_dir(directory.path(), "rnk05/synthetic-full-high", &schema, true)
        .unwrap()
        .unwrap();
    let values = &fixture.values_f32;
    assert_eq!(fixture.tensors["input.X"].dims(), &[4, 2]);
    assert_eq!(fixture.tensors["input.y"].dims(), &[4]);
    assert_eq!(fixture.tensors["input.row_ids"].dims(), &[4]);
    assert_eq!(fixture.tensors["input.row_ids"].dtype(), DType::I64);
    assert_eq!(values["input.row_ids"], [10.0, 11.0, 20.0, 21.0]);
    let report = fixture_array::<REPORT_WIDTH>(&fixture, "report");
    let table = fixture_array::<TABLE_WIDTH>(&fixture, "table");
    let interactions = fixture_array::<INTERACTION_WIDTH>(&fixture, "interactions");
    let building_blocks = fixture_array::<BUILDING_BLOCK_WIDTH>(&fixture, "building_blocks");
    let raw_column_values = &values["col_full"];
    let raw_columns: Vec<[f32; RAW_COLUMN_WIDTH]> = raw_column_values
        .chunks_exact(RAW_COLUMN_WIDTH)
        .map(|row| row.try_into().unwrap())
        .collect();

    let full = LandmarkOutput::from_descriptor_parts(
        &report,
        &table,
        &interactions,
        &building_blocks,
        &raw_columns,
    );
    assert_eq!(
        full.vector(),
        &fixture_array::<VECTOR_WIDTH>(&fixture, "v256")
    );
    assert_eq!(
        full.columns().iter().flatten().copied().collect::<Vec<_>>(),
        values["col"]
    );

    let mut sampled_report = report;
    sampled_report[0] += 1.0;
    let sampled = LandmarkOutput::from_descriptor_parts(
        &sampled_report,
        &table,
        &interactions,
        &building_blocks,
        &raw_columns,
    );
    let sampled_rows = values["sampled_rows"][0] as usize;
    let full_rows = values["full_rows"][0] as usize;
    let agreement = compare_sampled_to_full(&sampled, &full, sampled_rows, full_rows).unwrap();
    let compared = VECTOR_WIDTH + RANKER_COLUMN_COUNT * RANKER_COLUMN_WIDTH;
    assert_eq!(agreement.sampled_rows, 2);
    assert_eq!(agreement.full_rows, 4);
    assert_eq!(agreement.values_compared, compared);
    assert_eq!(agreement.exact_matches, compared - 1);
    assert_eq!(agreement.max_abs_error, 1.0);
    assert!((agreement.mean_abs_error - 1.0 / compared as f64).abs() < f64::EPSILON);

    let latency = values["latency_ms"]
        .iter()
        .map(|milliseconds| Duration::from_secs_f64(f64::from(*milliseconds) / 1_000.0))
        .collect::<Vec<_>>();
    let latency = summarize_latency(&latency).unwrap();
    assert_eq!(latency.samples, 3);
    assert_eq!(latency.min_ms, 1.0);
    assert!((latency.median_ms - 2.0).abs() < 1e-9);
    assert_eq!(latency.max_ms, 3.0);
}

#[test]
fn schema_rejects_missing_extra_and_wrong_shape_tensors() {
    let directory = tempdir().unwrap();
    fs::write(
        directory.path().join("synthetic.safetensors"),
        fixture_bytes(),
    )
    .unwrap();

    let wrong_shape = [
        TensorSpec::new("input", vec![4]),
        TensorSpec::new("extra", vec![1]),
    ];
    assert!(
        load_from_dir(directory.path(), "synthetic", &wrong_shape, true)
            .unwrap_err()
            .to_string()
            .contains("expected [4]")
    );

    let extra = [TensorSpec::new("input", vec![2, 2])];
    assert!(load_from_dir(directory.path(), "synthetic", &extra, true)
        .unwrap_err()
        .to_string()
        .contains("tensor names"));

    let missing = [
        TensorSpec::new("input", vec![2, 2]),
        TensorSpec::new("expected", vec![2, 2]),
    ];
    assert!(load_from_dir(directory.path(), "synthetic", &missing, true)
        .unwrap_err()
        .to_string()
        .contains("missing required tensor `expected`"));
}

#[test]
fn missing_fixture_skips_unless_parity_is_strict() {
    let directory = tempdir().unwrap();
    let schema = [TensorSpec::new("input", vec![2, 2])];

    assert!(load_from_dir(directory.path(), "absent", &schema, false)
        .unwrap()
        .is_none());
    assert!(load_from_dir(directory.path(), "absent", &schema, true)
        .unwrap_err()
        .to_string()
        .contains("missing golden fixture"));
}

#[test]
fn missing_fixture_obeys_strict_environment_and_prints_skip_diagnostic() {
    if let Some(mode) = env::var_os(TEST_PROBE_ENV) {
        let schema = [TensorSpec::new("input", vec![2, 2])];
        match mode.to_string_lossy().as_ref() {
            "strict" => assert!(load_named("absent", &schema)
                .unwrap_err()
                .to_string()
                .contains("missing golden fixture")),
            "skip" => assert!(load_named("absent", &schema).unwrap().is_none()),
            other => panic!("unknown test probe mode: {other}"),
        }
        return;
    }

    let directory = tempdir().unwrap();
    for (mode, strict) in [("strict", "1"), ("skip", "0")] {
        let output = Command::new(env::current_exe().unwrap())
            .arg("--exact")
            .arg("tests::missing_fixture_obeys_strict_environment_and_prints_skip_diagnostic")
            .arg("--nocapture")
            .env(TEST_PROBE_ENV, mode)
            .env(GOLDEN_DIR_ENV, directory.path())
            .env(STRICT_ENV, strict)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{mode} environment probe failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        if mode == "skip" {
            let output = format!(
                "{}{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(output.contains("skipping golden fixture: missing golden fixture"));
        }
    }
}
