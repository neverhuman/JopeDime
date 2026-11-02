use super::*;

#[test]
#[ignore = "requires externally staged, hash-pinned RES-07 CPU fixtures and bundle"]
fn matches_res07_real_landmark_output_assembly() {
    use super::super::adapter::real_cpu_parity::Fixtures;

    // PROVISIONAL, WATCHER-SET; owner may override. This reuses the frozen
    // RES-07/RNK-12 CPU parity arithmetic, not an owner-approved RNK-05 rule.
    const ATOL: f64 = 1e-5;
    const RTOL: f64 = 1e-4;

    fn assert_matches_golden(context: &str, name: &str, actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len(), "{context}: {name} length");
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            let actual = f64::from(actual);
            let expected = f64::from(expected);
            assert!(
                actual.is_finite() && expected.is_finite(),
                "{context}: {name}[{index}] must be finite"
            );
            let tolerance = ATOL + RTOL * expected.abs();
            let error = (actual - expected).abs();
            assert!(
                error <= tolerance,
                "{context}: {name}[{index}] actual={actual}, expected={expected}, error={error}, tolerance={tolerance}"
            );
        }
    }

    let fixtures = Fixtures::open();
    assert_eq!(fixtures.contexts.len(), 50);

    for context in &fixtures.contexts {
        let name = format!("landmarks/{context}.safetensors");
        let fixture = fixtures.load(&name);
        let schema = &fixtures.index["files"][&name]["tensors"];
        let input_shape = schema["input.X"]["shape"].as_array().unwrap();
        let fit_rows = input_shape[0].as_u64().unwrap() as usize;
        let feature_count = input_shape[1].as_u64().unwrap() as usize;

        assert!((1..=1_024).contains(&fit_rows), "{context}: FIT row cap");
        assert_eq!(
            schema["input.row_ids"]["shape"][0].as_u64().unwrap() as usize,
            fit_rows,
            "{context}: source row IDs must cover the FIT table"
        );
        assert_eq!(
            schema["input.y"]["shape"][0].as_u64().unwrap() as usize,
            fit_rows,
            "{context}: target rows must match the FIT table"
        );
        assert_eq!(
            schema["col_full"]["shape"][0].as_u64().unwrap() as usize,
            feature_count,
            "{context}: raw column output count must match feature count"
        );

        let vector = fixture.f32("v256");
        assert_eq!(vector.len(), VECTOR_WIDTH, "{context}: v256 width");
        let report: [f32; REPORT_WIDTH] = vector[..REPORT_WIDTH].try_into().unwrap();
        let table: [f32; TABLE_WIDTH] = vector[REPORT_WIDTH..REPORT_WIDTH + TABLE_WIDTH]
            .try_into()
            .unwrap();
        let interaction_start = REPORT_WIDTH + TABLE_WIDTH;
        let interactions: [f32; INTERACTION_WIDTH] = vector
            [interaction_start..interaction_start + INTERACTION_WIDTH]
            .try_into()
            .unwrap();
        let building_block_start = interaction_start + INTERACTION_WIDTH;
        let building_blocks: [f32; BUILDING_BLOCK_WIDTH] =
            vector[building_block_start..].try_into().unwrap();

        let raw_column_values = fixture.f32("col_full");
        assert_eq!(
            raw_column_values.len() % RAW_COLUMN_WIDTH,
            0,
            "{context}: raw column width"
        );
        let raw_columns = raw_column_values
            .chunks_exact(RAW_COLUMN_WIDTH)
            .map(|row| row.try_into().unwrap())
            .collect::<Vec<[f32; RAW_COLUMN_WIDTH]>>();

        let actual = LandmarkOutput::from_descriptor_parts(
            &report,
            &table,
            &interactions,
            &building_blocks,
            &raw_columns,
        );
        assert_matches_golden(context, "v256", actual.vector().as_slice(), &vector);

        let expected_columns = fixture.f32("col");
        let actual_columns = actual
            .columns()
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        assert_matches_golden(context, "col", &actual_columns, &expected_columns);
    }
}
