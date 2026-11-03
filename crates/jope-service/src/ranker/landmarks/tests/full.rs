use super::*;

#[test]
#[ignore = "requires hash-pinned RES-07 inputs and captured full-recipe outputs"]
fn matches_res07_full_recipe_outputs_and_measures_rust_latency() {
    use std::time::Instant;

    const ATOL: f64 = 1e-5;
    const RTOL: f64 = 1e-4;
    fn check(context: &str, name: &str, got: &[f32], expected: &[f32]) {
        assert_eq!(got.len(), expected.len(), "{context}: {name} shape");
        for (i, (&a, &g)) in got.iter().zip(expected).enumerate() {
            assert!(
                a.is_finite() && g.is_finite(),
                "{context}: {name}[{i}] nonfinite"
            );
            let error = (f64::from(a) - f64::from(g)).abs();
            let tolerance = ATOL + RTOL * f64::from(g).abs();
            assert!(
                error <= tolerance,
                "{context}: {name}[{i}] actual={a} golden={g} error={error} tolerance={tolerance}"
            );
        }
    }

    let fixtures = super::super::super::adapter::real_cpu_parity::Fixtures::open();
    assert_eq!(fixtures.contexts.len(), 50);
    let mut elapsed = Vec::with_capacity(fixtures.contexts.len());
    let mut v_compared = 0;
    let mut raw_compared = 0;
    let mut ranker_compared = 0;
    let mut max_error = 0.0_f64;
    for context in &fixtures.contexts {
        let name = format!("landmarks/{context}.safetensors");
        let fixture = fixtures.load(&name);
        let schema = &fixtures.index["files"][&name]["tensors"];
        let shape = schema["input.X"]["shape"].as_array().unwrap();
        let rows = shape[0].as_u64().unwrap() as usize;
        let features = shape[1].as_u64().unwrap() as usize;
        assert!((1..=1_024).contains(&rows), "{context}: FIT row cap");
        let x = fixture.f32("input.X");
        let y = fixture.f32("input.y");
        let start = Instant::now();
        let got = fit_transform_and_extract_bundle_high(&x, rows, features, &y)
            .unwrap_or_else(|error| panic!("{context}: full Rust recipe failed: {error}"));
        elapsed.push(start.elapsed());

        let golden_vector = fixture.f32("v256");
        let raw_golden = fixture.f32("col_full");
        let raw_actual: Vec<f32> = got.raw_columns().iter().flatten().copied().collect();
        let ranker_golden = fixture.f32("col");
        let ranker_actual: Vec<f32> = got.columns().iter().flatten().copied().collect();
        check(context, "v256", got.vector(), &golden_vector);
        check(context, "col_full", &raw_actual, &raw_golden);
        check(context, "col", &ranker_actual, &ranker_golden);
        v_compared += golden_vector.len();
        raw_compared += raw_golden.len();
        ranker_compared += ranker_golden.len();
        for (a, g) in got
            .vector()
            .iter()
            .zip(&golden_vector)
            .chain(raw_actual.iter().zip(&raw_golden))
            .chain(ranker_actual.iter().zip(&ranker_golden))
        {
            max_error = max_error.max((f64::from(*a) - f64::from(*g)).abs());
        }
    }
    let latency = summarize_latency(&elapsed).unwrap();
    eprintln!("RNK-05 Rust full-recipe parity: contexts={} v256={} col_full={} col={} misses=0 max_abs_error={max_error:.9} landmarks.ms samples={} min={:.6} median={:.6} max={:.6}", fixtures.contexts.len(), v_compared, raw_compared, ranker_compared, latency.samples, latency.min_ms, latency.median_ms, latency.max_ms);
}

#[test]
#[ignore = "requires hash-pinned RES-07 sampled/full training capture"]
fn reports_res07_sampled_full_rust_agreement() {
    use sha2::{Digest, Sha256};
    use std::{env, fs, time::Instant};

    const ATOL: f64 = 1e-5;
    const RTOL: f64 = 1e-4;
    let train_path = env::var("JOPE_RNK05_TRAIN_CSV").unwrap();
    let indices_path = env::var("JOPE_RNK05_SAMPLE_INDICES").unwrap();
    let capture_path = env::var("JOPE_RNK05_CAPTURE_JSON").unwrap();
    let report_path = env::var("JOPE_RNK05_REPORT_JSON").unwrap();
    let train_bytes = fs::read(train_path).unwrap();
    let indices_bytes = fs::read(indices_path).unwrap();
    let capture: serde_json::Value =
        serde_json::from_slice(&fs::read(capture_path).unwrap()).unwrap();
    let train_sha = format!("{:x}", Sha256::digest(&train_bytes));
    let indices_sha = format!("{:x}", Sha256::digest(&indices_bytes));
    assert_eq!(train_sha, capture["train_file_sha256"]);
    assert_eq!(indices_sha, capture["sample_row_indices_sha256"]);
    assert_eq!(capture["schema"], "rnk05.sampled-full-recipe-capture/v1");

    let rows: Vec<Vec<f32>> = String::from_utf8(train_bytes)
        .unwrap()
        .lines()
        .map(|line| {
            line.split(',')
                .map(|value| value.parse::<f32>().unwrap())
                .collect()
        })
        .collect();
    let full_rows = rows.len();
    let features = rows[0].len() - 1;
    assert_eq!(
        full_rows,
        capture["training_rows"].as_u64().unwrap() as usize
    );
    assert_eq!(
        features + 1,
        capture["training_columns_including_target"]
            .as_u64()
            .unwrap() as usize
    );
    assert_eq!(indices_bytes.len() % 4, 0);
    let indices: Vec<usize> = indices_bytes
        .chunks_exact(4)
        .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()) as usize)
        .collect();
    assert_eq!(
        indices.len(),
        capture["sample_rows"].as_u64().unwrap() as usize
    );
    assert!(indices.iter().all(|&index| index < full_rows));
    let flatten_x = |selected: &[usize]| {
        selected
            .iter()
            .flat_map(|&i| rows[i][..features].iter().copied())
            .collect::<Vec<_>>()
    };
    let flatten_y = |selected: &[usize]| {
        selected
            .iter()
            .map(|&i| rows[i][features])
            .collect::<Vec<_>>()
    };
    let all_indices: Vec<usize> = (0..full_rows).collect();
    let sampled_x = flatten_x(&indices);
    let sampled_y = flatten_y(&indices);
    let full_x = flatten_x(&all_indices);
    let full_y = flatten_y(&all_indices);

    let started = Instant::now();
    let sampled =
        fit_transform_and_extract_bundle_high(&sampled_x, indices.len(), features, &sampled_y)
            .unwrap();
    let sampled_ms = started.elapsed().as_secs_f64() * 1_000.0;
    let started = Instant::now();
    let full = build_high_full_table(&full_x, full_rows, features, &full_y).unwrap();
    let full_ms = started.elapsed().as_secs_f64() * 1_000.0;
    assert!(indices.len() <= 1_024 && full_rows > indices.len());

    let check_golden = |label: &str, got: &[f32], expected: &[f32]| {
        assert_eq!(got.len(), expected.len(), "{label} shape");
        for (index, (&actual, &golden)) in got.iter().zip(expected).enumerate() {
            assert!(
                actual.is_finite() && golden.is_finite(),
                "{label}[{index}] nonfinite"
            );
            let error = (f64::from(actual) - f64::from(golden)).abs();
            assert!(
                error <= ATOL + RTOL * f64::from(golden).abs(),
                "{label}[{index}] actual={actual} golden={golden} error={error}"
            );
        }
    };
    for (name, output) in [("sampled", &sampled), ("full", &full)] {
        let expected_vector: Vec<f32> =
            serde_json::from_value(capture["outputs"][name]["vector256"].clone()).unwrap();
        let expected_columns: Vec<Vec<f32>> =
            serde_json::from_value(capture["outputs"][name]["col_full"].clone()).unwrap();
        let expected_flat: Vec<f32> = expected_columns.into_iter().flatten().collect();
        let actual_columns: Vec<f32> = output.raw_columns().iter().flatten().copied().collect();
        check_golden(&format!("{name}.v256"), output.vector(), &expected_vector);
        check_golden(&format!("{name}.col_full"), &actual_columns, &expected_flat);
    }
    let agreement = compare_sampled_to_full(&sampled, &full, indices.len(), full_rows).unwrap();
    let report = serde_json::json!({
        "schema": "rnk05.sampled-full-rust-agreement/v1",
        "dataset_id": capture["dataset_id"],
        "dataset_canonical_sha256": capture["dataset_canonical_sha256"],
        "training_sha256": train_sha,
        "sample_row_indices_sha256": indices_sha,
        "sample_method": capture["sample_method"],
        "sample_seed": capture["sample_seed"],
        "sampled_rows": agreement.sampled_rows,
        "full_rows": agreement.full_rows,
        "values_compared": agreement.values_compared,
        "exact_matches": agreement.exact_matches,
        "max_abs_error": agreement.max_abs_error,
        "mean_abs_error": agreement.mean_abs_error,
        "sampled_landmarks_ms": sampled_ms,
        "full_landmarks_ms": full_ms,
        "golden_tolerance": "finite; abs(actual-golden) <= 1e-5 + 1e-4*abs(golden)",
        "golden_misses": 0
    });
    fs::write(report_path, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
}
