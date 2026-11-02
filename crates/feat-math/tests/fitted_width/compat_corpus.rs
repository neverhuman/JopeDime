use super::*;

// ---------------------------------------------------------------------------------------------
// PR 01's compatibility corpus
// ---------------------------------------------------------------------------------------------

fn fixture(name: &str) -> serde_json::Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/compat_freeze")
        .join(name);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{path:?}: {e}"))
}

/// Deterministic regression data, copied from `tests/golden.rs` through `compat_freeze.rs`
/// (both hash-pinned, so copied rather than shared).
fn toy_data(n: usize, cols: usize, seed: u64) -> (Rows, Vec<f64>) {
    let mut rng = Pcg32::new(seed, 17);
    let mut x = Vec::new();
    let mut y = Vec::new();
    for _ in 0..n {
        let row: Vec<f64> = (0..cols).map(|_| rng.next_f64() * 4.0 - 2.0).collect();
        let target = 1.7 * row[0] - 0.9 * row[1]
            + 0.5 * (3.0 * row[2]).sin()
            + 0.3 * row[3] * row[4]
            + 0.1 * rng.gaussian();
        x.push(row);
        y.push(target);
    }
    (x, y)
}

/// The corpus's fitted surface: the first 64 rows train, the last 16 are predicted.
fn corpus_data() -> (Rows, Vec<f64>, Rows) {
    let (x, y) = toy_data(80, 5, 42);
    (x[..64].to_vec(), y[..64].to_vec(), x[64..].to_vec())
}

fn row_text(values: &[f64]) -> String {
    values
        .iter()
        .map(|value| format!("{:016x}", value.to_bits()))
        .collect::<Vec<_>>()
        .join(" ")
}

/// The widths a pipeline model's rows have, replayed stage by stage on `x`: the input width,
/// then after every stage. Each is checked against the static walk on the way.
fn replay_widths(name: &str, model: &Model, x: &Rows) -> Vec<usize> {
    let Model::Pipeline { ops, head, .. } = model else {
        return Vec::new();
    };
    let mut current = x.clone();
    sanitize(&mut current);
    let mut width = current[0].len();
    let mut widths = vec![width];
    for (index, op) in ops.iter().enumerate() {
        assert!(
            required_input_width(op).admits(width),
            "{name}: stage {index} at width {width}"
        );
        let expected = output_width(op, width).unwrap_or_else(|e| panic!("{name}: {e}"));
        current = apply_op(op, &current).unwrap();
        assert!(
            current.iter().all(|row| row.len() == expected),
            "{name}: stage {index} output_width {expected}"
        );
        width = expected;
        widths.push(width);
    }
    assert!(
        required_head_width(head).admits(width),
        "{name}: the head at width {width}"
    );
    widths
}

#[test]
fn compat_corpus_try_predict_equals_predict_bit_for_bit() {
    let emitted = fixture("emitted_bytes.v1.json");
    let serde = fixture("model_serde.v1.json");
    let programs = emitted["programs"].as_array().expect("programs");
    let recorded = serde["models"].as_array().expect("models");
    assert_eq!(programs.len(), 55);
    let (train, target, held_out) = corpus_data();
    let (mut fitted, mut skipped) = (0, 0);
    let (mut accepted_elsewhere, mut refused_elsewhere) = (0, 0);
    for entry in programs {
        let name = entry["name"].as_str().expect("name");
        let canonical = entry["canonical"].as_str().expect("canonical");
        if canonical.contains("(portfolio") {
            // As in compat_freeze: a portfolio fit takes minutes in a debug build.
            skipped += 1;
            continue;
        }
        let model = fit(&program(canonical), &train, &target)
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        fitted += 1;

        let predicted = model.predict(&held_out).unwrap();
        let tried = model
            .try_predict(&held_out)
            .unwrap_or_else(|mismatch| panic!("{name}: {mismatch}"));
        assert_eq!(bits(&tried), bits(&predicted), "{name}");
        let record = recorded
            .iter()
            .find(|record| record["name"] == name)
            .unwrap_or_else(|| panic!("{name}: no recorded predictions"));
        assert_eq!(
            row_text(&tried),
            record["predictions"].as_str().expect("predictions"),
            "{name}: the recorded predictions"
        );

        let widths = replay_widths(name, &model, &held_out);
        match name {
            // poly_interact appends 512 pairs to 70 columns; the row is capped at 512.
            "champion" => assert_eq!(widths, [5, 10, 70, 70, 70, 512, 512], "{name}"),
            // One column leaves pair_ratio no pair; the empty row is padded to one column.
            "empty_stage_output" => assert_eq!(widths, [5, 1, 1], "{name}"),
            _ => {}
        }

        // At any other width, try_predict never panics, and a value it returns is predict's.
        for (index, width) in [0usize, 1, 4, 6, 9].into_iter().enumerate() {
            let x = data(4, width, 50 + index as u64);
            let outcome = catch_unwind(AssertUnwindSafe(|| model.try_predict(&x)))
                .unwrap_or_else(|_| panic!("{name}: try_predict panicked at width {width}"));
            match outcome {
                Ok(values) => {
                    assert_eq!(bits(&values), bits(&model.predict(&x).unwrap()), "{name}");
                    accepted_elsewhere += 1;
                }
                Err(_) => refused_elsewhere += 1,
            }
        }
    }
    assert_eq!((fitted, skipped), (54, 1));
    assert!(accepted_elsewhere > 0 && refused_elsewhere > 0);
    println!(
        "CORPUS fitted {fitted}, skipped {skipped}; at widths 0, 1, 4, 6 and 9: \
         {accepted_elsewhere} accepted, {refused_elsewhere} refused"
    );
}

#[test]
fn stored_compat_models_try_predict_equals_predict() {
    let serde = fixture("model_serde.v1.json");
    let stored = serde["stored"].as_array().expect("stored");
    assert_eq!(stored.len(), 5);
    let (_, _, held_out) = corpus_data();
    for entry in stored {
        let name = entry["name"].as_str().expect("name");
        // The same in-memory model on both paths: a reload may differ in the last bits from the
        // recording build (compat_freeze explains why), never between two calls.
        let model: Model = serde_json::from_str(entry["json"].as_str().expect("json"))
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(
            bits(&model.try_predict(&held_out).unwrap()),
            bits(&model.predict(&held_out).unwrap()),
            "{name}"
        );
        replay_widths(name, &model, &held_out);
    }
}
