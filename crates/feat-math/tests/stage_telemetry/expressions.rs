use super::*;

/// PR 01's compatibility corpus, `compat_freeze/mathexpr_values.v1.json`: every row (the empty
/// row among them), in its order, and every expression source.
fn corpus() -> (Rows, Vec<String>) {
    let file = fixture_json("compat_freeze/mathexpr_values.v1.json");
    let rows: Rows = file["rows"]
        .as_array()
        .expect("rows")
        .iter()
        .map(|row| {
            // The empty row is recorded as "": no cells, not one empty token.
            row["values"]
                .as_str()
                .expect("row values")
                .split(' ')
                .filter(|token| !token.is_empty())
                .map(|token| {
                    f64::from_bits(
                        u64::from_str_radix(token, 16)
                            .unwrap_or_else(|error| panic!("{token}: {error}")),
                    )
                })
                .collect()
        })
        .collect();
    let expressions: Vec<String> = file["expressions"]
        .as_array()
        .expect("expressions")
        .iter()
        .map(|expression| expression["source"].as_str().expect("source").to_string())
        .collect();
    assert!(
        rows.iter().any(Vec::is_empty) && !expressions.is_empty(),
        "the corpus has an empty row and at least one expression"
    );
    (rows, expressions)
}

#[test]
fn expression_stages_equal_apply_op_on_every_corpus_expression_and_row() {
    let (rows, expressions) = corpus();
    let nonfinite = rows.iter().flatten().filter(|v| !v.is_finite()).count() as u64;
    assert!(nonfinite > 0, "the corpus has non-finite cells");
    let mut stages = 0;
    let mut faulted_stages = 0;
    let mut hard_stages = 0;
    for source in &expressions {
        let expr = MathExpr::parse(source).expect("a corpus expression parses");
        for append in [false, true] {
            let ops = [
                FittedOp::MapExpr {
                    source: source.clone(),
                    expr: expr.clone(),
                    append,
                },
                FittedOp::DeriveExpr {
                    source: source.clone(),
                    expr: expr.clone(),
                    append,
                },
            ];
            for op in &ops {
                let counts = check_op(&format!("{source} append={append}"), op, &rows);
                // Expression outputs are finite, so only merged input cells reach the epilogue.
                let merged = if append { nonfinite } else { 0 };
                assert_eq!(
                    counts.epilogue_sanitized, merged,
                    "{source} append={append}"
                );
                stages += 1;
                faulted_stages += usize::from(counts.faulted_cells.iter().any(|&n| n > 0));
                hard_stages += usize::from(counts.hard_rows > 0);
            }
        }
    }
    eprintln!(
        "BOUNDARY expressions={} rows={} stages={stages} faulted={faulted_stages} hard={hard_stages}",
        expressions.len(),
        rows.len()
    );
    assert!(faulted_stages > 0 && hard_stages > 0);
}

#[test]
fn every_corpus_expression_as_a_symbolic_program_fits_to_fits_bytes() {
    // fit_traced copies fit's linear scaling of a symbolic program. The corpus programs take its
    // slope branch only a few times, so every corpus expression is also fitted here, raw and
    // affine, on eight-column rows.
    let (x, y) = toy_data(80, 8, 42);
    let (train, target, test) = (x[..64].to_vec(), y[..64].to_vec(), x[64..].to_vec());
    let (corpus_rows, expressions) = corpus();
    let raw: Rows = corpus_rows
        .into_iter()
        .filter(|row| !row.is_empty())
        .collect();
    let mut slopes = 0;
    for source in &expressions {
        let expr = MathExpr::parse(source).expect("a corpus expression parses");
        // A symbolic program evaluates its expression per row, as a derive stage does.
        let derive = FittedOp::DeriveExpr {
            source: source.clone(),
            expr: expr.clone(),
            append: false,
        };
        for affine in [false, true] {
            let at = format!(
                "(sym {} \"{source}\")",
                if affine { "affine" } else { "raw" }
            );
            let program = GenomeIr::Symbolic {
                source: source.clone(),
                expr: expr.clone(),
                affine,
            };
            let model = fit(&program, &train, &target).expect("fits");
            let (traced, telemetry) = fit_traced(&program, &train, &target).expect("fits");
            assert_eq!(serialized(&traced), serialized(&model), "{at}");
            assert_eq!(telemetry.stages.len(), 1, "{at}");
            assert_eq!(
                fields(&telemetry.stages[0]),
                recount(&derive, &train),
                "{at}"
            );
            if let Model::Symbolic { a, .. } = &model {
                slopes += usize::from(affine && *a != 0.0);
            }
            for rows in [&test, &raw] {
                let (predictions, pass) = predict_traced(&model, rows).expect("no zero-width row");
                assert_eq!(
                    bits(&predictions),
                    bits(&model.predict(rows).unwrap()),
                    "{at}"
                );
                assert_eq!(fields(&pass.stages[0]), recount(&derive, rows), "{at}");
            }
        }
    }
    eprintln!(
        "SYMBOLIC expressions={} affine fits with a slope={slopes}",
        expressions.len()
    );
    assert!(slopes >= 40, "the slope branch is taken {slopes} times");
}
