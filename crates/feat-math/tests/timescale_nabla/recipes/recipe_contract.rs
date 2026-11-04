use super::*;

#[test]
fn irregular_recipes_fitted_and_checkpoints_match_native_bits() {
    let dir = tempfile::tempdir().unwrap().keep();
    let mut receipts = Vec::new();
    for gap in [0.125, 0.25] {
        for case in ["boundary", "numeric_groups", "minus_zero_labels"] {
            let p = program(gap);
            let mut rows = rows();
            rows[5].timestamp = 1.5;
            if case == "numeric_groups" {
                for row in &mut rows {
                    row.group =
                        GroupKey::numeric(if row.group == GroupKey::categorical("a").unwrap() {
                            -0.
                        } else {
                            3.
                        })
                        .unwrap();
                }
            }
            let labels = if case == "minus_zero_labels" {
                vec![-0.; 4]
            } else {
                vec![1., 2., 3., 4.]
            };
            let fold = FoldBoundary::new("recipe-cut", 4, 8).unwrap();
            let native = p
                .fit_training(&rows[..4], &labels, TaskMode::Forecasting, fold.clone())
                .unwrap();
            let fitted =
                feat_math::kernel::wire_v2::VersionedModel::from_json(&native.to_json().unwrap())
                    .unwrap()
                    .emit_time_gradient_fitted_cpp17()
                    .unwrap();
            assert_eq!(
                TimeGradientFittedCpp::from_codec_json(&fitted.codec)
                    .unwrap()
                    .source,
                fitted.source
            );
            let expected = reference(native, &rows);
            let mut incompatible = Vec::new();
            for (width, other_gap) in [(2, gap / 2.), (1, gap)] {
                let mut state = TimeGradientState::new(
                    TimeGradientConfig::new(2, 75, width, other_gap).unwrap(),
                    TaskMode::Forecasting,
                    p.input_identity(),
                    fold.clone(),
                )
                .unwrap();
                for row in &rows[..4] {
                    state
                        .push(row.group.clone(), row.timestamp, &row.values[..width])
                        .unwrap();
                }
                incompatible.push(state.to_bytes().unwrap());
            }
            for lang in [
                TimeGradientLanguage::Rust,
                TimeGradientLanguage::Python,
                TimeGradientLanguage::Cpp17,
            ] {
                let source =
                    feat_math::kernel::wire_v2::VersionedProgram::TimeGradientV2(p.clone())
                        .emit_time_gradient_recipe(lang)
                        .unwrap()
                        .source;
                let driver = match lang {
                    TimeGradientLanguage::Rust => rust_driver(&rows, &labels, &incompatible),
                    TimeGradientLanguage::Python => python_driver(&rows, &labels, &incompatible),
                    TimeGradientLanguage::Cpp17 => cpp_driver(&rows, &labels, &incompatible, false),
                };
                let name = format!("gap{:016x}-{case}-{lang:?}", gap.to_bits());
                let actual = execute(&dir, &name, &format!("{source}\n{driver}"), lang);
                assert_eq!(actual.replace(' ', ""), expected.replace(' ', ""), "{name}");
                fs::write(dir.join(format!("{name}.stdout")), &actual).unwrap();
                receipts.push(name);
            }
            let name = format!("gap{:016x}-{case}-fitted", gap.to_bits());
            let actual = execute(
                &dir,
                &name,
                &format!(
                    "{}\n{}",
                    fitted.source,
                    cpp_driver(&rows, &labels, &incompatible, true)
                ),
                TimeGradientLanguage::Cpp17,
            );
            assert_eq!(actual.replace(' ', ""), expected.replace(' ', ""), "{name}");
            fs::write(dir.join(format!("{name}.stdout")), actual).unwrap();
            receipts.push(name);
        }
    }
    fs::write(
        dir.join("artifacts.json"),
        serde_json::to_vec_pretty(&receipts).unwrap(),
    )
    .unwrap();
    println!("24 compiled B15 recipes/fitted at {}", dir.display());
}
