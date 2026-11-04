use super::*;

#[test]
fn fixed_five_recipes_fitted_and_checkpoints_match_native_bits() {
    let dir = tempfile::tempdir().unwrap().keep();
    let mut receipts = Vec::new();
    for window in [2, 4] {
        for case in ["boundary", "numeric_groups", "minus_zero_labels"] {
            let p = program(window);
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
                    .emit_path_stats_fitted_cpp17()
                    .unwrap();
            assert_eq!(
                PathStatsFittedCpp::from_codec_json(&fitted.codec)
                    .unwrap()
                    .source,
                fitted.source
            );
            let expected = reference(native, &rows);
            let mut incompatible = Vec::new();
            let mut state = PathState::new(
                PathStatsConfig::new(2, 64, if window == 2 { 4 } else { 2 }).unwrap(),
                TaskMode::Forecasting,
                p.input_identity(),
                fold.clone(),
            )
            .unwrap();
            for row in &rows[..4] {
                state
                    .push(row.group.clone(), row.timestamp, &row.values)
                    .unwrap();
            }
            incompatible.push(state.to_bytes().unwrap());
            for lang in [
                PathStatsLanguage::Rust,
                PathStatsLanguage::Python,
                PathStatsLanguage::Cpp17,
            ] {
                let source = p.emit_recipe(lang).unwrap().source;
                let driver = match lang {
                    PathStatsLanguage::Rust => rust_driver(&rows, &labels, &incompatible),
                    PathStatsLanguage::Python => python_driver(&rows, &labels, &incompatible),
                    PathStatsLanguage::Cpp17 => cpp_driver(&rows, &labels, &incompatible, false),
                };
                let name = format!("window{window}-{case}-{lang:?}");
                let actual = execute(&dir, &name, &format!("{source}\n{driver}"), lang);
                assert_eq!(actual.replace(' ', ""), expected.replace(' ', ""), "{name}");
                fs::write(dir.join(format!("{name}.stdout")), &actual).unwrap();
                receipts.push(name);
            }
            let name = format!("window{window}-{case}-fitted");
            let actual = execute(
                &dir,
                &name,
                &format!(
                    "{}\n{}",
                    fitted.source,
                    cpp_driver(&rows, &labels, &incompatible, true)
                ),
                PathStatsLanguage::Cpp17,
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
    println!("24 compiled B16 recipes/fitted at {}", dir.display());
}
