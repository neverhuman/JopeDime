use super::*;

#[test]
fn four_orders_train_and_continue_through_all_three_public_recipes_and_fitted_cpp() {
    use feat_math::kernel::export::Route;
    let original = rows();
    let mut mixed = original.clone();
    for (i, row) in mixed.iter_mut().enumerate() {
        row.timestamp = i as f64;
        row.group = match i {
            0 => GroupKey::NumericBits((-0.0f64).to_bits()),
            1 => GroupKey::numeric(0.).unwrap(),
            2 | 6 => GroupKey::numeric(-1.).unwrap(),
            3 => GroupKey::numeric(1.).unwrap(),
            4 => GroupKey::numeric(2.5).unwrap(),
            _ => GroupKey::categorical("category").unwrap(),
        };
    }
    let mut negative = original.clone();
    for (i, row) in negative.iter_mut().enumerate() {
        row.values = vec![if i < 4 { 0.0 } else { -(i as f64) }; 2];
    }
    let dir = tempfile::tempdir().unwrap().keep(); // Preserve generated proof artifacts.
    for (case, rows, labels) in [
        ("boundary", original.clone(), vec![0.25, 0.5, 1., 2.]),
        ("groups", mixed, vec![0.25, 0.5, 1., 2.]),
        ("negative-zero", negative, vec![-0.0; 4]),
    ] {
        for order in 1..=4 {
            if case != "boundary" && order != 1 {
                continue;
            }
            let program = program(order);
            let routed = VersionedProgram::from_bytes(&program.to_bytes().unwrap()).unwrap();
            let report = export_report(&routed);
            assert!(matches!(report.rust_recipe, Route::Ok));
            assert!(matches!(report.python_recipe, Route::Ok));
            assert!(matches!(report.cpp17_recipe, Route::Ok));
            assert!(matches!(report.cpp17_fitted, Route::Ok));
            let native = program
                .fit_training(
                    &rows[..4],
                    &labels,
                    TaskMode::Forecasting,
                    FoldBoundary::new("recipe-cut", 4, 8).unwrap(),
                )
                .unwrap();
            let mut incompatible = Vec::new();
            for (other_order, step, width) in [
                (if order == 4 { 1 } else { order + 1 }, 0.5, 2),
                (order, 1., 2),
                (order, 0.5, 1),
            ] {
                let p = DifferenceProgram::new(
                    DifferenceConfig::new(2, 59, other_order, step).unwrap(),
                    width,
                    program.input_identity().clone(),
                    program.downstream().clone(),
                )
                .unwrap();
                let mut training = rows[..4].to_vec();
                for row in &mut training {
                    row.values.truncate(width);
                }
                incompatible.push(state(
                    &p.fit_training(
                        &training,
                        &labels,
                        TaskMode::Forecasting,
                        FoldBoundary::new("recipe-cut", 4, 8).unwrap(),
                    )
                    .unwrap(),
                ));
            }
            incompatible.push(
                DifferenceState::new(
                    program.config(),
                    2,
                    TaskMode::Forecasting,
                    program.input_identity(),
                    FoldBoundary::new("recipe-cut", 4, 8).unwrap(),
                )
                .unwrap()
                .to_bytes()
                .unwrap(),
            );
            let tagged = VersionedModel::from_json(&native.to_json().unwrap()).unwrap();
            let fitted = tagged.emit_ordered_fitted_cpp17().unwrap();
            let expected = reference(native, &rows);
            if case == "negative-zero" {
                assert!(expected.contains("8000000000000000"));
            }
            for (lang, driver, name) in [
                (
                    DifferenceLanguage::Rust,
                    rust_driver(&rows, &labels, &incompatible),
                    "rust",
                ),
                (
                    DifferenceLanguage::Python,
                    python_driver(&rows, &labels, &incompatible),
                    "python",
                ),
                (
                    DifferenceLanguage::Cpp17,
                    cpp_driver(&rows, &labels, &incompatible, false),
                    "cpp",
                ),
            ] {
                let recipe = routed.emit_ordered_recipe(lang).unwrap();
                assert_eq!(recipe.program_identity, program.identity().unwrap());
                assert_eq!(
                    execute(
                        &dir,
                        &format!("{case}-order{order}-{name}"),
                        &(recipe.source + &driver),
                        lang
                    ),
                    expected
                );
            }
            assert_eq!(
                execute(
                    &dir,
                    &format!("{case}-order{order}-fitted"),
                    &(fitted.source + &cpp_driver(&rows, &labels, &incompatible, true)),
                    DifferenceLanguage::Cpp17
                ),
                expected
            );
        }
    }
    eprintln!("ordered recipe proof artifacts: {}", dir.display());
}
