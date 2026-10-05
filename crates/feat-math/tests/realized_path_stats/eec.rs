//! GP-created and edited wire-75 candidates retain working export closure.
use feat_math::kernel::ordered_axis::{FoldBoundary, OrderedInputIdentity, TaskMode};
use feat_math::kernel::realized_path_stats::{
    Edit, ExperimentalProfile, PathState, PathStatsLanguage, PathStatsSearch,
};
use feat_math::kernel::wire_v2::{VersionedModel, VersionedProgram};
use feat_math::GenomeIr;

#[test]
fn grow_change_remove_round_trip_and_compile_three_recipes_and_fitted_cpp() {
    let search = PathStatsSearch::new(
        1,
        OrderedInputIdentity::new("group", "time"),
        GenomeIr::parse("(pipe X (ridge lambda_frac=0.01))").unwrap(),
        TaskMode::Forecasting,
    )
    .unwrap();
    let profile = ExperimentalProfile {
        path_stats: true,
        prior: 0.125,
    };
    let mut rows = super::recipes::rows();
    // EEC uses a finite cohort admitted by both the ordered stage and its removed plain Ridge child.
    for (i, row) in rows.iter_mut().enumerate() {
        row.timestamp = (i / 2) as f64 * 2.0;
        for v in &mut row.values {
            if !v.is_finite() {
                *v = 0.0;
            }
        }
    }
    let labels = [0.25, 0.5, 1., 2.];
    let dir = tempfile::tempdir().unwrap().keep();
    let mut executed = 0;
    for seed in [7, 11] {
        let grow = search
            .edit(&search.base(), seed, Edit::Grow, profile)
            .unwrap();
        let change = search.edit(&grow, seed, Edit::Change, profile).unwrap();
        assert_ne!(grow.to_bytes().unwrap(), change.to_bytes().unwrap());
        let removed = search
            .edit(
                &change,
                seed,
                Edit::Drop,
                ExperimentalProfile {
                    path_stats: false,
                    prior: profile.prior,
                },
            )
            .unwrap();
        assert_eq!(
            removed.to_bytes().unwrap(),
            search.base().to_bytes().unwrap()
        );
        if seed == 7 {
            super::utility_exports::check(&removed, &rows, &labels, 4, &dir, "drop");
            executed += 4;
        }
        for (name, node) in [("grow", grow), ("change", change)] {
            let bytes = node.to_bytes().unwrap();
            let restored = VersionedProgram::from_bytes(&bytes).unwrap();
            assert_eq!(restored.to_bytes().unwrap(), bytes);
            let text = restored.to_lossless_text().unwrap();
            assert_eq!(
                VersionedProgram::parse_lossless_text(&text)
                    .unwrap()
                    .to_bytes()
                    .unwrap(),
                bytes
            );
            let VersionedProgram::PathStatsV2(ref program) = restored else {
                panic!("created wire64")
            };
            let native = program
                .fit_training(
                    &rows[..4],
                    &labels,
                    TaskMode::Forecasting,
                    FoldBoundary::new("recipe-cut", 4, 8).unwrap(),
                )
                .unwrap();
            let incompatible = vec![PathState::new(
                *program.config(),
                TaskMode::Forecasting,
                program.input_identity(),
                FoldBoundary::new("recipe-cut", 4, 8).unwrap(),
            )
            .unwrap()
            .to_bytes()
            .unwrap()];
            let fitted = VersionedModel::from_json(&native.to_json().unwrap())
                .unwrap()
                .emit_path_stats_fitted_cpp17()
                .unwrap();
            let expected = super::recipes::reference(native, &rows);
            for (language, driver, suffix) in [
                (
                    PathStatsLanguage::Rust,
                    super::recipes::rust_driver(&rows, &labels, &incompatible),
                    "rust",
                ),
                (
                    PathStatsLanguage::Python,
                    super::recipes::python_driver(&rows, &labels, &incompatible),
                    "python",
                ),
                (
                    PathStatsLanguage::Cpp17,
                    super::recipes::cpp_driver(&rows, &labels, &incompatible, false),
                    "cpp",
                ),
            ] {
                let recipe = restored.emit_path_stats_recipe(language).unwrap();
                assert_eq!(
                    super::recipes::execute(
                        &dir,
                        &format!("seed{seed}-{name}-{suffix}"),
                        &(recipe.source + &driver),
                        language
                    ),
                    expected
                );
                executed += 1;
            }
            assert_eq!(
                super::recipes::execute(
                    &dir,
                    &format!("seed{seed}-{name}-fitted"),
                    &(fitted.source
                        + &super::recipes::cpp_driver(&rows, &labels, &incompatible, true)),
                    PathStatsLanguage::Cpp17
                ),
                expected
            );
            executed += 1;
        }
    }
    assert_eq!(executed, 20);
    eprintln!("wire64 GP closure artifacts: {}", dir.display());
}
