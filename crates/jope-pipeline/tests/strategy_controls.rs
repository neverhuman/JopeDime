#[path = "../../feat-math/src/kernel/strategy/legacy.rs"]
mod legacy_control;
#[path = "../../feat-math/src/kernel/strategy/lite_race.rs"]
mod lite_race_control;

use feat_math::{evolve, metrics::r2, Rows};

struct Quiet;
impl evolve::Observer for Quiet {
    fn on_generation(&self, _: &evolve::GenStats) {}
    fn on_improvement(
        &self,
        _: &evolve::Scored,
        _: &feat_math::transpile::Emitted,
        _: &evolve::GenStats,
    ) {
    }
}

fn fixture() -> (Rows, Vec<f64>) {
    let x = (0..24)
        .map(|i| vec![i as f64 / 7.0, (i % 5) as f64, (i * i % 11) as f64])
        .collect();
    let y = (0..24).map(|i| 0.7 * i as f64 + (i % 3) as f64).collect();
    (x, y)
}

fn config(fitness: evolve::FitnessMode) -> evolve::EvolveConfig {
    evolve::EvolveConfig {
        pop: 8,
        gens: 2,
        folds: 2,
        seed: 0x18_2026,
        max_transforms: 2,
        seed_genomes: vec![
            "(pipe X (standardize) (ridge))".to_owned(),
            "(pipe X (robust_scale) (ridge))".to_owned(),
        ],
        fitness,
        ..evolve::EvolveConfig::default()
    }
}

fn assert_scores_equal(left: &[evolve::Scored], right: &[evolve::Scored]) {
    assert_eq!(left.len(), right.len());
    for (a, b) in left.iter().zip(right) {
        assert_eq!(a.dsl, b.dsl);
        assert_eq!(a.cv_mean.to_bits(), b.cv_mean.to_bits());
        assert_eq!(
            a.fold_scores
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            b.fold_scores
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn legacy_control_matches_frozen_loop_draw_order_and_champions() {
    assert_eq!(legacy_control::STRATEGY_ID, "feat-math.legacy-gp.v1");
    let (x, y) = fixture();
    let cfg = config(evolve::FitnessMode::Surrogate);
    let direct = evolve::evolve(&x, &y, &r2, &cfg, &Quiet);
    let controlled = legacy_control::run(&x, &y, &r2, &cfg, &Quiet);
    assert_scores_equal(&direct, &controlled);
}

#[test]
fn lite_race_control_preserves_same_cut_rungs_and_champions() {
    assert_eq!(lite_race_control::STRATEGY_ID, "feat-math.lite-race.v1");
    let (x, y) = fixture();
    let cfg = config(evolve::FitnessMode::LiteRace);
    let direct = evolve::evolve_lite_race(&x, &y, &r2, &cfg, &Quiet);
    let controlled = lite_race_control::run(&x, &y, &r2, &cfg, &Quiet);
    assert_scores_equal(&direct.ranked, &controlled.ranked);
    assert_eq!(direct.rung_evals, controlled.rung_evals);
    assert_eq!(direct.rung0_rows, controlled.rung0_rows);
    assert_eq!(direct.rung0_assignments, controlled.rung0_assignments);
}
