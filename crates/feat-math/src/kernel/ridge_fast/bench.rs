use crate::dsl::{ParamValue, Stage};
use crate::evolve::{
    evolve, random_canonical_pipeline, EvolveConfig, GenStats, Observer, RidgeScoring, Scored,
    HEAD_OPS,
};
use crate::heads::RidgeModel;
use crate::ops::{apply_op_train, fit_op};
use crate::transpile::Emitted;
use crate::{GenomeIr, Pcg32, Rows};
use sha2::{Digest, Sha256};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// The pinned 5000-row B11 fixture: twelve gaussian columns, two heavy-tailed columns that
/// activate the ridge clamp, one integer column and one near-constant column.
fn ridge_fixture() -> (Rows, Vec<f64>) {
    let mut rng = Pcg32::new(0xB11, 11);
    let x: Rows = (0..5000)
        .map(|_| {
            let mut row: Vec<f64> = (0..12).map(|_| rng.gaussian()).collect();
            row.push((2.0 * rng.gaussian()).exp());
            row.push((2.0 * rng.gaussian()).exp());
            row.push(f64::from(rng.next_u32() % 7));
            row.push(1.0 + 1e-9 * rng.gaussian());
            row
        })
        .collect();
    let y = x
        .iter()
        .map(|r| {
            1.5 * r[0] - 0.8 * r[1] * r[1]
                + 0.6 * (2.0 * r[2]).sin()
                + 0.4 * r[3] * r[4]
                + 0.1 * rng.gaussian()
        })
        .collect();
    (x, y)
}

/// Where a surrogate fold fit spends its time. Mirrors the search surrogate on random generated
/// genomes: transforms fitted and applied on a two-thirds training split, then the ridge head on
/// the final frame.
#[test]
#[ignore]
fn ridge_share_of_surrogate_fits() {
    let (x, y) = ridge_fixture();
    let train: Rows = x[..3333].to_vec();
    let target = &y[..3333];
    let (mut transforms, mut ridge, mut widths) = (0.0f64, 0.0f64, Vec::new());
    for seed in 0..120u64 {
        let GenomeIr::Pipeline(mut genome) =
            GenomeIr::parse(&random_canonical_pipeline(seed, 6)).unwrap()
        else {
            continue;
        };
        // The search surrogate: knobs capped as the search caps them, the head a plain ridge.
        for stage in &mut genome.stages {
            for (key, cap) in [("degree", 4i64), ("n", 64)] {
                if let Some(value) = stage.params.get_mut(key) {
                    if value.as_u64().is_some_and(|v| v as i64 > cap) {
                        *value = ParamValue::Int(cap);
                    }
                }
            }
        }
        if genome
            .stages
            .last()
            .is_some_and(|last| HEAD_OPS.contains(&last.op.as_str()))
        {
            genome.stages.pop();
        }
        let stages: Vec<Stage> = genome.stages;
        let started = Instant::now();
        let mut frame = train.clone();
        crate::sanitize(&mut frame);
        let mut ok = true;
        for stage in &stages {
            match fit_op(stage, &frame, target)
                .ok()
                .and_then(|op| apply_op_train(&op, &frame).ok())
            {
                Some(next) => frame = next,
                None => {
                    ok = false;
                    break;
                }
            }
        }
        if !ok {
            continue;
        }
        transforms += started.elapsed().as_secs_f64();
        let started = Instant::now();
        std::hint::black_box(RidgeModel::fit(&frame, target, 0.02));
        ridge += started.elapsed().as_secs_f64();
        widths.push(frame[0].len());
    }
    widths.sort_unstable();
    println!(
        "B11_SHARE genomes={} transforms_s={transforms:.3} ridge_s={ridge:.3} ridge_share={:.3} \
             width_median={} width_max={}",
        widths.len(),
        ridge / (ridge + transforms),
        widths[widths.len() / 2],
        widths[widths.len() - 1]
    );
}

/// Surrogate-search throughput on the pinned fixture. Prints wall time, evaluations per second
/// and a digest of the ranked result, so that two builds can be compared for speed and for bit
/// identity.
#[test]
#[ignore]
fn ridge_fast_v1_throughput() {
    struct Evals(AtomicU64);
    impl Observer for Evals {
        fn on_generation(&self, stats: &GenStats) {
            self.0.store(stats.evals, Ordering::Relaxed);
        }
        fn on_improvement(&self, _best: &Scored, _emitted: &Emitted, _stats: &GenStats) {}
    }

    let (x, y) = ridge_fixture();
    let metric = |truth: &[f64], predicted: &[f64]| crate::metrics::r2(truth, predicted);
    // Three pairs, the order alternating per pair, so drifting host load hits both arms alike.
    for run in 0..6 {
        let scoring = if (run % 2 == 0) == (run / 2 % 2 == 0) {
            RidgeScoring::Refit
        } else {
            RidgeScoring::Certified
        };
        let cfg = EvolveConfig {
            pop: 64,
            gens: 8,
            folds: 3,
            seed: 0x1EA7_BEEF,
            max_transforms: 6,
            ridge_scoring: scoring,
            ..EvolveConfig::default()
        };
        let obs = Evals(AtomicU64::new(0));
        let started = Instant::now();
        let ranked = evolve(&x, &y, &metric, &cfg, &obs);
        let secs = started.elapsed().as_secs_f64();
        let mut digest = Sha256::new();
        for scored in &ranked {
            digest.update(scored.dsl.as_bytes());
            digest.update(scored.cv_mean.to_bits().to_le_bytes());
            for fold in &scored.fold_scores {
                digest.update(fold.to_bits().to_le_bytes());
            }
        }
        let evals = obs.0.load(Ordering::Relaxed);
        println!(
            "B11_E2E run={run} scoring={scoring:?} secs={secs:.3} evals={evals} \
                 evals_per_sec={:.2} ranked={} digest={:x}",
            evals as f64 / secs,
            ranked.len(),
            digest.finalize()
        );
    }
}
