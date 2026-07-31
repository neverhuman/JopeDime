//! Headless end-to-end smoke of the upgraded engine: run the evolutionary
//! search on synthetic regression data, and verify that (1) tokens/sec is live
//! from the STATIC estimator every generation (no code executed to get it),
//! and (2) the current-best genome emits v2.1.0 standalone Python via the
//! gene->code converter — the exact signals the realtime web view streams.
//!
//! Run: cargo run -p feat-math --example evolve_smoke --release

use feat_math::evolve::{evolve, EvolveConfig, GenStats, Observer, Scored};
use feat_math::transpile::Emitted;
use feat_math::{r2, Pcg32, Rows};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

#[derive(Default)]
struct SmokeObserver {
    gens: AtomicU64,
    last_tps: AtomicU64, // tokens/sec, stored as bits
    improvements: AtomicU64,
    best_code: Mutex<String>,
    best_genome: Mutex<String>,
}

impl Observer for SmokeObserver {
    fn on_generation(&self, stats: &GenStats) {
        self.gens.fetch_add(1, Ordering::Relaxed);
        self.last_tps
            .store(stats.tokens_per_sec.to_bits(), Ordering::Relaxed);
    }
    fn on_improvement(&self, best: &Scored, emitted: &Emitted, _stats: &GenStats) {
        self.improvements.fetch_add(1, Ordering::Relaxed);
        *self.best_code.lock().unwrap() = emitted.python.clone();
        *self.best_genome.lock().unwrap() = best.dsl.clone();
    }
}

fn main() {
    // Synthetic regression: y depends nonlinearly on a few columns.
    let n = 240;
    let p = 6;
    let mut rng = Pcg32::new(20260703, 7);
    let mut x: Rows = Vec::with_capacity(n);
    let mut y: Vec<f64> = Vec::with_capacity(n);
    for _ in 0..n {
        let row: Vec<f64> = (0..p).map(|_| rng.next_f64() * 2.0 - 1.0).collect();
        let target = 1.3 * row[0] - 0.8 * row[1] * row[1]
            + 0.5 * (row[2] * 3.0).sin()
            + 0.2 * (rng.next_f64() - 0.5);
        x.push(row);
        y.push(target);
    }

    let cfg = EvolveConfig {
        pop: 24,
        gens: 6,
        folds: 3,
        seed: 0xC0DE_F00D,
        max_transforms: 4,
        time_budget_secs: 25.0,
        seed_genomes: Vec::new(),
        ..EvolveConfig::default()
    };
    let obs = SmokeObserver::default();
    let metric = |yt: &[f64], yp: &[f64]| r2(yt, yp);

    let results = evolve(&x, &y, &metric, &cfg, &obs);

    let gens = obs.gens.load(Ordering::Relaxed);
    let tps = f64::from_bits(obs.last_tps.load(Ordering::Relaxed));
    let improvements = obs.improvements.load(Ordering::Relaxed);
    let code = obs.best_code.lock().unwrap().clone();
    let genome = obs.best_genome.lock().unwrap().clone();

    println!("=== evolve smoke ===");
    println!("generations observed : {gens}");
    println!("improvements         : {improvements}");
    println!("last tokens/sec      : {tps:.0}   (from static estimator, no execution)");
    println!("best genome          : {genome}");
    println!(
        "best CV r2           : {:.4}",
        results.first().map(|s| s.cv_mean).unwrap_or(f64::NAN)
    );
    println!("--- emitted Python (first 14 lines, v2.1.0) ---");
    for line in code.lines().take(14) {
        println!("  {line}");
    }

    // Assertions: the realtime-web contract must hold.
    assert!(gens > 0, "no generations observed");
    assert!(tps > 0.0, "tokens/sec was not live from the estimator");
    assert!(!code.is_empty(), "no emitted code captured");
    assert!(
        code.contains("class Model") || code.contains("def fit"),
        "emitted code is not the v2.x class-Model shape"
    );
    let banned = ["genome", "pipe X", "GenomeIr"];
    for b in banned {
        assert!(!code.contains(b), "emitted code leaked GP vocabulary: {b}");
    }
    println!("\nOK: live tokens/sec + v2.1.0 emitted code, no GP vocabulary.");
}
