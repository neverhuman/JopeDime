//! End-to-end checks of the per-generation GP summary and the collapse alarm on synthetic
//! populations.

use jope_telemetry::alarm::{AlarmId, Engine, Thresholds};
use jope_telemetry::gp::{
    Candidate, Config, EditOp, EvalStatus, Genome, Interner, ProposalCounters, Provenance, Tracker,
    NO_PREDICTION_SIGNATURES,
};

struct Pool {
    clauses: Interner,
    ops: Interner,
}

impl Pool {
    fn cand(&mut self, dsl: &str, fitness: f64, gate: bool, behavior: Vec<f64>) -> Candidate {
        Candidate {
            program: jope_telemetry::gp::program_sha256(dsl),
            genome: Genome::from_dsl(dsl, &mut self.clauses, &mut self.ops),
            parent: None,
            edit_op: EditOp::EditParameter,
            status: EvalStatus::Ok,
            gate_passed: gate,
            fitness,
            improved: Some(fitness < -1.0),
            hwm_winner: false,
            complexity: Some(dsl.matches('(').count() as u32),
            proj16: Some(dsl.len() as u16),
            behavior,
            provenance: Provenance::Novel,
        }
    }
}

#[test]
fn summary_counts_duplicates_sentinels_parents_and_stagnation() {
    let mut p = Pool {
        clauses: Interner::default(),
        ops: Interner::default(),
    };
    let mut gen0 = vec![
        p.cand("(pipe X (a q=1) (h))", -1.2, true, vec![0.1, 0.5, 0.9, 0.2]),
        p.cand("(pipe X (a q=1) (h))", -1.2, true, vec![0.1, 0.5, 0.9, 0.2]),
        p.cand(
            "(pipe X (b) (a q=2) (h))",
            -0.8,
            true,
            vec![0.9, 0.1, 0.3, 0.7],
        ),
        p.cand("(pipe X (c) (h2))", 100.0, false, vec![f64::NAN; 4]),
    ];
    gen0[3].status = EvalStatus::Timeout;
    gen0[2].parent = Some(gen0[0].genome.clone());
    gen0[2].edit_op = EditOp::EditInsertStage;

    let cfg = Config {
        op_vocab_size: Some(8),
        ..Default::default()
    };
    let mut t = Tracker::new("ctx", cfg);
    let g = t.summarize(
        0,
        0,
        &gen0,
        ProposalCounters {
            proposals: 10,
            seen_rejects: 6,
        },
    );

    assert_eq!(g.pop_size, 4);
    assert_eq!(g.unique_genotype_frac, Some(0.75));
    assert_eq!(g.dedup_reject_frac, Some(0.6));
    assert_eq!(g.status.timeout, 1);
    assert_eq!(g.status.gate_passed, 3);
    assert_eq!(
        g.fitness_passers.max,
        Some(-0.8),
        "the sentinel 100.0 never enters quantiles"
    );
    // (a q=1)(h) -> (b)(a q=2)(h): insert (b), substitute (a q=1) -> (a q=2).
    assert_eq!(g.parent_child_hist[2], 1);
    assert_eq!(g.ops_distinct, 5, "a, h, b, c, h2");
    assert_eq!(g.ops_unused, Some(3));
    assert!(g.op_entropy_norm.unwrap() > 0.0 && g.op_entropy_norm.unwrap() < 1.0);
    // Three candidates have a usable fitness profile; two of them are identical.
    assert_eq!(g.behavior.n_with_behavior, 3);
    assert!((g.behavior.fitness_profile_unique_frac.unwrap() - 2.0 / 3.0).abs() < 1e-12);
    assert!(g.behavior.prediction_diversity.is_none());
    assert_eq!(
        g.behavior.prediction_availability.as_deref(),
        Some(NO_PREDICTION_SIGNATURES)
    );
    assert!(
        g.behavior.archive_knn_mean.is_none(),
        "empty archive on the first generation"
    );
    assert_eq!(g.edit_ops[&EditOp::EditInsertStage].success_rate, Some(0.0));
    assert_eq!(g.best_fitness, Some(-1.2));
    assert_eq!(g.gens_since_improved, 0);

    // A generation with no improvement on the best: stagnation counts up, novelty is now
    // measured against the archive filled by generation 0.
    let gen1 = vec![
        p.cand("(pipe X (a q=3) (h))", -1.0, true, vec![0.1, 0.5, 0.9, 0.2]),
        p.cand("(pipe X (a q=4) (h))", -0.9, true, vec![0.2, 0.4, 0.8, 0.3]),
    ];
    let g1 = t.summarize(0, 1, &gen1, ProposalCounters::default());
    assert_eq!(g1.best_fitness, Some(-1.2));
    assert_eq!(g1.gens_since_improved, 1);
    assert!(g1.behavior.archive_knn_mean.is_some());
    assert_eq!(
        g1.dedup_reject_frac, None,
        "no proposals reported, so no fraction"
    );
}

#[test]
fn collapsed_population_raises_a2_after_three_generations() {
    let mut p = Pool {
        clauses: Interner::default(),
        ops: Interner::default(),
    };
    let mut t = Tracker::new(
        "ctx",
        Config {
            op_vocab_size: Some(40),
            ..Default::default()
        },
    );
    let mut e = Engine::new("run", Thresholds::default());
    let mut fired = Vec::new();
    for g in 0..3 {
        let pop: Vec<Candidate> = (0..8)
            .map(|_| p.cand("(pipe X (a q=1) (h))", -1.0, true, vec![0.1, 0.2, 0.3, 0.4]))
            .collect();
        let s = t.summarize(
            0,
            g,
            &pop,
            ProposalCounters {
                proposals: 20,
                seen_rejects: 15,
            },
        );
        assert_eq!(s.unique_genotype_frac, Some(0.125));
        fired.push(e.observe_generation(&s, g as f64));
    }
    assert!(fired[0].is_none() && fired[1].is_none());
    let a = fired[2]
        .as_ref()
        .expect("third consecutive collapsed generation alarms");
    assert_eq!(a.id, AlarmId::A2GpCollapse);
    for needle in [
        "unique_genotype_frac",
        "fitness_profile_unique_frac",
        "op_entropy_norm",
        "dedup_reject_frac",
    ] {
        assert!(
            a.reasons.iter().any(|r| r.contains(needle)),
            "{needle} missing from {:?}",
            a.reasons
        );
    }
}
