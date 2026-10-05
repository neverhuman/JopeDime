//! B11 certified ridge scoring tests: bit identity with `Refit`, the surrogate memo and its
//! exact keys, thread independence and refusals. A child module of `evolve`, so it reaches the
//! scorer's private items.

use super::*;
use crate::kernel::lexicase::{CaseSampling, EpsilonRule};
use std::sync::Mutex;

#[derive(Default)]
struct Log {
    scoring: Mutex<Vec<ScoringReceipt>>,
    selection: Mutex<Vec<SelectionReceipt>>,
}

impl Observer for Log {
    fn on_generation(&self, _stats: &GenStats) {}
    fn on_improvement(&self, _best: &Scored, _emitted: &Emitted, _stats: &GenStats) {}
    fn on_selection(&self, receipt: &SelectionReceipt) {
        self.selection.lock().unwrap().push(receipt.clone());
    }
    fn on_scoring(&self, receipt: &ScoringReceipt) {
        self.scoring.lock().unwrap().push(receipt.clone());
    }
}

fn fixture(seed: u64, n: usize) -> (Rows, Vec<f64>) {
    let mut rng = Pcg32::new(seed, 21);
    let x: Rows = (0..n)
        .map(|_| (0..4).map(|_| rng.gaussian()).collect())
        .collect();
    let y = x
        .iter()
        .map(|r| 2.0 * r[0] - r[1] * r[1] + (2.0 * r[2]).sin() + 0.1 * rng.gaussian())
        .collect();
    (x, y)
}

fn r2(truth: &[f64], predicted: &[f64]) -> f64 {
    crate::metrics::r2(truth, predicted)
}

fn dump(ranked: &[Scored]) -> Vec<(String, u64, Vec<u64>)> {
    ranked
        .iter()
        .map(|s| {
            let folds = s.fold_scores.iter().map(|v| v.to_bits()).collect();
            (s.dsl.clone(), s.cv_mean.to_bits(), folds)
        })
        .collect()
}

fn run(scoring: RidgeScoring, selection: ParentSelection) -> (Vec<Scored>, Log) {
    let (x, y) = fixture(3, 180);
    let cfg = EvolveConfig {
        pop: 16,
        gens: 4,
        seed: 0xB11,
        max_transforms: 4,
        selection,
        ridge_scoring: scoring,
        ..EvolveConfig::default()
    };
    let log = Log::default();
    let ranked = evolve(&x, &y, &r2, &cfg, &log);
    (ranked, log)
}

#[test]
fn certified_scoring_is_the_legacy_search_bit_for_bit() {
    let lexicase = ParentSelection::EpsilonLexicase(LexicaseConfig {
        epsilon: EpsilonRule::MedianAbsoluteDeviation,
        sampling: CaseSampling::All,
    });
    for selection in [ParentSelection::Tournament3, lexicase] {
        let (legacy, legacy_log) = run(RidgeScoring::Refit, selection);
        let (certified, certified_log) = run(RidgeScoring::Certified, selection);
        assert_eq!(dump(&legacy), dump(&certified), "{selection:?}");
        // Lexicase cases from the memo are the legacy residuals, so selection is unchanged.
        assert_eq!(
            *legacy_log.selection.lock().unwrap(),
            *certified_log.selection.lock().unwrap()
        );
        assert!(legacy_log.scoring.lock().unwrap().is_empty());
    }
}

#[test]
fn certified_scoring_reports_reuse_and_stage_cache_counters() {
    let (_, log) = run(RidgeScoring::Certified, ParentSelection::Tournament3);
    let receipts = log.scoring.into_inner().unwrap();
    assert_eq!(receipts.len(), 4);
    for (generation, pair) in receipts.windows(2).enumerate() {
        assert_eq!(pair[0].generation, generation);
        assert!(pair[1].candidates >= pair[0].candidates);
        assert!(pair[1].surrogate_reuses >= pair[0].surrogate_reuses);
    }
    let last = receipts.last().unwrap();
    assert!(last.surrogate_reuses <= last.candidates);
    // Head-only variation and shared prefixes are common in a short search.
    assert!(last.surrogate_reuses > 0, "{last:?}");
    assert!(last.stage_hits > 0, "{last:?}");
}

#[test]
fn a_different_genome_with_the_same_surrogate_reuses_its_scores_and_cases() {
    let (x, y) = fixture(5, 120);
    let k = 3;
    let assign: Vec<usize> = (0..y.len()).map(|i| i % k).collect();
    let parse = |text: &str| match GenomeIr::parse(text).unwrap() {
        GenomeIr::Pipeline(genome) => genome,
        other => panic!("{other:?}"),
    };
    // Different heads, so different genome keys, but one ridge surrogate.
    let a = parse("(pipe X (standardize) (poly_interact append=true) (gbm))");
    let b = parse("(pipe X (standardize) (poly_interact append=true) (elasticnet))");
    assert_ne!(a.to_string(), b.to_string());
    assert_eq!(
        search_surrogate(&a).to_string(),
        search_surrogate(&b).to_string()
    );
    let rows = lexicase_case_rows(y.len(), 7);
    let mut scoring = CertifiedScoring::new();
    let mut cases = HashMap::new();
    let first = scoring.score_unseen(
        vec![(a.to_string(), a.clone())],
        &x,
        &y,
        &assign,
        k,
        &r2,
        Some(&rows),
        &mut cases,
    );
    let second = scoring.score_unseen(
        vec![(b.to_string(), b.clone())],
        &x,
        &y,
        &assign,
        k,
        &r2,
        Some(&rows),
        &mut cases,
    );
    assert_eq!(
        (scoring.candidates, scoring.reuses, scoring.memo.len()),
        (2, 1, 1)
    );
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    let (legacy_scores, legacy_cases) =
        score_genome_cases(&b, &x, &y, &assign, k, &r2, &rows).unwrap();
    assert_eq!(bits(second[0].2.as_ref().unwrap()), bits(&legacy_scores));
    assert_eq!(bits(first[0].2.as_ref().unwrap()), bits(&legacy_scores));
    assert_eq!(bits(&cases[&b.to_string()]), bits(&legacy_cases));
    assert_eq!(bits(&cases[&a.to_string()]), bits(&legacy_cases));
}

#[test]
fn surrogates_that_print_alike_get_distinct_entries_and_their_own_scores() {
    let (x, y) = fixture(5, 120);
    let k = 3;
    let assign: Vec<usize> = (0..y.len()).map(|i| i % k).collect();
    let parse = |text: &str| match GenomeIr::parse(text).unwrap() {
        GenomeIr::Pipeline(genome) => genome,
        other => panic!("{other:?}"),
    };
    // Different heads, so the search's genome memo keeps them apart in both modes. Their
    // surrogates differ only in gamma, 1.0 against 1.0 + 5e-13, which prints as 1.0: the same
    // canonical text for two different computations.
    let a = parse("(pipe X (standardize) (rff_cos gamma=1.0 n=8 seed=3 append=true) (gbm))");
    let mut b =
        parse("(pipe X (standardize) (rff_cos gamma=1.0 n=8 seed=3 append=true) (elasticnet))");
    b.stages[1]
        .params
        .insert("gamma".to_string(), ParamValue::Float(1.0 + 5e-13));
    assert_ne!(1.0f64.to_bits(), (1.0f64 + 5e-13).to_bits());
    assert_ne!(a.to_string(), b.to_string());
    let (sa, sb) = (search_surrogate(&a), search_surrogate(&b));
    assert_eq!(sa.to_string(), sb.to_string());
    assert_ne!(exact_surrogate_key(&sa), exact_surrogate_key(&sb));
    let rows = lexicase_case_rows(y.len(), 7);
    let mut scoring = CertifiedScoring::new();
    let mut cases = HashMap::new();
    let mut score = |g: &DslExpr| {
        let scored = scoring.score_unseen(
            vec![(g.to_string(), g.clone())],
            &x,
            &y,
            &assign,
            k,
            &r2,
            Some(&rows),
            &mut cases,
        );
        let (_, misses, _) = scoring.cache.stats();
        (scored[0].2.clone().unwrap(), misses)
    };
    let (first, misses_a) = score(&a);
    let (second, misses_b) = score(&b);
    // Two memo entries and no reuse; the shared standardize stage hits on every fold, while
    // b's rff_cos block is computed afresh on every fold instead of reusing a's.
    assert_eq!(
        (scoring.candidates, scoring.reuses, scoring.memo.len()),
        (2, 0, 2)
    );
    assert!(misses_b >= misses_a + k as u64, "{misses_a} -> {misses_b}");
    // Each candidate gets exactly the score and cases Refit computes for it.
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    for (genome, scores) in [(&a, &first), (&b, &second)] {
        let (refit_scores, refit_cases) =
            score_genome_cases(genome, &x, &y, &assign, k, &r2, &rows).unwrap();
        assert_eq!(bits(scores), bits(&refit_scores));
        assert_eq!(bits(&cases[&genome.to_string()]), bits(&refit_cases));
    }
}

#[test]
fn print_alike_surrogates_that_run_different_programs_are_scored_apart_in_one_batch() {
    let (x, y) = fixture(5, 120);
    let k = 3;
    let assign: Vec<usize> = (0..y.len()).map(|i| i % k).collect();
    let parse = |text: &str| match GenomeIr::parse(text).unwrap() {
        GenomeIr::Pipeline(genome) => genome,
        other => panic!("{other:?}"),
    };
    // `k` = 2.0 and 2.0 - 5e-13 both print as 2.0, but `lag` truncates `k` to an integer, so
    // the second surrogate runs lag 1, not lag 2: a different program with a different Refit
    // score. A text key would hand it the first one's score.
    let a = parse("(pipe X (lag k=2.0) (gbm))");
    let mut b = parse("(pipe X (lag k=2.0) (elasticnet))");
    b.stages[0]
        .params
        .insert("k".to_string(), ParamValue::Float(2.0 - 5e-13));
    assert_eq!(
        search_surrogate(&a).to_string(),
        search_surrogate(&b).to_string()
    );
    let rows = lexicase_case_rows(y.len(), 7);
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    let refit = |g: &DslExpr| score_genome_cases(g, &x, &y, &assign, k, &r2, &rows).unwrap();
    let ((refit_a, _), (refit_b, _)) = (refit(&a), refit(&b));
    assert_ne!(
        bits(&refit_a),
        bits(&refit_b),
        "lag 2 and lag 1 must score apart"
    );
    let mut scoring = CertifiedScoring::new();
    let mut cases = HashMap::new();
    let scored = scoring.score_unseen(
        vec![(a.to_string(), a.clone()), (b.to_string(), b.clone())],
        &x,
        &y,
        &assign,
        k,
        &r2,
        Some(&rows),
        &mut cases,
    );
    assert_eq!(
        (scoring.candidates, scoring.reuses, scoring.memo.len()),
        (2, 0, 2)
    );
    assert_eq!(bits(scored[0].2.as_ref().unwrap()), bits(&refit_a));
    assert_eq!(bits(scored[1].2.as_ref().unwrap()), bits(&refit_b));
}

#[test]
fn certified_search_scores_print_alike_seeds_as_refit_does() {
    // The independent review's probe: both lag stages print as (lag k=2.0), but lag reads k by
    // truncation, so the first seed lags by one column and the second by two. Their whole
    // surrogates differ, so only the stage cache can confuse them; it keys on exact bits.
    let seeds = [
        "(pipe X (lag k=1.9999999999995) (standardize) (ridge))",
        "(pipe X (lag k=2.0) (sigmoid) (ridge))",
    ];
    let (x, y) = fixture(3, 180);
    let run = |scoring: RidgeScoring, threads: usize| {
        let cfg = EvolveConfig {
            pop: 3,
            gens: 1,
            seed: 0xB11,
            max_transforms: 3,
            seed_genomes: seeds.iter().map(|seed| seed.to_string()).collect(),
            ridge_scoring: scoring,
            ..EvolveConfig::default()
        };
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        dump(&pool.install(|| evolve(&x, &y, &r2, &cfg, &Log::default())))
    };
    let refit = run(RidgeScoring::Refit, 1);
    // Both seeds are scored and ranked, each under its canonical text.
    for seed in seeds {
        let canonical = GenomeIr::parse(seed).unwrap().canonical();
        assert!(
            refit.iter().any(|(dsl, _, _)| *dsl == canonical),
            "{canonical}"
        );
    }
    assert_eq!(run(RidgeScoring::Certified, 1), refit);
    for _ in 0..3 {
        assert_eq!(run(RidgeScoring::Certified, 4), refit);
    }
}

#[test]
fn certified_scoring_is_the_same_on_one_thread_and_on_four() {
    let pool = |threads| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
    };
    let certified = || run(RidgeScoring::Certified, ParentSelection::Tournament3);
    let (one, one_log) = pool(1).install(certified);
    let (four, four_log) = pool(4).install(certified);
    let (legacy, _) = pool(4).install(|| run(RidgeScoring::Refit, ParentSelection::Tournament3));
    assert_eq!(dump(&one), dump(&four));
    assert_eq!(dump(&four), dump(&legacy));
    // The memo counters follow the candidate stream alone. The stage-cache counters are
    // instrumentation: two threads building the same prefix both count a miss.
    let memo = |log: Log| {
        log.scoring
            .into_inner()
            .unwrap()
            .iter()
            .map(|r| (r.generation, r.candidates, r.surrogate_reuses))
            .collect::<Vec<_>>()
    };
    assert_eq!(memo(one_log), memo(four_log));
}

#[test]
fn certified_scoring_refuses_the_race_and_strict_policies() {
    let certified = RidgeScoring::Certified;
    assert!(certified
        .validate(FitnessMode::Surrogate, FaultPolicy::Legacy)
        .is_ok());
    assert!(certified
        .validate(FitnessMode::LiteRace, FaultPolicy::Legacy)
        .unwrap_err()
        .contains("surrogate search"));
    assert!(certified
        .validate(FitnessMode::Surrogate, FaultPolicy::RejectHard)
        .unwrap_err()
        .contains("default fault policy"));
    assert!(RidgeScoring::Refit
        .validate(FitnessMode::LiteRace, FaultPolicy::RejectHard)
        .is_ok());
    let (x, y) = fixture(4, 60);
    let cfg = EvolveConfig {
        pop: 4,
        fitness: FitnessMode::LiteRace,
        ridge_scoring: certified,
        ..EvolveConfig::default()
    };
    let refused =
        evolve_lite_race_with_exact_initial_population(&x, &y, &r2, &cfg, &Log::default(), &[]);
    assert!(refused.unwrap_err().contains("surrogate search"));
}

#[test]
#[should_panic(expected = "certified scoring needs the default fault policy")]
fn the_surrogate_search_panics_on_certified_scoring_under_a_strict_policy() {
    let (x, y) = fixture(4, 60);
    let cfg = EvolveConfig {
        pop: 4,
        gens: 1,
        fault_policy: FaultPolicy::RejectHard,
        ridge_scoring: RidgeScoring::Certified,
        ..EvolveConfig::default()
    };
    evolve(&x, &y, &r2, &cfg, &Log::default());
}

#[test]
#[should_panic(expected = "certified scoring applies to the surrogate search")]
fn the_race_panics_on_certified_scoring() {
    let (x, y) = fixture(4, 60);
    let cfg = EvolveConfig {
        pop: 4,
        gens: 1,
        fitness: FitnessMode::LiteRace,
        ridge_scoring: RidgeScoring::Certified,
        ..EvolveConfig::default()
    };
    evolve(&x, &y, &r2, &cfg, &Log::default());
}
