mod gpu_screen_tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    const PARITY_ABS_TOL: f64 = 5e-3;
    const TEST_EVALUATOR_RELEASE_SHA256: &str =
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    type Rung0TestRun = (
        Vec<Option<Vec<f64>>>,
        Rung0ExactSurrogateDedupReceipt,
        Vec<Option<Rung0ExactSurrogateCandidateReceipt>>,
        Vec<Vec<Rung0FoldRouteReceipt>>,
        GpuScreenPhaseTimings,
    );

    struct NullObs;
    impl Observer for NullObs {
        fn on_generation(&self, _stats: &GenStats) {}
        fn on_improvement(&self, _best: &Scored, _emitted: &Emitted, _stats: &GenStats) {}
    }

    struct XorShift(u64);
    impl XorShift {
        fn next_f64(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    fn synth(n: usize, seed: u64) -> (Rows, Vec<f64>) {
        let mut rng = XorShift(seed | 1);
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for _ in 0..n {
            let row: Vec<f64> = (0..6).map(|_| rng.next_f64() * 4.0 - 2.0).collect();
            let target = 0.8 * row[0] - 1.3 * row[1]
                + 0.5 * row[2] * row[3]
                + (row[4] * 1.7).sin()
                + 0.05 * (rng.next_f64() - 0.5);
            x.push(row);
            y.push(target);
        }
        (x, y)
    }

    fn candidates(count: usize, seed: u64) -> Vec<(String, DslExpr)> {
        let mut rng = Pcg32::new(seed, 0x5EED);
        let mut out = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for source in STARTERS {
            if let Ok(GenomeIr::Pipeline(genome)) = GenomeIr::parse(source) {
                let key = genome.to_string();
                if seen.insert(key.clone()) {
                    out.push((key, genome));
                }
            }
        }
        while out.len() < count {
            let genome = rand_genome(&mut rng, 4);
            let key = genome.to_string();
            if seen.insert(key.clone()) {
                out.push((key, genome));
            }
        }
        out
    }

    fn run_rung0(
        x: &Rows,
        y: &[f64],
        candidates: &[(String, DslExpr)],
        gpu: Option<&GpuRidgeScoreBatchFn<'_>>,
    ) -> Vec<Option<Vec<f64>>> {
        let policy = gpu.map(|_| GpuRidgeScorePolicy::new(1, 1).unwrap());
        run_rung0_with_policy(x, y, candidates, gpu, policy)
    }

    fn run_rung0_with_policy(
        x: &Rows,
        y: &[f64],
        candidates: &[(String, DslExpr)],
        gpu: Option<&GpuRidgeScoreBatchFn<'_>>,
        policy: Option<GpuRidgeScorePolicy>,
    ) -> Vec<Option<Vec<f64>>> {
        run_rung0_with_options(x, y, candidates, gpu, policy, None).0
    }

    fn run_rung0_with_options(
        x: &Rows,
        y: &[f64],
        candidates: &[(String, DslExpr)],
        gpu: Option<&GpuRidgeScoreBatchFn<'_>>,
        policy: Option<GpuRidgeScorePolicy>,
        dedup: Option<Rung0ExactSurrogateDedupPolicy>,
    ) -> Rung0TestRun {
        run_rung0_with_options_and_observer(x, y, candidates, gpu, policy, dedup, &NullObs)
    }

    fn run_rung0_with_options_and_observer(
        x: &Rows,
        y: &[f64],
        candidates: &[(String, DslExpr)],
        gpu: Option<&GpuRidgeScoreBatchFn<'_>>,
        policy: Option<GpuRidgeScorePolicy>,
        dedup: Option<Rung0ExactSurrogateDedupPolicy>,
        obs: &dyn Observer,
    ) -> Rung0TestRun {
        let n = x.len().min(y.len());
        let perm = permute_indices(n, 7);
        let rows0 = perm[..n.min(RACE_RUNG_ROWS[0])].to_vec();
        let assign0 = fold_assign(rows0.len(), RACE_RUNG_FOLDS[0], "invent_r0");
        let dedup_scope = dedup.map(|policy| {
            rung0_fold_scope_sha256(
                x,
                y,
                &rows0,
                &assign0,
                RACE_RUNG_FOLDS[0],
                &policy.evaluator_release_sha256,
            )
            .unwrap()
        });
        let metric = |actual: &[f64], predicted: &[f64]| crate::metrics::r2(actual, predicted);
        let ctx = RungCtx {
            x,
            y,
            metric: &metric,
            rows: [rows0.clone(), rows0.clone(), rows0],
            assigns: [assign0.clone(), assign0.clone(), assign0],
            caps: [f64::INFINITY; 3],
            t0: Instant::now(),
            overall_budget: 0.0,
            gpu_screen: gpu,
            gpu_screen_policy: policy,
            rung0_exact_surrogate_scope_sha256: dedup_scope,
            capture_cases: false,
        };
        let mut state = RaceState {
            memo: HashMap::new(),
            occurrences: HashMap::new(),
            spent: [0.0; 3],
            rung_evals: [0; 3],
            evals: 0,
            ledger: TokenLedger::new(),
            last_eval_tick: Instant::now(),
            gpu_screen_timings: GpuScreenPhaseTimings::default(),
            rung0_exact_surrogate_cache: HashMap::new(),
            rung0_exact_surrogate_dedup: Rung0ExactSurrogateDedupReceipt {
                requested: dedup.is_some(),
                active: dedup_scope.is_some(),
                authority_receipt_sha256: dedup
                    .map(|_| RUNG0_EXACT_SURROGATE_DEDUP_AUTHORITY_SHA256.to_string()),
                evaluator_release_sha256: dedup
                    .map(Rung0ExactSurrogateDedupPolicy::evaluator_release_sha256_hex),
                cache_scope: dedup.map(|_| "run_local".to_string()),
                wall_clock_completion_may_accelerate: dedup.is_some(),
                completed_candidate_score_bits_exact: dedup.is_some(),
                ..Rung0ExactSurrogateDedupReceipt::default()
            },
        };
        eval_rung(&ctx, &mut state, 0, 0, candidates, obs);
        let scores = candidates
            .iter()
            .map(|(key, _)| {
                state
                    .memo
                    .get(key)
                    .and_then(|entry| entry.rungs[0].clone())
                    .flatten()
            })
            .collect();
        let receipts = candidates
            .iter()
            .map(|(key, _)| {
                state
                    .memo
                    .get(key)
                    .and_then(|entry| entry.r0_exact_surrogate_dedup.clone())
            })
            .collect();
        let routes = candidates
            .iter()
            .map(|(key, _)| {
                state
                    .memo
                    .get(key)
                    .and_then(|entry| entry.r0_fold_routes.clone())
                    .unwrap()
            })
            .collect();
        (
            scores,
            state.rung0_exact_surrogate_dedup,
            receipts,
            routes,
            state.gpu_screen_timings,
        )
    }

    fn assert_same_shape(cpu: &[Option<Vec<f64>>], gpu: &[Option<Vec<f64>>]) {
        assert_eq!(cpu.len(), gpu.len());
        for (index, (cpu_scores, gpu_scores)) in cpu.iter().zip(gpu).enumerate() {
            assert_eq!(
                cpu_scores.is_some(),
                gpu_scores.is_some(),
                "candidate {index}: CPU/GPU scored-ness diverged"
            );
            if let (Some(cpu_scores), Some(gpu_scores)) = (cpu_scores, gpu_scores) {
                assert_eq!(
                    cpu_scores.len(),
                    gpu_scores.len(),
                    "candidate {index}: fold count diverged"
                );
            }
        }
    }

    fn exact_backend(problems: Vec<GpuRidgeScoreProblem>) -> Result<GpuRidgeScoreBatch, String> {
        let fold_bytes = vec![GpuRidgeScoreFoldBytes::default(); problems.len()];
        let scores = problems
            .into_iter()
            .map(|problem| {
                let mut rows: Rows = vec![vec![0.0; problem.p]; problem.train_n];
                // Column-major to row-major transpose. The index arithmetic
                // `column * train_n + row` is the operation being performed, not an
                // accidental way of walking a slice.
                #[allow(clippy::needless_range_loop)]
                for column in 0..problem.p {
                    for row in 0..problem.train_n {
                        rows[row][column] =
                            problem.train_x_cols[column * problem.train_n + row] as f64;
                    }
                }
                let target = problem
                    .train_y
                    .iter()
                    .map(|value| *value as f64)
                    .collect::<Vec<_>>();
                let model = RidgeModel::fit(&rows, &target, problem.lambda_frac);
                let validation = problem
                    .valid_x_rows
                    .chunks_exact(problem.p)
                    .map(|row| row.iter().map(|value| f64::from(*value)).collect())
                    .collect::<Rows>();
                crate::metrics::r2(&problem.valid_y, &model.predict(&validation))
            })
            .collect();
        Ok(GpuRidgeScoreBatch {
            scores,
            timings: GpuRidgeScoreTimings::default(),
            fold_bytes,
        })
    }

    #[test]
    fn gpu_screen_wave_seam_matches_cpu_path_with_exact_backend() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let (x, y) = synth(160, 0xFEED_CAFE);
        let candidates = candidates(192, 42);
        let calls = AtomicUsize::new(0);
        let backend = |problems: Vec<GpuRidgeScoreProblem>| {
            calls.fetch_add(1, Ordering::Relaxed);
            exact_backend(problems)
        };
        let cpu = run_rung0(&x, &y, &candidates, None);
        let gpu = run_rung0(&x, &y, &candidates, Some(&backend));
        assert!(calls.load(Ordering::Relaxed) > 0);
        assert_same_shape(&cpu, &gpu);
        let scored = cpu.iter().filter(|scores| scores.is_some()).count();
        assert!(scored * 2 >= candidates.len());
        for (index, (cpu_scores, gpu_scores)) in cpu.iter().zip(&gpu).enumerate() {
            if let (Some(cpu_scores), Some(gpu_scores)) = (cpu_scores, gpu_scores) {
                for (fold, (cpu_score, gpu_score)) in cpu_scores.iter().zip(gpu_scores).enumerate()
                {
                    let delta = (cpu_score - gpu_score).abs();
                    assert!(
                        delta <= PARITY_ABS_TOL,
                        "candidate {index} ({}) fold {fold}: CPU={cpu_score} backend={gpu_score} delta={delta}",
                        candidates[index].0
                    );
                }
            }
        }
    }

    #[test]
    fn gpu_screen_backend_error_falls_back_to_exact_cpu_path() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let (x, y) = synth(160, 0xBEEF_F00D);
        let candidates = candidates(96, 7);
        let calls = AtomicUsize::new(0);
        let backend = |_problems: Vec<GpuRidgeScoreProblem>| {
            calls.fetch_add(1, Ordering::Relaxed);
            Err("injected CUDA failure".to_string())
        };
        let cpu = run_rung0(&x, &y, &candidates, None);
        let fallback = run_rung0(&x, &y, &candidates, Some(&backend));
        assert!(calls.load(Ordering::Relaxed) > 0);
        assert_same_shape(&cpu, &fallback);
        for (index, (cpu_scores, fallback_scores)) in cpu.iter().zip(&fallback).enumerate() {
            if let (Some(cpu_scores), Some(fallback_scores)) = (cpu_scores, fallback_scores) {
                for (fold, (cpu_score, fallback_score)) in
                    cpu_scores.iter().zip(fallback_scores).enumerate()
                {
                    assert_eq!(
                        cpu_score.to_bits(),
                        fallback_score.to_bits(),
                        "candidate {index} fold {fold}: fallback changed CPU result"
                    );
                }
            }
        }
    }

    fn one_surrogate_candidates(count: usize) -> Vec<(String, DslExpr)> {
        (0..count)
            .map(|index| {
                let source = format!(
                    "(pipe X (standardize) (ridge lambda_frac={}))",
                    0.001 + index as f64 * 0.000_001
                );
                let GenomeIr::Pipeline(genome) = GenomeIr::parse(&source).unwrap() else {
                    unreachable!()
                };
                (genome.to_string(), genome)
            })
            .collect()
    }

    fn score_bits(scores: &[Option<Vec<f64>>]) -> Vec<Option<Vec<u64>>> {
        scores
            .iter()
            .map(|scores| {
                scores
                    .as_ref()
                    .map(|scores| scores.iter().map(|score| score.to_bits()).collect())
            })
            .collect()
    }

    #[test]
    fn exact_surrogate_dedup_is_bit_exact_ordered_and_cross_chunk_deterministic() {
        let (x, y) = synth(160, 0xDEC0_DED0);
        let candidates = one_surrogate_candidates(RACE_CHUNK * 2 + 1);
        let policy = Rung0ExactSurrogateDedupPolicy::from_authority_receipt(
            RUNG0_EXACT_SURROGATE_DEDUP_AUTHORITY_SHA256,
            TEST_EVALUATOR_RELEASE_SHA256,
        )
        .unwrap();

        let baseline = run_rung0_with_options(&x, &y, &candidates, None, None, None);
        let first = run_rung0_with_options(&x, &y, &candidates, None, None, Some(policy));
        let second = run_rung0_with_options(&x, &y, &candidates, None, None, Some(policy));

        assert_eq!(score_bits(&baseline.0), score_bits(&first.0));
        assert_eq!(score_bits(&first.0), score_bits(&second.0));
        assert_eq!(first.1, second.1);
        assert_eq!(first.2, second.2);
        assert!(!baseline.1.requested);
        assert_eq!(baseline.1.physical_candidate_evaluations, 0);
        assert!(first.1.requested && first.1.active);
        assert_eq!(
            first.1.evaluator_release_sha256.as_deref(),
            Some(TEST_EVALUATOR_RELEASE_SHA256)
        );
        assert_eq!(first.1.cache_scope.as_deref(), Some("run_local"));
        assert_eq!(first.1.logical_candidates, candidates.len() as u64);
        assert_eq!(first.1.physical_candidate_evaluations, 1);
        assert_eq!(first.1.unique_evaluation_signatures, 1);
        assert_eq!(first.1.reused_candidates, candidates.len() as u64 - 1);
        assert_eq!(first.1.rejected_chunks, 0);
        let physical_cpu_folds = first
            .3
            .iter()
            .flatten()
            .filter(|route| route.route == Rung0FoldRoute::CpuExact)
            .count();
        let reused_folds = first
            .3
            .iter()
            .flatten()
            .filter(|route| route.route == Rung0FoldRoute::ExactSurrogateReuse)
            .count();
        assert_eq!(physical_cpu_folds, RACE_RUNG_FOLDS[0]);
        assert_eq!(reused_folds, (candidates.len() - 1) * RACE_RUNG_FOLDS[0]);
        assert_eq!(first.4.cpu_folds as usize, physical_cpu_folds);
        assert_eq!(first.4.gpu_folds, 0);
        assert_eq!(
            first.1.physical_candidate_evaluations * RACE_RUNG_FOLDS[0] as u64,
            first.4.cpu_folds + first.4.gpu_folds
        );

        let representative = first.2[0].as_ref().unwrap();
        assert!(!representative.reused);
        assert_eq!(representative.representative_r0_seq, 0);
        for receipt in first.2.iter().skip(1) {
            let receipt = receipt.as_ref().unwrap();
            assert!(receipt.reused);
            assert_eq!(receipt.representative_r0_seq, 0);
            assert_eq!(
                receipt.evaluation_signature_sha256,
                representative.evaluation_signature_sha256
            );
        }

        let other_release = Rung0ExactSurrogateDedupPolicy::from_authority_receipt(
            RUNG0_EXACT_SURROGATE_DEDUP_AUTHORITY_SHA256,
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        )
        .unwrap();
        let other = run_rung0_with_options(&x, &y, &candidates, None, None, Some(other_release));
        assert_eq!(score_bits(&first.0), score_bits(&other.0));
        assert_ne!(
            first.2[0].as_ref().unwrap().evaluation_signature_sha256,
            other.2[0].as_ref().unwrap().evaluation_signature_sha256
        );
    }

    #[test]
    fn exact_surrogate_dedup_rejects_signature_collisions_and_partial_scatter() {
        let candidates = ["(pipe X (standardize) (ridge))", "(pipe X (rank) (ridge))"]
            .into_iter()
            .map(|source| {
                let GenomeIr::Pipeline(genome) = GenomeIr::parse(source).unwrap() else {
                    unreachable!()
                };
                (genome.to_string(), genome)
            })
            .collect::<Vec<_>>();
        let scope = [3_u8; 32];
        let cache = HashMap::new();
        let collision =
            build_rung0_exact_surrogate_plan_with(&candidates, &scope, &cache, |_, _| [9_u8; 32])
                .unwrap_err();
        assert_eq!(collision, Rung0ExactSurrogateReject::SignatureCollision);

        let duplicate_candidates = one_surrogate_candidates(2);
        let mut plan =
            build_rung0_exact_surrogate_plan(&duplicate_candidates, &scope, &cache).unwrap();
        plan.member_sources.pop();
        let partial = scatter_rung0_exact_surrogates(
            &plan,
            &RungChunkEvaluation {
                scores: vec![Some(vec![0.25, 0.5])],
                timings: GpuScreenPhaseTimings::default(),
                routes: vec![vec![
                    Rung0FoldRouteReceipt::cpu(Rung0FoldRoute::CpuExact);
                    RACE_RUNG_FOLDS[0]
                ]],
            },
            &cache,
            0,
            RACE_RUNG_FOLDS[0],
        )
        .err()
        .unwrap();
        assert_eq!(partial, Rung0ExactSurrogateReject::PartialMapping);
    }

    #[test]
    fn exact_surrogate_dedup_rejects_failed_representative_and_unknown_authority() {
        assert!(Rung0ExactSurrogateDedupPolicy::from_authority_receipt(
            &"0".repeat(64),
            TEST_EVALUATOR_RELEASE_SHA256,
        )
        .is_none());
        assert!(Rung0ExactSurrogateDedupPolicy::from_authority_receipt(
            RUNG0_EXACT_SURROGATE_DEDUP_AUTHORITY_SHA256,
            "",
        )
        .is_none());
        assert!(Rung0ExactSurrogateDedupPolicy::from_authority_receipt(
            RUNG0_EXACT_SURROGATE_DEDUP_AUTHORITY_SHA256,
            &"0".repeat(64),
        )
        .is_none());
        let candidates = one_surrogate_candidates(2);
        let scope = [4_u8; 32];
        let cache = HashMap::new();
        let plan = build_rung0_exact_surrogate_plan(&candidates, &scope, &cache).unwrap();
        let rejected = scatter_rung0_exact_surrogates(
            &plan,
            &RungChunkEvaluation {
                scores: vec![None],
                timings: GpuScreenPhaseTimings::default(),
                routes: vec![vec![
                    Rung0FoldRouteReceipt::cpu(Rung0FoldRoute::CpuExact);
                    RACE_RUNG_FOLDS[0]
                ]],
            },
            &cache,
            0,
            RACE_RUNG_FOLDS[0],
        )
        .err()
        .unwrap();
        assert_eq!(rejected, Rung0ExactSurrogateReject::RepresentativeFailure);
    }

    #[test]
    fn exact_surrogate_dedup_reduces_gpu_work_without_changing_score_bits() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let (x, y) = synth(160, 0xFA57_CADA);
        let candidates = one_surrogate_candidates(RACE_CHUNK * 2 + 1);
        let baseline_folds = AtomicUsize::new(0);
        let baseline_backend = |problems: Vec<GpuRidgeScoreProblem>| {
            baseline_folds.fetch_add(problems.len(), Ordering::Relaxed);
            exact_backend(problems)
        };
        let dedup_folds = AtomicUsize::new(0);
        let dedup_backend = |problems: Vec<GpuRidgeScoreProblem>| {
            dedup_folds.fetch_add(problems.len(), Ordering::Relaxed);
            exact_backend(problems)
        };
        let gpu_policy = GpuRidgeScorePolicy::new(1, 1);
        let dedup_policy = Rung0ExactSurrogateDedupPolicy::from_authority_receipt(
            RUNG0_EXACT_SURROGATE_DEDUP_AUTHORITY_SHA256,
            TEST_EVALUATOR_RELEASE_SHA256,
        )
        .unwrap();

        let baseline = run_rung0_with_options(
            &x,
            &y,
            &candidates,
            Some(&baseline_backend),
            gpu_policy,
            None,
        );
        let dedup = run_rung0_with_options(
            &x,
            &y,
            &candidates,
            Some(&dedup_backend),
            gpu_policy,
            Some(dedup_policy),
        );

        assert_eq!(score_bits(&baseline.0), score_bits(&dedup.0));
        assert_eq!(
            baseline_folds.load(Ordering::Relaxed),
            candidates.len() * RACE_RUNG_FOLDS[0]
        );
        assert_eq!(dedup_folds.load(Ordering::Relaxed), RACE_RUNG_FOLDS[0]);
        assert_eq!(dedup.1.physical_candidate_evaluations, 1);
        assert_eq!(dedup.1.reused_candidates, candidates.len() as u64 - 1);
        assert_eq!(dedup.4.gpu_folds, RACE_RUNG_FOLDS[0] as u64);
        assert_eq!(dedup.4.cpu_folds, 0);
        assert_eq!(
            dedup
                .3
                .iter()
                .flatten()
                .filter(|route| route.route == Rung0FoldRoute::Gpu)
                .count(),
            RACE_RUNG_FOLDS[0]
        );
        assert_eq!(
            dedup
                .3
                .iter()
                .flatten()
                .filter(|route| route.route == Rung0FoldRoute::ExactSurrogateReuse)
                .count(),
            (candidates.len() - 1) * RACE_RUNG_FOLDS[0]
        );
    }

    #[test]
    fn exact_surrogate_dedup_runs_bounded_and_preserves_common_completed_bits() {
        let (x, y) = synth(96, 0xB0_0DED);
        let seeds = one_surrogate_candidates(64);
        let config = EvolveConfig {
            pop: 64,
            gens: 1,
            time_budget_secs: 30.0,
            seed_genomes: seeds.iter().map(|(dsl, _)| dsl.clone()).collect(),
            fitness: FitnessMode::LiteRace,
            ..EvolveConfig::default()
        };
        let policy = Rung0ExactSurrogateDedupPolicy::from_authority_receipt(
            RUNG0_EXACT_SURROGATE_DEDUP_AUTHORITY_SHA256,
            TEST_EVALUATOR_RELEASE_SHA256,
        )
        .unwrap();
        let baseline = evolve_lite_race_with_gpu_policy(
            &x,
            &y,
            &crate::metrics::r2,
            &config,
            &NullObs,
            None,
            None,
        );
        let accelerated = evolve_lite_race_with_gpu_policy_and_exact_surrogate_dedup(
            &x,
            &y,
            &crate::metrics::r2,
            &config,
            &NullObs,
            None,
            None,
            Some(policy),
        );

        let completed_bits = |outcome: &RaceOutcome| {
            outcome
                .candidates
                .iter()
                .filter_map(|candidate| {
                    candidate.rung_fold_scores[0].as_ref().map(|scores| {
                        (
                            candidate.dsl.clone(),
                            scores
                                .iter()
                                .map(|score| score.to_bits())
                                .collect::<Vec<_>>(),
                        )
                    })
                })
                .collect::<BTreeMap<_, _>>()
        };
        let baseline_bits = completed_bits(&baseline);
        let accelerated_bits = completed_bits(&accelerated);
        let mut common = 0;
        for (dsl, bits) in &baseline_bits {
            if let Some(accelerated) = accelerated_bits.get(dsl) {
                assert_eq!(bits, accelerated, "bounded common candidate {dsl}");
                common += 1;
            }
        }

        assert!(common > 0);
        assert!(accelerated_bits.len() >= baseline_bits.len());
        assert!(accelerated.rung0_exact_surrogate_dedup.requested);
        assert!(accelerated.rung0_exact_surrogate_dedup.active);
        assert!(
            accelerated
                .rung0_exact_surrogate_dedup
                .wall_clock_completion_may_accelerate
        );
        assert!(
            accelerated
                .rung0_exact_surrogate_dedup
                .completed_candidate_score_bits_exact
        );
        assert!(accelerated.rung0_exact_surrogate_dedup.reused_candidates >= 1);
        assert!(
            accelerated
                .rung0_exact_surrogate_dedup
                .physical_candidate_evaluations
                < accelerated.rung0_exact_surrogate_dedup.logical_candidates
        );
        assert!(accelerated
            .rung0_exact_surrogate_dedup
            .activation_blocker
            .is_none());
    }

    struct CancelAfterRepresentative {
        checks: AtomicUsize,
    }

    impl Observer for CancelAfterRepresentative {
        fn on_generation(&self, _stats: &GenStats) {}
        fn on_improvement(&self, _best: &Scored, _emitted: &Emitted, _stats: &GenStats) {}
        fn is_cancelled(&self) -> bool {
            self.checks.fetch_add(1, Ordering::Relaxed) + 1 >= 6
        }
    }

    #[test]
    fn exact_surrogate_dedup_never_scatters_or_falls_back_after_cancellation() {
        let (x, y) = synth(160, 0x0CA1_1CE1);
        let candidates = one_surrogate_candidates(2);
        let policy = Rung0ExactSurrogateDedupPolicy::from_authority_receipt(
            RUNG0_EXACT_SURROGATE_DEDUP_AUTHORITY_SHA256,
            TEST_EVALUATOR_RELEASE_SHA256,
        )
        .unwrap();
        let observer = CancelAfterRepresentative {
            checks: AtomicUsize::new(0),
        };
        let result = run_rung0_with_options_and_observer(
            &x,
            &y,
            &candidates,
            None,
            None,
            Some(policy),
            &observer,
        );

        assert!(result.0.iter().all(Option::is_none));
        assert_eq!(result.1.physical_candidate_evaluations, 1);
        assert_eq!(result.1.reused_candidates, 0);
        assert_eq!(result.1.cancellation_rejections, 1);
        assert_eq!(result.1.rejected_chunks, 1);
        assert!(result.2.iter().all(Option::is_none));
    }

    #[test]
    fn rung0_trace_enrichment_replays_target_candidate_bit_exactly() {
        let (x, y) = synth(160, 0xA11C_E55E);
        let source = "(pipe X (target kind=rank_gauss) (standardize) (histgbm depth=4))";
        let GenomeIr::Pipeline(genome) = GenomeIr::parse(source).unwrap() else {
            unreachable!()
        };
        let candidates = vec![(genome.to_string(), genome)];
        let observed = run_rung0(&x, &y, &candidates, None)
            .pop()
            .flatten()
            .unwrap();
        let rows = permute_indices(x.len(), 7)[..x.len().min(RACE_RUNG_ROWS[0])].to_vec();
        let assignments = fold_assign(rows.len(), RACE_RUNG_FOLDS[0], "invent_r0");
        let trace =
            enrich_rung0_trace_candidate(&candidates[0].0, &x, &y, &rows, &assignments).unwrap();

        assert!(trace.surrogate_dsl.ends_with("(ridge))"));
        assert_eq!(trace.transform_prefixes.len(), 2);
        assert_eq!(trace.folds.len(), RACE_RUNG_FOLDS[0]);
        for (actual, replay) in observed.iter().zip(&trace.folds) {
            assert_eq!(actual.to_bits(), replay.cpu_replay_score.to_bits());
            assert!(replay.prepared_width > 0);
            assert!(replay.host_f64_bytes > replay.host_f32_bytes);
        }
    }

    #[test]
    fn rung0_trace_sequence_and_cpu_routes_are_dense() {
        let (x, y) = synth(96, 0x51A7_E001);
        let cfg = EvolveConfig {
            pop: 8,
            gens: 1,
            seed: 0x51A7_E001,
            fitness: FitnessMode::LiteRace,
            ..EvolveConfig::default()
        };
        let outcome = evolve_lite_race_with_gpu_policy(
            &x,
            &y,
            &crate::metrics::r2,
            &cfg,
            &NullObs,
            None,
            None,
        );
        let mut traced = outcome
            .candidates
            .iter()
            .filter_map(|candidate| candidate.r0_seq.map(|seq| (seq, candidate)))
            .collect::<Vec<_>>();
        traced.sort_by_key(|(seq, _)| *seq);
        assert_eq!(traced.len() as u64, outcome.rung_evals[0]);
        for (expected, (seq, candidate)) in traced.iter().enumerate() {
            assert_eq!(*seq, expected as u64);
            assert_eq!(candidate.r0_generation, Some(0));
            assert_eq!(candidate.r0_generation_slot, Some(expected));
            assert!(candidate
                .r0_fold_routes
                .as_ref()
                .unwrap()
                .iter()
                .all(|receipt| {
                    receipt.route == Rung0FoldRoute::CpuExact
                        && receipt.physical_h2d_bytes == 0
                        && receipt.physical_d2h_bytes == 0
                }));
        }
        assert_eq!(outcome.rung0_rows.len(), outcome.rung0_assignments.len());
        assert_ne!(
            outcome.rung0_rows,
            (0..outcome.rung0_rows.len()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn gpu_screen_requires_explicit_policy_and_routes_only_eligible_widths() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Mutex;

        let (x, y) = synth(160, 0xABCD_1234);
        let candidates = [
            "(pipe X (standardize) (ridge))",
            "(pipe X (poly_interact append=true) (ridge))",
            "(pipe X (target kind=rank_gauss) (standardize) (ridge))",
        ]
        .into_iter()
        .map(|source| {
            let GenomeIr::Pipeline(genome) = GenomeIr::parse(source).unwrap() else {
                unreachable!()
            };
            (genome.to_string(), genome)
        })
        .collect::<Vec<_>>();
        let calls = AtomicUsize::new(0);
        let seen_widths = Mutex::new(Vec::new());
        let backend = |problems: Vec<GpuRidgeScoreProblem>| {
            calls.fetch_add(1, Ordering::Relaxed);
            seen_widths
                .lock()
                .unwrap()
                .extend(problems.iter().map(|problem| problem.p));
            exact_backend(problems)
        };

        let cpu = run_rung0(&x, &y, &candidates, None);
        let fail_closed = run_rung0_with_policy(&x, &y, &candidates, Some(&backend), None);
        assert_eq!(calls.load(Ordering::Relaxed), 0);
        for (cpu_scores, fail_closed_scores) in cpu.iter().zip(&fail_closed) {
            assert_eq!(
                cpu_scores.as_ref().map(|scores| scores
                    .iter()
                    .map(|score| score.to_bits())
                    .collect::<Vec<_>>()),
                fail_closed_scores.as_ref().map(|scores| scores
                    .iter()
                    .map(|score| score.to_bits())
                    .collect::<Vec<_>>())
            );
        }

        let below_batch = run_rung0_with_policy(
            &x,
            &y,
            &candidates,
            Some(&backend),
            GpuRidgeScorePolicy::new(7, 3),
        );
        assert_eq!(calls.load(Ordering::Relaxed), 0);
        for (cpu_scores, below_batch_scores) in cpu.iter().zip(&below_batch) {
            assert_eq!(
                cpu_scores.as_ref().map(|scores| scores
                    .iter()
                    .map(|score| score.to_bits())
                    .collect::<Vec<_>>()),
                below_batch_scores.as_ref().map(|scores| scores
                    .iter()
                    .map(|score| score.to_bits())
                    .collect::<Vec<_>>())
            );
        }

        let policy = GpuRidgeScorePolicy::new(7, 2).unwrap();
        let hybrid = run_rung0_with_policy(&x, &y, &candidates, Some(&backend), Some(policy));
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        let widths = seen_widths.lock().unwrap();
        assert_eq!(widths.len(), RACE_RUNG_FOLDS[0]);
        assert!(widths.iter().all(|width| *width >= policy.min_p));
        drop(widths);
        assert_same_shape(&cpu, &hybrid);
        for (cpu_scores, hybrid_scores) in cpu.iter().zip(&hybrid) {
            if let (Some(cpu_scores), Some(hybrid_scores)) = (cpu_scores, hybrid_scores) {
                for (cpu_score, hybrid_score) in cpu_scores.iter().zip(hybrid_scores) {
                    assert!((cpu_score - hybrid_score).abs() <= PARITY_ABS_TOL);
                }
            }
        }
    }
}
