/// Prefer stronger survivors. Policy weights apply to *ranked* genomes, never
/// to seed-population order: those are different programs after the first evals.
fn policy_weighted_parent(
    keep: usize,
    random_draw: u64,
    supplied: Option<&SeedPopulation>,
    ranked_genomes: Option<&[String]>,
) -> usize {
    if keep == 0 {
        return 0;
    }
    if let (Some(population), Some(genomes)) = (supplied, ranked_genomes) {
        let mut by_genome = std::collections::BTreeMap::<&str, f64>::new();
        for candidate in &population.candidates {
            if let Some(logprob) = candidate.policy_logprob {
                by_genome.insert(candidate.genome.as_str(), logprob);
            }
        }
        // Offspring often have no policy score. A missing score is not an opinion
        // of zero: zero is the TOP of the log-probability range and would outweigh
        // every scored survivor. Impute the minimum observed score instead, so an
        // unscored offspring stays eligible at the weight of the least likely
        // scored survivor and steering survives past the first evolved child that
        // enters the elite pool. A non-finite score is corruption rather than
        // absence and still collapses the whole pool to the uniform draw.
        let mut observed = Vec::with_capacity(keep);
        let mut corrupt = false;
        for genome in genomes.iter().take(keep) {
            match by_genome.get(genome.as_str()).copied() {
                Some(value) if value.is_finite() => observed.push(Some(value)),
                Some(_) => {
                    corrupt = true;
                    break;
                }
                None => observed.push(None),
            }
        }
        let floor = observed
            .iter()
            .flatten()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let scores = if corrupt || observed.len() != keep || !floor.is_finite() {
            None
        } else {
            Some(
                observed
                    .iter()
                    .map(|value| value.unwrap_or(floor))
                    .collect::<Vec<_>>(),
            )
        };
        if let Some(scores) = scores.filter(|scores| scores.len() == keep) {
            let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mut acc = 0.0;
            let mut weights = Vec::with_capacity(keep);
            for (i, score) in scores.iter().enumerate() {
                let w = (score - maximum).exp().max(1e-6);
                acc += w;
                weights.push((i, acc));
            }
            if acc > 0.0 {
                let pick = unit_quantile(random_draw) * acc;
                return weights
                    .iter()
                    .find(|(_, c)| pick < *c)
                    .map(|(i, _)| *i)
                    .unwrap_or(keep - 1);
            }
        }
    }
    // Missing policy scores, or scores that cannot be aligned to ranked
    // survivors, use the independently seeded uniform parent draw.
    (random_draw % keep as u64) as usize
}

#[cfg(test)]
mod parent_selection_tests {
    use super::{policy_weighted_parent, SeedCandidate, SeedPopulation};

    fn population(logprobs: &[Option<f64>]) -> SeedPopulation {
        SeedPopulation {
            schema: "t".into(),
            prepared_sha256: String::new(),
            cut: 0,
            training_sha256: String::new(),
            context_sha256: String::new(),
            checkpoint_sha256: String::new(),
            candidates: logprobs
                .iter()
                .map(|p| SeedCandidate {
                    genome: "g".into(),
                    origin: "neural_policy".into(),
                    policy_logprob: *p,
                    proposal_record_sha256: None,
                })
                .collect(),
        }
    }

    #[test]
    fn unmatched_seeds_keep_uniform_parent() {
        assert_eq!(policy_weighted_parent(10, 7, None, None), 7);
        let pop = population(&[None; 10]);
        assert_eq!(policy_weighted_parent(10, 7, Some(&pop), None), 7);
    }

    #[test]
    fn historical_and_failed_slots_never_invent_physical_starts() {
        use serde_json::json;
        let original = json!({"metrics":{"physical_fit_starts":1,"evaluation_origin":"execution"}});
        let reused = json!({"metrics":{"physical_fit_starts":0,"evaluation_origin":"reuse"}});
        let known = super::execution_accounting(&[original.clone(), reused.clone()]);
        assert_eq!(known["logical_slots"], 2);
        assert_eq!(known["physical_fit_starts"], 1);
        assert_eq!(known["reused_completions"], 1);
        let mixed = super::execution_accounting(&[
            original,
            reused,
            json!({"failure":"lost_worker_reply"}),
            json!({"metrics":{"rmse":0.}}),
        ]);
        assert_eq!(mixed["logical_slots"], 4);
        assert_eq!(mixed["recorded_physical_fit_starts"], 1);
        assert!(mixed["physical_fit_starts"].is_null());
        assert_eq!(mixed["unknown_start_outcomes"], 2);
    }

    #[test]
    fn rank_weights_do_not_emit_behavior_probability_claims() {
        let old = serde_json::json!({"genome":"(pipe X (ridge))","origin":"fixture","policy_logprob":-2.5});
        let candidate: SeedCandidate = serde_json::from_value(old).unwrap();
        let saved = serde_json::to_value(&candidate).unwrap();
        assert_eq!(saved["rank_log_weight"], -2.5);
        assert!(saved.get("policy_logprob").is_none());
        let restored: SeedCandidate = serde_json::from_value(saved).unwrap();
        assert_eq!(restored.policy_logprob, candidate.policy_logprob);
        let duplicate = serde_json::json!({"genome":"g","origin":"fixture","rank_log_weight":-1.,"policy_logprob":-2.});
        assert!(serde_json::from_value::<SeedCandidate>(duplicate).is_err());
    }

    #[test]
    fn policy_scores_weight_ranked_genomes_not_seed_order() {
        let mut pop = population(&[Some(10.0), Some(0.0), Some(0.0)]);
        pop.candidates[0].genome = "a".into();
        pop.candidates[1].genome = "b".into();
        pop.candidates[2].genome = "c".into();
        let aligned = vec!["a".into(), "b".into(), "c".into()];
        assert_eq!(
            policy_weighted_parent(3, u64::MAX / 2, Some(&pop), Some(&aligned)),
            0
        );
        // Ranked order is not seed order. Weight the ranked genomes.
        let ranked = vec!["b".into(), "c".into(), "a".into()];
        assert_eq!(
            policy_weighted_parent(3, u64::MAX / 2, Some(&pop), Some(&ranked)),
            2
        );
        // Scores without a ranked alignment must not index the wrong array.
        assert_eq!(policy_weighted_parent(3, 7, Some(&pop), None), 7 % 3);
    }

    #[test]
    fn nonfinite_scores_fall_back_for_the_whole_pool() {
        let genomes: Vec<String> = vec!["a".into(), "b".into(), "offspring".into()];
        let mut pop = population(&[Some(-50.), Some(-49.), None]);
        for (candidate, genome) in pop.candidates.iter_mut().zip(&genomes) {
            candidate.genome = genome.clone();
        }
        // Corruption, not absence: refuse to steer on it.
        for bad in [Some(f64::NAN), Some(f64::INFINITY), Some(f64::NEG_INFINITY)] {
            pop.candidates[2].policy_logprob = bad;
            for attempt in 0..1000 {
                let draw = super::random_word(8129661450841087504, 0, attempt, "left_parent");
                assert_eq!(
                    policy_weighted_parent(3, draw, Some(&pop), Some(&genomes)),
                    policy_weighted_parent(3, draw, None, None)
                );
            }
        }
    }

    #[test]
    fn unscored_offspring_keep_steering_alive_at_the_floor() {
        // Regression test for the defect that silently disabled policy steering
        // after the first evolved child entered the elite pool: one unscored
        // genome used to collapse the entire survivor pool to the uniform draw.
        let genomes: Vec<String> = vec!["a".into(), "b".into(), "offspring".into()];
        let mut pop = population(&[Some(-1.), Some(-9.), None]);
        for (candidate, genome) in pop.candidates.iter_mut().zip(&genomes) {
            candidate.genome = genome.clone();
        }
        pop.candidates[2].policy_logprob = None;
        let mut counts = [0usize; 3];
        let mut differs_from_uniform = 0;
        for attempt in 0..5000 {
            let draw = super::random_word(8129661450841087504, 0, attempt, "left_parent");
            let parent = policy_weighted_parent(3, draw, Some(&pop), Some(&genomes));
            counts[parent] += 1;
            if parent != policy_weighted_parent(3, draw, None, None) {
                differs_from_uniform += 1;
            }
        }
        // Steering is live: the pool is no longer the uniform draw.
        assert!(differs_from_uniform > 0, "steering collapsed to uniform");
        // The strongly scored survivor dominates the weakly scored one.
        assert!(counts[0] > counts[1], "policy score ignored: {counts:?}");
        // The unscored offspring is imputed at the floor, so it is reachable but
        // never preferred over the best scored survivor.
        assert!(
            counts[0] > counts[2],
            "offspring outweighed the best parent"
        );
    }

    #[test]
    fn seeded_draws_cover_parents_and_operators_independently() {
        let mut counts = [[0usize; 3]; 10];
        let mut quantiles = std::collections::BTreeSet::new();
        let mut pop = population(&[Some(1000.); 10]);
        let genomes: Vec<String> = (0..10).map(|i| format!("g{i}")).collect();
        for (candidate, genome) in pop.candidates.iter_mut().zip(&genomes) {
            candidate.genome = genome.clone();
        }
        for attempt in 0..10000 {
            let draw = super::random_word(8129661450841087504, 0, attempt, "left_parent");
            assert_eq!(
                draw,
                super::random_word(8129661450841087504, 0, attempt, "left_parent")
            );
            assert_ne!(
                draw,
                super::random_word(8129661450841087504, 0, attempt, "right_parent")
            );
            let quantile = super::unit_quantile(draw);
            assert!((0.0..1.0).contains(&quantile));
            quantiles.insert(quantile.to_bits());
            let parent = policy_weighted_parent(10, draw, Some(&pop), Some(&genomes));
            let operator = super::random_word(8129661450841087504, 0, attempt, "operator") % 16;
            counts[parent][if operator < 2 {
                0
            } else if operator < 9 {
                1
            } else {
                2
            }] += 1;
        }
        assert_eq!(quantiles.len(), 10000);
        assert_eq!(super::unit_quantile(0), 0.);
        assert!(super::unit_quantile(u64::MAX) < 1.);
        // Wide deterministic bounds detect the former constant quantile and
        // parent/operator parity coupling without a flaky stochastic test.
        for row in counts {
            assert!((850..1150).contains(&row.iter().sum::<usize>()));
            assert!((80..175).contains(&row[0]));
            assert!((330..550).contains(&row[1]));
            assert!((330..550).contains(&row[2]));
        }
    }

    #[test]
    fn stray_result_does_not_clear_worker_failure() {
        let mut failure = Some("worker_exit:1".into());
        assert!(!super::take_clean_result(&mut failure, true));
        assert_eq!(failure.as_deref(), Some("worker_exit:1"));
        let mut ok = None;
        assert!(super::take_clean_result(&mut ok, true));
        assert!(ok.is_none());
        let mut missing = None;
        assert!(!super::take_clean_result(&mut missing, false));
        assert_eq!(missing.as_deref(), Some("missing_result"));
    }

    #[test]
    fn logprobs_follow_ranking_scores_not_seed_order() {
        let mut candidates = vec![
            SeedCandidate {
                genome: "worse".into(),
                origin: "neural_policy".into(),
                policy_logprob: None,
                proposal_record_sha256: None,
            },
            SeedCandidate {
                genome: "best".into(),
                origin: "neural_policy".into(),
                policy_logprob: None,
                proposal_record_sha256: None,
            },
        ];
        super::attach_policy_logprobs(&mut candidates, &[0.1, 2.0]).unwrap();
        assert!(candidates[1].policy_logprob.unwrap() > candidates[0].policy_logprob.unwrap());
    }
}
