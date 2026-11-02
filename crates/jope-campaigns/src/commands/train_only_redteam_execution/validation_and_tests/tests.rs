#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(id: char, loss: f64, complexity: usize) -> RankedCandidate {
        RankedCandidate {
            candidate: Candidate {
                canonical_genome: "(pipe X (ridge))".to_string(),
                semantic_sha256: id.to_string().repeat(64),
                complexity,
                head_family: "ridge".to_string(),
                roots: BTreeSet::from(["dl".to_string()]),
                proposal_origins: BTreeSet::new(),
                decoder_seeds: BTreeSet::new(),
                ancestry: Value::Null,
            },
            evaluation_path: PathBuf::from("unused"),
            evaluation_sha256: "0".repeat(64),
            robust_loss: loss,
            worst_fold_nrmse: loss,
        }
    }

    fn initial_candidate(island: &str, slot: usize, head_family: &str) -> Candidate {
        Candidate {
            canonical_genome: format!("(pipe {island} {slot} ({head_family}))"),
            semantic_sha256: format!("{island}-{slot}"),
            complexity: 1,
            head_family: head_family.to_string(),
            roots: BTreeSet::from([island.to_string()]),
            proposal_origins: BTreeSet::new(),
            decoder_seeds: BTreeSet::new(),
            ancestry: Value::Null,
        }
    }

    #[test]
    fn mutation_arm_frequency_contract_is_exact_over_full_cycles() {
        let mut counts = BTreeMap::new();
        for draw in 0..10_000 {
            *counts.entry(mutation_arm(draw).0).or_insert(0usize) += 1;
        }
        assert_eq!(counts["parameter_jitter"], 3_000);
        assert_eq!(counts["insertion"], 1_000);
        assert_eq!(counts["deletion"], 2_000);
        assert_eq!(counts["adjacent_swap"], 1_000);
        assert_eq!(counts["head_replacement"], 1_500);
        assert_eq!(counts["simplification"], 1_500);
    }

    #[test]
    fn nsga_order_preserves_loss_complexity_nondomination() {
        let population = vec![
            candidate('a', 0.10, 8),
            candidate('b', 0.11, 2),
            candidate('c', 0.12, 8),
            candidate('d', 0.13, 9),
        ];
        let order = nsga_order(&population);
        let first_front = &order[..2];
        assert!(first_front.contains(&0));
        assert!(first_front.contains(&1));
        assert!(
            order.iter().position(|index| *index == 2).unwrap()
                < order.iter().position(|index| *index == 3).unwrap()
        );
    }

    #[test]
    fn convex_alpha_is_bounded_and_improves_the_fit() {
        let truth = [0.0, 1.0, 2.0, 3.0];
        let current = [0.0, 0.5, 1.0, 1.5];
        let candidate = [0.0, 1.0, 2.0, 3.0];
        let alpha = optimal_convex_alpha(&truth, &current, &candidate).unwrap();
        assert_eq!(alpha, 1.0);
        let blended = add_prediction(&current, &candidate, alpha);
        assert_eq!(blended, truth);
    }

    #[test]
    fn smart_only_selector_freezes_without_a_g20_receipt() {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path();
        let dataset_hash = DATASETS[0];
        let dataset_root = root.join("datasets").join(dataset_hash);
        fs::create_dir_all(&dataset_root).unwrap();
        let train_path = root.join("train.csv");
        let csv = (0..30)
            .map(|row| format!("{},{}", row as f64, 2.0 * row as f64 + 1.0))
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(&train_path, format!("{csv}\n")).unwrap();
        let train_sha256 = sha256_file(&train_path).unwrap();
        let data = read_regression_csv(&train_path).unwrap();
        let config = TrainOnlyRedteamConfig::sealed(SMART_V3_CPU_RUN_ID, DEFAULT_PARENT_RUN, false);
        let bank = build_fold_bank(
            SMART_V3_CPU_RUN_ID,
            dataset_hash,
            &train_sha256,
            &data.y,
            &config,
        )
        .unwrap();
        let fold_bank_path = dataset_root.join("fold-banks.v1.json");
        let fold_bank_sha256 = write_receipt(&fold_bank_path, &bank).unwrap();
        let canonical_genome = "(pipe X (ridge))".to_string();
        let selected = Candidate {
            semantic_sha256: semantic_identity(&canonical_genome),
            canonical_genome,
            complexity: 1,
            head_family: "ridge".to_string(),
            roots: BTreeSet::from(["dl".to_string()]),
            proposal_origins: BTreeSet::new(),
            decoder_seeds: BTreeSet::new(),
            ancestry: Value::Null,
        };
        let ranked = RankedCandidate {
            candidate: selected,
            evaluation_path: PathBuf::from("unused"),
            evaluation_sha256: "0".repeat(64),
            robust_loss: 1.0,
            worst_fold_nrmse: 1.0,
        };
        let smart = SmartComplete {
            schema: "TrainOnlyRedteamSmartComplete.v1".to_string(),
            run_id: SMART_V3_CPU_RUN_ID.to_string(),
            dataset_hash: dataset_hash.to_string(),
            generations_completed: 10,
            stopped_early: false,
            screening_sha256: "1".repeat(64),
            generation_sha256: vec!["2".repeat(64); 10],
            archive_top20_by_island: ["dl", "transfer", "mixed"]
                .into_iter()
                .map(|island| (island.to_string(), vec![ranked.clone()]))
                .collect(),
            mutation_counts: BTreeMap::new(),
            development_only: true,
            confirmatory_valid: false,
        };
        write_receipt(&dataset_root.join("smart-v3-complete.v1.json"), &smart).unwrap();
        assert!(!dataset_root.join("g20-extension-complete.v1.json").exists());
        let dataset = DatasetInput {
            dataset_hash: dataset_hash.to_string(),
            train_path,
            train_sha256,
            rows: data.y.len(),
            features: data.features,
            context_sha256: "3".repeat(64),
            transfer_ranking_path: root.join("unused-transfer.json"),
            transfer_ranking_sha256: "4".repeat(64),
            parent_generation_paths: Vec::new(),
            parent_generation_sha256: Vec::new(),
            fold_bank_path,
            fold_bank_sha256,
        };
        let manifest = TrainOnlyRedteamManifest {
            schema: "TrainOnlyRedteamManifest.v1".to_string(),
            run_id: SMART_V3_CPU_RUN_ID.to_string(),
            config,
            config_sha256: "5".repeat(64),
            development_only: true,
            confirmatory_valid: false,
            test_reuse_disclosure: DISCLOSURE.to_string(),
            parent: ParentLink {
                run_id: DEFAULT_PARENT_RUN.to_string(),
                manifest_path: root.join("unused-parent.json"),
                manifest_sha256: "6".repeat(64),
                validation_path: root.join("unused-validation.json"),
                validation_sha256: "7".repeat(64),
            },
            proposal_bundle: ProposalBundleLink {
                path: root.join("unused-proposals.jsonl"),
                sha256: "8".repeat(64),
                exact_rows_per_dataset: 256,
                jittered_rows_per_dataset: 192,
                scaling_statistic_sha256: "9".repeat(64),
            },
            scaling_statistic_path: root.join("unused-scaling.json"),
            scaling_statistic_sha256: "9".repeat(64),
            datasets: vec![dataset.clone()],
            firewall: SelectionFirewall {
                selection_workers_receive_training_only: true,
                contains_test_paths: false,
                contains_baseline_paths: false,
                test_paths_constructed: false,
                baseline_paths_constructed: false,
            },
        };
        let (path, _) = finalize_smart_only_selector(root, &manifest, &dataset, 1)
            .unwrap()
            .unwrap();
        let (receipt, _): (SmartV3SelectorComplete, _) = read_receipt(&path).unwrap();
        assert_eq!(receipt.schema, "TrainOnlyRedteamSmartV3SelectorComplete.v1");
        assert_eq!(receipt.final_variant.candidates.len(), 1);
        assert!(receipt.selection_frozen_before_test_access);
    }

    #[test]
    fn initialization_head_cap_is_applied_per_island() {
        let populations = ["dl", "transfer", "mixed"]
            .into_iter()
            .map(|island| {
                let rows = (0..30)
                    .map(|slot| {
                        initial_candidate(island, slot, ["ridge", "elastic", "hist"][slot / 10])
                    })
                    .collect::<Vec<_>>();
                (island.to_string(), rows)
            })
            .collect::<BTreeMap<_, _>>();

        validate_initial_populations(&populations, ISLAND_SIZE, &["dl", "transfer", "mixed"], 10)
            .unwrap();
    }

    #[test]
    fn global_head_cap_is_impossible_for_the_screened_pilot_pool() {
        let populations = ["dl", "transfer", "mixed"]
            .into_iter()
            .map(|island| {
                let rows = (0..30)
                    .map(|slot| {
                        initial_candidate(island, slot, ["ridge", "elastic", "hist"][slot / 10])
                    })
                    .collect::<Vec<_>>();
                (island.to_string(), rows)
            })
            .collect::<BTreeMap<_, _>>();
        let global_head_counts = populations.values().flatten().fold(
            HashMap::<&str, usize>::new(),
            |mut counts, candidate| {
                *counts.entry(candidate.head_family.as_str()).or_default() += 1;
                counts
            },
        );

        assert_eq!(global_head_counts["ridge"], 30);
        assert!(global_head_counts.values().any(|count| *count > 10));
        assert!(validate_initial_populations(
            &populations,
            ISLAND_SIZE,
            &["dl", "transfer", "mixed"],
            10,
        )
        .is_ok());
    }
}
