use super::{fit_cut_summary_context, Candidate, RetrievalMode, RetrievalSeedProvider};
use jope_pipeline::{
    dataset, emit::Language, evidence, prepare, run_with_seed_provider, search::Budget,
    seed_provider::CutSeedProvider, sha256_hex, FitCutView, Policy, RunOptions, FROM_SEED_PROVIDER,
};

fn committed_dataset() -> String {
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../datasets/test/pmlb-229-pwlinear"
    )
    .to_owned()
}

fn copy_dataset_with_poisoned_holdout(
    dest_root: &std::path::Path,
    seed: u64,
    cut_index: usize,
) -> String {
    let src = std::path::PathBuf::from(committed_dataset());
    let resolved = dataset::resolve(&committed_dataset()).unwrap();
    let rows = dataset::training_rows(&resolved).unwrap();
    let prep = prepare::partition(&rows, &resolved.row.dataset_hash, seed, 1).unwrap();
    let prep_written = prep.write(&dest_root.join("ref-prep")).unwrap();
    let hold_set: std::collections::BTreeSet<usize> = prep_written.cuts[cut_index]
        .hold_indices
        .iter()
        .copied()
        .collect();

    let out = dest_root.join("poisoned-holdout");
    std::fs::create_dir_all(&out).unwrap();

    let train_text = std::fs::read_to_string(src.join("train.csv")).unwrap();
    let mut mutated_train = String::new();
    for (row_idx, line) in train_text.lines().enumerate() {
        if hold_set.contains(&row_idx) {
            let vals: Vec<f64> = line
                .split(',')
                .enumerate()
                .map(|(col_idx, s)| {
                    let v: f64 = s.parse().unwrap();
                    950.0 * v - 412.0 + (row_idx * 7 + col_idx) as f64
                })
                .collect();
            mutated_train.push_str(
                &vals
                    .iter()
                    .map(|v| format!("{v}"))
                    .collect::<Vec<_>>()
                    .join(","),
            );
            mutated_train.push('\n');
        } else {
            mutated_train.push_str(line);
            mutated_train.push('\n');
        }
    }

    let test_text = std::fs::read_to_string(src.join("test.csv")).unwrap();
    let mut mutated_test = String::new();
    for (row_idx, line) in test_text.lines().enumerate() {
        let vals: Vec<f64> = line
            .split(',')
            .map(|s| 1200.0 * s.parse::<f64>().unwrap() + 999.0 + row_idx as f64)
            .collect();
        mutated_test.push_str(
            &vals
                .iter()
                .map(|v| format!("{v}"))
                .collect::<Vec<_>>()
                .join(","),
        );
        mutated_test.push('\n');
    }

    std::fs::write(out.join("train.csv"), &mutated_train).unwrap();
    std::fs::write(out.join("test.csv"), &mutated_test).unwrap();

    let mut meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(src.join("metadata.json")).unwrap()).unwrap();
    meta["files"]["train"]["sha256"] = sha256_hex(mutated_train.as_bytes()).into();
    meta["files"]["train"]["bytes"] = mutated_train.len().into();
    meta["files"]["test"]["sha256"] = sha256_hex(mutated_test.as_bytes()).into();
    meta["files"]["test"]["bytes"] = mutated_test.len().into();
    std::fs::write(out.join("metadata.json"), meta.to_string()).unwrap();
    out.display().to_string()
}

fn candidate(
    id: &str,
    genome: &str,
    parent_fam: &str,
    ds_fam: &str,
    role: &str,
    ctx: [f64; 4],
    manifest_sha: &str,
) -> Candidate {
    Candidate {
        candidate_id: id.into(),
        genome: genome.into(),
        program_sha256: sha256_hex(genome.as_bytes()),
        parent_families: vec![parent_fam.into()],
        dataset_families: vec![ds_fam.into()],
        context: ctx.to_vec(),
        role: role.into(),
        role_manifest_sha256: manifest_sha.into(),
    }
}

#[test]
fn retrieval_seed_provider_feeds_family_disjoint_seeds_post_partition_without_holdout_leakage() {
    let tmp = tempfile::tempdir().unwrap();
    let role_sha = "a".repeat(64);
    let snap_sha = "b".repeat(64);

    let resolved_clean = dataset::resolve(&committed_dataset()).unwrap();
    let rows_clean = dataset::training_rows(&resolved_clean).unwrap();
    let prep_clean = prepare::partition(
        &rows_clean,
        &resolved_clean.row.dataset_hash,
        jope_pipeline::DEFAULT_SEED,
        1,
    )
    .unwrap()
    .write(&tmp.path().join("clean-prep"))
    .unwrap();
    let views_clean = FitCutView::from_preparation(&prep_clean, &resolved_clean.row.dataset_hash);
    let target_ctx = fit_cut_summary_context(&views_clean);

    let g_same_family = "(pipe X (robust_scale) (ridge))";
    let g_non_train = "(pipe X (blom) (ridge))";
    let g_disjoint_near = "(pipe X (standardize) (ridge))";
    let g_disjoint_far = "(pipe X (rank) (ridge))";

    let provider = RetrievalSeedProvider {
        request_id: "req-v2".into(),
        own_parent_families: vec!["parent-own".into()],
        own_dataset_families: vec!["pmlb-229".into()],
        own_program_sha256: Vec::new(),
        role_manifest_sha256: role_sha.clone(),
        snapshot_sha256: snap_sha,
        eligible_support: vec![
            "c-same-family".into(),
            "c-val-role".into(),
            "c-disjoint-near".into(),
            "c-disjoint-far".into(),
        ],
        candidates: vec![
            // Closest context, but shares dataset family -> MUST be excluded!
            candidate(
                "c-same-family",
                g_same_family,
                "parent-other-1",
                "pmlb-229",
                "train",
                target_ctx,
                &role_sha,
            ),
            // Disjoint family, but non-train role -> MUST be excluded!
            candidate(
                "c-val-role",
                g_non_train,
                "parent-other-2",
                "family-alpha",
                "val",
                target_ctx,
                &role_sha,
            ),
            // Disjoint family + train + near context -> selected #1!
            candidate(
                "c-disjoint-near",
                g_disjoint_near,
                "parent-other-3",
                "family-beta",
                "train",
                [
                    target_ctx[0] + 0.05,
                    target_ctx[1],
                    target_ctx[2],
                    target_ctx[3],
                ],
                &role_sha,
            ),
            // Disjoint family + train + farther context -> selected #2!
            candidate(
                "c-disjoint-far",
                g_disjoint_far,
                "parent-other-4",
                "family-gamma",
                "train",
                [
                    target_ctx[0] + 2.0,
                    target_ctx[1] + 1.0,
                    target_ctx[2],
                    target_ctx[3],
                ],
                &role_sha,
            ),
        ],
        mode: RetrievalMode::Cold,
    };

    // Check poisoned holdout dataset produces byte-identical FitCutViews and proposed SeedGenomes.
    let poisoned_ds =
        copy_dataset_with_poisoned_holdout(tmp.path(), jope_pipeline::DEFAULT_SEED, 0);
    let resolved_poison = dataset::resolve(&poisoned_ds).unwrap();
    let rows_poison = dataset::training_rows(&resolved_poison).unwrap();
    let prep_poison = prepare::partition(
        &rows_poison,
        &resolved_poison.row.dataset_hash,
        jope_pipeline::DEFAULT_SEED,
        1,
    )
    .unwrap()
    .write(&tmp.path().join("poison-prep"))
    .unwrap();
    let views_poison =
        FitCutView::from_preparation(&prep_poison, &resolved_poison.row.dataset_hash);

    assert_eq!(views_clean, views_poison);
    assert_eq!(views_clean[0].fit_digest(), views_poison[0].fit_digest());
    assert_eq!(
        fit_cut_summary_context(&views_clean),
        fit_cut_summary_context(&views_poison)
    );
    let seeds_clean = provider.propose_seeds(&views_clean, 4).unwrap();
    let seeds_poison = provider.propose_seeds(&views_poison, 4).unwrap();
    assert_eq!(seeds_clean, seeds_poison);
    assert_eq!(seeds_clean.len(), 2);
    assert_eq!(seeds_clean[0].genome, g_disjoint_near);
    assert_eq!(seeds_clean[1].genome, g_disjoint_far);

    // Run full pipeline end-to-end on both clean and poisoned-holdout datasets.
    let mk_opts = |ds: String, out: &std::path::Path| RunOptions {
        budget: Budget {
            evaluations: 16,
            seconds: 300,
        },
        rng_seed: 19,
        cuts: 1,
        languages: vec![Language::Rust],
        policy: Policy::Auto,
        source_commit: Some("81dd03702a846d652f15a1f56480d8f074bddec4".into()),
        ..RunOptions::new(ds, out)
    };
    let out_clean = tmp.path().join("run-clean");
    let out_poison = tmp.path().join("run-poison");
    let receipt_clean =
        run_with_seed_provider(&mk_opts(committed_dataset(), &out_clean), Some(&provider)).unwrap();
    let receipt_poison =
        run_with_seed_provider(&mk_opts(poisoned_ds, &out_poison), Some(&provider)).unwrap();

    let verified = evidence::verify_run(&out_clean.join("receipt.json")).unwrap();
    let seeding_clean = receipt_clean.search.seeding.as_ref().unwrap();
    let seeding_poison = receipt_poison.search.seeding.as_ref().unwrap();
    assert_eq!(seeding_clean, seeding_poison);
    assert_eq!(seeding_clean.seeds.len(), 2);
    assert_eq!(seeding_clean.seeds[0].genome, g_disjoint_near);
    assert_eq!(seeding_clean.seeds[1].genome, g_disjoint_far);
    assert!(seeding_clean
        .seeds
        .iter()
        .all(|s| s.source == FROM_SEED_PROVIDER));
    assert!(verified.cuts[0]
        .1
        .iter()
        .any(|o| o.genome == g_disjoint_near && o.generation == 0));
    assert!(verified.cuts[0]
        .1
        .iter()
        .any(|o| o.genome == g_disjoint_far && o.generation == 0));
    assert_eq!(
        std::fs::read(out_clean.join("prepared/provider-seeds.json")).unwrap(),
        std::fs::read(out_poison.join("prepared/provider-seeds.json")).unwrap()
    );
}

#[test]
fn retrieval_seed_provider_validates_at_zero_limit_and_does_not_starve_past_duplicates() {
    let tmp = tempfile::tempdir().unwrap();
    let resolved = dataset::resolve(&committed_dataset()).unwrap();
    let rows = dataset::training_rows(&resolved).unwrap();
    let prep = prepare::partition(
        &rows,
        &resolved.row.dataset_hash,
        jope_pipeline::DEFAULT_SEED,
        1,
    )
    .unwrap()
    .write(&tmp.path().join("prep"))
    .unwrap();
    let views = FitCutView::from_preparation(&prep, &resolved.row.dataset_hash);
    let ctx = fit_cut_summary_context(&views);
    let role_sha = "c".repeat(64);
    let snap_sha = "d".repeat(64);

    // 1. Invalid manifest/snapshot digest MUST fail closed even when limit == 0.
    let mut bad_provider = RetrievalSeedProvider {
        request_id: "req-edge".into(),
        own_parent_families: vec!["p-own".into()],
        own_dataset_families: vec!["d-own".into()],
        own_program_sha256: Vec::new(),
        role_manifest_sha256: "not-hex".into(),
        snapshot_sha256: snap_sha.clone(),
        eligible_support: vec!["c1".into()],
        candidates: Vec::new(),
        mode: RetrievalMode::Cold,
    };
    assert!(bad_provider.propose_seeds(&views, 0).is_err());
    bad_provider.role_manifest_sha256 = role_sha.clone();
    bad_provider.snapshot_sha256 = "short".into();
    assert!(bad_provider.propose_seeds(&views, 0).is_err());

    // 2. Top ranks holding BOOTSTRAP_GENOME + canonical duplicate must not starve rank 4 at limit=2.
    let g_boot = jope_pipeline::search::BOOTSTRAP_GENOME;
    let g_dup_a = "(pipe X (standardize) (ridge))";
    let g_dup_b = "  (pipe X (standardize) (ridge))  ";
    let g_second = "(pipe X (rank) (ridge))";

    let dedup_provider = RetrievalSeedProvider {
        request_id: "req-dedup".into(),
        own_parent_families: vec!["p-own".into()],
        own_dataset_families: vec!["d-own".into()],
        own_program_sha256: Vec::new(),
        role_manifest_sha256: role_sha.clone(),
        snapshot_sha256: snap_sha,
        eligible_support: vec!["r1".into(), "r2".into(), "r3".into(), "r4".into()],
        candidates: vec![
            candidate("r1", g_boot, "p1", "f1", "train", ctx, &role_sha),
            candidate(
                "r2",
                g_dup_a,
                "p2",
                "f2",
                "train",
                [ctx[0] + 0.01, ctx[1], ctx[2], ctx[3]],
                &role_sha,
            ),
            candidate(
                "r3",
                g_dup_b,
                "p3",
                "f3",
                "train",
                [ctx[0] + 0.02, ctx[1], ctx[2], ctx[3]],
                &role_sha,
            ),
            candidate(
                "r4",
                g_second,
                "p4",
                "f4",
                "train",
                [ctx[0] + 0.03, ctx[1], ctx[2], ctx[3]],
                &role_sha,
            ),
        ],
        mode: RetrievalMode::Cold,
    };

    assert!(dedup_provider.propose_seeds(&views, 0).unwrap().is_empty());
    let selected = dedup_provider.propose_seeds(&views, 2).unwrap();
    assert_eq!(selected.len(), 2);
    assert_eq!(selected[0].genome, g_dup_a);
    assert_eq!(selected[1].genome, g_second);
}

#[test]
fn retrieval_seed_provider_handles_empty_eligible_pool_under_auto_and_require_policies() {
    let tmp = tempfile::tempdir().unwrap();
    let resolved = dataset::resolve(&committed_dataset()).unwrap();
    let rows = dataset::training_rows(&resolved).unwrap();
    let prep = prepare::partition(
        &rows,
        &resolved.row.dataset_hash,
        jope_pipeline::DEFAULT_SEED,
        1,
    )
    .unwrap()
    .write(&tmp.path().join("prep-empty"))
    .unwrap();
    let views = FitCutView::from_preparation(&prep, &resolved.row.dataset_hash);
    let ctx = fit_cut_summary_context(&views);
    let role_sha = "e".repeat(64);
    let snap_sha = "f".repeat(64);

    // Every candidate is filtered out (same family or non-train role).
    let empty_provider = RetrievalSeedProvider {
        request_id: "req-empty".into(),
        own_parent_families: vec!["p-own".into()],
        own_dataset_families: vec!["pmlb-229".into()],
        own_program_sha256: Vec::new(),
        role_manifest_sha256: role_sha.clone(),
        snapshot_sha256: snap_sha,
        eligible_support: vec!["c-own".into(), "c-val".into()],
        candidates: vec![
            candidate(
                "c-own",
                "(pipe X (standardize) (ridge))",
                "p-other",
                "pmlb-229",
                "train",
                ctx,
                &role_sha,
            ),
            candidate(
                "c-val",
                "(pipe X (rank) (ridge))",
                "p-other-2",
                "other-fam",
                "val",
                ctx,
                &role_sha,
            ),
        ],
        mode: RetrievalMode::Cold,
    };

    assert!(empty_provider.propose_seeds(&views, 4).unwrap().is_empty());

    let mk_opts = |policy: Policy, out: &std::path::Path| RunOptions {
        budget: Budget {
            evaluations: 12,
            seconds: 300,
        },
        rng_seed: 19,
        cuts: 1,
        languages: vec![Language::Rust],
        policy,
        source_commit: Some("6c7ec97a497d9686e966549c61bfd20e99773c65".into()),
        ..RunOptions::new(committed_dataset(), out)
    };

    let auto_receipt = run_with_seed_provider(
        &mk_opts(Policy::Auto, &tmp.path().join("auto-empty")),
        Some(&empty_provider),
    )
    .unwrap();
    let auto_seeding = auto_receipt.search.seeding.as_ref().unwrap();
    assert_eq!(auto_seeding.offered, 0);
    assert!(auto_seeding.seeds.is_empty());

    let req_err = run_with_seed_provider(
        &mk_opts(Policy::Require, &tmp.path().join("require-empty")),
        Some(&empty_provider),
    );
    assert!(req_err.is_err());
}

#[test]
fn donor_dataset_shuffle_permutes_across_distinct_donor_families_without_rotating_coordinates() {
    let role_sha = "1".repeat(64);
    let snap_sha = "2".repeat(64);
    let query_ctx = [10.0, 0.0, 0.0, 0.0];

    let req = super::RetrievalRequest {
        request_id: "req-donor-shuffle".into(),
        own_parent_families: vec!["p-own".into()],
        own_dataset_families: vec!["d-own".into()],
        own_program_sha256: Vec::new(),
        role_manifest_sha256: role_sha.clone(),
        context: query_ctx.to_vec(),
        eligible_support: vec!["c-a".into(), "c-b".into(), "c-c".into()],
        cap: 1,
        requested_by_source: std::collections::BTreeMap::from([(
            jope_pipeline::seeds::SeedSourceV2::OtherLineageRetrieval,
            1,
        )]),
        snapshot_sha256: snap_sha,
        cold_cost_ms: 5,
        warm_cost_ms: 1,
        cache_warmup_ms: 2,
    };

    // c-a matches query_ctx; c-b matches 1-coord rotated [0,0,0,10]; c-c is a distinct donor family.
    let candidates = vec![
        candidate(
            "c-a",
            "(pipe X (standardize) (ridge))",
            "p-a",
            "fam-a",
            "train",
            query_ctx,
            &role_sha,
        ),
        candidate(
            "c-b",
            "(pipe X (rank) (ridge))",
            "p-b",
            "fam-b",
            "train",
            [0.0, 0.0, 0.0, 10.0],
            &role_sha,
        ),
        candidate(
            "c-c",
            "(pipe X (robust_scale) (ridge))",
            "p-c",
            "fam-c",
            "train",
            [0.0, 5.0, 5.0, 0.0],
            &role_sha,
        ),
    ];

    let cold = super::materialize(&req, &candidates, RetrievalMode::Cold).unwrap();
    let rotated = super::materialize(&req, &candidates, RetrievalMode::ShuffledContext).unwrap();
    let donor = super::materialize(&req, &candidates, RetrievalMode::DonorDatasetShuffle).unwrap();
    let donor_repeat =
        super::materialize(&req, &candidates, RetrievalMode::DonorDatasetShuffle).unwrap();

    assert_eq!(cold.selected_candidate_ids, vec!["c-a"]);
    assert_eq!(rotated.selected_candidate_ids, vec!["c-b"]);
    assert_eq!(
        donor.selected_candidate_ids,
        donor_repeat.selected_candidate_ids
    );
    assert_ne!(donor.selected_candidate_ids, vec!["c-a"]);
    assert_eq!(donor.mode, RetrievalMode::DonorDatasetShuffle);
    assert_eq!(
        serde_json::to_string(&RetrievalMode::DonorDatasetShuffle).unwrap(),
        "\"donor_dataset_shuffle\""
    );

    // Refuses when fewer than 2 distinct donor dataset families exist.
    let single_family = vec![candidates[0].clone()];
    assert!(super::materialize(&req, &single_family, RetrievalMode::DonorDatasetShuffle).is_err());
}
