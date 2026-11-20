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
