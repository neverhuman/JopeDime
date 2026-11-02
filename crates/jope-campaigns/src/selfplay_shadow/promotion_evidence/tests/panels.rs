use super::*;

fn panel_pool() -> SealedPanelPoolV2 {
    let entries = (0..1_000)
        .map(|index| {
            let pool = match index {
                0..=699 => PanelPoolKindV2::Training,
                700..=799 => PanelPoolKindV2::Development,
                800..=949 => PanelPoolKindV2::Promotion,
                _ => PanelPoolKindV2::Audit,
            };
            SealedPanelEntryV2 {
                task: Task::Binary,
                context_identity: format!("context-{index}"),
                context_sha256: digest(&format!("context-{index}")),
                source_family_sha256: digest(&format!("family-{index}")),
                near_duplicate_cluster_sha256: digest(&format!("cluster-{index}")),
                origin: DatasetOriginV2::Real,
                pool,
                promotion_panel: (pool == PanelPoolKindV2::Promotion).then(|| "panel-a".to_owned()),
            }
        })
        .collect();
    SealedPanelPoolV2 {
        schema: SEALED_PANEL_POOL_V2_SCHEMA.to_owned(),
        entries,
        created_at_ms: 1,
    }
}

#[test]
fn sealed_panels_are_family_disjoint_and_single_use() {
    let pool = panel_pool();
    pool.validate().unwrap();
    let contexts = pool
        .promotion_panel(Task::Binary, "panel-a")
        .into_iter()
        .take(PROMOTION_MIN_CONTEXTS)
        .map(|entry| entry.context_sha256.clone())
        .collect();
    let receipt = SealedPanelConsumptionV1 {
        schema: SEALED_PANEL_CONSUMPTION_V1_SCHEMA.to_owned(),
        task: Task::Binary,
        panel: "panel-a".to_owned(),
        candidate_checkpoint_sha256: digest("candidate"),
        run_index: 0,
        seed: 9,
        context_sha256: contexts,
        consumed_at_ms: 1,
    };
    validate_panel_consumption(&pool, &receipt, &[]).unwrap();
    assert!(validate_panel_consumption(&pool, &receipt, std::slice::from_ref(&receipt)).is_err());
}

#[test]
fn sealed_panel_builder_creates_three_real_disjoint_panels_per_task() {
    let candidates = (0..5_120)
        .map(|index| SealedPanelCandidateV2 {
            task: if index % 2 == 0 {
                Task::Binary
            } else {
                Task::Regression
            },
            context_identity: format!("context-{index}"),
            context_sha256: Sha256Digest::parse(format!("{index:064x}")).unwrap(),
            source_family_sha256: Sha256Digest::parse(format!(
                "{:064x}",
                10_000 + (index / 4) * 2 + index % 2
            ))
            .unwrap(),
            near_duplicate_cluster_sha256: Sha256Digest::parse(format!(
                "{:064x}",
                20_000 + (index / 4) * 2 + index % 2
            ))
            .unwrap(),
            origin: if index < 1_024 {
                DatasetOriginV2::Real
            } else {
                DatasetOriginV2::Synthetic
            },
            base_seal_sha256: digest("base-seal"),
            catalog_sha256: digest("catalog"),
            train_sha256: digest(&format!("train-{index}")),
            test_sha256: digest(&format!("test-{index}")),
            context_facts_sha256: digest(&format!("facts-{index}")),
            prior_use_index_sha256: digest("prior-use"),
            zero_prior_use_proof_sha256: sealed_zero_prior_use_proof_v2(
                &Sha256Digest::parse(format!("{index:064x}")).unwrap(),
                &digest("base-seal"),
                &digest("catalog"),
                &digest("prior-use"),
                1,
            ),
            admission_watermark: 1,
        })
        .collect();
    let pool = build_sealed_panel_pool_v2(candidates, 1).unwrap();
    pool.validate().unwrap();
    for task in [Task::Binary, Task::Regression] {
        let panels = pool
            .entries
            .iter()
            .filter(|entry| entry.task == task && entry.pool == PanelPoolKindV2::Promotion)
            .filter_map(|entry| entry.promotion_panel.as_deref())
            .collect::<BTreeSet<_>>();
        assert_eq!(panels.len(), 3);
        assert!(panels
            .iter()
            .all(|panel| pool.promotion_panel(task, panel).len() == 128));
        let mut family_panels = BTreeMap::new();
        for entry in pool
            .entries
            .iter()
            .filter(|entry| entry.task == task && entry.pool == PanelPoolKindV2::Promotion)
        {
            let panel = entry.promotion_panel.as_deref().unwrap();
            assert!(family_panels
                .insert(&entry.source_family_sha256, panel)
                .is_none_or(|existing| existing == panel));
        }
    }
}
