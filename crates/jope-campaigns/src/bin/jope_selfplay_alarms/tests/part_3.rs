#[test]
fn the_live_micro_cohort_set_raises_deficiency_on_binary_only() {
    let temp = TempDir::new().unwrap();
    let root = proposal_root(&temp, live_micro_cohort_summary());
    let alarms = proposal_scan(&root);
    let names = alarm_names(&alarms);
    // 19/64 = 0.297 is past the warn floor; worst context 20/64 = 0.3125 is
    // above the critical ratio, so this is a warning, not a critical.
    let row = alarms
        .rows
        .iter()
        .find(|row| row["alarm"] == "proposal_deficiency:binary")
        .unwrap_or_else(|| panic!("{names:?}"));
    assert_eq!(row["severity"], "warning");
    let observed = row["observed"].as_str().unwrap();
    assert!(observed.contains("19/64 contexts short"), "{observed}");
    assert!(observed.contains("worst 20/64"), "{observed}");
    assert!(observed.contains("583 rows borrowed"), "{observed}");
    // Regression filled every context itself and its modal head is 0.223.
    assert!(
        !names.iter().any(|name| name.ends_with(":regression")),
        "{names:?}"
    );
    // logit_ridge at 0.878 is the binary branch's NORMAL skew (an independent
    // 4-context offline run of the same checkpoint measured 0.883), so it must
    // NOT light the head alarm — otherwise the alarm is on permanently and
    // says nothing when the branch actually collapses.
    assert!(
        !names.contains(&"proposal_head_collapse:binary".to_owned()),
        "{names:?}"
    );
}

#[test]
fn a_binary_branch_drifting_past_its_measured_band_warns_before_it_collapses() {
    let temp = TempDir::new().unwrap();
    let mut summary = live_micro_cohort_summary();
    // Same 3,513 model-sampled binary proposals, redistributed so the modal
    // head holds 0.93: past the 0.90 measured-healthy band, short of 0.95.
    summary["native_head_counts_by_task"]["binary"] =
        json!({"logit_ridge": 3267, "tab_boost": 200, "extra_trees": 46});
    let root = proposal_root(&temp, summary);
    let alarms = proposal_scan(&root);
    let row = alarms
        .rows
        .iter()
        .find(|row| row["alarm"] == "proposal_head_collapse:binary")
        .unwrap_or_else(|| panic!("{:?}", alarm_names(&alarms)));
    assert_eq!(row["severity"], "warning");
    assert!(row["observed"]
        .as_str()
        .unwrap()
        .contains("logit_ridge holds 0.93"));
}

/// Whole-catalog set catalog-62054d7b05cb, same checkpoint, 2026-07-28
/// 20:47Z: 132,669 binary proposals, every one of them `logistic`, and a
/// binary decode validity of 0.0407.
#[test]
fn the_live_whole_catalog_set_is_a_critical_binary_head_collapse() {
    let temp = TempDir::new().unwrap();
    let root = proposal_root(
        &temp,
        json!({
            "schema": "JopeTestOnlyDlProposalSet.v1",
            "model_checkpoint_sha256": "fbcd5459f2c18f6cd5e4c23eef76a508a91d653a82f27c6ea41bcee8cafa2592",
            "proposals_per_dataset": 64,
            "native_head_counts_by_task": {
                "binary": {"logistic": 132_669},
                "regression": {
                    "gpr": 184_692, "ridge": 173_227, "histgbm2": 131_428,
                    "elasticnet": 119_461, "ensemble_ridge_gbm": 62_100,
                    "ensemble": 46_134, "histgbm": 36_543, "gbm": 29_992,
                    "ksmooth": 20_007
                }
            },
            "proposal_validity_by_task": {
                "binary": 0.040735005833279964,
                "regression": 0.9996332708535528
            }
        }),
    );
    let alarms = proposal_scan(&root);
    let names = alarm_names(&alarms);
    let head = alarms
        .rows
        .iter()
        .find(|row| row["alarm"] == "proposal_head_collapse:binary")
        .unwrap_or_else(|| panic!("{names:?}"));
    assert_eq!(head["severity"], "critical");
    assert!(head["observed"]
        .as_str()
        .unwrap()
        .contains("logistic holds 1.0000"));
    let validity = alarms
        .rows
        .iter()
        .find(|row| row["alarm"] == "proposal_task_validity:binary")
        .unwrap_or_else(|| panic!("{names:?}"));
    assert_eq!(validity["severity"], "critical");
    assert!(validity["observed"].as_str().unwrap().starts_with("0.0407"));
    assert!(
        !names.iter().any(|name| name.ends_with(":regression")),
        "{names:?}"
    );
}

#[test]
fn a_context_the_pool_could_not_fill_is_always_critical() {
    let temp = TempDir::new().unwrap();
    let root = proposal_root(
        &temp,
        json!({
            "schema": "JopeTestOnlyDlProposalSet.v1",
            "proposals_per_dataset": 64,
            "deficiency_by_task": {"binary": {
                "contexts": 64, "deficient_contexts": 1,
                "deficient_fraction": 1.0 / 64.0, "min_native_unique": 60,
                "backfilled_rows": 4, "unfilled_contexts": 1
            }}
        }),
    );
    let alarms = proposal_scan(&root);
    let names = alarm_names(&alarms);
    // One thin context out of 64 is under every deficiency threshold, so the
    // only thing that can be loud here is the below-budget file itself.
    assert!(
        !names.contains(&"proposal_deficiency:binary".to_owned()),
        "{names:?}"
    );
    let row = alarms
        .rows
        .iter()
        .find(|row| row["alarm"] == "proposal_unfilled:binary")
        .unwrap_or_else(|| panic!("{names:?}"));
    assert_eq!(row["severity"], "critical");
}

#[test]
fn a_collapsed_worst_context_is_critical_even_when_few_contexts_are_short() {
    let temp = TempDir::new().unwrap();
    let root = proposal_root(
        &temp,
        json!({
            "schema": "JopeTestOnlyDlProposalSet.v1",
            "proposals_per_dataset": 64,
            "deficiency_by_task": {"binary": {
                "contexts": 640, "deficient_contexts": 12,
                "deficient_fraction": 12.0 / 640.0,
                // the original severing failure: 34 of 64. 16 is a quarter.
                "min_native_unique": 16, "backfilled_rows": 300,
                "unfilled_contexts": 0
            }}
        }),
    );
    let alarms = proposal_scan(&root);
    let row = alarms
        .rows
        .iter()
        .find(|row| row["alarm"] == "proposal_deficiency:binary")
        .expect("deficiency alarm");
    assert_eq!(row["severity"], "critical");
}

#[test]
fn a_summary_without_the_new_fields_is_silent_rather_than_falsely_healthy() {
    let temp = TempDir::new().unwrap();
    // The pre-2026-07-28 summary shape: no deficiency_by_task, no head
    // histogram. Measuring nothing must not report health.
    let root = proposal_root(
        &temp,
        json!({
            "schema": "JopeTestOnlyDlProposalSet.v1",
            "proposals_per_dataset": 64,
            "counts": {"binary": 132_669, "regression": 803_584},
            "deficient_contexts": {"binary:27390545fb3675f1": 61},
            "proposal_validity_by_task": {"binary": 0.9, "regression": 0.99}
        }),
    );
    assert!(proposal_scan(&root).rows.is_empty());
}

#[test]
fn the_summary_beside_the_archive_is_used_when_the_named_one_was_reaped() {
    let temp = TempDir::new().unwrap();
    let root = proposal_root(&temp, live_micro_cohort_summary());
    // GC reaps training/tournament-work/, which is where the controller
    // lane's `summary` pointer aims.
    std::fs::remove_file(root.join("proposals/named.summary.json")).unwrap();
    let names = alarm_names(&proposal_scan(&root));
    assert!(
        names.contains(&"proposal_deficiency:binary".to_owned()),
        "{names:?}"
    );
}

#[test]
fn no_published_proposal_set_is_silent() {
    let temp = TempDir::new().unwrap();
    std::fs::create_dir_all(temp.path().join("proposals")).unwrap();
    assert!(proposal_scan(temp.path()).rows.is_empty());
}

// -----------------------------------------------------------------------
// Corpus freeze guardrails
// -----------------------------------------------------------------------

fn freeze_root(temp: &TempDir) -> PathBuf {
    let root = temp.path().to_path_buf();
    std::fs::create_dir_all(root.join("ops")).unwrap();
    std::fs::create_dir_all(root.join("training")).unwrap();
    root
}

fn freeze_scan(root: &Path) -> Alarms {
    let mut alarms = Alarms { rows: Vec::new() };
    check_corpus_freeze_guard(root, &mut alarms);
    alarms
}

fn write_pointers(root: &Path, frozen: u64, labels: u64) {
    std::fs::write(
        root.join("training/current-corpus.json"),
        serde_json::to_vec(&json!({"watermark": frozen})).unwrap(),
    )
    .unwrap();
    std::fs::write(
        root.join("ops/last-ingest.json"),
        serde_json::to_vec(&json!({"exact_rows_total": labels})).unwrap(),
    )
    .unwrap();
}
