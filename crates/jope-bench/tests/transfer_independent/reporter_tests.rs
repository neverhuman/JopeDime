use super::*;
use super::values::sha256_hex;

// Cross-implementation agreement with the reporter, from the files it reads and writes.

fn store<T: Serialize>(root: &Path, name: &str, value: &T) -> Asset {
    let path = root.join(name);
    std::fs::write(&path, serde_json::to_vec(value).unwrap()).unwrap();
    Asset {
        sha256: sha256_hex(&std::fs::read(&path).unwrap()),
        path,
    }
}

fn specification(root: &Path) -> Specification {
    let common = store(root, "asset.json", &json!({"fixture":true}));
    let rows: Vec<_> = (0..50)
        .map(|i| Row {
            id: format!("r{i}"),
            group: format!("g{i:02}"),
            outer: (i % 5) as u8,
        })
        .collect();
    let labels: BTreeMap<_, _> = rows
        .iter()
        .enumerate()
        .map(|(i, r)| (r.id.clone(), i as f64))
        .collect();
    Specification {
        schema: "jope.transfer-spec/v1".into(),
        round: "fixture-only".into(),
        source_sha: "a".repeat(40),
        procedure: common.clone(),
        encoder: common.clone(),
        assets: vec![],
        arms: vec![
            Arm {
                id: "parent".into(),
                checkpoint: common.clone(),
            },
            Arm {
                id: "challenger".into(),
                checkpoint: common.clone(),
            },
        ],
        seeds: vec![42],
        datasets: vec![Dataset {
            id: "d1".into(),
            family: "f1".into(),
            features: common.clone(),
            labels: store(root, "labels.json", &labels),
            rows,
            evaluation_exposure: "untouched".into(),
            checkpoint_exposure: "untouched".into(),
            encoder_exposure: "untouched".into(),
            exposure_evidence: common,
        }],
        limits: Limits {
            max_searches: 60,
            candidates_per_search: 1,
            candidate_fit_seconds: 1,
            max_fit_seconds: 360,
            cpus: 1,
            memory_bytes: 1 << 20,
        },
    }
}

struct Reported {
    _root: tempfile::TempDir,
    plan: PathBuf,
    freeze: PathBuf,
    report: Value,
}

/// A complete frozen round scored by the reporter, optionally with constant labels.
fn reported(constant: Option<f64>) -> Reported {
    let root = tempfile::tempdir().unwrap();
    let path = root.path();
    let mut spec = specification(path);
    if let Some(value) = constant {
        let labels: BTreeMap<_, _> = spec.datasets[0]
            .rows
            .iter()
            .map(|r| (r.id.clone(), value))
            .collect();
        spec.datasets[0].labels = store(path, "constant-labels.json", &labels);
    }
    let input = store(path, "specification.json", &spec);
    let plan_path = path.join("plan.json");
    let planned = transfer::plan(&input.path, &plan_path).unwrap();
    assert_eq!(planned["ok"], true);
    let plan_bytes = std::fs::read(&plan_path).unwrap();
    let plan: Plan = serde_json::from_slice(&plan_bytes).unwrap();
    let plan_sha256 = sha256_hex(&plan_bytes);
    let labels: BTreeMap<String, f64> =
        serde_json::from_slice(&std::fs::read(&spec.datasets[0].labels.path).unwrap()).unwrap();
    let mut predictions = Vec::new();
    for (i, s) in plan.searches.iter().enumerate() {
        let search_sha256 = sha256_hex(&serde_json::to_vec(s).unwrap());
        for arm in &spec.arms {
            // Different per-fold recipes are intentional, and both arms lack absolute skill.
            let recipe = store(
                path,
                &format!("recipe-{i}-{}.json", arm.id),
                &format!("recipe-{i}-{}", arm.id),
            );
            let selection = SelectionReceipt {
                schema: "jope.transfer-nested-selection/v1".into(),
                search_sha256: search_sha256.clone(),
                procedure_sha256: spec.procedure.sha256.clone(),
                recipe_sha256: recipe.sha256.clone(),
                inner_score: s.inner_score.clone(),
                label_dependencies: s.train.clone(),
                complete_search_repeated: true,
            };
            let receipt = store(path, &format!("selection-{i}-{}.json", arm.id), &selection);
            let offset = if arm.id == "parent" { 100.0 } else { 90.0 };
            let prediction = Prediction {
                schema: "jope.transfer-procedure-audit/v1".into(),
                plan_sha256: plan_sha256.clone(),
                search_sha256: search_sha256.clone(),
                arm: arm.id.clone(),
                checkpoint_sha256: arm.checkpoint.sha256.clone(),
                procedure_sha256: spec.procedure.sha256.clone(),
                recipe_sha256: recipe.sha256.clone(),
                recipe,
                selection_receipt: receipt,
                fitting_rows: s.train.clone(),
                selection_label_rows: s.train.clone(),
                predictions: s
                    .score
                    .iter()
                    .map(|_| {
                        s.train.iter().map(|r| labels[r]).sum::<f64>() / s.train.len() as f64
                            + offset
                    })
                    .collect(),
                completed_sequence: (i + 1) as u64,
            };
            predictions.push(store(
                path,
                &format!("prediction-{i}-{}.json", arm.id),
                &prediction,
            ));
        }
    }
    let freeze = store(
        path,
        "freeze.json",
        &Freeze {
            schema: "jope.transfer-freeze/v1".into(),
            plan_sha256: plan_sha256.clone(),
            predictions,
            failures: vec![],
            frozen_sequence: 100,
        },
    );
    let release = store(
        path,
        "release.json",
        &Release {
            schema: "jope.transfer-release/v1".into(),
            plan_sha256,
            freeze_sha256: freeze.sha256.clone(),
            released_sequence: 102,
            first_scored_result_sequence: 101,
            executor_source_sha: spec.source_sha.clone(),
            isolation_evidence: spec.procedure.clone(),
            all_predictions_committed_before_scoring: true,
            scoring_labels_excluded_from_selection_and_fitting: true,
            no_cross_fold_result_feedback: true,
            independently_verified: true,
        },
    );
    let output = path.join("report.json");
    let returned = transfer::report(
        &plan_path,
        &freeze.path,
        &release.path,
        &release.sha256,
        &output,
    )
    .unwrap();
    let report: Value = serde_json::from_slice(&std::fs::read(&output).unwrap()).unwrap();
    assert_eq!(returned, report);
    Reported {
        _root: root,
        plan: plan_path,
        freeze: freeze.path,
        report,
    }
}

#[test]
fn independent_scorer_reproduces_the_report() {
    // A separate implementation recomputes baselines, losses, per-unit minima and the
    // family-weighted headline from the plan, labels and frozen predictions, and agrees exactly
    // with the reporter for a complete round and for an incomplete zero-baseline round.
    for constant in [None, Some(3.0)] {
        let fixture = reported(constant);
        let read = |path: &Path| -> Value {
            serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
        };
        let independent = score(&read(&fixture.plan), &read(&fixture.freeze)).unwrap();
        assert_eq!(
            compare(&independent, &fixture.report),
            Vec::<String>::new(),
            "{constant:?}"
        );
        assert_eq!(json!(independent.complete), fixture.report["ok"]);
        assert_eq!(independent.complete, constant.is_none());
    }
}
