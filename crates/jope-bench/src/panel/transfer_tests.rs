use super::*;

fn store<T: Serialize>(root: &Path, name: &str, value: &T) -> Asset {
    let path = root.join(name);
    std::fs::write(&path, serde_json::to_vec(value).unwrap()).unwrap();
    Asset {
        sha256: hash(&std::fs::read(&path).unwrap()),
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
struct Fixture {
    _root: tempfile::TempDir,
    plan: PathBuf,
    freeze: PathBuf,
    release: PathBuf,
    release_hash: String,
    output: PathBuf,
}
fn fixture() -> Fixture {
    fixture_with_labels(None)
}
fn fixture_with_labels(constant: Option<f64>) -> Fixture {
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
    let plan = build(spec.clone()).unwrap();
    assert!(plan.failures.is_empty());
    let plan_asset = store(path, "plan.json", &plan);
    let labels: BTreeMap<String, f64> =
        serde_json::from_slice(&asset(&spec.datasets[0].labels).unwrap()).unwrap();
    let mut predictions = Vec::new();
    for (i, s) in plan.searches.iter().enumerate() {
        for arm in &spec.arms {
            // Different per-fold recipes are intentional, and both arms lack absolute skill.
            let recipe_asset = store(
                path,
                &format!("recipe-{i}-{}.json", arm.id),
                &format!("recipe-{i}-{}", arm.id),
            );
            let recipe = recipe_asset.sha256.clone();
            let selection = SelectionReceipt {
                schema: "jope.transfer-nested-selection/v1".into(),
                search_sha256: digest(s).unwrap(),
                procedure_sha256: spec.procedure.sha256.clone(),
                recipe_sha256: recipe.clone(),
                inner_score: s.inner_score.clone(),
                label_dependencies: s.train.clone(),
                complete_search_repeated: true,
            };
            let receipt = store(path, &format!("selection-{i}-{}.json", arm.id), &selection);
            let p = Prediction {
                schema: PROTOCOL.into(),
                plan_sha256: plan_asset.sha256.clone(),
                search_sha256: digest(s).unwrap(),
                arm: arm.id.clone(),
                checkpoint_sha256: arm.checkpoint.sha256.clone(),
                procedure_sha256: spec.procedure.sha256.clone(),
                recipe_sha256: recipe,
                recipe: recipe_asset,
                selection_receipt: receipt,
                fitting_rows: s.train.clone(),
                selection_label_rows: s.train.clone(),
                predictions: s
                    .score
                    .iter()
                    .map(|_| {
                        s.train.iter().map(|r| labels[r]).sum::<f64>() / s.train.len() as f64
                            + if arm.id == "parent" { 100.0 } else { 90.0 }
                    })
                    .collect(),
                completed_sequence: (i + 1) as u64,
            };
            predictions.push(store(path, &format!("prediction-{i}-{}.json", arm.id), &p));
        }
    }
    let freeze = Freeze {
        schema: "jope.transfer-freeze/v1".into(),
        plan_sha256: plan_asset.sha256.clone(),
        predictions,
        failures: vec![],
        frozen_sequence: 100,
    };
    let freeze_asset = store(path, "freeze.json", &freeze);
    let release = Release {
        schema: "jope.transfer-release/v1".into(),
        plan_sha256: plan_asset.sha256,
        freeze_sha256: freeze_asset.sha256,
        released_sequence: 102,
        first_scored_result_sequence: 101,
        executor_source_sha: spec.source_sha,
        isolation_evidence: spec.procedure,
        all_predictions_committed_before_scoring: true,
        scoring_labels_excluded_from_selection_and_fitting: true,
        no_cross_fold_result_feedback: true,
        independently_verified: true,
    };
    let release_asset = store(path, "release.json", &release);
    Fixture {
        plan: plan_asset.path,
        freeze: freeze_asset.path,
        release: release_asset.path,
        release_hash: release_asset.sha256,
        output: path.join("report.json"),
        _root: root,
    }
}
fn run(f: &Fixture) -> Result<Value> {
    report(&f.plan, &f.freeze, &f.release, &f.release_hash, &f.output)
}
fn change_release(f: &mut Fixture, update: impl FnOnce(&mut Release)) {
    let mut r: Release = read(&f.release).unwrap();
    update(&mut r);
    std::fs::write(&f.release, serde_json::to_vec(&r).unwrap()).unwrap();
    f.release_hash = hash(&std::fs::read(&f.release).unwrap());
}
#[test]
fn min_is_per_unit_and_keeps_negative_and_one_sided_gains() {
    assert_eq!(
        gains(&[(1.0, 1.0, 1.0); 5], (1.0, 0.8, 1.0))
            .unwrap()
            .contribution,
        0.0
    );
    assert_eq!(
        gains(&[(1.0, 0.8, 1.0); 5], (1.0, 1.0, 1.0))
            .unwrap()
            .contribution,
        0.0
    );
    let a = gains(&[(1.0, 0.7, 1.0); 5], (1.0, 1.1, 1.0)).unwrap();
    let b = gains(&[(1.0, 1.1, 1.0); 5], (1.0, 0.7, 1.0)).unwrap();
    assert!((average(&[a, b]).contribution + 0.1).abs() < 1e-12);
    let g = gains(&[(2.0, 1.5, 1.0); 5], (2.0, 1.5, 1.0)).unwrap();
    assert_eq!(g.audit_skill, -0.5);
    assert!(gains(&[(1.0, 0.0, 0.0); 5], (1.0, 0.0, 1.0)).is_err());
    assert!(gains(&[(1.0, f64::NAN, 1.0); 5], (1.0, 0.0, 1.0)).is_err());
    let a = [
        (2.0, 1.0, 1.0),
        (2.0, 1.0, 2.0),
        (2.0, 1.0, 4.0),
        (2.0, 1.0, 5.0),
        (2.0, 1.0, 10.0),
    ];
    assert!((gains(&a, (2.0, 1.0, 1.0)).unwrap().audit - 0.41).abs() < 1e-12);
}
#[test]
fn independent_python_scorer_reproduces_the_report() {
    // W10-T09: a separate implementation recomputes baselines, losses, per-unit minima and the
    // family-weighted headline from the plan, labels and frozen predictions, and agrees exactly
    // with the reporter for a complete round and for an incomplete zero-baseline round.
    for constant in [None, Some(3.0)] {
        let f = fixture_with_labels(constant);
        let reported = run(&f).unwrap();
        let output = std::process::Command::new("python3")
            .arg(Path::new(env!("CARGO_MANIFEST_DIR")).join("src/panel/transfer_score.py"))
            .arg("--plan")
            .arg(&f.plan)
            .arg("--freeze")
            .arg(&f.freeze)
            .arg("--report")
            .arg(&f.output)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let independent: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(independent["disagreements"], json!([]));
        assert_eq!(independent["complete"], reported["ok"]);
    }
}
#[test]
fn plan_is_thirty_searches_per_arm_and_label_free_searches() {
    let root = tempfile::tempdir().unwrap();
    let spec = specification(root.path());
    let p = build(spec.clone()).unwrap();
    assert_eq!(p.planned_searches, 60);
    assert_eq!(p.searches.len(), 30);
    assert_eq!(p.estimated_candidate_fits, 360);
    for s in &p.searches {
        assert!(s.score.iter().all(|r| !s.train.contains(r)));
        assert_eq!(
            s.inner_score.iter().map(Vec::len).sum::<usize>(),
            s.train.len()
        );
        assert!(s.inner_score.iter().flatten().all(|r| s.train.contains(r)));
    }
    let mut changed = spec;
    changed.datasets[0].labels = store(
        root.path(),
        "changed-labels.json",
        &json!({"scoring_labels":"changed"}),
    );
    assert_eq!(p.searches, build(changed).unwrap().searches);
}
#[test]
fn feasibility_exposure_resources_and_duplicate_aliases_fail_explicitly() {
    let root = tempfile::tempdir().unwrap();
    let mut s = specification(root.path());
    s.datasets[0].rows.truncate(5);
    assert!(!build(s).unwrap().failures.is_empty());
    let mut s = specification(root.path());
    s.limits.max_searches = 59;
    assert!(!build(s).unwrap().failures.is_empty());
    let mut s = specification(root.path());
    s.datasets[0].encoder_exposure = "unknown".into();
    assert!(!build(s).unwrap().failures.is_empty());
    let mut s = specification(root.path());
    let mut duplicate = s.datasets[0].clone();
    duplicate.id = "alias".into();
    duplicate.family = "fake-family".into();
    s.datasets.push(duplicate);
    assert!(build(s).is_err());
    let mut s = specification(root.path());
    s.datasets[0].rows[1].group = s.datasets[0].rows[0].group.clone();
    assert!(build(s).is_err());
}
#[test]
fn complete_frozen_fixture_accepts_different_recipes_and_warns_no_absolute_skill() {
    let f = fixture();
    let r = run(&f).unwrap();
    assert_eq!(r["ok"], true);
    assert_eq!(r["valid_units"], 5);
    assert!(r["kpi_percentage_points"].as_f64().unwrap() > 0.0);
    assert_eq!(
        r["absolute_skill_statement"],
        "no absolute skill demonstrated"
    );
    assert!(run(&f).is_err(), "immutable output must not be overwritten");
}
#[test]
fn missing_pair_blocks_all_result_release() {
    let mut f = fixture();
    let mut freeze: Freeze = read(&f.freeze).unwrap();
    freeze.predictions.pop();
    std::fs::write(&f.freeze, serde_json::to_vec(&freeze).unwrap()).unwrap();
    let h = hash(&std::fs::read(&f.freeze).unwrap());
    change_release(&mut f, |r| r.freeze_sha256 = h);
    let r = run(&f).unwrap();
    assert_eq!(r["ok"], false);
    assert_eq!(r["scores_released"], false);
    assert_eq!(r["planned_units"], 5);
    assert!(r.get("units").is_none());
}
#[test]
fn premature_release_and_altered_hashes_are_rejected() {
    let mut f = fixture();
    change_release(&mut f, |r| r.first_scored_result_sequence = 99);
    assert!(run(&f).is_err());
    let f = fixture();
    std::fs::write(&f.freeze, b"{}").unwrap();
    assert!(run(&f).is_err());
    let f = fixture();
    let freeze: Freeze = read(&f.freeze).unwrap();
    std::fs::write(&freeze.predictions[0].path, b"{}").unwrap();
    assert!(run(&f).is_err());
    let mut f = fixture();
    change_release(&mut f, |r| r.independently_verified = false);
    assert!(run(&f).is_err());
}
#[test]
fn scoring_row_in_selection_closure_is_rejected_even_after_rehash() {
    let mut f = fixture();
    let mut freeze: Freeze = read(&f.freeze).unwrap();
    let mut p: Prediction = read(&freeze.predictions[0].path).unwrap();
    p.selection_label_rows.push("forbidden".into());
    std::fs::write(&freeze.predictions[0].path, serde_json::to_vec(&p).unwrap()).unwrap();
    freeze.predictions[0].sha256 = hash(&std::fs::read(&freeze.predictions[0].path).unwrap());
    std::fs::write(&f.freeze, serde_json::to_vec(&freeze).unwrap()).unwrap();
    let h = hash(&std::fs::read(&f.freeze).unwrap());
    change_release(&mut f, |r| r.freeze_sha256 = h);
    assert!(run(&f).is_err());
}

#[test]
fn failed_terminal_outcome_retains_identity_and_reason() {
    let mut f = fixture();
    let mut freeze: Freeze = read(&f.freeze).unwrap();
    let removed = freeze.predictions.pop().unwrap();
    let prediction: Prediction = read(&removed.path).unwrap();
    freeze.failures.push(FailedPrediction {
        search_sha256: prediction.search_sha256.clone(),
        arm: prediction.arm,
        status: "failed".into(),
        reason: "fixture budget exhausted".into(),
        completed_sequence: 90,
    });
    std::fs::write(&f.freeze, serde_json::to_vec(&freeze).unwrap()).unwrap();
    let h = hash(&std::fs::read(&f.freeze).unwrap());
    change_release(&mut f, |r| r.freeze_sha256 = h);
    let report = run(&f).unwrap();
    assert_eq!(report["ok"], false);
    assert_eq!(report["terminal_outcomes"], 60);
    assert_eq!(
        report["unavailable"][0]["search_sha256"],
        prediction.search_sha256
    );
    assert_eq!(
        report["unavailable"][0]["reason"],
        "fixture budget exhausted"
    );
}
#[test]
fn equal_family_weighting_differs_from_pooling_datasets() {
    let low = gains(&[(1.0, 1.2, 1.0); 5], (1.0, 1.2, 1.0)).unwrap();
    let high = gains(&[(1.0, 0.4, 1.0); 5], (1.0, 0.4, 1.0)).unwrap();
    let family_a = average(&[average(&[low, low]), average(&[low, low])]);
    let family_b = average(&[average(&[high, high])]);
    let global = average(&[family_a, family_b]);
    assert!((global.contribution - 0.2).abs() < 1e-12);
    assert!((average(&[low, low, high]).contribution - global.contribution).abs() > 0.1);
    assert_ne!(sign(0.0), sign(0.1));
    assert_eq!(sign(0.0), sign(-0.0));
}
/// CPU contract fixture only: no production evaluation executor is admitted by this test.
/// Select a constant-offset recipe by inner CV and refit using only permitted labels.
fn frozen_fixture_selection(search: &Search, labels: &BTreeMap<String, f64>) -> (usize, Vec<f64>) {
    let offsets = [0.0, 2.0];
    let scores: Vec<_> = offsets
        .iter()
        .map(|offset| {
            search
                .inner_score
                .iter()
                .map(|held| {
                    let train: Vec<_> = search.train.iter().filter(|r| !held.contains(r)).collect();
                    let mean =
                        train.iter().map(|r| labels[*r]).sum::<f64>() / train.len() as f64 + offset;
                    held.iter().map(|r| (labels[r] - mean).powi(2)).sum::<f64>() / held.len() as f64
                })
                .sum::<f64>()
        })
        .collect();
    let winner = usize::from(scores[1] < scores[0]);
    let mean = search.train.iter().map(|r| labels[r]).sum::<f64>() / search.train.len() as f64
        + offsets[winner];
    (winner, vec![mean; search.score.len()])
}
#[test]
fn own_scoring_label_changes_cannot_change_fixture_selection_or_frozen_predictions() {
    let root = tempfile::tempdir().unwrap();
    let spec = specification(root.path());
    let plan = build(spec.clone()).unwrap();
    let labels: BTreeMap<String, f64> =
        serde_json::from_slice(&asset(&spec.datasets[0].labels).unwrap()).unwrap();
    for search in &plan.searches {
        let original = frozen_fixture_selection(search, &labels);
        let mut changed = labels.clone();
        for id in &search.score {
            changed.insert(id.clone(), -99999.0);
        }
        assert_eq!(original, frozen_fixture_selection(search, &changed));
        let original_score = rmse(
            &search.score.iter().map(|r| labels[r]).collect::<Vec<_>>(),
            &original.1,
        )
        .unwrap();
        let altered_score = rmse(
            &search.score.iter().map(|r| changed[r]).collect::<Vec<_>>(),
            &original.1,
        )
        .unwrap();
        assert_ne!(original_score, altered_score);
    }
}

#[test]
fn zero_baseline_retains_every_unit_without_global_score() {
    let f = fixture_with_labels(Some(1.0));
    let r = run(&f).unwrap();
    assert_eq!(r["ok"], false);
    assert!(r["kpi_percentage_points"].is_null());
    assert_eq!(r["planned_units"], 5);
    assert_eq!(r["units"].as_array().unwrap().len(), 5);
    assert_eq!(r["failures"].as_array().unwrap().len(), 5);
}
#[test]
fn nonfinite_baseline_overflow_writes_explicit_failure_report() {
    let f = fixture_with_labels(Some(1e308));
    assert!(run(&f).is_err());
    let r: Value = read(&f.output).unwrap();
    assert_eq!(r["ok"], false);
    assert_eq!(r["claimed_planned_units_unverified"], 5);
}
#[test]
fn adaptive_panel_cannot_be_redeclared_untouched() {
    let root = tempfile::tempdir().unwrap();
    let mut s = specification(root.path());
    s.datasets[0].id = "230_machine_cpu".into();
    assert!(build(s)
        .unwrap()
        .failures
        .iter()
        .any(|f| f.contains("existing adaptive panel")));
}
