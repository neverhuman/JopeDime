use super::values::{search_digest, sha256_hex};
use super::*;

// Synthetic plans whose gains are exact by construction (no reporter involved).

type Outcome = dyn Fn(&str, Option<u64>, &str, &[f64], f64) -> Vec<f64>;

struct Synthetic {
    _root: tempfile::TempDir,
    root: PathBuf,
    specification: Value,
    searches: Vec<Value>,
    predictions: Vec<Value>,
}

impl Synthetic {
    fn new(datasets: &[(&str, &str)], outcome: &Outcome) -> Self {
        let root = tempfile::tempdir().unwrap();
        let mut fixture = Synthetic {
            root: root.path().to_path_buf(),
            _root: root,
            specification: json!({"arms":[{"id":"parent"},{"id":"challenger"}],"seeds":[7],"datasets":[]}),
            searches: Vec::new(),
            predictions: Vec::new(),
        };
        for (dataset, family) in datasets {
            let labels: BTreeMap<String, f64> = (0..20)
                .map(|i| (format!("{dataset}-r{i}"), f64::from(i)))
                .collect();
            let asset = fixture.asset(&format!("{dataset}-labels.json"), &json!(labels));
            fixture.specification["datasets"]
                .as_array_mut()
                .unwrap()
                .push(json!({"id":dataset,"family":family,"labels":asset}));
            // Sorted row identities, as the labels map orders them.
            let rows: Vec<&String> = labels.keys().collect();
            for outer in 0..5_usize {
                for audit in [Some(0), Some(1), Some(2), Some(3), Some(4), None] {
                    let score: Vec<String> = rows[outer * 4..(outer + 1) * 4]
                        .iter()
                        .map(|r| r.to_string())
                        .collect();
                    let train: Vec<String> = rows
                        .iter()
                        .filter(|r| !score.contains(r))
                        .map(|r| r.to_string())
                        .collect();
                    let search = json!({"dataset":dataset,"seed":7,"outer":outer,"audit":audit,
                        "train":train,"score":score,"inner_score":[train]});
                    fixture.searches.push(search.clone());
                    let mean = train.iter().map(|r| labels[r]).sum::<f64>() / train.len() as f64;
                    let actual: Vec<f64> = score.iter().map(|r| labels[r]).collect();
                    for arm in ["parent", "challenger"] {
                        let values = outcome(dataset, audit, arm, &actual, mean);
                        let name = format!("p-{}.json", fixture.predictions.len());
                        let prediction = json!({"search_sha256":search_digest(&search).unwrap(),
                            "arm":arm,"predictions":values});
                        let asset = fixture.asset(&name, &prediction);
                        fixture.predictions.push(asset);
                    }
                }
            }
        }
        fixture
    }

    fn asset(&self, name: &str, value: &Value) -> Value {
        let path = self.root.join(name);
        std::fs::write(&path, serde_json::to_vec(value).unwrap()).unwrap();
        json!({"path": path, "sha256": sha256_hex(&std::fs::read(&path).unwrap())})
    }

    fn plan(&self) -> Value {
        json!({"specification": self.specification, "searches": self.searches, "failures": []})
    }

    fn freeze(&self) -> Value {
        json!({"predictions": self.predictions, "failures": []})
    }
}

/// The parent predicts the training mean (the gain denominator) and the challenger is exact:
/// every gain is one.
fn perfect_challenger(
    _dataset: &str,
    _audit: Option<u64>,
    arm: &str,
    actual: &[f64],
    mean: f64,
) -> Vec<f64> {
    if arm == "challenger" {
        actual.to_vec()
    } else {
        vec![mean; actual.len()]
    }
}

fn almost_equal(left: f64, right: f64) -> bool {
    (left - right).abs() < 0.5e-12
}

#[test]
fn exact_gain_is_one_hundred_points() {
    let fixture = Synthetic::new(&[("d1", "f1")], &perfect_challenger);
    let result = score(&fixture.plan(), &fixture.freeze()).unwrap();
    assert!(result.complete);
    assert!(almost_equal(result.kpi_percentage_points.unwrap(), 100.0));
}

#[test]
fn min_is_taken_per_unit_and_families_weigh_equally() {
    // d2's challenger equals the parent on outer searches: its unit minimum is zero.
    let outer_regression = |dataset: &str, audit: Option<u64>, arm: &str, actual: &[f64], mean| {
        if dataset == "d2" && audit.is_none() {
            vec![mean; actual.len()]
        } else {
            perfect_challenger(dataset, audit, arm, actual, mean)
        }
    };
    let fixture = Synthetic::new(
        &[("d1", "f1"), ("d2", "f2"), ("d3", "f2")],
        &outer_regression,
    );
    let result = score(&fixture.plan(), &fixture.freeze()).unwrap();
    // f1 = 1.0; f2 = mean(d2 = 0.0, d3 = 1.0) = 0.5; families equally: 0.75.
    assert!(almost_equal(result.kpi_percentage_points.unwrap(), 75.0));
    assert!(almost_equal(result.family("f2").unwrap()[2], 0.5));
}

#[test]
fn missing_pair_and_zero_baseline_never_become_zero_gains() {
    let fixture = Synthetic::new(&[("d1", "f1")], &perfect_challenger);
    let mut freeze = fixture.freeze();
    freeze["predictions"].as_array_mut().unwrap().pop();
    let result = score(&fixture.plan(), &freeze).unwrap();
    assert!(!result.complete);
    assert!(result.kpi_percentage_points.is_none());
    assert!(result.reasons.iter().any(|r| r.starts_with("missing")));

    let mut fixture = Synthetic::new(&[("d1", "f1")], &perfect_challenger);
    let original: BTreeMap<String, f64> = serde_json::from_slice(
        &std::fs::read(
            fixture.specification["datasets"][0]["labels"]["path"]
                .as_str()
                .unwrap(),
        )
        .unwrap(),
    )
    .unwrap();
    let constant: BTreeMap<&String, f64> = original.keys().map(|r| (r, 3.0)).collect();
    fixture.specification["datasets"][0]["labels"] =
        fixture.asset("constant.json", &json!(constant));
    let result = score(&fixture.plan(), &fixture.freeze()).unwrap();
    assert!(!result.complete);
    assert!(result.kpi_percentage_points.is_none());
    assert_eq!(result.units.len(), 5);
}

#[test]
fn tampered_or_unplanned_predictions_are_refused() {
    let fixture = Synthetic::new(&[("d1", "f1")], &perfect_challenger);
    let freeze = fixture.freeze();
    std::fs::write(
        freeze["predictions"][0]["path"].as_str().unwrap(),
        br#"{"tampered": true}"#,
    )
    .unwrap();
    let error = score(&fixture.plan(), &freeze).unwrap_err();
    assert!(error.contains("altered asset"), "{error}");

    let fixture = Synthetic::new(&[("d1", "f1")], &perfect_challenger);
    let mut plan = fixture.plan();
    plan["searches"].as_array_mut().unwrap().remove(0);
    let error = score(&plan, &fixture.freeze()).unwrap_err();
    assert!(error.contains("unplanned prediction"), "{error}");
}

#[test]
fn comparison_detects_a_different_headline() {
    let fixture = Synthetic::new(&[("d1", "f1")], &perfect_challenger);
    let result = score(&fixture.plan(), &fixture.freeze()).unwrap();
    let units: Vec<Value> = result
        .units
        .iter()
        .map(|u| {
            let g = u.gains.unwrap();
            json!({"dataset":u.dataset,"seed":u.seed,"outer":u.outer,
                "gains":{"audit":g[0],"outer":g[1],"contribution":g[2]}})
        })
        .collect();
    let losses: Vec<Value> = result
        .losses
        .iter()
        .map(|((search, arm), (rmse, baseline))| {
            json!({"search_sha256":search,"arm":arm,"rmse":rmse,"training_mean_rmse":baseline})
        })
        .collect();
    let mut report = json!({"ok":true,"kpi_percentage_points":100.0,
        "families":{"f1":{"audit":1.0,"outer":1.0,"contribution":1.0}},
        "units":units,"losses":losses});
    assert_eq!(compare(&result, &report), Vec::<String>::new());
    report["kpi_percentage_points"] = json!(100.5);
    assert!(compare(&result, &report)
        .iter()
        .any(|p| p.starts_with("headline")));
    report["ok"] = json!(false);
    assert!(compare(&result, &report)
        .iter()
        .any(|p| p.starts_with("completeness")));
}

#[test]
fn comparison_detects_unit_family_and_loss_drift() {
    let fixture = Synthetic::new(&[("d1", "f1")], &perfect_challenger);
    let result = score(&fixture.plan(), &fixture.freeze()).unwrap();
    let report = |units: Vec<Value>, losses: Vec<Value>, family_outer: f64| {
        json!({"ok":true,"kpi_percentage_points":100.0,
            "families":{"f1":{"audit":1.0,"outer":family_outer,"contribution":1.0}},
            "units":units,"losses":losses})
    };
    let units = || -> Vec<Value> {
        result
            .units
            .iter()
            .map(|u| {
                json!({"dataset":u.dataset,"seed":u.seed,"outer":u.outer,
                    "gains":{"audit":1.0,"outer":1.0,"contribution":1.0}})
            })
            .collect()
    };
    let losses = || -> Vec<Value> {
        result
            .losses
            .iter()
            .map(|((search, arm), (rmse, baseline))| {
                json!({"search_sha256":search,"arm":arm,"rmse":rmse,"training_mean_rmse":baseline})
            })
            .collect()
    };
    assert!(compare(&result, &report(units(), losses(), 1.0)).is_empty());
    assert_eq!(
        compare(&result, &report(units(), losses(), 0.9)),
        ["family f1 outer"]
    );
    let mut drifted = units();
    drifted[2]["gains"]["contribution"] = json!(0.5);
    assert_eq!(
        compare(&result, &report(drifted, losses(), 1.0)),
        ["unit d1 seed 7 outer 2"]
    );
    let mut missing = units();
    missing.pop();
    assert_eq!(
        compare(&result, &report(missing, losses(), 1.0)),
        ["unit d1 seed 7 outer 4"]
    );
    let mut rows = losses();
    rows[0]["rmse"] = json!(rows[0]["rmse"].as_f64().unwrap() + 1e-3);
    rows.pop();
    let problems = compare(&result, &report(units(), rows, 1.0));
    assert_eq!(problems.len(), 2, "{problems:?}");
    assert!(problems[0].starts_with("loss "));
    assert_eq!(problems[1], "loss row count differs");
}

// ---------------------------------------------------------------------------------------------
