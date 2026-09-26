//! W10-T09: an independent recomputation of the transfer KPI from the plan, the hash-pinned
//! labels and the frozen predictions.
//!
//! The oracle below reads only plan, freeze and report JSON and the assets they pin. It never
//! links `jope_bench::panel::transfer`'s scoring code: it recomputes the training-mean baseline
//! and each arm's RMSE per search, the five-audit mean gain and the outer gain per
//! dataset/seed/outer unit, their per-unit minimum, and the equal-weight means over units within
//! datasets, datasets within families and families. Missing or failed pairs, zero or nonfinite
//! baselines and invalid values leave the score incomplete; nothing becomes a zero gain. Given a
//! reporter output it lists every unit, family, loss and headline disagreement. Release and
//! isolation evidence is the reporter's responsibility and is not assessed here. The crate's
//! public types and entry points are used only to build the cross-implementation fixture.

use jope_bench::panel::transfer::{
    self, Arm, Asset, Dataset, Freeze, Limits, Plan, Prediction, Release, Row, SelectionReceipt,
    Specification,
};
use serde::Serialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};

const GAIN_FIELDS: [&str; 3] = ["audit", "outer", "contribution"];
/// The planner's search identity: SHA-256 of its compact, field-ordered JSON encoding.
const SEARCH_FIELDS: [&str; 7] = [
    "dataset",
    "seed",
    "outer",
    "audit",
    "train",
    "score",
    "inner_score",
];

type Oracle<T> = Result<T, String>;

fn require(condition: bool, message: impl Into<String>) -> Oracle<()> {
    if condition {
        Ok(())
    } else {
        Err(message.into())
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// A JSON member that must exist, as a subscript would demand.
fn field<'a>(value: &'a Value, key: &str) -> Oracle<&'a Value> {
    value
        .as_object()
        .and_then(|object| object.get(key))
        .ok_or_else(|| format!("missing field {key}"))
}

fn array<'a>(value: &'a Value, key: &str) -> Oracle<&'a Vec<Value>> {
    field(value, key)?
        .as_array()
        .ok_or_else(|| format!("{key} is not an array"))
}

fn string<'a>(value: &'a Value, key: &str) -> Oracle<&'a str> {
    field(value, key)?
        .as_str()
        .ok_or_else(|| format!("{key} is not a string"))
}

/// Values as an f-string renders them.
fn text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Null => "None".into(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        other => other.to_string(),
    }
}

/// Equality of identity members; numbers compare by value, as the planner's integers do.
fn same(left: &Value, right: &Value) -> bool {
    match (left.as_f64(), right.as_f64()) {
        (Some(left), Some(right)) => left == right,
        _ => left == right,
    }
}

fn truthy(value: Option<&Value>) -> bool {
    match value {
        None | Some(Value::Null) => false,
        Some(Value::Bool(flag)) => *flag,
        Some(Value::Number(number)) => number.as_f64().is_some_and(|n| n != 0.0),
        Some(Value::String(text)) => !text.is_empty(),
        Some(Value::Array(items)) => !items.is_empty(),
        Some(Value::Object(members)) => !members.is_empty(),
    }
}

fn load(asset: &Value) -> Oracle<Value> {
    let path = string(asset, "path")?;
    let data = std::fs::read(path).map_err(|error| format!("read {path}: {error}"))?;
    require(
        field(asset, "sha256")?.as_str() == Some(sha256_hex(&data).as_str()),
        format!("altered asset {path}"),
    )?;
    serde_json::from_slice(&data).map_err(|error| format!("parse {path}: {error}"))
}

fn search_digest(search: &Value) -> Oracle<String> {
    let mut encoded = String::from("{");
    for (position, name) in SEARCH_FIELDS.iter().enumerate() {
        if position > 0 {
            encoded.push(',');
        }
        encoded.push_str(&json!(name).to_string());
        encoded.push(':');
        encoded.push_str(&field(search, name)?.to_string());
    }
    encoded.push('}');
    Ok(sha256_hex(encoded.as_bytes()))
}

/// Correctly rounded sum of the values (Shewchuk partials with half-even correction).
fn exact_sum(values: &[f64]) -> Oracle<f64> {
    let mut partials: Vec<f64> = Vec::new();
    let (mut special, mut infinite) = (0.0_f64, 0.0_f64);
    for &value in values {
        let mut x = value;
        let mut kept = 0;
        for index in 0..partials.len() {
            let mut y = partials[index];
            if x.abs() < y.abs() {
                std::mem::swap(&mut x, &mut y);
            }
            let hi = x + y;
            let lo = y - (hi - x);
            if lo != 0.0 {
                partials[kept] = lo;
                kept += 1;
            }
            x = hi;
        }
        partials.truncate(kept);
        if x != 0.0 {
            if !x.is_finite() {
                require(!value.is_finite(), "intermediate overflow in exact sum")?;
                if value.is_infinite() {
                    infinite += value;
                }
                special += value;
                partials.clear();
            } else {
                partials.push(x);
            }
        }
    }
    if special != 0.0 {
        require(!infinite.is_nan(), "-inf + inf in exact sum")?;
        return Ok(special);
    }
    let mut remaining = partials.len();
    let mut hi = 0.0;
    if remaining > 0 {
        remaining -= 1;
        hi = partials[remaining];
        let mut lo = 0.0;
        while remaining > 0 {
            let x = hi;
            remaining -= 1;
            let y = partials[remaining];
            hi = x + y;
            lo = y - (hi - x);
            if lo != 0.0 {
                break;
            }
        }
        if remaining > 0
            && ((lo < 0.0 && partials[remaining - 1] < 0.0)
                || (lo > 0.0 && partials[remaining - 1] > 0.0))
        {
            let y = lo * 2.0;
            let x = hi + y;
            if y == x - hi {
                hi = x;
            }
        }
    }
    Ok(hi)
}

fn mean(values: &[f64]) -> Oracle<f64> {
    require(!values.is_empty(), "mean of no values")?;
    Ok(exact_sum(values)? / values.len() as f64)
}

/// Compensated (Neumaier) running sum of squared errors.
fn compensated_sum(values: impl Iterator<Item = f64>) -> f64 {
    let (mut total, mut compensation) = (0.0_f64, 0.0_f64);
    for value in values {
        let next = total + value;
        compensation += if total.abs() >= value.abs() {
            (total - next) + value
        } else {
            (value - next) + total
        };
        total = next;
    }
    if compensation != 0.0 && compensation.is_finite() {
        total += compensation;
    }
    total
}

fn rmse(actual: &[f64], predicted: &Value) -> Oracle<f64> {
    let predicted = predicted
        .as_array()
        .ok_or("prediction count differs from scoring rows")?;
    require(
        actual.len() == predicted.len() && !actual.is_empty(),
        "prediction count differs from scoring rows",
    )?;
    let predicted = predicted
        .iter()
        .map(|value| value.as_f64().filter(|x| x.is_finite()))
        .collect::<Option<Vec<f64>>>()
        .ok_or("nonfinite prediction")?;
    let squared = actual.iter().zip(&predicted).map(|(a, p)| (a - p).powi(2));
    let value = (compensated_sum(squared) / actual.len() as f64).sqrt();
    require(value.is_finite(), "nonfinite loss")?;
    Ok(value)
}

#[derive(Debug)]
struct Unit {
    dataset: Value,
    seed: Value,
    outer: Value,
    gains: Option<[f64; 3]>,
}

#[derive(Debug)]
struct Score {
    complete: bool,
    kpi_percentage_points: Option<f64>,
    families: Vec<(String, [f64; 3])>,
    units: Vec<Unit>,
    reasons: Vec<String>,
    losses: Vec<((String, String), (f64, f64))>,
}

impl Score {
    fn loss(&self, search: &str, arm: &str) -> Option<(f64, f64)> {
        self.losses
            .iter()
            .find(|((s, a), _)| s == search && a == arm)
            .map(|(_, loss)| *loss)
    }
    fn family(&self, name: &str) -> Option<[f64; 3]> {
        self.families
            .iter()
            .find(|(family, _)| family == name)
            .map(|(_, gains)| *gains)
    }
}

fn score(plan: &Value, freeze: &Value) -> Oracle<Score> {
    let specification = field(plan, "specification")?;
    let arms: BTreeSet<&str> = array(specification, "arms")?
        .iter()
        .map(|arm| string(arm, "id"))
        .collect::<Oracle<_>>()?;
    require(
        arms.contains("parent") && arms.contains("challenger"),
        "parent and challenger arms required",
    )?;
    // Plan order; a repeated identity keeps its first position and its last value.
    let mut searches: Vec<(String, &Value)> = Vec::new();
    for search in array(plan, "searches")? {
        let digest = search_digest(search)?;
        match searches.iter_mut().find(|(known, _)| *known == digest) {
            Some(entry) => entry.1 = search,
            None => searches.push((digest, search)),
        }
    }
    let mut predictions: HashMap<(String, String), Value> = HashMap::new();
    for asset in array(freeze, "predictions")? {
        let prediction = load(asset)?;
        let search = field(&prediction, "search_sha256")?.as_str();
        let arm = field(&prediction, "arm")?.as_str();
        let (Some(search), Some(arm)) = (search, arm) else {
            return Err("unplanned prediction".into());
        };
        require(
            searches.iter().any(|(known, _)| known == search) && arms.contains(arm),
            "unplanned prediction",
        )?;
        let key = (search.to_owned(), arm.to_owned());
        require(!predictions.contains_key(&key), "duplicate prediction")?;
        predictions.insert(key, field(&prediction, "predictions")?.clone());
    }
    let failures = array(freeze, "failures")?;
    let mut incomplete: Vec<String> = array(plan, "failures")?
        .iter()
        .map(|failure| format!("plan: {}", text(failure)))
        .collect();
    for failure in failures {
        incomplete.push(format!(
            "terminal {}: {} {}",
            text(field(failure, "status")?),
            text(field(failure, "search_sha256")?),
            text(field(failure, "arm")?)
        ));
    }
    for (search, _) in &searches {
        for arm in &arms {
            let terminal = failures.iter().any(|failure| {
                failure["search_sha256"].as_str() == Some(search.as_str())
                    && failure["arm"].as_str() == Some(*arm)
            });
            if !predictions.contains_key(&(search.clone(), arm.to_string())) && !terminal {
                incomplete.push(format!("missing {search} {arm}"));
            }
        }
    }
    if !incomplete.is_empty() {
        return Ok(Score {
            complete: false,
            kpi_percentage_points: None,
            families: Vec::new(),
            units: Vec::new(),
            reasons: incomplete,
            losses: Vec::new(),
        });
    }
    let datasets = array(specification, "datasets")?;
    let mut labels: HashMap<&str, Value> = HashMap::new();
    for dataset in datasets {
        labels.insert(string(dataset, "id")?, load(field(dataset, "labels")?)?);
    }
    let mut losses = Vec::new();
    for (search, planned) in &searches {
        let table = labels
            .get(string(planned, "dataset")?)
            .ok_or("search names an unknown dataset")?;
        let label = |row: &Value| -> Oracle<f64> {
            row.as_str()
                .and_then(|row| table.get(row))
                .and_then(Value::as_f64)
                .ok_or_else(|| format!("no numeric label for row {}", text(row)))
        };
        let train = array(planned, "train")?
            .iter()
            .map(label)
            .collect::<Oracle<Vec<_>>>()?;
        let actual = array(planned, "score")?
            .iter()
            .map(label)
            .collect::<Oracle<Vec<_>>>()?;
        let baseline_value = mean(&train)?;
        let baseline = rmse(&actual, &json!(vec![baseline_value; actual.len()]))?;
        for arm in &arms {
            let key = (search.clone(), arm.to_string());
            // Controls are scored as separate loss rows, never pooled.
            losses.push((key.clone(), (rmse(&actual, &predictions[&key])?, baseline)));
        }
    }
    let loss = |search: &str, arm: &str| {
        losses
            .iter()
            .find(|((s, a), _)| s == search && a == arm)
            .map(|(_, loss)| *loss)
            .ok_or_else(|| format!("no loss for {search} {arm}"))
    };
    let seeds = array(specification, "seeds")?;
    let (mut units, mut reasons) = (Vec::new(), Vec::new());
    let mut families: Vec<(String, Vec<[f64; 3]>)> = Vec::new();
    for dataset in datasets {
        let id = field(dataset, "id")?;
        let mut dataset_units = Vec::new();
        for seed in seeds {
            for outer in 0..5_u64 {
                let outer = json!(outer);
                let mut audit = Vec::new();
                let mut last = Vec::new();
                for (search, planned) in &searches {
                    if same(field(planned, "dataset")?, id)
                        && same(field(planned, "seed")?, seed)
                        && same(field(planned, "outer")?, &outer)
                    {
                        if field(planned, "audit")?.is_null() {
                            last.push(search.as_str());
                        } else {
                            audit.push(search.as_str());
                        }
                    }
                }
                let values = audit
                    .iter()
                    .chain(&last)
                    .map(|search| {
                        let (parent, baseline) = loss(search, "parent")?;
                        let (challenger, _) = loss(search, "challenger")?;
                        Ok((parent, challenger, baseline))
                    })
                    .collect::<Oracle<Vec<_>>>()?;
                let mut unit = Unit {
                    dataset: id.clone(),
                    seed: seed.clone(),
                    outer: outer.clone(),
                    gains: None,
                };
                if audit.len() != 5
                    || last.len() != 1
                    || values.iter().any(|&(_, _, b)| b <= 0.0 || !b.is_finite())
                {
                    reasons.push(format!(
                        "{} seed {} outer {}: invalid baseline or fold set",
                        text(id),
                        text(seed),
                        outer
                    ));
                    units.push(unit);
                    continue;
                }
                let audit_gain = mean(
                    &values[..5]
                        .iter()
                        .map(|&(p, c, b)| (p - c) / b)
                        .collect::<Vec<_>>(),
                )?;
                let (p, c, b) = values[5];
                let outer_gain = (p - c) / b;
                let contribution = if outer_gain < audit_gain {
                    outer_gain
                } else {
                    audit_gain
                };
                let gains = [audit_gain, outer_gain, contribution];
                unit.gains = Some(gains);
                units.push(unit);
                dataset_units.push(gains);
            }
        }
        if dataset_units.len() == 5 * seeds.len() {
            let mut means = [0.0; 3];
            for (index, value) in means.iter_mut().enumerate() {
                *value = mean(&dataset_units.iter().map(|u| u[index]).collect::<Vec<_>>())?;
            }
            let family = string(dataset, "family")?;
            match families.iter_mut().find(|(name, _)| name == family) {
                Some((_, members)) => members.push(means),
                None => families.push((family.to_owned(), vec![means])),
            }
        }
    }
    let mut family_means = Vec::new();
    for (name, members) in &families {
        let mut means = [0.0; 3];
        for (index, value) in means.iter_mut().enumerate() {
            *value = mean(&members.iter().map(|d| d[index]).collect::<Vec<_>>())?;
        }
        family_means.push((name.clone(), means));
    }
    let complete = reasons.is_empty() && !family_means.is_empty();
    let kpi_percentage_points = if complete {
        Some(100.0 * mean(&family_means.iter().map(|(_, g)| g[2]).collect::<Vec<_>>())?)
    } else {
        None
    };
    Ok(Score {
        complete,
        kpi_percentage_points,
        families: family_means,
        units,
        reasons,
        losses,
    })
}

fn close(a: Option<f64>, b: Option<f64>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => (a - b).abs() <= 1e-9 * 1.0_f64.max(a.abs()).max(b.abs()),
        _ => false,
    }
}

/// Every disagreement between the independent score and the reporter, or none.
fn compare(result: &Score, report: &Value) -> Vec<String> {
    let mut problems = Vec::new();
    let ok = report.get("ok");
    if truthy(ok) != result.complete {
        problems.push(format!(
            "completeness: report ok={} independent={}",
            text(ok.unwrap_or(&Value::Null)),
            text(&json!(result.complete))
        ));
    }
    let headline = report.get("kpi_percentage_points");
    if !result.complete {
        if headline.is_some_and(|value| !value.is_null()) {
            problems
                .push("report released a headline the independent score cannot complete".into());
        }
        return problems;
    }
    if !close(
        result.kpi_percentage_points,
        headline.and_then(Value::as_f64),
    ) {
        problems.push(format!(
            "headline {} != {:?}",
            text(headline.unwrap_or(&Value::Null)),
            result.kpi_percentage_points
        ));
    }
    for (name, gains) in &result.families {
        let theirs = report
            .get("families")
            .and_then(|families| families.get(name));
        for (index, gain_field) in GAIN_FIELDS.iter().enumerate() {
            let reported = theirs
                .and_then(|t| t.get(gain_field))
                .and_then(Value::as_f64);
            if !close(Some(gains[index]), reported) {
                problems.push(format!("family {name} {gain_field}"));
            }
        }
    }
    let empty = Vec::new();
    let reported_units = report
        .get("units")
        .and_then(Value::as_array)
        .unwrap_or(&empty);
    for unit in &result.units {
        // A repeated reported unit identity keeps its last value.
        let theirs = reported_units
            .iter()
            .rev()
            .find(|u| {
                same(&u["dataset"], &unit.dataset)
                    && same(&u["seed"], &unit.seed)
                    && same(&u["outer"], &unit.outer)
            })
            .and_then(|u| u.get("gains"))
            .filter(|gains| !gains.is_null());
        let agrees = theirs.is_some_and(|theirs| {
            GAIN_FIELDS.iter().enumerate().all(|(index, gain_field)| {
                close(
                    unit.gains.map(|g| g[index]),
                    theirs.get(gain_field).and_then(Value::as_f64),
                )
            })
        });
        if !agrees {
            problems.push(format!(
                "unit {} seed {} outer {}",
                text(&unit.dataset),
                text(&unit.seed),
                text(&unit.outer)
            ));
        }
    }
    let rows = report
        .get("losses")
        .and_then(Value::as_array)
        .unwrap_or(&empty);
    for row in rows {
        let mine = match (row["search_sha256"].as_str(), row["arm"].as_str()) {
            (Some(search), Some(arm)) => result.loss(search, arm),
            _ => None,
        };
        if let Some((rmse, baseline)) = mine {
            if !(close(Some(rmse), row["rmse"].as_f64())
                && close(Some(baseline), row["training_mean_rmse"].as_f64()))
            {
                problems.push(format!(
                    "loss {} {}",
                    text(&row["search_sha256"]),
                    text(&row["arm"])
                ));
            }
        }
    }
    if rows.len() != result.losses.len() {
        problems.push("loss row count differs".into());
    }
    problems
}

// ---------------------------------------------------------------------------------------------
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
