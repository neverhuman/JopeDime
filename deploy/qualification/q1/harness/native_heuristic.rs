//! Development-only reference adapter. Never a learned or sealed-Q1 evaluator.
use feat_math::evolve::api::{evaluate_candidate, CvFold, EvaluationFidelity, STARTERS};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicUsize, Ordering};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let request: Value = serde_json::from_reader(std::io::stdin())?;
    if request["schema"] != "jope.q1-request/v1"
        || request["mode"] != "heuristic_smoke"
        || request["guidance"] != "heuristic"
        || request["metric"] != "rmse"
        || request["dataset"]["task"] != "regression"
    {
        return Err("reference adapter refuses learned/qualification requests".into());
    }
    let budget = usize::try_from(request["budget"].as_u64().ok_or("missing budget")?)?;
    if !(1..=4).contains(&budget) || budget > STARTERS.len() {
        return Err("smoke budget must be 1..4".into());
    }
    let x: Vec<Vec<f64>> = serde_json::from_value(request["payload"]["x"].clone())?;
    let y: Vec<f64> = serde_json::from_value(request["payload"]["y"].clone())?;
    if x.len() < 16
        || x.len() > 128
        || y.len() != x.len()
        || x.iter().any(|r| {
            r.is_empty() || r.len() > 8 || r.len() != x[0].len() || r.iter().any(|v| !v.is_finite())
        })
        || y.iter().any(|v| !v.is_finite())
    {
        return Err("invalid bounded numeric smoke matrix".into());
    }
    let folds: Vec<CvFold> = request["payload"]["folds"]
        .as_array()
        .ok_or("missing folds")?
        .iter()
        .map(|v| {
            Ok(CvFold {
                train: serde_json::from_value(v["train"].clone())?,
                validation: serde_json::from_value(v["validation"].clone())?,
            })
        })
        .collect::<Result<_, serde_json::Error>>()?;
    if folds.len() != 2 {
        return Err("smoke adapter requires two folds".into());
    }
    let scored = AtomicUsize::new(0);
    let metric = |actual: &[f64], pred: &[f64]| {
        scored.fetch_add(1, Ordering::Relaxed);
        (actual
            .iter()
            .zip(pred)
            .map(|(a, p)| (a - p).powi(2))
            .sum::<f64>()
            / actual.len() as f64)
            .sqrt()
    };
    let mut best: Option<(f64, &str)> = None;
    let mut failures = Vec::new();
    let mut completed = 0;
    for (attempt, source) in STARTERS.iter().take(budget).enumerate() {
        match evaluate_candidate(
            source,
            &x,
            &y,
            &folds,
            EvaluationFidelity::Surrogate,
            &metric,
        ) {
            Ok(scores) => {
                completed += 1;
                let loss = scores.iter().sum::<f64>() / scores.len() as f64;
                if best.is_none_or(|(old, _)| loss < old) {
                    best = Some((loss, source));
                }
            }
            Err(error) => failures.push(json!({"attempt":attempt,"reason":format!("{error:#}")})),
        }
    }
    let (loss, program) = best.map_or((Value::Null, Value::Null), |(loss, source)| {
        (json!(loss), json!(source))
    });
    let response = json!({"schema":"jope.q1-response/v1","request_sha256":request["request_sha256"],
        "status":if best.is_some(){"complete"}else{"failed"},"guidance":"heuristic",
        "metric":"rmse","charged":budget,"completed":completed,"loss":loss,
        "fold_scores_completed":scored.load(Ordering::Relaxed),"candidate_failures":failures,
        "champion_program":program,"adapter":"rust-reference-smoke-only"});
    println!("{}", serde_json::to_string(&response)?);
    Ok(())
}
