use super::*;

fn strict_finite(v: Option<&Value>) -> Option<f64> {
    let n = v?.as_number()?.as_f64()?;
    n.is_finite().then_some(n)
}

fn league_evidence(row: &Value) -> Result<Map<String, Value>, String> {
    let folds = get(row, "folds")
        .and_then(Value::as_u64)
        .filter(|n| *n >= 2)
        .ok_or("LeagueLabel.v1 folds must be an integer of at least two")?;
    let mut evidence = Map::new();
    evidence.insert("folds".into(), Value::from(folds));
    for field in ["fold_plan", "evidence_tier", "evaluator_version"] {
        let value = get(row, field)
            .and_then(Value::as_str)
            .map(str::to_owned)
            .filter(|s| !s.trim().is_empty())
            .ok_or_else(|| format!("LeagueLabel.v1 {field} must be a nonempty string"))?;
        evidence.insert(field.into(), Value::String(value.trim().into()));
    }
    let runtime = strict_finite(get(row, "runtime_seconds"))
        .filter(|x| *x >= 0.0)
        .ok_or("LeagueLabel.v1 runtime_seconds must be finite and nonnegative")?;
    evidence.insert("runtime_seconds".into(), Value::from(runtime));
    let raw = get(row, "provenance")
        .and_then(Value::as_object)
        .ok_or("LeagueLabel.v1 provenance must be an object")?;
    let mut safe = Map::new();
    if let Some(v) = raw.get("gp_task_id") {
        let s = v
            .as_str()
            .map(str::to_owned)
            .filter(|s| !s.trim().is_empty())
            .ok_or("LeagueLabel.v1 provenance.gp_task_id must be nonempty")?;
        safe.insert("gp_task_id".into(), Value::String(s.trim().into()));
    }
    if let Some(v) = raw.get("oof_r2") {
        let x = strict_finite(Some(v)).ok_or("LeagueLabel.v1 provenance.oof_r2 must be finite")?;
        safe.insert("oof_r2".into(), Value::from(x));
    }
    if let Some(v) = raw.get("fold_r2") {
        let values = v
            .as_array()
            .filter(|a| a.len() == folds as usize)
            .ok_or("LeagueLabel.v1 provenance.fold_r2 must cover every fold")?;
        let numbers = values
            .iter()
            .map(|v| {
                strict_finite(Some(v))
                    .filter(|x| x.is_finite())
                    .ok_or("LeagueLabel.v1 provenance.fold_r2 must contain finite scores")
            })
            .collect::<Result<Vec<_>, _>>()?;
        safe.insert("fold_r2".into(), serde_json::json!(numbers));
    }
    if let Some(v) = raw.get("robust_formula") {
        let s = v
            .as_str()
            .map(str::to_owned)
            .filter(|s| !s.trim().is_empty())
            .ok_or("LeagueLabel.v1 provenance.robust_formula must be nonempty")?;
        safe.insert("robust_formula".into(), Value::String(s.trim().into()));
    }
    evidence.insert("provenance".into(), Value::Object(safe));
    Ok(evidence)
}

fn reward_pair_identity(
    dataset: &str,
    task: &str,
    folds: u64,
    fold_plan: &str,
    evaluator: &str,
) -> String {
    use sha2::{Digest, Sha256};
    // Keys and escaping follow Python's sorted, compact json.dumps contract.
    let payload = serde_json::json!({"dataset":dataset,"evaluator_version":evaluator,"fold_plan":fold_plan,
        "folds":folds,"schema":"RewardPairIdentity.v1","task":task});
    let canonical = super::python_canonical_json(&payload);
    format!(
        "reward-pair-v1:{}",
        hex::encode(Sha256::digest(canonical.as_bytes()))
    )
}

pub(super) fn parse_row(row: &Value) -> Result<Option<Map<String, Value>>, String> {
    let committed = row;
    let row = wrapped_label(row);
    let schema = text(get(row, "schema")).unwrap_or_default();
    let mut out = Map::new();
    if schema == LEAGUE_LABEL_SCHEMA {
        if text(get(row, "split"))
            .unwrap_or_else(|| "train".into())
            .to_lowercase()
            != "train"
        {
            return Ok(None);
        }
        let genome =
            first_truthy(row, "canonical_genome", "semantic_genome").and_then(|v| text(Some(v)));
        let dataset = text(get(row, "dataset"));
        let (Some(genome), Some(dataset)) = (genome, dataset) else {
            return Ok(None);
        };
        let ev = league_evidence(row)?;
        let provenance = get(&Value::Object(ev.clone()), "provenance")
            .cloned()
            .unwrap_or(Value::Null);
        let prov = provenance.as_object();
        let folds = get(row, "folds").and_then(Value::as_u64).unwrap();
        let fold_plan = text(get(row, "fold_plan")).unwrap();
        let evaluator = text(get(row, "evaluator_version")).unwrap();
        let task = text(get(row, "task"))
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "regression".into());
        let failure = get(row, "failure");
        out.insert("dataset_hash".into(), Value::String(dataset.clone()));
        out.insert("canonical_genome".into(), Value::String(genome));
        out.insert(
            "engine".into(),
            get(row, "engine").cloned().unwrap_or(Value::Null),
        );
        out.insert(
            "reward".into(),
            finite(get(row, "robust_score"))
                .map(Value::from)
                .unwrap_or(Value::Null),
        );
        for (field, source) in [
            ("oof_r2", "oof_r2"),
            ("fold_r2", "fold_r2"),
            ("gp_task_id", "gp_task_id"),
            ("robust_formula", "robust_formula"),
        ] {
            out.insert(
                field.into(),
                prov.and_then(|p| p.get(source))
                    .cloned()
                    .unwrap_or(Value::Null),
            );
        }
        for field in [
            "folds",
            "fold_plan",
            "evaluator_version",
            "evidence_tier",
            "runtime_seconds",
        ] {
            out.insert(
                field.into(),
                get(&Value::Object(ev.clone()), field)
                    .cloned()
                    .unwrap_or(Value::Null),
            );
        }
        out.insert(
            "reward_pair_identity".into(),
            Value::String(reward_pair_identity(
                &dataset, &task, folds, &fold_plan, &evaluator,
            )),
        );
        out.insert(
            "p_bad".into(),
            Value::from(if truthy(failure) { 1.0 } else { 0.0 }),
        );
        out.insert("failed".into(), Value::Bool(truthy(failure)));
        out.insert("task_type".into(), Value::String(task));
        out.insert(
            "semantic_plan_hash".into(),
            get(row, "semantic_genome").cloned().unwrap_or(Value::Null),
        );
        out.insert("reward_std".into(), Value::Null);
        out.insert("n_evidence".into(), Value::Null);
        out.insert(
            "promotion_eligible".into(),
            Value::Bool(failure == Some(&Value::Null) || failure.is_none()),
        );
    } else {
        let genome = get(row, "canonical_genome").and_then(|v| text(Some(v)));
        let dataset = first_truthy(row, "dataset_hash", "dataset_id").and_then(|v| text(Some(v)));
        let (Some(genome), Some(dataset)) = (genome, dataset) else {
            return Ok(None);
        };
        let reward = ["reward", "mean_realized", "realized_score"]
            .into_iter()
            .find_map(|k| finite(get(row, k)));
        let failed = truthy(get(row, "failure"))
            || finite(get(row, "n_failed")).unwrap_or(0.0) > 0.0 && reward.is_none();
        out.insert("dataset_hash".into(), Value::String(dataset));
        out.insert("canonical_genome".into(), Value::String(genome));
        out.insert(
            "engine".into(),
            get(row, "engine").cloned().unwrap_or(Value::Null),
        );
        out.insert(
            "reward".into(),
            reward.map(Value::from).unwrap_or(Value::Null),
        );
        out.insert(
            "oof_r2".into(),
            finite(get(row, "oof_r2"))
                .or(reward)
                .map(Value::from)
                .unwrap_or(Value::Null),
        );
        out.insert(
            "p_bad".into(),
            finite(get(row, "p_bad"))
                .map(Value::from)
                .unwrap_or(Value::Null),
        );
        out.insert("failed".into(), Value::Bool(failed));
        out.insert(
            "task_type".into(),
            Value::String(
                text(get(row, "task"))
                    .filter(|s| !s.is_empty())
                    .unwrap_or_else(|| "regression".into()),
            ),
        );
        out.insert(
            "semantic_plan_hash".into(),
            get(row, "semantic_plan_hash")
                .cloned()
                .unwrap_or(Value::Null),
        );
        out.insert(
            "reward_std".into(),
            finite(get(row, "reward_std"))
                .map(Value::from)
                .unwrap_or(Value::Null),
        );
        for (field, value) in [
            (
                "n_evidence",
                get(row, "n_evidence").cloned().unwrap_or(Value::Null),
            ),
            ("fold_r2", Value::Null),
            ("folds", Value::Null),
            ("fold_plan", Value::Null),
            ("evaluator_version", Value::Null),
            ("evidence_tier", Value::Null),
            ("runtime_seconds", Value::Null),
            ("gp_task_id", Value::Null),
            ("robust_formula", Value::Null),
            ("reward_pair_identity", Value::Null),
        ] {
            out.insert(field.into(), value);
        }
        out.insert(
            "promotion_eligible".into(),
            Value::Bool(
                get(row, "promotion_eligible")
                    .and_then(Value::as_bool)
                    .unwrap_or(true),
            ),
        );
    }
    for (target, a, b) in [
        ("label_watermark", "watermark", "_watermark"),
        ("committed_at_ms", "committed_at_ms", "_committed_at_ms"),
    ] {
        out.insert(
            target.into(),
            first_truthy(committed, a, b)
                .cloned()
                .unwrap_or(Value::Null),
        );
    }
    out.insert(
        "replay_hot_until_ms".into(),
        get(committed, "_replay_hot_until_ms")
            .cloned()
            .unwrap_or(Value::Null),
    );
    Ok(Some(out))
}
