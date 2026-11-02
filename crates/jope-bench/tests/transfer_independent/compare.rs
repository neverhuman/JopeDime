use super::*;

fn close(a: Option<f64>, b: Option<f64>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => (a - b).abs() <= 1e-9 * 1.0_f64.max(a.abs()).max(b.abs()),
        _ => false,
    }
}

/// Every disagreement between the independent score and the reporter, or none.
pub(super) fn compare(result: &Score, report: &Value) -> Vec<String> {
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
