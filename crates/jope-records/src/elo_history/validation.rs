fn by_rating(ratings: &BTreeMap<String, f64>) -> Vec<&str> {
    let mut ordered: Vec<(&String, f64)> = ratings.iter().map(|(m, r)| (m, *r)).collect();
    ordered.sort_by(|a, b| {
        (-a.1)
            .partial_cmp(&-b.1)
            .expect("finite ratings")
            .then_with(|| a.0.cmp(b.0))
    });
    ordered.into_iter().map(|(m, _)| m.as_str()).collect()
}

/// How far the retained ratings (the iterate at the historical cap) sit from a longer fit.
pub fn sensitivity(root: &Path, history: &Value, cap: usize, zstd: Zstd) -> Result<Value> {
    let result = replay(root, index(history, "replay")?, zstd)?;
    let (longer, iterations, converged) = fit(&result.ladder.pairs, cap);
    let retained = by_rating(&result.ladder.ratings);
    let extended = by_rating(&longer);
    require!(!longer.is_empty(), "max() arg is an empty sequence");
    let shift = longer
        .iter()
        .map(|(m, r)| (r - result.ladder.ratings[m]).abs())
        .fold(f64::NEG_INFINITY, |a, b| if b > a { b } else { a });
    let mut dict = Dict::new();
    dict.insert("history", index(history, "id")?.clone());
    dict.insert("historical_cap", ITERATIONS.into());
    dict.insert("historical_converged", result.ladder.converged.into());
    dict.insert("extended_iterations", iterations.into());
    dict.insert("extended_converged", converged.into());
    dict.insert("max_abs_rating_shift", Value::Float(shift));
    dict.insert(
        "positions_changed",
        retained
            .iter()
            .zip(&extended)
            .filter(|(a, b)| a != b)
            .count()
            .into(),
    );
    dict.insert(
        "top10_unchanged",
        (retained[..retained.len().min(10)] == extended[..extended.len().min(10)]).into(),
    );
    Ok(Value::Dict(dict))
}

fn truthy_field(value: &Value, key: &str) -> bool {
    value.get(key).is_some_and(Value::truthy)
}

/// A history's disposition rules and retained-file digests.
pub fn check_history(
    root: &Path,
    history: &Value,
    seen: &mut HashSet<Key>,
    zstd: Zstd,
) -> Result<()> {
    let complete = history.as_dict().is_some() && COMMON.iter().all(|f| truthy_field(history, f));
    require!(
        complete,
        "{}: incomplete history",
        if history.as_dict().is_some() {
            history
                .get("id")
                .map_or_else(|| "None".to_owned(), Value::py_str)
        } else {
            history.py_str()
        }
    );
    let id = index(history, "id")?;
    let hid = id.py_str();
    let key = key_of(id)?;
    require!(!seen.contains(&key), "{hid}: duplicate history");
    seen.insert(key);
    let disposition = index(history, "disposition")?;
    require!(
        disposition
            .as_str()
            .is_some_and(|d| DISPOSITIONS.contains(&d)),
        "{hid}: disposition"
    );
    let disposition = disposition.as_str().expect("checked above");
    require!(
        index(history, "limitations")?.as_list().is_some(),
        "{hid}: limitations"
    );
    if let Some(retained) = history.get("retained") {
        for item in py::iterate(retained).ok_or_else(|| shape(format!("{hid}: retained")))? {
            load(root, &item, zstd)?;
        }
    }
    if disposition == REPRODUCIBLE {
        require!(
            history.get("replay").and_then(Value::as_dict).is_some(),
            "{hid}: reproducible history needs a replay"
        );
        let receipts = history.get("receipts");
        require!(
            receipts.and_then(Value::as_dict).is_some()
                && receipts.is_some_and(
                    |r| truthy_field(r, "original_execution") && truthy_field(r, "algorithm")
                ),
            "{hid}: original execution and algorithm receipts"
        );
        return Ok(());
    }
    require!(
        history.get("replay").is_none(),
        "{hid}: {disposition} carries no replay"
    );
    require!(
        history
            .get("missing")
            .and_then(Value::as_list)
            .is_some_and(|m| !m.is_empty()),
        "{hid}: {disposition} must name what is missing"
    );
    require!(
        !["ratings", "ranking", "leaderboard"]
            .iter()
            .any(|k| history.get(k).is_some()),
        "{hid}: {disposition} history cannot carry ratings or rankings"
    );
    if disposition == "scorer-parity-only" {
        require!(
            truthy_field(history, "parity"),
            "{hid}: scorer parity needs its parity test"
        );
    }
    Ok(())
}
