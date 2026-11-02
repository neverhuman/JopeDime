//! Candidate ordering helpers ported from `jope.policy.decode_ranking`.
use serde_json::Value;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

/// Final candidate ordering. The global selection-key winner always occupies slot zero;
/// the engine diversity floor may only fill later slots.
pub fn rank_candidates<F>(
    valid: &[Value],
    n: i64,
    stability_threshold: f64,
    selection_key: F,
) -> Vec<usize>
where
    F: Fn(usize) -> f64,
{
    let ranked_order = sorted_indices(valid.len(), |a, b| {
        descending(selection_key(a), selection_key(b))
    });
    let Some(&top) = ranked_order.first() else {
        return ranked_order;
    };
    let engines: BTreeSet<String> = valid.iter().map(|row| engine(row)).collect();
    let mut floor = Vec::new();
    if n >= engines.len() as i64 {
        for target_engine in engines {
            if engine(&valid[top]) == target_engine {
                continue;
            }
            if let Some(&candidate) = ranked_order.iter().find(|&&i| {
                i != top
                    && !floor.contains(&i)
                    && engine(&valid[i]) == target_engine
                    && python_float_or_zero(valid[i].get("p_bad")) <= stability_threshold
                    && !truthy(valid[i].get("candidate_risk_reason"))
            }) {
                floor.push(candidate);
            }
        }
    }
    let mut result = vec![top];
    result.extend(floor.iter().copied());
    result.extend(
        ranked_order
            .into_iter()
            .filter(|i| *i != top && !floor.contains(i)),
    );
    result
}

/// Selector 2.0 portfolio: reranker order, then uncertainty, diversity, and engine slots.
pub fn rank_candidates_with_reranker(
    valid: &[Value],
    n: i64,
    stability_threshold: f64,
) -> Vec<usize> {
    let base_order = sorted_indices(valid.len(), |a, b| {
        let a_row = &valid[a];
        let b_row = &valid[b];
        descending(
            numeric(
                a_row,
                "selector_reranker_score",
                numeric(a_row, "selection_key", f64::NEG_INFINITY),
            ),
            numeric(
                b_row,
                "selector_reranker_score",
                numeric(b_row, "selection_key", f64::NEG_INFINITY),
            ),
        )
        .then_with(|| {
            descending(
                numeric(a_row, "selection_key", f64::NEG_INFINITY),
                numeric(b_row, "selection_key", f64::NEG_INFINITY),
            )
        })
        .then_with(|| {
            descending(
                numeric(a_row, "reward", f64::NEG_INFINITY),
                numeric(b_row, "reward", f64::NEG_INFINITY),
            )
        })
        .then_with(|| proposal_index(a_row).cmp(&proposal_index(b_row)))
    });
    let limit = n.max(0) as usize;
    if limit == 0 {
        return base_order;
    }
    let safe = |i: usize| {
        !truthy(valid[i].get("candidate_risk_reason"))
            && numeric(&valid[i], "p_bad", 0.0) <= stability_threshold
    };
    let mut selected = Vec::new();
    let mut selected_set = BTreeSet::new();
    let add = |i: usize, selected: &mut Vec<usize>, selected_set: &mut BTreeSet<usize>| {
        if selected_set.insert(i) {
            selected.push(i);
        }
        selected.len() >= limit
    };

    let initial_target = 3.min(limit);
    for &i in &base_order {
        if safe(i) {
            add(i, &mut selected, &mut selected_set);
            if selected.len() >= initial_target {
                break;
            }
        }
    }
    if selected.len() >= limit {
        return append_remaining(selected, &selected_set, &base_order);
    }

    let mut uncertainty: Vec<usize> = base_order
        .iter()
        .copied()
        .filter(|i| !selected_set.contains(i) && safe(*i))
        .collect();
    uncertainty.sort_by(|&a, &b| {
        descending(
            numeric(&valid[a], "reward_std", f64::NEG_INFINITY),
            numeric(&valid[b], "reward_std", f64::NEG_INFINITY),
        )
        .then_with(|| {
            descending(
                numeric(&valid[a], "selection_key", f64::NEG_INFINITY),
                numeric(&valid[b], "selection_key", f64::NEG_INFINITY),
            )
        })
    });
    if let Some(&i) = uncertainty.first() {
        if add(i, &mut selected, &mut selected_set) {
            return append_remaining(selected, &selected_set, &base_order);
        }
    }

    let seen_diversity: BTreeSet<String> = selected
        .iter()
        .map(|&i| coalesced_string(valid[i].get("candidate_diversity_key")))
        .collect();
    if let Some(&i) = base_order.iter().find(|&&i| {
        !selected_set.contains(&i)
            && safe(i)
            && valid[i]
                .get("novel_vs_historical")
                .map_or(true, truthy_value)
            && !seen_diversity.contains(&coalesced_string(valid[i].get("candidate_diversity_key")))
    }) {
        if add(i, &mut selected, &mut selected_set) {
            return append_remaining(selected, &selected_set, &base_order);
        }
    }

    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for &i in &selected {
        *counts.entry(engine_coalesced(&valid[i])).or_default() += 1;
    }
    let engines: BTreeSet<String> = valid.iter().map(engine_coalesced).collect();
    let mut underrepresented: Vec<String> = engines.into_iter().collect();
    underrepresented.sort_by_key(|e| (counts.get(e).copied().unwrap_or(0), e.clone()));
    for target_engine in underrepresented {
        if let Some(&i) = base_order.iter().find(|&&i| {
            !selected_set.contains(&i) && safe(i) && engine_coalesced(&valid[i]) == target_engine
        }) {
            if add(i, &mut selected, &mut selected_set) {
                return append_remaining(selected, &selected_set, &base_order);
            }
        }
    }
    for &i in &base_order {
        if add(i, &mut selected, &mut selected_set) {
            break;
        }
    }
    append_remaining(selected, &selected_set, &base_order)
}

/// Compute the v3 head/family diversity label used by the Python decoder.
pub fn candidate_diversity_key_v3(
    actions: &Value,
    head_type: Option<&Value>,
    family_bitmap: Option<&Value>,
) -> String {
    let transforms = actions
        .get("transforms")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    let first = transforms
        .iter()
        .find(|stage| stage.is_object())
        .and_then(|stage| stage.get("op"))
        .map(python_string)
        .unwrap_or_else(|| "none".to_owned());
    let action_head = actions
        .get("head")
        .filter(|v| v.is_object())
        .and_then(|v| v.get("op"));
    let head = if truthy(head_type) {
        python_string(head_type.unwrap())
    } else if truthy(action_head) {
        python_string(action_head)
    } else {
        "unknown".to_owned()
    };
    let mut active: Vec<String> = family_bitmap
        .and_then(Value::as_object)
        .into_iter()
        .flat_map(|map| map.iter())
        .filter(|(_, value)| truthy_value(value))
        .map(|(key, _)| key.clone())
        .collect();
    active.sort();
    let family = active.first().cloned().unwrap_or(first);
    format!("{head}:{family}")
}

/// Numeric JSON field converted to finite f64, or `default`.
pub fn row_numeric(row: &Value, key: &str, default: f64) -> f64 {
    numeric(row, key, default)
}

fn numeric(row: &Value, key: &str, default: f64) -> f64 {
    match row.get(key) {
        Some(Value::Number(number)) => number.as_f64().filter(|v| v.is_finite()).unwrap_or(default),
        Some(Value::Bool(value)) => {
            if *value {
                1.0
            } else {
                0.0
            }
        }
        _ => default,
    }
}

fn python_float_or_zero(value: Option<&Value>) -> f64 {
    match value {
        None | Some(Value::Null) | Some(Value::Bool(false)) => 0.0,
        Some(Value::Bool(true)) => 1.0,
        Some(Value::Number(number)) => number.as_f64().unwrap_or(f64::NAN),
        Some(Value::String(value)) if value.is_empty() => 0.0,
        Some(Value::String(value)) => value.parse().unwrap_or(f64::NAN),
        Some(_) => f64::NAN,
    }
}

fn proposal_index(row: &Value) -> i128 {
    match row.get("proposal_index") {
        Some(Value::Number(number)) => number
            .as_i64()
            .map(i128::from)
            .or_else(|| number.as_u64().map(i128::from))
            .unwrap_or(1_000_000_000),
        Some(Value::Bool(value)) => i128::from(*value),
        _ => 1_000_000_000,
    }
}

fn engine(row: &Value) -> String {
    row.get("engine")
        .map(python_string)
        .unwrap_or_else(|| "<missing>".to_owned())
}

fn engine_coalesced(row: &Value) -> String {
    row.get("engine")
        .filter(|value| truthy_value(value))
        .map(python_string)
        .unwrap_or_default()
}

fn truthy(value: Option<&Value>) -> bool {
    value.is_some_and(truthy_value)
}

fn truthy_value(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::Number(value) => value.as_f64().is_some_and(|number| number != 0.0),
        Value::String(value) => !value.is_empty(),
        Value::Array(value) => !value.is_empty(),
        Value::Object(value) => !value.is_empty(),
    }
}

fn coalesced_string(value: Option<&Value>) -> String {
    value
        .filter(|v| truthy_value(v))
        .map(python_string)
        .unwrap_or_default()
}

fn python_string(value: &Value) -> String {
    match value {
        Value::Null => "None".to_owned(),
        Value::Bool(true) => "True".to_owned(),
        Value::Bool(false) => "False".to_owned(),
        Value::String(value) => value.clone(),
        other => other.to_string(),
    }
}

fn descending(a: f64, b: f64) -> Ordering {
    b.partial_cmp(&a).unwrap_or(Ordering::Equal)
}

fn sorted_indices<F>(len: usize, compare: F) -> Vec<usize>
where
    F: Fn(usize, usize) -> Ordering,
{
    let mut indices: Vec<usize> = (0..len).collect();
    indices.sort_by(compare);
    indices
}

fn append_remaining(
    selected: Vec<usize>,
    selected_set: &BTreeSet<usize>,
    base: &[usize],
) -> Vec<usize> {
    let mut result = selected;
    result.extend(base.iter().copied().filter(|i| !selected_set.contains(i)));
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn sacred_global_winner_precedes_viable_engine_floor() {
        let rows = vec![
            json!({"engine":"z", "p_bad":0.9}),
            json!({"engine":"a", "p_bad":0.0}),
            json!({"engine":"z", "p_bad":0.0}),
        ];
        assert_eq!(
            rank_candidates(&rows, 2, 0.2, |i| [9.0, 8.0, 7.0][i]),
            vec![0, 1, 2]
        );
    }

    #[test]
    fn reranker_orders_stably_and_prioritizes_safety_and_uncertainty() {
        let rows = vec![
            json!({"selector_reranker_score":3.0,"selection_key":1.0,"reward":1.0,"proposal_index":0,"p_bad":0.0,"reward_std":0.1}),
            json!({"selector_reranker_score":2.0,"selection_key":2.0,"reward":2.0,"proposal_index":1,"p_bad":0.0,"reward_std":0.8}),
            json!({"selector_reranker_score":1.0,"selection_key":3.0,"reward":3.0,"proposal_index":2,"p_bad":0.9,"reward_std":9.0}),
            json!({"selector_reranker_score":0.0,"selection_key":0.0,"reward":0.0,"proposal_index":3,"p_bad":0.0,"reward_std":0.2}),
        ];
        assert_eq!(
            rank_candidates_with_reranker(&rows, 3, 0.2),
            vec![0, 1, 3, 2]
        );
    }

    #[test]
    fn reranker_uses_uncertainty_then_novel_diversity() {
        let rows = vec![
            json!({"selection_key":4.0,"reward_std":0.1,"candidate_diversity_key":"a","engine":"x"}),
            json!({"selection_key":3.0,"reward_std":0.9,"candidate_diversity_key":"a","engine":"x"}),
            json!({"selection_key":2.0,"reward_std":0.2,"candidate_diversity_key":"b","engine":"y","novel_vs_historical":true}),
        ];
        assert_eq!(rank_candidates_with_reranker(&rows, 2, 0.2), vec![0, 1, 2]);
        assert_eq!(rank_candidates_with_reranker(&rows, 4, 0.2), vec![0, 1, 2]);
    }

    #[test]
    fn diversity_key_selects_first_active_sorted_family() {
        let actions = json!({"transforms":[null,{"op":"scale"}],"head":{"op":"linear"}});
        let families = json!({"z":true,"a":1,"off":false});
        assert_eq!(
            candidate_diversity_key_v3(&actions, None, Some(&families)),
            "linear:a"
        );
        assert_eq!(
            candidate_diversity_key_v3(&json!({}), None, None),
            "unknown:none"
        );
    }

    #[test]
    fn numeric_fields_accept_boolean_and_reject_non_finite_or_non_numeric() {
        assert_eq!(row_numeric(&json!({"x":true}), "x", -1.0), 1.0);
        assert_eq!(row_numeric(&json!({"x":"2"}), "x", -1.0), -1.0);
        assert_eq!(row_numeric(&json!({}), "x", -1.0), -1.0);
    }
}
