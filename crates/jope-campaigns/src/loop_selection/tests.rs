use super::{model_specs, select_candidates};
use serde_json::{json, Value};

fn candidate(mu: f64, sigma: f64, engine: &str, head: &str) -> Value {
    json!({"mu":mu,"sigma":sigma,"engine":engine,"head_type":head})
}

#[test]
fn stored_grammar_specs_win_and_empty_specs_fall_back() {
    let stored = vec!["checkpoint-spec"];
    assert_eq!(
        model_specs(Some(&stored), || vec!["repository-spec"]),
        stored
    );
    assert_eq!(
        model_specs(Some(&[] as &[&str]), || vec!["repository-spec"]),
        vec!["repository-spec"]
    );
    assert_eq!(
        model_specs(None::<&[&str]>, || vec!["repository-spec"]),
        vec!["repository-spec"]
    );
}

#[test]
fn exploit_selection_keeps_mu_order_and_engine_head_diversity() {
    let pool = vec![
        candidate(0.9, 0.1, "a", "linear"),
        candidate(0.8, 0.2, "a", "linear"),
        candidate(0.7, 0.3, "a", "tree"),
        candidate(0.6, 0.4, "b", "linear"),
    ];
    let selected = select_candidates(&pool, 3, 0).unwrap();
    assert_eq!(selected[0], pool[0]);
    assert_eq!(selected[1], pool[2]);
    assert!(selected.iter().any(|row| row["engine"] == "b"));
}

#[test]
fn exploration_selects_uncertain_candidates_in_stable_descending_order() {
    let pool = vec![
        candidate(0.9, 0.1, "a", "linear"),
        candidate(0.8, 0.7, "a", "tree"),
        candidate(0.7, 0.7, "b", "linear"),
        candidate(0.6, 0.2, "b", "tree"),
    ];
    let selected = select_candidates(&pool, 1, 2).unwrap();
    assert_eq!(selected[0], pool[0]);
    // The engine floor is applied before uncertainty exploration, matching Python.
    assert_eq!(selected[1], pool[2]);
    assert_eq!(selected[2], pool[1]);
}

#[test]
fn selection_rejects_missing_or_non_numeric_rank_fields() {
    assert!(select_candidates(&[json!({"sigma":0.1,"engine":"a"})], 1, 0).is_err());
    assert!(select_candidates(&[json!({"mu":"high","sigma":0.1,"engine":"a"})], 1, 0).is_err());
}
