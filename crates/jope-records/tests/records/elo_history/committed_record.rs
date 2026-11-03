use super::*;

#[test]
fn every_reproducible_history_is_replayed_or_explicitly_not_here() {
    let summary = committed();
    let reproducible = sorted_ids(
        record()
            .get("histories")
            .and_then(Value::as_list)
            .unwrap()
            .iter()
            .filter(|h| h.get("disposition") == Some(&Value::from(elo_history::REPRODUCIBLE)))
            .map(|h| h.get("id").unwrap().py_str()),
    );
    let replayed = summary.replayed.clone();
    let missing: Vec<String> = summary.not_replayed_here.keys().cloned().collect();
    assert_eq!(
        sorted_ids(replayed.iter().chain(&missing).cloned()),
        reproducible
    );
    assert!(replayed.iter().all(|id| !missing.contains(id)));
    assert!(FAST.iter().all(|id| replayed.iter().any(|r| r == id)));
    // The decoder is built in, so every reproducible history replays on every host.
    assert!(missing.is_empty());
}

#[test]
fn only_replayed_histories_enter_exact_totals() {
    let summary = committed();
    let every = sorted_ids(
        record()
            .get("histories")
            .and_then(Value::as_list)
            .unwrap()
            .iter()
            .map(|h| h.get("id").unwrap().py_str()),
    );
    let excluded: Vec<String> = every
        .into_iter()
        .filter(|id| !summary.replayed.contains(id))
        .collect();
    assert_eq!(summary.excluded_from_exact_totals, excluded);
    let json = summary.to_json();
    let totals = json.get("exact_totals").unwrap();
    assert_eq!(
        totals.get("histories"),
        Some(&Value::from(summary.replays.len()))
    );
    assert_eq!(
        totals.get("rankable_rows"),
        Some(&Value::from(
            summary
                .replays
                .values()
                .map(|r| r.rankable_rows)
                .sum::<usize>()
        ))
    );
    for id in summary
        .irrecoverable
        .iter()
        .chain(&summary.scorer_parity_only)
    {
        assert!(!summary.replays.contains_key(id), "{id}");
    }
}

#[test]
fn committed_order_of_exchangeable_methods_is_reported_not_trusted() {
    let summary = committed();
    let replays = &summary.replays;
    assert!(replays["H-same-cut-v1-deep"].byte_identical);
    let screen = &replays["H-same-cut-v1-screen"];
    assert!(!screen.byte_identical);
    let pair = groups(&[&["baseline.elasticnet", "baseline.ridge"]]);
    assert_eq!(screen.order_differs_from_replay, pair);
    assert!(screen.exchangeable_groups.contains(&pair[0]));
    let published = &replays["H-sotarena-golden45-published"];
    assert_eq!(
        published.order_differs_from_replay,
        groups(&[&["ngboost", "xgboost_hist_trees900_depth4_lr0p02"]])
    );
    assert!(!published.converged); // the retained ratings are the 250th iterate
    assert_eq!(published.iterations, ITERATIONS);
    assert!(replays["H-sotarena-golden45-benchmark-valid"].byte_identical);
}

#[test]
fn every_rating_is_historical_ranking_only() {
    let json = committed().to_json();
    assert_eq!(
        json.get("consumer"),
        Some(&Value::from("historical-ranking-only"))
    );
    assert_eq!(
        json.get("refused_consumers").map(Value::repr).as_deref(),
        Some("['teaching', 'hwm', 'transfer', 'promotion']")
    );
    let replays = json.get("replays").and_then(Value::as_dict).unwrap();
    assert!(replays
        .values()
        .all(|r| r.get("consumer") == Some(&Value::from("historical-ranking-only"))));
}

#[test]
fn lineage_edges_and_derived_view() {
    let summary = committed();
    let count = |lineage: &str, field: &str| {
        summary.lineages[lineage]
            .get(field)
            .and_then(Value::as_int)
            .unwrap()
    };
    let a100 = "L-lineage-b-a100-recovery";
    assert_eq!(
        (
            count(a100, "checkpoints"),
            count(a100, "retained_parent_edges"),
            count(a100, "unretained_parent_edges")
        ),
        (78, 74, 4)
    );
    let audit = "L-native-audit-108-blocks";
    assert_eq!(
        (
            count(audit, "checkpoints"),
            count(audit, "retained_parent_edges")
        ),
        (108, 73)
    );
    assert_eq!(
        summary
            .derived_view_rows
            .iter()
            .map(|(k, v)| (k.as_str(), *v))
            .collect::<Vec<_>>(),
        [("V-plot-historical-elo", 661)]
    );
}

#[test]
fn without_a_zstd_decoder_golden45_is_not_replayed_and_require_all_fails() {
    // Addition: the Python replay's behaviour on a host without `zstd`, recorded in the
    // slice-2 baseline (`elo_history_nozstd.out`): the Golden45 ladders are listed, never
    // counted, and `--require-all` refuses.
    // Keep only the six reproducible histories involved in this assertion. Other historical
    // dispositions include retained files that are intentionally no longer present in-tree.
    let mut document = record();
    at(&mut document, "histories")
        .as_list_mut()
        .expect("histories")
        .retain(|history| {
            let id = history.get("id").map(Value::py_str).unwrap_or_default();
            FAST.contains(&id.as_str()) || GOLDEN45.contains(&id.as_str())
        });
    let copy = Copy::new(document);
    dump(&copy.path, &copy.document);
    let summary = elo_history::verify(&copy.path, &copy.root, Zstd::Unavailable)
        .expect("verifies without zstd");
    assert_eq!(summary.replayed, FAST);
    let reason = "assets/regression/sotarena-tabular-regression-v2/results/tabular-regression-v2/\
                      scores.csv.zst: no zstd decoder on this host";
    assert_eq!(
        summary
            .not_replayed_here
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect::<Vec<_>>(),
        GOLDEN45.map(|id| (id, reason))
    );
    let totals = summary.to_json();
    assert_eq!(
        totals
            .get("exact_totals")
            .map(|t| pyjson::dumps(t, true, None))
            .as_deref(),
        Some(r#"{"histories": 4, "pairwise_matches": 1800, "rankable_rows": 360}"#)
    );
    assert!(GOLDEN45
        .iter()
        .all(|id| summary.excluded_from_exact_totals.iter().any(|e| e == id)));
    assert_eq!(summary.derived_view_rows["V-plot-historical-elo"], 661);
    assert_refused(
        summary.require_all(),
        r"^not replayed here: \['H-sotarena-golden45-benchmark-valid', 'H-sotarena-golden45-published'\]$",
    );
    copy
        .verify()
        .expect("fixture verifies with the built-in decoder")
        .require_all()
        .expect("the retained histories replay with the built-in decoder");
}
