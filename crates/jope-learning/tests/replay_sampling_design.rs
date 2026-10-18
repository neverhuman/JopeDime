//! Bounded acceptance tests for I36.

use jope_learning::replay::sample::{
    BottomKSnapshot, InclusionProbability, ReplayDesign, ReplayRecord, ReplayRole,
};
use std::collections::{BTreeMap, BTreeSet};

fn design(capacity: usize, max_bytes: u64) -> ReplayDesign {
    ReplayDesign {
        seed: b"frozen-seed-v1".to_vec(),
        capacity,
        max_bytes,
    }
}

fn row(id: &str, role: ReplayRole, payload_bytes: u64, parents: &[&str]) -> ReplayRecord {
    ReplayRecord {
        id: id.to_owned(),
        role,
        payload_bytes,
        parents: parents
            .iter()
            .map(|p| (*p).to_owned())
            .collect::<BTreeSet<_>>(),
    }
}

fn ids(snapshot: &BottomKSnapshot) -> Vec<String> {
    snapshot.selected_ids.clone()
}

#[test]
fn restart_and_disjoint_merge_reproduce_exact_selection_and_byte_cap() {
    let all = (0..37)
        .map(|i| row(&format!("r{i:02}"), ReplayRole::General, 7, &[]))
        .collect::<Vec<_>>();
    let restarted = BottomKSnapshot::build(design(9, 63), "whole", all.clone()).unwrap();
    let restarted_again = BottomKSnapshot::build(design(9, 63), "whole", all.clone()).unwrap();
    assert_eq!(ids(&restarted), ids(&restarted_again));
    let persisted = serde_json::to_vec(&restarted).unwrap();
    let restored: BottomKSnapshot = serde_json::from_slice(&persisted).unwrap();
    assert_eq!(restored, restarted);
    assert_eq!(restarted.selected_bytes, 63);

    let left = BottomKSnapshot::build(design(9, 63), "left", all[..18].to_vec()).unwrap();
    let right = BottomKSnapshot::build(design(9, 63), "right", all[18..].to_vec()).unwrap();
    let merged = left.merge(&right).unwrap();
    assert_eq!(ids(&restarted), ids(&merged));
    assert_eq!(merged.selected_bytes, 63);
    assert_eq!(merged.population_count, 37);
    assert!(merged.selected_bytes <= merged.design.max_bytes);
    assert!(left.merge(&left).is_err());
}

#[test]
fn adaptive_reused_threshold_does_not_create_known_unconditional_pi_or_ht_claim() {
    // Counterexample: let rank U be uniform on [0,1), and reuse the same U to set adaptive
    // threshold T=U. Selection is U<T, so its actual inclusion probability is 0. If T were an
    // independent uniform threshold, the same rule would have probability 1/2. Reusing the
    // model-dependent hash rank therefore invalidates the apparent independent-threshold pi.
    let reused_rank_decisions = (0..100)
        .map(|i| {
            let u = i as f64 / 100.0;
            let adaptive_threshold = u;
            u < adaptive_threshold
        })
        .collect::<Vec<_>>();
    assert!(reused_rank_decisions.iter().all(|selected| !selected));
    let selected = BottomKSnapshot::build(
        design(1, 10),
        "adaptive-example",
        vec![
            row("x", ReplayRole::General, 1, &[]),
            row("y", ReplayRole::General, 1, &[]),
        ],
    )
    .unwrap();
    assert_eq!(
        selected.inclusion_probability("x"),
        InclusionProbability::Known {
            numerator: 1,
            denominator: 2
        }
    );
    assert_eq!(
        selected.inclusion_probability("not-in-snapshot"),
        InclusionProbability::Unknown
    );
    // That probability belongs to the declared fixed bottom-k design only. The adaptive
    // threshold has no API path into it and its contribution cannot be relabeled as an HT total.
    let mut adaptive_contribution = BTreeMap::new();
    adaptive_contribution.insert("adaptive-x".to_owned(), 1.0);
    assert!(selected.ht_total(&adaptive_contribution).is_err());
}

#[test]
fn protected_failures_parents_and_exposure_ancestry_survive_training_exclusion() {
    let population = vec![
        row("accepted-parent", ReplayRole::General, 2, &[]),
        row(
            "rare-failure",
            ReplayRole::RareFailure,
            2,
            &["accepted-parent"],
        ),
        row(
            "accepted-exposure",
            ReplayRole::AcceptedExposure,
            2,
            &["accepted-parent"],
        ),
        row("ordinary-a", ReplayRole::General, 2, &[]),
        row("ordinary-b", ReplayRole::General, 2, &[]),
    ];
    let snapshot = BottomKSnapshot::build(design(3, 6), "protected", population).unwrap();
    for id in ["accepted-parent", "rare-failure", "accepted-exposure"] {
        assert!(snapshot.selected_ids.iter().any(|selected| selected == id));
    }
    assert_eq!(snapshot.records.len(), 3);
    assert_eq!(snapshot.selected_bytes, 6);

    let missing_ancestry = vec![row(
        "rare-failure",
        ReplayRole::RareFailure,
        2,
        &["missing-parent"],
    )];
    assert!(BottomKSnapshot::build(design(3, 10), "missing", missing_ancestry).is_err());
    assert!(BottomKSnapshot::build(
        design(2, 3),
        "too-large-protected",
        vec![
            row("p", ReplayRole::AcceptedParent, 2, &[]),
            row("f", ReplayRole::RareFailure, 2, &["p"]),
        ]
    )
    .is_err());
}

#[test]
fn probability_and_estimator_contracts_fail_closed() {
    let census = BottomKSnapshot::build(
        design(4, 40),
        "census",
        vec![
            row("a", ReplayRole::General, 2, &[]),
            row("b", ReplayRole::General, 2, &[]),
        ],
    )
    .unwrap();
    assert_eq!(
        census.inclusion_probability("a"),
        InclusionProbability::Known {
            numerator: 1,
            denominator: 1
        }
    );
    let contributions = BTreeMap::from([("a".to_owned(), 4.0), ("b".to_owned(), 6.0)]);
    assert_eq!(census.ht_total(&contributions).unwrap(), 10.0);
    assert_eq!(
        census
            .ratio_mean(
                &contributions,
                &BTreeMap::from([("a".to_owned(), 1.0), ("b".to_owned(), 1.0)])
            )
            .unwrap(),
        5.0
    );
    assert!(census
        .ratio_mean(&contributions, &BTreeMap::from([("a".to_owned(), 1.0)]))
        .is_err());
    assert!(census
        .ratio_mean(
            &contributions,
            &BTreeMap::from([("a".to_owned(), 0.0), ("b".to_owned(), 0.0)])
        )
        .is_err());
}
