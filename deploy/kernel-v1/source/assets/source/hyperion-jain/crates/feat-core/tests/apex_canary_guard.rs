//! apex_canary_guard — the CANARY quarantine invariant.
//!
//! A GP survivor whose expression references a CANARY negative-control column must be removed BEFORE
//! feature reduction, so 0 canary-derived features can reach the final selection. This exercises the
//! quarantine in `build_augmented_from_search` directly and deterministically: one clean survivor is
//! kept, one canary-referencing survivor is dropped. If a change re-admits canary survivors into the
//! model matrix, this test fails `cargo test -p feat-core`.
use feat_core::buildmatrix::{build_augmented_from_search, FeatureInfo};
use feat_core::canary::{canary_search_col, CanaryProfile, CanarySpec};
use feat_core::data::{Dataset, Matrix};
use feat_core::dsl::Expr;
use feat_core::gp::SurvivorRec;

/// Column-major dataset with `cols` distinct, finite columns.
fn ds(cols: usize, n: usize) -> Dataset {
    let mut data = vec![0f32; cols * n];
    for j in 0..cols {
        for (i, v) in data[j * n..(j + 1) * n].iter_mut().enumerate() {
            *v = (j * 10 + i + 1) as f32;
        }
    }
    Dataset {
        x: Matrix::new(data, n, cols),
        y: vec![0.0; n],
        y_multi: None,
        names: (0..cols).map(|j| format!("c{j}")).collect(),
    }
}

fn surv(expr: Expr, formula: &str) -> SurvivorRec {
    SurvivorRec {
        expr,
        formula: formula.into(),
        signal: 0.5,
        proxy_gain: 0.5,
        lin_corr: 0.1,
        maxcorr_library: 0.1,
        target_corr: 0.2,
        depth: 1,
        nodes: 1,
    }
}

fn base_feature() -> FeatureInfo {
    FeatureInfo {
        source: "base".into(),
        formula: "c0".into(),
        expr: Expr::Col(0),
        fill: 0.0,
        scale_min: 0.0,
        scale_span: 1.0,
        canary: None,
        canary_refs: Vec::new(),
    }
}

#[test]
fn canary_referencing_survivors_are_quarantined_before_feature_reduction() {
    let n = 6;
    let visible = ds(1, n); // one visible base column
    let search = ds(3, n); // base0, base1, and a canary column at search index 2

    // canary search col = search_width.max(source_index + 1) + canary_index = 2.max(1) + 0 = 2
    let spec = CanarySpec {
        version: 1,
        name: "CANARY_FEATURE_0000__c0".into(),
        source_name: "c0".into(),
        source_index: 0,
        canary_index: 0,
        search_width: 2,
        seed: 1,
        profile: CanaryProfile::Continuous {
            missing_rate: 0.0,
            quantiles: vec![0.0, 1.0],
        },
    };
    assert_eq!(canary_search_col(&spec), 2);

    let survivors = vec![
        surv(Expr::Col(0), "clean_uses_base"), // references base col 0 → kept
        surv(Expr::Col(2), "leans_on_canary"), // references canary col 2 → purged
    ];
    let aug = build_augmented_from_search(
        &visible,
        &visible,
        &search,
        &search,
        &survivors,
        10,
        &[base_feature()],
        &[spec],
    );

    let formulas: Vec<&str> = aug.features.iter().map(|f| f.formula.as_str()).collect();
    assert!(
        formulas.contains(&"clean_uses_base"),
        "a clean GP survivor must be kept: {formulas:?}"
    );
    assert!(
        !formulas.contains(&"leans_on_canary"),
        "a canary-referencing survivor MUST be quarantined before feature reduction — \
         re-admitting it would let a negative-control leak into selection: {formulas:?}"
    );
    // base (1) + clean GP feature (1); the canary one is dropped, and no retained feature refs a canary.
    assert_eq!(aug.features.len(), 2);
    assert_eq!(aug.train.c, 2);
    assert!(aug.features.iter().all(|f| f.canary_refs.is_empty()));
}
