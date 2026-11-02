use jope_learning::pools::evaluate::fold_of;
use jope_learning::pools::pool::{Member, Pool, Status, POOL_SCHEMA};
use jope_learning::pools::selectors::{
    rank_targets, ContextualSurrogateQHat, CvEstimate, CvUnavailable, CvUnavailableReason,
    ExternalProgramPrior, Features, Selector,
};
use std::collections::BTreeMap;

fn fixture(n: usize) -> (Vec<Pool>, Features, Features) {
    let pools: Vec<_> = (0..n)
        .map(|i| Pool {
            schema: POOL_SCHEMA.into(),
            context_id: format!("{i:064x}"),
            parent_id: format!("parent-{i}"),
            role: "train".into(),
            protocol_sha256: "ab".repeat(32),
            incumbent_q: Some(0.0),
            members: (0..3)
                .map(|j| Member {
                    program_sha256: format!("{j:064x}"),
                    status: Status::Complete,
                    q: Some((i % 2 == j % 2) as u8 as f64),
                    tokens: Some(vec![j as u32]),
                })
                .collect(),
        })
        .collect();
    let cf = Features::context_id_descriptors(&pools, 1).unwrap();
    let pf = Features {
        dim: 1,
        rows: BTreeMap::from_iter((0..3).map(|j| (format!("{j:064x}"), vec![j as f64]))),
    };
    (pools, cf, pf)
}

fn record_evidence(name: &str, value: &impl serde::Serialize) {
    if let Some(path) = std::env::var_os("I03_EVIDENCE_DIR") {
        let path = std::path::PathBuf::from(path);
        std::fs::create_dir_all(&path).unwrap();
        std::fs::write(
            path.join(format!("{name}.json")),
            serde_json::to_vec_pretty(value).unwrap(),
        )
        .unwrap();
    }
}

fn expect_unavailable(
    pools: &[Pool],
    cf: &Features,
    pf: &Features,
    lambda: f64,
    reason: CvUnavailableReason,
) -> CvEstimate {
    let error = match ContextualSurrogateQHat::fit(
        &pools.iter().collect::<Vec<_>>(),
        cf,
        pf,
        2,
        2.0,
        lambda,
    ) {
        Ok(_) => panic!("expected unavailable {reason:?}"),
        Err(error) => error,
    };
    let estimate = error
        .downcast_ref::<CvUnavailable>()
        .expect("typed unavailable")
        .0
        .clone();
    let CvEstimate::Unavailable {
        reason: actual,
        counts,
        ..
    } = &estimate
    else {
        panic!("expected missing metric")
    };
    assert_eq!(*actual, reason, "{error}");
    assert_eq!(
        counts.attempted_units,
        counts.scored_units + counts.failed_units + counts.excluded_units
    );
    let serialized = serde_json::to_value(&estimate).unwrap();
    assert!(serialized.get("metric").is_none());
    assert!(estimate.loss().is_none());
    record_evidence(
        &format!(
            "unavailable-{reason:?}-{}",
            jope_learning::canon::canonical_sha256(&estimate).unwrap()
        ),
        &estimate,
    );
    estimate
}

include!("pools_cv_unavailable/folds.rs");
include!("pools_cv_unavailable/failures.rs");
include!("pools_cv_unavailable/support.rs");
