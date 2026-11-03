use super::*;
use jope_learning::retention::distilled as distilled_api;

#[test]
fn distilled_table_round_trips_and_reconciles() {
    let mut l6 = row('c', 'b', Some(0.8961), Outcome::Ok);
    l6.score_name = Some("l6".into());
    l6.l6_delta = Some(0.8961 - 0.897327);
    l6.qualified = true;
    l6.split_role = SplitRole::Val263;
    let mut failed = row('d', 'a', None, Outcome::Timeout);
    failed.program = None;
    failed.rank_ctx = None;
    let rows = vec![
        row('b', 'a', Some(0.61), Outcome::Ok),
        row('a', 'a', Some(0.59), Outcome::Ok),
        l6,
        failed,
    ];
    let bytes = distilled_api::write(
        &rows,
        &known(),
        &[("source_manifest".into(), "fixture".into())],
    )
    .unwrap();
    let back = distilled_api::read(&bytes, &known()).unwrap();
    let mut expected = rows.clone();
    expected.sort_by(|x, y| x.program_sha.cmp(&y.program_sha));
    assert_eq!(back, expected, "rows come back sorted by identity");
    assert_eq!(
        distilled_api::write(
            &rows,
            &known(),
            &[("source_manifest".into(), "fixture".into())]
        )
        .unwrap(),
        bytes,
        "deterministic bytes"
    );
    let keys: BTreeSet<String> = rows.iter().map(|r| r.source_key.clone()).collect();
    distilled_api::reconcile(&back, &keys).unwrap();
    let mut more = keys.clone();
    more.insert("experiments/gp-teacher-v6/attempts/zz.json".into());
    assert!(distilled_api::reconcile(&back, &more)
        .unwrap_err()
        .to_string()
        .contains("missing"));
}

#[test]
fn distilled_rows_that_would_corrupt_meaning_are_refused() {
    let k = known();
    let mut r = row('a', 'a', Some(0.5), Outcome::Ok);
    r.score_name = Some("l7_mystery".into());
    assert!(
        distilled_api::validate(&r, &k).is_err(),
        "unknown score identity"
    );
    let mut r = row('a', 'a', Some(0.5), Outcome::Ok);
    r.qualified = true;
    assert!(
        distilled_api::validate(&r, &k).is_err(),
        "qualified without a paired L6 eval"
    );
    let r = row('a', 'a', Some(0.5), Outcome::Failed);
    assert!(
        distilled_api::validate(&r, &k).is_err(),
        "scored row with a failure outcome"
    );
    let mut r = row('a', 'a', Some(f64::NAN), Outcome::Ok);
    r.score = Some(f64::NAN);
    assert!(distilled_api::validate(&r, &k).is_err(), "non-finite score");
    let mut r = row('a', 'a', None, Outcome::Failed);
    r.superseded_by = Some("champ".into());
    assert!(
        distilled_api::validate(&r, &k).is_err(),
        "superseded_by without champion_id"
    );
    let dup = vec![
        row('a', 'a', Some(0.5), Outcome::Ok),
        row('a', 'a', Some(0.5), Outcome::Ok),
    ];
    assert!(distilled_api::write(&dup, &k, &[])
        .unwrap_err()
        .to_string()
        .contains("duplicate"));
}
