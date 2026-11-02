use super::*;

#[test]
fn claims_are_refused_outside_reproducible_histories_and_historical_consumers() {
    let document = record();
    let granted = authorize(&document, "H-same-cut-v2-deep", "ranking", CONSUMER).unwrap();
    assert_eq!(
        granted.get("consumer"),
        Some(&Value::from("historical-ranking-only"))
    );
    let refusals = [
        (
            ("H-same-cut-v2-deep", "ranking", "hwm"),
            "hwm consumers are refused",
        ),
        (
            ("H-same-cut-v2-deep", "rating", "teaching"),
            "teaching consumers are refused",
        ),
        (
            ("H-same-cut-v2-deep", "rating", "transfer"),
            "transfer consumers are refused",
        ),
        (
            ("H-same-cut-v2-deep", "ranking", "promotion"),
            "promotion consumers are refused",
        ),
        (
            ("H-same-cut-v2-deep", "continuity", CONSUMER),
            "continuity across ladders",
        ),
        (
            ("H-legacy-round1-internal-elo", "ranking", CONSUMER),
            "irrecoverable history supports no ranking",
        ),
        (
            ("H-sotarena-golden42-v1", "rating", CONSUMER),
            "irrecoverable history supports no rating",
        ),
        (
            ("H-campaign-redteam-bradley-terry", "ranking", CONSUMER),
            "scorer-parity-only history supports",
        ),
        (("H-unknown", "ranking", CONSUMER), "unknown history"),
    ];
    for ((history, claim, consumer), reason) in refusals {
        assert_refused(authorize(&document, history, claim, consumer), reason);
    }
    // Addition: an unknown consumer or claim is refused by name.
    assert_refused(
        authorize(&document, "H-same-cut-v2-deep", "ranking", "press"),
        "^H-same-cut-v2-deep: unknown consumer 'press'$",
    );
    assert_refused(
        authorize(&document, "H-same-cut-v2-deep", "trend", CONSUMER),
        "^H-same-cut-v2-deep: unknown claim 'trend'$",
    );
}

type Mutate = fn(&mut Value);

#[test]
fn record_cannot_upgrade_a_disposition() {
    const LEGACY: &str = "H-legacy-round1-internal-elo";
    const PARITY: &str = "H-campaign-redteam-bradley-terry";
    let mutations: [(&str, Mutate, &str); 11] = [
        (
            "irrecoverable relabelled reproducible",
            |d| {
                set(
                    history(d, LEGACY),
                    "disposition",
                    elo_history::REPRODUCIBLE.into(),
                )
            },
            "needs a replay",
        ),
        (
            "irrecoverable with a replay",
            |d| {
                let replay = history(d, "H-same-cut-v1-deep")
                    .get("replay")
                    .unwrap()
                    .clone();
                set(history(d, LEGACY), "replay", replay);
            },
            "carries no replay",
        ),
        (
            "irrecoverable with ratings",
            |d| set(history(d, LEGACY), "ratings", json(r#"{"gen30": 1119.2}"#)),
            "cannot carry ratings",
        ),
        (
            "irrecoverable without what is missing",
            |d| set(history(d, LEGACY), "missing", Value::List(vec![])),
            "must name what is missing",
        ),
        (
            "parity without its test",
            |d| drop(pop(history(d, PARITY), "parity")),
            "parity test",
        ),
        (
            "parity with a ranking",
            |d| set(history(d, PARITY), "ranking", strs(&["a"])),
            "cannot carry",
        ),
        (
            "reproducible without receipts",
            |d| drop(pop(history(d, "H-same-cut-v1-deep"), "receipts")),
            "receipts",
        ),
        (
            "unknown disposition",
            |d| set(history(d, LEGACY), "disposition", "recovered".into()),
            "disposition",
        ),
        (
            "duplicate history",
            |d| {
                let copy = history(d, LEGACY).clone();
                at(d, "histories").as_list_mut().unwrap().push(copy);
            },
            "duplicate history",
        ),
        (
            "no limitations",
            |d| set(history(d, LEGACY), "limitations", Value::List(vec![])),
            "incomplete history",
        ),
        (
            "altered retained observation",
            |d| {
                set(
                    nth(at(history(d, LEGACY), "retained"), 0),
                    "sha256",
                    "1".repeat(64).into(),
                )
            },
            "digest mismatch",
        ),
    ];
    for (label, mutate, reason) in mutations {
        let mut document = fast(record());
        mutate(&mut document);
        let result = Copy::new(document).verify().map(|_| ());
        assert!(result.is_err(), "{label}: accepted");
        assert_refused(result, reason);
    }
}
