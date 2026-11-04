use super::*;

#[test]
fn positive_controls() {
    let strong = board(&[
        ("a", "strong", "1.0"),
        ("a", "weak", "2.0"),
        ("b", "strong", "2.0"),
        ("b", "weak", "3.0"),
    ]);
    assert!(strong.ratings["strong"] > strong.ratings["weak"]);
    let mean = strong.ratings.values().sum::<f64>() / 2.0;
    assert_eq!(py::round(mean - 1500.0, 9), 0.0);
    let tied = board(&[
        ("a", "left", "1.0"),
        ("a", "right", &pyjson::float_repr(1.0 + 1e-13)),
    ]);
    assert_eq!(
        tied.ratings
            .iter()
            .map(|(k, v)| (k.as_str(), *v))
            .collect::<Vec<_>>(),
        [("left", 1500.0), ("right", 1500.0)]
    );
    let alone = board(&[("a", "solo", "1.0"), ("b", "x", "1.0"), ("b", "y", "2.0")]);
    assert!(!alone.ratings.contains_key("solo"));
    let flipped = board_with(&[("a", "strong", "1.0"), ("a", "weak", "2.0")], false);
    assert!(flipped.ratings["weak"] > flipped.ratings["strong"]);
}

#[test]
fn equal_counts_and_totals_are_exchangeable_despite_different_records() {
    let result = board(&[
        ("d1", "a", "1"),
        ("d1", "c", "2"),
        ("d1", "b", "3"),
        ("d1", "x", "4"),
        ("d2", "b", "1"),
        ("d2", "x", "2"),
        ("d2", "a", "3"),
        ("d2", "c", "4"),
    ]);
    assert_eq!(result.exchangeable, groups(&[&["a", "b"], &["c", "x"]]));
    assert_ne!(result.pairs.get("a", "c"), result.pairs.get("b", "c"));
    assert_eq!(py::round(result.ratings["a"] - result.ratings["b"], 9), 0.0);
}

#[test]
fn disconnected_groups_are_reported() {
    let result = board(&[
        ("d1", "a", "1"),
        ("d1", "b", "2"),
        ("d2", "c", "1"),
        ("d2", "d", "2"),
    ]);
    assert_eq!(result.components, [2, 2]);
}

#[test]
fn identical_score_vectors_are_reported_as_possible_aliases() {
    let result = board(&[
        ("d1", "a", "1"),
        ("d1", "a2", "1"),
        ("d1", "b", "2"),
        ("d2", "a", "3"),
        ("d2", "a2", "3"),
        ("d2", "b", "1"),
    ]);
    assert_eq!(result.identical_scores, groups(&[&["a", "a2"]]));
}

#[test]
fn duplicate_delivery_cannot_add_or_replace_a_match() {
    let rows = [("d1", "a", "1.0"), ("d1", "b", "2.0")];
    for duplicate in [("d1", "a", "1.0"), ("d1", "a", "3.0")] {
        let mut all = rows.to_vec();
        all.push(duplicate);
        assert_refused(parse_scores(&scores_csv(&all), "t"), "duplicate delivery");
    }
    let missing = scores_csv(&rows) + &"t,d1,c,same-cut,rmse,,missing,True,k,r,,,\n".repeat(2);
    assert_eq!(parse_scores(&missing, "t").unwrap().len(), 4); // unrankable rows add no match
                                                               // Addition: the message names the cell as Python's tuple repr did.
    let mut all = rows.to_vec();
    all.push(("d1", "a", "3.0"));
    assert_refused(
        parse_scores(&scores_csv(&all), "t"),
        r"^scores\.csv line 4: duplicate delivery of \('d1', 'a'\) cannot add or replace a match$",
    );
}

#[test]
fn replay_is_deterministic() {
    let document = record();
    let spec = document
        .get("histories")
        .and_then(Value::as_list)
        .unwrap()
        .iter()
        .find(|h| h.get("id") == Some(&Value::from("H-same-cut-v2-screen")))
        .and_then(|h| h.get("replay"))
        .unwrap()
        .clone();
    let first = elo_history::replay(&repo(), &spec, Zstd::Builtin).unwrap();
    let second = elo_history::replay(&repo(), &spec, Zstd::Builtin).unwrap();
    assert_eq!(first.ladder.csv, second.ladder.csv);
    let bits = |l: &Ladder| {
        l.ratings
            .iter()
            .map(|(m, r)| (m.clone(), r.to_bits()))
            .collect::<Vec<_>>()
    };
    assert_eq!(bits(&first.ladder), bits(&second.ladder));
}
