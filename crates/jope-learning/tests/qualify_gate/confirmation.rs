use super::*;

#[test]
fn an_arena_must_be_a_usable_predeclared_confirmation_arena() {
    let cases: Vec<(&str, Mutation)> = vec![
        (
            "challenger is the incumbent",
            Box::new(|m| m["challenger"]["sha256"] = json!(sha(1))),
        ),
        (
            "adopted sealed roster",
            Box::new(|m| m["roster_role"] = json!("sealed_final")),
        ),
        (
            "not independent confirmation",
            Box::new(|m| m["roster_role"] = json!("development")),
        ),
        (
            "not independent confirmation",
            Box::new(|m| m["roster_role"] = json!("meta_training")),
        ),
        (
            "more than alpha_total",
            Box::new(|m| m["looks"][1]["alpha"] = json!(0.045)),
        ),
        ("numbered", Box::new(|m| m["looks"][1]["index"] = json!(3))),
        ("min_families", Box::new(|m| m["min_families"] = json!(13))),
        ("min_families", Box::new(|m| m["min_families"] = json!(1))),
        (
            "family_weighting",
            Box::new(|m| m["family_weighting"] = json!("context")),
        ),
        (
            "missing field `family_weighting`",
            Box::new(|m| {
                m.as_object_mut().unwrap().remove("family_weighting");
            }),
        ),
        (
            "leaves fewer than min_families",
            Box::new(|m| m["max_incomplete_families"] = json!(5)),
        ),
        ("unique", Box::new(|m| m["families"][1] = json!("fam-01"))),
        (
            "resamples",
            Box::new(|m| m["bootstrap"]["resamples"] = json!(999)),
        ),
        ("finer than", Box::new(|m| m["contenders"] = json!(50))),
        (
            "practical_margin is negative",
            Box::new(|m| m["practical_margin"] = json!(-0.1)),
        ),
        (
            "lowercase hex sha256",
            Box::new(|m| m["incumbent"]["sha256"] = json!("ea078456")),
        ),
        (
            "max_regression is negative",
            Box::new(|m| m["safety"]["max_regression"] = json!(-1)),
        ),
        (
            "manifest schema",
            Box::new(|m| m["schema"] = json!("jope.other/v1")),
        ),
    ];
    for (expected, mutate) in cases {
        let mut document = manifest();
        mutate(&mut document);
        let manifest = bytes(&document);
        let results = bytes(&results(&manifest, rows(clear_lift)));
        let error = promotion::decide(&manifest, &results).unwrap_err();
        assert!(error.contains(expected), "{expected}: {error}");
    }
}

#[test]
fn contenders_split_the_alpha_of_a_look() {
    let mut shared = manifest();
    shared["contenders"] = json!(4);
    let shared = bytes(&shared);
    let report = promotion::decide(&shared, &bytes(&results(&shared, rows(clear_lift)))).unwrap();
    assert!((report.alpha_per_comparison - 0.0025).abs() < 1e-15);
}

#[test]
fn the_bootstrap_replays_from_its_seed() {
    let manifest = bytes(&manifest());
    let noisy = rows(|family, context| {
        if (family + context) % 3 == 0 {
            0.04
        } else {
            -0.01
        }
    });
    let results = bytes(&results(&manifest, noisy));
    let first = promotion::decide(&manifest, &results).unwrap();
    let second = promotion::decide(&manifest, &results).unwrap();
    assert_eq!(first, second);
    assert_eq!(
        serde_json::to_string(&first).unwrap(),
        serde_json::to_string(&second).unwrap()
    );
}
