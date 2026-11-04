use super::*;

#[test]
fn write_tags_enforce_retention_invariants() {
    let t = tags();
    let pairs: Vec<(&str, &str)> = t
        .pairs()
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect();
    assert_eq!(
        pairs,
        vec![
            ("tier", "T2"),
            ("run_id", "run-20261013-a"),
            ("family", "gp-teacher-v6"),
            ("qualified", "false"),
            ("superseded_by", "none"),
            ("kind", "pack")
        ]
    );
    assert_eq!(
        t.header_value(),
        "tier=T2&run_id=run-20261013-a&family=gp-teacher-v6&qualified=false&superseded_by=none&kind=pack"
    );
    // qualified results, splits and evidence are T0.
    assert!(WriteTags::new(Tier::T2, "r", "f", true, None, ObjectKind::Result).is_err());
    assert!(WriteTags::new(Tier::T1, "r", "f", false, None, ObjectKind::Split).is_err());
    assert!(WriteTags::new(Tier::T3, "r", "f", false, None, ObjectKind::Evidence).is_err());
    assert!(WriteTags::new(Tier::T0, "r", "f", true, None, ObjectKind::Result).is_ok());
    // Only a qualified champion supersedes.
    let unq = Supersession {
        champion_run_id: "champ".into(),
        champion_qualified: false,
    };
    let q = Supersession {
        champion_run_id: "champ".into(),
        champion_qualified: true,
    };
    assert!(WriteTags::new(Tier::T3, "r", "f", false, Some(&unq), ObjectKind::Result).is_err());
    let ok = WriteTags::new(Tier::T3, "r", "f", false, Some(&q), ObjectKind::Result).unwrap();
    assert!(ok
        .pairs()
        .contains(&("superseded_by".into(), "champ".into())));
    // S3 tag charset.
    assert!(WriteTags::new(Tier::T2, "run#1", "f", false, None, ObjectKind::Result).is_err());
}
