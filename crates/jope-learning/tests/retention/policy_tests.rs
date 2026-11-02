use super::*;

#[test]
fn hard_t0_cannot_be_removed_by_policy() {
    // A policy that calls splits/ and the sealed bucket expendable still yields KEEP_T0.
    let f = fixture();
    let mut p: Value = serde_json::from_str(DEFAULT_POLICY_JSON).unwrap();
    p["rules"].as_array_mut().unwrap().push(json!({"bucket": DATA, "prefix": "splits/", "kind": "gp-result-unconsumed", "family": "x", "why": "hostile"}));
    p["rules"].as_array_mut().unwrap().push(
        json!({"bucket": SEALED, "prefix": "", "kind": "archive", "family": "x", "why": "hostile"}),
    );
    put(
        &f.path("hostile-policy.json"),
        &serde_json::to_vec(&p).unwrap(),
    );
    let mut cfg = config(&f);
    cfg.policy = Some(f.path("hostile-policy.json"));
    let b = build(&cfg).unwrap();
    assert_eq!(action(&b, &format!("s3://{DATA}/splits/")), Action::KeepT0);
    assert_eq!(
        action(&b, &format!("s3://{SEALED}/objects/")),
        Action::KeepT0
    );
    assert!(hard_t0("veox-jopedime-archive-use2", "evidence/x/").is_some());
    assert!(hard_t0(
        "veox-jopedime-archive-use2",
        "staging/node-high-value/sealed-meta-test-winners/"
    )
    .is_some());
    assert!(hard_t0(
        "veox-jopedime-learning-use2",
        "experiments/gp-teacher-v6/contexts/"
    )
    .is_none());
}

#[test]
fn policy_validation_rejects_unsafe_settings() {
    let base: Value = serde_json::from_str(DEFAULT_POLICY_JSON).unwrap();
    Policy::parse(DEFAULT_POLICY_JSON.as_bytes()).unwrap();
    for (field, value) in [
        ("l6_direction", json!("higher_is_better")),
        ("noncurrent_days", json!(7)),
        ("pack_bytes", json!(1024)),
        ("reservoir_rate", json!(0.0)),
    ] {
        let mut p = base.clone();
        p[field] = value;
        assert!(
            Policy::parse(&serde_json::to_vec(&p).unwrap()).is_err(),
            "{field} accepted"
        );
    }
    // D1: structured refs and non-scan roles can never be demoted to prose mentions.
    for (field, value) in [
        ("prose_extensions", json!(["md", "json"])),
        ("prose_extensions", json!(["jsonl"])),
        ("prose_extensions", json!(["MD"])),
        ("prose_extensions", json!(["parquet"])),
        ("prose_roles", json!(["val263"])),
        ("prose_roles", json!(["evidence", "prereg"])),
        ("prose_roles", json!(["prose-mention"])),
    ] {
        let mut p = base.clone();
        p["reference_formats"][field] = value.clone();
        assert!(
            Policy::parse(&serde_json::to_vec(&p).unwrap()).is_err(),
            "reference_formats.{field} = {value} accepted"
        );
    }
}

#[test]
fn prose_evidence_mentions_are_tv_not_pins() {
    // Owner decision D1 (2026-10-13T12:36Z): JSON/JSONL evidence pins T0; an md/txt/log mention
    // is training value only (blocks EXPIRE, allows COMPACT/COLD); unruled formats still pin.
    let f = fixture();
    let l = |p: &str| format!("s3://{LEARN}/{p}");
    put(
        &f.path("evid/chat/AGENT_CHAT.md"),
        format!("ran s3://{LEARN}/experiments/gp-teacher-v6/contexts/cccc/x.json today\n")
            .as_bytes(),
    );
    put(
        &f.path("evid/logs/run.log"),
        format!("GET s3://{LEARN}/experiments/regression-evolution/v1/attempts/a2 200\n")
            .as_bytes(),
    );
    put(
        &f.path("evid/receipts/r.json"),
        &serde_json::to_vec(&json!({"bucket": LEARN, "key": "staging/lineage-a/preserved-archives-20260908-F-D-E/a.tar"}))
            .unwrap(),
    );
    put(
        &f.path("evid/notes.yaml"),
        format!("ref: s3://{LEARN}/experiments/gp-teacher-v6/contexts/bbbb/y\n").as_bytes(),
    );
    let mut cfg = config(&f);
    cfg.scans.push((Role::Evidence, f.path("evid")));
    let b = build(&cfg).unwrap();

    let cccc = row(&b, &l("experiments/gp-teacher-v6/contexts/cccc/"));
    assert_eq!(cccc.action, Action::Compact, "{}", cccc.reason);
    assert_eq!(cccc.t0, None);
    assert_eq!(cccc.refs_by_role.get("prose-mention"), Some(&1));
    assert!(cccc.tv.value && cccc.tv.uses.contains(&"prose-mention"));
    assert!(
        !expire_allowed(&cccc.tv, true, true),
        "a prose mention blocks EXPIRE"
    );
    assert!(cccc
        .ref_sources_sample
        .contains(&"prose-mention:chat/AGENT_CHAT.md".to_string()));
    let attempts = row(&b, &l("experiments/regression-evolution/v1/attempts/"));
    assert_eq!(attempts.action, Action::DistillCold, "{}", attempts.reason);
    assert!(attempts.tv.uses.contains(&"prose-mention"));
    assert_eq!(
        action(
            &b,
            &l("staging/lineage-a/preserved-archives-20260908-F-D-E/")
        ),
        Action::KeepT0,
        "structured JSON evidence pins"
    );
    let bbbb = row(&b, &l("experiments/gp-teacher-v6/contexts/bbbb/"));
    assert_eq!(bbbb.action, Action::KeepT0, "unruled yaml still pins");
    assert_eq!(bbbb.refs_by_role.get("evidence"), Some(&1));
    assert_eq!(
        action(&b, &l("experiments/mystery/")),
        Action::KeepT0,
        "main-tree prose is outside the ruling and still pins"
    );
    let ev = b.lock.scans.iter().find(|s| s.role == "evidence").unwrap();
    assert_eq!((ev.object_refs, ev.prose_refs), (4, 2));
    assert!(b.report_md.contains("(2 prose mentions, D1)"));

    // Without the policy entry (the pre-D1 behaviour) the same mentions pin.
    let mut p: Value = serde_json::from_str(DEFAULT_POLICY_JSON).unwrap();
    p["reference_formats"]["prose_roles"] = json!([]);
    put(&f.path("pre-d1.json"), &serde_json::to_vec(&p).unwrap());
    cfg.policy = Some(f.path("pre-d1.json"));
    let pre = build(&cfg).unwrap();
    assert_eq!(
        action(&pre, &l("experiments/gp-teacher-v6/contexts/cccc/")),
        Action::KeepT0
    );
    assert_eq!(
        action(&pre, &l("experiments/regression-evolution/v1/attempts/")),
        Action::KeepT0
    );
}

#[test]
fn oversized_scan_files_that_name_a_bucket_fail_closed() {
    // A file over the scan cap is not parsed. One that names no in-scope bucket could not pin
    // anything and is only counted; one that does fails the run, because skipping it could hide
    // a T0 pin (the 2026-10-06 archive cursors pinned a COMPACT group that way).
    let f = fixture();
    let cc = format!("s3://{LEARN}/experiments/gp-teacher-v6/contexts/cccc/");
    let filler = "x".repeat(10_000);
    put(
        &f.path("evid/big-unrelated.jsonl"),
        format!("{{\"note\": \"{filler}\"}}\n").as_bytes(),
    );
    let mut cfg = config(&f);
    cfg.scans.push((Role::Evidence, f.path("evid")));
    cfg.scan_max_bytes = 4096;
    let b = build(&cfg).unwrap();
    let ev = b.lock.scans.iter().find(|s| s.role == "evidence").unwrap();
    assert_eq!((ev.files_over_cap, ev.files_read), (1, 0));
    assert_eq!(action(&b, &cc), Action::Compact);

    put(
        &f.path("evid/archive-cursor.json"),
        &serde_json::to_vec(&json!({"note": filler, "objects": [
            {"bucket": LEARN, "key": "experiments/gp-teacher-v6/contexts/cccc/x7"}
        ]}))
        .unwrap(),
    );
    let err = format!("{:#}", build(&cfg).err().unwrap());
    assert!(
        err.contains("archive-cursor.json") && err.contains("over the scan cap"),
        "{err}"
    );
    // With a cap that covers it the pin is read and cccc is T0.
    cfg.scan_max_bytes = 1 << 20;
    assert_eq!(action(&build(&cfg).unwrap(), &cc), Action::KeepT0);
}

#[test]
fn expire_needs_all_three_conditions() {
    let no_tv = Tv {
        value: false,
        uses: vec![],
    };
    let tv = Tv {
        value: true,
        uses: vec!["unknown-fail-closed"],
    };
    assert!(expire_allowed(&no_tv, true, true));
    assert!(!expire_allowed(&tv, true, true));
    assert!(!expire_allowed(&no_tv, false, true));
    assert!(!expire_allowed(&no_tv, true, false));
}
