use super::*;

#[test]
fn every_action_path_matches_the_binding_rules() {
    let f = fixture();
    let b = build(&config(&f)).unwrap();
    let l = |p: &str| format!("s3://{LEARN}/{p}");
    assert_eq!(
        action(&b, &l("#direct")),
        Action::Hold,
        "unmatched root objects default to HOLD"
    );
    assert_eq!(action(&b, &l("experiments/#direct")), Action::Hold);
    assert_eq!(
        action(&b, &l("experiments/gp-teacher-v6/contexts/aaaa/")),
        Action::KeepHot,
        "bucketless frozen24 key ref keeps it in place"
    );
    assert_eq!(
        action(&b, &l("experiments/gp-teacher-v6/contexts/bbbb/")),
        Action::KeepHot,
        "live writer"
    );
    assert_eq!(
        action(&b, &l("experiments/gp-teacher-v6/contexts/cccc/")),
        Action::Compact,
        "closed, small objects; skipped dir pins nothing"
    );
    assert_eq!(
        action(
            &b,
            &l("experiments/gp-policy-dl/v2-trainonly/program-results/")
        ),
        Action::KeepT0,
        "val263 reference"
    );
    assert_eq!(
        action(&b, &l("experiments/gp-policy-dl/v2-trainonly/gp-hwm/")),
        Action::KeepHot,
        "train reference by key"
    );
    assert_eq!(
        action(&b, &l("experiments/gp-policy-dl/v2-trainonly/checkpoints/")),
        Action::Hold,
        "no checkpoint->L6 linkage"
    );
    assert_eq!(
        action(&b, &l("experiments/regression-evolution/v1/attempts/")),
        Action::DistillCold,
        "inventory copy and elided key do not pin"
    );
    assert_eq!(
        action(&b, &l("experiments/regression-evolution/v1/contexts/")),
        Action::KeepT0,
        "holds GP rows of a val263 context"
    );
    assert_eq!(
        action(&b, &l("experiments/mystery/")),
        Action::KeepT0,
        "main-tree object reference pins"
    );
    assert_eq!(
        action(
            &b,
            &l("staging/lineage-a/preserved-archives-20260908-F-D-E/")
        ),
        Action::Cold
    );
    assert_eq!(action(&b, &format!("s3://{DATA}/splits/")), Action::KeepT0);
    assert_eq!(
        action(&b, &format!("s3://{SEALED}/objects/")),
        Action::KeepT0
    );
    assert_eq!(
        action(&b, &format!("s3://{LEGACY}/data/")),
        Action::Hold,
        "out of scope"
    );
    assert!(
        b.rows.iter().all(|r| r.action != Action::Expire),
        "no EXPIRE at prefix level"
    );
    assert!(b.rows.iter().all(|r| r.tv.value), "TV fails closed to true");
    assert_eq!(
        b.summary.totals.objects,
        ledger()["totals"]["objects"].as_u64().unwrap()
    );
    assert_eq!(b.summary.refs_bucketless_resolved, 1);
    assert_eq!(
        b.summary.refs_unmatched_in_scope, 0,
        "globs and placeholders are prefix mentions"
    );
    assert!(b.summary.prefix_mentions >= 3);
    for g in ["#direct", "experiments/#direct"] {
        assert!(
            row(&b, &l(g)).refs_by_role.is_empty(),
            "{g}: a glob/placeholder must not pin a group"
        );
    }
}

#[test]
fn tv_index_and_cost_fields_are_derived_from_the_inputs() {
    let f = fixture();
    let b = build(&config(&f)).unwrap();
    let hwm = row(
        &b,
        &format!("s3://{LEARN}/experiments/gp-policy-dl/v2-trainonly/gp-hwm/"),
    );
    let ix = hwm.index.as_ref().unwrap();
    assert_eq!(
        (ix.rows, ix.scored_rows, ix.contexts_with_alternatives),
        (2, 2, 1)
    );
    for u in ["ce-target", "negative-example", "knn-teacher", "reservoir"] {
        assert!(
            hwm.tv.uses.contains(&u),
            "gp-hwm TV lacks {u}: {:?}",
            hwm.tv.uses
        );
    }
    assert_eq!(hwm.pinned_programs.get("train"), Some(&1));
    let cold = row(
        &b,
        &format!("s3://{LEARN}/staging/lineage-a/preserved-archives-20260908-F-D-E/"),
    );
    assert_eq!(cold.cost.class_after, "DEEP_ARCHIVE");
    assert!(
        cold.cost.already_applied_by_lifecycle,
        "p1 rule already transitions it"
    );
    assert!(cold.cost.after_usd_mo < cold.cost.now_usd_mo);
    let dc = row(
        &b,
        &format!("s3://{LEARN}/experiments/regression-evolution/v1/attempts/"),
    );
    assert!(dc.cost.after_needs_e1_of_originals);
    assert!(
        dc.cost.one_time_off_aws_usd > dc.cost.one_time_in_region_usd,
        "reading from xbabe adds egress"
    );
    let unknown = row(
        &b,
        &format!("s3://{LEARN}/experiments/gp-teacher-v6/contexts/cccc/"),
    );
    assert_eq!(unknown.tv.uses, vec!["unknown-fail-closed"]);
    let mystery = row(&b, &format!("s3://{LEARN}/experiments/mystery/"));
    assert_eq!(
        mystery.ref_sources_sample,
        vec!["main-manifest:docs/notes.md".to_string()]
    );
    let ckpt = row(
        &b,
        &format!("s3://{LEARN}/experiments/gp-policy-dl/v2-trainonly/checkpoints/"),
    );
    assert!(
        ckpt.lifecycle
            .tagged
            .iter()
            .any(|t| t.starts_with("checkpoint-blobs-gir")),
        "tag rules are listed, not applied"
    );
    assert!(ckpt.lifecycle.cold_class.is_none());
}

#[test]
fn lifecycle_hazards_and_notes_are_reported() {
    let f = fixture();
    let b = build(&config(&f)).unwrap();
    assert!(
        b.summary
            .hazards
            .iter()
            .any(|h| h.kind == "expire-current-over-protected" && h.detail.contains("bad-expire")),
        "{:?}",
        b.summary.hazards
    );
    assert!(b
        .summary
        .notes
        .iter()
        .any(|n| n.kind == "lifecycle-not-supplied" && n.detail.contains(SEALED)));
    // D2 (owner, 2026-10-13T12:42:52Z): 90 d is the T0 noncurrent window too; nothing asks 365 d.
    let all = |b: &Built| -> Vec<String> {
        b.summary
            .hazards
            .iter()
            .chain(&b.summary.notes)
            .map(|n| n.kind.clone())
            .collect()
    };
    assert!(!all(&b)
        .iter()
        .any(|k| k.contains("365") || k.contains("noncurrent")));
    let lc = json!({"Rules": [{"ID": "short", "Status": "Enabled", "Filter": {"Prefix": ""}, "NoncurrentVersionExpiration": {"NoncurrentDays": 60}}]});
    put(
        &f.path("lc/veox-jopedime-learning-use2.json"),
        &serde_json::to_vec(&lc).unwrap(),
    );
    let b = build(&config(&f)).unwrap();
    assert!(
        b.summary
            .hazards
            .iter()
            .any(|h| h.kind == "t0-noncurrent-below-policy" && h.detail.contains("60d")),
        "{:?}",
        b.summary.hazards
    );
}

#[test]
fn plan_is_deterministic_and_never_overwritten() {
    let f = fixture();
    let a = build(&config(&f)).unwrap();
    let b = build(&config(&f)).unwrap();
    assert_eq!(a.plan_jsonl, b.plan_jsonl);
    assert_eq!(a.summary.plan_sha256, sha256_hex(&a.plan_jsonl));
    let out = f.path("out/plan1");
    write(&a, &out).unwrap();
    let side = fs::read_to_string(out.join("retention-plan.jsonl.sha256")).unwrap();
    assert!(side.starts_with(&a.summary.plan_sha256));
    let err = write(&b, &out).unwrap_err().to_string();
    assert!(err.contains("not empty"), "{err}");
    // A second plan diffs against the first.
    let mut cfg = config(&f);
    cfg.previous_plan = Some(out.join("retention-plan.jsonl"));
    let c = build(&cfg).unwrap();
    let d = c.summary.diff_vs_previous.unwrap();
    assert_eq!(d.get("unchanged").copied(), Some(c.rows.len() as u64));
}

#[test]
fn inputs_fail_closed() {
    let f = fixture();
    // Sidecar disagreement.
    put(
        &f.path("in/ledger.sha256"),
        b"0000000000000000000000000000000000000000000000000000000000000000  ledger.json\n",
    );
    let err = format!("{:#}", build(&config(&f)).err().unwrap());
    assert!(err.contains("disagrees with sidecar"), "{err}");

    // Unknown score identity.
    let f = fixture();
    let mut ix = index();
    ix["runs"][0]["score_name"] = json!("l7_mystery");
    put(&f.path("in/index.json"), &serde_json::to_vec(&ix).unwrap());
    let err = format!("{:#}", build(&config(&f)).err().unwrap());
    assert!(err.contains("unknown score identity"), "{err}");

    // Roster member hash mismatch.
    let f = fixture();
    put(
        &f.path("roster/train.jsonl"),
        format!("{}\n", json!({"context_id": C1})).as_bytes(),
    );
    let err = format!("{:#}", build(&config(&f)).err().unwrap());
    assert!(err.contains("roster manifest"), "{err}");

    // Expected sha256 prefix.
    let f = fixture();
    let mut cfg = config(&f);
    cfg.gp_index.expect_prefix = Some("deadbeef".into());
    let err = format!("{:#}", build(&cfg).err().unwrap());
    assert!(err.contains("does not start with the expected"), "{err}");

    // Ledger that does not reconcile.
    let f = fixture();
    let mut l = ledger();
    l["totals"]["objects"] = json!(1);
    put(&f.path("in/ledger.json"), &serde_json::to_vec(&l).unwrap());
    let lb = fs::read(f.path("in/ledger.json")).unwrap();
    put(
        &f.path("in/ledger.sha256"),
        format!("{}  ledger.json\n", sha256_hex(&lb)).as_bytes(),
    );
    let err = format!("{:#}", build(&config(&f)).err().unwrap());
    assert!(err.contains("totals say"), "{err}");

    // Incomplete bucket.
    let mut l = ledger();
    l["buckets"][0]["complete"] = json!(false);
    assert!(flatten(&l).is_err());
    let mut l = ledger();
    l["capped"] = json!(true);
    assert!(flatten(&l).is_err());
}
