use jope_records::research::wave_git::*;
use std::collections::BTreeSet;

fn input() -> WaveGitInput {
    let fields = [
        "wave_index",
        "wave_id",
        "version",
        "source_commit",
        "training_identity",
        "archive_manifest",
        "archive_status",
        "capture_status",
        "stage",
        "disposition",
        "ancestry",
        "evidence_roots",
        "question",
        "hypothesis",
        "controls",
        "execution",
        "effect",
        "changed",
        "counterevidence",
        "falsifier",
        "reconstruction",
        "metric:loss",
    ];
    WaveGitInput {
        wave_id: "wave-alpha".into(),
        version: "v-one".into(),
        source_commit: None,
        training_identity_sha256: "a".repeat(64),
        archive_manifest_sha256: "b".repeat(64),
        archive_status: "LOCAL_COMMITTED_REMOTE_PENDING".into(),
        capture_status: "COMPLETE".into(),
        stage: Stage::Shadow,
        disposition: "aborted".into(),
        ancestry: vec!["parent-a".into()],
        evidence_roots: vec![EvidenceRoot {
            sha256: "c".repeat(64),
            member_sha256: "e".repeat(64),
            version: "local-v1".into(),
            physical_version: None,
            bytes: 1024,
            codec: "parquet-zstd".into(),
            rows: 1,
        }],
        context: [
            ("question", "Does the candidate improve?"),
            ("hypothesis", "Prospective comparison"),
            ("controls", "Frozen baseline"),
            ("execution", "Some attempts stopped"),
            ("effect", "Estimate remains uncertain"),
            ("stage", "Shadow evidence"),
            ("changed", "No confirmed consumer change"),
            ("counterevidence", "Failed attempts retained"),
            ("falsifier", "Repeat under frozen controls"),
            ("reconstruction", "Use the immutable evidence root"),
        ]
        .into_iter()
        .map(|(k, v)| (k.into(), v.into()))
        .collect(),
        assertions: vec![NumericAssertion {
            id: "loss".into(),
            value: Number::Float(0.25),
            unit: "loss/example".into(),
            numerator: 2,
            denominator: 8,
            status: "inconclusive".into(),
            source_field: "training.loss".into(),
            expression: "mean(loss)".into(),
            evidence_sha256: "c".repeat(64),
            evidence_version: "local-v1".into(),
            export_allowed: true,
        }],
        disclosure: DisclosurePolicy {
            role: "public".into(),
            policy_id: "public-fixture".into(),
            sha256: "d".repeat(64),
            allowed_fields: fields
                .into_iter()
                .map(str::to_owned)
                .collect::<BTreeSet<_>>(),
        },
        additional_review_bytes: 0,
        budget_exception: None,
    }
}

#[test]
fn repeated_generation_is_byte_identical() {
    let i = input();
    assert_eq!(compile(&i).unwrap(), compile(&i).unwrap());
}
#[test]
fn numeric_facts_bind_value_unit_denominator_source_and_version() {
    let p = compile(&input()).unwrap();
    let m = String::from_utf8(p.metrics).unwrap();
    for s in [
        "0.25",
        "loss/example",
        "\"denominator\":8",
        "training.loss",
        "mean(loss)",
        "local-v1",
        "inconclusive",
    ] {
        assert!(m.contains(s));
    }
}
#[test]
fn only_allowlisted_disclosure_is_rendered() {
    let mut i = input();
    i.disclosure.allowed_fields.remove("counterevidence");
    assert!(compile(&i).is_err());
}
#[test]
fn private_role_cannot_compile_public_projection() {
    let mut i = input();
    i.disclosure.role = "sealed".into();
    assert!(compile(&i).is_err());
}
#[test]
fn numeric_prose_is_rejected_instead_of_laundered() {
    let mut i = input();
    i.context
        .insert("effect".into(), "improved by twelve percent".into());
    assert!(compile(&i).is_err());
}
#[test]
fn unknown_source_commit_remains_null_and_future_git_hash_is_absent() {
    let p = compile(&input()).unwrap();
    let m = String::from_utf8(p.manifest).unwrap();
    assert!(m.contains("\"source_commit\":null"));
    assert!(m.contains("GIT_PUBLICATION_PENDING"));
    assert!(!m.contains("git_commit"));
}
#[test]
fn stage_and_disposition_are_never_upgraded_by_summary() {
    let p = compile(&input()).unwrap();
    let c = String::from_utf8(p.card).unwrap();
    assert!(
        c.contains("shadow")
            && c.contains("aborted")
            && c.contains("descriptive; no promotion authority")
    );
}
#[test]
fn traversal_wave_ids_are_refused() {
    let mut i = input();
    i.wave_id = "../private".into();
    assert!(compile(&i).is_err());
}
#[test]
fn each_file_and_total_bytes_are_measured() {
    let p = compile(&input()).unwrap();
    assert!(p.byte_counts["wave.manifest.json"] <= MANIFEST_LIMIT);
    assert!(p.byte_counts["card.json"] <= CARD_LIMIT);
    assert!(p.byte_counts["context.md"] <= CONTEXT_LIMIT);
    assert!(p.byte_counts["metrics.json"] <= METRICS_LIMIT);
    assert!(p.index_row.len() <= INDEX_ROW_LIMIT);
    assert!(p.total_bytes <= TOTAL_LIMIT);
}
#[test]
fn over_budget_requires_reviewed_reason_and_preserves_all_bytes() {
    let mut i = input();
    i.additional_review_bytes = TOTAL_LIMIT + 1;
    assert!(compile(&i).is_err());
    i.budget_exception = Some(("review-123".into(), "material limitation preserved".into()));
    let p = compile(&i).unwrap();
    assert!(p.total_bytes > TOTAL_LIMIT);
    assert!(String::from_utf8(p.manifest)
        .unwrap()
        .contains("material limitation preserved"));
}
#[test]
fn index_accepts_sorted_multiple_versions_for_one_wave() {
    let a = compile(&input()).unwrap().index_row;
    let mut i = input();
    i.version = "v-two".into();
    let b = compile(&i).unwrap().index_row;
    assert!(validate_index_rows(&[a, b]).is_ok());
}
#[test]
fn index_rejects_unsorted_or_duplicate_wave_versions() {
    let a = compile(&input()).unwrap().index_row;
    let mut i = input();
    i.wave_id = "wave-zeta".into();
    let z = compile(&i).unwrap().index_row;
    assert!(validate_index_rows(&[z, a.clone()]).is_err());
    assert!(validate_index_rows(&[a.clone(), a]).is_err());
}
#[test]
fn evidence_roots_are_stable_under_input_order() {
    let mut i = input();
    i.evidence_roots.push(EvidenceRoot {
        sha256: "f".repeat(64),
        member_sha256: "e".repeat(64),
        version: "v2".into(),
        physical_version: None,
        bytes: 32,
        codec: "parquet-zstd".into(),
        rows: 1,
    });
    let p = compile(&i).unwrap();
    i.evidence_roots.reverse();
    assert_eq!(p, compile(&i).unwrap());
}
#[test]
fn malformed_numeric_assertion_is_rejected() {
    let mut i = input();
    i.assertions[0].denominator = 0;
    assert!(compile(&i).is_err());
}

#[test]
fn small_projection_fixture_matches_all_emitted_files() {
    let mut i = input();
    i.wave_id = "fixture-wave".into();
    let p = compile(&i).unwrap();
    assert_eq!(
        p.manifest,
        include_str!("fixtures/wave-evidence-v1/git-projection/wave.manifest.json")
            .trim_end()
            .as_bytes()
    );
    assert_eq!(
        p.card,
        include_str!("fixtures/wave-evidence-v1/git-projection/card.json")
            .trim_end()
            .as_bytes()
    );
    assert_eq!(
        p.context,
        include_bytes!("fixtures/wave-evidence-v1/git-projection/context.md")
    );
    assert_eq!(
        p.metrics,
        include_str!("fixtures/wave-evidence-v1/git-projection/metrics.json")
            .trim_end()
            .as_bytes()
    );
}
