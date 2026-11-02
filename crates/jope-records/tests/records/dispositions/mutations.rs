use super::*;

#[test]
fn mutations_are_rejected() {
    let other_finding_ci = || Value::from("evidence/S03-G04/ci.json");
    let mutations: Vec<(&str, Mutate, &str)> = vec![
        (
            "unknown finding",
            update_record(vec![("finding", "S99-G1".into())]),
            "unknown finding",
        ),
        (
            "unsupported disposition",
            update_record(vec![("disposition", "closed".into())]),
            ": disposition$",
        ),
        (
            "digest mismatch",
            update_reference("hosted_ci", "sha256", "0".repeat(64).into()),
            "digest mismatch",
        ),
        (
            "path traversal",
            update_reference(
                "hosted_ci",
                "path",
                "evidence/S03-G03/../../register.json".into(),
            ),
            "path traversal",
        ),
        (
            "other finding directory",
            update_reference("hosted_ci", "path", other_finding_ci()),
            "outside evidence/S03-G03/",
        ),
        (
            "symlinked evidence",
            Box::new(symlink_evidence),
            "evidence symlink",
        ),
        (
            "evidence names another finding",
            edit_evidence("hosted_ci", vec![("finding", "S03-G04".into())]),
            "another finding",
        ),
        (
            "evidence names another stage",
            edit_evidence("hosted_ci", vec![("stage", "fixture_proof".into())]),
            "another stage",
        ),
        (
            "evidence names another commit",
            edit_evidence("hosted_ci", vec![("source_commit", PRE.into())]),
            "another commit",
        ),
        (
            "hosted CI failed",
            edit_evidence("hosted_ci", vec![("conclusion", "failure".into())]),
            "hosted CI evidence binding",
        ),
        (
            "hosted CI other head",
            edit_evidence("hosted_ci", vec![("head_sha", PRE.into())]),
            "hosted CI evidence binding",
        ),
        (
            "synthetic claims hosted CI",
            edit_evidence(
                "hosted_ci",
                vec![("evidence_kind", "synthetic-fixture".into())],
            ),
            "cannot back hosted_ci",
        ),
        (
            "defect without hosted CI",
            pop_stage("hosted_ci"),
            "needs source, fixture and hosted CI",
        ),
        (
            "defect without source",
            pop_stage("source_implementation"),
            "needs source, fixture and hosted CI",
        ),
        (
            "positive not observed",
            edit_evidence("fixture_proof", vec![("passed_tests", Value::List(vec![]))]),
            "not observed passing",
        ),
        (
            "positive also failed",
            edit_evidence(
                "fixture_proof",
                vec![(
                    "failed_tests",
                    strs(&[
                        &format!("{TARGET}::{TEST}_pre"),
                        &format!("{TARGET}::{TEST}"),
                    ]),
                )],
            ),
            "roundtrip not observed passing",
        ),
        (
            "pre-fix not observed",
            edit_evidence("fixture_proof", vec![("failed_tests", Value::List(vec![]))]),
            "pre-fix failure not observed",
        ),
        (
            "pre-fix commit equals fix",
            Box::new(|_, record| {
                let test = crate::nth(at(stage(record, "fixture_proof"), "tests"), 0);
                set(test, "pre_fix_commit", FIXED.into());
            }),
            "distinct pre-fix commit",
        ),
        (
            "no negative test",
            Box::new(|_, record| {
                at(stage(record, "fixture_proof"), "tests")
                    .as_list_mut()
                    .expect("tests")
                    .remove(1);
            }),
            "pre-fix failure, negative and positive",
        ),
        (
            "duplicate stage",
            Box::new(|_, record| {
                let copy = stage(record, "hosted_ci").clone();
                at(record, "attestations")
                    .as_list_mut()
                    .expect("attestations")
                    .push(copy);
            }),
            "duplicate hosted_ci",
        ),
        (
            "historical unknown closed",
            update_record(vec![
                ("disposition", "historical-unknown".into()),
                ("reason", "lost".into()),
            ]),
            "historical-unknown cannot close",
        ),
        (
            "missing evidence closed",
            update_record(vec![
                ("disposition", "missing-execution-evidence".into()),
                ("reason", "none".into()),
            ]),
            "missing-execution-evidence cannot close",
        ),
        (
            "deferral of regression",
            update_record(vec![
                ("disposition", "deferred-non-regression".into()),
                ("closed", false.into()),
                ("deferred_subcases", strs(&["regression"])),
            ]),
            "deferral names only",
        ),
        (
            "live package without live stage",
            Box::new(|fixture, record| *record = fixture.defect("S08-G03", true)),
            "W8 requires live qualification",
        ),
        (
            "no decision owner",
            update_record(vec![("decided_by", Value::Dict(Dict::new()))]),
            "decided_by",
        ),
        (
            "empty scope",
            update_record(vec![("scope", " ".into())]),
            ": scope$",
        ),
    ];
    // Each mutation must fail for its own reason, not an incidental earlier check.
    assert_eq!(mutations.len(), 26);
    for (label, mutate, reason) in mutations {
        let fixture = Fixture::new();
        let mut record = fixture.defect("S03-G03", true);
        mutate(&fixture, &mut record);
        fixture.write(vec![record]);
        let result = fixture.verify();
        assert!(result.is_err(), "{label}: accepted");
        assert_refused(result, reason);
    }
}
