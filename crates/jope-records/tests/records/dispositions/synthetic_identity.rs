use super::*;

/// Marks an evidence body with the public conformance fixtures' synthetic identity.
fn marked(body: &Value) -> Value {
    let mut body = body.clone();
    set(
        &mut body,
        "synthetic_identity",
        "jope.core-ip-conformance/v1".into(),
    );
    body
}

#[test]
fn synthetic_identity_cannot_close_through_hosted_or_live_evidence() {
    // W12-T07 (reproduced by repro.log): every evidence body of a W8 closure carries the
    // conformance synthetic identity while declaring hosted-ci and live-host kinds.
    let closure = |mark_ci: bool, mark_live: bool| {
        let fixture = Fixture::new();
        let finding = "S08-G03";
        let mut record = fixture.defect(finding, true);
        let live_body = fixture.body(finding, "live_qualification", "live-host", &[]);
        let live_body = if mark_live {
            marked(&live_body)
        } else {
            live_body
        };
        let live = fixture.evidence(finding, "live.json", &live_body);
        at(&mut record, "attestations")
            .as_list_mut()
            .expect("attestations")
            .push(fixture.attestation("live_qualification", live));
        for name in ["source_implementation", "fixture_proof", "hosted_ci"] {
            if name == "hosted_ci" && !mark_ci {
                continue;
            }
            let reference = first_reference(&mut record, name);
            let path = fixture
                .root
                .join(reference.get("path").and_then(Value::as_str).expect("path"));
            dump_with(&path, &marked(&load(&path)), true, None);
            set(reference, "sha256", digest(&path).into());
        }
        fixture.write(vec![record]);
        fixture
    };
    // Every case runs before the verdict, so a failure lists each gap rather than the first.
    let mut accepted = Vec::new();
    for (label, mark_ci, reason) in [
        (
            "marked hosted CI and live bodies",
            true,
            "^S08-G03: synthetic identity cannot be hosted-ci evidence$",
        ),
        (
            "marked live body",
            false,
            "^S08-G03: synthetic identity cannot be live-host evidence$",
        ),
    ] {
        match closure(mark_ci, true).verify() {
            Ok(summary) => accepted.push(format!("{label}: ACCEPTED {summary:?}")),
            Err(error) => assert_refused::<()>(Err(error), reason),
        }
    }
    assert!(
        accepted.is_empty(),
        "synthetic identity closed a finding:\n{}",
        accepted.join("\n")
    );
    // Positive control: marked source trace and synthetic fixture evidence keep their stages
    // when hosted CI and live evidence are not synthetic.
    let fixture = closure(false, false);
    assert_eq!(
        fixture
            .verify()
            .expect("real hosted and live evidence")
            .closed,
        1
    );
}

#[test]
fn synthetic_identity_cannot_be_declared_real_corpus_evidence() {
    // W12-T07 addition: a synthetic-marked fixture cannot pass as real-corpus fixture proof.
    let fixture = Fixture::new();
    let mut record = fixture.defect("S03-G03", false);
    let reference = first_reference(&mut record, "fixture_proof");
    let path = fixture
        .root
        .join(reference.get("path").and_then(Value::as_str).expect("path"));
    let mut body = marked(&load(&path));
    set(&mut body, "evidence_kind", "real-corpus-fixture".into());
    dump_with(&path, &body, true, None);
    set(reference, "sha256", digest(&path).into());
    fixture.write(vec![record]);
    assert_refused(
        fixture.verify(),
        "^S03-G03: synthetic identity cannot be real-corpus-fixture evidence$",
    );
}
