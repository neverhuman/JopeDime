use super::*;
#[test]
fn membership_excludes_outer_and_sealed_without_target_fields() {
    let value = serde_json::json!({"protocol":"p","seed":0,"source_sha256":"s","rows":[{"id":"outer","group":"a","source_index":0,"fold":0},{"id":"fit","group":"b","source_index":1,"fold":1},{"id":"sealed","group":"c","source_index":2,"fold":null}]});
    let assignments: Assignments = serde_json::from_value(value.clone()).unwrap();
    assert_eq!(fitting_ids(&assignments), vec!["fit"]);
    let mut injected = value;
    injected["rows"][0]["target"] = json!(123);
    assert!(serde_json::from_value::<Assignments>(injected).is_err());
}

use jope_datasets::staging_disposition::committed;

fn pinned(dir: &Path, name: &str, bytes: &[u8]) -> Artifact {
    let path = dir.join(name);
    fs::write(&path, bytes).unwrap();
    Artifact {
        path,
        sha256: digest(bytes),
    }
}

/// A fitting-input manifest naming `source`, one projected input per raw
/// digest, and the inner folds on the first. No named object exists.
fn closure(source: &str, raw: &[&str], folds: &[&str]) -> Vec<u8> {
    let inputs = raw
        .iter()
        .enumerate()
        .map(|(i, sha)| {
            let gp = if i == 0 {
                json!({"folds":folds.iter().map(|f| json!([{"path":"/absent/fold.json","sha256":f},[]])).collect::<Vec<_>>()})
            } else {
                Value::Null
            };
            json!({"id":format!("d{i}"),"raw":{"path":format!("/absent/raw-{i}.json"),"sha256":sha},"gp":gp})
        })
        .collect::<Vec<_>>();
    serde_json::to_vec(&json!({"schema":"jope.inference-fitting-inputs/v1",
        "projection_source_sha":source,"inputs":inputs}))
    .unwrap()
}

/// W1 follow-up 2d at the planner's receiving path.
#[test]
fn the_planner_admits_only_closures_outside_the_historical_staging_path() {
    let (d, p) = committed().unwrap();
    let temp = tempfile::tempdir().unwrap();
    let dir = temp.path();
    let source = "0123456789abcdef0123456789abcdef01234567";
    let (r0, r1, f0) = (
        digest(b"fresh raw 0"),
        digest(b"fresh raw 1"),
        digest(b"fresh fold 0"),
    );
    // The historical member with inner folds (197_cpu_act).
    let member = &d.input_closure.members[0];
    assert!(!member.gp_fold_sha256.is_empty());

    // Positive control: a fresh closure is admitted, from the pinned bytes.
    let bytes = closure(source, &[&r0, &r1], &[&f0]);
    let pin = pinned(dir, "fresh.json", &bytes);
    let run = attestation(dir, &p, &bytes, false);
    let (consumed, receipt) = admit_closure(&pin, Some(&run)).unwrap();
    assert_eq!(receipt["corrected_run"]["accepted"], true);
    assert_eq!(consumed, bytes);
    assert_eq!(receipt["objects_checked"], 3);
    assert_eq!(receipt["projection_source_sha"], source);
    assert_eq!(receipt["staging_protocol_id"], p.protocol_id.as_str());

    // The historical projection source, one historical projected input or
    // one historical inner-fold input: each is refused before any named
    // object is opened (none exists).
    for (name, bytes, expected) in [
        (
            "source",
            closure(&d.study.source_sha, &[&r0, &r1], &[&f0]),
            "historical staging source",
        ),
        (
            "raw",
            closure(source, &[&r0, &member.raw_sha256], &[&f0]),
            "d1: projected fitting input",
        ),
        (
            "fold",
            closure(source, &[&r0, &r1], &[&f0, &member.gp_fold_sha256[2]]),
            "d0: inner fold 1 input",
        ),
    ] {
        let err = admit_closure(&pinned(dir, &format!("{name}.json"), &bytes), Some(&run))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains(expected) && err.contains("historical"),
            "{name}: {err}"
        );
    }

    // Bytes replaced after pinning are never consumed.
    fs::write(&pin.path, closure(source, &[&r1, &r0], &[&f0])).unwrap();
    let err = admit_closure(&pin, Some(&run)).unwrap_err().to_string();
    assert!(err.contains("manifest changed"), "{err}");
    // A symlink is refused even when it resolves to the pinned bytes.
    let target = pinned(dir, "target.json", &bytes);
    let link = dir.join("link.json");
    std::os::unix::fs::symlink(&target.path, &link).unwrap();
    let err = admit_closure(
        &Artifact {
            path: link,
            sha256: target.sha256,
        },
        Some(&run),
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("regular file"), "{err}");
}

/// The producer's attestation (`jope-datasets staging-disposition
/// attest`) for `bytes`, pinned; `outer_labels_parsed` is its declaration.
fn attestation(
    dir: &Path,
    protocol: &jope_datasets::staging_disposition::CorrectedProtocol,
    bytes: &[u8],
    outer_labels_parsed: bool,
) -> Artifact {
    use jope_datasets::staging_disposition::{attest_corrected_run, ProjectionAttestation};
    let run = attest_corrected_run(
        protocol,
        bytes,
        ProjectionAttestation {
            inputs_pre_projected: true,
            outer_scoring_labels_parsed: outer_labels_parsed,
            development_targets_parsed_outside_fitting_rows: false,
            sealed_tables_opened: false,
        },
        None,
    )
    .unwrap();
    let encoded = run.encode().unwrap();
    pinned(
        dir,
        &format!("run-{}.json", &digest(&encoded)[..12]),
        &encoded,
    )
}

/// W1-T07: a fresh closure alone is not admitted; the planner requires the
/// producer's corrected-run attestation for exactly those bytes.
#[test]
fn the_planner_requires_the_producers_corrected_run_attestation() {
    let (_, p) = committed().unwrap();
    let temp = tempfile::tempdir().unwrap();
    let dir = temp.path();
    let source = "0123456789abcdef0123456789abcdef01234567";
    let bytes = closure(
        source,
        &[&digest(b"attested raw 0")],
        &[&digest(b"attested fold 0")],
    );
    let pin = pinned(dir, "fresh.json", &bytes);
    let refused = admit_closure(&pin, None);
    assert!(
        matches!(&refused, Err(e) if e.to_string().contains("corrected-run attestation")),
        "a closure was admitted without the producer's corrected-run attestation"
    );
    // A run attesting other bytes, or declaring parsed outer labels.
    let other = closure(
        source,
        &[&digest(b"other raw 0")],
        &[&digest(b"other fold 0")],
    );
    for (run, expected) in [
        (
            attestation(dir, &p, &other, false),
            "not the run's pinned manifest",
        ),
        (
            attestation(dir, &p, &bytes, true),
            "fitting-only projection",
        ),
    ] {
        let err = format!("{:#}", admit_closure(&pin, Some(&run)).unwrap_err());
        assert!(err.contains(expected), "{err}");
    }
    // A pinned run replaced after pinning is not consumed.
    let honest = attestation(dir, &p, &bytes, false);
    let swapped = Artifact {
        path: attestation(dir, &p, &other, false).path,
        sha256: honest.sha256.clone(),
    };
    let err = format!("{:#}", admit_closure(&pin, Some(&swapped)).unwrap_err());
    assert!(err.contains("attestation changed"), "{err}");
    // Positive control: the producer's attestation of exactly these bytes.
    let (consumed, receipt) = admit_closure(&pin, Some(&honest)).unwrap();
    assert_eq!(consumed, bytes);
    assert_eq!(receipt["corrected_run"]["accepted"], true);
    assert_eq!(
        receipt["corrected_run"]["protocol_id"],
        p.protocol_id.as_str()
    );
}

/// W1 follow-up 2d at plan admission (`read`: run, cold and v3 report).
#[test]
fn a_plan_pinning_any_historical_object_is_never_admitted() {
    let (d, _) = committed().unwrap();
    let fresh = |s: &str| Artifact {
        path: PathBuf::from(format!("/absent/{s}")),
        sha256: digest(s.as_bytes()),
    };
    let inputs = ["a", "b"]
        .map(|id| Dataset {
            id: id.into(),
            raw: fresh(&format!("raw-{id}")),
            gp: None,
        })
        .to_vec();
    // Positive control.
    assert_eq!(
        refuse_historical_plan(&fresh("manifest"), &inputs).unwrap(),
        3
    );
    // The retained historical manifest (the local v3 plan built on it).
    let retained = Artifact {
        path: PathBuf::from("/absent/retained-fitting-inputs.json"),
        sha256: d.input_closure.manifest_sha256.clone(),
    };
    let err = refuse_historical_plan(&retained, &inputs)
        .unwrap_err()
        .to_string();
    assert!(err.contains("fitting-input manifest"), "{err}");
    // A fresh manifest re-pinning one historical projected input.
    let mut reused = inputs.clone();
    reused[1].raw.sha256 = d.input_closure.members[3].raw_sha256.clone();
    let err = refuse_historical_plan(&fresh("manifest"), &reused)
        .unwrap_err()
        .to_string();
    assert!(err.contains("b: projected fitting input"), "{err}");
}
