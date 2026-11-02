use super::*;

// ---------------------------------------------------------------------------
// Review of #111 (W1-T04, route R8 at the outer release authority): every
// grant field, every recorded digest and every consumed object comes from
// exactly the bytes the authority verified or wrote. The test-only
// instrumentation point `manifest::after_verification` replaces a file
// between its check (or its write) and its consumption.
// ---------------------------------------------------------------------------

pub(super) const OTHER_TEST: &str = "12,1000\n13,1000\n14,1000\n";

/// An exported selected subject of the view fixture, the sealed labels its
/// preparation binds (written after export), and another label file.
pub(super) struct OuterCase {
    root: PathBuf,
    prepared: PathBuf,
    output: PathBuf,
    seeds: PathBuf,
    predictor: PathBuf,
    sealed: PathBuf,
    other: PathBuf,
    // Rust drops fields in declaration order: stop the service before its files.
    _authority: Fixture,
    _temp: tempfile::TempDir,
}

pub(super) fn outer_case() -> OuterCase {
    let (temp, prepared) = view_fixture();
    let root = temp.path().canonicalize().unwrap();
    let prepared = root.join(prepared.strip_prefix(temp.path()).unwrap());
    let mut p: Value = read_json(&prepared.join("prepared.json")).unwrap();
    p["original_test_sha256"] = json!(hash(SEALED_TEST.as_bytes()));
    write_json(&prepared.join("prepared.json"), &p).unwrap();
    let mut authority = Fixture::activated(&root, vec![cut_training_hash(&prepared)]);
    authority.start();
    let (output, seeds, predictor) = exported_subject(&authority, &root, &prepared);
    let sealed = root.join("data/originals/sealed-test.csv");
    fs::write(&sealed, SEALED_TEST).unwrap();
    let other = root.join("data/originals/other-labels.csv");
    fs::write(&other, OTHER_TEST).unwrap();
    OuterCase {
        root,
        prepared,
        output,
        seeds,
        predictor,
        sealed,
        other,
        _authority: authority,
        _temp: temp,
    }
}

impl OuterCase {
    fn release(&self, name: &str) -> outer_release::Authority {
        outer_release::Authority::initialize(&self.root.join(name)).unwrap()
    }
    fn grant(&self, name: &str, test: &Path) -> Result<Value> {
        let path = self.release(name).grant(&outer_release::GrantRequest {
            search: handoff::Inputs {
                root: &self.root,
                output: &self.output,
                seeds: Some(&self.seeds),
                vocabulary: None,
            },
            predictor: &self.predictor,
            test,
        })?;
        read_json(&path)
    }
    fn handoff(&self) -> PathBuf {
        self.predictor.with_extension("handoff.json")
    }
}

/// How often a receiver reports `target` verified while `run` runs.
pub(super) fn verifications(target: &Path, run: impl FnOnce()) -> usize {
    let target = target.canonicalize().unwrap();
    let count = std::rc::Rc::new(std::cell::Cell::new(0));
    let seen = std::rc::Rc::clone(&count);
    let _hook = crate::manifest::verification_hook::install(move |path| {
        if path.canonicalize().ok().as_deref() == Some(target.as_path()) {
            seen.set(seen.get() + 1);
        }
    });
    run();
    count.get()
}

/// Replace `target` (new inode, via rename) with `bytes` at its `nth` reported
/// verification: the last one, as counted by an identical run.
pub(super) fn swap_at(
    target: &Path,
    nth: usize,
    bytes: Vec<u8>,
) -> crate::manifest::verification_hook::Guard {
    assert!(nth > 0, "the receiver never reported {target:?} verified");
    let target = target.canonicalize().unwrap();
    let mut seen = 0;
    crate::manifest::verification_hook::install(move |path| {
        if path.canonicalize().ok().as_deref() == Some(target.as_path()) {
            seen += 1;
            if seen == nth {
                let staged = target.with_extension("swap");
                fs::write(&staged, &bytes).unwrap();
                fs::rename(&staged, &target).unwrap();
            }
        }
    })
}

/// The reviewed finding [P1]: `grant_body` hashed `prepared.json` against the
/// sealed selection, then reopened it. A preparation substituted between the
/// two supplied the grant's sealed-label identity and metadata.
#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
pub(super) fn an_outer_grant_takes_its_labels_and_metadata_from_the_verified_preparation() {
    let case = outer_case();
    let prepared = case.prepared.join("prepared.json");
    let bound: Value = read_json(&prepared).unwrap();
    // Positive control: an honest grant binds the preparation's sealed labels.
    let mut honest = Value::Null;
    let nth = verifications(&prepared, || {
        honest = case.grant("honest", &case.sealed).unwrap();
    });
    assert_eq!(
        honest["scoring_labels"]["sha256"],
        hash(SEALED_TEST.as_bytes())
    );
    assert_eq!(honest["preparation"]["dataset"], bound["dataset"]);
    // A preparation naming another label file and other metadata replaces
    // the verified one right after its check.
    let mut substituted = bound.clone();
    substituted["original_test_sha256"] = json!(hash(OTHER_TEST.as_bytes()));
    substituted["dataset"] = json!("substituted-dataset");
    substituted["family"] = json!("substituted-family");
    substituted["clean_status"] = json!("substituted");
    let _hook = swap_at(
        &prepared,
        nth,
        serde_json::to_vec_pretty(&substituted).unwrap(),
    );
    let granted = case.grant("swapped", &case.other).unwrap();
    assert_eq!(
        granted["scoring_labels"]["sha256"],
        hash(SEALED_TEST.as_bytes()),
        "a grant was issued for labels the sealed preparation never bound: {granted}"
    );
    for key in ["dataset", "family", "clean_status"] {
        assert_eq!(granted["preparation"][key], bound[key], "{key}: {granted}");
    }
    // The scorer checks the named file against the verified identity, so the
    // substituted labels are never parsed.
    let outer = outer_release::Authority::open(&case.root.join("swapped")).unwrap();
    let grant_path = case
        .root
        .join("swapped/grants")
        .join(format!("{}.json", granted["grant_id"].as_str().unwrap()));
    let receipt = case.root.join("swapped-score.json");
    let error = refusal(outer.score(&grant_path, &case.predictor, &case.other, &receipt));
    assert!(
        error.contains("differ from the granted identity"),
        "{error}"
    );
}

/// The same receiver, for the sealed selection: the seal the handoff records
/// is the seal that is compared with the replayed selection.
#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
pub(super) fn an_outer_grant_is_bound_to_the_seal_it_checked() {
    let case = outer_case();
    let mut handoff: Value = read_json(&case.handoff()).unwrap();
    let seal = case.root.join(handoff["seal"]["path"].as_str().unwrap());
    let replayed = fs::read(&seal).unwrap();
    // Positive control, counting the seal's verifications.
    let nth = verifications(&seal, || {
        case.grant("honest", &case.sealed).unwrap();
    });
    // The handoff records a seal that is not the replayed selection.
    let mut tampered: Value = serde_json::from_slice(&replayed).unwrap();
    tampered["stop_reason"] = json!("tampered");
    let tampered = serde_json::to_vec_pretty(&tampered).unwrap();
    fs::write(&seal, &tampered).unwrap();
    handoff["seal"]["sha256"] = json!(hash(&tampered));
    write_json(&case.handoff(), &handoff).unwrap();
    // Negative control: refused without a replacement.
    let error = refusal(case.grant("tampered", &case.sealed));
    assert!(
        error.contains("differs from the replayed selection"),
        "{error}"
    );
    // The replayed seal replaces it right after its check against the handoff.
    let _hook = swap_at(&seal, nth, replayed);
    let error = refusal(case.grant("swapped", &case.sealed));
    assert!(
        error.contains("differs from the replayed selection"),
        "a grant was issued for a handoff whose seal is not the replayed selection: {error}"
    );
}

/// The same receiver, for the fitted subject: the predictor the grant names
/// is the predictor verified as the selected final refit.
#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
pub(super) fn an_outer_grant_is_for_the_predictor_it_verified() {
    let case = outer_case();
    let mut handoff: Value = read_json(&case.handoff()).unwrap();
    let exported = fs::read(&case.predictor).unwrap();
    // Positive control, counting the predictor's verifications.
    let mut honest = Value::Null;
    let nth = verifications(&case.predictor, || {
        honest = case.grant("honest", &case.sealed).unwrap();
    });
    assert_eq!(honest["subject"]["predictor"]["sha256"], hash(&exported));
    // The handoff records a predictor that declares another training scope.
    let mut forged: Value = serde_json::from_slice(&exported).unwrap();
    forged["training_sha256"] = json!("0".repeat(64));
    let forged = serde_json::to_vec_pretty(&forged).unwrap();
    fs::write(&case.predictor, &forged).unwrap();
    handoff["predictor"]["sha256"] = json!(hash(&forged));
    write_json(&case.handoff(), &handoff).unwrap();
    // Negative control: refused without a replacement.
    let error = refusal(case.grant("forged", &case.sealed));
    assert!(error.contains("another training scope"), "{error}");
    // The verified export replaces it right after the subject check.
    let _hook = swap_at(&case.predictor, nth, exported);
    let result = case.grant("swapped", &case.sealed);
    assert!(
        matches!(&result, Err(e) if format!("{e:#}").contains("another training scope")),
        "a grant was issued for a subject never verified as the final refit: {result:?}"
    );
}

/// The same receiver, for the handoff: the digest the grant records is the
/// digest of the handoff whose content was checked and recorded.
#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
pub(super) fn an_outer_grant_records_the_handoff_it_checked() {
    let case = outer_case();
    let recorded = fs::read(case.handoff()).unwrap();
    // Positive control, counting the handoff's verifications.
    let mut honest = Value::Null;
    let nth = verifications(&case.handoff(), || {
        honest = case.grant("honest", &case.sealed).unwrap();
    });
    assert_eq!(honest["subject"]["handoff"]["sha256"], hash(&recorded));
    // Another handoff replaces it right after it was read.
    let mut other: Value = serde_json::from_slice(&recorded).unwrap();
    other["final_refit"]["model_sha256"] = json!("f".repeat(64));
    let _hook = swap_at(
        &case.handoff(),
        nth,
        serde_json::to_vec_pretty(&other).unwrap(),
    );
    let granted = case.grant("swapped", &case.sealed).unwrap();
    assert_eq!(
        granted["subject"]["handoff"]["sha256"],
        hash(&recorded),
        "the grant records a handoff other than the one it checked: {granted}"
    );
    assert_eq!(
        granted["subject"]["model_sha256"],
        honest["subject"]["model_sha256"]
    );
}

/// The same receiver, for the final refit's fit completion
/// (`lineage::verify_final_refit`): the completion consumed is the one whose
/// digest the predictor records.
#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
pub(super) fn an_outer_grant_consumes_the_fit_completion_it_verified() {
    let case = outer_case();
    let exported: Value = read_json(&case.predictor).unwrap();
    let completion = PathBuf::from(exported["fit_completion"]["path"].as_str().unwrap());
    // Positive control, counting the completion's verifications.
    let mut honest = Value::Null;
    let nth = verifications(&completion, || {
        honest = case.grant("honest", &case.sealed).unwrap();
    });
    let _hook = swap_at(&completion, nth, b"{}".to_vec());
    let granted = case
        .grant("swapped", &case.sealed)
        .map_err(|e| format!("{e:#}"));
    assert!(
        matches!(&granted, Ok(g) if g["subject"]["fit_key"] == honest["subject"]["fit_key"]),
        "the refit check consumed a completion other than the verified one: {granted:?}"
    );
}

/// Replace the first file the authority reports written under `dir` with
/// `edit` applied to it.
pub(super) fn swap_written(
    dir: PathBuf,
    edit: impl Fn(&mut Value) + 'static,
) -> crate::manifest::verification_hook::Guard {
    let mut done = false;
    crate::manifest::verification_hook::install(move |path| {
        if !done && path.parent() == Some(dir.as_path()) {
            done = true;
            let mut value: Value = read_json(path).unwrap();
            edit(&mut value);
            let staged = path.with_extension("swap");
            fs::write(&staged, serde_json::to_vec_pretty(&value).unwrap()).unwrap();
            fs::rename(&staged, path).unwrap();
        }
    })
}

/// The authority's own records: the ledger binds the digest of exactly the
/// grant and withheld score the authority wrote, never a later reread.
#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
pub(super) fn the_release_ledger_binds_the_bytes_the_authority_wrote() {
    let case = outer_case();
    let other = hash(OTHER_TEST.as_bytes());
    // A grant replaced right after it is written, naming other labels.
    let dir = case.root.join("grant-swap");
    let outer = case.release("grant-swap");
    let grants = dir.canonicalize().unwrap().join("grants");
    let relative = case.other.strip_prefix(&case.root).unwrap().to_path_buf();
    let path = {
        let _hook = swap_written(grants, move |grant| {
            grant["scoring_labels"]["path"] = json!(relative);
            grant["scoring_labels"]["sha256"] = json!(other);
        });
        outer
            .grant(&outer_release::GrantRequest {
                search: handoff::Inputs {
                    root: &case.root,
                    output: &case.output,
                    seeds: Some(&case.seeds),
                    vocabulary: None,
                },
                predictor: &case.predictor,
                test: &case.sealed,
            })
            .unwrap()
    };
    let receipt = case.root.join("grant-swap-score.json");
    let error = refusal(outer.score(&path, &case.predictor, &case.other, &receipt));
    assert!(
        error.contains("not recorded by this release authority"),
        "labels the preparation never bound were scored under a replaced grant: {error}"
    );

    // A withheld score replaced right after it is written.
    let dir = case.root.join("withheld-swap");
    let outer = case.release("withheld-swap");
    let withheld = dir.canonicalize().unwrap().join("withheld");
    let path = outer
        .grant(&outer_release::GrantRequest {
            search: handoff::Inputs {
                root: &case.root,
                output: &case.output,
                seeds: Some(&case.seeds),
                vocabulary: None,
            },
            predictor: &case.predictor,
            test: &case.sealed,
        })
        .unwrap();
    let scored = {
        let _hook = swap_written(withheld, |result| {
            result["metrics"]["rmse"] = json!(0.0);
        });
        outer
            .score(
                &path,
                &case.predictor,
                &case.sealed,
                &case.root.join("withheld-score.json"),
            )
            .unwrap()
    };
    let grant_id = scored["grant_id"].as_str().unwrap().to_string();
    let visible = case.root.join("withheld-swap-released.json");
    let released = outer
        .release(&grant_id, &visible)
        .map_err(|e| format!("{e:#}"));
    assert!(
        matches!(&released, Err(e) if e.contains("withheld score changed after scoring")),
        "a withheld score other than the one computed was released: {released:?}"
    );
    assert!(!visible.exists());
}

// ---------------------------------------------------------------------------
// Review of #111, second finding: the exported predictor's own model
// parameters are bound to the verified final refit.
// ---------------------------------------------------------------------------

/// Add one to the first floating-point parameter of `value` (depth first).
pub(super) fn change_one_parameter(value: &mut Value) -> bool {
    if value.is_f64() {
        *value = json!(value.as_f64().unwrap() + 1.0);
        return true;
    }
    match value {
        Value::Array(items) => items.iter_mut().any(change_one_parameter),
        Value::Object(members) => members.values_mut().any(change_one_parameter),
        _ => false,
    }
}

/// `lineage::verify_final_refit` bound the exported predictor's recipe,
/// training scope and fit-completion reference, but not the model parameters
/// the predictor serves. Those must be the model the verified completion
/// retained (its `model_sha256`, as the fit authority defines it).
#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
pub(super) fn an_exported_predictors_model_parameters_are_those_of_its_verified_final_refit() {
    let case = outer_case();
    let inputs = handoff::Inputs {
        root: &case.root,
        output: &case.output,
        seeds: Some(&case.seeds),
        vocabulary: None,
    };
    let (_, verified) = handoff::seal(&inputs).unwrap();
    let exported: Value = read_json(&case.predictor).unwrap();
    // Positive control: the honest final refit passes, its model parameters
    // encode to exactly the retained model bytes, and a grant is issued.
    let refit = lineage::verify_final_refit(&case.root, &verified, &case.predictor).unwrap();
    let retained = case
        ._authority
        .authority
        .join("objects")
        .join(&refit.model_sha256);
    assert_eq!(
        serde_json::to_vec(&exported["model"]).unwrap(),
        fs::read(retained).unwrap()
    );
    case.grant("honest", &case.sealed).unwrap();
    // One model parameter changed; recipe, training scope and completion
    // reference unchanged, and the handoff records the changed predictor.
    let mut forged = exported.clone();
    assert!(change_one_parameter(&mut forged["model"]), "{exported}");
    let forged = serde_json::to_vec_pretty(&forged).unwrap();
    fs::write(&case.predictor, &forged).unwrap();
    let mut handoff: Value = read_json(&case.handoff()).unwrap();
    handoff["predictor"]["sha256"] = json!(hash(&forged));
    write_json(&case.handoff(), &handoff).unwrap();
    let checked = lineage::verify_final_refit(&case.root, &verified, &case.predictor)
        .map(|r| r.model_sha256)
        .map_err(|e| format!("{e:#}"));
    let granted = case
        .grant("forged", &case.sealed)
        .map(|g| g["subject"].clone())
        .map_err(|e| format!("{e:#}"));
    let refused = "final refit model parameters are not the verified completion's model";
    assert!(
        matches!(&checked, Err(e) if e.contains(refused))
            && matches!(&granted, Err(e) if e.contains(refused)),
        "other model parameters passed as the verified final refit: \
         refit check {checked:?}; grant {granted:?}"
    );
    let grants = case.root.join("forged/grants");
    assert_eq!(fs::read_dir(grants).map(|d| d.count()).unwrap_or(0), 0);
}

/// The grant names the model identity its final-refit check verified, never
/// the handoff's unverified record of it.
#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
pub(super) fn an_outer_grant_names_the_model_its_final_refit_verified() {
    let case = outer_case();
    let honest = case.grant("honest", &case.sealed).unwrap();
    let mut handoff: Value = read_json(&case.handoff()).unwrap();
    let verified = handoff["final_refit"]["model_sha256"].clone();
    assert_eq!(honest["subject"]["model_sha256"], verified);
    handoff["final_refit"]["model_sha256"] = json!("e".repeat(64));
    write_json(&case.handoff(), &handoff).unwrap();
    let granted = case.grant("misrecorded", &case.sealed).unwrap();
    assert_eq!(
        granted["subject"]["model_sha256"], verified,
        "the grant names a model identity its final refit never verified: {granted}"
    );
}
