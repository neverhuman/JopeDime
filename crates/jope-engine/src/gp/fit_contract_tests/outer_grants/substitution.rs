use super::*;
use crate::gp::outer_release as release_authority;

/// The same receiver, for the final refit's fit completion
/// (`lineage::verify_final_refit`): the completion consumed is the one whose
/// digest the predictor records.
#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
pub(crate) fn an_outer_grant_consumes_the_fit_completion_it_verified() {
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

/// The authority's own records: the ledger binds the digest of exactly the
/// grant and withheld score the authority wrote, never a later reread.
#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
pub(crate) fn the_release_ledger_binds_the_bytes_the_authority_wrote() {
    let case = outer_case();
    let other = hash(OTHER_TEST.as_bytes());
    // A grant replaced right after it is written, naming other labels.
    let dir = case.root.join("grant-swap");
    let outer = release_authority::Authority::initialize(&case.root.join("grant-swap")).unwrap();
    let grants = dir.canonicalize().unwrap().join("grants");
    let relative = case.other.strip_prefix(&case.root).unwrap().to_path_buf();
    let path = {
        let _hook = swap_written(grants, move |grant| {
            grant["scoring_labels"]["path"] = json!(relative);
            grant["scoring_labels"]["sha256"] = json!(other);
        });
        outer
            .grant(&release_authority::GrantRequest {
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
    let outer = release_authority::Authority::initialize(&case.root.join("withheld-swap")).unwrap();
    let withheld = dir.canonicalize().unwrap().join("withheld");
    let path = outer
        .grant(&release_authority::GrantRequest {
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

/// `lineage::verify_final_refit` bound the exported predictor's recipe,
/// training scope and fit-completion reference, but not the model parameters
/// the predictor serves. Those must be the model the verified completion
/// retained (its `model_sha256`, as the fit authority defines it).
#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
pub(crate) fn an_exported_predictors_model_parameters_are_those_of_its_verified_final_refit() {
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
