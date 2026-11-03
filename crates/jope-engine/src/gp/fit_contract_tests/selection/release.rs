use super::*;

/// W1-T06 at the search: what one search reads, fits, proposes and scores.
/// Backing rows outside every membership are forbidden parents: proposals,
/// fitting inputs and scores are identical and every fit is reused. The
/// cut's held-out labels are its authorized selection labels: fitting inputs
/// stay identical (fits reused) while scores change. A training label is an
/// authorized fitting input: the fitted subject changes.
#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
pub(crate) fn search_proposals_fits_and_scores_depend_only_on_their_permitted_rows() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let variants = [
        ("baseline", vec![]),
        (
            "unreferenced-backing-rows-changed",
            vec![(10, "900"), (11, "-7")],
        ),
        ("heldout-labels-changed", vec![(8, "150"), (9, "95.5")]),
        ("training-label-changed", vec![(3, "140")]),
    ];
    let prepared: Vec<PathBuf> = variants
        .iter()
        .map(|(name, labels)| label_variant(&root, name, labels))
        .collect();
    let training: BTreeSet<String> = prepared.iter().map(|p| cut_training_hash(p)).collect();
    assert_eq!(
        training.len(),
        2,
        "only the training-label variant changes its fitting rows"
    );
    let mut authority = Fixture::activated(&root, training.into_iter().collect());
    authority.start();
    // Duplicate initial proposals force variation draws whose parents are
    // chosen from measured candidates: later proposals depend on scores.
    let run = |index: usize| -> (Vec<Value>, i64) {
        let (name, _) = &variants[index];
        let seeds = root.join(format!("seeds-{name}.json"));
        seed_population(
            &prepared[index],
            &seeds,
            &[
                PROGRAM,
                "(pipe X   (ridge))",
                "(pipe X (standardize) (ridge))",
                "(pipe X  (ridge))",
            ],
            |_| None,
        );
        let output = root.join(format!("search-{name}"));
        success(
            search_client(&authority, &root, &prepared[index], &output, 4)
                .arg("--seeds")
                .arg(&seeds)
                .output()
                .unwrap(),
        );
        let rows: Vec<Value> = read_json(&output.join("observations.json")).unwrap();
        (rows, authority.entries().0)
    };
    // Permitted identifier differences: `prepared_sha256` and the seed file's
    // `seeds_sha256` name the (changed) preparation; they are not compared.
    let project = |rows: &[Value], key: &str| -> Vec<Value> {
        rows.iter()
            .map(|row| match key {
                "proposals" => {
                    let mut source = row["lineage"]["source"].clone();
                    if let Some(source) = source.as_object_mut() {
                        source.remove("seeds_sha256");
                    }
                    json!([row["genome"], source])
                }
                "fitting_inputs" => json!([
                    row["training_sha256"],
                    row["metrics"]["score_identity"]["fit_key"]
                ]),
                "fitted_state" => row["metrics"]["score_identity"]["model_sha256"].clone(),
                "scores" => json!([row["failure"].is_null(), row["metrics"]["rmse"]]),
                "predictions" => row["metrics"]["semantic_sha256"].clone(),
                _ => unreachable!(),
            })
            .collect()
    };
    let keys = [
        "proposals",
        "fitting_inputs",
        "fitted_state",
        "scores",
        "predictions",
    ];
    let (baseline, fits) = run(0);
    assert!(
        baseline
            .iter()
            .filter(|row| row["failure"].is_null())
            .count()
            >= 2
    );
    assert!(baseline[1]["lineage"]["source"]["kind"] == "variation");
    // (variant, per-key expectation: Some(true) equal, Some(false) differs at
    // slot 0, None declared; whether the fit registry must grow). A changed
    // score may steer later variation to new programs, so the held-out
    // variant's proposals and registry growth are recorded, not asserted.
    let expectations = [
        (
            1usize,
            [Some(true), Some(true), Some(true), Some(true), Some(true)],
            Some(false),
        ),
        (2, [None, None, None, Some(false), None], None),
        (3, [None, Some(false), Some(false), None, None], Some(true)),
    ];
    for (index, expected, new_fits) in expectations {
        let before_fits = authority.entries().0;
        let (rows, after_fits) = run(index);
        let mut comparisons = Vec::new();
        for (key, expect) in keys.iter().zip(expected) {
            let (a, b) = (project(&baseline, key), project(&rows, key));
            // The first slot is the same program in every variant.
            let (a0, b0) = (&a[0], &b[0]);
            match expect {
                Some(true) => assert_eq!(a, b, "{}: {key} changed", variants[index].0),
                Some(false) => assert_ne!(a0, b0, "{}: {key} ignored its input", variants[index].0),
                None => {}
            }
            comparisons.push(json!({"artifact":key,"expected":match expect {
                    Some(true) => "equal", Some(false) => "differs (slot 0)", None => "declared"},
                "observed":if a == b {"equal"} else {"differs"},"baseline":a,"mutated":b}));
        }
        if index == 2 {
            // Selection labels are not fitting inputs: slot 0 reuses its fit.
            assert_eq!(
                project(&baseline, "fitting_inputs")[0],
                project(&rows, "fitting_inputs")[0]
            );
            assert_eq!(rows[0]["metrics"]["evaluation_origin"], "reuse");
        }
        if let Some(grows) = new_fits {
            assert_eq!(
                after_fits > before_fits,
                grows,
                "{}: fit registry growth {before_fits} -> {after_fits}",
                variants[index].0
            );
        }
        if let Some(dir) = std::env::var_os("JOPE_W1_RECEIPTS") {
            let dir = Path::new(&dir).join("counterfactual");
            fs::create_dir_all(&dir).unwrap();
            write_json(
                &dir.join(format!("search-{}.json", variants[index].0)),
                &json!({"schema":"jope.w1-counterfactual-receipt/v1","task":"W1-T06",
                    "mutation":format!("search-{}", variants[index].0),
                    "perturbed":variants[index].1.iter().map(|(r, l)| json!({"backing_row":r,"label":l})).collect::<Vec<_>>(),
                    "frozen":"engine source, seed population programs, rng seed, evaluator, budget (4 evaluations), cut membership",
                    "tolerance":"exact equality of recorded identities and f64 bits; fits are single-threaded fixed-seed",
                    "permitted_identifier_differences":["prepared_sha256","lineage.source.seeds_sha256"],
                    "comparisons":comparisons,"fit_registry_entries":{"before":before_fits,"after":after_fits},
                    "receiver":"gp::search_seeded_with_rng_in_root through the fixture CLI"}),
            )
            .unwrap();
        }
    }
    assert!(fits > 0);
    authority.stop_service();
}

/// Raw-unit sealed final-test rows for the view fixture (`x`, `target`).
pub(crate) const SEALED_TEST: &str = "12,136\n13,139\n14,142.5\n";

/// A completed search of the view fixture and its exported selected predictor.
pub(super) fn exported_subject(
    authority: &Fixture,
    root: &Path,
    prepared: &Path,
) -> (PathBuf, PathBuf, PathBuf) {
    let (output, seeds) = handoff_search(authority, root, prepared);
    let predictor = root.join("selected.json");
    success(
        export_selected(authority, root, prepared, &output, &seeds, &predictor)
            .output()
            .unwrap(),
    );
    (output, seeds, predictor)
}

/// W1-T04, consumer-map route R14: `score` hashed whatever test file the
/// caller named against the caller's own selection receipt, so any label file
/// could be scored — it was never bound to the preparation's sealed-test
/// identity — and the metrics were published straight to the caller.
#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
pub(crate) fn a_selection_receipt_alone_neither_reads_caller_chosen_labels_nor_publishes_a_score() {
    let (temp, prepared) = view_fixture();
    let root = temp.path().canonicalize().unwrap();
    let prepared = root.join(prepared.strip_prefix(temp.path()).unwrap());
    let mut authority = Fixture::activated(&root, vec![cut_training_hash(&prepared)]);
    authority.start();
    let (_, _, predictor) = exported_subject(&authority, &root, &prepared);
    authority.stop_service();
    // Caller-chosen labels: not the preparation's sealed identity ("1"*64).
    let chosen = root.join("caller-chosen.csv");
    fs::write(&chosen, SEALED_TEST).unwrap();
    let p: Prepared = read_json(&prepared.join("prepared.json")).unwrap();
    assert_ne!(file_hash(&chosen).unwrap(), p.original_test_sha256);
    let evidence = root.join("selection-evidence.json");
    write_json(
        &evidence,
        &json!({"predictor_sha256":file_hash(&predictor).unwrap(),"partition":"meta_validation"}),
    )
    .unwrap();
    let receipt = root.join("selection.json");
    write_json(
        &receipt,
        &SelectionReceipt {
            schema: "parallel.selection.v1".into(),
            predictor_sha256: file_hash(&predictor).unwrap(),
            final_test_sha256: file_hash(&chosen).unwrap(),
            selection_evidence_path: evidence.clone(),
            selection_evidence_sha256: file_hash(&evidence).unwrap(),
            selected_on: "meta_validation".into(),
            frozen: true,
        },
    )
    .unwrap();
    let output = root.join("outer-score.json");
    let scored = score(&predictor, &chosen, &receipt, &output);
    assert!(
        scored.is_err(),
        "scored caller-chosen labels without a release grant: {}",
        fs::read_to_string(&output).unwrap_or_default()
    );
    let error = format!("{:#}", scored.unwrap_err());
    assert!(error.contains("outer release"), "{error}");
    assert!(!output.exists(), "a refused score was published");
}
