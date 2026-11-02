use super::*;

/// Seed population whose programs equal `base`'s but whose origin labels are
/// all `origin`: only the labels differ.
pub(super) fn relabelled_population(base: &Path, path: &Path, origin: &str) {
    let mut population: SeedPopulation = read_json(base).unwrap();
    for candidate in &mut population.candidates {
        candidate.origin = origin.into();
    }
    write_json(path, &population).unwrap();
}

#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
pub(super) fn a_paired_comparison_names_its_sole_arm_difference_and_refuses_any_other() {
    let (temp, prepared) = view_fixture();
    let root = temp.path().canonicalize().unwrap();
    let mut authority = Fixture::activated(&root, vec![cut_training_hash(&prepared)]);
    authority.start();
    let neural = root.join("seeds-neural.json");
    seed_population(
        &prepared,
        &neural,
        &["(pipe X (standardize) (ridge))", UNFITTABLE, PROGRAM],
        |_| None,
    );
    let shuffled = root.join("seeds-shuffled.json");
    relabelled_population(&neural, &shuffled, "shuffled_context");
    let run = |name: &str, extra: &[&std::ffi::OsStr]| {
        let output = root.join(name);
        success(
            search_client(&authority, &root, &prepared, &output, 3)
                .args(extra)
                .output()
                .unwrap(),
        );
        output
    };
    let flag = std::ffi::OsStr::new;
    let model = run("arm-model", &[flag("--seeds"), neural.as_os_str()]);
    let relabelled = run("arm-relabelled", &[flag("--seeds"), shuffled.as_os_str()]);
    let generic = run("arm-generic", &[]);
    let reseeded = run(
        "arm-reseeded",
        &[
            flag("--seeds"),
            shuffled.as_os_str(),
            flag("--rng-seed"),
            flag("7"),
        ],
    );
    fn arm<'a>(label: &'a str, output: &'a Path) -> paired::Arm<'a> {
        paired::Arm { label, output }
    }
    // Origin labels are the sole difference: every shared program carries the
    // same measurement, admissibility and budget in both arms.
    let manifest = paired::compare(
        &root,
        &[arm("model", &model), arm("relabelled", &relabelled)],
        &["seeds_sha256"],
    )
    .unwrap();
    assert_eq!(
        manifest["observed_request_differences"],
        json!(["seeds_sha256"])
    );
    assert_eq!(manifest["shared_programs"], 3);
    assert_eq!(
        manifest["shared_programs_with_divergent_outcomes"],
        json!([])
    );
    let arms = manifest["arms"].as_array().unwrap();
    assert_eq!(arms[0]["winner"]["program"], arms[1]["winner"]["program"]);
    assert_eq!(
        arms[0]["winner"]["objective_rmse_original_units"],
        arms[1]["winner"]["objective_rmse_original_units"]
    );
    for key in [
        "completed_evaluations",
        "measured",
        "failed",
        "proposal_draws",
    ] {
        assert_eq!(arms[0]["counters"][key], arms[1]["counters"][key], "{key}");
    }
    assert_eq!(arms[0]["generation_zero"]["origins"]["neural_policy"], 3);
    assert_eq!(arms[1]["generation_zero"]["origins"]["shuffled_context"], 3);
    // The second arm reused the first arm's fits: counted apart, not hidden.
    assert_eq!(arms[0]["counters"]["recorded_physical_fit_starts"], 2);
    assert_eq!(arms[1]["counters"]["recorded_physical_fit_starts"], 0);
    assert_eq!(arms[1]["counters"]["reused_completions"], 2);
    // The deterministic fit failure has no recorded start outcome in either arm.
    assert_eq!(arms[0]["counters"]["unknown_start_outcomes"], 1);
    assert!(manifest["feasibility_protocol"].is_string());
    // Seeded against generic: the seed population is the declared difference.
    paired::compare(
        &root,
        &[arm("model", &model), arm("generic", &generic)],
        &["seeds_sha256"],
    )
    .unwrap();
    // Undeclared, overdeclared and empty declarations are refused.
    for (arms, declared, expected) in [
        (
            vec![arm("model", &model), arm("reseeded", &reseeded)],
            vec!["seeds_sha256"],
            "not exactly the declared",
        ),
        (
            vec![arm("model", &model), arm("relabelled", &relabelled)],
            vec!["seeds_sha256", "rng_seed"],
            "not exactly the declared",
        ),
        (
            vec![arm("model", &model), arm("relabelled", &relabelled)],
            vec![],
            "must declare",
        ),
        (
            vec![arm("model", &model), arm("model", &relabelled)],
            vec!["seeds_sha256"],
            "unique",
        ),
    ] {
        let error = format!(
            "{:#}",
            paired::compare(&root, &arms, &declared).unwrap_err()
        );
        assert!(error.contains(expected), "{declared:?}: {error}");
    }
    let manifest = paired::compare(
        &root,
        &[arm("model", &model), arm("reseeded", &reseeded)],
        &["seeds_sha256", "rng_seed"],
    )
    .unwrap();
    assert_eq!(
        manifest["observed_request_differences"],
        json!(["rng_seed", "seeds_sha256"])
    );
    // A shared program whose measurement differs between arms is refused.
    let rows: Vec<Value> = read_json(&relabelled.join("observations.json")).unwrap();
    let files = ["observations.jsonl", "observations.json", "search.json"];
    let original: Vec<Vec<u8>> = files
        .iter()
        .map(|name| fs::read(relabelled.join(name)).unwrap())
        .collect();
    let measured = rows
        .iter()
        .position(|row| row["failure"].is_null())
        .unwrap();
    let mut forged = rows.clone();
    forged[measured]["metrics"]["rmse"] = json!(0.5);
    republish_ledger(&relabelled, &forged);
    assert!(paired::compare(
        &root,
        &[arm("model", &model), arm("relabelled", &relabelled)],
        &["seeds_sha256"],
    )
    .is_err());
    for (name, bytes) in files.iter().zip(&original) {
        fs::write(relabelled.join(name), bytes).unwrap();
    }
    authority.stop_service();
}

#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
pub(super) fn a_refused_proposal_in_one_slot_does_not_shift_another_slots_random_streams() {
    let (temp, prepared) = view_fixture();
    let root = temp.path().canonicalize().unwrap();
    let mut authority = Fixture::activated(&root, vec![cut_training_hash(&prepared)]);
    authority.start();
    let run = |seeds: &Path, name: &str| -> Vec<Value> {
        let output = root.join(name);
        success(
            search_client(&authority, &root, &prepared, &output, 3)
                .arg("--seeds")
                .arg(seeds)
                .output()
                .unwrap(),
        );
        read_json(&output.join("observations.json")).unwrap()
    };
    // Arm B: slot 1's initial proposal duplicates slot 0 and is refused, so
    // slot 1 is filled by a variation draw. In both arms slot 2's initial
    // proposal is a duplicate too, so slot 2 is filled by variation.
    let b = root.join("streams-b.json");
    seed_population(
        &prepared,
        &b,
        &[PROGRAM, "(pipe X   (ridge))", "(pipe X  (ridge))"],
        |_| None,
    );
    let rows_b = run(&b, "streams-arm-b");
    assert_eq!(rows_b[1]["lineage"]["source"]["kind"], "variation");
    // Arm A seeds slot 1 with exactly the program arm B's variation produced:
    // both arms enter slot 2 with the same ledger, parent pool and evaluated
    // set, and differ only by one refused draw in slot 1.
    let a = root.join("streams-a.json");
    let slot_one = rows_b[1]["genome"].as_str().unwrap();
    seed_population(
        &prepared,
        &a,
        &[PROGRAM, slot_one, "(pipe X  (ridge))"],
        |_| None,
    );
    let rows_a = run(&a, "streams-arm-a");
    assert_eq!(
        rows_a[1]["lineage"]["source"]["kind"],
        "seed_population_slot"
    );
    for row in 0..2 {
        assert_eq!(rows_a[row]["genome"], rows_b[row]["genome"]);
        for key in ["rmse", "semantic_sha256"] {
            assert_eq!(rows_a[row]["metrics"][key], rows_b[row]["metrics"][key]);
        }
    }
    let counter = |row: &Value| row["proposal_attempts"].as_u64().unwrap();
    assert!(
        counter(&rows_b[1]) > counter(&rows_a[1]),
        "arm B drew more proposals before slot 2"
    );
    // Slot 2 walks the same keyed stream sequence in both arms: the same
    // accepted draw, words, operator, parents and program.
    let slot_two = |rows: &[Value]| {
        let source = &rows[2]["lineage"]["source"];
        json!({"draw":counter(&rows[2]) - counter(&rows[1]) - 1,
            "kind":source["kind"],"operator":source["operator"],"parents":source["parents"],
            "variation_seed":source["variation_seed"],"operator_word":source["operator_word"],
            "genome":rows[2]["genome"]})
    };
    assert_eq!(
        slot_two(&rows_a),
        slot_two(&rows_b),
        "a refused proposal in slot 1 shifted slot 2's random streams"
    );
    for rows in [&rows_a, &rows_b] {
        let source = &rows[2]["lineage"]["source"];
        assert_eq!(source["kind"], "variation", "{source}");
        assert_eq!(source["slot"], 2, "{source}");
        assert_eq!(source["slot_draw"], slot_two(rows)["draw"], "{source}");
    }
    authority.stop_service();
}

/// Rewrite the fixture preparation's family partition (identity recomputed by
/// every consumer, as a producer of that partition would publish it).
pub(super) fn set_partition(prepared: &Path, partition: &str) {
    let mut p: Value = read_json(&prepared.join("prepared.json")).unwrap();
    p["meta_partition"] = json!(partition);
    write_json(&prepared.join("prepared.json"), &p).unwrap();
}

/// W1-T04, consumer-map route R13: `--qualification` admitted a search of a
/// sealed `meta_test` family. Its observations are teacher/learning inputs, so
/// the sealed family would reach learning through the search. The refusal
/// precedes every view load, output reservation and fit claim.
#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
pub(super) fn qualification_search_of_a_sealed_meta_test_family_is_refused_before_any_view_or_fit()
{
    let (temp, prepared) = view_fixture();
    let root = temp.path().canonicalize().unwrap();
    let mut authority = Fixture::activated(&root, vec![cut_training_hash(&prepared)]);
    authority.start();
    set_partition(&prepared, "meta_test");
    let sealed = root.join("search-sealed-family");
    let refused = search_client(&authority, &root, &prepared, &sealed, 1)
        .output()
        .unwrap();
    let error = String::from_utf8_lossy(&refused.stderr);
    assert!(
        !refused.status.success(),
        "a sealed meta_test family was searched under --qualification"
    );
    assert!(error.contains("sealed meta_test"), "{error}");
    assert!(
        !sealed.exists(),
        "a refused search left search state behind"
    );
    authority.assert_no_fit_starts();
    // Positive control: the identical view as a meta-validation family is
    // still admitted for qualification (adaptive development evidence).
    set_partition(&prepared, "meta_validation");
    let admitted = root.join("search-validation-family");
    success(
        search_client(&authority, &root, &prepared, &admitted, 1)
            .output()
            .unwrap(),
    );
    let search: Value = read_json(&admitted.join("search.json")).unwrap();
    assert_eq!(search["final_tests_opened"], false);
    assert_eq!(authority.entries(), (1, 1));
    authority.stop_service();
}

/// The view fixture's preparation over its own backing file, with the given
/// raw label cells replaced. Cut 0 trains on rows 0..8 and validates on 8, 9;
/// rows 10 and 11 sit in the backing file outside every membership, where a
/// shared source keeps rows no stage may use.
pub(super) fn label_variant(root: &Path, name: &str, labels: &[(usize, &str)]) -> PathBuf {
    fs::create_dir_all(root.join("manifests")).unwrap();
    fs::write(root.join("Cargo.toml"), "[workspace]\n").unwrap();
    fs::create_dir_all(root.join("data/originals")).unwrap();
    let mut raw = (0..12)
        .map(|i| vec![i.to_string(), (100 + i * 3).to_string()])
        .collect::<Vec<_>>();
    for (row, label) in labels {
        raw[*row][1] = label.to_string();
    }
    let backing = format!("data/originals/{name}.csv");
    let mut source = "x,target\n".to_string();
    for row in &raw {
        source.push_str(&format!("{},{}\n", row[0], row[1]));
    }
    fs::write(root.join(&backing), source).unwrap();
    let pp = Preprocessing::fit(&raw[..8], "regression").unwrap();
    let prepared = root.join("data/prepared").join(name);
    let dir = prepared.join("cut-0");
    fs::create_dir_all(&dir).unwrap();
    write_json(&dir.join("preprocessing.json"), &pp).unwrap();
    let tr = csv_bytes(&pp.transform(&raw[..8]).unwrap()).unwrap();
    let va = csv_bytes(&pp.transform(&raw[8..10]).unwrap()).unwrap();
    let p = json!({"schema":"parallel.prepared.v1","dataset":"example","family":"example",
        "task":"regression","meta_partition":"meta_train","seed":SEED,"family_partition_frozen":false,
        "clean_evaluation":true,"clean_status":"raw_reconstructed","original_train_sha256":"0".repeat(64),
        "original_test_sha256":"1".repeat(64),"selected_train_indices":(0..10).collect::<Vec<_>>(),
        "selected_test_indices":[0,1],"target_last":true,"rows_total":12,
        "cuts":[{"index":0,"train_indices":(0..8).collect::<Vec<_>>(),"validation_indices":[8,9],
            "training_sha256":hash(&tr),"validation_sha256":hash(&va),
            "preprocessing_sha256":file_hash(&dir.join("preprocessing.json")).unwrap()}],
        "storage":{"source":backing,"source_sha256":file_hash(&root.join(&backing)).unwrap(),
            "feature_columns":["x"],"target_column":"target",
            "original_training_row_map":(0..10).collect::<Vec<_>>()}});
    write_json(&prepared.join("prepared.json"), &p).unwrap();
    prepared
}

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
pub(super) fn search_proposals_fits_and_scores_depend_only_on_their_permitted_rows() {
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
pub(super) const SEALED_TEST: &str = "12,136\n13,139\n14,142.5\n";

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
pub(super) fn a_selection_receipt_alone_neither_reads_caller_chosen_labels_nor_publishes_a_score() {
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
