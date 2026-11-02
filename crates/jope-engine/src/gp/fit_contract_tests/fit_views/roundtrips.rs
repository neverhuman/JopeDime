use super::*;

#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
pub(super) fn fitted_predictor_roundtrip_matches_native_gp_and_restores_target_units() {
    let tmp = tempfile::tempdir().unwrap();
    let (train, training) = raw_training(tmp.path());
    let model = tmp.path().join("model.json");
    let mut authority = Fixture::activated(tmp.path(), vec![training]);
    authority.start();
    success(
        authority
            .command()
            .args(["fit", "--train"])
            .arg(&train)
            .args(["--genome", PROGRAM, "--output"])
            .arg(&model)
            .output()
            .unwrap(),
    );
    authority.assert_one_original_fit();
    let features = tmp.path().join("features.csv");
    let x = vec![vec![2.0, 4.0], vec![10.0, 100.0]];
    fs::write(&features, csv_bytes(&x).unwrap()).unwrap();
    let out = tmp.path().join("predictions.csv");
    predict(&model, &features, &out).unwrap();
    let predicted = fs::read_to_string(out)
        .unwrap()
        .lines()
        .map(|s| s.parse::<f64>().unwrap())
        .collect::<Vec<_>>();
    assert!(predicted.iter().all(|p| *p > 90.0 && *p < 190.0));
    let p: Predictor = read_json(&model).unwrap();
    let decoded: Model = serde_json::from_slice(&serde_json::to_vec(&p.model).unwrap()).unwrap();
    let tx = x
        .iter()
        .map(|r| {
            p.preprocessing
                .features(&r.iter().map(f64::to_string).collect::<Vec<_>>())
                .unwrap()
        })
        .collect::<Vec<_>>();
    assert_eq!(p.model.predict(&tx), decoded.predict(&tx));
    let expected = p
        .model
        .predict(&tx)
        .unwrap()
        .iter()
        .map(|v| v * p.preprocessing.target_scale + p.preprocessing.target_mean)
        .collect::<Vec<_>>();
    assert_eq!(predicted, expected);
    let completion_ref = p.fit_completion.as_ref().unwrap();
    let completion: Value = read_json(Path::new(completion_ref["path"].as_str().unwrap())).unwrap();
    // Library half of `verify-fit-completion`: the committed original is
    // returned byte-for-byte and is bound to this authority instance only.
    let instance = authority.open().instance().to_owned();
    let verified = authority
        .open()
        .verify_committed_completion(completion_ref["sha256"].as_str().unwrap())
        .unwrap();
    assert_eq!(serde_json::to_value(&verified).unwrap(), completion);
    assert_eq!(verified.authority_instance, instance);
    assert_ne!(verified.authority_instance, "f".repeat(64));

    // Evicting a retained model leaves the original exclusion intact. Exporting
    // under a different destination cannot authorize replacement fitting.
    fs::remove_file(
        authority
            .authority
            .join("objects")
            .join(completion["model_sha256"].as_str().unwrap()),
    )
    .unwrap();
    let replacement = tmp.path().join("replacement-model.json");
    let unavailable = authority
        .command()
        .args(["fit", "--train"])
        .arg(&train)
        .args(["--genome", PROGRAM, "--output"])
        .arg(&replacement)
        .output()
        .unwrap();
    assert!(!unavailable.status.success());
    assert!(!replacement.exists());
    authority.assert_one_original_fit();
    let stats = authority.stop_service();
    assert_eq!(stats["worker_spawns"], 1);
}

#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
pub(super) fn sealed_memory_view_matches_file_evaluation_and_rejects_writes() {
    use std::io::Write;
    use std::os::fd::AsRawFd;
    let tmp = tempfile::tempdir().unwrap();
    let rows = (0..30)
        .map(|i| vec![i as f64, (i * i) as f64, 3.0 + i as f64 * 0.8])
        .collect::<Vec<_>>();
    let mut memory = memory_view(&rows[..24], &rows[24..]).unwrap();
    assert!(memory.write_all(b"corrupt").is_err());
    let train = tmp.path().join("train.csv");
    let heldout = tmp.path().join("heldout.csv");
    let genome = tmp.path().join("p.dsl");
    fs::write(&train, csv_bytes(&rows[..24]).unwrap()).unwrap();
    fs::write(&heldout, csv_bytes(&rows[24..]).unwrap()).unwrap();
    fs::write(&genome, PROGRAM).unwrap();
    let a = tmp.path().join("a.json");
    let b = tmp.path().join("b.json");
    let mut authority = Fixture::activated(tmp.path(), vec![training_hash(&rows[..24])]);
    authority.start();
    success(
        authority
            .command()
            .args(["evaluate-worker", "--train"])
            .arg(&train)
            .arg("--validation")
            .arg(&heldout)
            .arg("--genome")
            .arg(&genome)
            .arg("--output")
            .arg(&a)
            .output()
            .unwrap(),
    );
    success(
        authority
            .command()
            .args(["evaluate-memory", "--fd"])
            .arg(memory.as_raw_fd().to_string())
            .arg("--genome")
            .arg(&genome)
            .arg("--output")
            .arg(&b)
            .output()
            .unwrap(),
    );
    let a: Value = read_json(&a).unwrap();
    let b: Value = read_json(&b).unwrap();
    assert_eq!(a["semantic_sha256"], b["semantic_sha256"]);
    assert_eq!(a["rmse"], b["rmse"]);
    assert_eq!(a["physical_fit_starts"], 1);
    assert_eq!(b["physical_fit_starts"], 0);
    assert_eq!(b["evaluation_origin"], "reuse");
    authority.assert_one_original_fit();
    let stats = authority.stop_service();
    assert_eq!(stats["worker_spawns"], 1);
    assert_eq!(stats["committed"], 1);
    assert_eq!(stats["reused"], 1);
    assert_eq!(stats["authority_opens"], 1);
}

#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
pub(super) fn fitted_view_predictor_uses_the_exact_cut_and_original_target_units() {
    let (temp, prepared) = view_fixture();
    let output = temp.path().join("predictor.json");
    let mut authority = Fixture::activated(temp.path(), vec![cut_training_hash(&prepared)]);
    authority.start();
    success(
        authority
            .command()
            .args(["fit-view", "--root"])
            .arg(temp.path())
            .arg("--prepared")
            .arg(&prepared)
            .args(["--cut", "0", "--genome", PROGRAM, "--output"])
            .arg(&output)
            .output()
            .unwrap(),
    );
    let predictor: Predictor = read_json(&output).unwrap();
    let p: Prepared = read_json(&prepared.join("prepared.json")).unwrap();
    assert_eq!(predictor.training_sha256, p.cuts[0].training_sha256);
    let rows = load_view(&prepared, 0, true).unwrap();
    let x = rows.iter().map(|r| r[..1].to_vec()).collect::<Vec<_>>();
    let y = rows.iter().map(|r| r[1]).collect::<Vec<_>>();
    let expected = feat_math::fit(&GenomeIr::parse(PROGRAM).unwrap(), &x, &y).unwrap();
    assert_eq!(predictor.model.predict(&x), expected.predict(&x));
    let features = temp.path().join("features.csv");
    fs::write(&features, "3\n7\n").unwrap();
    let predictions = temp.path().join("predictions.csv");
    predict(&output, &features, &predictions).unwrap();
    let predictions = fs::read_to_string(predictions).unwrap();
    assert!(predictions
        .lines()
        .all(|r| r.parse::<f64>().unwrap() > 100.0));
    assert!(!prepared.join("cut-0/train.csv").exists());
    authority.assert_one_original_fit();
}

#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
pub(super) fn isolated_fitted_view_export_binds_success_and_preserves_worker_failures() {
    let (temp, prepared) = view_fixture();
    let executable = std::env::current_exe().unwrap();
    let mut authority = Fixture::activated(temp.path(), vec![cut_training_hash(&prepared)]);
    authority.start();
    let isolated = |genome: &str, output: &Path| {
        authority
            .command()
            .args(["fit-view-isolated", "--executable"])
            .arg(&executable)
            .arg("--root")
            .arg(temp.path())
            .arg("--prepared")
            .arg(&prepared)
            .args(["--cut", "0", "--genome", genome, "--output"])
            .arg(output)
            .output()
            .unwrap()
    };
    let output = temp.path().join("isolated.json");
    success(isolated(PROGRAM, &output));
    let receipt: Value = read_json(&output.with_extension("fit.json")).unwrap();
    assert_eq!(receipt["passed"], true);
    assert_eq!(receipt["predictor_sha256"], file_hash(&output).unwrap());
    assert_eq!(receipt["threads"], 1);
    assert_eq!(receipt["memory_limit_gib"], 8);
    let failed = temp.path().join("failed.json");
    assert!(!isolated("(pipe X (unknown))", &failed).status.success());
    let receipt: Value = read_json(&failed.with_extension("fit.json")).unwrap();
    assert_eq!(receipt["passed"], false);
    assert!(!receipt["failure"].as_str().unwrap().is_empty());
    assert!(!failed.exists());
    assert!(!fs::read_to_string(failed.with_extension("fit.log"))
        .unwrap()
        .is_empty());
    // The unknown operator is a deterministic numerical failure: it reaches the
    // authorized worker once and is retained as that original terminal outcome.
    assert_eq!(authority.entry_states(), ["completed", "terminal_failure"]);
}
