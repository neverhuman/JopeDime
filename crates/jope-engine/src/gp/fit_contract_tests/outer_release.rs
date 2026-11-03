use super::*;

/// Copy a retained receipt to `$JOPE_W1_RECEIPTS/<group>/` when set.
pub(crate) fn retain(group: &str, name: &str, path: &Path) {
    if let Some(dir) = std::env::var_os("JOPE_W1_RECEIPTS") {
        let dir = Path::new(&dir).join(group);
        fs::create_dir_all(&dir).unwrap();
        fs::copy(path, dir.join(name)).unwrap();
    }
}

pub(crate) fn refusal<T: std::fmt::Debug>(result: Result<T>) -> String {
    format!(
        "{:#}",
        result.expect_err("the release authority accepted a request it must refuse")
    )
}

/// W1-T04 acceptance: sealed labels reach only a granted outer scorer; the
/// score stays withheld until the separate authority releases it, once.
#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
pub(crate) fn outer_scores_need_a_grant_read_only_the_granted_labels_and_release_once() {
    use crate::gp::outer_release::{Authority as Release, GrantRequest};
    let (temp, prepared) = view_fixture();
    let root = temp.path().canonicalize().unwrap();
    let prepared = root.join(prepared.strip_prefix(temp.path()).unwrap());
    // The preparation binds its sealed-test identity; the file itself does
    // not exist while search, selection and export run (denied upstream).
    let mut p: Value = read_json(&prepared.join("prepared.json")).unwrap();
    p["original_test_sha256"] = json!(hash(SEALED_TEST.as_bytes()));
    write_json(&prepared.join("prepared.json"), &p).unwrap();
    let sealed = root.join("data/originals/sealed-test.csv");
    let mut authority = Fixture::activated(&root, vec![cut_training_hash(&prepared)]);
    authority.start();
    let (output, seeds, predictor) = exported_subject(&authority, &root, &prepared);
    assert!(!sealed.exists());
    let fits = authority.entries();
    fs::write(&sealed, SEALED_TEST).unwrap();
    let outer_dir = root.join("outer-authority");
    let outer = Release::initialize(&outer_dir).unwrap();
    let search = handoff::Inputs {
        root: &root,
        output: &output,
        seeds: Some(&seeds),
        vocabulary: None,
    };
    // Grants: a symlinked label path and an unexported subject are refused.
    let link = root.join("data/originals/sealed-link.csv");
    std::os::unix::fs::symlink(&sealed, &link).unwrap();
    let error = refusal(outer.grant(&GrantRequest {
        search,
        predictor: &predictor,
        test: &link,
    }));
    assert!(error.contains("symlink"), "{error}");
    let orphan = root.join("unexported.json");
    fs::copy(&predictor, &orphan).unwrap();
    let error = refusal(outer.grant(&GrantRequest {
        search,
        predictor: &orphan,
        test: &sealed,
    }));
    assert!(error.contains("handoff"), "{error}");
    let grant = outer
        .grant(&GrantRequest {
            search,
            predictor: &predictor,
            test: &sealed,
        })
        .unwrap();
    let granted: Value = read_json(&grant).unwrap();
    assert_eq!(granted["scoring_labels"]["role"], "sealed_final");
    assert_eq!(
        granted["scoring_labels"]["path"],
        "data/originals/sealed-test.csv"
    );
    assert_eq!(
        granted["scoring_labels"]["sha256"],
        hash(SEALED_TEST.as_bytes())
    );
    assert_eq!(
        granted["subject"]["predictor"]["sha256"],
        file_hash(&predictor).unwrap()
    );
    let error = refusal(outer.grant(&GrantRequest {
        search,
        predictor: &predictor,
        test: &sealed,
    }));
    assert!(error.contains("already granted"), "{error}");

    // Refused label reads, each before the first label byte is read.
    let receipts = root.join("outer-receipts");
    let mut forged = granted.clone();
    forged["scoring_labels"]["path"] = json!("data/originals/raw.csv");
    let forged_path = root.join("forged-grant.json");
    write_json(&forged_path, &forged).unwrap();
    let refused_forged = receipts.join("refused-forged-grant.json");
    let error = refusal(outer.score(&forged_path, &predictor, &sealed, &refused_forged));
    assert!(error.contains("not recorded"), "{error}");
    let other_model = root.join("other-model.json");
    let reencoded: Value = read_json(&predictor).unwrap();
    fs::write(&other_model, serde_json::to_vec(&reencoded).unwrap()).unwrap();
    let refused_model = receipts.join("refused-other-model.json");
    let error = refusal(outer.score(&grant, &other_model, &sealed, &refused_model));
    assert!(error.contains("not the granted fitted subject"), "{error}");
    let copy = root.join("data/originals/sealed-copy.csv");
    fs::write(&copy, SEALED_TEST).unwrap();
    let refused_copy = receipts.join("refused-other-label-file.json");
    let error = refusal(outer.score(&grant, &predictor, &copy, &refused_copy));
    assert!(error.contains("not the granted one"), "{error}");
    for path in [&refused_forged, &refused_model, &refused_copy] {
        let receipt: Value = read_json(path).unwrap();
        assert_eq!(receipt["status"], "refused", "{receipt}");
        assert_eq!(receipt["labels_opened"], false, "{receipt}");
    }

    // The valid score, through the shipped `score` entry: withheld.
    let scored_receipt = receipts.join("scored-withheld.json");
    success(
        authority
            .command()
            .args(["score", "--model"])
            .arg(&predictor)
            .arg("--test")
            .arg(&sealed)
            .arg("--receipt")
            .arg(&grant)
            .arg("--output")
            .arg(&scored_receipt)
            .env(crate::gp::outer_release::AUTHORITY_ENV, &outer_dir)
            .output()
            .unwrap(),
    );
    let scored: Value = read_json(&scored_receipt).unwrap();
    assert_eq!(scored["status"], "scored_withheld");
    assert_eq!(scored["released"], false);
    assert!(!fs::read_to_string(&scored_receipt)
        .unwrap()
        .contains("rmse"));
    // A retry cannot obtain a second look.
    let refused_retry = receipts.join("refused-retry.json");
    let error = refusal(outer.score(&grant, &predictor, &sealed, &refused_retry));
    assert!(error.contains("consumed"), "{error}");

    // Release, once; the released metrics are the granted labels' metrics.
    let grant_id = granted["grant_id"].as_str().unwrap();
    let visible = root.join("released-outer-score.json");
    outer.release(grant_id, &visible).unwrap();
    let released: Value = read_json(&visible).unwrap();
    let fitted: Predictor = read_json(&predictor).unwrap();
    let rows: Vec<(&str, f64)> = SEALED_TEST
        .lines()
        .map(|line| {
            let (x, y) = line.split_once(',').unwrap();
            (x, y.parse().unwrap())
        })
        .collect();
    let x: Vec<Vec<f64>> = rows
        .iter()
        .map(|(x, _)| fitted.preprocessing.features(&[x.to_string()]).unwrap())
        .collect();
    let predicted = fitted.model.predict(&x).unwrap();
    let squared = rows
        .iter()
        .zip(&predicted)
        .map(|((_, y), p)| {
            (y - (p * fitted.preprocessing.target_scale + fitted.preprocessing.target_mean)).powi(2)
        })
        .sum::<f64>();
    let rmse = (squared / rows.len() as f64).sqrt();
    let reported = released["metrics"]["rmse"].as_f64().unwrap();
    assert!(
        (reported - rmse).abs() <= 1e-12 * rmse.abs().max(1.0),
        "released RMSE {reported} is not the granted labels' RMSE {rmse}"
    );
    assert_eq!(released["scoring_labels"]["rows"], 3);
    assert_eq!(released["per_row_values_included"], false);
    assert_eq!(
        released["preparation"]["clean_status"], p["clean_status"],
        "a released score must carry its preparation's evaluation status"
    );
    let again = root.join("released-again.json");
    let error = refusal(outer.release(grant_id, &again));
    assert!(error.contains("already released"), "{error}");
    assert!(!again.exists());

    // The append-only history records every decision; only the one granted
    // attempt ever opened sealed labels. No fit was started for scoring.
    let events = outer.events().unwrap();
    let kinds: Vec<&str> = events
        .iter()
        .map(|event| event["event"].as_str().unwrap())
        .collect();
    assert_eq!(
        kinds,
        [
            "grant_refused",
            "grant_refused",
            "granted",
            "grant_refused",
            "score_refused",
            "score_refused",
            "score_refused",
            "score_started",
            "scored",
            "score_refused",
            "released",
            "release_refused"
        ]
    );
    assert!(events
        .iter()
        .filter(|event| event["event"] == "score_refused")
        .all(|event| event["labels_opened"] == false));
    assert_eq!(authority.entries(), fits);
    let ledger = outer_dir.join("ledger.jsonl");
    for (name, path) in [
        ("ledger.jsonl", &ledger),
        ("grant.json", &grant),
        ("scored-withheld-receipt.json", &scored_receipt),
        ("released-outer-score.json", &visible),
        ("refused-label-read-forged-grant.json", &refused_forged),
        ("refused-label-read-other-file.json", &refused_copy),
        ("refused-retry.json", &refused_retry),
    ] {
        retain("outer-release", name, path);
    }
    // Removing a recorded event breaks the chain: the authority refuses to open.
    let original = fs::read(&ledger).unwrap();
    let kept: Vec<u8> = original
        .split_inclusive(|b| *b == b'\n')
        .enumerate()
        .filter(|(i, _)| *i != 10)
        .flat_map(|(_, line)| line.iter().copied())
        .collect();
    fs::write(&ledger, kept).unwrap();
    assert!(refusal(Release::open(&outer_dir)).contains("chain"));
    fs::write(&ledger, original).unwrap();
    Release::open(&outer_dir).unwrap();
    authority.stop_service();
}

/// The shipped `outer-release` clap command, parsed from argv and run by the
/// same `outer_release_cli::run` the binary calls. The shipped binary cannot
/// produce an exported subject (every fit is fenced), so its positive path is
/// proven here; `tests/outer_release_cli.rs` proves the binary's wiring.
#[derive(clap::Parser)]
#[command(name = "jope-parallel")]
pub(crate) struct OuterReleaseArgv {
    #[command(subcommand)]
    command: OuterReleaseVerb,
}
#[derive(clap::Subcommand)]
pub(crate) enum OuterReleaseVerb {
    OuterRelease {
        #[command(subcommand)]
        command: outer_release_cli::OuterReleaseCommand,
    },
}
pub(crate) fn outer_release_cli_run(argv: &[&str]) -> Result<Value> {
    use clap::Parser;
    let OuterReleaseVerb::OuterRelease { command } = OuterReleaseArgv::try_parse_from(
        std::iter::once("jope-parallel").chain(argv.iter().copied()),
    )?
    .command;
    outer_release_cli::run(command)
}

/// W1 follow-up 2b: init, grant, score, release and events through the CLI
/// verbs, with the same one-grant, one-attempt, one-release rules.
#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
pub(crate) fn the_outer_release_cli_grants_scores_and_releases_once() {
    let (temp, prepared) = view_fixture();
    let root = temp.path().canonicalize().unwrap();
    let prepared = root.join(prepared.strip_prefix(temp.path()).unwrap());
    let mut p: Value = read_json(&prepared.join("prepared.json")).unwrap();
    p["original_test_sha256"] = json!(hash(SEALED_TEST.as_bytes()));
    write_json(&prepared.join("prepared.json"), &p).unwrap();
    let mut authority = Fixture::activated(&root, vec![cut_training_hash(&prepared)]);
    authority.start();
    let (output, seeds, predictor) = exported_subject(&authority, &root, &prepared);
    authority.stop_service();
    let sealed = root.join("data/originals/sealed-test.csv");
    fs::write(&sealed, SEALED_TEST).unwrap();
    let dir = root.join("outer-cli-authority");
    let s = |p: &Path| p.to_str().unwrap().to_owned();
    let (a, r) = (s(&dir), s(&root));
    let init = outer_release_cli_run(&["outer-release", "init", "--authority", &a]).unwrap();
    assert_eq!(init["instance"].as_str().map(str::len), Some(64));
    let grant_args = [
        "outer-release",
        "grant",
        "--authority",
        &a,
        "--root",
        &r,
        "--search-output",
        &s(&output),
        "--seeds",
        &s(&seeds),
        "--predictor",
        &s(&predictor),
        "--test",
        &s(&sealed),
    ];
    let granted = outer_release_cli_run(&grant_args).unwrap();
    let grant_id = granted["grant_id"].as_str().unwrap().to_owned();
    let error = format!("{:#}", outer_release_cli_run(&grant_args).unwrap_err());
    assert!(error.contains("already granted"), "{error}");
    let receipt = root.join("cli-scored.json");
    let score = |receipt: &Path| {
        outer_release_cli_run(&[
            "outer-release",
            "score",
            "--authority",
            &a,
            "--grant",
            granted["grant"].as_str().unwrap(),
            "--model",
            &s(&predictor),
            "--test",
            &s(&sealed),
            "--receipt",
            &s(receipt),
        ])
    };
    let scored = score(&receipt).unwrap();
    assert_eq!(scored["status"], "scored_withheld");
    assert!(!fs::read_to_string(&receipt).unwrap().contains("rmse"));
    assert!(score(&root.join("cli-retry.json")).is_err());
    let visible = root.join("cli-released.json");
    let release = [
        "outer-release",
        "release",
        "--authority",
        &a,
        "--grant-id",
        &grant_id,
        "--output",
        &s(&visible),
    ];
    outer_release_cli_run(&release).unwrap();
    let released: Value = read_json(&visible).unwrap();
    assert!(released["metrics"]["rmse"].as_f64().unwrap().is_finite());
    assert!(outer_release_cli_run(&release).is_err());
    let events = outer_release_cli_run(&["outer-release", "events", "--authority", &a]).unwrap();
    let kinds: Vec<&str> = events["events"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e["event"].as_str().unwrap())
        .collect();
    assert_eq!(
        kinds,
        [
            "granted",
            "grant_refused",
            "score_started",
            "scored",
            "score_refused",
            "released",
            "release_refused"
        ]
    );
    // An unknown verb or a missing required flag is a parse refusal, not a call.
    assert!(outer_release_cli_run(&["outer-release", "publish", "--authority", &a]).is_err());
    assert!(outer_release_cli_run(&["outer-release", "grant", "--authority", &a]).is_err());
}

/// W1 follow-up 2c (consumer-map route R2 at the export worker): the fitted
/// predictor export worker loaded its own view from the prepared path and
/// resolved the backing table under the nearest ancestor holding
/// `manifests/` and `Cargo.toml` - not under the search's declared root. The
/// parent must project the fitting-only input under the declared root and
/// hand the worker only that sealed projection.
#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
pub(crate) fn the_export_worker_receives_its_projected_input_and_never_resolves_the_view_itself() {
    let temp = tempfile::tempdir().unwrap();
    let outer = temp.path().canonicalize().unwrap();
    let root = outer.join("declared");
    fs::create_dir_all(&root).unwrap();
    let prepared = view_fixture_in(&root, "example", &[8, 9]);
    let mut authority = Fixture::activated(&root, vec![cut_training_hash(&prepared)]);
    authority.start();
    let (output, seeds) = handoff_search(&authority, &root, &prepared);
    // The declared root is a relocated copy without workspace markers, and an
    // ancestor is a workspace holding a byte-identical backing table.
    fs::remove_dir_all(root.join("manifests")).unwrap();
    fs::remove_file(root.join("Cargo.toml")).unwrap();
    let backing = root.join("data/originals/raw.csv");
    let decoy = outer.join("data/originals/raw.csv");
    fs::create_dir_all(outer.join("manifests")).unwrap();
    fs::write(outer.join("Cargo.toml"), "[workspace]\n").unwrap();
    fs::create_dir_all(decoy.parent().unwrap()).unwrap();
    fs::copy(&backing, &decoy).unwrap();

    // The backing table exists only outside the declared root: the export
    // must refuse rather than read it there.
    let away = root.join("raw.csv.away");
    fs::rename(&backing, &away).unwrap();
    let outside = root.join("exported-outside.json");
    let refused = export_selected(&authority, &root, &prepared, &output, &seeds, &outside)
        .output()
        .unwrap();
    assert!(
        !refused.status.success(),
        "the export read its fitting view from outside the declared root"
    );
    assert!(!outside.exists());
    let handoff: Value = read_json(&outside.with_extension("handoff.json")).unwrap();
    assert_eq!(handoff["status"], "export_failed");
    let receipt: Value = read_json(&outside.with_extension("fit.json")).unwrap();
    assert_eq!(receipt["passed"], false);
    assert!(receipt["worker_argv"].as_array().is_none_or(Vec::is_empty));

    // The backing table only inside the declared root, the decoy ancestor
    // still a workspace without it: the export succeeds, and its worker was
    // handed the sealed projection and no prepared path.
    fs::rename(&away, &backing).unwrap();
    fs::remove_file(&decoy).unwrap();
    let inside = root.join("exported.json");
    success(
        export_selected(&authority, &root, &prepared, &output, &seeds, &inside)
            .output()
            .unwrap(),
    );
    let receipt: Value = read_json(&inside.with_extension("fit.json")).unwrap();
    assert_eq!(receipt["passed"], true);
    assert_eq!(receipt["worker_input"]["kind"], "sealed_memory_projection");
    assert_eq!(receipt["artifact_root"], json!(root));
    let argv = receipt["worker_argv"].as_array().unwrap();
    assert_eq!(argv[0], "fit-memory");
    assert!(argv
        .iter()
        .all(|a| !a.as_str().unwrap().contains("prepared")));
    let handoff: Value = read_json(&inside.with_extension("handoff.json")).unwrap();
    assert_eq!(handoff["status"], "exported");
    let exported: Predictor = read_json(&inside).unwrap();
    let p: Prepared = read_json(&prepared.join("prepared.json")).unwrap();
    assert_eq!(exported.training_sha256, p.cuts[0].training_sha256);
    authority.stop_service();
}

// ---------------------------------------------------------------------------
// W1 route R2 at the public fitted-view fit and the legacy isolated export:
// the cut's fitting rows are projected under a declared root (or, with no
// root, only from a contained view's own tables). Neither ever resolves a
// pointer view's backing table by walking the view's ancestors.
// ---------------------------------------------------------------------------
