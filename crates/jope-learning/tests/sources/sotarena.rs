use super::*;

fn score_line(track: &str, ds: &str, method: &str, value: f64) -> String {
    format!(
        "{track},{ds},{method},public-development,rmse,{value},ok,True,reproducible-run-report,2026-06-30T00:40:20+00:00,{},results/external-references/golden42/{ds}/{method}.json,Exact Golden42 report; method identity is the report filename.\n",
        sha256_hex(format!("{track}{ds}{method}").as_bytes())
    )
}

#[test]
fn sotarena_rows_append_verify_and_query() {
    let tmp = tempfile::tempdir().unwrap();
    let results = tmp.path().join("results");
    let v1 = HEADER.to_string()
        + &score_line(
            "tabular-regression-v1",
            "019de5d76a718163",
            "ada_boost",
            0.0733,
        )
        + &score_line(
            "tabular-regression-v1",
            "019de5d76a718163",
            "jope-dime",
            0.0611,
        )
        + &score_line("tabular-regression-v1", "0259e0b6af043f58", "ridge", 0.5);
    let v2 = HEADER.to_string()
        + &score_line(
            "tabular-regression-v2",
            "019de5d76a718163",
            "ada_boost",
            0.0733,
        )
        + &score_line(
            "tabular-regression-v2",
            "019de5d76a718163",
            "jope-dime",
            0.0598,
        )
        + &score_line("tabular-regression-v2", "ZTS_small", "ada_boost", 0.2142);
    fs::create_dir_all(results.join("tabular-regression-v1")).unwrap();
    fs::create_dir_all(results.join("tabular-regression-v2")).unwrap();
    fs::create_dir_all(results.join("external-references")).unwrap();
    fs::write(results.join("tabular-regression-v1/scores.csv"), &v1).unwrap();
    fs::write(results.join("tabular-regression-v2/scores.csv"), &v2).unwrap();
    assert_eq!(sotarena::track_versions(&results).unwrap().len(), 2);
    let state_dir = tmp.path().join("state");
    let rep = jl(
        &state_dir,
        &[
            "scores",
            "append",
            "--source",
            "sotarena",
            results.to_str().unwrap(),
            "--git-sha",
            "e5db30a07c093661c8f6940d3e22192511f8388b",
        ],
    )
    .ok()
    .json();
    let rep = rep.as_array().unwrap();
    assert_eq!(rep.len(), 2);
    assert_eq!(rep[0]["rows"], 3);
    assert_eq!(rep[1]["rows"], 3);
    assert_eq!(
        rep[1]["batch"]["prev_batch_sha256"],
        rep[0]["batch"]["sha256"]
    );
    // The published csv is kept byte-identical with its hash.
    let pub_v2 =
        state_dir.join("scores/sotarena/tabular-regression-v2/tabular-regression-v2/scores.csv");
    assert_eq!(fs::read_to_string(&pub_v2).unwrap(), v2);
    assert!(fs::read_to_string(pub_v2.with_extension("csv.sha256"))
        .unwrap()
        .starts_with(&sha256_hex(v2.as_bytes())));
    let v = jl(&state_dir, &["scores", "verify-ledger"]).ok().json();
    assert_eq!(
        (v["rows"].as_u64(), v["ledger"]["batches"].as_u64()),
        (Some(6), Some(2))
    );
    // Re-append: every row is a duplicate, no new batch.
    let rep = jl(
        &state_dir,
        &[
            "scores",
            "append",
            "--source",
            "sotarena",
            results.to_str().unwrap(),
            "--git-sha",
            "e5db30a07c093661c8f6940d3e22192511f8388b",
        ],
    )
    .ok()
    .json();
    assert_eq!(rep[0]["rows"], 0);
    assert_eq!(rep[0]["duplicates_skipped"], 3);
    assert!(rep[0]["batch"].is_null());
    let best = jl(&state_dir, &["scores", "query", "--best-per-dataset"])
        .ok()
        .json();
    let best = best.as_array().unwrap();
    assert_eq!(best.len(), 3);
    let top = best
        .iter()
        .find(|b| b["dataset_hash"] == "019de5d76a718163")
        .unwrap();
    assert_eq!(
        (
            top["method_id"].as_str(),
            top["value"].as_f64(),
            top["version"].as_str()
        ),
        (
            Some("jope-dime"),
            Some(0.0598),
            Some("tabular-regression-v2")
        )
    );
    assert_eq!(
        (top["rows"].as_u64(), top["methods"].as_u64()),
        (Some(4), Some(2))
    );
    let best = jl(
        &state_dir,
        &[
            "scores",
            "query",
            "--best-per-dataset",
            "--version",
            "tabular-regression-v1",
        ],
    )
    .ok()
    .json();
    assert_eq!(
        best.as_array()
            .unwrap()
            .iter()
            .find(|b| b["dataset_hash"] == "019de5d76a718163")
            .unwrap()["value"],
        0.0611
    );
    // Tampering with a value breaks the row hash and the batch seal.
    let ledger = jope_learning::ledger::Ledger::open(&state_dir, "scores");
    let (_, first) = ledger.batches().unwrap().into_iter().next().unwrap();
    fs::write(
        &first,
        fs::read_to_string(&first)
            .unwrap()
            .replace("0.0611", "0.0001"),
    )
    .unwrap();
    let r = jl(&state_dir, &["scores", "verify-ledger"]);
    assert_eq!(r.status, 1);
    let v = r.json();
    assert!(!v["invalid_rows"].as_array().unwrap().is_empty());
    assert!(!v["ledger"]["breaks"].as_array().unwrap().is_empty());
}
