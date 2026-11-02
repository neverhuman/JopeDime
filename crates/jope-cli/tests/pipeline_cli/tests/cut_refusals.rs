use super::*;

/// A cut count outside the fold partition exits 2 and leaves nothing behind, for `evolve` and
/// `run`, so the corrected command runs into the same `--output`. (`evolve --rounds 2 --cuts 6`
/// used to leave `round-001/`, and the corrected retry failed with "already holds a lineage".)
#[test]
fn a_refused_cut_count_exits_2_leaves_nothing_and_the_corrected_retry_runs() {
    let temp = tempfile::tempdir().unwrap();
    let output = temp.path().join("lineage");
    let evolve = |cuts: &str, budget: &str| {
        jope()
            .args(["pipeline", "evolve", "--dataset"])
            .arg(dataset())
            .arg("--output")
            .arg(&output)
            .args(["--rounds", "2", "--cuts", cuts, "--budget", budget])
            .args(["--policy", "skip", "--emit", "rust"])
            .output()
            .unwrap()
    };
    for cuts in ["6", "0"] {
        let result = evolve(cuts, "evaluations=16,seconds=60");
        let stderr = String::from_utf8_lossy(&result.stderr);
        assert_eq!(result.status.code(), Some(2), "{stderr}");
        assert!(
            stderr.contains(&format!("--cuts must be 1..=5, got {cuts}")),
            "{stderr}"
        );
        assert!(!output.exists(), "--cuts {cuts} left {}", output.display());
    }
    let run = temp.path().join("run");
    let result = jope()
        .args(run_args(&run, "evaluations=4,seconds=60"))
        .args(["--cuts", "6"])
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert_eq!(result.status.code(), Some(2), "{stderr}");
    assert!(!run.exists(), "run --cuts 6 left {}", run.display());
    // The corrected command, into the same --output (a small budget keeps it quick).
    let result = evolve("3", "evaluations=10,seconds=60");
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let lineage: Value =
        serde_json::from_slice(&std::fs::read(output.join("lineage.json")).unwrap()).unwrap();
    assert_eq!(lineage["rounds"].as_array().map(Vec::len), Some(2));
    assert_eq!(lineage["cuts"], serde_json::json!([0, 1, 2]));
}

/// `evolve` into an `--output` that already holds something (here a stale `control/`) exits 2
/// before it writes anything; it used to run every round first, write `lineage.json` and only then
/// refuse, so the corrected retry was refused as "already holds a lineage". Once the directory is
/// emptied the same command runs into it.
#[test]
fn evolve_into_a_non_empty_output_exits_2_first_and_runs_once_it_is_emptied() {
    let temp = tempfile::tempdir().unwrap();
    let output = temp.path().join("lineage");
    std::fs::create_dir_all(output.join("control")).unwrap();
    std::fs::write(output.join("control/receipt.json"), b"{}\n").unwrap();
    let evolve = || {
        jope()
            .args(["pipeline", "evolve", "--dataset"])
            .arg(dataset())
            .arg("--output")
            .arg(&output)
            .args([
                "--rounds",
                "1",
                "--control",
                "--budget",
                "evaluations=4,seconds=60",
            ])
            .args(["--policy", "skip", "--emit", "rust"])
            .output()
            .unwrap()
    };
    let result = evolve();
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert_eq!(result.status.code(), Some(2), "{stderr}");
    assert!(stderr.contains("is not empty (control)"), "{stderr}");
    let names = |dir: &Path| {
        let mut names: Vec<String> = std::fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    };
    assert_eq!(
        names(&output),
        ["control"],
        "the refusal wrote into --output"
    );
    assert_eq!(names(&output.join("control")), ["receipt.json"]);
    std::fs::remove_dir_all(output.join("control")).unwrap();
    let result = evolve();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let lineage: Value =
        serde_json::from_slice(&std::fs::read(output.join("lineage.json")).unwrap()).unwrap();
    assert_eq!(lineage["rounds"].as_array().map(Vec::len), Some(1));
    assert!(lineage["control"].is_object());
}
