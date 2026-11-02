use super::*;

#[test]
fn written_report_keeps_python_spelling() {
    let root = tempfile::tempdir().unwrap();
    let journal = root.path().join("journal");
    let output = root.path().join("output");
    write_file(
        &journal.join("launch.json"),
        r#"{"exit_code": 0, "t": 1e-05, "big": 1e16, "neg": -0.0, "n": 12345678901234567890, "s": "tab\there é"}"#,
    );
    write_file(
        &journal.join("window.json"),
        r#"{"started": 1.5, "list": [], "obj": {}}"#,
    );
    write_file(
        &journal.join("warm-é1-0/receipt.json"),
        r#"{"mode": "warm", "dataset": "é1", "seed": 0, "elapsed_seconds": 2.5, "failure": "bad, \"quoted\"\nline", "success": true}"#,
    );
    let plan = r#"{"schema": "jope.inference-benchmark/v2", "experiment": {"methods_sha256": "m", "source_sha": "s"}, "binary": {"sha256": "b"}, "inputs": [{"id": "é1", "raw": {"sha256": "r"}}], "census": {"warm": 1, "cold": 0, "warmup": 0, "profile": 0}}"#;
    write(plan.as_bytes(), &journal, &output).unwrap();

    let prediction = PREDICTION_BACKENDS
            .iter()
            .flat_map(|backend| {
                BATCH_ROWS.iter().map(move |rows| {
                    format!(
                        "    {{\n      \"dataset\": \"\\u00e91\",\n      \"backend\": \"{backend}\",\n      \"batch_rows\": {rows},\n      \"status\": \"unexecuted\",\n      \"accepted\": false\n    }}"
                    )
                })
            })
            .collect::<Vec<_>>()
            .join(",\n");
    let expected = format!(
        r#"{{
  "schema": "jope.inference-benchmark/v2",
  "controller": {{
    "status": "completed",
    "launch": {{
      "exit_code": 0,
      "t": 1e-05,
      "big": 1e+16,
      "neg": -0.0,
      "n": 12345678901234567890,
      "s": "tab\there \u00e9"
    }},
    "qualification_status": "not_executed_or_interrupted"
  }},
  "qualified": false,
  "qualification": {{}},
  "publication_headline": false,
  "completed_successfully": 0,
  "expected": 1,
  "census": {{
    "modes": {{
      "warm": 1,
      "cold": 0,
      "warmup": 0,
      "profile": 0
    }},
    "source": "plan"
  }},
  "reconciliation": {{
    "planned": 1,
    "receipts": 1,
    "accepted": 0,
    "rejected": 1,
    "missing": 0,
    "unplanned_receipts": [],
    "reconciled": true,
    "failure_or_missing_rate": 1.0
  }},
  "missing": [],
  "failures": [
    {{
      "dataset": "\u00e91",
      "mode": "warm",
      "seed": 0,
      "failure": "bad, \"quoted\"\nline"
    }}
  ],
  "warm_cumulative_backend_milestones": [],
  "cold_observations": [],
  "full_gp": [],
  "prediction": [
{prediction}
  ],
  "window": {{
    "started": 1.5,
    "list": [],
    "obj": {{}}
  }},
  "warm_complete_request_distribution": {{
    "n": 0,
    "median_seconds": null,
    "p95_seconds": null,
    "mean_seconds": null
  }},
  "boundary": "{BOUNDARY}"
}}
"#
    );
    assert_eq!(
        fs::read_to_string(output.join("summary.json")).unwrap(),
        expected
    );
    assert_eq!(
            fs::read_to_string(output.join("requests.csv")).unwrap(),
            "dataset,mode,seed,elapsed_seconds,accepted,failure\r\né1,warm,0,2.5,False,\"bad, \"\"quoted\"\"\nline\"\r\n"
        );
    let methodology = fs::read_to_string(output.join("methodology.md")).unwrap();
    assert!(methodology.starts_with(&format!("# GPU timing study\n\n{BOUNDARY}\n\n")));
    assert!(methodology.contains("Qualified: False. Completed warm Python and Rust paths: 0/1. Planned direct slots: 1, accepted 0, missing 0, unplanned receipts 0. Missing attempts are unexecuted, never zero latency."));
    assert!(methodology.ends_with("containing I/O and teardown.\n"));
    // An unsupported contract is refused before any output changes.
    let v1 = plan.replace("inference-benchmark/v2", "inference-benchmark/v1");
    let error = write(v1.as_bytes(), &journal, &root.path().join("v1")).unwrap_err();
    assert!(format!("{error:#}").contains("receipt version unsupported"));
    assert!(!root.path().join("v1").exists());
}
