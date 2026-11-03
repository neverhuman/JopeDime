use super::*;

use super::args::parse_value;
use clap::Parser;
use serde_json::json;
use std::path::PathBuf;

#[derive(Parser)]
struct Args {
    #[command(subcommand)]
    command: crate::cli::ApplianceCommand,
}

#[cfg(not(windows))]
#[test]
fn request_boundary_requires_acknowledged_pause_or_cancel_and_preserves_control_errors() {
    use jope_client::{
        transport::{header, HeaderMap},
        Request, Response,
    };
    use std::{io::Cursor, sync::Mutex};
    struct Control {
        current: Mutex<RunSnapshot>,
        actions: Mutex<Vec<String>>,
        refuse: bool,
    }
    impl Transport for Control {
        fn execute(&self, request: Request) -> jope_client::Result<Response> {
            let mut current = self.current.lock().unwrap();
            if request.method().as_str() == "POST" {
                if self.refuse {
                    return Err(jope_client::Error::Protocol(
                        "unknown control commit outcome",
                    ));
                }
                let value: Value = serde_json::from_slice(request.body()).unwrap();
                let action = value["action"].as_str().unwrap();
                self.actions.lock().unwrap().push(action.into());
                current.state = match action {
                    "pause" => RunState::Paused,
                    "cancel" => RunState::Cancelled,
                    _ => panic!("unexpected action"),
                };
                current.revision.0 += 1;
            }
            let mut headers = HeaderMap::new();
            headers.insert(header::CONTENT_TYPE, "application/json".parse().unwrap());
            headers.insert("jope-api-version", "1.0".parse().unwrap());
            Ok(Response::new(
                200,
                headers,
                Cursor::new(serde_json::to_vec(&*current).unwrap()),
            ))
        }
    }
    let snapshot: RunSnapshot = serde_json::from_value(json!({
        "run_id":"run_01JBD8Q6N4V2T5Y8K3M7P9R1S0", "revision":"1",
        "state":"running", "phase":"searching", "pending":null, "effort":"fast",
        "budget_used":{"evaluations":"1","wall_seconds":"1"},
        "budget_total":{"evaluations":"128","wall_seconds":"60"}, "best":null,
        "event_watermark":"0", "warnings":[], "completion_reason":null
    }))
    .unwrap();
    for (count, refused, action) in [(1, false, "pause"), (2, false, "cancel"), (1, true, "")] {
        let control = Control {
            current: Mutex::new(snapshot.clone()),
            actions: Mutex::new(vec![]),
            refuse: refused,
        };
        let interrupts = RunInterrupts::delivered_for_test(count);
        let mut stderr = vec![];
        let error = finish_interrupt(&control, &snapshot, &interrupts, &mut stderr).unwrap_err();
        assert_eq!(stderr, format!("{}\n", snapshot.run_id).as_bytes());
        if refused {
            assert!(error.to_string().contains("unknown control commit outcome"));
            assert_eq!(interrupted_exit_code(&error), None);
            assert!(control.actions.lock().unwrap().is_empty());
        } else {
            assert_eq!(interrupted_exit_code(&error), Some(130));
            assert_eq!(*control.actions.lock().unwrap(), [action]);
        }
    }
}

#[cfg(not(windows))]
#[test]
fn cleanup_failure_around_an_acknowledgment_does_not_become_exit130() {
    let acknowledged: anyhow::Error = super::super::inproc::Interrupted {
        run_id: "run_01JBD8Q6N4V2T5Y8K3M7P9R1S0".parse().unwrap(),
        state: RunState::Paused,
    }
    .into();
    assert_eq!(interrupted_exit_code(&acknowledged), Some(130));
    let error = acknowledged.context("response reader shutdown failed");
    assert_eq!(interrupted_exit_code(&error), None);
    assert_eq!(dispatch_exit_code(&error), None);
}

#[test]
fn batch_discovery_failures_keep_catalog_exit_codes_and_do_not_mask_cleanup_errors() {
    for (error, expected) in [
        (jope_client::discover::Error::NoEngine, 9),
        (jope_client::discover::Error::InvalidEndpoint, 2),
        (
            jope_client::discover::Error::Busy {
                pid: Some(123),
                command: "owned-fixture".into(),
            },
            9,
        ),
    ] {
        let error: anyhow::Error = error.into();
        assert_eq!(dispatch_exit_code(&error), Some(expected));
        assert_eq!(
            dispatch_exit_code(&error.context("response shutdown failed")),
            None
        );
    }
    assert_eq!(
        dispatch_exit_code(&DispatchUnavailable::EndpointCredentials.into()),
        Some(9)
    );
    #[cfg(not(feature = "engine"))]
    assert_eq!(
        dispatch_exit_code(&DispatchUnavailable::LocalHandlers.into()),
        Some(2)
    );
}

fn m1_argv(path: &std::path::Path) -> Vec<String> {
    [
        "jopedime",
        "run",
        "--dataset",
        "ds_21fbfe8b53dbcd605d79486b",
        "--target",
        "target",
        "--task",
        "regression",
        "--split",
        r#"{"kind":"random","holdout_fraction":0.2,"group_column":null,"time_column":null}"#,
        "--objective",
        "rmse",
        "--effort",
        "fast",
        "--budget",
        r#"{"evaluations":"1","wall_seconds":"60"}"#,
        "--landmarks",
        "none",
        "--guidance",
        "heuristic",
        "--device",
        "cpu",
        "--seed",
        "73",
        "--json",
        "--m1-plan",
    ]
    .into_iter()
    .map(str::to_owned)
    .chain(std::iter::once(path.to_string_lossy().into_owned()))
    .collect()
}

#[test]
fn hidden_m1_requires_json_preserves_public_requirements_and_conflicts_with_transport_modes() {
    use clap::CommandFactory;
    assert_eq!(parse_value::<String>("regression").unwrap(), "regression");
    assert_eq!(
        parse_value::<String>(r#""regression""#).unwrap(),
        "regression"
    );
    assert_eq!(
        parse_value::<Value>(r#"{"kind":"random"}"#).unwrap(),
        json!({"kind": "random"})
    );
    let error = parse_value::<bool>("not a boolean").unwrap_err();
    let message = error.to_string();
    assert!(message.contains("expected"), "{message}");
    assert!(message.contains("invalid type"), "{message}");

    let argv = m1_argv(std::path::Path::new("prepared.json"));
    let parsed = Args::try_parse_from(&argv).unwrap();
    let crate::cli::ApplianceCommand::Run(args) = parsed.command else {
        panic!("run command")
    };
    assert_eq!(args.m1_plan, Some("prepared.json".into()));
    assert!(!serde_json::to_string(&args.run_spec().unwrap())
        .unwrap()
        .contains("m1"));
    let mut cli = crate::cli::Cli::command();
    let help = cli.find_subcommand_mut("run").unwrap().render_long_help();
    assert!(!help.to_string().contains("m1"));
    let without_json: Vec<_> = argv.iter().filter(|a| a.as_str() != "--json").collect();
    assert!(Args::try_parse_from(without_json).is_err());
    assert!(Args::try_parse_from(["jopedime", "run", "--json"]).is_err());
    for extra in [
        vec!["--detach"],
        vec!["--dry-run"],
        vec!["--events", "ndjson"],
        vec!["--wait", "0"],
    ] {
        let mut conflict = argv.clone();
        conflict.extend(extra.into_iter().map(str::to_owned));
        assert!(Args::try_parse_from(conflict).is_err());
    }
}

#[cfg(feature = "engine")]
fn m1_fixture(directory: &std::path::Path) -> std::path::PathBuf {
    use jope_service::discover::r#loop::Plan;
    let raw = include_bytes!("../../../../../test-support/datasets/v1/d2-regression/train.csv");
    assert_eq!(
        jope_runtime::sha256_hex(raw),
        "d9a0b97e91ff9d04bc87fefa4671a49c40cee21dfac05be58a7fc7ce08045931"
    );
    let mut names: Vec<_> = (0..21).map(|j| format!("x{j}")).collect();
    names.push("target".into());
    let mut csv = names.join(",").into_bytes();
    csv.push(b'\n');
    csv.extend(raw);
    let mut table =
        jope_runtime::table::parse_csv(&csv, &jope_runtime::limits::Limits::default()).unwrap();
    let (fit, _) = jope_service::discover::splits::random_split(
        table.rows.len(),
        table.rows.len() * 4 / 5,
        73,
        "dsc08/fit",
    )
    .unwrap();
    let rows: Vec<_> = fit.into_iter().take(30).collect();
    table.rows = rows.iter().map(|&r| table.rows[r].clone()).collect();
    let bytes = jope_service::dataset::jmat::encode(&table).unwrap();
    let matrix = directory.join("fit.jmat");
    std::fs::write(&matrix, &bytes).unwrap();
    std::fs::write(
        directory.join("source-rows.json"),
        serde_json::to_vec(&rows).unwrap(),
    )
    .unwrap();
    let plan = Plan {
        run_id: "run_01JBD8Q6N4V2T5Y8K3M7P9R1S0".parse().unwrap(),
        dataset_id: "ds_21fbfe8b53dbcd605d79486b".parse().unwrap(),
        problem_digest: jope_runtime::sha256_hex(&bytes),
        bundle_digest: jope_runtime::sha256_hex(b"DSC08 heuristic M1, no learned bundle"),
        binary_version: "dsc08f01-cli-probe-v1".into(),
        matrix_path: matrix,
        matrix_sha256: jope_runtime::sha256_hex(&bytes),
        target: "target".into(),
        features: names[..21].to_vec(),
        objective: "rmse".into(),
        seed: 73,
        planned_workers: 8,
        effort: jope_api::Effort::Fast,
        evaluations: 1,
        wall_seconds: 60,
        evaluation_wall_ms: 2000,
        evaluation_rss_bytes: 128 * 1024 * 1024,
        groups: None,
        times: None,
        gap_rows: 0,
    };
    let path = directory.join("m1.json");
    std::fs::write(
        &path,
        serde_json::to_vec_pretty(
            &json!({"home":directory.join("store"),"plan":plan,"live_workers":2}),
        )
        .unwrap(),
    )
    .unwrap();
    path
}

#[cfg(feature = "engine")]
#[test]
fn hidden_m1_refuses_malformed_or_contradictory_input_before_creating_a_store() {
    let directory = tempfile::tempdir().unwrap();
    let path = m1_fixture(directory.path());
    let mut argv = m1_argv(&path);
    let objective = argv.iter().position(|a| a == "--objective").unwrap() + 1;
    argv[objective] = "mse".into();
    let crate::cli::ApplianceCommand::Run(args) = Args::try_parse_from(argv).unwrap().command
    else {
        panic!("run")
    };
    let error = dispatch(args).unwrap_err();
    assert_eq!(
        jope_runtime::error::kind_of(&error),
        jope_runtime::FailureKind::Input
    );
    assert!(!directory.path().join("store").exists());
    std::fs::write(&path, b"not json").unwrap();
    let crate::cli::ApplianceCommand::Run(args) =
        Args::try_parse_from(m1_argv(&path)).unwrap().command
    else {
        panic!("run")
    };
    assert_eq!(
        jope_runtime::error::kind_of(&dispatch(args).unwrap_err()),
        jope_runtime::FailureKind::Input
    );
    assert!(!directory.path().join("store").exists());
}

#[cfg(not(feature = "engine"))]
#[test]
fn hidden_m1_without_engine_refuses_as_backend_before_reading_a_plan() {
    let crate::cli::ApplianceCommand::Run(args) =
        Args::try_parse_from(m1_argv(std::path::Path::new("absent.json")))
            .unwrap()
            .command
    else {
        panic!("run")
    };
    assert_eq!(
        jope_runtime::error::kind_of(&dispatch(args).unwrap_err()),
        jope_runtime::FailureKind::Backend
    );
}

#[cfg(feature = "engine")]
#[test]
#[ignore = "actual CLI gate; build the current jopedime with dev-keys, set JOPE_M1_CLI, and explicitly run this test"]
fn hidden_m1_actual_cli_runs_real_fit_without_a_daemon_and_refuses_damaged_package() {
    use std::process::Command;
    let executable =
        PathBuf::from(std::env::var_os("JOPE_M1_CLI").expect("build/select the current CLI first"));
    let directory = tempfile::tempdir().unwrap();
    let input = m1_fixture(directory.path());
    let argv = m1_argv(&input);
    let output = Command::new(&executable)
        .args(&argv[1..])
        .env_clear()
        .env("JOPEDIME_HOME", directory.path().join("absent-daemon"))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(output.stdout.iter().filter(|&&b| b == b'\n').count(), 1);
    assert_eq!(value["stage"], "search_finished");
    assert!(value["champion"]["cv_value"].as_f64().unwrap().is_finite());
    assert_eq!(value["nondeterminism"]["count"], "0");
    let text = String::from_utf8(output.stdout.clone()).unwrap();
    for forbidden in [
        "(pipe", "\"dsl\"", "admitted", "stalled", "round-", "slot-", "rung-",
    ] {
        assert!(!text.contains(forbidden), "public leak {forbidden}");
    }
    assert!(!directory.path().join("absent-daemon").exists());

    use jope_assets::writer::{assemble_development, stage, write_new, StageOptions};
    let code = std::fs::read(&executable).unwrap();
    let staged = stage(
        &code,
        vec![],
        &StageOptions {
            product: "jopedime".into(),
            product_version: env!("CARGO_PKG_VERSION").into(),
            target: "x86_64-unknown-linux-gnu".into(),
            profile: "fixture".into(),
            scope: jope_assets::Scope::Development,
            source: Default::default(),
            methods: vec!["rust.ridge".into()],
            key_id: jope_assets::verify::development_key().id,
            frame_bytes: 1 << 20,
            level: 3,
        },
    )
    .unwrap();
    let mut package = assemble_development(&code, &staged).unwrap();
    // Change an executable-region byte while preserving the loader header.
    package[code.len() - 1] ^= 1;
    let damaged = directory.path().join("damaged-jopedime");
    write_new(&package, &damaged).unwrap();
    let refused = Command::new(&damaged)
        .args(&argv[1..])
        .env_clear()
        .output()
        .unwrap();
    assert_eq!(
        refused.status.code(),
        Some(3),
        "{}",
        String::from_utf8_lossy(&refused.stderr)
    );
    assert!(refused.stdout.is_empty());
    if let Some(root) = std::env::var_os("JOPE_M1_RECEIPTS") {
        let root = PathBuf::from(root);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("stdout.json"), &output.stdout).unwrap();
        std::fs::write(root.join("integrity.stderr"), &refused.stderr).unwrap();
        for name in ["fit.jmat", "source-rows.json", "m1.json"] {
            std::fs::copy(directory.path().join(name), root.join(name)).unwrap();
        }
        std::fs::write(root.join("cli.json"), serde_json::to_vec_pretty(&json!({"executable":executable,"sha256":jope_runtime::sha256_hex(&code),"argv":argv,"success_exit":output.status.code(),"integrity_exit":refused.status.code()})).unwrap()).unwrap();
    }
}

#[test]
fn run_flags_round_trip_to_authoritative_runspec_json() {
    let parsed = Args::try_parse_from([
        "jopedime",
        "run",
        "--dataset",
        "ds_0123456789abcdef01234567",
        "--target",
        "y",
        "--task",
        "regression",
        "--ignore-column",
        "x1",
        "--ignore-column",
        "x2",
        "--test-dataset",
        "ds_abcdef0123456789abcdef01",
        "--split",
        r#"{"kind":"random","holdout_fraction":0.2,"group_column":null,"time_column":null}"#,
        "--objective",
        "rmse",
        "--effort",
        "medium",
        "--budget",
        r#"{"evaluations":"100","wall_seconds":"60"}"#,
        "--landmarks",
        "full",
        "--guidance",
        "auto",
        "--language",
        "python",
        "--language",
        "rust",
        "--require-all-languages",
        "--device",
        "cpu",
        "--seed",
        "7",
        "--dry-run",
        "--plan-digest",
        "abc",
    ])
    .unwrap();
    let crate::cli::ApplianceCommand::Run(args) = parsed.command else {
        panic!("run command")
    };
    let spec = args.run_spec().unwrap();
    let json = serde_json::to_value(spec).unwrap();
    assert_eq!(json["dataset"], "ds_0123456789abcdef01234567");
    assert_eq!(json["test_dataset"], "ds_abcdef0123456789abcdef01");
    assert_eq!(json["ignore_columns"], json!(["x1", "x2"]));
    assert_eq!(json["languages"], json!(["python", "rust"]));
    assert_eq!(json["dry_run"], true);
    assert_eq!(json["plan_digest"], "abc");
    let _: RunSpec = serde_json::from_value(json).unwrap();
}

#[test]
fn api_json_output_is_one_golden_document_line() {
    let line =
        encode_json_line(&json!({"run_id":"run_01J00000000000000000000000","state":"completed"}))
            .unwrap();
    assert_eq!(
        line,
        "{\"run_id\":\"run_01J00000000000000000000000\",\"state\":\"completed\"}\n"
    );
}

#[test]
fn events_and_export_receipt_is_one_final_golden_json_line() {
    let receipt = receipt_value(
        &json!({"run_id":"run_01J00000000000000000000000","state":"completed"}),
        Some(json!({"artifact_id":"art_0123456789abcdef01234567"})),
    );
    assert_eq!(
            encode_json_line(&receipt).unwrap(),
            "{\"export\":{\"artifact_id\":\"art_0123456789abcdef01234567\"},\"kind\":\"receipt\",\"run\":{\"run_id\":\"run_01J00000000000000000000000\",\"state\":\"completed\"}}\n"
        );
}

#[test]
fn detach_and_interrupt_report_run_id_on_stderr() {
    let mut stderr = Vec::new();
    write_run_id(&mut stderr, &"run_01J00000000000000000000000").unwrap();
    assert_eq!(stderr, b"run_01J00000000000000000000000\n");
}

#[test]
fn noninteractive_modes_parse_without_prompting() {
    let csv = Args::try_parse_from([
        "jopedime",
        "run",
        "data/train.csv",
        "--target",
        "target",
        "--effort",
        "high",
        "--lang",
        "rust,python",
        "--out",
        "model",
        "--json",
    ])
    .unwrap();
    let crate::cli::ApplianceCommand::Run(csv) = csv.command else {
        panic!("expected run command")
    };
    assert_eq!(
        csv.input.as_deref(),
        Some(std::path::Path::new("data/train.csv"))
    );
    assert_eq!(csv.languages, ["rust", "python"]);
    assert!(csv.dataset.is_none());
    let required = [
        "--dataset",
        "ds_0123456789abcdef01234567",
        "--target",
        "y",
        "--task",
        "regression",
        "--split",
        r#"{"kind":"random","holdout_fraction":0.2,"group_column":null,"time_column":null}"#,
        "--objective",
        "rmse",
        "--effort",
        "medium",
        "--budget",
        r#"{"evaluations":"100","wall_seconds":"60"}"#,
        "--landmarks",
        "full",
        "--guidance",
        "auto",
        "--device",
        "cpu",
        "--seed",
        "7",
    ];
    for mode in [
        vec!["--detach"],
        vec!["--events", "ndjson", "--after-event", "7"],
        vec!["--out", "model-bundle"],
    ] {
        let mut argv = vec!["jopedime", "run"];
        argv.extend(required);
        argv.extend(mode);
        Args::try_parse_from(argv).unwrap();
    }
}

#[cfg(feature = "engine")]
#[test]
fn run_uses_the_typed_in_process_api_contract() {
    use jope_client::{InProcess, Transport};
    let home = tempfile::tempdir().unwrap();
    let engine = jope_service::engine::InProcessEngine::open(home.path()).unwrap();
    let client =
        InProcess::with_engine(std::sync::Arc::new(engine), 43123, &"ab".repeat(32)).unwrap();
    let parsed = Args::try_parse_from([
        "jopedime",
        "run",
        "--dataset",
        "ds_0123456789abcdef01234567",
        "--target",
        "y",
        "--task",
        "regression",
        "--split",
        r#"{"kind":"random","holdout_fraction":0.2,"group_column":null,"time_column":null}"#,
        "--objective",
        "rmse",
        "--effort",
        "medium",
        "--budget",
        r#"{"evaluations":"100","wall_seconds":"60"}"#,
        "--landmarks",
        "full",
        "--guidance",
        "auto",
        "--device",
        "cpu",
        "--seed",
        "7",
        "--dry-run",
    ])
    .unwrap();
    let crate::cli::ApplianceCommand::Run(args) = parsed.command else {
        panic!("run command")
    };
    let spec = args.run_spec().unwrap();
    assert!(matches!(
        client.run_start(&spec, Some("bat02_test")),
        Err(jope_client::Error::Problem(problem))
            if problem.code == jope_api::error::ErrorCode::CapabilityUnavailable
    ));
}
