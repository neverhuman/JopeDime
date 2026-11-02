use super::*;

#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
fn configured_authority_refuses_every_public_fit_entry_without_verified_activation() {
    let mut data = Data::new();
    assert_every_entry_refused_without_verified_activation(&data, "configured");
    let stats = data.authority.shutdown();
    assert_eq!(stats["worker_spawns"], 0);
    assert_eq!(stats["committed"], 0);
}

/// Shipped build: the isolated export worker inherits a fully configured
/// authority, is refused at activation and still receipts its failure.
#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
fn isolated_model_export_is_refused_without_verified_activation() {
    let data = Data::new();
    let spec = data.root.join("export-configured.request.json");
    write_json(
        &spec,
        &json!({"prepared":data.prepared,"output":data.root.join("export-configured.json"),
        "expected_error":UNVERIFIED_ACTIVATION}),
    )
    .unwrap();
    let mut command = Command::new(std::env::current_exe().unwrap());
    command
        .args(["--exact", "isolated_export_child_fixture", "--nocapture"])
        .env("JOPE_SYNTHETIC_EXPORT_TEST", &spec);
    data.authority.configure(&mut command);
    fixture::success(command.output().unwrap());
    data.authority.assert_no_fit_starts();
}

/// Shipped build: `namespace-activate` has no verifier for any capability, and
/// refusing changes neither dispatch history nor activation state.
#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
fn namespace_activate_refuses_every_capability_without_changing_authority_state() {
    let data = Data::new();
    let journal = data.authority.authority.join("dispatch.jsonl");
    let before = file_hash(&journal).unwrap();
    for (scope, capability, message) in [
        (
            SCOPE,
            "synthetic_qualification",
            "namespace activation unavailable",
        ),
        (
            SCOPE,
            "regression_collection",
            "namespace activation unavailable",
        ),
        (
            SCOPE,
            "regression_training",
            "namespace activation unavailable",
        ),
        (
            SCOPE,
            "regression_promotion",
            "namespace activation unavailable",
        ),
        (
            SCOPE,
            "arbitrary_execution",
            "unsupported activation capability",
        ),
        (
            "not-an-authorized-scope",
            "synthetic_qualification",
            "unknown activation scope",
        ),
    ] {
        let result = data
            .authority
            .command()
            .args(["namespace-activate", "--authority"])
            .arg(&data.authority.authority)
            .arg("--witness")
            .arg(&data.authority.witness)
            .args(["--scope", scope, "--capability", capability])
            .output()
            .unwrap();
        let error = String::from_utf8_lossy(&result.stderr);
        assert!(!result.status.success(), "{scope}/{capability} activated");
        assert!(error.contains(message), "{scope}/{capability}: {error}");
        assert!(result.stdout.is_empty());
    }
    assert_eq!(
        file_hash(&journal).unwrap(),
        before,
        "refused activation changed dispatch history"
    );
    let rows: i64 = read_only(&data.authority)
        .query_row("SELECT count(*) FROM activations", [], |r| r.get(0))
        .unwrap();
    assert_eq!(rows, 0);
    assert_every_entry_refused_without_verified_activation(&data, "after-activate");
}
