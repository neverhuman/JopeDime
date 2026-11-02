use super::*;

/// `census collect` observes a host, so it runs as the test-only fixture executable: the
/// production command line observing the fixture host.
fn collector() -> PathBuf {
    fixture_host::executable()
}
/// `census merge` and `preflight` read only pinned files and observe no host: the product binary.
fn product() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_jope-native-loop"))
}
fn run(exe: &Path, prefix: &[&str], args: &[&str]) -> std::process::Output {
    Command::new(exe)
        .args(prefix)
        .args(args)
        .env("CUDA_VISIBLE_DEVICES", "")
        .output()
        .unwrap()
}
/// `tree` without the root directory's own entry (its mtime changes when a file is added).
fn contents(root: &Path) -> std::collections::BTreeMap<PathBuf, String> {
    let mut t = tree(root);
    t.remove(Path::new(""));
    t
}
fn collect_argv(spec: &CollectSpec, out: &Path) -> Vec<String> {
    let s = |p: &Path| p.display().to_string();
    vec![
        "census".into(),
        "collect".into(),
        "--out".into(),
        s(out),
        "--address".into(),
        spec.address.clone(),
        "--certificate".into(),
        s(&spec.certificate),
        "--source-commit".into(),
        spec.source_commit.clone(),
        "--executable".into(),
        s(&spec.executable),
        "--runtime-library".into(),
        s(&spec.runtime_libraries[0]),
        "--compiler".into(),
        s(&spec.compiler),
        "--installed-policy".into(),
        s(&spec.installed_policies[0]),
    ]
}

#[test]
fn cli_collect_merge_and_preflight_round_trip_without_side_effects() {
    let (_t, root) = tempdir();
    let spec = install(&root);
    let out = root.join("host.census.json");
    let argv = collect_argv(&spec, &out);
    let argv: Vec<&str> = argv.iter().map(String::as_str).collect();
    let (collector, product) = (collector(), product());
    let before = contents(&root);
    let o = run(&collector, &[], &argv);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let report: serde_json::Value = serde_json::from_slice(&o.stdout).unwrap();
    let written = fs::read(&out).unwrap();
    assert_eq!(report["census_sha256"], sha(&written));
    assert_eq!(
        report["gpu_probe"],
        "not_queried_cuda_visible_devices_empty"
    );
    assert_eq!(report["allocation_held"], false);
    // The collector wrote its one output and nothing else.
    let mut after = contents(&root);
    assert!(after.remove(Path::new("host.census.json")).is_some());
    assert_eq!(after, before);
    // A census is never replaced.
    let before = contents(&root);
    let o = run(&collector, &[], &argv);
    assert!(!o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("already exists"));
    assert_eq!(contents(&root), before);
    // Legacy loop flags are refused for census operations.
    let o = run(&collector, &["--tenant", "t"], &argv);
    assert!(!o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("no fixed-loop arguments"));

    let base = HostCensus::from_bytes(&written).unwrap();
    // The command line observed the fixture host and hashed the installed worker it was given.
    let identity = &base.hosts[0].identity;
    assert_eq!(
        (
            identity.host.as_str(),
            identity.machine_id.as_str(),
            identity.boot_id.as_str()
        ),
        (
            fixture_host::HOST,
            fixture_host::MACHINE_ID,
            fixture_host::BOOT_ID
        )
    );
    assert_eq!(
        identity.executable_sha256,
        sha(b"fixture worker executable")
    );
    let now = base.captured_unix_ms;
    let mut documents = Vec::new();
    for i in 0..3 {
        let doc = if i == 0 {
            written.clone()
        } else {
            serde_json::to_vec(&synthetic(&base, i, now)).unwrap()
        };
        let path = root.join(format!("doc-{i}.json"));
        fs::write(&path, &doc).unwrap();
        documents.push(format!("{}={}", path.display(), sha(&doc)));
    }
    let merged_path = root.join("merged.json");
    let mut merge = vec![
        "census".to_string(),
        "merge".into(),
        "--out".into(),
        merged_path.display().to_string(),
        "--max-age-ms".into(),
        "600000".into(),
        "--clock-skew-ms".into(),
        "2000".into(),
    ];
    // A wrong pin refuses the merge and writes nothing.
    let mut wrong = merge.clone();
    for (i, d) in documents.iter().enumerate() {
        wrong.push("--document".into());
        wrong.push(if i == 1 {
            format!("{}={}", d.rsplit_once('=').unwrap().0, "0".repeat(64))
        } else {
            d.clone()
        });
    }
    let o = run(
        &product,
        &wrong.iter().map(String::as_str).collect::<Vec<_>>(),
        &[],
    );
    assert!(!o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("was altered"));
    assert!(!merged_path.exists());
    for d in &documents {
        merge.push("--document".into());
        merge.push(d.clone());
    }
    let o = run(
        &product,
        &merge.iter().map(String::as_str).collect::<Vec<_>>(),
        &[],
    );
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let report: serde_json::Value = serde_json::from_slice(&o.stdout).unwrap();
    let merged_bytes = fs::read(&merged_path).unwrap();
    assert_eq!(report["census_sha256"], sha(&merged_bytes));
    assert_eq!(report["documents"].as_array().unwrap().len(), 3);
    let merged = HostCensus::from_bytes(&merged_bytes).unwrap();

    // Plan from the merged identities and preflight through the operator CLI.
    let hosts: Vec<_> = merged
        .hosts
        .iter()
        .map(|h| (h.identity.clone(), None))
        .collect();
    let plan_now = now_ms();
    let c = plan_for(plan_now, &hosts);
    let dir = root.join("run");
    let plan = controller::plan(&dir, &bytes(&c), plan_now, None).unwrap();
    let o = run(
        &product,
        &[
            "preflight",
            "--run-dir",
            dir.to_str().unwrap(),
            "--plan-sha256",
            &plan.plan_sha256,
            "--census",
            merged_path.to_str().unwrap(),
        ],
        &[],
    );
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let receipt: serde_json::Value = serde_json::from_slice(&o.stdout).unwrap();
    assert_eq!(receipt["allocation_held"], false);
    assert_eq!(receipt["census_sha256"], sha(&merged_bytes));
    assert_eq!(receipt["hosts"].as_object().unwrap().len(), 3);
}
