use super::*;

/// Exercise candidate selection, refusals and failure visibility under an explicit custody
/// directory, and prove every fixture byte, link, mode and mtime unchanged.
#[test]
#[ignore = "fixture runner: needs JOPE_MAINTENANCE_SCRATCH outside /tmp and working trees"]
fn scan_preserves_fixtures_and_fails_visibly() {
    let scratch = scratch("scan");
    let tools = tools(&scratch);
    let helper = std::env::var("JOPE_MAINTENANCE_HELPER")
        .map(PathBuf::from)
        .unwrap_or_else(|_| tools.join("cargo-build-dirs.sh"));
    let root = scratch.join("fixtures");
    fs::create_dir(&root).unwrap();
    for name in [
        "stale/.rustc_info.json",
        "fresh/.rustc_info.json",
        "target/unknown",
        "generic/CACHEDIR.TAG",
        "stale/debug/receipt.bin",
        "stale/release/proof",
        "stale/tmp/evidence",
        "stale/nested/.rustc_info.json",
        "fingerprint/release/.fingerprint/marker",
        "fresh/debug/recent",
        "recent-nested/.rustc_info.json",
        "recent-nested/nested/.rustc_info.json",
        "odd\nname/.rustc_info.json",
        "gitguard/.rustc_info.json",
        "gitguard/.git/objects/pack",
    ] {
        let path = root.join(name);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, b"\x00preserve\xff\n").unwrap();
    }
    let outside = scratch.join("outside");
    fs::create_dir(&outside).unwrap();
    fs::write(outside.join(".rustc_info.json"), "{}").unwrap();
    std::os::unix::fs::symlink(&outside, root.join("linked")).unwrap();
    fs::create_dir(root.join("marker-link")).unwrap();
    std::os::unix::fs::symlink(
        outside.join(".rustc_info.json"),
        root.join("marker-link/.rustc_info.json"),
    )
    .unwrap();
    fs::create_dir_all(root.join("relsym/release")).unwrap();
    std::os::unix::fs::symlink(&outside, root.join("relsym/release/.fingerprint")).unwrap();
    let locked = root.join("inaccessible");
    fs::create_dir(&locked).unwrap();
    let old = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
        - 72 * 3600;
    let mut everything = vec![root.clone()];
    let mut stack = vec![root.clone()];
    while let Some(dir) = stack.pop() {
        for entry in fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            everything.push(path.clone());
            if fs::symlink_metadata(&path).unwrap().is_dir() {
                stack.push(path);
            }
        }
    }
    everything.sort_by_key(|p| std::cmp::Reverse(p.components().count()));
    for path in &everything {
        touch_old(path, old);
    }
    for recent in [
        "fresh/debug/recent",
        "recent-nested/nested/.rustc_info.json",
    ] {
        let status = Command::new("touch")
            .arg(root.join(recent))
            .status()
            .unwrap();
        assert!(status.success());
    }
    let before = inventory(&root);
    let mut results: Vec<Value> = vec![];
    let mut record = |label: &str, c: &Call| {
        results.push(Value::Dict(
            [
                ("test", Value::from(label)),
                ("exit", Value::from(i64::from(c.code))),
                ("stdout", Value::from(c.stdout.clone())),
                ("stderr", Value::from(c.stderr.clone())),
            ]
            .into_iter()
            .collect(),
        ));
    };

    let c = call(&helper, &[&root, Path::new("--stale"), Path::new("48")]);
    assert_eq!(c.code, 0, "{}", c.stderr);
    record("stale and marker selection", &c);
    let doc = pyjson::parse_str(&c.stdout).unwrap();
    let names: std::collections::BTreeSet<String> = doc
        .get("candidates")
        .and_then(Value::as_list)
        .unwrap()
        .iter()
        .map(|r| {
            Path::new(r.get("path").and_then(Value::as_str).unwrap())
                .file_name()
                .unwrap()
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    assert_eq!(
        names,
        ["stale", "fingerprint", "odd\nname", "gitguard"]
            .map(String::from)
            .into(),
        "nested Cargo output must not be double-counted"
    );
    let c = call(&helper, &[&root.join("fingerprint/release")]);
    assert_eq!(c.code, 0, "{}", c.stderr);
    record("fingerprint cannot escape supplied root", &c);
    assert_eq!(
        pyjson::parse_str(&c.stdout).unwrap().get("candidates"),
        Some(&Value::List(vec![]))
    );
    let missing = scratch.join("missing");
    let linked = root.join("linked");
    for (label, args, expected) in [
        (
            "delete rejected",
            vec![root.as_path(), Path::new("--delete")],
            2,
        ),
        (
            "delete before root validation",
            vec![missing.as_path(), Path::new("--delete")],
            2,
        ),
        ("missing age", vec![root.as_path(), Path::new("--stale")], 2),
        (
            "invalid age",
            vec![root.as_path(), Path::new("--stale"), Path::new("wat")],
            2,
        ),
        (
            "negative age",
            vec![root.as_path(), Path::new("--stale"), Path::new("-1")],
            2,
        ),
        ("missing root", vec![missing.as_path()], 1),
        ("symlink root", vec![linked.as_path()], 1),
    ] {
        let c = call(&helper, &args);
        assert_eq!(c.code, expected, "{label}: {}", c.stderr);
        assert!(
            !c.stderr.is_empty() && !c.stdout.contains("\"complete\": true"),
            "{label}"
        );
        record(label, &c);
    }
    // Traversal failure: an unreadable directory fails the scan instead of being skipped.
    // Permission changes leave mtimes alone; each mode is restored exactly.
    let mode_of = |p: &Path| fs::symlink_metadata(p).unwrap().permissions().mode() & 0o7777;
    let locked_mode = mode_of(&locked);
    fs::set_permissions(&locked, fs::Permissions::from_mode(0o000)).unwrap();
    let c = call(&helper, &[&root, Path::new("--stale"), Path::new("48")]);
    fs::set_permissions(&locked, fs::Permissions::from_mode(locked_mode)).unwrap();
    assert!(
        c.code != 0 && !c.stdout.contains("\"complete\": true"),
        "traversal failure was suppressed (running as root?)"
    );
    record("traversal failure", &c);
    // A descendant the candidate search never enters (.git) still fails the mtime walk.
    let git_objects = root.join("gitguard/.git/objects");
    let objects_mode = mode_of(&git_objects);
    fs::set_permissions(&git_objects, fs::Permissions::from_mode(0o000)).unwrap();
    let c = call(&helper, &[&root, Path::new("--stale"), Path::new("48")]);
    fs::set_permissions(&git_objects, fs::Permissions::from_mode(objects_mode)).unwrap();
    assert!(
        c.code != 0,
        "candidate mtime traversal failure was suppressed"
    );
    record("descendant mtime traversal failure", &c);
    // Measurement failures (the scan is the library's; the measurement is injected).
    for (label, failure) in [
        (
            "measurement failure",
            Failure::new("Command '['du']' returned non-zero exit status 1."),
        ),
        (
            "measurement timeout",
            Failure::new("Command '['du']' timed out after 600 seconds"),
        ),
    ] {
        let outcome = scan(&root, Some(48), &never_interrupted, &|_, _| {
            Err(failure.clone())
        });
        assert_eq!(outcome.unwrap_err(), failure, "{label}");
    }
    // A real failing du (a vanished path) is an error, never a number.
    assert!(allocated(&scratch.join("absent"), false, &never_interrupted).is_err());
    let after = inventory(&root);
    assert_eq!(
        before, after,
        "fixture contents, membership, links, modes or mtimes changed"
    );
    // The report wrapper writes a failure receipt for a refused configuration.
    let reports = scratch.join("reports");
    fs::create_dir(&reports).unwrap();
    let config = scratch.join("invalid-config.json");
    fs::write(
        &config,
        r#"{"cargo_roots": [], "stale_hours": 48, "measurement_roots": []}"#,
    )
    .unwrap();
    let wrapper = helper.with_file_name("jope-build-report.sh");
    let c = call(
        &wrapper,
        &[
            Path::new("--config"),
            &config,
            Path::new("--custody"),
            &reports,
        ],
    );
    assert_ne!(c.code, 0);
    let run = fs::read_dir(&reports)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let receipt =
        pyjson::parse_str(&fs::read_to_string(run.join("receipt.json")).unwrap()).unwrap();
    assert_eq!(receipt.get("status"), Some(&Value::from("failed")));
    assert_eq!(receipt.get("exit_status"), Some(&Value::Int(1)));
    assert!(!fs::read_to_string(run.join("failure.txt"))
        .unwrap()
        .is_empty());
    fs::write(
        scratch.join("results.json"),
        pyjson::dumps(
            &Value::Dict(
                [
                    ("tests", Value::List(results)),
                    ("wrapper_failure", receipt),
                    ("fixtures", Value::from(before.len())),
                    ("preservation", Value::from("exact unchanged file/directory membership, contents, symlinks, modes and mtimes")),
                ]
                .into_iter()
                .collect(),
            ),
            false,
            Some(2),
        ),
    )
    .unwrap();
}
