use super::*;

#[test]
fn changed_serving_dependency_refuses_and_required_failures_block() {
    let root = scratch();
    let (a, _) = fixture(Task::Regression, "ridge");
    let q = qualify(root.path(), &a, Backend::Python);
    // A pinned stand-in for the declared interpreter, so it can be changed safely.
    let tool = root.path().join("python3-pinned");
    fs::write(&tool, b"pinned interpreter bytes").unwrap();
    let tool = fs::canonicalize(&tool).unwrap();
    let mut declared = profile(&q);
    declared.serve.insert("python3".into(), pin(&tool));
    let (manifest, files) = seal(
        &q.bundle,
        &q.plan,
        &q.result,
        declared,
        &q.built,
        &q.executed,
        true,
    )
    .unwrap();
    let python = root.path().join("python");
    let identity = install(&python, &manifest, &files).unwrap();
    let loaded = load(&python, &identity).unwrap();
    fs::write(&tool, b"changed interpreter bytes").unwrap();
    let refused = load(&python, &identity).unwrap_err();
    assert!(format!("{refused:#}").contains("serving dependency changed"));

    let js = qualify(root.path(), &a, Backend::JavaScript);
    let (manifest, files) = seal_default(&js, false);
    let javascript = root.path().join("javascript");
    let js_identity = install(&javascript, &manifest, &files).unwrap();
    let js_loaded = load(&javascript, &js_identity).unwrap();
    let request = PackageRequest {
        required: vec![Backend::Python],
        optional: vec![Backend::JavaScript],
    };
    let complete = assess(
        &request,
        &[
            (Backend::Python, Ok(loaded.clone())),
            (Backend::JavaScript, Ok(js_loaded.clone())),
        ],
    )
    .unwrap();
    assert_eq!(complete["state"], "complete");
    let partial = assess(
        &request,
        &[
            (Backend::Python, Ok(loaded.clone())),
            (Backend::JavaScript, Err("node absent".into())),
        ],
    )
    .unwrap();
    assert_eq!(partial["state"], "partial");
    assert_eq!(partial["optional_failures"][0]["backend"], "javascript");
    let blocked = assess(
        &request,
        &[
            (Backend::Python, Err(format!("{refused:#}"))),
            (Backend::JavaScript, Ok(js_loaded.clone())),
        ],
    )
    .unwrap();
    assert_eq!(blocked["state"], "blocked");
    // An absent required backend blocks; nothing is inferred from another backend.
    assert_eq!(
        assess(&request, &[(Backend::JavaScript, Ok(js_loaded.clone()))]).unwrap()["state"],
        "blocked"
    );
    // A package sealed as optional cannot satisfy a required slot, nor can a stray backend.
    let swapped = PackageRequest {
        required: vec![Backend::JavaScript],
        optional: vec![Backend::Python],
    };
    assert!(assess(&swapped, &[(Backend::JavaScript, Ok(js_loaded.clone()))]).is_err());
    assert!(assess(&request, &[(Backend::Rust, Err("x".into()))]).is_err());
    for value in [complete, partial, blocked] {
        assert_eq!(value["release"], "not-decided");
    }
}
