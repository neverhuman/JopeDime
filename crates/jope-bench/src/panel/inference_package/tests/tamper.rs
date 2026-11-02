use super::*;

#[test]
fn intact_packages_load_and_tampered_installations_refuse_before_serving() {
    let root = scratch();
    let (a, _) = fixture(Task::Regression, "ridge");
    let (b, _) = fixture(Task::Binary, "logit_ridge");
    let a_bytes = a.bytes().unwrap();
    let mut refused = Vec::new();
    let mut accepted = Vec::new();
    for backend in [Backend::Python, Backend::Rust, Backend::Cpp17] {
        let q = qualify(root.path(), &a, backend);
        let (manifest, files) = seal_default(&q, true);
        let intact = root.path().join(format!("{backend:?}-intact"));
        let identity = install(&intact, &manifest, &files).unwrap();
        let loaded = load(&intact, &identity).unwrap();
        assert_eq!(loaded.manifest.fitted_sha256, q.plan.fitted_sha256);
        assert_eq!(
            loaded.bundle.identity().unwrap(),
            q.bundle.identity().unwrap()
        );
        if backend != Backend::Rust {
            // The installed bytes, not the build directory, serve the qualified output.
            let module = if backend == Backend::Python {
                "model"
            } else {
                "boundary"
            };
            let served = serve_python(&intact, module);
            assert_eq!(digest(&served), q.result.prediction_sha256);
        }
        let main = match backend {
            Backend::Python => "model.py",
            Backend::Rust => "model.rs",
            _ => "model.cpp",
        };
        let mut mutations: Vec<Mutation> = vec![
            (
                "changed-source",
                Box::new(move |d: &Path| {
                    let mut text = fs::read(d.join(main)).unwrap();
                    text.push(b' ');
                    fs::write(d.join(main), text).unwrap();
                }),
            ),
            (
                "missing-wrapper",
                Box::new(move |d: &Path| {
                    let wrapper = if main == "model.cpp" {
                        "boundary.py"
                    } else {
                        main
                    };
                    fs::remove_file(d.join(wrapper)).unwrap();
                }),
            ),
            (
                "extra-file",
                Box::new(|d: &Path| fs::write(d.join("sitecustomize.py"), "import os\n").unwrap()),
            ),
            (
                "linked-file",
                Box::new(move |d: &Path| {
                    // Identical bytes, but reached through a link outside the package.
                    let outside = d.with_extension("link-target");
                    fs::copy(d.join(main), &outside).unwrap();
                    fs::remove_file(d.join(main)).unwrap();
                    std::os::unix::fs::symlink(&outside, d.join(main)).unwrap();
                }),
            ),
            (
                "rebound-wrong-backend-receipt",
                Box::new(|d: &Path| {
                    let mut value: serde_json::Value =
                        serde_json::from_slice(&fs::read(d.join(PACKAGE_FILE)).unwrap()).unwrap();
                    value["qualification"]["backend"] = serde_json::json!("javascript");
                    fs::write(
                        d.join(PACKAGE_FILE),
                        serde_json::to_vec_pretty(&value).unwrap(),
                    )
                    .unwrap();
                }),
            ),
            (
                "rebound-other-state-receipt",
                Box::new(|d: &Path| {
                    let mut value: serde_json::Value =
                        serde_json::from_slice(&fs::read(d.join(PACKAGE_FILE)).unwrap()).unwrap();
                    value["qualification"]["fitted_sha256"] = serde_json::json!("0".repeat(64));
                    fs::write(
                        d.join(PACKAGE_FILE),
                        serde_json::to_vec_pretty(&value).unwrap(),
                    )
                    .unwrap();
                }),
            ),
        ];
        if matches!(backend, Backend::Cpp17) {
            mutations.push((
                "substituted-executable",
                Box::new(|d: &Path| {
                    let mut bytes = fs::read(d.join("model.so")).unwrap();
                    let last = bytes.len() - 1;
                    bytes[last] ^= 1;
                    fs::write(d.join("model.so"), bytes).unwrap();
                }),
            ));
            mutations.push((
                "missing-stage-codec",
                Box::new(|d: &Path| fs::remove_file(d.join("codec.json")).unwrap()),
            ));
            mutations.push((
                "missing-native-library",
                Box::new(|d: &Path| fs::remove_file(d.join("model.so")).unwrap()),
            ));
        }
        for (label, mutate) in &mutations {
            let directory = root.path().join(format!("{backend:?}-{label}"));
            copy(&intact, &directory);
            mutate(&directory);
            // A rebound manifest is presented with its own new digest, so the internal
            // bindings, not only the retained reference, must refuse it.
            let presented = if label.starts_with("rebound") {
                digest(&fs::read(directory.join(PACKAGE_FILE)).unwrap())
            } else {
                identity.clone()
            };
            match load(&directory, &presented) {
                Ok(_) => accepted.push(format!("{backend:?}/{label}")),
                Err(_) => refused.push(format!("{backend:?}/{label}")),
            }
        }
        // Stale reference: a different package installed where A's reference points.
        let other = qualify(root.path(), &b, backend);
        let (other_manifest, other_files) = seal_default(&other, true);
        let stale = root.path().join(format!("{backend:?}-stale"));
        install(&stale, &other_manifest, &other_files).unwrap();
        assert!(load(&stale, &identity).is_err());
        // Seal-time refusals: wrong-backend receipt, receipt from another state/plan,
        // unqualified native bytes and a profile that is not the executed one.
        let rust = qualify(
            root.path(),
            &a,
            if backend == Backend::Rust {
                Backend::Python
            } else {
                Backend::Rust
            },
        );
        assert!(seal(
            &q.bundle,
            &q.plan,
            &rust.result,
            profile(&q),
            &q.built,
            &q.executed,
            true
        )
        .is_err());
        assert!(seal(
            &q.bundle,
            &q.plan,
            &other.result,
            profile(&q),
            &q.built,
            &q.executed,
            true
        )
        .is_err());
        let mut changed = profile(&q);
        changed.build_commands.push(vec!["true".into()]);
        assert!(seal(
            &q.bundle,
            &q.plan,
            &q.result,
            changed,
            &q.built,
            &q.executed,
            true
        )
        .is_err());
        if backend == Backend::Cpp17 {
            let mut built = q.built.clone();
            built.get_mut("model.so").unwrap().push(0);
            assert!(seal(
                &q.bundle,
                &q.plan,
                &q.result,
                profile(&q),
                &built,
                &q.executed,
                true
            )
            .is_err());
            assert!(seal(
                &q.bundle,
                &q.plan,
                &q.result,
                profile(&q),
                &BTreeMap::new(),
                &q.executed,
                true
            )
            .is_err());
        }
        // The intact installation still loads; nothing above altered it.
        load(&intact, &identity).unwrap();
    }
    assert_eq!(
        a.bytes().unwrap(),
        a_bytes,
        "packaging is read-only for the fitted subject"
    );
    assert!(
        accepted.is_empty(),
        "accepted tampered packages: {accepted:?}"
    );
    assert_eq!(refused.len(), 6 + 6 + 9);
}

/// Reproduction of the gap this module closes: the existing public manifest binds only
/// source files, so a substituted native library beside intact sources passes
/// `PublicBundle::validate` and silently serves other predictions. `load` refuses it.
#[test]
fn public_manifest_alone_cannot_see_a_substituted_native_library() {
    let root = scratch();
    let (a, _) = fixture(Task::Regression, "ridge");
    let (b, _) = fixture(Task::Binary, "logit_ridge");
    let q = qualify(root.path(), &a, Backend::Cpp17);
    let other = qualify(root.path(), &b, Backend::Cpp17);
    let (manifest, files) = seal_default(&q, true);
    let installed = root.path().join("installed");
    let identity = install(&installed, &manifest, &files).unwrap();
    assert_eq!(
        digest(&serve_python(&installed, "boundary")),
        q.result.prediction_sha256
    );
    fs::write(installed.join("model.so"), &other.built["model.so"]).unwrap();
    let public = PublicBundle {
        manifest: manifest.public.clone(),
        files: q
            .bundle
            .files
            .keys()
            .map(|name| {
                (
                    name.clone(),
                    fs::read_to_string(installed.join(name)).unwrap(),
                )
            })
            .collect(),
    };
    public.validate().unwrap();
    assert_ne!(
        digest(&serve_python(&installed, "boundary")),
        q.result.prediction_sha256
    );
    let refused = load(&installed, &identity).unwrap_err();
    assert!(format!("{refused:#}").contains("shipping file changed: model.so"));
}
