use super::*;

/// Relocate the pointer view's declared `root` below a workspace ancestor
/// `outer` that holds a byte-identical copy of the backing table, and remove
/// the root's own workspace markers. Returns the decoy copy.
pub(super) fn decoy_workspace_ancestor(outer: &Path, root: &Path) -> PathBuf {
    fs::remove_dir_all(root.join("manifests")).unwrap();
    fs::remove_file(root.join("Cargo.toml")).unwrap();
    let decoy = outer.join("data/originals/raw.csv");
    fs::create_dir_all(outer.join("manifests")).unwrap();
    fs::write(outer.join("Cargo.toml"), "[workspace]\n").unwrap();
    fs::create_dir_all(decoy.parent().unwrap()).unwrap();
    fs::copy(root.join("data/originals/raw.csv"), &decoy).unwrap();
    decoy
}

/// A contained (table) view of the same cut: `train.csv`/`validation.csv`
/// beside its preparation state, no backing table and no root.
pub(super) fn contained_view_fixture(root: &Path, pointer: &Path, name: &str) -> PathBuf {
    let prepared = root.join("data/prepared").join(name);
    let dir = prepared.join("cut-0");
    fs::create_dir_all(&dir).unwrap();
    for (file, training) in [("train.csv", true), ("validation.csv", false)] {
        let rows = crate::prepare::load_view_in_root(root, pointer, 0, training).unwrap();
        fs::write(dir.join(file), csv_bytes(&rows).unwrap()).unwrap();
    }
    fs::copy(
        pointer.join("cut-0/preprocessing.json"),
        dir.join("preprocessing.json"),
    )
    .unwrap();
    let mut p: Value = read_json(&pointer.join("prepared.json")).unwrap();
    p.as_object_mut().unwrap().remove("storage");
    p["dataset"] = json!(name);
    write_json(&prepared.join("prepared.json"), &p).unwrap();
    prepared
}

#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
pub(super) fn the_public_fit_view_projects_under_its_declared_root_and_never_resolves_an_ancestor()
{
    let temp = tempfile::tempdir().unwrap();
    let outer = temp.path().canonicalize().unwrap();
    let root = outer.join("declared");
    fs::create_dir_all(&root).unwrap();
    let prepared = view_fixture_in(&root, "example", &[8, 9]);
    let training = cut_training_hash(&prepared);
    let mut authority = Fixture::activated(&root, vec![training.clone()]);
    authority.start();
    let decoy = decoy_workspace_ancestor(&outer, &root);
    let fit_view = |output: &Path| {
        authority
            .command()
            .args(["fit-view", "--root"])
            .arg(&root)
            .arg("--prepared")
            .arg(&prepared)
            .args(["--cut", "0", "--genome", PROGRAM, "--output"])
            .arg(output)
            .output()
            .unwrap()
    };

    // The backing table exists only outside the declared root: the fit must
    // refuse rather than read it from the workspace ancestor.
    let backing = root.join("data/originals/raw.csv");
    let away = root.join("raw.csv.away");
    fs::rename(&backing, &away).unwrap();
    let outside = root.join("fitted-outside.json");
    let refused = fit_view(&outside);
    assert!(
        !refused.status.success(),
        "the public fit-view read its view from outside the declared root"
    );
    assert!(
        String::from_utf8_lossy(&refused.stderr).contains("backing table"),
        "{}",
        String::from_utf8_lossy(&refused.stderr)
    );
    assert!(!outside.exists());
    authority.assert_no_fit_starts();

    // Positive control: the backing table only inside the declared root, the
    // decoy ancestor still a workspace without it.
    fs::rename(&away, &backing).unwrap();
    fs::remove_file(&decoy).unwrap();
    let inside = root.join("fitted.json");
    success(fit_view(&inside));
    let predictor: Predictor = read_json(&inside).unwrap();
    assert_eq!(predictor.training_sha256, training_of(&prepared));
    authority.assert_one_original_fit();
    authority.stop_service();
}

/// The cut's recorded training identity.
pub(super) fn training_of(prepared: &Path) -> String {
    let p: Prepared = read_json(&prepared.join("prepared.json")).unwrap();
    p.cuts[0].training_sha256.clone()
}

#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
pub(super) fn the_legacy_isolated_export_never_resolves_a_pointer_view_from_its_ancestors() {
    let temp = tempfile::tempdir().unwrap();
    let outer = temp.path().canonicalize().unwrap();
    let root = outer.join("declared");
    fs::create_dir_all(&root).unwrap();
    let prepared = view_fixture_in(&root, "example", &[8, 9]);
    let contained = contained_view_fixture(&root, &prepared, "contained");
    let training = cut_training_hash(&prepared);
    let executable = std::env::current_exe().unwrap();
    let mut authority = Fixture::activated(&root, vec![training.clone()]);
    authority.start();
    let _decoy = decoy_workspace_ancestor(&outer, &root);
    let isolated = |view: &Path, root: Option<&Path>, output: &Path| {
        let mut command = authority.command();
        command
            .args(["fit-view-isolated", "--executable"])
            .arg(&executable);
        if let Some(root) = root {
            command.arg("--root").arg(root);
        }
        command
            .arg("--prepared")
            .arg(view)
            .args(["--cut", "0", "--genome", PROGRAM, "--output"])
            .arg(output)
            .output()
            .unwrap()
    };

    // No declared root: the pointer view is refused before any worker starts;
    // its backing table is never looked up in the workspace ancestor.
    let unrooted = root.join("unrooted.json");
    let refused = isolated(&prepared, None, &unrooted);
    assert!(
        !refused.status.success(),
        "the legacy export resolved a pointer view from the view's ancestors"
    );
    assert!(!unrooted.exists());
    let receipt: Value = read_json(&unrooted.with_extension("fit.json")).unwrap();
    assert_eq!(receipt["passed"], false);
    assert_eq!(receipt["view_resolution"], "contained_view");
    assert!(receipt["worker_argv"].as_array().is_none_or(Vec::is_empty));
    assert!(
        receipt["failure"]
            .as_str()
            .unwrap()
            .contains("declared data root"),
        "{receipt}"
    );
    authority.assert_no_fit_starts();

    // Positive controls: the same pointer view under its declared root, and a
    // contained view without any root, each export through the sealed
    // projection.
    for (name, view, declared) in [
        ("rooted", &prepared, Some(root.as_path())),
        ("contained", &contained, None),
    ] {
        let output = root.join(format!("{name}.json"));
        success(isolated(view, declared, &output));
        let receipt: Value = read_json(&output.with_extension("fit.json")).unwrap();
        assert_eq!(receipt["passed"], true, "{name}");
        assert_eq!(receipt["worker_input"]["kind"], "sealed_memory_projection");
        assert_eq!(
            receipt["view_resolution"],
            if declared.is_some() {
                "declared_root"
            } else {
                "contained_view"
            }
        );
        let exported: Predictor = read_json(&output).unwrap();
        assert_eq!(exported.training_sha256, training_of(view));
    }
    authority.stop_service();
}
