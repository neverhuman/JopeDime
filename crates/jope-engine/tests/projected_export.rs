//! W1 follow-up 2c: the fitted-view export worker (`fit-memory`, hidden)
//! receives its projected fitting input as sealed memory and opens no
//! prepared path. The shipped binary still fences every fit until a
//! production namespace verifier exists; the positive export through the
//! fixture authority is
//! `gp::fit_contract_tests::the_export_worker_receives_its_projected_input_and_never_resolves_the_view_itself`.

use jope_parallel::{
    gp,
    manifest::{file_hash, hash, write_json},
    prepare::{csv_bytes, load_view_in_root, Preprocessing},
};
use std::{
    fs,
    os::fd::AsRawFd,
    path::{Path, PathBuf},
};
#[path = "support/fit_authority.rs"]
mod authority;

#[test]
fn shipped_fit_memory_is_fenced_and_refuses_a_mismatched_or_unsealed_projection() {
    let tmp = tempfile::tempdir().unwrap();
    let raw = (0..12)
        .map(|i| vec![i.to_string(), (100 + 3 * i).to_string()])
        .collect::<Vec<_>>();
    let pp = Preprocessing::fit(&raw, "regression").unwrap();
    let rows = pp.transform(&raw).unwrap();
    let pp_bytes = serde_json::to_vec_pretty(&pp).unwrap();
    let memory = gp::fit_memory_input(&rows, &pp_bytes).unwrap();
    let training = hash(&csv_bytes(&rows).unwrap());
    let preprocessing = hash(&pp_bytes);
    let mut authority = authority::Fixture::new(tmp.path(), vec![authority::training_hash(&rows)]);
    {
        let run = |fd: i32, training: &str, output: &std::path::Path| {
            authority
                .command()
                .args(["fit-memory", "--fd", &fd.to_string()])
                .args(["--training-sha256", training])
                .args(["--preprocessing-sha256", &preprocessing])
                .args(["--genome", authority::PROGRAM, "--output"])
                .arg(output)
                .output()
                .unwrap()
        };
        // Fully configured and in scope: still refused at namespace activation.
        let fenced = tmp.path().join("fenced.json");
        let result = run(memory.as_raw_fd(), &training, &fenced);
        let error = String::from_utf8_lossy(&result.stderr);
        assert!(
            !result.status.success(),
            "shipped fit-memory ran without activation"
        );
        assert!(error.contains(authority::UNVERIFIED_ACTIVATION), "{error}");
        assert!(!fenced.exists());
        // A projection that is not the declared cut is refused before any claim.
        let mismatched = tmp.path().join("mismatched.json");
        let result = run(memory.as_raw_fd(), &"0".repeat(64), &mismatched);
        let error = String::from_utf8_lossy(&result.stderr);
        assert!(!result.status.success());
        assert!(error.contains("training identity"), "{error}");
        assert!(!mismatched.exists());
        // An unsealed regular file is not a projected input.
        let plain = std::fs::File::create(tmp.path().join("not-memory")).unwrap();
        let unsealed = tmp.path().join("unsealed.json");
        let result = run(plain.as_raw_fd(), &training, &unsealed);
        let error = String::from_utf8_lossy(&result.stderr);
        assert!(!result.status.success());
        assert!(error.contains("sealed memory"), "{error}");
        assert!(!unsealed.exists());
        authority.assert_no_fit_starts();
    }
    let stats = authority.shutdown();
    assert_eq!(stats["worker_spawns"], 0);
    assert_eq!(stats["committed"], 0);
}

/// A pointer view (backing table `data/originals/raw.csv`) under `root`,
/// which carries no workspace markers of its own.
fn pointer_view(root: &Path) -> PathBuf {
    let raw = (0..12)
        .map(|i| vec![i.to_string(), (100 + i * 3).to_string()])
        .collect::<Vec<_>>();
    let mut source = "x,target\n".to_string();
    for row in &raw {
        source.push_str(&format!("{},{}\n", row[0], row[1]));
    }
    fs::create_dir_all(root.join("data/originals")).unwrap();
    fs::write(root.join("data/originals/raw.csv"), source).unwrap();
    let pp = Preprocessing::fit(&raw[..8], "regression").unwrap();
    let prepared = root.join("data/prepared/example");
    let dir = prepared.join("cut-0");
    fs::create_dir_all(&dir).unwrap();
    write_json(&dir.join("preprocessing.json"), &pp).unwrap();
    let tr = csv_bytes(&pp.transform(&raw[..8]).unwrap()).unwrap();
    let va = csv_bytes(&pp.transform(&raw[8..10]).unwrap()).unwrap();
    let p = serde_json::json!({"schema":"parallel.prepared.v1","dataset":"example","family":"example","task":"regression","meta_partition":"meta_train","seed":7,"family_partition_frozen":false,"clean_evaluation":true,"clean_status":"raw_reconstructed","original_train_sha256":"0".repeat(64),"original_test_sha256":"1".repeat(64),"selected_train_indices":(0..10).collect::<Vec<_>>(),"selected_test_indices":[0,1],"target_last":true,"rows_total":12,"cuts":[{"index":0,"train_indices":(0..8).collect::<Vec<_>>(),"validation_indices":[8,9],"training_sha256":hash(&tr),"validation_sha256":hash(&va),"preprocessing_sha256":file_hash(&dir.join("preprocessing.json")).unwrap()}],"storage":{"source":"data/originals/raw.csv","source_sha256":file_hash(&root.join("data/originals/raw.csv")).unwrap(),"feature_columns":["x"],"target_column":"target","original_training_row_map":(0..10).collect::<Vec<_>>()}});
    write_json(&prepared.join("prepared.json"), &p).unwrap();
    prepared
}

/// W1 route R2 at the public `fit-view` verb: the shipped binary projects the
/// cut's fitting rows under its declared `--root`. With the backing table
/// only in a workspace ancestor it refuses at the projection, before any fit
/// request; with the table under the root only the activation fence refuses.
#[test]
fn shipped_fit_view_projects_under_its_declared_root_before_the_fence() {
    let tmp = tempfile::tempdir().unwrap();
    let outer = tmp.path().canonicalize().unwrap();
    let root = outer.join("declared");
    let prepared = pointer_view(&root);
    let rows = load_view_in_root(&root, &prepared, 0, true).unwrap();
    // The declared root sits below a workspace ancestor holding a
    // byte-identical copy of the backing table.
    let decoy = outer.join("data/originals/raw.csv");
    fs::create_dir_all(outer.join("manifests")).unwrap();
    fs::write(outer.join("Cargo.toml"), "[workspace]\n").unwrap();
    fs::create_dir_all(decoy.parent().unwrap()).unwrap();
    fs::copy(root.join("data/originals/raw.csv"), &decoy).unwrap();
    let mut authority = authority::Fixture::new(&root, vec![authority::training_hash(&rows)]);
    {
        let fit_view = |output: &Path| {
            authority
                .command()
                .arg("--root")
                .arg(&root)
                .args(["fit-view", "--prepared"])
                .arg(&prepared)
                .args(["--cut", "0", "--genome", authority::PROGRAM, "--output"])
                .arg(output)
                .output()
                .unwrap()
        };
        let backing = root.join("data/originals/raw.csv");
        let away = root.join("raw.csv.away");
        fs::rename(&backing, &away).unwrap();
        let outside = root.join("outside.json");
        let result = fit_view(&outside);
        let error = String::from_utf8_lossy(&result.stderr);
        assert!(!result.status.success());
        assert!(
            !error.contains(authority::UNVERIFIED_ACTIVATION),
            "shipped fit-view projected its view from outside the declared root: {error}"
        );
        assert!(error.contains("backing table"), "{error}");
        assert!(!outside.exists());
        authority.assert_no_fit_starts();

        // Positive control: the table under the declared root is projected,
        // and only the activation fence refuses the fit.
        fs::rename(&away, &backing).unwrap();
        fs::remove_file(&decoy).unwrap();
        let fenced = root.join("fenced.json");
        let result = fit_view(&fenced);
        let error = String::from_utf8_lossy(&result.stderr);
        assert!(
            !result.status.success(),
            "shipped fit-view ran without activation"
        );
        assert!(error.contains(authority::UNVERIFIED_ACTIVATION), "{error}");
        assert!(!fenced.exists());
        authority.assert_no_fit_starts();
    }
    let stats = authority.shutdown();
    assert_eq!(stats["worker_spawns"], 0);
    assert_eq!(stats["committed"], 0);
}
