//! W1 consumer-map routes R1, R11 and R8 at the preparation receiver.
//!
//! An environment variable is not an input: changing `JOPE_REGISTRY` (and so
//! `jope_paths::quant_root`) must not change which backing object preparation
//! reads (R1) or whether a family partition is frozen (R11). The historical
//! training split is consumed only as the bytes the audit bound (R8).
//!
//! The process environment is shared by the tests of one binary, so every
//! test that sets a variable holds `ENV` and restores what it set.

#[path = "support/legacy_dataset.rs"]
mod legacy;

use jope_parallel::manifest::{file_hash, read_json, write_json};
use legacy::{Fixture, Spec};
use serde_json::{json, Value};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::Mutex,
};

static ENV: Mutex<()> = Mutex::new(());

/// Run `f` with `JOPE_REGISTRY` set to `registry`, restoring the prior value.
fn with_registry<T>(registry: &Path, f: impl FnOnce() -> T) -> T {
    let previous = std::env::var_os("JOPE_REGISTRY");
    std::env::set_var("JOPE_REGISTRY", registry);
    let result = f();
    match previous {
        Some(value) => std::env::set_var("JOPE_REGISTRY", value),
        None => std::env::remove_var("JOPE_REGISTRY"),
    }
    result
}

fn meta_path(f: &Fixture) -> PathBuf {
    f.legacy().join("meta.json")
}

fn set_meta(f: &Fixture, key: &str, value: Value) {
    let mut meta: Value = read_json(&meta_path(f)).unwrap();
    meta[key] = value;
    write_json(&meta_path(f), &meta).unwrap();
}

/// Backing allowlist entry for `path`, served at `destination` under the root.
fn allow(f: &Fixture, path: &Path, destination: &str) -> Value {
    let target = f.root.join(destination);
    fs::create_dir_all(target.parent().unwrap()).unwrap();
    if target != path {
        fs::copy(path, &target).unwrap();
    }
    json!({"source":path,"destination":destination,"bytes":fs::metadata(path).unwrap().len(),
        "sha256":file_hash(path).unwrap(),"dataset":legacy::DATASET,"role":"backing_source"})
}

fn write_allowlist(f: &Fixture, entries: Vec<Value>) {
    write_json(
        &f.root.join("inputs/backing.json"),
        &json!({"schema":"parallel.transfer.v1","files":entries}),
    )
    .unwrap();
}

fn prepared_source(prepared: &Path) -> Value {
    let p: Value = read_json(&prepared.join("prepared.json")).unwrap();
    p["storage"]["source_sha256"].clone()
}

/// R1: the recorded backing path of a historical dataset no longer exists.
/// Two allowlisted copies share its relative name: the original bytes and an
/// alternate version whose rows outside the historical training table differ.
/// Which one preparation reads must not depend on `JOPE_REGISTRY`.
#[test]
fn a_missing_recorded_backing_path_is_never_re_rooted_by_the_environment() {
    let _env = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let f = Fixture::new(Spec::default());
    let original = f.backing();
    // Historical host path, absent here.
    set_meta(
        &f,
        "source_path",
        json!(f.root.join("retired-host/hyper/data/raw.csv")),
    );
    // Mirror A (identical bytes) and mirror B (alternate version).
    let mirror_a = f.root.join("mirror-a/hyper/data/raw.csv");
    fs::create_dir_all(mirror_a.parent().unwrap()).unwrap();
    fs::copy(&original, &mirror_a).unwrap();
    let mut alternate = f.spec.clone();
    alternate.source[35].y = 999.0;
    let mirror_b = f.root.join("mirror-b/hyper/data/raw.csv");
    fs::create_dir_all(mirror_b.parent().unwrap()).unwrap();
    let mut text = "x1,x2,c,target\n".to_string();
    for row in &alternate.source {
        text.push_str(&format!("{},{},{},{}\n", row.x1, row.x2, row.c, row.y));
    }
    fs::write(&mirror_b, text).unwrap();
    assert_ne!(file_hash(&mirror_a).unwrap(), file_hash(&mirror_b).unwrap());
    write_allowlist(
        &f,
        vec![
            allow(&f, &original, "data/originals/raw.csv"),
            allow(&f, &mirror_b, "data/originals/raw-b.csv"),
        ],
    );
    let audit = f.audit("r1").unwrap();
    let registry = |mirror: &str| f.root.join(mirror).join("quant/regression");
    let run = |label: &str, mirror: &str| {
        with_registry(&registry(mirror), || f.prepare(&audit, label))
            .map(|prepared| prepared_source(&prepared))
            .map_err(|e| format!("{e:#}"))
    };
    let under_a = run("env-a", "mirror-a");
    let under_b = run("env-b", "mirror-b");
    assert_eq!(
        under_a, under_b,
        "JOPE_REGISTRY selected the backing object: {under_a:?} vs {under_b:?}"
    );
    // The object read is the one relocation the allowlist names (mirror B),
    // and the receipt says so.
    assert_eq!(under_a, Ok(json!(file_hash(&mirror_b).unwrap())));
    let receipt: Value = read_json(
        &f.root
            .join("prepared-env-a")
            .join(legacy::DATASET)
            .join("reconstruction.json"),
    )
    .unwrap();
    let resolution = &receipt["receipt"]["source_resolution"];
    assert_eq!(resolution["kind"], "allowlisted_relocation");
    assert_eq!(resolution["allowlisted_source"], json!(mirror_b));

    // Negative: two allowlisted relocations of the same relative identity.
    write_allowlist(
        &f,
        vec![
            allow(&f, &mirror_a, "data/originals/raw-a.csv"),
            allow(&f, &mirror_b, "data/originals/raw-b.csv"),
        ],
    );
    let ambiguous = run("ambiguous", "mirror-a");
    assert!(
        ambiguous
            .as_ref()
            .is_err_and(|e| e.contains("exactly one required")),
        "{ambiguous:?}"
    );
    // Negative: no allowlisted relocation; the environment still cannot supply one.
    write_allowlist(&f, vec![allow(&f, &original, "data/originals/raw.csv")]);
    let none = run("none", "mirror-a");
    assert!(
        none.as_ref().is_err_and(|e| e.contains("relocations")),
        "{none:?}"
    );
    for label in ["ambiguous", "none"] {
        assert!(!f
            .root
            .join(format!("prepared-{label}"))
            .join(legacy::DATASET)
            .exists());
    }
}

/// R1 positive control: a recorded backing path that exists is read as
/// recorded, independent of the environment.
#[test]
fn an_existing_recorded_backing_path_is_read_as_recorded() {
    let _env = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let f = Fixture::new(Spec::default());
    let audit = f.audit("recorded").unwrap();
    let a = with_registry(&f.root.join("x/quant/regression"), || {
        f.prepare(&audit, "a")
    })
    .unwrap();
    let b = with_registry(&f.root.join("y/quant/regression"), || {
        f.prepare(&audit, "b")
    })
    .unwrap();
    assert_eq!(prepared_source(&a), json!(file_hash(&f.backing()).unwrap()));
    assert_eq!(
        fs::read(a.join("prepared.json")).unwrap(),
        fs::read(b.join("prepared.json")).unwrap()
    );
    let receipt: Value = read_json(&a.join("reconstruction.json")).unwrap();
    assert_eq!(
        receipt["receipt"]["source_resolution"]["kind"],
        "recorded_path"
    );
}

/// R11: the same audit and inputs, prepared under two values of
/// `JOPE_REGISTRY`, must not differ in whether the family partition is frozen.
#[test]
fn the_frozen_family_partition_is_never_decided_by_the_environment() {
    let _env = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let f = Fixture::new(Spec::default());
    let audit = f.audit("r11").unwrap();
    let report: Value = read_json(&audit.join("audit.json")).unwrap();
    assert_eq!(report["complete"], true, "fixture audit must be complete");
    let frozen = |label: &str, registry: PathBuf| {
        let prepared = with_registry(&registry, || f.prepare(&audit, label)).unwrap();
        read_json::<Value>(&prepared.join("prepared.json")).unwrap()["family_partition_frozen"]
            .clone()
    };
    let over_audited_root = frozen("inside", f.root.join("legacy/regression"));
    let elsewhere = frozen("outside", f.root.join("elsewhere/regression"));
    assert_eq!(
        over_audited_root, elsewhere,
        "JOPE_REGISTRY decided family_partition_frozen"
    );
    assert_eq!(
        over_audited_root, false,
        "no explicit authority: not frozen"
    );

    // Positive control: the caller declares the audited tree authoritative.
    let declared = |label: &str, root: &Path| {
        let output = f
            .root
            .join(format!("prepared-{label}"))
            .join(legacy::DATASET);
        jope_parallel::prepare::run_with_admission(
            &audit.join("audit.json"),
            legacy::DATASET,
            &output,
            false,
            true,
            &f.root.join("inputs/backing.json"),
            None,
            None,
            Some(root),
        )
        .unwrap()
    };
    let p = declared("declared", &f.root.join("legacy"));
    assert!(p.family_partition_frozen);
    let receipt: Value = read_json(
        &f.root
            .join("prepared-declared")
            .join(legacy::DATASET)
            .join("reconstruction.json"),
    )
    .unwrap();
    assert_eq!(
        receipt["family_partition_frozen_authority"]["kind"],
        "declared_inventory_root"
    );
    // Negative: a declared root the audit did not cover.
    assert!(!declared("declared-other", &f.root.join("elsewhere")).family_partition_frozen);
}

/// R8: the historical training split changed after the audit bound its bytes
/// (`transfer.json`). Preparation must refuse, not prepare the new bytes.
#[test]
fn a_training_split_changed_after_its_audit_is_refused() {
    let _env = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let mut f = Fixture::new(Spec::default());
    let audit = f.audit("r8").unwrap();
    f.spec.train.swap(0, 1);
    f.write();
    let error = f
        .prepare(&audit, "changed-after-audit")
        .map(|_| ())
        .map_err(|e| format!("{e:#}"));
    assert!(
        error.as_ref().is_err_and(|e| e.contains("training split")),
        "prepared a training split other than the audited bytes: {error:?}"
    );
    assert!(!f
        .root
        .join("prepared-changed-after-audit")
        .join(legacy::DATASET)
        .exists());
    // Positive control: the audited bytes prepare, and the recorded training
    // identity is the audited digest of exactly the consumed bytes.
    f.spec.train.swap(0, 1);
    f.write();
    let prepared = f.prepare(&audit, "as-audited").unwrap();
    let transfer: Value = read_json(&audit.join("transfer.json")).unwrap();
    let audited = transfer["files"]
        .as_array()
        .unwrap()
        .iter()
        .find(|e| e["source"] == json!(f.train_csv()))
        .expect("the audit binds the training split")["sha256"]
        .clone();
    assert_eq!(f.prepared(&prepared).original_train_sha256, audited);
    let receipt: Value = read_json(&prepared.join("reconstruction.json")).unwrap();
    assert!(receipt["training_split_identity_source"]
        .as_str()
        .unwrap()
        .starts_with("audited"));
}
