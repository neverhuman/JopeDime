//! W1 follow-up 2e: the W1-T06 counterfactual harness extended to the
//! teacher's input admission, `label_admission::verify_inputs`. This is the
//! receiving path that binds an admitted teacher label to the view it was
//! evaluated on: under the declared root it loads the cut's training and
//! validation views and returns the ordered training identity.
//!
//! * **Forbidden:** the sealed final-test split (labels, order, categories
//!   and a denied read) for every cut, and one cut's held-out rows for that
//!   cut's admitted training identity.
//! * **Authorized controls:** the same held-out rows for every other cut, and
//!   a training label of the cut, must change the admitted identity.
//! * **Refusals:** stale preparation bytes and a family in the sealed meta
//!   partition.
//!
//! The shared fixture audits one family, which the audit always places in
//! `meta_test`, and the teacher never admits that partition. The teacher
//! proofs therefore add one distinct, approved, never-prepared companion
//! dataset of a family that ranks after the example's. The two-family audit
//! then places the example's family in `meta_train`. The example's own
//! inputs and decision are unchanged.
//!
//! Tolerance: zero. The admitted identity is a SHA-256 digest of
//! deterministic single-threaded output. One receipt per mutation is written
//! to `$JOPE_W1_RECEIPTS/counterfactual/teacher-*.json`.

#[path = "support/legacy_dataset.rs"]
mod legacy;

use jope_parallel::{
    corpus::ContextRow,
    label_admission,
    manifest::{file_hash, hash, read_json, write_json, SEED},
    prepare::csv_bytes,
};
use legacy::{Fixture, Run, Spec, DATASET};
use serde_json::{json, Value};
use std::{fs, path::Path};

const TOLERANCE: &str = "exact byte equality of SHA-256 digests (zero tolerance)";
const COMPANION: &str = "companion";

/// The first companion family name the audit ranks after the example's family.
fn companion_family() -> String {
    let rank = |family: &str| hash(format!("{SEED}:family:{family}").as_bytes());
    let example = rank("example-family");
    (0..)
        .map(|i| format!("companion-family-{i}"))
        .find(|f| rank(f) > example)
        .unwrap()
}

/// Rewrite every input from the spec, then add the companion dataset: other
/// rows, another source identity and its own approved decision.
fn write(f: &Fixture) {
    f.write();
    let dir = f.root.join("legacy/regression").join(COMPANION);
    fs::create_dir_all(&dir).unwrap();
    let rows = |n: usize, shift: f64| {
        (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                vec![t, 0.5, 1.0 / 3.0, (t + shift) / (1.0 + shift)]
            })
            .collect::<Vec<_>>()
    };
    fs::write(dir.join("train.csv"), csv_bytes(&rows(24, 0.25)).unwrap()).unwrap();
    fs::write(dir.join("test.csv"), csv_bytes(&rows(6, 0.5)).unwrap()).unwrap();
    let mut meta: Value = read_json(&f.legacy().join("meta.json")).unwrap();
    meta["dataset_hash"] = json!(COMPANION);
    meta["source_dataset_id"] = json!("companion-source");
    write_json(&dir.join("meta.json"), &meta).unwrap();
    let path = f.root.join("inputs/decisions.json");
    let mut decisions: Value = read_json(&path).unwrap();
    let mut companion = decisions[0].clone();
    companion["key"] = json!(format!("dataset:{COMPANION}"));
    companion["family"] = json!(companion_family());
    decisions.as_array_mut().unwrap().push(companion);
    write_json(&path, &decisions).unwrap();
}

/// A fixture whose example family the teacher may admit.
fn fixture() -> Fixture {
    let f = Fixture::new(Spec::default());
    write(&f);
    f
}

/// The corpus context row a teacher label of `cut` names. The corpus record
/// itself is not read by input admission.
fn context(f: &Fixture, run: &Run, cut: usize) -> ContextRow {
    let p = f.prepared(&run.prepared);
    ContextRow {
        dataset: DATASET.into(),
        family: p.family,
        cut,
        partition: p.meta_partition,
        path: "corpus/record.json".into(),
        sha256: "0".repeat(64),
        prepared: run.prepared.strip_prefix(&f.root).unwrap().to_path_buf(),
        prepared_sha256: file_hash(&run.prepared.join("prepared.json")).unwrap(),
    }
}

/// The teacher's admitted training identity for every cut of one run.
fn admitted(f: &Fixture, run: &Run) -> Vec<String> {
    assert!(
        run.audit_error.is_none() && run.prepare_error.is_none(),
        "{:?} {:?}",
        run.audit_error,
        run.prepare_error
    );
    let p = f.prepared(&run.prepared);
    assert!(
        matches!(p.meta_partition.as_str(), "meta_train" | "meta_validation"),
        "fixture family is not teacher-eligible: {}",
        p.meta_partition
    );
    (0..p.cuts.len())
        .map(|cut| label_admission::verify_inputs(&f.root, &context(f, run, cut), &p).unwrap())
        .collect()
}

fn receipt(mutation: &str, class: &str, perturbed: &str, base: &[String], mutated: &[String]) {
    if let Some(dir) = std::env::var_os("JOPE_W1_RECEIPTS") {
        let dir = Path::new(&dir).join("counterfactual");
        fs::create_dir_all(&dir).unwrap();
        let value = json!({"mutation":mutation,"class":class,"perturbed":perturbed,
            "receiver":"jope_parallel::label_admission::verify_inputs",
            "base_admitted_training_sha256":base,"mutated_admitted_training_sha256":mutated,
            "tolerance":TOLERANCE});
        fs::write(
            dir.join(format!("teacher-{mutation}.json")),
            serde_json::to_vec_pretty(&value).unwrap(),
        )
        .unwrap();
    }
}

/// Source indices of cut `cut`'s held-out rows and training rows.
fn memberships(f: &Fixture, run: &Run, cut: usize) -> (Vec<usize>, Vec<usize>) {
    let p = f.prepared(&run.prepared);
    let map = &p.storage.as_ref().unwrap().original_training_row_map;
    let c = &p.cuts[cut];
    (
        c.validation_indices.iter().map(|&i| map[i]).collect(),
        c.train_indices.iter().map(|&i| map[i]).collect(),
    )
}

#[test]
fn sealed_split_mutations_never_reach_the_teachers_admitted_inputs() {
    type Mutate = fn(&mut Spec);
    let cases: [(&str, &str, Mutate); 4] = [
        (
            "sealed-labels-changed",
            "every sealed final-test label replaced",
            |s| {
                for (i, row) in s.test.iter_mut().enumerate() {
                    row.y = 205.0 - 3.25 * i as f64;
                }
            },
        ),
        (
            "sealed-labels-shuffled",
            "sealed final-test labels permuted across rows",
            |s| {
                let labels: Vec<f64> = s.test.iter().map(|r| r.y).collect();
                for (i, row) in s.test.iter_mut().enumerate() {
                    row.y = labels[(i + 3) % labels.len()];
                }
            },
        ),
        (
            "sealed-rows-reordered",
            "sealed final-test rows reversed",
            |s| s.test.reverse(),
        ),
        (
            "sealed-categories-changed",
            "every sealed final-test category rotated to another recorded category",
            |s| {
                for row in &mut s.test {
                    row.c = match row.c.as_str() {
                        "red" => "blue",
                        "blue" => "green",
                        _ => "red",
                    }
                    .into();
                }
            },
        ),
    ];
    for (name, perturbed, mutate) in cases {
        let mut f = fixture();
        let base_run = f.run("baseline");
        let base = admitted(&f, &base_run);
        mutate(&mut f.spec);
        write(&f);
        let run = f.run(name);
        let mutated = admitted(&f, &run);
        // The mutation is real: the preparation names the changed sealed file.
        assert_ne!(
            file_hash(&run.prepared.join("prepared.json")).unwrap(),
            file_hash(&base_run.prepared.join("prepared.json")).unwrap(),
            "{name} changed nothing"
        );
        assert_eq!(
            mutated, base,
            "{name} reached the teacher's admitted inputs"
        );
        receipt(name, "forbidden", perturbed, &base, &mutated);
    }
}

/// Denied read: with the sealed split absent, the teacher's admission of every
/// cut reproduces the baseline. A family in the sealed meta partition is
/// refused.
#[test]
fn the_teachers_input_admission_never_opens_the_sealed_split() {
    let f = fixture();
    let run = f.run("baseline");
    let base = admitted(&f, &run);
    fs::rename(f.test_csv(), f.root.join("sealed-test-moved-away.csv")).unwrap();
    let denied = admitted(&f, &run);
    assert_eq!(denied, base);
    receipt(
        "sealed-split-denied-read",
        "forbidden",
        "sealed test.csv removed after preparation; any open by admission fails",
        &base,
        &denied,
    );
    // The shared single-family fixture is always audited into `meta_test`:
    // the teacher refuses its preparation outright.
    let sealed = Fixture::new(Spec::default());
    let sealed_run = sealed.run("sealed-family");
    let p = sealed.prepared(&sealed_run.prepared);
    assert_eq!(p.meta_partition, "meta_test");
    let row = context(&sealed, &sealed_run, 0);
    let error = label_admission::verify_inputs(&sealed.root, &row, &p).unwrap_err();
    assert!(
        format!("{error:#}").contains("admission sealed"),
        "{error:#}"
    );
}

/// Fold purity at the teacher: cut 0's held-out rows never reach cut 0's
/// admitted training identity; they are training rows of every other cut,
/// whose identity must change.
#[test]
fn a_cuts_held_out_rows_never_reach_its_admitted_training_identity() {
    let mut f = fixture();
    let base_run = f.run("baseline");
    let base = admitted(&f, &base_run);
    let (held, _) = memberships(&f, &base_run, 0);
    assert!(held.len() >= 2);
    for &i in &held {
        f.spec.source[i].y += 17.5;
    }
    write(&f);
    let run = f.run("cut0-heldout-labels-changed");
    assert_eq!(memberships(&f, &run, 0).0, held);
    let mutated = admitted(&f, &run);
    assert_eq!(
        mutated[0], base[0],
        "cut 0's held-out labels reached its admitted training identity"
    );
    for k in 1..base.len() {
        assert_ne!(
            mutated[k], base[k],
            "authorized: cut {k} trains on those rows"
        );
    }
    receipt(
        "cut0-heldout-labels-changed",
        "forbidden_for_cut0_authorized_elsewhere",
        "labels of cut 0's held-out rows changed",
        &base,
        &mutated,
    );
}

/// Authorized control: a training label of cut 0 changes cut 0's admitted
/// identity. A teacher label naming the earlier preparation bytes is refused.
#[test]
fn a_training_label_changes_the_admitted_identity_and_stale_bytes_are_refused() {
    let mut f = fixture();
    let base_run = f.run("baseline");
    let base = admitted(&f, &base_run);
    let fitted = memberships(&f, &base_run, 0).1[0];
    f.spec.source[fitted].y += 9.0;
    write(&f);
    let run = f.run("cut0-training-label-changed");
    let mutated = admitted(&f, &run);
    assert_ne!(
        mutated[0], base[0],
        "a training label did not reach the teacher"
    );
    receipt(
        "cut0-training-label-changed",
        "authorized_control",
        "one label of cut 0's training membership changed",
        &base,
        &mutated,
    );
    let mut stale = context(&f, &run, 0);
    stale.prepared_sha256 = file_hash(&base_run.prepared.join("prepared.json")).unwrap();
    let error =
        label_admission::verify_inputs(&f.root, &stale, &f.prepared(&run.prepared)).unwrap_err();
    assert!(
        format!("{error:#}").contains("admission prepared artifact changed"),
        "{error:#}"
    );
}
