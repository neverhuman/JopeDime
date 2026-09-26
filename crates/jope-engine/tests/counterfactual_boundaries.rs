//! W1-T06: counterfactual receiving-path proofs for the legacy registry path
//! (historical splits → `audit::run` → `prepare::run` → pointer views).
//!
//! Each proof freezes every permitted input, the source code, the random
//! streams (fixed `SEED`; folds and caps depend only on row positions and
//! feature identities) and the execution settings, then perturbs one thing.
//!
//! * **Forbidden-only** perturbations touch inputs a stage may not depend on:
//!   the sealed final-test split, one cut's held-out labels/features (for that
//!   cut's fitting state) and backing rows outside every membership. The
//!   restricted outputs must be byte-identical; the only permitted differences
//!   are identifiers that name the changed input (listed per receipt).
//! * **Authorized** perturbations touch an input the output is supposed to
//!   depend on and must change it — a proof that ignores both is no proof.
//! * **Refusals** (duplicate joins, symlink escapes, stale caches, denied
//!   reads) must stop at the receiver without publishing a usable view.
//!
//! Tolerance: zero. Every compared artifact is a SHA-256 of bytes produced by
//! deterministic single-threaded code from byte-identical permitted inputs, so
//! equality is exact byte equality and no floating comparison is used.
//!
//! One receipt per mutation is written to `$JOPE_W1_RECEIPTS/counterfactual/`
//! when that variable names a directory; the assertions run regardless.

#[path = "support/legacy_dataset.rs"]
mod legacy;

use jope_parallel::{manifest::hash, prepare::csv_bytes};
use legacy::{Fixture, Run, Spec};
use serde_json::{json, Value};
use std::{collections::BTreeMap, fs, path::Path};

const CUTS: usize = 5;
const TOLERANCE: &str = "exact byte equality of SHA-256 digests (zero tolerance): deterministic \
single-threaded producers, fixed seed, byte-identical permitted inputs";

type Snapshot = BTreeMap<String, String>;

fn sha(bytes: &[u8]) -> String {
    hash(bytes)
}

fn file(path: &Path) -> String {
    sha(&fs::read(path).unwrap())
}

fn without(value: &Value, pointers: &[&str]) -> String {
    let mut value = value.clone();
    for pointer in pointers {
        let (parent, key) = pointer.rsplit_once('/').unwrap();
        if let Some(object) = value.pointer_mut(parent).and_then(Value::as_object_mut) {
            object.remove(key);
        }
    }
    sha(&serde_json::to_vec(&value).unwrap())
}

/// Digest every restricted output of one completed run, including the views
/// the receiver returns for each cut.
fn snapshot(f: &Fixture, run: &Run) -> Snapshot {
    assert!(run.audit_error.is_none(), "audit: {:?}", run.audit_error);
    assert!(
        run.prepare_error.is_none(),
        "prepare: {:?}",
        run.prepare_error
    );
    let mut s = Snapshot::new();
    s.insert(
        "audit/audit.json".into(),
        file(&run.audit.join("audit.json")),
    );
    s.insert(
        "audit/transfer.json".into(),
        file(&run.audit.join("transfer.json")),
    );
    let prepared: Value =
        serde_json::from_slice(&fs::read(run.prepared.join("prepared.json")).unwrap()).unwrap();
    s.insert(
        "prepared/prepared.json".into(),
        file(&run.prepared.join("prepared.json")),
    );
    s.insert(
        "prepared/prepared.json-without-input-identifiers".into(),
        without(
            &prepared,
            &[
                "/original_test_sha256",
                "/storage/source_sha256",
                "/original_train_sha256",
            ],
        ),
    );
    let reconstruction: Value =
        serde_json::from_slice(&fs::read(run.prepared.join("reconstruction.json")).unwrap())
            .unwrap();
    s.insert(
        "prepared/reconstruction.json".into(),
        file(&run.prepared.join("reconstruction.json")),
    );
    s.insert(
        "prepared/reconstruction.json-without-source-identity".into(),
        without(&reconstruction, &["/receipt/source_sha256"]),
    );
    let cuts = prepared["cuts"].as_array().unwrap();
    assert_eq!(cuts.len(), CUTS);
    for (k, cut) in cuts.iter().enumerate() {
        s.insert(
            format!("cut-{k}/fitting-entry"),
            sha(&serde_json::to_vec(&json!([
                cut["train_indices"],
                cut["validation_indices"],
                cut["training_sha256"],
                cut["preprocessing_sha256"]
            ]))
            .unwrap()),
        );
        s.insert(
            format!("cut-{k}/preprocessing.json"),
            file(&run.prepared.join(format!("cut-{k}/preprocessing.json"))),
        );
        for (name, training) in [("training-view", true), ("validation-view", false)] {
            let rows = f.view(&run.prepared, k, training).unwrap();
            s.insert(format!("cut-{k}/{name}"), sha(&csv_bytes(&rows).unwrap()));
        }
    }
    s
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Expect {
    Equal,
    Differs,
    /// Recorded, not asserted: the stage legitimately reads the changed input.
    Declared,
}

struct Proof<'a> {
    mutation: &'a str,
    class: &'a str,
    perturbed: &'a str,
    frozen: &'a str,
    notes: Vec<String>,
    comparisons: Vec<Value>,
    refusals: Vec<Value>,
}

impl<'a> Proof<'a> {
    fn new(mutation: &'a str, class: &'a str, perturbed: &'a str) -> Self {
        Self {
            mutation,
            class,
            perturbed,
            frozen: "engine source; fixture generator; SEED; folds/caps (row positions and \
                feature identities only); every input byte except the perturbation; \
                single-threaded execution",
            notes: vec![],
            comparisons: vec![],
            refusals: vec![],
        }
    }

    fn compare(
        &mut self,
        baseline: &Snapshot,
        mutated: &Snapshot,
        expect: impl Fn(&str) -> Expect,
    ) {
        for (artifact, before) in baseline {
            let after = &mutated[artifact];
            let expected = expect(artifact);
            let observed = if before == after { "equal" } else { "differs" };
            match expected {
                Expect::Equal => assert_eq!(
                    before, after,
                    "{}: restricted output {artifact} changed",
                    self.mutation
                ),
                Expect::Differs => assert_ne!(
                    before, after,
                    "{}: {artifact} ignored an input it must depend on",
                    self.mutation
                ),
                Expect::Declared => {}
            }
            self.comparisons.push(json!({"artifact":artifact,
                "expected":format!("{expected:?}").to_lowercase(),"observed":observed,
                "baseline_sha256":before,"mutated_sha256":after}));
        }
    }

    fn refused(&mut self, what: &str, result: anyhow::Result<impl std::fmt::Debug>, needle: &str) {
        let error = match result {
            Ok(value) => panic!("{}: {what} was accepted: {value:?}", self.mutation),
            Err(error) => format!("{error:#}"),
        };
        assert!(
            error.contains(needle),
            "{}: {what}: expected {needle:?}, got {error}",
            self.mutation
        );
        self.refusals
            .push(json!({"receiver":what,"refused":true,"reason":error}));
    }

    fn note(&mut self, note: impl Into<String>) {
        self.notes.push(note.into());
    }

    fn finish(self) {
        let receipt = json!({"schema":"jope.w1-counterfactual-receipt/v1","task":"W1-T06",
            "mutation":self.mutation,"class":self.class,"perturbed":self.perturbed,
            "frozen":self.frozen,"tolerance":TOLERANCE,"comparisons":self.comparisons,
            "refusals":self.refusals,"notes":self.notes,
            "receiver":"jope_parallel::{audit::run, prepare::run, prepare::load_view_in_root}"});
        if let Some(dir) = std::env::var_os("JOPE_W1_RECEIPTS") {
            let dir = Path::new(&dir).join("counterfactual");
            fs::create_dir_all(&dir).unwrap();
            fs::write(
                dir.join(format!("{}.json", self.mutation)),
                serde_json::to_vec_pretty(&receipt).unwrap(),
            )
            .unwrap();
        }
    }
}

/// Restricted outputs of one cut's fitting state.
fn fitting_state(artifact: &str, cut: usize) -> bool {
    [
        format!("cut-{cut}/fitting-entry"),
        format!("cut-{cut}/preprocessing.json"),
        format!("cut-{cut}/training-view"),
    ]
    .iter()
    .any(|a| a == artifact)
}

fn baseline(spec: Spec) -> (Fixture, Run, Snapshot) {
    let f = Fixture::new(spec);
    let run = f.run("baseline");
    let snap = snapshot(&f, &run);
    (f, run, snap)
}

/// Source indices of cut `cut`'s held-out rows.
fn held_out_sources(f: &Fixture, run: &Run, cut: usize) -> Vec<usize> {
    let p = f.prepared(&run.prepared);
    let map = &p.storage.as_ref().unwrap().original_training_row_map;
    p.cuts[cut]
        .validation_indices
        .iter()
        .map(|&i| map[i])
        .collect()
}

fn training_sources(f: &Fixture, run: &Run, cut: usize) -> Vec<usize> {
    let p = f.prepared(&run.prepared);
    let map = &p.storage.as_ref().unwrap().original_training_row_map;
    p.cuts[cut].train_indices.iter().map(|&i| map[i]).collect()
}

/// Sealed final-test labels, rows and categories are forbidden to every
/// upstream stage. Only the identifiers of the sealed file may change.
#[test]
fn sealed_split_perturbations_leave_every_upstream_output_unchanged() {
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
            "sealed final-test labels permuted across rows (target shuffle)",
            |s| {
                let labels: Vec<f64> = s.test.iter().map(|r| r.y).collect();
                for (i, row) in s.test.iter_mut().enumerate() {
                    row.y = labels[(i + 3) % labels.len()];
                }
            },
        ),
        (
            "sealed-rows-reordered",
            "sealed final-test rows reversed (row reordering)",
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
        let (mut f, _, base) = baseline(Spec::default());
        mutate(&mut f.spec);
        f.write();
        let run = f.run(name);
        let mutated = snapshot(&f, &run);
        let mut proof = Proof::new(name, "forbidden_only", perturbed);
        proof.compare(&base, &mutated, |artifact| match artifact {
            // Identifiers of the changed sealed file itself.
            "audit/transfer.json" | "prepared/prepared.json" => Expect::Differs,
            _ => Expect::Equal,
        });
        proof.note(
            "permitted differences: transfer.json (content hash of test.csv) and \
             prepared.json original_test_sha256; prepared.json without it is byte-identical",
        );
        proof.finish();
    }
}

/// Denied read: with the sealed split absent, preparation and every view
/// reproduce the baseline bytes — they never open it.
#[test]
fn preparation_and_views_never_open_the_sealed_split() {
    let (f, run, base) = baseline(Spec::default());
    fs::rename(f.test_csv(), f.root.join("sealed-test-moved-away.csv")).unwrap();
    let prepared = f.prepare(&run.audit, "sealed-split-absent").unwrap();
    let mutated = snapshot(
        &f,
        &Run {
            audit: run.audit.clone(),
            prepared,
            audit_error: None,
            prepare_error: None,
        },
    );
    let mut proof = Proof::new(
        "sealed-split-denied-read",
        "forbidden_only",
        "sealed test.csv removed after the audit; any open by preparation or a view fails",
    );
    proof.compare(&base, &mutated, |_| Expect::Equal);
    proof.note("same audit directory reused; every prepared byte and view is identical");
    proof.finish();
}

/// Fold purity: one cut's held-out labels, label order and categories are
/// forbidden to that cut's fitting state; the same rows are authorized
/// fitting inputs for every other cut, whose state must change.
#[test]
fn held_out_rows_of_a_cut_never_reach_its_fitting_state() {
    type Mutate = fn(&mut Spec, &[usize]);
    let cases: [(&str, &str, Mutate); 3] = [
        (
            "cut0-heldout-labels-changed",
            "labels of cut 0's held-out rows changed in the backing source and historical table",
            |s, held| {
                for &i in held {
                    s.source[i].y += 17.5;
                }
            },
        ),
        (
            "cut0-heldout-labels-shuffled",
            "labels permuted among cut 0's held-out rows only (target shuffle)",
            |s, held| {
                let labels: Vec<f64> = held.iter().map(|&i| s.source[i].y).collect();
                for (k, &i) in held.iter().enumerate() {
                    s.source[i].y = labels[(k + 1) % labels.len()];
                }
            },
        ),
        (
            "cut0-heldout-categories-changed",
            "categories of cut 0's held-out rows rotated to other recorded categories",
            |s, held| {
                for &i in held {
                    s.source[i].c = match s.source[i].c.as_str() {
                        "red" => "green",
                        "green" => "blue",
                        _ => "red",
                    }
                    .into();
                }
            },
        ),
    ];
    for (name, perturbed, mutate) in cases {
        let (mut f, run, base) = baseline(Spec::default());
        let held = held_out_sources(&f, &run, 0);
        assert!(held.len() >= 2);
        mutate(&mut f.spec, &held);
        f.write();
        let run = f.run(name);
        let mutated = snapshot(&f, &run);
        // Fold assignment is label- and category-independent: same memberships.
        assert_eq!(held_out_sources(&f, &run, 0), held);
        let mut proof = Proof::new(name, "forbidden_only_for_cut0_fitting", perturbed);
        proof.compare(&base, &mutated, |artifact| {
            if fitting_state(artifact, 0) {
                Expect::Equal
            } else if artifact == "cut-0/validation-view"
                || (1..CUTS).any(|k| artifact == format!("cut-{k}/training-view"))
            {
                // Authorized: cut 0's scoring rows, and fitting rows elsewhere.
                Expect::Differs
            } else {
                Expect::Declared
            }
        });
        proof.note(
            "cut 0's fitting entry, preprocessing and training view are byte-identical; its \
             validation view (the cut's selection labels) and every other cut's training \
             view change: those consumers are authorized to read the perturbed rows",
        );
        proof.note(
            "declared, not asserted: audit.json and prepared.json read development labels \
             of the historical training table (adaptive development evidence)",
        );
        proof.finish();
    }
}

/// Authorized positive control: a training label of cut 0 must change cut 0's
/// fitting state. Reusing the stale cached state for it is then refused.
#[test]
fn a_training_label_changes_its_fitting_state_and_stale_state_is_refused() {
    let (mut f, run, base) = baseline(Spec::default());
    let fitted = training_sources(&f, &run, 0)[0];
    f.spec.source[fitted].y += 9.0;
    f.write();
    let mutated_run = f.run("cut0-training-label-changed");
    let mutated = snapshot(&f, &mutated_run);
    let mut proof = Proof::new(
        "cut0-training-label-changed",
        "authorized",
        "one label of cut 0's training membership changed",
    );
    proof.compare(&base, &mutated, |artifact| {
        if fitting_state(artifact, 0) {
            Expect::Differs
        } else {
            Expect::Declared
        }
    });
    proof.finish();

    // Stale cache: the mutated preparation keeps the baseline cut-0 state and
    // hashes, as a cache keyed on anything but the fitting rows would.
    let stale = &mutated_run.prepared;
    fs::copy(
        run.prepared.join("cut-0/preprocessing.json"),
        stale.join("cut-0/preprocessing.json"),
    )
    .unwrap();
    let mut p: Value =
        serde_json::from_slice(&fs::read(stale.join("prepared.json")).unwrap()).unwrap();
    let old: Value =
        serde_json::from_slice(&fs::read(run.prepared.join("prepared.json")).unwrap()).unwrap();
    for key in ["preprocessing_sha256", "training_sha256"] {
        p["cuts"][0][key] = old["cuts"][0][key].clone();
    }
    fs::write(stale.join("prepared.json"), serde_json::to_vec(&p).unwrap()).unwrap();
    let mut proof = Proof::new(
        "stale-cached-preparation-state",
        "refusal",
        "cut 0 learned state and hashes reused from before its training label changed",
    );
    proof.refused(
        "load_view_in_root(training)",
        f.view(stale, 0, true),
        "not the state learned from its training membership",
    );
    proof.refused(
        "load_view_in_root(validation)",
        f.view(stale, 0, false),
        "not the state learned from its training membership",
    );
    proof.finish();
}

/// Backing rows outside every membership are forbidden parents: changing them
/// changes only the backing identity. A preparation made before the change is
/// stale and refused, never silently re-read.
#[test]
fn backing_rows_outside_every_membership_are_forbidden_parents() {
    let (mut f, run, base) = baseline(Spec::default());
    let outside = 35;
    assert!(f.train_position(outside).is_none());
    f.spec.source[outside].y = 999.0;
    f.spec.source[outside].c = "green".into();
    f.write();
    let mut proof = Proof::new(
        "unreferenced-backing-row-changed",
        "forbidden_only",
        "label and category of a backing row that no membership references",
    );
    proof.refused(
        "load_view_in_root on the preparation made before the change",
        f.view(&run.prepared, 0, true),
        "backing table changed",
    );
    let mutated_run = f.run("unreferenced-backing-row-changed");
    let mutated = snapshot(&f, &mutated_run);
    proof.compare(&base, &mutated, |artifact| match artifact {
        "prepared/prepared.json" | "prepared/reconstruction.json" => Expect::Differs,
        _ => Expect::Equal,
    });
    proof.note(
        "permitted differences: storage.source_sha256 and the reconstruction receipt's \
         source_sha256 name the changed backing bytes; the backing reader digests and \
         tokenizes the whole file but selects only mapped rows",
    );
    proof.finish();
}

/// Row identity: reordering the historical training table is a new
/// preparation identity, never a silent reinterpretation of the old one.
#[test]
fn reordered_historical_rows_are_a_new_preparation_identity() {
    let (mut f, run, base) = baseline(Spec::default());
    f.spec.train.reverse();
    f.write();
    let mutated_run = f.run("historical-training-rows-reordered");
    let mutated = snapshot(&f, &mutated_run);
    let mut proof = Proof::new(
        "historical-training-rows-reordered",
        "new_identity",
        "historical training table rows reversed",
    );
    proof.compare(&base, &mutated, |artifact| match artifact {
        "prepared/prepared.json" => Expect::Differs,
        _ => Expect::Declared,
    });
    let old = f.prepared(&run.prepared);
    let new = f.prepared(&mutated_run.prepared);
    assert_ne!(old.original_train_sha256, new.original_train_sha256);
    for cut in 0..CUTS {
        f.view(&run.prepared, cut, true).unwrap();
        f.view(&mutated_run.prepared, cut, true).unwrap();
    }
    proof.note("original_train_sha256 differs; both preparations stay separately loadable");
    proof.note(
        "audit.json is recorded, not asserted: its training-table summary (counts and \
         value sets) is order-independent, so it is legitimately equal. The first run of \
         this proof (w1b-t06-prefix) wrongly expected it to differ; that expectation, not \
         the audit, was corrected",
    );
    proof.finish();
}

/// Group aliases: feature-identical rows (one entity under two row ids) stay
/// on one side of every cut, so neither can score the other.
#[test]
fn feature_identical_rows_never_straddle_a_cut() {
    let mut spec = Spec::default();
    let (a, b) = (3, 17);
    spec.source[b] = legacy::Row {
        y: spec.source[b].y,
        ..spec.source[a].clone()
    };
    let f = Fixture::new(spec);
    let run = f.run("feature-alias");
    assert!(run.prepare_error.is_none(), "{:?}", run.prepare_error);
    let mut proof = Proof::new(
        "group-alias-feature-identical-rows",
        "grouping",
        "two training rows made feature-identical with different labels",
    );
    for cut in 0..CUTS {
        let held = held_out_sources(&f, &run, cut);
        assert_eq!(
            held.contains(&a),
            held.contains(&b),
            "cut {cut} splits an entity alias across fitting and scoring"
        );
        proof.note(format!(
            "cut {cut}: both rows {}",
            if held.contains(&a) {
                "held out"
            } else {
                "in training"
            }
        ));
    }
    proof.finish();
}

/// Duplicate-expanding join: one backing row listed twice in the historical
/// training table must not enter a fitting membership twice.
#[test]
fn a_duplicate_join_onto_one_backing_row_is_refused() {
    let mut spec = Spec::default();
    spec.train.push(5);
    let f = Fixture::new(spec);
    let run = f.run("duplicate-join");
    let mut proof = Proof::new(
        "duplicate-join",
        "refusal",
        "historical training table repeats backing row 5",
    );
    match &run.prepare_error {
        Some(error) => {
            assert!(error.contains("duplicate"), "{error}");
            proof
                .refusals
                .push(json!({"receiver":"prepare::run","refused":true,"reason":error}));
            assert!(!run.prepared.exists(), "a refused preparation left output");
        }
        None => panic!("a duplicate-expanding join was prepared"),
    }
    proof.finish();
}

/// Genuine duplicates in the backing source are distinct rows and remain
/// admissible: each historical copy maps to its own backing row.
#[test]
fn identical_backing_rows_map_to_distinct_pointers() {
    let mut spec = Spec::default();
    spec.source[20] = spec.source[5].clone();
    let f = Fixture::new(spec);
    let run = f.run("backing-duplicates");
    assert!(run.prepare_error.is_none(), "{:?}", run.prepare_error);
    let p = f.prepared(&run.prepared);
    let map = &p.storage.as_ref().unwrap().original_training_row_map;
    let positions = [5, 20].map(|s| f.train_position(s).unwrap());
    assert_ne!(map[positions[0]], map[positions[1]]);
    for cut in 0..CUTS {
        f.view(&run.prepared, cut, true).unwrap();
    }
}

/// Symlink escapes: a backing source or a historical split reached through a
/// symlink is refused by its receiver.
#[test]
fn symlinked_inputs_are_refused_at_their_receivers() {
    let outside = tempfile::tempdir().unwrap();
    // Backing source escape.
    let (f, run, _) = baseline(Spec::default());
    let copy = outside.path().join("raw.csv");
    fs::copy(f.backing(), &copy).unwrap();
    fs::remove_file(f.backing()).unwrap();
    std::os::unix::fs::symlink(&copy, f.backing()).unwrap();
    let mut proof = Proof::new(
        "symlink-escapes",
        "refusal",
        "backing source and historical splits replaced by symlinks to identical bytes outside the root",
    );
    proof.refused(
        "load_view_in_root(backing symlink)",
        f.view(&run.prepared, 0, true),
        "symlink",
    );
    // Historical split escapes, one at a time on a fresh fixture.
    for split in ["test.csv", "train.csv"] {
        let f = Fixture::new(Spec::default());
        let target = outside.path().join(split);
        let path = f.legacy().join(split);
        fs::copy(&path, &target).unwrap();
        fs::remove_file(&path).unwrap();
        std::os::unix::fs::symlink(&target, &path).unwrap();
        let run = f.run(&format!("symlinked-{split}"));
        let error = run
            .prepare_error
            .clone()
            .unwrap_or_else(|| panic!("a symlinked {split} was audited and prepared"));
        assert!(!run.prepared.exists());
        let audit: Value =
            serde_json::from_slice(&fs::read(run.audit.join("audit.json")).unwrap()).unwrap();
        let issues = audit["datasets"][0]["issues"].to_string();
        assert!(issues.contains("symlink"), "{split}: {issues}");
        proof.refusals.push(
            json!({"receiver":format!("audit::run + prepare::run ({split})"),
            "refused":true,"audit_issues":audit["datasets"][0]["issues"],"reason":error}),
        );
    }
    proof.finish();
}
