use super::prepare::materialize;
use super::*;

fn recipe(method: &str, loss: Option<f64>) -> ConfirmedRecipe {
    ConfirmedRecipe {
        method: method.into(),
        diagnostic: false,
        completed_folds: if loss.is_some() { 5 } else { 0 },
        mean_loss: loss,
        fold_losses: [loss; 5],
    }
}
#[test]
fn fitted_winner_uses_losses_and_identity_never_backend_support() {
    let recipes = vec![
        recipe("external.catboost.default", Some(0.2)),
        recipe("rust.ridge", Some(0.3)),
        recipe("asm.neural", None),
    ];
    assert_eq!(
        winner(&recipes).unwrap().unwrap().method,
        "external.catboost.default"
    );
    assert_eq!(
        winner(&[recipe("z", Some(1.0)), recipe("a", Some(1.0))])
            .unwrap()
            .unwrap()
            .method,
        "a"
    );
    assert!(winner(&[recipe("unavailable", None)]).unwrap().is_none());
}
#[test]
fn fitted_winner_refuses_incomplete_or_nonfinite_means() {
    let mut bad = recipe("x", Some(0.0));
    bad.completed_folds = 4;
    assert!(winner(&[bad]).is_err());
    assert!(winner(&[recipe("x", Some(f64::NAN))]).is_err());
}

use super::super::inference::tests::fixture;
use super::super::methods::{digest, FittedRustArtifact};
use jope_datasets::catalog::Task;
use jope_pipeline::experiment::{Column, ColumnKind};

fn head_state(value: &mut Value) -> Option<&mut serde_json::Map<String, Value>> {
    match value {
        Value::Object(map) => {
            if map.contains_key("means") && map.contains_key("w") {
                return Some(map);
            }
            map.values_mut().find_map(head_state)
        }
        Value::Array(items) => items.iter_mut().find_map(head_state),
        _ => None,
    }
}
/// Two fitted states of one registered recipe. B differs from A only in the learned
/// weight of the numeric column, and every row sets that column to its fitted mean,
/// so both states predict bit-identical values on these convenient rows.
fn same_recipe_pair() -> (
    FittedRustArtifact,
    FittedRustArtifact,
    Vec<Vec<Option<String>>>,
) {
    let (a, _) = fixture(Task::Regression, "ridge");
    let mut value = serde_json::to_value(&a).unwrap();
    let head = head_state(&mut value).expect("ridge head state");
    let mean = head["means"][0].as_f64().unwrap();
    let weights = head.get_mut("w").unwrap().as_array_mut().unwrap();
    let weight = weights[1].as_f64().unwrap();
    weights[1] = json!(weight * 2.0 + 1.0);
    let b: FittedRustArtifact = serde_json::from_value(value).unwrap();
    b.validate().unwrap();
    assert_eq!(a.method, b.method, "one recipe");
    let amount = format!("{mean:?}");
    let rows = vec![
        vec![Some(amount.clone()), Some("NA".into())],
        vec![Some(amount.clone()), None],
        vec![Some(amount), Some("unseen".into())],
    ];
    (a, b, rows)
}
fn raw_columns() -> Vec<Column> {
    vec![
        Column {
            name: "amount".into(),
            kind: ColumnKind::Numeric,
        },
        Column {
            name: "country".into(),
            kind: ColumnKind::Categorical,
        },
    ]
}
/// The authenticated original worker output that `verify` reads from output.json.
fn worker_output(artifact: &FittedRustArtifact, rows: &[Vec<Option<String>>]) -> (Value, String) {
    let sha = digest(&artifact.bytes().unwrap());
    let predictions = artifact.predict_raw(&raw_columns(), rows).unwrap();
    (
        json!({"schema":"jope.panel-rust-worker-output/v1","input_sha256":"0".repeat(64),
                "artifact_sha256":sha,"predictions":predictions,"artifact":artifact}),
        sha,
    )
}
/// Unvalidated write, so a tampered entry reaches `verify_entries` itself.
fn write_public(public: &Path, entries: &[Entry]) {
    for entry in entries {
        if let Entry::Prepared {
            backend, bundle, ..
        } = entry
        {
            let directory = public.join(name(*backend));
            fs::create_dir_all(&directory).unwrap();
            for (file, text) in &bundle.files {
                fs::write(directory.join(file), text).unwrap();
            }
            fs::write(
                directory.join("manifest.json"),
                serde_json::to_vec(&bundle.manifest).unwrap(),
            )
            .unwrap();
        }
    }
}
/// Re-seal a prepared entry around substituted public files, keeping checksums,
/// bundle identity and executed plan files internally consistent. The only
/// remaining inconsistency is with the exact fitted state.
fn reseal(entry: &Entry, files: BTreeMap<String, String>) -> Entry {
    let Entry::Prepared {
        backend,
        bundle,
        qualification,
    } = entry
    else {
        panic!("prepared entry")
    };
    let mut bundle = bundle.clone();
    let mut qualification = qualification.clone();
    for name in bundle.files.keys() {
        qualification.files.remove(name);
    }
    qualification.files.extend(files.clone());
    bundle.manifest.files = files
        .iter()
        .map(|(name, text)| (name.clone(), digest(text.as_bytes())))
        .collect();
    bundle.files = files;
    qualification.bundle_sha256 = bundle.identity().unwrap();
    Entry::Prepared {
        backend: *backend,
        bundle,
        qualification,
    }
}
fn prepared(entry: &Entry) -> (&PublicBundle, &QualificationPlan) {
    let Entry::Prepared {
        bundle,
        qualification,
        ..
    } = entry
    else {
        panic!("prepared entry")
    };
    (bundle, qualification)
}

#[test]
fn same_recipe_state_swaps_and_missing_stages_fail_even_with_equal_predictions() {
    let (a, b, rows) = same_recipe_pair();
    let (a_bytes, b_bytes) = (a.bytes().unwrap(), b.bytes().unwrap());
    let entries_a = prepare_entries(&FittedInferenceView::registered(&a).unwrap(), &rows).unwrap();
    let entries_b = prepare_entries(&FittedInferenceView::registered(&b).unwrap(), &rows).unwrap();
    let (output_a, sha_a) = worker_output(&a, &rows);
    let (output_b, sha_b) = worker_output(&b, &rows);
    // Convenient rows: identical native predictions, distinct fitted states and bundles.
    assert_eq!(output_a["predictions"], output_b["predictions"]);
    assert_ne!(sha_a, sha_b);
    let mean: f64 = rows[0][0].as_ref().unwrap().parse().unwrap();
    let off = vec![vec![Some(format!("{:?}", mean + 2.0)), None]];
    let (far, _) = worker_output(&a, &off);
    let (far_b, _) = worker_output(&b, &off);
    assert_ne!(
        far["predictions"], far_b["predictions"],
        "states differ elsewhere"
    );
    for (ea, eb) in entries_a.iter().zip(&entries_b) {
        let ((bundle_a, plan_a), (bundle_b, plan_b)) = (prepared(ea), prepared(eb));
        assert_ne!(bundle_a.identity().unwrap(), bundle_b.identity().unwrap());
        assert_eq!(
            serde_json::to_value(&plan_a.expected).unwrap(),
            serde_json::to_value(&plan_b.expected).unwrap()
        );
        assert_ne!(plan_a.fitted_sha256, plan_b.fitted_sha256);
    }

    // Parity alone cannot tell the states apart: B's emitted Python passes A's plan.
    let scratch = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/fitted-inference-tests");
    fs::create_dir_all(&scratch).unwrap();
    let root = tempfile::tempdir_in(scratch).unwrap();
    let (_, plan_a) = prepared(&entries_a[0]);
    let (bundle_b, _) = prepared(&entries_b[0]);
    let run = root.path().join("parity-blind");
    fs::create_dir(&run).unwrap();
    for (file, text) in plan_a.files.iter().chain(&bundle_b.files) {
        fs::write(run.join(file), text).unwrap();
    }
    let output = std::process::Command::new("python3")
        .arg("qualification.py")
        .current_dir(&run)
        .output()
        .unwrap();
    assert!(output.status.success());
    inference::verify_qualification(plan_a, &plan_a.identity().unwrap(), &output.stdout)
        .expect("equal predictions pass parity");

    // Positive control through the production materializer.
    let public = root.path().join("intact").join("public");
    materialize(&public, &entries_a).unwrap();
    verify_entries(&entries_a, &output_a, &sha_a, &public).unwrap();

    let python = 0;
    let rust = 1;
    let cpp17 = 2;
    let (bundle_a, _) = prepared(&entries_a[python]);
    let mut attempts: Vec<(&str, Vec<Entry>)> = Vec::new();
    // Same-recipe state swap: B's public code, every identity rebound except the fitted one.
    let mut swap = entries_a.clone();
    swap[python] = reseal(
        &entries_a[python],
        prepared(&entries_b[python]).0.files.clone(),
    );
    attempts.push(("state-swap", swap));
    // The executed plan runs B's code while the shipped bundle stays A's.
    let mut executed = entries_a.clone();
    if let Entry::Prepared { qualification, .. } = &mut executed[python] {
        qualification.files.insert(
            "model.py".into(),
            prepared(&entries_b[python]).0.files["model.py"].clone(),
        );
    }
    attempts.push(("executed-file-swap", executed));
    // Whole receipt swap: B's complete entry carries B's fitted checksum.
    let mut receipt = entries_a.clone();
    receipt[python] = entries_b[python].clone();
    attempts.push(("receipt-swap", receipt));
    // Missing qualification stage: the harness file is absent from the plan.
    let mut stage = entries_a.clone();
    if let Entry::Prepared { qualification, .. } = &mut stage[python] {
        qualification.files.remove("qualification.py");
    }
    attempts.push(("missing-harness-stage", stage));
    // Missing wrapper: the C++ typed-table host boundary is dropped and resealed.
    let mut wrapper = entries_a.clone();
    let mut files = prepared(&entries_a[cpp17]).0.files.clone();
    files.remove("boundary.py");
    wrapper[cpp17] = reseal(&entries_a[cpp17], files);
    attempts.push(("missing-wrapper", wrapper));
    // Corrupted fitted constant in the shipped source, resealed.
    let mut constant = entries_a.clone();
    let mut files = bundle_a.files.clone();
    let source = files.get_mut("model.py").unwrap();
    let start = source.find("\\\"weights\\\":[").unwrap() + "\\\"weights\\\":[".len();
    let end = start + source[start..].find(',').unwrap();
    let weight: f64 = source[start..end].parse().unwrap();
    source.replace_range(start..end, &format!("{:?}", weight + 1.0));
    constant[python] = reseal(&entries_a[python], files);
    attempts.push(("corrupted-constant", constant));
    // Omitted operation: the missing-indicator encoding line is removed from Rust source.
    let mut omitted = entries_a.clone();
    let mut files = prepared(&entries_a[rust]).0.files.clone();
    let source = files.get_mut("model.rs").unwrap();
    let line = "x.push(if absent { 1.0 } else { 0.0 });\n";
    assert!(source.contains(line));
    *source = source.replacen(line, "", 1);
    omitted[rust] = reseal(&entries_a[rust], files);
    attempts.push(("omitted-operation", omitted));
    // Hidden support: an emitted backend is recorded as unsupported.
    let mut hidden = entries_a.clone();
    hidden[python] = Entry::Unsupported {
        backend: Backend::Python,
        reason: json!({"kind":"cpp_matrix"}),
    };
    attempts.push(("hidden-support", hidden));
    // Poisoned native reference.
    let mut poisoned = entries_a.clone();
    if let Entry::Prepared { qualification, .. } = &mut poisoned[python] {
        qualification.expected = serde_json::from_value(far_b["predictions"].clone()).unwrap();
    }
    attempts.push(("poisoned-reference", poisoned));

    let mut accepted = Vec::new();
    for (label, entries) in &attempts {
        let public = root.path().join(label).join("public");
        write_public(&public, entries);
        if verify_entries(entries, &output_a, &sha_a, &public).is_ok() {
            accepted.push(*label);
        }
    }
    // No authority-bearing input changed: both fitted subjects and A's receipt are intact.
    assert_eq!(a.bytes().unwrap(), a_bytes);
    assert_eq!(b.bytes().unwrap(), b_bytes);
    assert_eq!(worker_output(&a, &rows).0, output_a);
    verify_entries(&entries_a, &output_a, &sha_a, &public).unwrap();
    assert!(
        accepted.is_empty(),
        "accepted tampered entries: {accepted:?}"
    );
}
