// Extracted from inference_contract_tests.rs; test behavior is unchanged.
use super::*;

#[test]
fn emitted_backends_match_native_on_raw_edge_inputs_and_refuse_malformed_rows() {
    let root = scratch();
    let cases = raw_cases(&HOSTILE_CATEGORIES);
    let mut all = Vec::new();
    // ridge/logit: all five backends; multiclass: no scalar C++ ABI; branch heads: Python/Rust.
    for (task, family, backends) in [
        (Task::Regression, "ridge", 5),
        (Task::Binary, "logit_ridge", 5),
        (Task::Multiclass, "multiclass_ridge", 3),
        (Task::Binary, "tab_boost", 2),
        (Task::Regression, "extra_trees", 2),
    ] {
        let artifact = hostile_fixture(task, family);
        let before = artifact.bytes().unwrap();
        let (found, tallies) = divergences(
            &format!("{task:?}-{family}"),
            &artifact,
            &cases,
            root.path(),
        );
        all.extend(found);
        assert_tallies(&cases, &tallies, backends);
        // Execution through every backend is read-only for the fitted subject.
        assert_eq!(artifact.bytes().unwrap(), before);
    }
    assert!(
        all.is_empty(),
        "{} divergences:\n{}",
        all.len(),
        all.join("\n")
    );
}

/// Positive control for the comparator itself: one changed fitted constant in the
/// emitted Python source, or one dropped output row, is reported as a divergence.
#[test]
fn raw_input_comparator_detects_changed_constants_and_truncated_outputs() {
    let root = scratch();
    let artifact = hostile_fixture(Task::Regression, "ridge");
    let view = FittedInferenceView::registered(&artifact).unwrap();
    let ExportOutcome::Emitted { bundle } = export_inference(&view, Backend::Python).unwrap()
    else {
        panic!("ridge Python export")
    };
    let cases = vec![case(
        "valid",
        json!([["1", "NA"], ["2", null], [null, "unseen"]]),
        false,
    )];
    let expected = view
        .predict_raw(&columns(), cases[0].typed.as_ref().unwrap())
        .unwrap();
    let intact = execute(&root.path().join("intact"), &artifact, &bundle, &cases);
    let actual: Predictions =
        serde_json::from_value(intact[0].as_ref().unwrap()["ok"].clone()).unwrap();
    compare(ParityPolicy::ExactBits, &actual, &expected).unwrap();
    // Truncation and reordering cannot pass the cardinality/row-order comparison.
    let Predictions::Values(values) = &actual else {
        panic!("regression values")
    };
    let truncated = Predictions::Values(values[..2].to_vec());
    assert!(compare(ParityPolicy::ExactBits, &truncated, &expected).is_err());
    let mut reordered = values.clone();
    reordered.swap(0, 1);
    assert_ne!(values[0].to_bits(), values[1].to_bits());
    assert!(compare(
        ParityPolicy::ExactBits,
        &Predictions::Values(reordered),
        &expected
    )
    .is_err());
    // A changed learned weight inside the serialized state (the intercept).
    let mut tampered = bundle.clone();
    let source = tampered.files.get_mut("model.py").unwrap();
    let marker = "\\\"weights\\\":[";
    let start = source.find(marker).expect("serialized weights") + marker.len();
    let end = start + source[start..].find(',').unwrap();
    let weight: f64 = source[start..end].parse().unwrap();
    let changed = weight + 1e-3 * (1.0 + weight.abs());
    source.replace_range(start..end, &format!("{changed:?}"));
    let result = execute(&root.path().join("tampered"), &artifact, &tampered, &cases);
    let actual: Predictions =
        serde_json::from_value(result[0].as_ref().unwrap()["ok"].clone()).unwrap();
    assert!(compare(ParityPolicy::ExactBits, &actual, &expected).is_err());
}

/// W9-T06 source determinism, a separate contract from binary reproducibility:
/// re-emitting the same fitted capsule, directly or after an exact-bytes reload, yields
/// byte-identical public bundles and private plans for every exported backend.
#[test]
fn re_emission_of_an_identical_capsule_is_source_deterministic() {
    let rows = vec![
        vec![Some("1".to_owned()), Some("NA".to_owned())],
        vec![None, None],
    ];
    let mut compared = 0;
    for (task, family) in [
        (Task::Regression, "ridge"),
        (Task::Binary, "tab_boost"),
        (Task::Multiclass, "multiclass_ridge"),
    ] {
        let artifact = hostile_fixture(task, family);
        let bytes = artifact.bytes().unwrap();
        let reloaded = FittedRustArtifact::from_bytes(&bytes, &digest(&bytes)).unwrap();
        for backend in [
            Backend::Python,
            Backend::Rust,
            Backend::JavaScript,
            Backend::Cpp17,
            Backend::Cpp20,
        ] {
            let first = export_fitted(&artifact, backend).unwrap();
            let second = export_fitted(&reloaded, backend).unwrap();
            assert_eq!(
                serde_json::to_vec(&first).unwrap(),
                serde_json::to_vec(&second).unwrap()
            );
            if let ExportOutcome::Emitted { bundle } = first {
                let a = qualification_plan(&artifact, &bundle, &rows).unwrap();
                let b = qualification_plan(&reloaded, &bundle, &rows).unwrap();
                assert_eq!(a.identity().unwrap(), b.identity().unwrap());
                compared += 1;
            }
        }
    }
    assert_eq!(compared, 5 + 2 + 3);
}

/// W9-T03: fitted strings are data. Qualification argv is the fixed harness for every
/// backend; no category, class or column string ever reaches a command line.
#[test]
fn hostile_fitted_strings_never_reach_command_arguments() {
    let rows: Vec<Vec<Option<String>>> = HOSTILE_CATEGORIES
        .iter()
        .map(|value| vec![Some("1".to_owned()), Some((*value).to_owned())])
        .collect();
    for (task, family) in [(Task::Regression, "ridge"), (Task::Binary, "logit_ridge")] {
        let artifact = hostile_fixture(task, family);
        let mut needles: Vec<&str> = HOSTILE_CATEGORIES
            .iter()
            .copied()
            .filter(|value| value.len() >= 3)
            .collect();
        needles.extend(artifact.classes.iter().map(String::as_str));
        for backend in [
            Backend::Python,
            Backend::Rust,
            Backend::JavaScript,
            Backend::Cpp17,
            Backend::Cpp20,
        ] {
            let ExportOutcome::Emitted { bundle } = export_fitted(&artifact, backend).unwrap()
            else {
                panic!("{family} {backend:?}")
            };
            let plan = qualification_plan(&artifact, &bundle, &rows).unwrap();
            verify_plan_binding(&plan, &bundle).unwrap();
            for argument in plan.commands.iter().flatten() {
                for needle in &needles {
                    assert!(
                        !argument.contains(needle),
                        "{argument:?} carries {needle:?}"
                    );
                }
            }
        }
    }
}

/// K06: the pre-fit inventory is a tested registry, not narrative. Every registered
/// recipe is fitted once and its actual export outcome for each backend must equal the
/// inventory claim; an unsupported outcome leaves the fitted state byte-identical.
#[test]
fn registered_inference_inventory_matches_actual_export_outcomes() {
    let inventory = registered_inference_inventory().unwrap();
    let mut checked = 0;
    let mut mismatches = Vec::new();
    for task in [Task::Regression, Task::Binary, Task::Multiclass] {
        for method in rust_methods(task).unwrap() {
            let id = method.id.clone();
            let claim = inventory
                .iter()
                .find(|row| row["method"] == id.as_str() && row["task"] == json!(task))
                .unwrap_or_else(|| panic!("{id} absent from inventory"));
            let artifact = fit_method(task, method, &hostile_categories(), &|_| 3, false).unwrap();
            let before = artifact.bytes().unwrap();
            for (backend, key) in [
                (Backend::Python, "standalone_python"),
                (Backend::Rust, "standalone_rust"),
                (Backend::JavaScript, "javascript"),
                (Backend::Cpp17, "cpp17_cpp20"),
                (Backend::Cpp20, "cpp17_cpp20"),
            ] {
                let emitted = match export_fitted(&artifact, backend).unwrap() {
                    ExportOutcome::Emitted { .. } => true,
                    ExportOutcome::Unsupported { .. } => false,
                };
                if emitted != (claim[key]["supported"] == true) {
                    mismatches.push(format!("{id}/{backend:?}: emitted={emitted}"));
                }
            }
            assert_eq!(artifact.bytes().unwrap(), before);
            checked += 1;
        }
    }
    assert_eq!(checked, 26);
    assert!(mismatches.is_empty(), "{mismatches:?}");
}

fn deletions(value: &Value, path: &mut Vec<String>, found: &mut Vec<Vec<String>>) {
    match value {
        Value::Object(map) => {
            for (key, child) in map {
                path.push(key.clone());
                found.push(path.clone());
                deletions(child, path, found);
                path.pop();
            }
        }
        Value::Array(items) => {
            for (index, child) in items.iter().enumerate() {
                path.push(index.to_string());
                deletions(child, path, found);
                path.pop();
            }
        }
        _ => {}
    }
}
fn delete(value: &mut Value, path: &[String]) {
    let (last, parents) = path.split_last().unwrap();
    let mut at = value;
    for key in parents {
        at = match at {
            Value::Object(map) => map.get_mut(key).unwrap(),
            Value::Array(items) => &mut items[key.parse::<usize>().unwrap()],
            _ => unreachable!(),
        };
    }
    at.as_object_mut().unwrap().remove(last);
}

/// No learned state is reconstructed with a default. Deleting any single field of a
/// fitted envelope either fails decode/validation, or (only for an absent null option)
/// re-serializes to the identical bytes; the exact-bytes loader refuses every deletion.
#[test]
fn missing_learned_state_is_refused_never_defaulted() {
    for (task, family) in [
        (Task::Regression, "ridge"),
        (Task::Binary, "tab_boost"),
        (Task::Multiclass, "multiclass_ridge"),
    ] {
        let artifact = hostile_fixture(task, family);
        let bytes = artifact.bytes().unwrap();
        let sha = digest(&bytes);
        let original: Value = serde_json::from_slice(&bytes).unwrap();
        let mut paths = Vec::new();
        deletions(&original, &mut Vec::new(), &mut paths);
        assert!(paths.len() > 20);
        let (mut refused, mut identical, mut defaulted) = (0, 0, Vec::new());
        for path in &paths {
            let mut value = original.clone();
            delete(&mut value, path);
            let changed = serde_json::to_vec(&value).unwrap();
            assert!(FittedRustArtifact::from_bytes(&changed, &sha).is_err());
            match FittedRustArtifact::from_bytes(&changed, &digest(&changed)) {
                Err(_) => refused += 1,
                Ok(decoded) if decoded.bytes().unwrap() == bytes => identical += 1,
                Ok(decoded) => {
                    // A defaulted state never inherits the original fitted identity.
                    assert_ne!(
                        FittedInferenceView::registered(&decoded)
                            .unwrap()
                            .fitted_sha256(),
                        sha
                    );
                    defaulted.push(path.join("/"));
                }
            }
        }
        assert!(refused > 0);
        assert!(
            defaulted.is_empty(),
            "{task:?}-{family}: {defaulted:?} ({identical} null options)"
        );
    }
}

#[test]
fn onehot_limit_boundary_and_all_missing_training_column_keep_native_semantics() {
    let root = scratch();
    // 300 categories; the first 256 occur twice and the rest once, so the fitted
    // max_onehot=256 limit is exercised on both sides of the boundary.
    let categories: Vec<Option<String>> = (0..300).map(|i| Some(format!("c{i:03}"))).collect();
    let artifact = fit(
        Task::Regression,
        "ridge",
        &categories,
        &|position| if position < 256 { 2 } else { 1 },
        false,
    )
    .unwrap();
    let probes: Vec<String> = [
        "c000", "c254", "c255", "c256", "c257", "c299", "c300", "C000",
    ]
    .map(str::to_owned)
    .to_vec();
    let probe_refs: Vec<&str> = probes.iter().map(String::as_str).collect();
    let cases = raw_cases(&probe_refs);
    let (found, tallies) = divergences("onehot-boundary", &artifact, &cases, root.path());
    assert!(found.is_empty(), "{}", found.join("\n"));
    assert_tallies(&cases, &tallies, 5);
    // The public state carries exactly the fitted top-256 vocabulary: the first
    // excluded category is absent from emitted source, the last included one present.
    let view = FittedInferenceView::registered(&artifact).unwrap();
    let ExportOutcome::Emitted { bundle } = export_inference(&view, Backend::Rust).unwrap() else {
        panic!("ridge Rust export")
    };
    let source = &bundle.files["model.rs"];
    assert!(source.contains("c255"));
    assert!(!source.contains("c256") && !source.contains("c299"));

    // Entirely missing numeric training column: the native fitter admits it, and every
    // exported backend reproduces that fitted state (fill and indicator) exactly.
    let mut categories: Vec<Option<String>> = vec![None];
    categories.extend(["a", "b", "c"].map(|v| Some(v.to_owned())));
    let artifact = fit(Task::Regression, "ridge", &categories, &|_| 4, true).unwrap();
    let cases = raw_cases(&["a", "b"]);
    let (found, tallies) = divergences(
        "all-missing-training-column",
        &artifact,
        &cases,
        root.path(),
    );
    assert!(found.is_empty(), "{}", found.join("\n"));
    assert_tallies(&cases, &tallies, 5);
}
