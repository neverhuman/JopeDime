//! W6-T03: comparisons and runtime observations at the training input boundary.
//!
//! CPU-only tiny fixtures. The corpus tests run under default features; the
//! comparison/objective tests need the `neural` feature. None of them trains,
//! loads CUDA or reads a real corpus.

use jope_parallel::{corpus, manifest::*};
use serde_json::{json, Value};
use std::{fs, path::Path};

const TRANSFORMS: [&str; 8] = [
    "standardize",
    "robust_scale",
    "norm_row",
    "row_stats",
    "rank",
    "blom",
    "sigmoid",
    "step",
];

fn genome(n: usize) -> String {
    format!("(pipe X ({}) (ridge))", TRANSFORMS[n])
}

fn observation(family: &str, n: usize) -> Value {
    let genome = genome(n);
    json!({"dataset":family,"family":family,"cut":0,"partition":"meta_train",
        "training_sha256":"train","validation_sha256":"validation","genome":genome,
        "program_sha256":hash(genome.as_bytes()),"metrics":{"rmse":n as f64+1.},
        "runtime_seconds":0.1,"evaluation_bias":"fixture","failure":null})
}

/// Two training families with one context each; `edit` may rewrite the
/// ledger rows of `family-a` before they are written.
fn corpus_fixture(
    root: &Path,
    edit: impl Fn(&mut Vec<Value>),
) -> (std::path::PathBuf, Vec<std::path::PathBuf>) {
    let mut contexts = Vec::new();
    let mut observations = Vec::new();
    for family in ["family-a", "family-b"] {
        let prepared = root.join(family);
        write_json(&prepared.join("prepared.json"), &json!({
            "schema":"parallel.prepared.v1","dataset":family,"family":family,"task":"regression",
            "meta_partition":"meta_train","seed":17,"clean_evaluation":false,"clean_status":"fixture",
            "original_train_sha256":"original","original_test_sha256":"sealed",
            "selected_train_indices":[0,1],"selected_test_indices":[],"target_last":true,"rows_total":2,
            "cuts":[{"index":0,"train_indices":[0],"validation_indices":[1],"training_sha256":"train",
                "validation_sha256":"validation","preprocessing_sha256":"preprocessing"}]})).unwrap();
        let prepared_sha = file_hash(&prepared.join("prepared.json")).unwrap();
        let path = prepared.join("context.bin");
        fs::write(&path, b"frozen context fixture").unwrap();
        write_json(&path.with_extension("json"),&json!({"schema":"parallel.encoding.v1","dataset":family,
            "family":family,"partition":"meta_train","cut":0,"prepared_sha256":prepared_sha,
            "training_sha256":"train","selected_training_rows_sha256":"rows","preprocessing_sha256":"preprocessing",
            "encoder_spec_sha256":"assembly","numerical_backend":"fixture","dataset_dimensions":1536,
            "target_dimensions":512,"output_sha256":file_hash(&path).unwrap()})).unwrap();
        contexts.push(
            json!({"dataset":family,"family":family,"cut":0,"partition":"meta_train",
            "path":path.strip_prefix(root).unwrap(),"sha256":file_hash(&path).unwrap(),
            "prepared":family,"prepared_sha256":prepared_sha}),
        );
        let observed = root.join(format!("observations-{family}"));
        let mut rows = (0..8).map(|n| observation(family, n)).collect::<Vec<_>>();
        if family == "family-a" {
            edit(&mut rows);
        }
        write_json(&observed.join("observations.json"), &rows).unwrap();
        write_json(&observed.join("search.json"), &json!({})).unwrap();
        observations.push(observed);
    }
    let contexts_path = root.join("contexts.json");
    write_json(
        &contexts_path,
        &json!({"schema":"parallel.context_index.v1","qualification":true,
        "assembly_sha256":"assembly","encodings":contexts}),
    )
    .unwrap();
    (contexts_path, observations)
}

fn vocabulary() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/grammar/featmath.json")
}

fn labels(output: &Path) -> Vec<Value> {
    fs::read_to_string(output.join("labels.jsonl"))
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

#[test]
fn censored_fit_runtime_is_excluded_and_cannot_shadow_a_completed_measurement() {
    // A numerical fit that timed out while the candidate reported no failure
    // was labeled with its elapsed (censored) time, and because the first
    // label of a program/cut wins it also shadowed the completed measurement.
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    let (contexts, observations) = corpus_fixture(root, |rows| {
        let mut censored = rows[0].clone();
        censored["numerical_runtime_seconds"] = json!(30.0);
        censored["fit_outcome"] = json!("timeout");
        let mut completed = rows[1].clone();
        completed["numerical_runtime_seconds"] = json!(0.25);
        completed["fit_outcome"] = json!("success");
        rows[1] = completed;
        rows.insert(0, censored);
    });
    let output = root.join("corpus");
    corpus::build(root, &contexts, &vocabulary(), &observations, &output, true).unwrap();
    let labels = labels(&output);
    let runtime_of = |program: &str| {
        labels
            .iter()
            .find(|row| row["context"] == 0 && row["program"] == program)
            .map(|row| row["runtime_seconds"].as_f64().unwrap())
    };
    // The completed measurement of the same program supplies its label.
    assert_eq!(runtime_of(&genome(0)), Some(0.1));
    // A completed numerical fit takes precedence over the candidate wall time.
    assert_eq!(runtime_of(&genome(1)), Some(0.25));
    let provenance: Value = read_json(&output.join("provenance.json")).unwrap();
    assert_eq!(provenance["counts"]["runtime_not_completed"], 1);
    assert!(provenance["counts"]["duplicate_program_cut"].is_null());
    let index: Value = read_json(&output.join("index.json")).unwrap();
    assert_eq!(index["records"], 16);
}

#[test]
fn completed_fit_outcomes_and_legacy_rows_remain_labels() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    let (contexts, observations) = corpus_fixture(root, |rows| {
        rows[2]["numerical_runtime_seconds"] = json!(0.5);
        rows[2]["fit_outcome"] = json!("success");
        rows[3]["fit_outcome"] = Value::Null;
    });
    let output = root.join("corpus");
    corpus::build(root, &contexts, &vocabulary(), &observations, &output, true).unwrap();
    let provenance: Value = read_json(&output.join("provenance.json")).unwrap();
    assert!(provenance["counts"]["runtime_not_completed"].is_null());
    let index: Value = read_json(&output.join("index.json")).unwrap();
    assert_eq!(index["records"], 16);
}

#[test]
fn missing_or_invalid_runtime_refuses_the_corpus_instead_of_becoming_zero() {
    // Regression protection for existing behavior: a missing runtime is an
    // error for the whole build, never a zero-second (fast) label, and a
    // present-but-null numerical time never falls back to another scope.
    for (case, edit) in [
        ("runtime null", json!({"runtime_seconds":null})),
        ("runtime absent", json!({"runtime_seconds":"__remove__"})),
        (
            "numerical null",
            json!({"numerical_runtime_seconds":null,"runtime_seconds":0.1}),
        ),
        ("negative", json!({"runtime_seconds":-1.0})),
        ("text", json!({"runtime_seconds":"fast"})),
    ] {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path();
        let (contexts, observations) = corpus_fixture(root, |rows| {
            for (key, value) in edit.as_object().unwrap() {
                if value == "__remove__" {
                    rows[4].as_object_mut().unwrap().remove(key);
                } else {
                    rows[4][key] = value.clone();
                }
            }
        });
        let output = root.join("corpus");
        assert!(
            corpus::build(root, &contexts, &vocabulary(), &observations, &output, true).is_err(),
            "{case}: accepted"
        );
        assert!(!output.exists(), "{case}: published a partial corpus");
    }
}

#[cfg(feature = "neural")]
mod comparisons {
    use jope_parallel::{
        corpus::{Group, Index},
        training::{
            objective::{composite, validate_rows, Outputs, RowLabels, SampledComparisons},
            sample, sample_replay_fresh, validation_panel,
        },
    };
    use std::collections::BTreeMap;
    use tch::{Kind, Tensor};

    /// Five groups of eight records; group 4 is meta_validation. Group `g`
    /// has context `10 + g` and records `8g..8g+8`, champion first.
    fn index() -> Index {
        Index {
            schema: "parallel.teaching_corpus.v1".into(),
            qualification: true,
            vocabulary_sha256: "v".into(),
            context_index_sha256: "c".into(),
            records_sha256: "r".into(),
            record_bytes: 532,
            records: 40,
            legal_sets: vec![],
            contexts: vec![],
            labels_sha256: None,
            groups: (0..5)
                .map(|g| Group {
                    context: 10 + g,
                    family: format!("family{}", g / 2),
                    partition: if g == 4 {
                        "meta_validation"
                    } else {
                        "meta_train"
                    }
                    .into(),
                    records: (g * 8..g * 8 + 8).collect(),
                    max_length: 8,
                })
                .collect(),
        }
    }

    fn families(index: &Index, partition: &str) -> BTreeMap<String, Vec<usize>> {
        let mut families = BTreeMap::<String, Vec<usize>>::new();
        for (g, group) in index.groups.iter().enumerate() {
            if group.partition == partition {
                families.entry(group.family.clone()).or_default().push(g);
            }
        }
        families
    }

    /// Labels a record would carry in `index()`: record `8g + k` has value
    /// and percentile decreasing with `k`.
    fn row(record: usize) -> RowLabels {
        let k = record % 8;
        RowLabels {
            record,
            context: 10 + record / 8,
            value: 1.0 / (1.0 + k as f32 / 3.0),
            percentile: 1.0 - k as f32 / 7.0,
            runtime: (0.1 * (k + 1) as f32).ln_1p(),
        }
    }

    #[test]
    fn sampled_training_panel_and_fresh_comparisons_pass_the_boundary() {
        let index = index();
        let checker = SampledComparisons::new(&index).unwrap();
        let train = families(&index, "meta_train");
        for serial in (0..512).step_by(64) {
            let ids = sample(&index, &train, serial, 64, 12);
            checker.check_ids(&index, &ids, "meta_train").unwrap();
            let rows = ids.iter().map(|&id| row(id)).collect::<Vec<_>>();
            checker.check_rows(&index, &rows).unwrap();
        }
        let fresh = BTreeMap::from([("family0".to_string(), vec![0usize])]);
        let fresh_by_teacher = BTreeMap::from([(0usize, vec![5usize, 6])]);
        let ids = sample_replay_fresh(&index, &train, &fresh, &fresh_by_teacher, 0, 64, 12, true);
        checker.check_ids(&index, &ids, "meta_train").unwrap();
        for panel in validation_panel(&index, 2, 12).unwrap() {
            checker
                .check_ids(&index, &panel, "meta_validation")
                .unwrap();
            // A validation comparison is never accepted as training input.
            assert!(checker.check_ids(&index, &panel, "meta_train").is_err());
        }
    }

    #[test]
    fn invalid_sampled_comparisons_are_refused_before_records_are_read() {
        let index = index();
        let checker = SampledComparisons::new(&index).unwrap();
        let valid: Vec<usize> = (0..8).chain(8..16).collect();
        checker.check_ids(&index, &valid, "meta_train").unwrap();
        let mut cases: Vec<(&str, Vec<usize>)> = Vec::new();
        let mut cross = valid.clone();
        cross[3] = 11; // a record of another group: a cross-identity pair
        cases.push(("mixes corpus groups", cross));
        let mut rotated = valid.clone();
        rotated.swap(0, 1); // champion no longer first
        cases.push(("group champion", rotated));
        let mut duplicate = valid.clone();
        duplicate[5] = duplicate[4];
        cases.push(("repeats a record", duplicate));
        let mut outside = valid.clone();
        outside[0] = 400;
        cases.push(("outside every corpus group", outside));
        cases.push(("complete comparisons", valid[..7].to_vec()));
        cases.push(("complete comparisons", Vec::new()));
        let validation: Vec<usize> = (32..40).collect();
        cases.push(("partition", validation));
        for (reason, ids) in cases {
            let error = checker.check_ids(&index, &ids, "meta_train").unwrap_err();
            assert!(error.to_string().contains(reason), "{reason}: {error}");
        }
        // Labels must carry the context of the group they were sampled from.
        let mut rows = valid.iter().map(|&id| row(id)).collect::<Vec<_>>();
        checker.check_rows(&index, &rows).unwrap();
        rows[2].context = 11;
        assert!(checker.check_rows(&index, &rows).is_err());
    }

    #[test]
    fn corpus_group_membership_must_be_unambiguous() {
        let mut index = index();
        index.groups[1].records[3] = 2; // record 2 is also in group 0
        assert!(SampledComparisons::new(&index).is_err());
        let mut index = self::index();
        index.groups[0].records[7] = 99; // outside the corpus
        assert!(SampledComparisons::new(&index).is_err());
    }

    struct Leaves {
        logits: Tensor,
        rewards: Tensor,
        runtime: Tensor,
        stability: Tensor,
        ablation: Tensor,
        tokens: Tensor,
        legal: Tensor,
    }

    /// Minimal valid tensors for `b` rows: BOS, one body token, EOS, PAD.
    fn leaves(b: i64) -> Leaves {
        let (t, v) = (4i64, 5i64);
        let tokens = Tensor::from_slice(&[1i64, 3, 2, 0])
            .repeat([b, 1])
            .reshape([b, t]);
        let mut legal = vec![0u8; (b * (t - 1) * v) as usize];
        for i in 0..b as usize {
            for (j, set) in [vec![3usize, 4], vec![2], vec![0]].iter().enumerate() {
                for &k in set {
                    legal[(i * 3 + j) * v as usize + k] = 1;
                }
            }
        }
        let leaf = |shape: &[i64]| {
            (Tensor::arange(
                shape.iter().product::<i64>(),
                (Kind::Float, tch::Device::Cpu),
            ) * 0.01)
                .sin()
                .reshape(shape)
                .set_requires_grad(true)
        };
        Leaves {
            logits: leaf(&[b, t + 1, v]),
            rewards: leaf(&[b, 4]),
            runtime: leaf(&[b]).abs().detach().set_requires_grad(true),
            stability: leaf(&[b]),
            ablation: leaf(&[b]),
            legal: Tensor::from_slice(&legal)
                .reshape([b, t - 1, v])
                .to_kind(Kind::Bool),
            tokens,
        }
    }

    fn objective(l: &Leaves, rows: &[RowLabels]) -> anyhow::Result<(Tensor, [f64; 5])> {
        composite(
            &Outputs {
                logits: &l.logits,
                rewards: &l.rewards,
                runtime: &l.runtime,
                stability: &l.stability,
                ablation: &l.ablation,
            },
            std::iter::empty(),
            &l.tokens,
            &l.tokens.ne(0),
            &l.legal,
            rows,
        )
    }

    #[test]
    fn malformed_supervision_is_refused_before_any_loss_or_gradient() {
        // Before this boundary, each case below produced a finite loss: the
        // ranking terms silently dropped nonfinite values and split a mixed
        // comparison into sub-groups, a duplicate counted twice, and an
        // out-of-range percentile or a negative "missing" runtime sentinel
        // was trained on as if it were a measurement.
        let valid = (0..16).map(row).collect::<Vec<_>>();
        let cases: Vec<(&str, Box<dyn Fn(&mut Vec<RowLabels>)>)> = vec![
            ("spans contexts", Box::new(|r| r[3].context = 99)),
            ("repeats a record", Box::new(|r| r[5].record = r[2].record)),
            ("invalid value", Box::new(|r| r[4].value = f32::NAN)),
            ("invalid value", Box::new(|r| r[4].value = 0.0)),
            ("invalid value", Box::new(|r| r[4].value = 1.5)),
            (
                "invalid percentile",
                Box::new(|r| r[6].percentile = f32::NAN),
            ),
            ("invalid percentile", Box::new(|r| r[6].percentile = 1.5)),
            ("invalid runtime", Box::new(|r| r[7].runtime = -1.0)),
            (
                "invalid runtime",
                Box::new(|r| r[7].runtime = f32::INFINITY),
            ),
            ("complete comparisons", Box::new(|r| r.truncate(7))),
        ];
        for (reason, edit) in &cases {
            let mut rows = valid.clone();
            edit(&mut rows);
            let error = validate_rows(&rows).unwrap_err();
            assert!(error.to_string().contains(reason), "{reason}: {error}");
            let l = leaves(rows.len() as i64);
            let error = objective(&l, &rows)
                .err()
                .unwrap_or_else(|| panic!("{reason}: composite accepted malformed supervision"));
            assert!(error.to_string().contains(reason), "{reason}: {error}");
            // No loss exists, so nothing can be backpropagated or applied.
            for t in [&l.logits, &l.rewards, &l.runtime, &l.ablation] {
                assert!(!t.grad().defined(), "{reason}: gradient was formed");
            }
        }
        // Positive control on the same fixture.
        let l = leaves(16);
        let (loss, terms) = objective(&l, &valid).unwrap();
        assert!(loss.double_value(&[]).is_finite());
        assert!(terms.iter().all(|x| x.is_finite()));
        loss.backward();
        assert!(l.rewards.grad().defined());
    }

    #[test]
    fn pair_reversal_changes_direction_without_inventing_comparisons() {
        // A reversed label order is still a valid comparison: the boundary
        // accepts it and the objective's ranking gradient flips sign.
        let forward = (0..8).map(row).collect::<Vec<_>>();
        let mut reversed = forward.clone();
        for k in 0..8 {
            reversed[k].value = forward[7 - k].value;
            reversed[k].percentile = forward[7 - k].percentile;
        }
        let gradient = |rows: &[RowLabels]| {
            let l = leaves(8);
            let (loss, _) = objective(&l, rows).unwrap();
            loss.backward();
            Vec::<f64>::try_from(
                l.rewards
                    .grad()
                    .sum_dim_intlist(&[1i64][..], false, Kind::Double),
            )
            .unwrap()
        };
        let a = gradient(&forward);
        let b = gradient(&reversed);
        // Row 0 is pushed up when it is the best and down when it is the worst.
        assert!(a[0] < 0.0 && b[0] > 0.0, "{a:?} {b:?}");
        assert!(a[7] > 0.0 && b[7] < 0.0, "{a:?} {b:?}");
    }
}
