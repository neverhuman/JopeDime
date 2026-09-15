//! Explicit continuation-objective corpus. Direct-score ledgers are preserved, never rewritten.
use crate::{
    manifest::{hash, resolve},
    native_genome::{self, Arm, Candidate, ContinuationTarget},
    native_graph::Artifact,
};
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::Write,
    path::{Path, PathBuf},
};
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TargetSet {
    pub schema: String,
    pub run: String,
    pub generation: u64,
    pub source_graph_state: PathBuf,
    pub qualification: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preparation_overlay: Option<Artifact>,
    pub parent_corpus: Artifact,
    pub vocabulary: Artifact,
    pub admission_policy: Artifact,
    pub frozen_scientific_policy: Artifact,
    pub targets: Vec<Artifact>,
    pub training_arms: BTreeSet<Arm>,
}
fn immutable(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path.parent().context("corpus parent absent")?;
    fs::create_dir_all(parent)?;
    let (mut f, tmp) = tempfile::Builder::new()
        .prefix(".continuation-pending-")
        .tempfile_in(parent)?
        .keep()?;
    f.write_all(bytes)?;
    f.sync_all()?;
    match crate::native_persistence::publish_staged(&tmp, path) {
        Ok(()) => (),
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => ensure!(
            fs::read(path)? == bytes,
            "conflicting continuation corpus bytes"
        ),
        Err(e) => return Err(e.into()),
    }
    fs::File::open(parent)?.sync_all()?;
    crate::native_persistence::discard_staging(&tmp)?;
    fs::File::open(parent)?.sync_all()?;
    Ok(())
}
/// Reconstruct source closure from immutable native admissions and producer receipts.
/// Original archived admissions remain mandatory; no graph edge is replaced with a success claim.
pub fn input_closure(
    root: &Path,
    manifest: &Artifact,
    graph_state: &Path,
) -> Result<BTreeMap<PathBuf, String>> {
    let set: TargetSet = native_genome::load(root, manifest)?;
    crate::manifest::relative(&set.source_graph_state)?;
    ensure!(
        root.join(&set.source_graph_state) == graph_state,
        "source graph state differs from target-set binding"
    );
    let mut pending = vec![
        manifest.clone(),
        set.parent_corpus,
        set.vocabulary,
        set.admission_policy,
        set.frozen_scientific_policy,
    ];
    pending.extend(set.targets);
    pending.extend(set.preparation_overlay);
    let mut all = BTreeMap::new();
    fn refs(v: &Value, out: &mut Vec<Artifact>) {
        if let Some(o) = v.as_object() {
            if let (Some(path), Some(sha)) = (
                o.get("path").and_then(Value::as_str),
                o.get("sha256").and_then(Value::as_str),
            ) {
                if !Path::new(path).is_absolute()
                    && sha.len() == 64
                    && sha
                        .bytes()
                        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
                {
                    out.push(Artifact {
                        path: path.into(),
                        sha256: sha.into(),
                    });
                }
            }
            for x in o.values() {
                refs(x, out)
            }
        } else if let Some(a) = v.as_array() {
            for x in a {
                refs(x, out)
            }
        }
    }
    while let Some(a) = pending.pop() {
        if let Some(old) = all.get(&a.path) {
            ensure!(old == &a.sha256, "source closure conflicting identity");
            continue;
        }
        ensure!(
            all.len() < 1_000_000,
            "source closure exceeds one million files"
        );
        let path = resolve(root, &a.path)?;
        ensure!(
            fs::symlink_metadata(&path)?.is_file(),
            "source closure special file"
        );
        ensure!(
            crate::manifest::file_hash(&path)? == a.sha256,
            "source closure file changed"
        );
        let length = fs::metadata(&path)?.len();
        let typed_large =
            length > 16 * 1024 * 1024 && path.extension().is_some_and(|x| x == "json");
        if length <= 16 * 1024 * 1024 || typed_large {
            let maximum = if typed_large {
                native_genome::DATASET_METADATA_LIMIT
            } else {
                16 * 1024 * 1024
            };
            let parsed =
                serde_json::from_slice::<Value>(&native_genome::checked(root, &a, maximum)?);
            if typed_large {
                ensure!(parsed.is_ok(), "large JSON corpus metadata is malformed");
            }
            if let Ok(value) = parsed {
                if typed_large {
                    ensure!(
                        matches!(
                            value["schema"].as_str(),
                            Some(
                                "parallel.teaching_corpus.v1"
                                    | "parallel.teaching_shards.v1"
                                    | "parallel.context_index.v1"
                                    | "jain.corpus_files.v1"
                                    | "jain.corpus_parts.v1"
                            )
                        ),
                        "large JSON source is not a supported typed corpus metadata manifest"
                    );
                }
                refs(&value, &mut pending);
                match value["schema"].as_str() {
                    Some("jope.native_graph_result.v1") => {
                        let result: crate::native_graph::VerifiedResult =
                            serde_json::from_value(value)?;
                        let key = hash(&serde_json::to_vec(&(
                            &result.run,
                            result.generation,
                            &result.task_id,
                        ))?);
                        let archived = graph_state
                            .join("archived-admissions")
                            .join(format!("{key}.json"));
                        // Completed predecessors must have finished the durable archive transition.
                        // Never capture an active pathname that the coordinator later removes.
                        let admission_path = archived;
                        let admission_ref = Artifact {
                            path: admission_path.strip_prefix(root)?.into(),
                            sha256: result.admission_sha256,
                        };
                        let admission: crate::native_graph::Admission =
                            native_genome::load(root, &admission_ref)?;
                        ensure!(
                            admission.manifest.sha256 == result.task_manifest_sha256,
                            "source admission task differs"
                        );
                        let task_path = resolve(root, &admission.manifest.path)?;
                        let (task, input) = crate::smartcluster_worker::verify_task_inputs(
                            root,
                            &task_path,
                            &admission.manifest.sha256,
                        )?;
                        let stem = task_path
                            .file_stem()
                            .and_then(|s| s.to_str())
                            .context("source task stem")?;
                        pending.push(Artifact {
                            path: task_path
                                .with_file_name(format!("{stem}.input.json"))
                                .strip_prefix(root)?
                                .into(),
                            sha256: task.input_manifest_sha256,
                        });
                        pending.push(admission_ref);
                        for (path, sha256) in input.input_artifacts {
                            pending.push(Artifact { path, sha256 })
                        }
                        if let Some(c) = input.closure {
                            pending.push(Artifact {
                                path: c.path,
                                sha256: c.sha256,
                            })
                        }
                        if let crate::smartcluster_worker::Operation::Proposal { output, .. } =
                            input.operation
                        {
                            let producer: Value = native_genome::load(root, &result.output)?;
                            let artifacts: BTreeMap<PathBuf, String> =
                                serde_json::from_value(producer["artifacts"].clone())?;
                            for (path, sha256) in artifacts {
                                pending.push(Artifact {
                                    path: output.join(path),
                                    sha256,
                                })
                            }
                        }
                    }
                    Some("jope.native_genome_completion.v1") => {
                        let result: native_genome::GenomeResult = serde_json::from_value(value)?;
                        let base = result
                            .search
                            .path
                            .parent()
                            .context("original search parent")?;
                        for (path, sha256) in result.artifacts {
                            pending.push(Artifact {
                                path: base.join(path),
                                sha256,
                            })
                        }
                        for (path, sha256) in native_genome::required_inputs(root, &result.task)? {
                            pending.push(Artifact { path, sha256 })
                        }
                    }
                    _ => (),
                }
            }
        }
        all.insert(a.path, a.sha256);
    }
    Ok(all)
}

/// Actual existing strict validators decide every source outcome before deriving a new target.
/// This function cannot create an admission policy or change original fitted program identity.
pub fn build(root: &Path, manifest: &Artifact, output: &Path) -> Result<()> {
    crate::manifest::relative(output)?;
    let output = root.join(output);
    let mut files = render(root, manifest)?;
    let index = files
        .remove("index.json")
        .context("rendered index absent")?;
    fs::create_dir_all(&output)?;
    for (name, bytes) in files {
        immutable(&output.join(name), &bytes)?;
    }
    immutable(&output.join("index.json"), &index)
}
/// Pure deterministic rendering also drives independent native Graph verification.
pub(crate) fn render(root: &Path, manifest: &Artifact) -> Result<BTreeMap<String, Vec<u8>>> {
    let set: TargetSet = native_genome::load(root, manifest)?;
    ensure!(
        set.schema == "jope.continuation_target_set.v1"
            && !set.run.is_empty()
            && !set.targets.is_empty()
            && set.targets.len() <= 100000
            && !set.training_arms.is_empty(),
        "target-set bounds/schema"
    );
    let parent_index: Value = native_genome::load_corpus_index(root, &set.parent_corpus)?;
    let previous_set = set
        .parent_corpus
        .path
        .parent()
        .context("parent corpus directory")?
        .join("target-set.json");
    if parent_index["label_policy"] == "strict_continuation_midrank_v1" {
        ensure!(
            crate::manifest::file_hash(&resolve(root, &previous_set)?)?
                == parent_index["target_set_sha256"],
            "previous continuation target set changed"
        );
        let previous: TargetSet = crate::manifest::read_json(&resolve(root, &previous_set)?)?;
        ensure!(
            previous.run == set.run
                && previous.generation < set.generation
                && previous.vocabulary == set.vocabulary
                && previous.frozen_scientific_policy == set.frozen_scientific_policy
                && previous.training_arms == set.training_arms
                && previous.qualification == set.qualification,
            "continuation corpus changed prior objective or run"
        );
        let identities: BTreeSet<_> = set.targets.iter().map(|a| (&a.path, &a.sha256)).collect();
        ensure!(
            previous
                .targets
                .iter()
                .all(|a| identities.contains(&(&a.path, &a.sha256))),
            "previous continuation targets omitted"
        );
    }

    let scientific: Value = native_genome::load(root, &set.frozen_scientific_policy)?;
    ensure!(
        scientific["schema"] == "jope.native_learning_policy.v1"
            && scientific["training_target"] == "continuation_only_v1"
            && scientific["training_arms"] == serde_json::to_value(&set.training_arms)?,
        "continuation objective not explicitly frozen"
    );
    let vocab_bytes = native_genome::checked(root, &set.vocabulary, 16 * 1024 * 1024)?;
    let vocab: crate::grammar::Vocab = serde_json::from_slice(&vocab_bytes)?;
    let mut groups = BTreeMap::<String, Vec<(Artifact, ContinuationTarget, Candidate)>>::new();
    let excluded: BTreeSet<String> =
        serde_json::from_value(scientific["excluded_families"].clone())?;
    let mut budgets = BTreeMap::new();
    let mut seen_candidates = BTreeSet::new();
    let mut seen = BTreeSet::new();
    let mut refused_targets = Vec::new();
    for reference in &set.targets {
        ensure!(
            seen.insert(reference.sha256.clone()),
            "duplicate target identity"
        );
        let target: ContinuationTarget = native_genome::load(root, reference)?;
        let verified = native_genome::target(root, &target.continuation)?;
        ensure!(
            serde_json::to_value(&target)? == serde_json::to_value(&verified)?,
            "continuation target differs from original results"
        );
        let task_result: native_genome::GenomeResult =
            native_genome::load(root, &target.continuation)?;
        let task: native_genome::GenomeTask = native_genome::load(root, &task_result.task)?;
        ensure!(
            task.admission_policy == set.admission_policy
                && task.vocabulary == set.vocabulary
                && task.qualification == set.qualification
                && task.preparation_overlay == set.preparation_overlay,
            "target scope/vocabulary changed"
        );
        let candidate: Candidate = native_genome::load(root, &target.candidate)?;
        ensure!(
            candidate.run == set.run
                && candidate.generation <= set.generation
                && candidate.context.partition == "meta_train",
            "foreign generation or sealed target"
        );
        ensure!(
            !excluded.contains(&candidate.context.family),
            "excluded family target"
        );
        ensure!(
            seen_candidates.insert((
                candidate.generation,
                candidate.arm,
                hash(&serde_json::to_vec(&candidate.context)?),
                candidate.genome_sha256.clone()
            )),
            "duplicate candidate outcome cannot become independent training evidence"
        );
        let budget = (task.evaluations, task.seconds);
        if let Some(old) = budgets.insert(candidate.generation, budget) {
            ensure!(
                old == budget,
                "unmatched continuation budget within generation"
            );
        }
        // Every outcome remains in the target set, including failures and comparison-only arms.
        if !set.training_arms.contains(&candidate.arm) {
            continue;
        }
        let key = hash(&serde_json::to_vec(&candidate.context)?);
        if let Some(reason) = training_refusal(&target) {
            refused_targets.push(json!({"target":reference,"context_sha256":key,"arm":candidate.arm,
                "reason":reason,"completed_children":target.completed_children,
                "completed_evaluations":target.completed_evaluations,"failed_evaluations":target.failed_evaluations,
                "fit_runtime_seconds":target.fit_runtime_seconds}));
            continue;
        }
        groups
            .entry(key)
            .or_default()
            .push((reference.clone(), target, candidate));
    }
    ensure!(
        !groups.is_empty(),
        "no admitted continuation training targets; training held"
    );
    let (groups, undersized) = eligible_contexts(groups, MINIMUM_ADMITTED_TARGETS_PER_CONTEXT);
    let mut excluded_contexts = Vec::new();
    for (key, group) in &undersized {
        let row = &group[0].2.context;
        excluded_contexts.push(json!({"context_sha256":key,"dataset":row.dataset,"family":row.family,
            "cut":row.cut,"admitted_targets":group.len(),"minimum":MINIMUM_ADMITTED_TARGETS_PER_CONTEXT,
            "reason":"fewer_than_minimum_admitted_targets",
            "targets":group.iter().map(|x| x.0.clone()).collect::<Vec<_>>()}));
    }
    ensure!(
        !groups.is_empty(),
        "no usable continuation context has eight admitted targets; training held"
    );
    let mut records = Vec::new();
    let mut labels = Vec::new();
    let mut contexts = Vec::new();
    let mut index_groups = Vec::new();
    let mut legal_sets = vec![vec![0usize]];
    let mut legal_ids = BTreeMap::from([(vec![0usize], 0usize)]);
    let mut families = BTreeSet::new();
    let mut record_count = 0usize;
    for (_, mut group) in groups {
        ensure!(
            group.len() >= MINIMUM_ADMITTED_TARGETS_PER_CONTEXT,
            "continuation context lacks eight independently identified targets"
        );
        group.sort_by(|a, b| {
            a.1.quality_rmse_cut_normalized
                .unwrap()
                .total_cmp(&b.1.quality_rmse_cut_normalized.unwrap())
                .then(a.0.sha256.cmp(&b.0.sha256))
        });
        let context = contexts.len();
        let row = group[0].2.context.clone();
        families.insert(row.family.clone());
        contexts.push(row.clone());
        let scores: Vec<f64> = group
            .iter()
            .map(|x| x.1.quality_rmse_cut_normalized.unwrap())
            .collect();
        let percentiles = crate::teaching_store::midranks(&scores);
        let median = scores[scores.len() / 2].max(1e-8);
        let start = record_count;
        let mut max_length = 0;
        for (i, (reference, target, candidate)) in group.into_iter().enumerate() {
            let tokens = vocab.encode(&candidate.genome)?;
            ensure!(
                tokens.len() <= crate::corpus::SEQUENCE,
                "unrepresentable exact parent program"
            );
            let mut record = crate::corpus::Record {
                tokens: [0; crate::corpus::SEQUENCE],
                legal: [0; crate::corpus::SEQUENCE],
                length: tokens.len(),
                context,
                value: (1. / (1. + scores[i] / median)) as f32,
                percentile: percentiles[i],
                runtime: target.fit_runtime_seconds.ln_1p() as f32,
            };
            let mut grammar = crate::grammar::State::default();
            grammar.sampling = false;
            for (j, token) in tokens.iter().enumerate() {
                record.tokens[j] = *token as i64;
                if j == 0 {
                    continue;
                }
                let legal = grammar.legal(&vocab);
                let id = *legal_ids.entry(legal.clone()).or_insert_with(|| {
                    let n = legal_sets.len();
                    legal_sets.push(legal);
                    n
                });
                ensure!(
                    id < u16::MAX as usize,
                    "continuation grammar table overflow"
                );
                record.legal[j] = id as i64;
                grammar.advance(&vocab, *token)?;
            }
            records.extend(record.bytes());
            max_length = max_length.max(record.length);
            serde_json::to_writer(
                &mut labels,
                &json!({"schema":"jope.continuation_training_label.v1","record":record_count,"context":context,
                "program":candidate.genome,"program_sha256":candidate.genome_sha256,"target":reference,
                "target_kind":target.target_kind,"protocol":target.protocol,"completed_children":target.completed_children,
                "rmse_cut_normalized":scores[i],"fit_runtime_seconds":target.fit_runtime_seconds,"runtime_seconds":target.fit_runtime_seconds,
                "direct_scores_are_separate":true,"heldout_final_test_used":false}),
            )?;
            labels.push(b'\n');
            record_count += 1;
        }
        index_groups.push(crate::corpus::Group {
            context,
            family: row.family,
            partition: row.partition,
            records: (start..record_count).collect(),
            max_length,
        });
    }
    ensure!(
        families.len() >= 2,
        "continuation corpus lacks two training families"
    );
    let context_bytes = serde_json::to_vec(
        &json!({"schema":"parallel.context_index.v1","qualification":set.qualification,"encodings":contexts}),
    )?;
    let index = crate::corpus::Index {
        schema: "parallel.teaching_corpus.v1".into(),
        qualification: set.qualification,
        vocabulary_sha256: set.vocabulary.sha256.clone(),
        context_index_sha256: hash(&context_bytes),
        records_sha256: hash(&records),
        record_bytes: crate::corpus::RECORD_BYTES,
        records: record_count,
        legal_sets,
        contexts,
        groups: index_groups,
    };
    let mut files = BTreeMap::from([
        ("records.bin".into(), records.clone()),
        ("labels.jsonl".into(), labels.clone()),
        ("vocabulary.json".into(), vocab_bytes),
        ("contexts.json".into(), context_bytes),
        (
            "target-set.json".into(),
            native_genome::checked(root, manifest, 16 * 1024 * 1024)?,
        ),
    ]);
    let mut index_value = serde_json::to_value(&index)?;
    index_value["label_policy"] = json!("strict_continuation_midrank_v1");
    index_value["target_set_sha256"] = json!(manifest.sha256);
    if let Some(a) = &set.preparation_overlay {
        index_value["preparation_overlay"] = serde_json::to_value(a)?;
    }
    let index_bytes = serde_json::to_vec(&index_value)?;
    files.insert("provenance.json".into(),serde_json::to_vec(&json!({"schema":"jope.continuation_corpus_provenance.v1","index_sha256":hash(&index_bytes),"labels_sha256":hash(&labels),
        "parent_corpus":set.parent_corpus,"target_set":manifest,"label_policy":"strict_continuation_midrank_v1","frozen_scientific_policy":set.frozen_scientific_policy,
        "direct_history_preserved":true,"protocol":native_genome::CONTINUATION_PROTOCOL,
        "training_admission":{"minimum_completed_children":native_genome::MINIMUM_COMPLETED_CHILDREN,
            "completed_child":"continuation row with null original failure and measured RMSE; parent row excluded",
            "minimum_admitted_targets_per_context":MINIMUM_ADMITTED_TARGETS_PER_CONTEXT},
        "refused_targets":refused_targets,"excluded_contexts":excluded_contexts,"targets":set.targets}))?);
    files.insert("index.json".into(), index_bytes);
    Ok(files)
}
/// plain_seeded_continuation_v1 corpus minimum: undersized contexts are refused, never padded.
pub(crate) const MINIMUM_ADMITTED_TARGETS_PER_CONTEXT: usize = 8;
/// Refused continuation targets stay in the target set and provenance but never become labels.
fn training_refusal(target: &ContinuationTarget) -> Option<String> {
    if target.training_admission != native_genome::TRAINING_ADMITTED {
        Some(target.training_admission.clone())
    } else if target.quality_rmse_cut_normalized.is_none() {
        Some("no_measured_quality".into())
    } else {
        None
    }
}
/// Deterministic split by context key; eligible contexts keep all of their admitted targets.
fn eligible_contexts<T>(
    groups: BTreeMap<String, Vec<T>>,
    minimum: usize,
) -> (BTreeMap<String, Vec<T>>, Vec<(String, Vec<T>)>) {
    let mut eligible = BTreeMap::new();
    let mut undersized = Vec::new();
    for (key, group) in groups {
        if group.len() >= minimum {
            eligible.insert(key, group);
        } else {
            undersized.push((key, group));
        }
    }
    (eligible, undersized)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn output_conflicts_preserve_original_bytes() {
        let t = tempfile::tempdir().unwrap();
        let p = t.path().join("x");
        immutable(&p, b"old").unwrap();
        assert!(immutable(&p, b"new").is_err());
        assert_eq!(fs::read(p).unwrap(), b"old");
    }
    #[test]
    fn large_original_index_transitive_edges_are_retained_and_corruption_refused() {
        let t = tempfile::tempdir().unwrap();
        let write = |name: &str, bytes: &[u8]| {
            fs::write(t.path().join(name), bytes).unwrap();
            Artifact {
                path: name.into(),
                sha256: hash(bytes),
            }
        };
        let leaf = write("original.bin", b"original dependency bytes");
        let index = write(
            "index.json",
            &serde_json::to_vec(&json!({"schema":"parallel.teaching_shards.v1",
            "retained_metadata":"x".repeat(17 * 1024 * 1024),"shards":[leaf]}))
            .unwrap(),
        );
        let other = write("policy.json", b"{}");
        let set = TargetSet {
            preparation_overlay: None,
            schema: "jope.continuation_target_set.v1".into(),
            run: "r".into(),
            generation: 0,
            source_graph_state: "archived".into(),
            qualification: false,
            parent_corpus: index,
            vocabulary: other.clone(),
            admission_policy: other.clone(),
            frozen_scientific_policy: other,
            targets: vec![],
            training_arms: BTreeSet::from([Arm::Model]),
        };
        let manifest = write("targets.json", &serde_json::to_vec(&set).unwrap());
        let closure = input_closure(t.path(), &manifest, &t.path().join("archived")).unwrap();
        assert_eq!(closure.get(&leaf.path), Some(&leaf.sha256));
        fs::write(t.path().join(&leaf.path), b"changed dependency").unwrap();
        assert!(input_closure(t.path(), &manifest, &t.path().join("archived")).is_err());
    }
    #[test]
    fn undersized_contexts_are_refused_while_eligible_contexts_pack() {
        let groups = BTreeMap::from([
            ("a".to_string(), vec![0; 8]),
            ("b".to_string(), vec![0; 7]),
            ("c".to_string(), vec![0; 12]),
        ]);
        let (eligible, undersized) =
            eligible_contexts(groups, MINIMUM_ADMITTED_TARGETS_PER_CONTEXT);
        assert_eq!(eligible.keys().cloned().collect::<Vec<_>>(), vec!["a", "c"]);
        assert_eq!(eligible["c"].len(), 12);
        assert_eq!(undersized.len(), 1);
        assert_eq!((undersized[0].0.as_str(), undersized[0].1.len()), ("b", 7));
        let only_small = BTreeMap::from([("z".to_string(), vec![0; 3])]);
        let (eligible, undersized) = eligible_contexts(only_small, 8);
        assert!(eligible.is_empty());
        assert_eq!(undersized.len(), 1);
    }
    #[test]
    fn refused_continuation_targets_never_become_training_labels() {
        let a = Artifact {
            path: "target.json".into(),
            sha256: "a".repeat(64),
        };
        let mut target = ContinuationTarget {
            schema: "jope.continuation_target.v2".into(),
            protocol: native_genome::CONTINUATION_PROTOCOL.into(),
            candidate: a.clone(),
            direct: a.clone(),
            continuation: a.clone(),
            parent_genome_sha256: "b".repeat(64),
            winning_original_observation: Some(a.clone()),
            winning_row: Some(0),
            quality_rmse_cut_normalized: Some(0.5),
            fit_runtime_seconds: 31.,
            completed_evaluations: 1,
            failed_evaluations: 1,
            completed_children: 0,
            training_admission: native_genome::REFUSED_NO_COMPLETED_CHILD.into(),
            evaluation_budget: 24,
            search_seconds_budget: 54,
            target_kind: "best_observed_outcome_after_plain_seeded_continuation_v1".into(),
        };
        assert_eq!(
            training_refusal(&target).as_deref(),
            Some(native_genome::REFUSED_NO_COMPLETED_CHILD)
        );
        target.completed_children = 1;
        target.training_admission = native_genome::TRAINING_ADMITTED.into();
        assert_eq!(training_refusal(&target), None);
        target.quality_rmse_cut_normalized = None;
        assert_eq!(
            training_refusal(&target).as_deref(),
            Some("no_measured_quality")
        );
    }
}
