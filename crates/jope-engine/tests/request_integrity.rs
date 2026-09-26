//! W6 follow-ups: verified resident corpus loading and explicitly declared
//! continuation modes on training requests (CPU only).

use jope_parallel::{
    corpus::{ContextRow, Group, Index, Record, RECORD_BYTES, SEQUENCE},
    manifest::{file_hash, hash, write_json},
    smartcluster_worker::Operation,
    task_contract::{classify_continuation, ContinuationMode},
    teaching_store,
};
use serde_json::{json, Value};
use std::{fs, path::Path};

// ---------------------------------------------------------------------------
// Verified resident corpus loading
// ---------------------------------------------------------------------------

fn record(g: usize, k: usize) -> Record {
    let mut r = Record {
        tokens: [0; SEQUENCE],
        legal: [0; SEQUENCE],
        length: 3,
        context: g,
        value: 1.0 / (1.0 + k as f32),
        percentile: 1.0 - k as f32 / 7.0,
        runtime: 0.1,
    };
    r.tokens[..3].copy_from_slice(&[1, 3 + k as i64, 2]);
    r
}

/// A two-group `parallel.teaching_corpus.v1` directory; returns (root, corpus).
fn corpus(dir: &Path) -> (std::path::PathBuf, std::path::PathBuf) {
    let root = dir.canonicalize().unwrap();
    let corpus = root.join("corpus");
    fs::create_dir_all(&corpus).unwrap();
    fs::copy(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/grammar/featmath.json"),
        corpus.join("vocabulary.json"),
    )
    .unwrap();
    let bytes = (0..16)
        .flat_map(|i| record(i / 8, i % 8).bytes())
        .collect::<Vec<_>>();
    fs::write(corpus.join("records.bin"), bytes).unwrap();
    let contexts = (0..2)
        .map(|c| {
            let path = corpus.join(format!("context-{c}.bin"));
            fs::write(&path, vec![c as u8; 8192]).unwrap();
            ContextRow {
                dataset: format!("d{c}"),
                family: format!("f{c}"),
                cut: 0,
                partition: "meta_train".into(),
                path: path.strip_prefix(&root).unwrap().to_path_buf(),
                sha256: file_hash(&path).unwrap(),
                prepared: format!("p{c}").into(),
                prepared_sha256: "p".repeat(64),
            }
        })
        .collect();
    let index = Index {
        schema: "parallel.teaching_corpus.v1".into(),
        qualification: true,
        vocabulary_sha256: file_hash(&corpus.join("vocabulary.json")).unwrap(),
        context_index_sha256: "declared".into(),
        records_sha256: file_hash(&corpus.join("records.bin")).unwrap(),
        record_bytes: RECORD_BYTES,
        records: 16,
        legal_sets: vec![vec![0]],
        contexts,
        labels_sha256: None,
        groups: (0..2)
            .map(|g| Group {
                context: g,
                family: format!("f{g}"),
                partition: "meta_train".into(),
                records: (g * 8..g * 8 + 8).collect(),
                max_length: 3,
            })
            .collect(),
    };
    write_json(&corpus.join("index.json"), &index).unwrap();
    (root, corpus)
}

fn groups(index: &Index) -> Value {
    serde_json::to_value(&index.groups).unwrap()
}

#[test]
fn a_resident_load_from_verified_bytes_prevents_substitution() {
    let dir = tempfile::tempdir().unwrap();
    let (root, corpus) = corpus(dir.path());
    let verified = fs::read(corpus.join("index.json")).unwrap();
    let original: Index = serde_json::from_slice(&verified).unwrap();
    // A different, valid index is written after verification.
    let mut swapped: Index = serde_json::from_slice(&verified).unwrap();
    swapped.groups.swap(0, 1);
    write_json(&corpus.join("index.json"), &swapped).unwrap();
    // The path-reading loader returns the substitute (the detected case).
    let reread = teaching_store::load(&root, &corpus).unwrap().0;
    assert_eq!(groups(&reread), groups(&swapped));
    // The verified-bytes loader returns exactly the verified index.
    let resident = teaching_store::load_resident_verified(&root, &corpus, &verified).unwrap();
    assert_eq!(groups(&resident.0), groups(&original));
    assert_eq!(resident.1.get(9).tokens[1], 3 + 1);
    // It is cached under the verified content, not the path's current bytes.
    let again = teaching_store::load_resident_verified(&root, &corpus, &verified).unwrap();
    assert!(std::rc::Rc::ptr_eq(&resident, &again));
    let other = teaching_store::load_resident(&root, &corpus).unwrap();
    assert!(!std::rc::Rc::ptr_eq(&resident, &other));
    assert_eq!(groups(&other.0), groups(&swapped));
}

#[test]
fn verified_bytes_still_verify_every_payload_they_name() {
    let dir = tempfile::tempdir().unwrap();
    let (root, corpus) = corpus(dir.path());
    // Verified index bytes naming a records digest the payload does not have.
    let mut index: Index =
        serde_json::from_slice(&fs::read(corpus.join("index.json")).unwrap()).unwrap();
    index.records_sha256 = "0".repeat(64);
    let bytes = serde_json::to_vec(&index).unwrap();
    assert!(teaching_store::load_resident_verified(&root, &corpus, &bytes).is_err());
    // Malformed verified bytes are refused, not re-read from the path.
    assert!(teaching_store::load_resident_verified(&root, &corpus, b"{\"schema\":").is_err());
    // Positive control: the unmodified bytes load, and their digest keys them.
    let good = fs::read(corpus.join("index.json")).unwrap();
    let loaded = teaching_store::load_resident_verified(&root, &corpus, &good).unwrap();
    assert_eq!(loaded.0.records, 16);
    assert_eq!(hash(&good), file_hash(&corpus.join("index.json")).unwrap());
}

// ---------------------------------------------------------------------------
// Declared continuation modes on requests
// ---------------------------------------------------------------------------

#[test]
fn the_continuation_mode_is_a_closed_serialized_vocabulary() {
    for (mode, text) in [
        (ContinuationMode::ColdInitialization, "cold_initialization"),
        (ContinuationMode::WeightWarmStart, "weight_warm_start"),
        (
            ContinuationMode::NewCorpusTransition,
            "new_corpus_transition",
        ),
        (ContinuationMode::ExactResume, "exact_resume"),
    ] {
        assert_eq!(mode.as_str(), text);
        assert_eq!(ContinuationMode::parse(text).unwrap(), mode);
        assert_eq!(serde_json::to_value(mode).unwrap(), json!(text));
        assert_eq!(
            serde_json::from_value::<ContinuationMode>(json!(text)).unwrap(),
            mode
        );
    }
    assert!(ContinuationMode::parse("resume").is_err());
    assert!(serde_json::from_value::<ContinuationMode>(json!("warm")).is_err());
    let carry = json!({"optimizer":"carry"});
    assert_eq!(
        classify_continuation(false, true, Some(&carry)).unwrap(),
        ContinuationMode::NewCorpusTransition
    );
    assert!(classify_continuation(true, true, None).is_err());
}

fn train_operation(mode: Option<&str>) -> Value {
    let mut operation = json!({"kind":"Train","corpus":"corpus","run":"run","launch":"launch",
        "micro":256,"checkpoint_steps":1000,"initialize":"parent","replay_policy":"policy.json",
        "initial_optimizer_step":60000,"requested_step_increment":3000,
        "target_optimizer_step":63000});
    if let Some(mode) = mode {
        operation["continuation_mode"] = json!(mode);
    }
    operation
}

#[test]
fn smartcluster_train_operations_carry_an_optional_declaration() {
    // A request frozen before the field existed deserializes and serializes
    // back byte for byte: its implicit meaning is unchanged.
    let old = train_operation(None);
    let parsed: Operation = serde_json::from_value(old.clone()).unwrap();
    assert_eq!(serde_json::to_value(&parsed).unwrap(), old);
    let Operation::Train {
        continuation_mode, ..
    } = &parsed
    else {
        panic!("not a train operation")
    };
    assert_eq!(*continuation_mode, None);
    // A declared request round-trips its declaration.
    let declared = train_operation(Some("new_corpus_transition"));
    let parsed: Operation = serde_json::from_value(declared.clone()).unwrap();
    assert_eq!(serde_json::to_value(&parsed).unwrap(), declared);
    // Unknown modes and unknown fields are refused.
    assert!(serde_json::from_value::<Operation>(train_operation(Some("resume"))).is_err());
    let mut unknown = train_operation(None);
    unknown["mode"] = json!("exact_resume");
    assert!(serde_json::from_value::<Operation>(unknown).is_err());
}

#[cfg(feature = "neural")]
mod neural {
    use super::*;
    use jope_parallel::{
        gpu_worker::Task, neural::AdamW, training::continuation::declared_lineage,
    };
    use tch::{nn, Device, Kind};

    fn train_task(mode: Option<&str>) -> Value {
        let mut task = json!({"kind":"Train","corpus":"corpus","run":"run","launch":"launch",
            "world":8,"micro":256,"steps":3000,"checkpoint_steps":1000,"resume":false,
            "initialize":"parent","replay_policy":null});
        if let Some(mode) = mode {
            task["continuation_mode"] = json!(mode);
        }
        task
    }

    #[test]
    fn gpu_worker_train_tasks_carry_an_optional_declaration() {
        let old = train_task(None);
        let parsed: Task = serde_json::from_value(old.clone()).unwrap();
        assert_eq!(serde_json::to_value(&parsed).unwrap(), old);
        let declared = train_task(Some("weight_warm_start"));
        let parsed: Task = serde_json::from_value(declared.clone()).unwrap();
        assert_eq!(serde_json::to_value(&parsed).unwrap(), declared);
        assert!(serde_json::from_value::<Task>(train_task(Some("warm"))).is_err());
    }

    #[test]
    fn a_declared_lineage_mode_must_match_the_request() {
        use ContinuationMode::*;
        let carry = json!({"optimizer":"carry"});
        let reset = json!({"optimizer":"reset"});
        // Implicit requests keep their meaning and say they are implicit.
        let r = declared_lineage(None, false, true, Some(&carry), None).unwrap();
        assert_eq!(
            (
                r["attempt_mode"].clone(),
                r["lineage_mode"].clone(),
                r["declaration"].clone()
            ),
            (
                json!("new_corpus_transition"),
                json!("new_corpus_transition"),
                json!("implicit")
            )
        );
        // Matching explicit declarations.
        for (declared, parent, policy) in [
            (ColdInitialization, false, None),
            (WeightWarmStart, true, None),
            (WeightWarmStart, true, Some(&reset)),
            (NewCorpusTransition, true, Some(&carry)),
        ] {
            let r = declared_lineage(Some(declared), false, parent, policy, None).unwrap();
            assert_eq!(r["declaration"], "explicit");
            assert_eq!(r["lineage_mode"], declared.as_str());
        }
        // A declaration that disagrees with the request is refused.
        for (declared, parent, policy) in [
            (ColdInitialization, true, None),
            (WeightWarmStart, true, Some(&carry)),
            (NewCorpusTransition, true, Some(&reset)),
            (NewCorpusTransition, false, None),
        ] {
            assert!(
                declared_lineage(Some(declared), false, parent, policy, None).is_err(),
                "{declared:?} accepted"
            );
        }
        // Exact resume is an attempt property, never a lineage declaration.
        assert!(declared_lineage(Some(ExactResume), false, false, None, None).is_err());
        // A resume validates the declaration against the recorded lineage.
        let lineage = json!({"kind":"warm_start","continuation_mode":"new_corpus_transition"});
        let r = declared_lineage(
            Some(NewCorpusTransition),
            true,
            false,
            Some(&carry),
            Some(&lineage),
        )
        .unwrap();
        assert_eq!(
            (r["attempt_mode"].clone(), r["lineage_mode"].clone()),
            (json!("exact_resume"), json!("new_corpus_transition"))
        );
        assert!(
            declared_lineage(Some(WeightWarmStart), true, false, None, Some(&lineage)).is_err()
        );
        // A legacy lineage without a recorded mode stays unknown and cannot
        // be declared into certainty.
        let legacy = json!({"kind":"fresh"});
        let r = declared_lineage(None, true, false, None, Some(&legacy)).unwrap();
        assert_eq!(r["lineage_mode"], "unknown_legacy");
        assert_eq!(r["declaration"], "implicit");
        assert!(
            declared_lineage(Some(ColdInitialization), true, false, None, Some(&legacy)).is_err()
        );
        // A resume must bring its lineage record.
        assert!(declared_lineage(None, true, false, None, None).is_err());
    }

    #[test]
    fn a_representative_caller_is_unblocked_by_a_stale_temporary() {
        // AdamW::save publishes optimizer.json through write_json.
        let dir = tempfile::tempdir().unwrap();
        let pid = std::process::id();
        fs::write(dir.path().join(format!("optimizer.partial.{pid}")), b"{").unwrap();
        let vs = nn::VarStore::new(Device::Cpu);
        let w = vs.root().var("w", &[2], nn::Init::Const(1.0));
        let mut optimizer = AdamW::new(1e-2, 0.0);
        w.square().sum(Kind::Float).backward();
        optimizer.update_reference(&vs).unwrap();
        optimizer.save(&dir.path().join("optimizer.ot")).unwrap();
        let loaded = AdamW::load(&dir.path().join("optimizer.ot"), Device::Cpu).unwrap();
        assert_eq!(loaded.step, 1);
    }
}
