#![cfg(feature = "neural")]
//! W6-T01: freeze resolved training inputs (CPU only, tiny fixture corpus).
//!
//! The corpus is a real `parallel.teaching_corpus.v1` directory loaded by the
//! real teaching store; staging uses the same functions `rank_impl` uses
//! before moving tensors to a device. No device, model or training run.

use jope_parallel::{
    corpus::{ContextRow, Group, Index, Record, RECORD_BYTES, SEQUENCE},
    manifest::{file_hash, hash, write_json},
    teaching_store,
    training::inputs,
};
use serde_json::{json, Value};
use std::{
    fs,
    path::{Path, PathBuf},
};

struct Fixture {
    _dir: tempfile::TempDir,
    root: PathBuf,
    corpus: PathBuf,
}

fn context_bytes(c: usize) -> Vec<u8> {
    (0..2048)
        .flat_map(|i| (((i + 7 * c) % 97) as f32 * 0.01).to_le_bytes())
        .collect()
}

/// Program `k` of every group: BOS, one body token, EOS. Both groups hold the
/// same eight programs, so 16 available records are 8 unique programs.
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
    r.legal[1] = 1;
    r.legal[2] = 2;
    r
}

fn vocabulary_len(corpus: &Path) -> usize {
    let vocabulary: Value =
        serde_json::from_slice(&fs::read(corpus.join("vocabulary.json")).unwrap()).unwrap();
    vocabulary["tokens"].as_array().unwrap().len()
}

fn build_index(corpus: &Path, root: &Path, contexts: usize) -> Index {
    let rows = (0..contexts)
        .map(|c| {
            let path = corpus.join(format!("context-{c}.bin"));
            ContextRow {
                dataset: format!("dataset-{c}"),
                family: format!("family-{c}"),
                cut: 0,
                partition: "meta_train".into(),
                path: path.strip_prefix(root).unwrap().to_path_buf(),
                sha256: file_hash(&path).unwrap(),
                prepared: format!("prepared-{c}").into(),
                prepared_sha256: "p".repeat(64),
            }
        })
        .collect();
    Index {
        schema: "parallel.teaching_corpus.v1".into(),
        qualification: true,
        vocabulary_sha256: file_hash(&corpus.join("vocabulary.json")).unwrap(),
        context_index_sha256: "declared-context-index".into(),
        records_sha256: file_hash(&corpus.join("records.bin")).unwrap(),
        record_bytes: RECORD_BYTES,
        records: 16,
        legal_sets: vec![vec![0], (3..11).collect(), vec![2]],
        contexts: rows,
        groups: (0..2)
            .map(|g| Group {
                context: g,
                family: format!("family-{g}"),
                partition: "meta_train".into(),
                records: (g * 8..g * 8 + 8).collect(),
                max_length: 3,
            })
            .collect(),
    }
}

fn fixture() -> Fixture {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().canonicalize().unwrap();
    let corpus = root.join("corpus");
    fs::create_dir_all(&corpus).unwrap();
    fs::copy(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/grammar/featmath.json"),
        corpus.join("vocabulary.json"),
    )
    .unwrap();
    for c in 0..2 {
        fs::write(corpus.join(format!("context-{c}.bin")), context_bytes(c)).unwrap();
    }
    let bytes = (0..16)
        .flat_map(|i| record(i / 8, i % 8).bytes())
        .collect::<Vec<_>>();
    fs::write(corpus.join("records.bin"), bytes).unwrap();
    let index = build_index(&corpus, &root, 2);
    write_json(&corpus.join("index.json"), &index).unwrap();
    Fixture {
        _dir: dir,
        root,
        corpus,
    }
}

impl Fixture {
    fn verified_index(&self) -> (Value, String) {
        let bytes = fs::read(self.corpus.join("index.json")).unwrap();
        (serde_json::from_slice(&bytes).unwrap(), hash(&bytes))
    }
}

fn clone_index(index: &Index) -> Index {
    serde_json::from_value(serde_json::to_value(index).unwrap()).unwrap()
}

#[test]
fn cached_context_staging_is_keyed_by_content_not_by_a_declared_digest() {
    // The trainer's resident context tensor was cached under the index's
    // declared `context_index_sha256`. Two indexes declaring the same digest
    // but listing different context artifacts shared one cached tensor, so a
    // cached load could differ from a cold load.
    let f = fixture();
    let (index, _) = teaching_store::load(&f.root, &f.corpus).unwrap();
    let mut other = clone_index(&index);
    fs::write(f.corpus.join("context-9.bin"), context_bytes(9)).unwrap();
    other.contexts[1].path = f
        .corpus
        .join("context-9.bin")
        .strip_prefix(&f.root)
        .unwrap()
        .to_path_buf();
    other.contexts[1].sha256 = file_hash(&f.corpus.join("context-9.bin")).unwrap();
    assert_eq!(other.context_index_sha256, index.context_index_sha256);
    assert_ne!(
        inputs::context_identity(&other).unwrap(),
        inputs::context_identity(&index).unwrap(),
        "different context content shared one cache identity"
    );
    assert_ne!(
        inputs::stage_contexts(&f.root, &other).unwrap(),
        inputs::stage_contexts(&f.root, &index).unwrap()
    );
    // Identical content keeps one identity, so reuse remains possible.
    assert_eq!(
        inputs::context_identity(&clone_index(&index)).unwrap(),
        inputs::context_identity(&index).unwrap()
    );
}

#[test]
fn cold_and_cached_loading_stage_identical_authorized_inputs() {
    let f = fixture();
    let (verified, index_sha) = f.verified_index();
    let vocabulary = vocabulary_len(&f.corpus);
    let (cold_a, records_a) = teaching_store::load(&f.root, &f.corpus).unwrap();
    let (cold_b, records_b) = teaching_store::load(&f.root, &f.corpus).unwrap();
    let resident = teaching_store::load_resident(&f.root, &f.corpus).unwrap();
    let cached = teaching_store::load_resident(&f.root, &f.corpus).unwrap();
    assert!(std::rc::Rc::ptr_eq(&resident, &cached));
    let staged = |index: &Index, records: &teaching_store::TeachingRecords| {
        inputs::bind_resident(&verified, index).unwrap();
        (
            inputs::stage_records(index, vocabulary, |i| records.get(i)).unwrap(),
            inputs::stage_contexts(&f.root, index).unwrap(),
            inputs::digest(
                &inputs::freeze(index, &index_sha, &index.vocabulary_sha256, None).unwrap(),
            )
            .unwrap(),
        )
    };
    let a = staged(&cold_a, &records_a);
    assert_eq!(a, staged(&cold_b, &records_b));
    assert_eq!(a, staged(&resident.0, &resident.1));
    assert_eq!(a, staged(&cached.0, &cached.1));
    // Independently expected layout.
    let (records, contexts, _) = a;
    for i in 0..16 {
        let row = &records.tokens[i * SEQUENCE..(i + 1) * SEQUENCE];
        assert_eq!(&row[..3], &[1, 3 + (i % 8) as i16, 2]);
        assert!(row[3..].iter().all(|&t| t == 0));
        assert_eq!(records.context_ids[i], (i / 8) as i64);
    }
    assert_eq!(records.masks.iter().filter(|&&m| m == 1).count(), 1 + 8 + 1);
    let expected_contexts = (0..2)
        .flat_map(|c| {
            context_bytes(c)
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    assert_eq!(contexts, expected_contexts);
}

#[test]
fn changed_missing_or_substituted_inputs_are_refused_before_training() {
    type Case = Box<dyn Fn(&Fixture) -> anyhow::Result<()>>;
    let stage = |f: &Fixture| -> anyhow::Result<()> {
        let (verified, index_sha) = f.verified_index();
        let (index, records) = teaching_store::load(&f.root, &f.corpus)?;
        inputs::bind_resident(&verified, &index)?;
        let vocabulary =
            inputs::read_verified(&f.corpus.join("vocabulary.json"), &index.vocabulary_sha256)?;
        let vocabulary: Value = serde_json::from_slice(&vocabulary)?;
        let len = vocabulary["tokens"].as_array().unwrap().len();
        inputs::stage_contexts(&f.root, &index)?;
        inputs::stage_records(&index, len, |i| records.get(i))?;
        inputs::freeze(&index, &index_sha, &index.vocabulary_sha256, None)?;
        Ok(())
    };
    // Positive control.
    stage(&fixture()).unwrap();
    let rewrite_index = |f: &Fixture, edit: &dyn Fn(&mut Index)| {
        let (mut index, _) = teaching_store::load(&f.root, &f.corpus).unwrap();
        edit(&mut index);
        write_json(&f.corpus.join("index.json"), &index).unwrap();
    };
    let cases: Vec<(&str, Case)> = vec![
        (
            "context byte changed",
            Box::new(move |f| {
                let path = f.corpus.join("context-1.bin");
                let mut bytes = fs::read(&path)?;
                bytes[100] ^= 1;
                fs::write(path, bytes)?;
                stage(f)
            }),
        ),
        (
            "context missing",
            Box::new(move |f| {
                fs::remove_file(f.corpus.join("context-0.bin"))?;
                stage(f)
            }),
        ),
        (
            "context not admitted",
            Box::new(move |f| {
                rewrite_index(f, &|index| index.contexts[0].partition = "meta_test".into());
                stage(f)
            }),
        ),
        (
            "context dimensions",
            Box::new(move |f| {
                let path = f.corpus.join("context-0.bin");
                fs::write(&path, &context_bytes(0)[..4096])?;
                let digest = file_hash(&path)?;
                rewrite_index(f, &|index| index.contexts[0].sha256 = digest.clone());
                stage(f)
            }),
        ),
        (
            "records changed",
            Box::new(move |f| {
                let path = f.corpus.join("records.bin");
                let mut bytes = fs::read(&path)?;
                bytes[3] ^= 1;
                fs::write(path, bytes)?;
                stage(f)
            }),
        ),
        (
            "vocabulary changed",
            Box::new(move |f| {
                let path = f.corpus.join("vocabulary.json");
                let mut bytes = fs::read(&path)?;
                bytes.push(b' ');
                fs::write(path, bytes)?;
                stage(f)
            }),
        ),
        (
            "vocabulary missing",
            Box::new(move |f| {
                fs::remove_file(f.corpus.join("vocabulary.json"))?;
                stage(f)
            }),
        ),
        (
            "record names a context outside the corpus",
            Box::new(move |f| {
                rewrite_index(f, &|index| {
                    index.contexts.truncate(1);
                    index.groups[1].context = 0;
                });
                stage(f)
            }),
        ),
        (
            "legal set names a token outside the vocabulary",
            Box::new(move |f| {
                rewrite_index(f, &|index| index.legal_sets[1].push(100_000));
                stage(f)
            }),
        ),
    ];
    for (case, run) in cases {
        let f = fixture();
        assert!(run(&f).is_err(), "{case}: accepted");
    }
}

#[test]
fn inputs_substituted_after_verification_are_refused() {
    // Mutable path substitution: the trainer verified index.json's bytes, then
    // the resident loader read the path again. A different, valid index
    // written in between becomes the resident view.
    let f = fixture();
    let (verified, _) = f.verified_index();
    let (mut swapped, _) = teaching_store::load(&f.root, &f.corpus).unwrap();
    swapped.groups.swap(0, 1);
    write_json(&f.corpus.join("index.json"), &swapped).unwrap();
    let (resident, _) = teaching_store::load(&f.root, &f.corpus).unwrap();
    assert_ne!(
        serde_json::to_value(&resident.groups).unwrap(),
        verified["groups"],
        "the loader re-reads the path"
    );
    let error = inputs::bind_resident(&verified, &resident).unwrap_err();
    assert!(error.to_string().contains("differ"), "{error}");
    // The same pattern for the vocabulary: hashing a path and reading it again
    // accepts substituted content; reading once does not.
    let path = f.corpus.join("vocabulary.json");
    let expected = file_hash(&path).unwrap();
    let original = fs::read(&path).unwrap();
    let mut substituted: Value = serde_json::from_slice(&original).unwrap();
    substituted["tokens"][3] = json!("ENGINE=substituted");
    fs::write(&path, serde_json::to_vec(&substituted).unwrap()).unwrap();
    assert!(inputs::read_verified(&path, &expected).is_err());
    fs::write(&path, &original).unwrap();
    assert_eq!(inputs::read_verified(&path, &expected).unwrap(), original);
}

#[test]
fn the_frozen_manifest_separates_availability_programs_and_presentations() {
    let f = fixture();
    let (_, index_sha) = f.verified_index();
    let (index, records) = teaching_store::load(&f.root, &f.corpus).unwrap();
    // 16 available records, 8 unique programs (each group repeats the other's).
    let unique = inputs::unique_programs(0..16, |i| records.get(i));
    assert_eq!(unique, 8);
    let frozen =
        inputs::freeze(&index, &index_sha, &index.vocabulary_sha256, Some(unique)).unwrap();
    assert_eq!(frozen["schema"], inputs::SCHEMA);
    assert_eq!(frozen["corpus"]["index_sha256"], index_sha);
    assert_eq!(
        frozen["availability"],
        json!({"partitions":{"meta_train":{"groups":2,"records":16,"families":2}},
            "records_in_groups":16,"records_outside_groups":0})
    );
    assert_eq!(
        frozen["unique_programs"],
        json!({"status":"computed","count":8})
    );
    assert!(frozen["exposure_units"]["presentations"]
        .as_str()
        .unwrap()
        .contains("never new data"));
    assert_eq!(frozen["revocation"], "not_available_in_this_source");
    // Launch-time freezing never estimates unique programs.
    let launch = inputs::freeze(&index, &index_sha, &index.vocabulary_sha256, None).unwrap();
    assert_eq!(
        launch["unique_programs"],
        json!({"status":"not_computed_at_launch","count":null})
    );
    // The digest is stable and moves with any resolved identity.
    assert_eq!(
        inputs::digest(&launch).unwrap(),
        inputs::digest(
            &inputs::freeze(&index, &index_sha, &index.vocabulary_sha256, None).unwrap()
        )
        .unwrap()
    );
    let mut other = clone_index(&index);
    other.contexts[0].sha256 = "0".repeat(64);
    assert_ne!(
        inputs::digest(
            &inputs::freeze(&other, &index_sha, &other.vocabulary_sha256, None).unwrap()
        )
        .unwrap(),
        inputs::digest(&launch).unwrap()
    );
    // A vocabulary other than the corpus's is refused.
    assert!(inputs::freeze(&index, &index_sha, &"f".repeat(64), None).is_err());
}
