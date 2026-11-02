//! W1 route R8 at the teacher's receivers (round 6): each receiver consumes
//! exactly the bytes it verified. The instrumentation point
//! `manifest::after_verification` replaces a file between its digest check and
//! its consumption; the receiver must still consume the verified bytes, and a
//! caller's copy that is not the verified bytes is refused.
use super::admission_tests::{attach_completion, Fixture};
use crate::{
    corpus::ContextRow,
    label_admission as admission,
    manifest::{verification_hook, *},
    prepare::Prepared,
    teaching_store as store,
};
use serde_json::{json, Value};
use std::{fs, path::Path};

/// The strict-admission fixture of `admission_tests` (three families, one
/// frozen admission policy, completions committed by a test authority).
struct Strict {
    f: Fixture,
    expected: std::path::PathBuf,
    policy: Value,
}

fn strict() -> Strict {
    let f = Fixture::new();
    fs::copy(std::env::current_exe().unwrap(), f.root().join("evaluator")).unwrap();
    let evaluator = file_hash(&f.root().join("evaluator")).unwrap();
    let mut contexts: Value = read_json(&f.contexts()).unwrap();
    for (i, family) in ["family-a", "family-b", "family-c"].iter().enumerate() {
        let pp = f.root().join(family).join("cut-0/preprocessing.json");
        write_json(&pp,&json!({"schema":"parallel.preprocessing.v1","columns":[],"labels":[],"target_mean":0.,"target_scale":2.})).unwrap();
        let pp_sha = file_hash(&pp).unwrap();
        let prepared = f.root().join(family).join("prepared.json");
        let bytes = crate::prepare::csv_bytes(&[vec![1., 2.]]).unwrap();
        fs::write(pp.with_file_name("train.csv"), &bytes).unwrap();
        fs::write(pp.with_file_name("validation.csv"), &bytes).unwrap();
        let mut p: Value = read_json(&prepared).unwrap();
        p["cuts"][0]["preprocessing_sha256"] = json!(pp_sha);
        p["cuts"][0]["training_sha256"] = json!(hash(&bytes));
        p["cuts"][0]["validation_sha256"] = json!(hash(&bytes));
        write_json(&prepared, &p).unwrap();
        let psha = file_hash(&prepared).unwrap();
        contexts["encodings"][i]["prepared_sha256"] = json!(psha);
        let encoding = f.root().join(family).join("context.json");
        let mut e: Value = read_json(&encoding).unwrap();
        e["prepared_sha256"] = json!(psha);
        write_json(&encoding, &e).unwrap();
        let path = f.ledger(family);
        let mut rows: Vec<Value> = read_json(&path).unwrap();
        for transform in ["standardize", "robust_scale"] {
            let mut row = rows[0].clone();
            let program = format!("(pipe X ({transform}) (rank) (ridge))");
            row["genome"] = json!(program);
            row["program_sha256"] = json!(hash(program.as_bytes()));
            rows.push(row);
        }
        for row in &mut rows {
            row["evaluation_origin"] = json!("execution");
            row["metrics"]["evaluation_origin"] = json!("execution");
            row["task"] = json!("regression");
            row["prepared_sha256"] = json!(psha);
            row["evaluator_sha256"] = json!(evaluator);
            row["training_sha256"] = json!(hash(&bytes));
            row["validation_sha256"] = json!(hash(&bytes));
            row["evaluator_protocol"] = json!(admission::PROTOCOL);
            row["resource_protocol"] = json!(admission::RESOURCE);
            row["preprocessing_sha256"] = json!(pp_sha);
            row["metric_units"] = json!("cut_normalized_target");
            row["metrics"]["units"] = json!("cut_normalized_target");
            row["target_scale"] = json!(2.);
            row["vocabulary_sha256"] = json!(file_hash(&f.vocab()).unwrap());
            row["evaluation_seed"] = json!(0);
            row["feature_view_sha256"] = json!(admission::raw_view(&hash(&bytes), &pp_sha));
            row["evaluation_identity_sha256"] = json!(hash(
                &serde_json::to_vec(&admission::identity(row)).unwrap()
            ));
        }
        write_json(&path, &rows).unwrap();
        f.strict_receipt(&path, &rows[0]);
    }
    write_json(&f.contexts(), &contexts).unwrap();
    let expected = f.root().join("admission.json");
    let authority: Value = read_json(&f.root().join("test-authority/authority.json")).unwrap();
    let rows: Vec<Value> = read_json(&f.ledger("family-a")).unwrap();
    let completion = admission::artifact(f.root(), &rows[1]["clean_completion"]).unwrap();
    let native = admission::artifact(f.root(), &completion["fit_completion"]).unwrap();
    write_json(&expected,&json!({"schema":"parallel.label_admission.v2","evaluator":"evaluator",
        "evaluator_sha256":evaluator,"evaluator_protocol":admission::PROTOCOL,
        "authority_root":"test-authority","authority_witness":"test-witness","authority_instance":authority["instance"],
        "runtime_sha256":native["fit_identity"]["runtime_sha256"],
        "numerical_contract_sha256":native["fit_identity"]["numerical_contract_sha256"]})).unwrap();
    let policy = read_json(&expected).unwrap();
    Strict {
        f,
        expected,
        policy,
    }
}

/// Family A's context row, its preparation parsed from the bytes on disk and
/// its ledger rows.
fn family_a(s: &Strict) -> (ContextRow, Prepared, Vec<Value>) {
    let contexts: Value = read_json(&s.f.contexts()).unwrap();
    let context: ContextRow = serde_json::from_value(contexts["encodings"][0].clone()).unwrap();
    let prepared: Prepared =
        serde_json::from_slice(&fs::read(s.f.root().join("family-a/prepared.json")).unwrap())
            .unwrap();
    let rows = read_json(&s.f.ledger("family-a")).unwrap();
    (context, prepared, rows)
}

/// Replace `target` (new inode, via rename) with `bytes` the first time a
/// receiver reports it verified.
fn replace_on_verification(target: &Path, bytes: Vec<u8>) -> verification_hook::Guard {
    let target = fs::canonicalize(target).unwrap();
    let mut done = false;
    verification_hook::install(move |path| {
        if !done && fs::canonicalize(path).ok().as_deref() == Some(target.as_path()) {
            done = true;
            let staged = target.with_extension("replacement");
            fs::write(&staged, &bytes).unwrap();
            fs::rename(&staged, &target).unwrap();
        }
    })
}

fn failure(result: anyhow::Result<impl Sized>) -> String {
    match result {
        Ok(_) => "accepted".into(),
        Err(e) => format!("{e:#}"),
    }
}

/// Family A's admission receiver with its cut verified once, before any
/// replacement is armed (`verify_inputs` itself verifies the same state).
struct ScaleCase {
    s: Strict,
    context: ContextRow,
    prepared: Prepared,
    rows: Vec<Value>,
    vocabulary: String,
    training: String,
    state: std::path::PathBuf,
}

fn scale_case() -> ScaleCase {
    let s = strict();
    let (context, prepared, rows) = family_a(&s);
    let vocabulary = file_hash(&s.f.vocab()).unwrap();
    let training = admission::verify_inputs(s.f.root(), &context, &prepared).unwrap();
    let state = s.f.root().join("family-a/cut-0/preprocessing.json");
    ScaleCase {
        s,
        context,
        prepared,
        rows,
        vocabulary,
        training,
        state,
    }
}

impl ScaleCase {
    fn validate(&self, row: &Value) -> anyhow::Result<()> {
        let root = self.s.f.root();
        let mut authority = admission::approved_authority(root, &self.s.policy)?;
        admission::validate_preverified(
            root,
            row,
            &self.context,
            &self.prepared,
            crate::label_admission::AdmissionInputs {
                vocabulary_sha: &self.vocabulary,
                expected: &self.s.policy,
                fit_training: &self.training,
            },
            &mut authority,
        )
    }
}

/// A well-formed preparation state with another target scale.
fn rescaled() -> Vec<u8> {
    serde_json::to_vec(&json!({"schema":"parallel.preprocessing.v1","columns":[],
        "labels":[],"target_mean":0.,"target_scale":3.}))
    .unwrap()
}

fn publish(s: &Strict, label: &str) -> rusqlite::Connection {
    let root = s.f.root();
    let cache = root.join(format!("runs/{label}/store"));
    store::publish_admitted(
        root,
        crate::teaching_store::PublishRequest {
            contexts_path: &s.f.contexts(),
            vocabulary: &s.f.vocab(),
            sources: &s.f.observations(),
            output: &root.join(label),
            store: &cache,
            qualification: true,
        },
        &s.expected,
        |_, _, _| Ok(()),
    )
    .unwrap();
    rusqlite::Connection::open(cache.join("store.sqlite")).unwrap()
}

/// The committed search-receipt digest of family A's ledger and the number of
/// labels admitted for its context.
fn family_a_state(db: &rusqlite::Connection) -> (String, usize) {
    let digest = db
        .query_row(
            "SELECT search_digest FROM sources WHERE path LIKE '%family-a/observations.json'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    let labels = db
        .query_row("SELECT count(*) FROM programs WHERE context=0", [], |r| {
            r.get(0)
        })
        .unwrap();
    (digest, labels)
}

/// A published strict corpus and its directory.
fn published(label: &str) -> (Strict, std::path::PathBuf) {
    let s = strict();
    publish(&s, label);
    let corpus = s.f.root().join(label);
    (s, corpus)
}

/// The first shard of a published corpus, its `shard.json` path, the bound
/// bytes and a well-formed manifest that disagrees with the binding.
fn first_shard(s: &Strict, corpus: &Path) -> (store::Shard, std::path::PathBuf, Vec<u8>, Vec<u8>) {
    let m: store::Manifest = read_json(&corpus.join("index.json")).unwrap();
    let shard = m.shards[0].clone();
    let path = resolve(s.f.root(), &shard.path).unwrap().join("shard.json");
    let good = fs::read(&path).unwrap();
    let mut bad: Value = serde_json::from_slice(&good).unwrap();
    bad["max_length"] = json!(shard.max_length + 1);
    (shard, path, good, serde_json::to_vec(&bad).unwrap())
}

/// A published corpus, the caller's verified index of it, the bytes of the
/// index its provenance names, and an internally consistent alternative
/// index whose first shard's labels name another program.
struct LabelCase {
    s: Strict,
    corpus: std::path::PathBuf,
    index: crate::corpus::Index,
    bound: Vec<u8>,
    forged: Vec<u8>,
}

fn label_case(label: &str) -> LabelCase {
    let (s, corpus) = published(label);
    let root = s.f.root();
    let index = store::read_index(root, &corpus).unwrap();
    let bound = fs::read(corpus.join("index.json")).unwrap();
    let mut m: Value = serde_json::from_slice(&bound).unwrap();
    let shard = resolve(root, Path::new(m["shards"][0]["path"].as_str().unwrap())).unwrap();
    let labels = fs::read_to_string(shard.join("labels.jsonl")).unwrap();
    let forged_labels = labels
        .lines()
        .map(|line| {
            let mut row: Value = serde_json::from_str(line).unwrap();
            row["program"] = json!("(pipe X (forged) (ridge))");
            format!("{}\n", serde_json::to_string(&row).unwrap())
        })
        .collect::<String>();
    fs::create_dir_all(root.join("forged-shard")).unwrap();
    fs::write(root.join("forged-shard/labels.jsonl"), &forged_labels).unwrap();
    m["shards"][0]["path"] = json!("forged-shard");
    m["shards"][0]["labels_sha256"] = json!(hash(forged_labels.as_bytes()));
    let forged = serde_json::to_vec_pretty(&m).unwrap();
    LabelCase {
        s,
        corpus,
        index,
        bound,
        forged,
    }
}

impl LabelCase {
    fn programs(&self) -> Result<Vec<String>, String> {
        store::teacher_programs(self.s.f.root(), &self.corpus, &self.index, 10)
            .map(|p| p.into_values().flatten().collect())
            .map_err(|e| format!("{e:#}"))
    }
}

fn forged(programs: &[String]) -> bool {
    programs.iter().any(|p| p.contains("forged"))
}

// ---------------------------------------------------------------------------
// W1 route R8 at the corpus vocabulary readers after `load` (round 8): the
// native loop's candidate expansion consumes exactly the vocabulary bytes the
// verified corpus index binds; `vocabulary.json` is never reopened unbound.
// ---------------------------------------------------------------------------

/// Another well-formed vocabulary: the given one with two tokens swapped.
fn permuted_vocabulary(path: &Path) -> Vec<u8> {
    let mut v: Value = read_json(path).unwrap();
    v["tokens"].as_array_mut().unwrap().swap(0, 1);
    serde_json::to_vec_pretty(&v).unwrap()
}

fn vocabulary_tokens(root: &Path, corpus: &Path) -> Result<Vec<String>, String> {
    store::read_index_and_vocabulary(root, corpus)
        .map(|(_, vocab)| vocab.tokens)
        .map_err(|e| format!("{e:#}"))
}

// ---------------------------------------------------------------------------
// W2-T04 (G-11c): the resident teaching view is keyed by the corpus it was
// loaded from. Another corpus directory with the same index bytes but a
// changed payload is verified, never served from the cache.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// W1 round 8: a legacy (`parallel.teaching_corpus.v1`) corpus's teacher labels
// are authenticated by a labels digest the verified index binds.
// `provenance.json` is bound only by its own back-reference to the index, so
// it never supplies that digest; without the binding the labels are unbound.
// ---------------------------------------------------------------------------

/// A published corpus exported to the legacy layout (`teaching_store::export`).
fn legacy_export(label: &str) -> (Strict, std::path::PathBuf) {
    let (s, corpus) = published(label);
    let legacy = s.f.root().join(format!("{label}-legacy"));
    store::export(s.f.root(), &corpus, &legacy).unwrap();
    (s, legacy)
}

fn legacy_programs(root: &Path, legacy: &Path) -> Result<Vec<String>, String> {
    let index = store::read_index(root, legacy).map_err(|e| format!("{e:#}"))?;
    store::teacher_programs(root, legacy, &index, 10)
        .map(|p| p.into_values().flatten().collect())
        .map_err(|e| format!("{e:#}"))
}

use super::verify_shard_metadata;

#[path = "read_once_tests/label_tests.rs"]
mod label_tests;
#[path = "read_once_tests/receipt_tests.rs"]
mod receipt_tests;
#[path = "read_once_tests/scale_tests.rs"]
mod scale_tests;
