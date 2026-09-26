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
            &self.vocabulary,
            &self.s.policy,
            &self.training,
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

/// Route R8 at `label_admission::validate_preverified` (label_admission.rs
/// :475-481 before the fix): the target scale an observation is admitted with
/// is read from exactly the preparation state whose digest was checked.
#[test]
fn an_honest_scale_is_admitted_against_the_verified_state_even_if_replaced_after_its_check() {
    let c = scale_case();
    // Positive control.
    c.validate(&c.rows[1]).unwrap();
    let _hook = replace_on_verification(&c.state, rescaled());
    let admitted = c.validate(&c.rows[1]).map_err(|e| format!("{e:#}"));
    assert_eq!(
        admitted,
        Ok(()),
        "the admission consumed a preparation state other than the verified one"
    );
}

/// The same receiver, other direction: an observation claiming the
/// replacement's scale, with its own committed completion, is refused with or
/// without the replacement.
#[test]
fn a_forged_scale_is_refused_even_if_the_state_is_replaced_after_its_check() {
    let c = scale_case();
    let mut forged = c.rows[1].clone();
    forged["target_scale"] = json!(3.);
    forged["evaluation_identity_sha256"] = json!(hash(
        &serde_json::to_vec(&admission::identity(&forged)).unwrap()
    ));
    attach_completion(
        c.s.f.root(),
        c.s.f.ledger("family-a").parent().unwrap(),
        900,
        &mut forged,
    );
    // Negative control: refused without a replacement.
    let refused = failure(c.validate(&forged));
    assert!(refused.contains("preprocessing/scale"), "{refused}");
    let _hook = replace_on_verification(&c.state, rescaled());
    let refused = failure(c.validate(&forged));
    assert!(
        refused.contains("preprocessing/scale"),
        "an observation was admitted with the scale of a state other than the verified one: \
         {refused}"
    );
}

/// Route R8 at `label_admission::verify_inputs` (label_admission.rs:240-266
/// before the fix), the receiver behind native_genome.rs:347-349 and the
/// teaching store's preparation map: provenance is checked against the hashed
/// preparation bytes, never against a caller's separate parse.
#[test]
fn the_teachers_input_admission_binds_the_callers_preparation_to_the_hashed_bytes() {
    let s = strict();
    let root = s.f.root();
    let (mut context, prepared, _) = family_a(&s);
    let path = root.join("family-a/prepared.json");
    // Positive control.
    admission::verify_inputs(root, &context, &prepared).unwrap();
    // The bound preparation is not a clean evaluation.
    let mut unclean: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    unclean["clean_evaluation"] = json!(false);
    let unclean = serde_json::to_vec(&unclean).unwrap();
    fs::write(&path, &unclean).unwrap();
    context.prepared_sha256 = hash(&unclean);
    // Negative control: the caller's parse of the bound bytes is refused.
    let honest: Prepared = serde_json::from_slice(&unclean).unwrap();
    let refused = failure(admission::verify_inputs(root, &context, &honest));
    assert!(refused.contains("provenance conflict"), "{refused}");
    // A caller's parse of other bytes (what native_genome.rs:347-349 held
    // after an unverified read) claims a clean evaluation.
    let refused = failure(admission::verify_inputs(root, &context, &prepared));
    assert!(
        refused.contains("differs from the bound"),
        "provenance was checked against a parse other than the hashed bytes: {refused}"
    );
}

/// The same receiver: the preparation is replaced after its digest check, and
/// the admitted training identity is still the one of the verified
/// preparation (its views are not resolved from a reopened `prepared.json`).
#[test]
fn a_preparation_replaced_after_its_check_never_decides_the_admitted_views() {
    let s = strict();
    let root = s.f.root();
    let (context, prepared, _) = family_a(&s);
    let path = root.join("family-a/prepared.json");
    // Positive control.
    let admitted = admission::verify_inputs(root, &context, &prepared).unwrap();
    let mut other: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    other["cuts"][0]["training_sha256"] = json!("0".repeat(64));
    let _hook = replace_on_verification(&path, serde_json::to_vec(&other).unwrap());
    let raced = admission::verify_inputs(root, &context, &prepared).map_err(|e| format!("{e:#}"));
    assert_eq!(
        raced,
        Ok(admitted),
        "the admitted views were resolved from a preparation other than the verified one"
    );
}

fn publish(s: &Strict, label: &str) -> rusqlite::Connection {
    let root = s.f.root();
    let cache = root.join(format!("runs/{label}/store"));
    store::publish_admitted(
        root,
        &s.f.contexts(),
        &s.f.vocab(),
        &s.f.observations(),
        &root.join(label),
        &cache,
        true,
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

/// Route R8 at the teaching store's ledger ingestion (teaching_store.rs:551,
/// 572 before the fix): the search receipt an observation is validated
/// against is the receipt whose digest the ledger commits.
#[test]
fn a_search_receipt_is_validated_from_the_bytes_its_ledger_commits() {
    let opened = |valid: &[u8]| {
        let mut receipt: Value = serde_json::from_slice(valid).unwrap();
        receipt["final_tests_opened"] = json!(true);
        serde_json::to_vec(&receipt).unwrap()
    };
    // Positive control: nothing replaced.
    let s = strict();
    let search = s.f.ledger("family-a").with_file_name("search.json");
    let valid = fs::read(&search).unwrap();
    let (digest, labels) = family_a_state(&publish(&s, "control"));
    assert_eq!(digest, hash(&valid));
    assert!(labels > 0);

    // The committed receipt opened final tests; a valid receipt replaces it
    // after its digest was taken. No label may be admitted under it.
    let s = strict();
    let search = s.f.ledger("family-a").with_file_name("search.json");
    let valid = fs::read(&search).unwrap();
    let committed = opened(&valid);
    fs::write(&search, &committed).unwrap();
    let db = {
        let _hook = replace_on_verification(&search, valid);
        publish(&s, "opened")
    };
    let (digest, labels) = family_a_state(&db);
    assert_eq!(digest, hash(&committed));
    assert_eq!(
        labels, 0,
        "labels were admitted under a search receipt other than the one the ledger commits"
    );

    // The other direction: a valid committed receipt replaced by one that
    // opened final tests. The committed receipt decides.
    let s = strict();
    let search = s.f.ledger("family-a").with_file_name("search.json");
    let valid = fs::read(&search).unwrap();
    let db = {
        let _hook = replace_on_verification(&search, opened(&valid));
        publish(&s, "valid")
    };
    let (digest, labels) = family_a_state(&db);
    assert_eq!(digest, hash(&valid));
    assert!(
        labels > 0,
        "the committed valid receipt was not the one the admission validated"
    );
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

/// Route R8 at `teaching_store::verify_shard_metadata` (:100-106 before the
/// fix): the shard binding is checked on exactly the verified `shard.json`.
#[test]
fn an_honest_shard_manifest_replaced_after_its_check_is_still_accepted() {
    let (s, corpus) = published("shard-honest");
    let (shard, path, _, bad) = first_shard(&s, &corpus);
    // Positive control.
    super::verify_shard_metadata(s.f.root(), &shard).unwrap();
    let _hook = replace_on_verification(&path, bad);
    let checked = super::verify_shard_metadata(s.f.root(), &shard)
        .map(|_| ())
        .map_err(|e| format!("{e:#}"));
    assert_eq!(
        checked,
        Ok(()),
        "the binding was checked on a shard manifest other than the verified one"
    );
}

/// The same receiver, other direction: a bound `shard.json` that disagrees
/// with its binding is refused, even if a matching one replaces it after the
/// digest check.
#[test]
fn a_shard_manifest_disagreeing_with_its_binding_is_refused_even_if_replaced() {
    let (s, corpus) = published("shard-forged");
    let (shard, path, good, bad) = first_shard(&s, &corpus);
    fs::write(&path, &bad).unwrap();
    let mut forged = shard.clone();
    forged.sha256 = hash(&bad);
    // Negative control: refused without a replacement.
    let refused = failure(super::verify_shard_metadata(s.f.root(), &forged));
    assert!(refused.contains("shard binding changed"), "{refused}");
    let _hook = replace_on_verification(&path, good);
    let refused = failure(super::verify_shard_metadata(s.f.root(), &forged));
    assert!(
        refused.contains("shard binding changed"),
        "a shard manifest that disagrees with its binding was accepted: {refused}"
    );
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

/// Route R8 at `teaching_store::teacher_programs` (:1243-1248 before the
/// fix): the index whose labels are read is the index whose digest the
/// provenance names, read once.
#[test]
fn teacher_labels_come_from_the_named_index_even_if_it_is_replaced_after_its_read() {
    let c = label_case("labels-honest");
    // Positive control.
    let programs = c.programs().unwrap();
    assert!(!programs.is_empty() && !forged(&programs));
    let _hook = replace_on_verification(&c.corpus.join("index.json"), c.forged.clone());
    let raced = c.programs();
    assert!(
        matches!(&raced, Ok(p) if !p.is_empty() && !forged(p)),
        "the labels were read from an index other than the one read and checked: {raced:?}"
    );
}

/// The same receiver, other direction: an index the provenance does not name
/// is refused, even if the named index replaces it after its read.
#[test]
fn teacher_labels_of_an_unnamed_index_are_refused_even_if_it_is_replaced_after_its_read() {
    let c = label_case("labels-forged");
    fs::write(c.corpus.join("index.json"), &c.forged).unwrap();
    // Negative control: refused without a replacement.
    let refused = c.programs().unwrap_err();
    assert!(refused.contains("label index changed"), "{refused}");
    let _hook = replace_on_verification(&c.corpus.join("index.json"), c.bound.clone());
    let raced = c.programs();
    assert!(
        matches!(&raced, Err(e) if e.contains("label index changed")),
        "teacher labels of an index the provenance does not name were returned: {raced:?}"
    );
}

/// The same receiver, bound to its caller (`load_resident_verified`,
/// `read_index`): the labels read are those of the index the caller verified.
#[test]
fn teacher_labels_are_those_of_the_callers_verified_index() {
    let mut c = label_case("labels-caller");
    // Positive control.
    assert!(!forged(&c.programs().unwrap()));
    c.index.records_sha256 = "0".repeat(64);
    let refused = c.programs();
    assert!(
        matches!(&refused, Err(e) if e.contains("label index changed")),
        "labels were read for an index other than the caller's verified index: {refused:?}"
    );
}
