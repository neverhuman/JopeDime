//! Append-only teaching ledgers and immutable, independently reusable groups.
//! The v1 mode retains reference label semantics; strict production uses original-precision midranks with original execution evidence.
use crate::{
    corpus::{self, ContextRow, Group, Index, Record, Records, RECORD_BYTES, SEQUENCE},
    encoding::Encoding,
    grammar::{State, Vocab},
    manifest::*,
    prepare::Prepared,
};
use anyhow::{ensure, Context, Result};
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};

pub const SCHEMA: &str = "parallel.teaching_shards.v1";
type Programs = BTreeMap<String, (Vec<usize>, f64, f64, String)>;

#[derive(Clone, Serialize, Deserialize)]
pub struct Shard {
    pub context: usize,
    pub path: PathBuf,
    pub sha256: String,
    pub records: usize,
    pub max_length: usize,
    pub records_sha256: String,
    pub labels_sha256: String,
    pub legal_sets: Vec<Vec<usize>>,
}

#[derive(Serialize, Deserialize)]
pub struct Manifest {
    pub schema: String,
    pub qualification: bool,
    pub label_policy: String,
    #[serde(default)]
    pub semantic_dependencies_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub admission_policy: Option<Value>,
    pub vocabulary_sha256: String,
    pub context_index_sha256: String,
    pub contexts: Vec<ContextRow>,
    pub records: usize,
    pub record_bytes: usize,
    pub records_sha256: String,
    pub groups_sha256: String,
    pub shards: Vec<Shard>,
    pub sources: Vec<Value>,
    pub counts: BTreeMap<String, usize>,
}

/// First-label freeze: already-admitted programs keep value/percentile.
pub fn frozen_z_first_labels(
    rmses: &[f64],
    previous: &BTreeMap<String, (f32, f32)>,
    programs: &[String],
) -> (Vec<f32>, Vec<f32>, usize) {
    assert_eq!(rmses.len(), programs.len());
    assert!(rmses.len() >= 2 && rmses.windows(2).all(|w| w[0] <= w[1]));
    let median = rmses[rmses.len() / 2].max(1e-8);
    let n = rmses.len();
    let mut values = Vec::with_capacity(n);
    let mut percentiles = Vec::with_capacity(n);
    let mut frozen = 0usize;
    for (i, program) in programs.iter().enumerate() {
        if let Some(&(value, percentile)) = previous.get(program) {
            values.push(value);
            percentiles.push(percentile);
            frozen += 1;
        } else {
            values.push((1. / (1. + rmses[i] / median)) as f32);
            percentiles.push(1. - i as f32 / (n - 1) as f32);
        }
    }
    (values, percentiles, frozen)
}

pub fn midranks(scores: &[f64]) -> Vec<f32> {
    assert!(scores.len() >= 2 && scores.windows(2).all(|w| w[0] <= w[1]));
    let mut targets = vec![0.; scores.len()];
    let mut start = 0;
    while start < scores.len() {
        let mut end = start + 1;
        while end < scores.len() && scores[start] == scores[end] {
            end += 1;
        }
        let target = (1.0 - (start + end - 1) as f64 / 2.0 / (scores.len() - 1) as f64) as f32;
        targets[start..end].fill(target);
        start = end;
    }
    targets
}

fn verify_shard_metadata(root: &Path, shard: &Shard) -> Result<PathBuf> {
    let path = resolve(root, &shard.path)?;
    // Route R8: the binding is checked on exactly the verified manifest bytes.
    let saved: Shard = serde_json::from_slice(&read_verified(
        &path.join("shard.json"),
        &shard.sha256,
        "shard manifest",
    )?)?;
    // The manifest's own digest is carried by its parent, avoiding self-hashing.
    let mut expected = shard.clone();
    expected.sha256.clear();
    ensure!(
        serde_json::to_value(saved)? == serde_json::to_value(expected)?,
        "shard binding changed"
    );
    Ok(path)
}

fn verify_shard(root: &Path, shard: &Shard) -> Result<PathBuf> {
    let path = verify_shard_metadata(root, shard)?;
    ensure!(
        file_hash(&path.join("records.bin"))? == shard.records_sha256
            && file_hash(&path.join("labels.jsonl"))? == shard.labels_sha256
            && path.join("records.bin").metadata()?.len() == (shard.records * RECORD_BYTES) as u64,
        "shard content changed"
    );
    Ok(path)
}

/// Program, value and ranking target assigned when packing a shard.
type AssignedLabels = Vec<(String, f32, f32)>;

#[derive(Clone, Copy)]
struct GroupLabels<'a> {
    pub vocab: &'a Vocab,
    pub binding: &'a str,
    pub policy: &'a str,
    pub lineage: &'a BTreeMap<String, Value>,
    pub previous: &'a BTreeMap<String, (f32, f32)>,
}

fn pack_group(
    root: &Path,
    store: &Path,
    context: usize,
    programs: Programs,
    labels: GroupLabels<'_>,
) -> Result<(Shard, AssignedLabels)> {
    let GroupLabels {
        vocab,
        binding,
        policy,
        lineage,
        previous,
    } = labels;

    let key = hash(&serde_json::to_vec(&(
        binding, context, &programs, lineage,
    ))?);
    let output = store.join("shards").join(&key);
    if output.is_dir() {
        let mut shard: Shard = read_json(&output.join("shard.json"))?;
        shard.sha256 = file_hash(&output.join("shard.json"))?;
        verify_shard(root, &shard)?;
        return Ok((shard, Vec::new()));
    }
    let staging = store
        .join("shards")
        .join(format!(".partial-{key}-{}", std::process::id()));
    // Staging is disposable derived data. Committed names are never overwritten.
    if staging.exists() {
        fs::remove_dir_all(&staging)?;
    }
    fs::create_dir_all(&staging)?;
    let mut file = BufWriter::new(fs::File::create(staging.join("records.bin"))?);
    let mut labels = BufWriter::new(fs::File::create(staging.join("labels.jsonl"))?);
    let mut rows: Vec<_> = programs.into_iter().collect();
    rows.sort_by(|a, b| a.1 .1.total_cmp(&b.1 .1).then(a.0.cmp(&b.0)));
    let program_ids: Vec<String> = rows.iter().map(|r| r.0.clone()).collect();
    let rmses: Vec<f64> = rows.iter().map(|r| r.1 .1).collect();
    let median = rmses[rmses.len() / 2].max(1e-8);
    let (values, targets) = if policy == "frozen_z_v1" {
        let (values, targets, _) = frozen_z_first_labels(&rmses, previous, &program_ids);
        (values, targets)
    } else if policy != "reference_v1" {
        (
            (0..rows.len())
                .map(|i| (1. / (1. + rmses[i] / median)) as f32)
                .collect(),
            midranks(&rmses),
        )
    } else {
        (
            (0..rows.len())
                .map(|i| (1. / (1. + rmses[i] / median)) as f32)
                .collect(),
            (0..rows.len())
                .map(|i| 1. - i as f32 / (rows.len() - 1) as f32)
                .collect(),
        )
    };
    let mut sets = BTreeMap::from([(vec![0], 0)]);
    let mut legal_sets = vec![vec![0]];
    let mut transitions = BTreeMap::<Vec<usize>, [i64; SEQUENCE]>::new();
    let mut max_length = 0;
    let count = rows.len();
    for (i, (program, (tokens, rmse, runtime, bias))) in rows.into_iter().enumerate() {
        crate::campaign::check_deadline()?;
        let mut record = Record {
            tokens: [0; SEQUENCE],
            legal: [0; SEQUENCE],
            length: tokens.len(),
            context,
            value: values[i],
            percentile: targets[i],
            runtime: runtime.ln_1p() as f32,
        };
        for (slot, &token) in record.tokens.iter_mut().zip(&tokens) {
            *slot = token as i64;
        }
        if let Some(legal) = transitions.get(&tokens) {
            record.legal = *legal;
        } else {
            let mut state = State::default();
            state.sampling = false;
            for (j, &token) in tokens.iter().enumerate().skip(1) {
                let legal = state.legal(vocab);
                let id = *sets.entry(legal.clone()).or_insert_with(|| {
                    let id = legal_sets.len();
                    legal_sets.push(legal);
                    id
                });
                record.legal[j] = id as i64;
                state.advance(vocab, token)?;
            }
            if transitions.len() < 65536 {
                transitions.insert(tokens, record.legal);
            }
        }
        ensure!(
            legal_sets.len() < u16::MAX as usize,
            "too many shard grammar states"
        );
        file.write_all(&record.bytes())?;
        let mut label = json!({"record":i,"context":context,"program":program,
            "rmse_cut_normalized":rmse,"runtime_seconds":runtime,"evaluation_bias":bias});
        if policy == "frozen_z_v1" {
            label["value"] = json!(values[i]);
            label["percentile"] = json!(targets[i]);
            label["frozen_z"] = json!(previous.contains_key(&program));
        }
        if policy == "strict_midrank_v4" {
            label["admission"] = lineage
                .get(&program)
                .context("accepted program source receipt absent")?
                .clone();
        }
        serde_json::to_writer(&mut labels, &label)?;
        labels.write_all(b"\n")?;
        max_length = max_length.max(record.length);
    }
    file.flush()?;
    file.get_ref().sync_all()?;
    labels.flush()?;
    labels.get_ref().sync_all()?;
    let mut shard = Shard {
        context,
        path: output.strip_prefix(root)?.into(),
        sha256: String::new(),
        records: count,
        max_length,
        records_sha256: file_hash(&staging.join("records.bin"))?,
        labels_sha256: file_hash(&staging.join("labels.jsonl"))?,
        legal_sets,
    };
    write_json(&staging.join("shard.json"), &shard)?;
    shard.sha256 = file_hash(&staging.join("shard.json"))?;
    fs::File::open(&staging)?.sync_all()?;
    fs::rename(&staging, &output)?;
    fs::File::open(output.parent().unwrap())?.sync_all()?;
    let assigned = program_ids
        .into_iter()
        .zip(values.into_iter().zip(targets))
        .map(|(p, (v, t))| (p, v, t))
        .collect();
    Ok((shard, assigned))
}

/// Immutable inputs and destinations for one teaching publication.
#[derive(Clone, Copy)]
pub struct PublishRequest<'a> {
    pub contexts_path: &'a Path,
    pub vocabulary: &'a Path,
    pub sources: &'a [PathBuf],
    pub output: &'a Path,
    pub store: &'a Path,
    pub qualification: bool,
}

/// Source order is an explicit append-only contract for legacy first-label
/// semantics. Mutated, removed or reordered receipts require a new store.
pub fn publish(
    root: &Path,
    request: PublishRequest<'_>,
    policy: &str,
    progress: impl FnMut(&str, usize, Option<usize>) -> Result<()>,
) -> Result<()> {
    ensure!(
        request.qualification,
        "production publication requires a frozen strict admission manifest"
    );
    publish_impl(root, request, policy, None, progress)
}

pub fn publish_admitted(
    root: &Path,
    request: PublishRequest<'_>,
    admission: &Path,
    progress: impl FnMut(&str, usize, Option<usize>) -> Result<()>,
) -> Result<()> {
    publish_impl(
        root,
        request,
        "strict_midrank_v4",
        Some(admission),
        progress,
    )
}

fn publish_impl(
    root: &Path,
    request: PublishRequest<'_>,
    policy: &str,
    admission: Option<&Path>,
    mut progress: impl FnMut(&str, usize, Option<usize>) -> Result<()>,
) -> Result<()> {
    let PublishRequest {
        contexts_path,
        vocabulary,
        sources,
        output,
        store,
        qualification,
    } = request;

    ensure!(
        [
            "reference_v1",
            "midrank_v2",
            "strict_midrank_v4",
            "frozen_z_v1"
        ]
        .contains(&policy),
        "unknown label policy"
    );
    ensure!(
        (policy == "strict_midrank_v4") == admission.is_some(),
        "strict admission requires a frozen evaluator manifest"
    );
    let root = root.canonicalize()?;
    let store = resolve(
        &root,
        if store.is_absolute() {
            store.strip_prefix(&root)?
        } else {
            store
        },
    )?;
    let output = resolve(
        &root,
        if output.is_absolute() {
            output.strip_prefix(&root)?
        } else {
            output
        },
    )?;
    ensure!(
        store.starts_with(root.join("runs"))
            && !output.starts_with(&store)
            && !store.starts_with(&output),
        "store/output overlap or non-private store"
    );
    fs::create_dir_all(store.join("shards"))?;
    let lock = fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(false)
        .open(store.join("writer.lock"))?;
    ensure!(lock.try_lock().is_ok(), "teaching store already owned");
    let context_bytes = fs::read(contexts_path)?;
    let ci: Value = serde_json::from_slice(&context_bytes)?;
    ensure!(
        ci["schema"] == "parallel.context_index.v1"
            && (qualification || ci["qualification"] == false),
        "context qualification mismatch"
    );
    let contexts: Vec<ContextRow> = serde_json::from_value(ci["encodings"].clone())?;
    let vocab_bytes = fs::read(vocabulary)?;
    let vocab: Vocab = serde_json::from_slice(&vocab_bytes)?;
    let expected: Option<Value> = admission.map(read_json).transpose()?;
    let mut authority = expected
        .as_ref()
        .map(|p| crate::label_admission::approved_authority(&root, p))
        .transpose()?;
    if let Some(expected) = &expected {
        ensure!(
            expected["schema"] == crate::label_admission::POLICY_SCHEMA
                && expected["evaluator_protocol"] == crate::label_admission::PROTOCOL,
            "unknown admission policy"
        );
        let executable = resolve(
            &root,
            Path::new(
                expected["evaluator"]
                    .as_str()
                    .context("admission evaluator absent")?,
            ),
        )?;
        ensure!(
            file_hash(&executable)? == expected["evaluator_sha256"],
            "admission evaluator changed"
        );
    }
    ensure!(
        vocab.tokens.len() < u16::MAX as usize,
        "vocabulary too large"
    );
    // The loaded packer binds transitive parser/native dependencies as well as
    // these reviewable source inputs. Any changed implementation gets a lineage.
    let packer_executable_sha256 = file_hash(Path::new("/proc/self/exe"))?;
    let semantic_sources = concat!(
        include_str!("corpus.rs"),
        include_str!("grammar.rs"),
        include_str!("grammar/automaton.rs"),
        include_str!("teaching_store.rs"),
        include_str!("task_contract.rs"),
        include_str!("prepare.rs"),
        include_str!("encoding.rs"),
        include_str!("manifest.rs"),
        include_str!("label_admission.rs"),
        include_str!("eval_registry.rs"),
        include_str!("eval_registry/authority.rs"),
        include_str!("eval_registry/service.rs"),
        include_str!("fit_boundary.rs")
    );
    let semantic_dependencies_sha256 = hash(&serde_json::to_vec(&(
        hash(semantic_sources.as_bytes()),
        packer_executable_sha256,
    ))?);
    let binding = hash(&serde_json::to_vec(&(
        hash(&context_bytes),
        hash(&vocab_bytes),
        qualification,
        policy,
        &expected,
        &semantic_dependencies_sha256,
    ))?);
    let mut prepared = BTreeMap::new();
    let mut by_cut = BTreeMap::new();
    for (i, c) in contexts.iter().enumerate() {
        ensure!(
            file_hash(&resolve(&root, &c.path)?)? == c.sha256,
            "context changed"
        );
        let e: Encoding = read_json(&resolve(&root, &c.path)?.with_extension("json"))?;
        ensure!(
            e.encoder_spec_sha256 == ci["assembly_sha256"]
                && e.prepared_sha256 == c.prepared_sha256,
            "encoding identity mismatch"
        );
        if !prepared.contains_key(&c.prepared) {
            let p = resolve(&root, &c.prepared)?.join("prepared.json");
            // Route R8: the map holds exactly the verified preparation bytes.
            let bytes = read_verified(&p, &c.prepared_sha256, "prepared identity")?;
            prepared.insert(
                c.prepared.clone(),
                serde_json::from_slice::<Prepared>(&bytes)?,
            );
        }
        ensure!(
            by_cut.insert((c.dataset.clone(), c.cut), i).is_none(),
            "duplicate context cut"
        );
    }
    let mut db = Connection::open(store.join("store.sqlite"))?;
    db.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL;
        CREATE TABLE IF NOT EXISTS binding (identity TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS sources (ordinal INTEGER PRIMARY KEY, path TEXT UNIQUE NOT NULL,
            digest TEXT NOT NULL, search_digest TEXT NOT NULL, counts TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS programs (context INTEGER NOT NULL, program TEXT NOT NULL,
            label TEXT NOT NULL, PRIMARY KEY(context,program));
        CREATE TABLE IF NOT EXISTS groups (context INTEGER PRIMARY KEY, shard TEXT, dirty INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS blocked (context INTEGER, program TEXT, PRIMARY KEY(context,program));
        CREATE TABLE IF NOT EXISTS quarantine (source TEXT, row_index INTEGER, context INTEGER, program TEXT, reason TEXT, observation TEXT);
        CREATE TABLE IF NOT EXISTS outcomes (source TEXT, row_index INTEGER, identity TEXT, observation TEXT,
            outcome TEXT NOT NULL, elapsed_seconds REAL, censor_seconds REAL,
            numerical_elapsed_seconds REAL, numerical_censor_seconds REAL);
        CREATE INDEX IF NOT EXISTS outcome_identity ON outcomes(identity);
        CREATE TABLE IF NOT EXISTS reuses (source TEXT, row_index INTEGER, observation TEXT,
            PRIMARY KEY(source,row_index));
        CREATE TABLE IF NOT EXISTS admissions (context INTEGER, program TEXT, source TEXT, row_index INTEGER,
            evaluation_sha256 TEXT, observation_sha256 TEXT, PRIMARY KEY(source,row_index));
        CREATE INDEX IF NOT EXISTS admission_program ON admissions(context,program);
        CREATE TABLE IF NOT EXISTS frozen_z (context INTEGER NOT NULL, program TEXT NOT NULL,
            value REAL NOT NULL, percentile REAL NOT NULL, PRIMARY KEY(context,program)) WITHOUT ROWID;")?;
    let old: Vec<String> = db
        .prepare("SELECT identity FROM binding")?
        .query_map([], |r| r.get(0))?
        .collect::<std::result::Result<_, _>>()?;
    if old.is_empty() {
        db.execute("INSERT INTO binding VALUES (?)", [&binding])?;
    } else {
        ensure!(
            old == vec![binding.clone()],
            "teaching store policy/input identity changed"
        );
    }
    let mut paths = Vec::new();
    for source in sources {
        let source = resolve(
            &root,
            if source.is_absolute() {
                source.strip_prefix(&root)?
            } else {
                source
            },
        )?;
        let mut found = walkdir::WalkDir::new(source)
            .into_iter()
            .collect::<std::result::Result<Vec<_>, _>>()?
            .into_iter()
            .filter(|e| {
                e.file_name() == "observations.json"
                    && e.path().with_file_name("search.json").is_file()
            })
            .map(|e| e.into_path())
            .collect::<Vec<_>>();
        found.sort();
        paths.extend(found);
    }
    ensure!(
        paths.iter().collect::<BTreeSet<_>>().len() == paths.len(),
        "duplicate source ledger"
    );
    let existing: usize = db.query_row("SELECT count(*) FROM sources", [], |r| r.get(0))?;
    ensure!(
        paths.len() >= existing,
        "committed teaching sources were removed"
    );
    let mut tokens = corpus::TokenCache::default();
    let mut ingested = 0;
    let mut verified_inputs = BTreeMap::<usize, std::result::Result<String, String>>::new();
    for (ordinal, path) in paths.iter().enumerate() {
        let relative = path.strip_prefix(&root)?.to_string_lossy().to_string();
        let bytes = fs::read(path)?;
        let digest = hash(&bytes);
        // Route R8: the receipt validated below is the one whose digest the
        // ledger commits; it is read once and never reopened.
        let (search_bytes, search_digest) =
            read_hashed(&path.with_file_name("search.json"), "search receipt")?;
        if ordinal < existing {
            let saved: (String, String, String) = db.query_row(
                "SELECT path,digest,search_digest FROM sources WHERE ordinal=?",
                [ordinal],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )?;
            ensure!(
                saved == (relative, digest, search_digest),
                "committed ledger changed or reordered"
            );
            continue;
        }
        let mut rows: Vec<Value> = serde_json::from_slice(&bytes)?;
        let observation_count = rows.len();
        let mut quarantined = Vec::new();
        let mut outcomes = Vec::new();
        let mut blocked = BTreeSet::new();
        let mut admissions = Vec::new();
        let mut reuses = Vec::new();
        if let Some(expected) = &expected {
            let search: Value = serde_json::from_slice(&search_bytes)?;
            ensure!(
                search["observations_sha256"] == digest,
                "completed search ledger digest changed"
            );
            let mut current = BTreeMap::<(usize, String), f64>::new();
            let mut failed = BTreeSet::new();
            let mut accepted = Vec::new();
            let row_count = rows.len();
            for (n, row) in rows.into_iter().enumerate() {
                let context = by_cut
                    .get(&(
                        row["dataset"].as_str().unwrap_or("").into(),
                        row["cut"].as_u64().unwrap_or(u64::MAX) as usize,
                    ))
                    .copied();
                let program = row["genome"].as_str().unwrap_or("").to_string();
                let validation = context.context("unknown teaching context").and_then(|c| {
                    let training = verified_inputs
                        .entry(c)
                        .or_insert_with(|| {
                            crate::label_admission::verify_inputs(
                                &root,
                                &contexts[c],
                                &prepared[&contexts[c].prepared],
                            )
                            .map_err(|e| format!("{e:#}"))
                        })
                        .as_ref()
                        .map_err(|e| anyhow::anyhow!("{e}"))?;
                    crate::label_admission::validate_receipt(&search, &row, row_count, expected)?;
                    crate::label_admission::validate_preverified(
                        &root,
                        &row,
                        &contexts[c],
                        &prepared[&contexts[c].prepared],
                        crate::label_admission::AdmissionInputs {
                            vocabulary_sha: &hash(&vocab_bytes),
                            expected,
                            fit_training: training,
                        },
                        authority.as_mut().context("approved authority absent")?,
                    )
                });
                let mut error = validation.err().map(|e| format!("{e:#}"));
                if error.is_none() {
                    if row["evaluation_origin"] == "reuse"
                        || row["metrics"]["evaluation_origin"] == "reuse"
                    {
                        reuses.push((n, row));
                        continue;
                    }
                    let context = context.unwrap();
                    let key = (context, program.clone());
                    let old: Option<String> = db
                        .query_row(
                            "SELECT label FROM programs WHERE context=? AND program=?",
                            params![context, &program],
                            |r| r.get(0),
                        )
                        .optional()?;
                    let prior = old
                        .map(|label| serde_json::from_str::<(Vec<usize>, f64, f64, String)>(&label))
                        .transpose()?
                        .map(|label| label.1);
                    if !row["failure"].is_null() {
                        outcomes.push((n, row.clone()));
                        failed.insert(key.clone());
                        if prior.is_some() || current.contains_key(&key) {
                            error = Some("unresolved success/failure disagreement".into());
                        } else {
                            continue;
                        }
                    } else {
                        let score = row["metrics"]["rmse"].as_f64().unwrap();
                        let prior_failure = db.query_row(
                            "SELECT EXISTS(SELECT 1 FROM outcomes WHERE identity=?)",
                            [serde_json::to_string(&crate::label_admission::identity(
                                &row,
                            ))?],
                            |r| r.get::<_, bool>(0),
                        )?;
                        if failed.contains(&key) || prior_failure {
                            error = Some("unresolved success/failure disagreement".into());
                        } else if prior.is_some_and(|p| p != score)
                            || current.get(&key).is_some_and(|p| *p != score)
                        {
                            error = Some("unresolved repeated-measurement disagreement".into());
                        } else {
                            current.insert(key, score);
                        }
                    }
                }
                if let Some(error) = error {
                    if let Some(context) = context {
                        blocked.insert((context, program.clone()));
                    }
                    quarantined.push((n, context, program, error, row));
                } else {
                    admissions.push((
                        context.unwrap(),
                        program,
                        n,
                        row["evaluation_identity_sha256"]
                            .as_str()
                            .unwrap()
                            .to_string(),
                        hash(&serde_json::to_vec(&row)?),
                    ));
                    accepted.push(row);
                }
            }
            rows = accepted
                .into_iter()
                .filter(|row| {
                    !blocked.contains(&(
                        by_cut[&(
                            row["dataset"].as_str().unwrap().into(),
                            row["cut"].as_u64().unwrap() as usize,
                        )],
                        row["genome"].as_str().unwrap().to_string(),
                    ))
                })
                .collect();
        }
        let mut contribution = corpus::scan_ledger(
            rows,
            path,
            crate::corpus::ScanInputs {
                by_cut: &by_cut,
                contexts: &contexts,
                prepared: &prepared,
                vocab: &vocab,
            },
            qualification,
            &mut tokens,
        )?;
        if admission.is_some() {
            contribution
                .counts
                .insert("observations_seen".into(), observation_count);
            contribution
                .counts
                .insert("admission_quarantined".into(), quarantined.len());
            contribution
                .counts
                .insert("admission_failed_outcomes".into(), outcomes.len());
            contribution
                .counts
                .insert("registry_reuse_not_observation".into(), reuses.len());
        }
        let transaction = db.transaction()?;
        for (context, program) in blocked {
            transaction.execute(
                "INSERT OR IGNORE INTO blocked VALUES (?,?)",
                params![context, &program],
            )?;
            if transaction.execute(
                "DELETE FROM programs WHERE context=? AND program=?",
                params![context, &program],
            )? > 0
            {
                transaction.execute("UPDATE groups SET dirty=1 WHERE context=?", [context])?;
            }
        }
        for (n, context, program, reason, row) in quarantined {
            transaction.execute(
                "INSERT INTO quarantine VALUES (?,?,?,?,?,?)",
                params![
                    &relative,
                    n,
                    context,
                    program,
                    reason,
                    serde_json::to_string(&row)?
                ],
            )?;
        }
        for (n, row) in outcomes {
            let completion = crate::label_admission::artifact(&root, &row["clean_completion"])?;
            let outcome = completion["outcome"]
                .as_str()
                .context("validated outcome absent")?;
            *contribution
                .counts
                .entry(format!("execution_{outcome}"))
                .or_default() += 1;
            let native = crate::label_admission::artifact(&root, &completion["fit_completion"])?;
            transaction.execute(
                "INSERT INTO outcomes VALUES (?,?,?,?,?,?,?,?,?)",
                params![
                    &relative,
                    n,
                    serde_json::to_string(&crate::label_admission::identity(&row))?,
                    serde_json::to_string(&row)?,
                    outcome,
                    completion["runtime_seconds"].as_f64(),
                    if outcome == "timeout" {
                        completion["deadline_seconds"].as_f64()
                    } else {
                        None
                    },
                    native["elapsed_seconds"].as_f64(),
                    if native["outcome"] == "timeout" {
                        native["deadline_seconds"].as_f64()
                    } else {
                        None
                    }
                ],
            )?;
        }
        for (n, row) in reuses {
            transaction.execute(
                "INSERT INTO reuses VALUES (?,?,?)",
                params![&relative, n, serde_json::to_string(&row)?],
            )?;
        }
        for (context, program, n, evaluation, observation) in admissions {
            transaction.execute(
                "INSERT INTO admissions VALUES (?,?,?,?,?,?)",
                params![context, program, &relative, n, evaluation, observation],
            )?;
        }
        for (context, programs) in contribution.selected {
            let mut changed = false;
            for (program, label) in programs {
                if transaction.query_row(
                    "SELECT EXISTS(SELECT 1 FROM blocked WHERE context=? AND program=?)",
                    params![context, &program],
                    |r| r.get::<_, bool>(0),
                )? {
                    continue;
                }
                let prior_best: Option<f64> = transaction.query_row(
                    "SELECT MIN(CAST(json_extract(label,'$[1]') AS REAL)) FROM programs WHERE context=?",
                    [context], |r| r.get(0))?;
                let inserted = transaction.execute(
                    "INSERT OR IGNORE INTO programs VALUES (?,?,?)",
                    params![context, program, serde_json::to_string(&label)?],
                )?;
                if inserted == 0 {
                    *contribution
                        .counts
                        .entry("duplicate_program_cut".into())
                        .or_default() += 1;
                } else {
                    changed = true;
                    *contribution
                        .counts
                        .entry("first_label_growth".into())
                        .or_default() += 1;
                    if prior_best.is_some_and(|best| label.1 < best) {
                        *contribution
                            .counts
                            .entry("new_teacher_beats".into())
                            .or_default() += 1;
                    }
                }
            }
            if changed {
                transaction.execute("INSERT INTO groups VALUES (?,NULL,1) ON CONFLICT(context) DO UPDATE SET dirty=1", [context])?;
            }
        }
        transaction.execute(
            "INSERT INTO sources VALUES (?,?,?,?,?)",
            params![
                ordinal,
                relative,
                digest,
                search_digest,
                serde_json::to_string(&contribution.counts)?
            ],
        )?;
        transaction.commit()?;
        ingested += 1;
        progress("search ledger committed", ordinal + 1, Some(paths.len()))?;
    }
    let groups: Vec<(usize, Option<String>, bool)> = db
        .prepare("SELECT context,shard,dirty FROM groups ORDER BY context")?
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?
        .collect::<std::result::Result<_, _>>()?;
    let mut shards = Vec::new();
    let mut rebuilt = 0;
    for (context, saved, dirty) in groups {
        if !dirty {
            if let Some(saved) = saved {
                let shard: Shard = serde_json::from_str(&saved)?;
                verify_shard(&root, &shard)?;
                shards.push(shard);
            }
            continue;
        }
        let programs: Programs = db
            .prepare("SELECT program,label FROM programs WHERE context=? ORDER BY program")?
            .query_map([context], |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
            })?
            .map(|r| {
                let (key, value) = r?;
                Ok((key, serde_json::from_str(&value)?))
            })
            .collect::<Result<_>>()?;
        let mut lineage = BTreeMap::new();
        if admission.is_some() {
            for program in programs.keys() {
                let occurrences: Vec<Value> = db.prepare(
                    "SELECT a.source,a.row_index,a.evaluation_sha256,a.observation_sha256,s.digest,s.search_digest
                     FROM admissions a JOIN sources s ON a.source=s.path WHERE a.context=? AND a.program=?
                     ORDER BY s.ordinal,a.row_index")?.query_map(params![context,program], |r| {
                        Ok(json!({"source":r.get::<_,String>(0)?,"row_index":r.get::<_,usize>(1)?,
                            "evaluation_identity_sha256":r.get::<_,String>(2)?,"observation_sha256":r.get::<_,String>(3)?,
                            "ledger_sha256":r.get::<_,String>(4)?,"search_receipt_sha256":r.get::<_,String>(5)?}))
                    })?.collect::<std::result::Result<_,_>>()?;
                let mut source = occurrences
                    .first()
                    .context("accepted program source receipt absent")?
                    .clone();
                source["supporting_occurrences"] = json!(occurrences);
                lineage.insert(program.clone(), source);
            }
        }
        let mut previous = BTreeMap::new();
        if policy == "frozen_z_v1" {
            let mut stmt =
                db.prepare("SELECT program,value,percentile FROM frozen_z WHERE context=?")?;
            previous = stmt
                .query_map([context], |r| {
                    Ok((r.get::<_, String>(0)?, (r.get(1)?, r.get(2)?)))
                })?
                .collect::<std::result::Result<_, _>>()?;
        }
        let packed = if programs.len() >= 8 {
            Some(pack_group(
                &root,
                &store,
                context,
                programs,
                crate::teaching_store::GroupLabels {
                    vocab: &vocab,
                    binding: &binding,
                    policy,
                    lineage: &lineage,
                    previous: &previous,
                },
            )?)
        } else {
            None
        };
        if let Some((ref _shard, ref assigned)) = packed {
            if policy == "frozen_z_v1" {
                for (program, value, percentile) in assigned {
                    db.execute(
                        "INSERT OR IGNORE INTO frozen_z VALUES (?,?,?,?)",
                        params![context, program, *value, *percentile],
                    )?;
                }
            }
        }
        let shard = packed.map(|(s, _)| s);
        db.execute(
            "UPDATE groups SET shard=?,dirty=0 WHERE context=?",
            params![
                shard.as_ref().map(serde_json::to_string).transpose()?,
                context
            ],
        )?;
        if let Some(shard) = shard {
            shards.push(shard);
        }
        rebuilt += 1;
        progress("teaching group committed", rebuilt, None)?;
    }
    let sources: Vec<(String, String, String)> = db
        .prepare("SELECT path,digest,counts FROM sources ORDER BY ordinal")?
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?
        .collect::<std::result::Result<_, _>>()?;
    let mut counts = BTreeMap::<String, usize>::new();
    let sources = sources.into_iter().map(|(path,digest,c)| -> Result<Value> {
        let c: BTreeMap<String,usize> = serde_json::from_str(&c)?;
        for (key,n) in &c { *counts.entry(key.clone()).or_default() += n; }
        Ok(json!({"path":path,"sha256":digest,"observations":c.get("observations_seen").copied().unwrap_or(0)}))
    }).collect::<Result<Vec<_>>>()?;
    let below: usize =
        db.query_row("SELECT count(*) FROM groups WHERE shard IS NULL", [], |r| {
            r.get(0)
        })?;
    if below > 0 {
        counts.insert("groups_below_eight".into(), below);
    }
    let first_label_growth = counts.remove("first_label_growth").unwrap_or(0);
    let new_teacher_beats = counts.remove("new_teacher_beats").unwrap_or(0);
    ensure!(
        shards
            .iter()
            .filter(|s| contexts[s.context].partition == "meta_train")
            .map(|s| &contexts[s.context].family)
            .collect::<BTreeSet<_>>()
            .len()
            >= 2,
        "insufficient teaching families"
    );
    let mut offsets = Vec::new();
    let mut records = 0usize;
    for shard in &shards {
        offsets.extend((records as u64).to_le_bytes());
        offsets.extend((shard.records as u64).to_le_bytes());
        records += shard.records;
    }
    let manifest = Manifest {
        schema: SCHEMA.into(),
        qualification,
        label_policy: policy.into(),
        semantic_dependencies_sha256,
        admission_policy: expected,
        vocabulary_sha256: hash(&vocab_bytes),
        context_index_sha256: hash(&context_bytes),
        contexts,
        records,
        record_bytes: RECORD_BYTES,
        records_sha256: hash(&serde_json::to_vec(&shards)?),
        groups_sha256: hash(&offsets),
        shards,
        sources,
        counts,
    };
    if output.exists() {
        ensure!(
            read_json::<Value>(&output.join("index.json"))? == serde_json::to_value(&manifest)?,
            "publication already exists with different observations"
        );
        load(&root, &output)?;
        return Ok(());
    }
    let staging = output.with_extension(format!("partial-{}", std::process::id()));
    if staging.exists() {
        fs::remove_dir_all(&staging)?;
    }
    fs::create_dir_all(&staging)?;
    fs::write(staging.join("groups.bin"), offsets)?;
    fs::write(staging.join("vocabulary.json"), vocab_bytes)?;
    write_json(&staging.join("index.json"), &manifest)?;
    write_json(
        &staging.join("provenance.json"),
        &json!({"schema":SCHEMA,
        "index_sha256":file_hash(&staging.join("index.json"))?,"label_policy":policy,
        "duplicate_program_cut":manifest.counts.get("duplicate_program_cut").copied().unwrap_or(0),
        "independent_evaluation":false,"final_tests_opened":false}),
    )?;
    write_json(
        &staging.join("packing-timing.json"),
        &json!({"new_ledgers":ingested,"rebuilt_groups":rebuilt,
        "reused_groups":manifest.shards.len().saturating_sub(rebuilt),
        "first_label_growth":first_label_growth,
        "new_teacher_beats":new_teacher_beats,
        "duplicate_program_cut":manifest.counts.get("duplicate_program_cut").copied().unwrap_or(0),
        "legacy_export_required_for_training":false}),
    )?;
    for entry in fs::read_dir(&staging)? {
        fs::File::open(entry?.path())?.sync_all()?;
    }
    fs::File::open(&staging)?.sync_all()?;
    progress(
        "manifest durable before publication",
        records,
        Some(records),
    )?;
    fs::rename(&staging, &output)?;
    fs::File::open(output.parent().context("publication parent")?)?.sync_all()?;
    progress("teaching manifest published", records, Some(records))?;
    Ok(())
}

pub struct TeachingRecords {
    parts: Vec<(usize, Records, Vec<usize>)>,
    count: usize,
}
type ResidentTeaching = (String, std::rc::Rc<(Index, TeachingRecords)>);

thread_local! {
    static RESIDENT: std::cell::RefCell<Option<ResidentTeaching>> = const { std::cell::RefCell::new(None) };
}

/// Retain the sealed view across tasks with the same immutable identity. Every
/// worker restart clears this cache and verifies the source content again.
pub fn load_resident(root: &Path, corpus: &Path) -> Result<std::rc::Rc<(Index, TeachingRecords)>> {
    load_resident_verified(root, corpus, &fs::read(corpus.join("index.json"))?)
}

/// Load the resident view of exactly `index_bytes`, which the caller has
/// already verified. The index path is never read again, so a file
/// substituted after verification cannot become the resident view; every
/// payload the bytes name is still verified by content (W6 follow-up).
pub fn load_resident_verified(
    root: &Path,
    corpus: &Path,
    index_bytes: &[u8],
) -> Result<std::rc::Rc<(Index, TeachingRecords)>> {
    // W2-T04 (G-11c): the view is of this corpus directory. Another directory
    // with the same index bytes has its own payload files (vocabulary, groups,
    // provenance, legacy records), which a cache hit would never verify.
    let key = format!(
        "{}:{}:{}",
        root.display(),
        corpus.canonicalize()?.display(),
        hash(index_bytes)
    );
    RESIDENT.with(|saved| {
        if let Some((identity, view)) = saved.borrow().as_ref() {
            if *identity == key {
                return Ok(std::rc::Rc::clone(view));
            }
        }
        let view = std::rc::Rc::new(load_impl(root, corpus, true, Some(index_bytes))?);
        *saved.borrow_mut() = Some((key, std::rc::Rc::clone(&view)));
        Ok(view)
    })
}
impl TeachingRecords {
    pub fn get(&self, i: usize) -> Record {
        assert!(i < self.count);
        let part = self.parts.partition_point(|(offset, _, _)| *offset <= i) - 1;
        let (offset, records, map) = &self.parts[part];
        let mut record = records.get(i - offset);
        if !map.is_empty() {
            for id in &mut record.legal {
                *id = map[*id as usize] as i64;
            }
        }
        record
    }
}

/// Every shard is content-verified on open. Local grammar IDs are explicitly
/// remapped in first-appearance order, identical to the reference packer.
pub fn load(root: &Path, corpus: &Path) -> Result<(Index, TeachingRecords)> {
    load_impl(root, corpus, true, None)
}

/// Catalogue readers do not stage the training payload or initialize CUDA.
pub fn read_index(root: &Path, corpus: &Path) -> Result<Index> {
    Ok(load_impl(root, corpus, false, None)?.0)
}

/// The verified corpus index and the corpus vocabulary (the native loop's
/// candidate expansion reads both). The vocabulary is read once against the
/// digest this index binds and parsed from those bytes (W1 route R8); a legacy
/// index, whose `load` never opens the vocabulary, gets the same check.
pub fn read_index_and_vocabulary(
    root: &Path,
    corpus: &Path,
) -> Result<(Index, crate::grammar::Vocab)> {
    let index = read_index(root, corpus)?;
    let vocab = crate::corpus::read_bound_vocabulary(corpus, &index.vocabulary_sha256)?;
    Ok((index, vocab))
}

fn load_impl(
    root: &Path,
    corpus: &Path,
    include_records: bool,
    index_bytes: Option<&[u8]>,
) -> Result<(Index, TeachingRecords)> {
    // Read the index once; its digest and its parsed value come from the
    // same bytes.
    let read;
    let index_bytes = match index_bytes {
        Some(bytes) => bytes,
        None => {
            read = fs::read(corpus.join("index.json"))?;
            &read[..]
        }
    };
    let index_sha256 = hash(index_bytes);
    let value: Value = serde_json::from_slice(index_bytes)?;
    if value["schema"] == "parallel.teaching_corpus.v1" {
        let index: Index = serde_json::from_value(value)?;
        let parts = if include_records {
            vec![(
                0,
                Records::open_verified(
                    &corpus.join("records.bin"),
                    index.records,
                    &index.records_sha256,
                )?,
                vec![],
            )]
        } else {
            vec![]
        };
        let records = TeachingRecords {
            parts,
            count: if include_records { index.records } else { 0 },
        };
        return Ok((index, records));
    }
    let m: Manifest = serde_json::from_value(value)?;
    ensure!(
        m.schema == SCHEMA
            && m.record_bytes == RECORD_BYTES
            && m.records_sha256 == hash(&serde_json::to_vec(&m.shards)?),
        "unsupported or changed teaching manifest"
    );
    ensure!(
        file_hash(&corpus.join("vocabulary.json"))? == m.vocabulary_sha256
            && file_hash(&corpus.join("groups.bin"))? == m.groups_sha256
            && read_json::<Value>(&corpus.join("provenance.json"))?["index_sha256"]
                == index_sha256.as_str(),
        "teaching manifest payload changed"
    );
    let mut sets = BTreeMap::from([(vec![0], 0)]);
    let mut legal_sets = vec![vec![0]];
    let mut parts = Vec::new();
    let mut groups = Vec::new();
    let mut count = 0;
    let mut offsets = Vec::new();
    let mut previous = None;
    for shard in &m.shards {
        ensure!(
            previous.is_none_or(|p| p < shard.context),
            "shard contexts not unique/ordered"
        );
        previous = Some(shard.context);
        let c = m
            .contexts
            .get(shard.context)
            .context("shard context outside manifest")?;
        ensure!(
            matches!(c.partition.as_str(), "meta_train" | "meta_validation") && shard.records >= 8,
            "invalid teaching group"
        );
        let path = verify_shard_metadata(root, shard)?;
        let mut map = Vec::new();
        for set in &shard.legal_sets {
            map.push(*sets.entry(set.clone()).or_insert_with(|| {
                let id = legal_sets.len();
                legal_sets.push(set.clone());
                id
            }));
        }
        ensure!(
            legal_sets.len() < u16::MAX as usize,
            "too many global grammar states"
        );
        offsets.extend((count as u64).to_le_bytes());
        offsets.extend((shard.records as u64).to_le_bytes());
        groups.push(Group {
            context: shard.context,
            family: c.family.clone(),
            partition: c.partition.clone(),
            records: (count..count + shard.records).collect(),
            max_length: shard.max_length,
        });
        if include_records {
            parts.push((
                count,
                Records::open_verified(
                    &path.join("records.bin"),
                    shard.records,
                    &shard.records_sha256,
                )?,
                map,
            ));
        }
        count += shard.records;
    }
    ensure!(
        count == m.records && hash(&offsets) == m.groups_sha256,
        "group offsets/counts changed"
    );
    Ok((
        Index {
            schema: SCHEMA.into(),
            qualification: m.qualification,
            vocabulary_sha256: m.vocabulary_sha256,
            context_index_sha256: m.context_index_sha256,
            records_sha256: hash(&serde_json::to_vec(&m.shards)?),
            record_bytes: RECORD_BYTES,
            records: count,
            legal_sets,
            contexts: m.contexts,
            groups,
            // Sharded labels are bound per shard by the manifest.
            labels_sha256: None,
        },
        TeachingRecords {
            parts,
            count: if include_records { count } else { 0 },
        },
    ))
}

/// Stream and verify original labels, retaining only the requested leaders.
/// Shard-local record identifiers never escape without an explicit offset.
pub fn teacher_programs(
    root: &Path,
    corpus: &Path,
    index: &Index,
    limit: usize,
) -> Result<BTreeMap<usize, Vec<String>>> {
    use sha2::{Digest, Sha256};
    use std::io::{BufRead, BufReader};
    let selected = index
        .groups
        .iter()
        .flat_map(|g| g.records.iter().take(limit).map(move |r| (*r, g.context)))
        .collect::<BTreeMap<_, _>>();
    // Route R8: the index is read once. The digest the provenance must name
    // and the label sources come from the same bytes, and those bytes must be
    // the index the caller verified (`load_resident_verified`, `read_index`).
    let (index_bytes, index_sha256) = read_hashed(&corpus.join("index.json"), "label index")?;
    let value: Value = serde_json::from_slice(&index_bytes)?;
    let provenance: Value = read_json(&corpus.join("provenance.json"))?;
    ensure!(
        provenance["index_sha256"] == index_sha256,
        "label index changed"
    );
    let sources = if value["schema"] == SCHEMA {
        let m: Manifest = serde_json::from_value(value)?;
        ensure!(
            hash(&serde_json::to_vec(&m.shards)?) == index.records_sha256,
            "label index changed"
        );
        m.shards
            .iter()
            .map(|s| {
                Ok((
                    resolve(root, &s.path)?.join("labels.jsonl"),
                    s.labels_sha256.clone(),
                    s.records,
                ))
            })
            .collect::<Result<Vec<_>>>()?
    } else {
        let legacy: Index = serde_json::from_value(value)?;
        ensure!(
            legacy.records_sha256 == index.records_sha256
                && legacy.labels_sha256 == index.labels_sha256,
            "label index changed"
        );
        // W1 round 8: the labels digest is the one the verified index binds.
        // provenance.json is bound only by its back-reference to the index, so
        // it never supplies the digest, and a disagreeing one is refused.
        let bound = legacy.labels_sha256.context(
            "legacy teaching corpus labels are unbound: the index binds no labels digest and \
             provenance.json is bound only by its back-reference to the index",
        )?;
        ensure!(
            provenance["labels_sha256"]
                .as_str()
                .is_none_or(|named| named == bound),
            "teacher labels changed"
        );
        vec![(corpus.join("labels.jsonl"), bound, index.records)]
    };
    let mut result = BTreeMap::<usize, Vec<String>>::new();
    let mut offset = 0;
    for (path, expected, count) in sources {
        let mut reader = BufReader::new(fs::File::open(path)?);
        let mut digest = Sha256::new();
        let mut line = Vec::new();
        let mut local = 0;
        while reader.read_until(b'\n', &mut line)? != 0 {
            digest.update(&line);
            if let Some(&context) = selected.get(&(offset + local)) {
                let row: Value = serde_json::from_slice(&line)?;
                ensure!(
                    row["record"] == local && row["context"] == context,
                    "teacher label coordinates changed"
                );
                result.entry(context).or_default().push(
                    row["program"]
                        .as_str()
                        .context("teacher program absent")?
                        .into(),
                );
            }
            local += 1;
            line.clear();
        }
        ensure!(
            local == count && hex::encode(digest.finalize()) == expected,
            "teacher labels changed"
        );
        offset += count;
    }
    ensure!(offset == index.records, "teacher label count changed");
    Ok(result)
}

/// Background compatibility export; it is never a prerequisite for training.
pub fn export(root: &Path, corpus: &Path, output: &Path) -> Result<()> {
    use std::io::{BufRead, BufReader};
    ensure!(!output.exists(), "legacy export exists");
    let m: Manifest = read_json(&corpus.join("index.json"))?;
    let (mut index, records) = load(root, corpus)?;
    let staging = output.with_extension(format!("partial-{}", std::process::id()));
    fs::create_dir_all(&staging)?;
    let mut binary = BufWriter::new(fs::File::create(staging.join("records.bin"))?);
    for i in 0..index.records {
        binary.write_all(&records.get(i).bytes())?;
    }
    binary.flush()?;
    binary.get_ref().sync_all()?;
    let mut labels = BufWriter::new(fs::File::create(staging.join("labels.jsonl"))?);
    let mut count = 0;
    for shard in &m.shards {
        for line in BufReader::new(fs::File::open(
            resolve(root, &shard.path)?.join("labels.jsonl"),
        )?)
        .lines()
        {
            let mut value: Value = serde_json::from_str(&line?)?;
            value["record"] = json!(count);
            count += 1;
            serde_json::to_writer(&mut labels, &value)?;
            labels.write_all(b"\n")?;
        }
    }
    labels.flush()?;
    labels.get_ref().sync_all()?;
    ensure!(count == index.records, "shard label count changed");
    index.schema = "parallel.teaching_corpus.v1".into();
    index.records_sha256 = file_hash(&staging.join("records.bin"))?;
    // W1 round 8: the legacy index binds its labels (provenance.json does not).
    index.labels_sha256 = Some(file_hash(&staging.join("labels.jsonl"))?);
    fs::copy(
        corpus.join("vocabulary.json"),
        staging.join("vocabulary.json"),
    )?;
    write_json(&staging.join("index.json"), &index)?;
    let families: BTreeSet<_> = index
        .groups
        .iter()
        .filter(|g| g.partition == "meta_train")
        .map(|g| &g.family)
        .collect();
    let label_policy = match m.label_policy.as_str() {
        "reference_v1" => "exact program parameters only; no numeric snapping; duplicate program/cut labels have one vote",
        "frozen_z_v1" => "frozen_z_v1; admitted value/percentile stay put; only new rows get a first label",
        "strict_midrank_v4" => "strict_midrank_v4; original committed execution required; exact f64 RMSE ties share targets",
        _ => "midrank_v2; observations preserved; exact f64 RMSE ties share targets",
    };
    write_json(
        &staging.join("provenance.json"),
        &json!({"sources":m.sources,"counts":m.counts,"selected_programs":index.records,
        "groups":index.groups.len(),"meta_training_families":families,"index_sha256":file_hash(&staging.join("index.json"))?,
        "labels_sha256":file_hash(&staging.join("labels.jsonl"))?,"evaluation_bias":"adaptive_cut_heldout_teacher",
        "label_policy":label_policy,"qualification":index.qualification,"independent_evaluation":false,"final_tests_opened":false}),
    )?;
    for e in fs::read_dir(&staging)? {
        fs::File::open(e?.path())?.sync_all()?;
    }
    fs::File::open(&staging)?.sync_all()?;
    fs::rename(staging, output)?;
    fs::File::open(output.parent().context("export parent")?)?.sync_all()?;
    Ok(())
}

#[cfg(test)]
mod frozen_z_tests {
    use super::*;

    #[test]
    fn losers_do_not_rewrite_admitted_z() {
        let programs = vec!["old".into(), "new".into()];
        let rmses = vec![0.5, 2.0];
        let mut previous = BTreeMap::new();
        previous.insert("old".into(), (0.33, 1.0));
        let (values, percentiles, frozen) = frozen_z_first_labels(&rmses, &previous, &programs);
        assert_eq!(frozen, 1);
        assert_eq!(values[0], 0.33);
        assert_eq!(percentiles[0], 1.0);
        assert_ne!((values[1], percentiles[1]), (0.33, 1.0));
    }
}

#[cfg(test)]
mod admission_tests;
#[cfg(test)]
mod read_once_tests;
