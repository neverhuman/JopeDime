//! Append-only teaching ledgers and immutable, independently reusable groups.
//! The v1 mode retains reference label semantics; midranks are an explicit v2 trial.
use crate::{corpus::{self, ContextRow, Group, Index, Record, Records, SEQUENCE, RECORD_BYTES},
    encoding::Encoding, grammar::{State, Vocab}, manifest::*, prepare::Prepared};
use anyhow::{ensure, Context, Result};
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{collections::{BTreeMap, BTreeSet}, fs, io::{BufWriter, Write},
    os::fd::AsRawFd, path::{Path, PathBuf}};

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

pub fn midranks(scores: &[f64]) -> Vec<f32> {
    assert!(scores.len() >= 2 && scores.windows(2).all(|w| w[0] <= w[1]));
    let mut targets = vec![0.; scores.len()];
    let mut start = 0;
    while start < scores.len() {
        let mut end = start + 1;
        while end < scores.len() && scores[start] == scores[end] { end += 1; }
        let target = (1.0 - (start + end - 1) as f64 / 2.0 / (scores.len() - 1) as f64) as f32;
        targets[start..end].fill(target);
        start = end;
    }
    targets
}

fn verify_shard_metadata(root: &Path, shard: &Shard) -> Result<PathBuf> {
    let path = resolve(root, &shard.path)?;
    ensure!(file_hash(&path.join("shard.json"))? == shard.sha256, "shard manifest changed");
    let saved: Shard = read_json(&path.join("shard.json"))?;
    // The manifest's own digest is carried by its parent, avoiding self-hashing.
    let mut expected = shard.clone(); expected.sha256.clear();
    ensure!(serde_json::to_value(saved)? == serde_json::to_value(expected)?, "shard binding changed");
    Ok(path)
}

fn verify_shard(root: &Path, shard: &Shard) -> Result<PathBuf> {
    let path = verify_shard_metadata(root, shard)?;
    ensure!(file_hash(&path.join("records.bin"))? == shard.records_sha256
        && file_hash(&path.join("labels.jsonl"))? == shard.labels_sha256
        && path.join("records.bin").metadata()?.len() == (shard.records * RECORD_BYTES) as u64,
        "shard content changed");
    Ok(path)
}

fn pack_group(root: &Path, store: &Path, context: usize, programs: Programs, vocab: &Vocab,
    binding: &str, policy: &str) -> Result<Shard> {
    let key = hash(&serde_json::to_vec(&(binding, context, &programs))?);
    let output = store.join("shards").join(&key);
    if output.is_dir() {
        let mut shard: Shard = read_json(&output.join("shard.json"))?;
        shard.sha256 = file_hash(&output.join("shard.json"))?;
        verify_shard(root, &shard)?;
        return Ok(shard);
    }
    let staging = store.join("shards").join(format!(".partial-{key}-{}", std::process::id()));
    // Staging is disposable derived data. Committed names are never overwritten.
    if staging.exists() { fs::remove_dir_all(&staging)?; }
    fs::create_dir_all(&staging)?;
    let mut file = BufWriter::new(fs::File::create(staging.join("records.bin"))?);
    let mut labels = BufWriter::new(fs::File::create(staging.join("labels.jsonl"))?);
    let mut rows: Vec<_> = programs.into_iter().collect();
    rows.sort_by(|a, b| a.1.1.total_cmp(&b.1.1).then(a.0.cmp(&b.0)));
    let median = rows[rows.len() / 2].1.1.max(1e-8);
    let targets = if policy != "reference_v1" {
        midranks(&rows.iter().map(|r| r.1.1).collect::<Vec<_>>())
    } else { (0..rows.len()).map(|i| 1. - i as f32 / (rows.len()-1) as f32).collect() };
    let mut sets = BTreeMap::from([(vec![0], 0)]);
    let mut legal_sets = vec![vec![0]];
    let mut transitions = BTreeMap::<Vec<usize>, [i64; SEQUENCE]>::new();
    let mut max_length = 0;
    let count = rows.len();
    for (i, (program, (tokens, rmse, runtime, bias))) in rows.into_iter().enumerate() {
        crate::campaign::check_deadline()?;
        let mut record = Record { tokens:[0; SEQUENCE], legal:[0; SEQUENCE],
            length:tokens.len(), context, value:(1./(1.+rmse/median)) as f32,
            percentile:targets[i], runtime:runtime.ln_1p() as f32 };
        for (slot, &token) in record.tokens.iter_mut().zip(&tokens) { *slot = token as i64; }
        if let Some(legal) = transitions.get(&tokens) { record.legal = *legal; }
        else {
            let mut state = State::default(); state.sampling = false;
            for (j, &token) in tokens.iter().enumerate().skip(1) {
                let legal = state.legal(vocab);
                let id = *sets.entry(legal.clone()).or_insert_with(|| {
                    let id = legal_sets.len(); legal_sets.push(legal); id
                });
                record.legal[j] = id as i64;
                state.advance(vocab, token)?;
            }
            if transitions.len() < 65536 { transitions.insert(tokens, record.legal); }
        }
        ensure!(legal_sets.len() < u16::MAX as usize, "too many shard grammar states");
        file.write_all(&record.bytes())?;
        serde_json::to_writer(&mut labels, &json!({"record":i,"context":context,"program":program,
            "rmse_cut_normalized":rmse,"runtime_seconds":runtime,"evaluation_bias":bias}))?;
        labels.write_all(b"\n")?;
        max_length = max_length.max(record.length);
    }
    file.flush()?; file.get_ref().sync_all()?;
    labels.flush()?; labels.get_ref().sync_all()?;
    let mut shard = Shard {context, path:output.strip_prefix(root)?.into(), sha256:String::new(),
        records:count, max_length, records_sha256:file_hash(&staging.join("records.bin"))?,
        labels_sha256:file_hash(&staging.join("labels.jsonl"))?, legal_sets};
    write_json(&staging.join("shard.json"), &shard)?;
    shard.sha256 = file_hash(&staging.join("shard.json"))?;
    fs::File::open(&staging)?.sync_all()?;
    fs::rename(&staging, &output)?;
    fs::File::open(output.parent().unwrap())?.sync_all()?;
    Ok(shard)
}

/// Source order is an explicit append-only contract for legacy first-label
/// semantics. Mutated, removed or reordered receipts require a new store.
pub fn publish(root: &Path, contexts_path: &Path, vocabulary: &Path, sources: &[PathBuf],
    output: &Path, store: &Path, qualification: bool, policy: &str,
    progress: impl FnMut(&str, usize, Option<usize>) -> Result<()>) -> Result<()> {
    publish_impl(root,contexts_path,vocabulary,sources,output,store,qualification,policy,None,progress)
}

pub fn publish_admitted(root: &Path, contexts_path: &Path, vocabulary: &Path, sources: &[PathBuf],
    output: &Path, store: &Path, qualification: bool, admission: &Path,
    progress: impl FnMut(&str, usize, Option<usize>) -> Result<()>) -> Result<()> {
    publish_impl(root,contexts_path,vocabulary,sources,output,store,qualification,"strict_midrank_v3",Some(admission),progress)
}

fn publish_impl(root: &Path, contexts_path: &Path, vocabulary: &Path, sources: &[PathBuf],
    output: &Path, store: &Path, qualification: bool, policy: &str, admission: Option<&Path>,
    mut progress: impl FnMut(&str, usize, Option<usize>) -> Result<()>) -> Result<()> {
    ensure!(["reference_v1", "midrank_v2", "strict_midrank_v3"].contains(&policy), "unknown label policy");
    ensure!((policy == "strict_midrank_v3") == admission.is_some(), "strict admission requires a frozen evaluator manifest");
    let root = root.canonicalize()?;
    let store = resolve(&root, if store.is_absolute() {store.strip_prefix(&root)?} else {store})?;
    let output = resolve(&root, if output.is_absolute() {output.strip_prefix(&root)?} else {output})?;
    ensure!(store.starts_with(root.join("runs")) && !output.starts_with(&store)
        && !store.starts_with(&output), "store/output overlap or non-private store");
    fs::create_dir_all(store.join("shards"))?;
    let lock = fs::OpenOptions::new().write(true).create(true).truncate(false).open(store.join("writer.lock"))?;
    ensure!(unsafe {libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB)} == 0, "teaching store already owned");
    let context_bytes = fs::read(contexts_path)?;
    let ci: Value = serde_json::from_slice(&context_bytes)?;
    ensure!(ci["schema"] == "parallel.context_index.v1" && (qualification || ci["qualification"] == false),
        "context qualification mismatch");
    let contexts: Vec<ContextRow> = serde_json::from_value(ci["encodings"].clone())?;
    let vocab_bytes = fs::read(vocabulary)?;
    let vocab: Vocab = serde_json::from_slice(&vocab_bytes)?;
    let expected: Option<Value> = admission.map(read_json).transpose()?;
    if let Some(expected) = &expected {
        let executable = resolve(&root,Path::new(expected["evaluator"].as_str().context("admission evaluator absent")?))?;
        ensure!(file_hash(&executable)? == expected["evaluator_sha256"], "admission evaluator changed");
    }
    ensure!(vocab.tokens.len() < u16::MAX as usize, "vocabulary too large");
    let binding = hash(&serde_json::to_vec(&(hash(&context_bytes), hash(&vocab_bytes), qualification, policy, &expected,
        hash(concat!(include_str!("corpus.rs"), include_str!("grammar.rs"), include_str!("teaching_store.rs")).as_bytes())))?);
    let mut prepared = BTreeMap::new(); let mut by_cut = BTreeMap::new();
    for (i,c) in contexts.iter().enumerate() {
        ensure!(file_hash(&resolve(&root, &c.path)?)? == c.sha256, "context changed");
        let e: Encoding = read_json(&resolve(&root, &c.path)?.with_extension("json"))?;
        ensure!(e.encoder_spec_sha256 == ci["assembly_sha256"] && e.prepared_sha256 == c.prepared_sha256,
            "encoding identity mismatch");
        if !prepared.contains_key(&c.prepared) {
            let p = resolve(&root, &c.prepared)?.join("prepared.json");
            ensure!(file_hash(&p)? == c.prepared_sha256, "prepared identity changed");
            prepared.insert(c.prepared.clone(), read_json::<Prepared>(&p)?);
        }
        ensure!(by_cut.insert((c.dataset.clone(), c.cut), i).is_none(), "duplicate context cut");
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
        CREATE TABLE IF NOT EXISTS outcomes (source TEXT, row_index INTEGER, identity TEXT, observation TEXT);")?;
    let old: Vec<String> = db.prepare("SELECT identity FROM binding")?.query_map([], |r| r.get(0))?
        .collect::<std::result::Result<_,_>>()?;
    if old.is_empty() { db.execute("INSERT INTO binding VALUES (?)", [&binding])?; }
    else { ensure!(old == vec![binding.clone()], "teaching store policy/input identity changed"); }
    let mut paths = Vec::new();
    for source in sources {
        let source = resolve(&root, if source.is_absolute() {source.strip_prefix(&root)?} else {source})?;
        let mut found = walkdir::WalkDir::new(source).into_iter().collect::<std::result::Result<Vec<_>,_>>()?
            .into_iter().filter(|e| e.file_name() == "observations.json"
                && e.path().with_file_name("search.json").is_file()).map(|e| e.into_path()).collect::<Vec<_>>();
        found.sort(); paths.extend(found);
    }
    ensure!(paths.iter().collect::<BTreeSet<_>>().len() == paths.len(), "duplicate source ledger");
    let existing: usize = db.query_row("SELECT count(*) FROM sources", [], |r| r.get(0))?;
    ensure!(paths.len() >= existing, "committed teaching sources were removed");
    let mut tokens = corpus::TokenCache::default();
    let mut ingested = 0;
    for (ordinal, path) in paths.iter().enumerate() {
        let relative = path.strip_prefix(&root)?.to_string_lossy().to_string();
        let bytes = fs::read(path)?; let digest = hash(&bytes);
        let search_digest = file_hash(&path.with_file_name("search.json"))?;
        if ordinal < existing {
            let saved: (String,String,String) = db.query_row("SELECT path,digest,search_digest FROM sources WHERE ordinal=?",
                [ordinal], |r| Ok((r.get(0)?,r.get(1)?,r.get(2)?)))?;
            ensure!(saved == (relative, digest, search_digest), "committed ledger changed or reordered");
            continue;
        }
        let mut rows: Vec<Value> = serde_json::from_slice(&bytes)?;
        let mut quarantined = Vec::new(); let mut outcomes = Vec::new(); let mut blocked = BTreeSet::new();
        if let Some(expected) = &expected {
            let search: Value = read_json(&path.with_file_name("search.json"))?;
            ensure!(search["observations_sha256"] == digest, "completed search ledger digest changed");
            let mut current = BTreeMap::<(usize,String),f64>::new();
            let mut accepted = Vec::new();
            for (n,row) in rows.into_iter().enumerate() {
                let context = by_cut.get(&(row["dataset"].as_str().unwrap_or("").into(),
                    row["cut"].as_u64().unwrap_or(u64::MAX) as usize)).copied();
                let program = row["genome"].as_str().unwrap_or("").to_string();
                let validation = context.context("unknown teaching context").and_then(|c|
                    crate::label_admission::validate(&root,&row,&contexts[c],&prepared[&contexts[c].prepared],&hash(&vocab_bytes),expected));
                let mut error = validation.err().map(|e| format!("{e:#}"));
                if error.is_none() && !row["failure"].is_null() {
                    outcomes.push((n, row)); continue;
                }
                if error.is_none() {
                    let context = context.unwrap(); let score = row["metrics"]["rmse"].as_f64().unwrap();
                    let old: Option<String> = db.query_row("SELECT label FROM programs WHERE context=? AND program=?",
                        params![context,&program],|r|r.get(0)).optional()?;
                    let prior = old.map(|label|serde_json::from_str::<(Vec<usize>,f64,f64,String)>(&label))
                        .transpose()?.map(|label|label.1);
                    if prior.is_some_and(|p|p != score) || current.get(&(context,program.clone())).is_some_and(|p|*p != score) {
                        error = Some("unresolved repeated-measurement disagreement".into());
                    } else { current.insert((context,program.clone()),score); }
                }
                if let Some(error) = error {
                    if let Some(context) = context { blocked.insert((context,program.clone())); }
                    quarantined.push((n,context,program,error,row));
                } else { accepted.push(row); }
            }
            rows = accepted.into_iter().filter(|row| !blocked.contains(&(
                by_cut[&(row["dataset"].as_str().unwrap().into(),row["cut"].as_u64().unwrap() as usize)],
                row["genome"].as_str().unwrap().to_string()))).collect();
        }
        let contribution = corpus::scan_ledger(rows, path, &by_cut,
            &contexts, &prepared, &vocab, qualification, &mut tokens)?;
        let transaction = db.transaction()?;
        for (context,program) in blocked {
            transaction.execute("INSERT OR IGNORE INTO blocked VALUES (?,?)",params![context,&program])?;
            if transaction.execute("DELETE FROM programs WHERE context=? AND program=?",params![context,&program])? > 0 {
                transaction.execute("UPDATE groups SET dirty=1 WHERE context=?",[context])?;
            }
        }
        for (n,context,program,reason,row) in quarantined {
            transaction.execute("INSERT INTO quarantine VALUES (?,?,?,?,?,?)",params![&relative,n,context,program,reason,serde_json::to_string(&row)?])?;
        }
        for (n,row) in outcomes {
            transaction.execute("INSERT INTO outcomes VALUES (?,?,?,?)",params![&relative,n,
                serde_json::to_string(&crate::label_admission::identity(&row))?,serde_json::to_string(&row)?])?;
        }
        for (context, programs) in contribution.selected {
            let mut changed = false;
            for (program,label) in programs {
                if transaction.query_row("SELECT EXISTS(SELECT 1 FROM blocked WHERE context=? AND program=?)",
                    params![context,&program],|r|r.get::<_,bool>(0))? {continue;}
                changed |= transaction.execute("INSERT OR IGNORE INTO programs VALUES (?,?,?)",
                    params![context, program, serde_json::to_string(&label)?])? != 0;
            }
            if changed { transaction.execute("INSERT INTO groups VALUES (?,NULL,1) ON CONFLICT(context) DO UPDATE SET dirty=1", [context])?; }
        }
        transaction.execute("INSERT INTO sources VALUES (?,?,?,?,?)", params![ordinal,relative,digest,search_digest,
            serde_json::to_string(&contribution.counts)?])?;
        transaction.commit()?;
        ingested += 1;
        progress("search ledger committed", ordinal+1, Some(paths.len()))?;
    }
    let groups: Vec<(usize,Option<String>,bool)> = db.prepare("SELECT context,shard,dirty FROM groups ORDER BY context")?
        .query_map([], |r| Ok((r.get(0)?,r.get(1)?,r.get(2)?)))?.collect::<std::result::Result<_,_>>()?;
    let mut shards = Vec::new(); let mut rebuilt = 0;
    for (context,saved,dirty) in groups {
        if !dirty {
            if let Some(saved) = saved { let shard: Shard = serde_json::from_str(&saved)?;
                verify_shard(&root,&shard)?; shards.push(shard); }
            continue;
        }
        let programs: Programs = db.prepare("SELECT program,label FROM programs WHERE context=? ORDER BY program")?
            .query_map([context], |r| Ok((r.get::<_,String>(0)?,r.get::<_,String>(1)?)))?
            .map(|r| {let (key,value) = r?; Ok((key,serde_json::from_str(&value)?))}).collect::<Result<_>>()?;
        let shard = if programs.len() >= 8 { Some(pack_group(&root,&store,context,programs,&vocab,&binding,policy)?) } else {None};
        db.execute("UPDATE groups SET shard=?,dirty=0 WHERE context=?",
            params![shard.as_ref().map(serde_json::to_string).transpose()?,context])?;
        if let Some(shard) = shard { shards.push(shard); }
        rebuilt += 1;
        progress("teaching group committed", rebuilt, None)?;
    }
    let sources: Vec<(String,String,String)> = db.prepare("SELECT path,digest,counts FROM sources ORDER BY ordinal")?
        .query_map([], |r| Ok((r.get(0)?,r.get(1)?,r.get(2)?)))?.collect::<std::result::Result<_,_>>()?;
    let mut counts = BTreeMap::<String,usize>::new();
    let sources = sources.into_iter().map(|(path,digest,c)| -> Result<Value> {
        let c: BTreeMap<String,usize> = serde_json::from_str(&c)?;
        for (key,n) in &c { *counts.entry(key.clone()).or_default() += n; }
        Ok(json!({"path":path,"sha256":digest,"observations":c.get("observations_seen").copied().unwrap_or(0)}))
    }).collect::<Result<Vec<_>>>()?;
    let below: usize = db.query_row("SELECT count(*) FROM groups WHERE shard IS NULL", [], |r| r.get(0))?;
    if below > 0 { counts.insert("groups_below_eight".into(), below); }
    ensure!(shards.iter().filter(|s| contexts[s.context].partition == "meta_train")
        .map(|s| &contexts[s.context].family).collect::<BTreeSet<_>>().len() >= 2, "insufficient teaching families");
    let mut offsets = Vec::new(); let mut records = 0usize;
    for shard in &shards { offsets.extend((records as u64).to_le_bytes());
        offsets.extend((shard.records as u64).to_le_bytes()); records += shard.records; }
    let manifest = Manifest {schema:SCHEMA.into(),qualification,label_policy:policy.into(),vocabulary_sha256:hash(&vocab_bytes),
        context_index_sha256:hash(&context_bytes),contexts,records,record_bytes:RECORD_BYTES,
        records_sha256:hash(&serde_json::to_vec(&shards)?),
        groups_sha256:hash(&offsets),shards,sources,counts};
    if output.exists() {
        ensure!(read_json::<Value>(&output.join("index.json"))? == serde_json::to_value(&manifest)?,
            "publication already exists with different observations");
        load(&root,&output)?;
        return Ok(());
    }
    let staging = output.with_extension(format!("partial-{}",std::process::id()));
    if staging.exists() { fs::remove_dir_all(&staging)?; }
    fs::create_dir_all(&staging)?;
    fs::write(staging.join("groups.bin"), offsets)?;
    fs::write(staging.join("vocabulary.json"), vocab_bytes)?;
    write_json(&staging.join("index.json"), &manifest)?;
    write_json(&staging.join("provenance.json"), &json!({"schema":SCHEMA,
        "index_sha256":file_hash(&staging.join("index.json"))?,"label_policy":policy,
        "independent_evaluation":false,"final_tests_opened":false}))?;
    write_json(&staging.join("packing-timing.json"), &json!({"new_ledgers":ingested,"rebuilt_groups":rebuilt,
        "reused_groups":manifest.shards.len().saturating_sub(rebuilt),"legacy_export_required_for_training":false}))?;
    for entry in fs::read_dir(&staging)? { fs::File::open(entry?.path())?.sync_all()?; }
    fs::File::open(&staging)?.sync_all()?;
    progress("manifest durable before publication", records, Some(records))?;
    fs::rename(&staging, &output)?; fs::File::open(output.parent().context("publication parent")?)?.sync_all()?;
    progress("teaching manifest published", records, Some(records))?;
    Ok(())
}

pub struct TeachingRecords { parts: Vec<(usize,Records,Vec<usize>)>, count: usize }
thread_local! {
    static RESIDENT: std::cell::RefCell<Option<(String,std::rc::Rc<(Index,TeachingRecords)>)>> = const { std::cell::RefCell::new(None) };
}

/// Retain the sealed view across tasks with the same immutable identity. Every
/// worker restart clears this cache and verifies the source content again.
pub fn load_resident(root:&Path,corpus:&Path)->Result<std::rc::Rc<(Index,TeachingRecords)>> {
    let key=format!("{}:{}",root.display(),file_hash(&corpus.join("index.json"))?);
    RESIDENT.with(|saved| {
        if let Some((identity,view))=saved.borrow().as_ref() {
            if *identity==key {return Ok(std::rc::Rc::clone(view));}
        }
        let view=std::rc::Rc::new(load(root,corpus)?);
        *saved.borrow_mut()=Some((key,std::rc::Rc::clone(&view)));
        Ok(view)
    })
}
impl TeachingRecords {
    pub fn get(&self, i: usize) -> Record {
        assert!(i < self.count);
        let part = self.parts.partition_point(|(offset,_,_)| *offset <= i)-1;
        let (offset,records,map) = &self.parts[part];
        let mut record = records.get(i-offset);
        if !map.is_empty() { for id in &mut record.legal { *id = map[*id as usize] as i64; } }
        record
    }
}

/// Every shard is content-verified on open. Local grammar IDs are explicitly
/// remapped in first-appearance order, identical to the reference packer.
pub fn load(root: &Path, corpus: &Path) -> Result<(Index,TeachingRecords)> {
    load_impl(root, corpus, true)
}

/// Catalogue readers do not stage the training payload or initialize CUDA.
pub fn read_index(root: &Path, corpus: &Path) -> Result<Index> {
    Ok(load_impl(root, corpus, false)?.0)
}

fn load_impl(root: &Path, corpus: &Path, include_records: bool) -> Result<(Index,TeachingRecords)> {
    let value: Value = read_json(&corpus.join("index.json"))?;
    if value["schema"] == "parallel.teaching_corpus.v1" {
        let index: Index = serde_json::from_value(value)?;
        let parts = if include_records {vec![(0,Records::open_verified(&corpus.join("records.bin"),index.records,&index.records_sha256)?,vec![])]} else {vec![]};
        let records = TeachingRecords {parts,count:if include_records {index.records} else {0}};
        return Ok((index,records));
    }
    let m: Manifest = serde_json::from_value(value)?;
    ensure!(m.schema == SCHEMA && m.record_bytes == RECORD_BYTES
        && m.records_sha256 == hash(&serde_json::to_vec(&m.shards)?), "unsupported or changed teaching manifest");
    ensure!(file_hash(&corpus.join("vocabulary.json"))? == m.vocabulary_sha256
        && file_hash(&corpus.join("groups.bin"))? == m.groups_sha256
        && read_json::<Value>(&corpus.join("provenance.json"))?["index_sha256"] == file_hash(&corpus.join("index.json"))?,
        "teaching manifest payload changed");
    let mut sets = BTreeMap::from([(vec![0],0)]); let mut legal_sets = vec![vec![0]];
    let mut parts = Vec::new(); let mut groups = Vec::new(); let mut count = 0; let mut offsets = Vec::new();
    let mut previous = None;
    for shard in &m.shards {
        ensure!(previous.is_none_or(|p| p < shard.context), "shard contexts not unique/ordered"); previous = Some(shard.context);
        let c = m.contexts.get(shard.context).context("shard context outside manifest")?;
        ensure!(matches!(c.partition.as_str(),"meta_train"|"meta_validation") && shard.records >= 8, "invalid teaching group");
        let path = verify_shard_metadata(root,shard)?;
        let mut map = Vec::new();
        for set in &shard.legal_sets {
            map.push(*sets.entry(set.clone()).or_insert_with(|| { let id = legal_sets.len(); legal_sets.push(set.clone()); id }));
        }
        ensure!(legal_sets.len() < u16::MAX as usize, "too many global grammar states");
        offsets.extend((count as u64).to_le_bytes()); offsets.extend((shard.records as u64).to_le_bytes());
        groups.push(Group {context:shard.context,family:c.family.clone(),partition:c.partition.clone(),
            records:(count..count+shard.records).collect(),max_length:shard.max_length});
        if include_records {
            parts.push((count,Records::open_verified(&path.join("records.bin"),shard.records,&shard.records_sha256)?,map));
        }
        count += shard.records;
    }
    ensure!(count == m.records && hash(&offsets) == m.groups_sha256, "group offsets/counts changed");
    Ok((Index {schema:SCHEMA.into(),qualification:m.qualification,vocabulary_sha256:m.vocabulary_sha256,
        context_index_sha256:m.context_index_sha256,records_sha256:hash(&serde_json::to_vec(&m.shards)?),
        record_bytes:RECORD_BYTES,records:count,legal_sets,contexts:m.contexts,groups}, TeachingRecords {parts,count:if include_records {count} else {0}}))
}

/// Stream and verify original labels, retaining only the requested leaders.
/// Shard-local record identifiers never escape without an explicit offset.
pub fn teacher_programs(root: &Path, corpus: &Path, index: &Index, limit: usize)
    -> Result<BTreeMap<usize,Vec<String>>> {
    use std::io::{BufRead,BufReader};
    use sha2::{Digest,Sha256};
    let selected = index.groups.iter().flat_map(|g| g.records.iter().take(limit)
        .map(move |r| (*r,g.context))).collect::<BTreeMap<_,_>>();
    let value: Value = read_json(&corpus.join("index.json"))?;
    let provenance: Value = read_json(&corpus.join("provenance.json"))?;
    ensure!(provenance["index_sha256"] == file_hash(&corpus.join("index.json"))?, "label index changed");
    let sources = if value["schema"] == SCHEMA {
        let m: Manifest = serde_json::from_value(value)?;
        m.shards.iter().map(|s| Ok((resolve(root,&s.path)?.join("labels.jsonl"),s.labels_sha256.clone(),s.records)))
            .collect::<Result<Vec<_>>>()?
    } else {
        vec![(corpus.join("labels.jsonl"),provenance["labels_sha256"].as_str().context("labels digest absent")?.into(),index.records)]
    };
    let mut result = BTreeMap::<usize,Vec<String>>::new(); let mut offset = 0;
    for (path,expected,count) in sources {
        let mut reader = BufReader::new(fs::File::open(path)?); let mut digest = Sha256::new();
        let mut line = Vec::new(); let mut local = 0;
        while reader.read_until(b'\n',&mut line)? != 0 {
            digest.update(&line);
            if let Some(&context) = selected.get(&(offset+local)) {
                let row: Value = serde_json::from_slice(&line)?;
                ensure!(row["record"] == local && row["context"] == context,"teacher label coordinates changed");
                result.entry(context).or_default().push(row["program"].as_str().context("teacher program absent")?.into());
            }
            local += 1; line.clear();
        }
        ensure!(local == count && hex::encode(digest.finalize()) == expected,"teacher labels changed");
        offset += count;
    }
    ensure!(offset == index.records,"teacher label count changed");
    Ok(result)
}

/// Background compatibility export; it is never a prerequisite for training.
pub fn export(root: &Path, corpus: &Path, output: &Path) -> Result<()> {
    use std::io::{BufRead,BufReader};
    ensure!(!output.exists(), "legacy export exists");
    let m: Manifest = read_json(&corpus.join("index.json"))?;
    let (mut index,records) = load(root,corpus)?;
    let staging = output.with_extension(format!("partial-{}",std::process::id()));
    fs::create_dir_all(&staging)?;
    let mut binary = BufWriter::new(fs::File::create(staging.join("records.bin"))?);
    for i in 0..index.records { binary.write_all(&records.get(i).bytes())?; }
    binary.flush()?; binary.get_ref().sync_all()?;
    let mut labels = BufWriter::new(fs::File::create(staging.join("labels.jsonl"))?); let mut count = 0;
    for shard in &m.shards {
        for line in BufReader::new(fs::File::open(resolve(root,&shard.path)?.join("labels.jsonl"))?).lines() {
            let mut value: Value = serde_json::from_str(&line?)?; value["record"] = json!(count); count += 1;
            serde_json::to_writer(&mut labels,&value)?; labels.write_all(b"\n")?;
        }
    }
    labels.flush()?; labels.get_ref().sync_all()?;
    ensure!(count == index.records, "shard label count changed");
    index.schema = "parallel.teaching_corpus.v1".into(); index.records_sha256 = file_hash(&staging.join("records.bin"))?;
    fs::copy(corpus.join("vocabulary.json"),staging.join("vocabulary.json"))?;
    write_json(&staging.join("index.json"),&index)?;
    let families: BTreeSet<_> = index.groups.iter().filter(|g| g.partition == "meta_train").map(|g| &g.family).collect();
    let label_policy = if m.label_policy == "reference_v1" {"exact program parameters only; no numeric snapping; duplicate program/cut labels have one vote"} else {"midrank_v2; observations preserved; exact f64 RMSE ties share targets"};
    write_json(&staging.join("provenance.json"),&json!({"sources":m.sources,"counts":m.counts,"selected_programs":index.records,
        "groups":index.groups.len(),"meta_training_families":families,"index_sha256":file_hash(&staging.join("index.json"))?,
        "labels_sha256":file_hash(&staging.join("labels.jsonl"))?,"evaluation_bias":"adaptive_cut_heldout_teacher",
        "label_policy":label_policy,"qualification":index.qualification,"independent_evaluation":false,"final_tests_opened":false}))?;
    for e in fs::read_dir(&staging)? { fs::File::open(e?.path())?.sync_all()?; }
    fs::File::open(&staging)?.sync_all()?; fs::rename(staging,output)?;
    fs::File::open(output.parent().context("export parent")?)?.sync_all()?;
    Ok(())
}
