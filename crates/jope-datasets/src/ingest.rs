//! `jope-datasets ingest`: enumerate a source tree, hash every candidate, de-duplicate, write the
//! reports and (unless `--dry-run`) upload blobs and manifests through the object store.

use crate::catalog::{CatalogRow, Ingest, IngestSource, Task};
use crate::hash::{digest_file, FileDigest};
use crate::meta::{aliases_for, build_row, task_from_meta, Hashed};
use crate::paths::{now_compact, now_rfc3339, relpath};
use crate::store::{PutOutcome, Store};
use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

pub const FILES: [&str; 3] = ["train.csv", "test.csv", "meta.json"];
const QUARANTINE: &str = "_dupes_quarantine";

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Source {
    Quant,
    Mirror,
}

impl Source {
    fn ingest_source(self) -> IngestSource {
        match self {
            Source::Quant => IngestSource::Quant,
            Source::Mirror => IngestSource::Mirror,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Options {
    pub source: Source,
    pub root: PathBuf,
    pub out: PathBuf,
    pub dry_run: bool,
    pub resume: bool,
    pub jobs: usize,
    pub limit: Option<usize>,
    /// Store URL for uploads (ignored on `--dry-run`).
    pub store_url: Option<String>,
}

#[derive(Clone, Debug)]
pub struct Candidate {
    pub relpath: String,
    pub dir: PathBuf,
    pub name: String,
    pub task: Option<Task>,
    /// Expected sha256 of train/test/meta from the mirror index.
    pub expected: Option<[String; 3]>,
    pub priority: u8,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Duplicate {
    pub path: String,
    pub rule: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub canonical: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dataset_hash: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RejectRecord {
    pub path: String,
    pub reason: String,
    pub detail: String,
}

#[derive(Default)]
pub struct Enumeration {
    pub candidates: Vec<Candidate>,
    pub duplicates: Vec<Duplicate>,
    pub rejects: Vec<RejectRecord>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Summary {
    pub source: String,
    pub root: String,
    pub dry_run: bool,
    pub candidates: usize,
    pub rows: usize,
    pub rows_by_task: BTreeMap<String, usize>,
    pub distinct_blobs: usize,
    pub blob_bytes: u64,
    pub duplicates: usize,
    pub duplicates_by_rule: BTreeMap<String, usize>,
    pub rejects: usize,
    pub rejects_by_reason: BTreeMap<String, usize>,
    pub uploaded: usize,
    pub already_present: usize,
    pub skipped_by_ledger: usize,
    pub elapsed_s: f64,
    pub out: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LedgerLine {
    pub sha256: String,
    pub bytes: u64,
    pub key: String,
    pub outcome: String,
    pub verified: bool,
    pub at: String,
}

fn is_dir(p: &Path) -> bool {
    p.is_dir()
}

fn sorted_names(dir: &Path) -> Result<Vec<String>> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .with_context(|| format!("read_dir {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.file_name().to_string_lossy().into_owned()))
        .collect();
    names.sort();
    Ok(names)
}

/// Quarantined copies below `dir` (`<dir>/_dupes_quarantine/<id>`) are listed, never ingested.
fn quarantine(dir: &Path, prefix: &str, en: &mut Enumeration) -> Result<()> {
    let q = dir.join(QUARANTINE);
    if !is_dir(&q) {
        return Ok(());
    }
    for name in sorted_names(&q)? {
        if crate::dataset_dir_hash(&name).is_some() && is_dir(&q.join(&name)) {
            en.duplicates.push(Duplicate {
                path: format!("{prefix}{QUARANTINE}/{name}"),
                rule: "quarantine".into(),
                canonical: None,
                dataset_hash: crate::dataset_dir_hash(&name).map(str::to_string),
            });
        }
    }
    Ok(())
}

fn candidate_or_reject(
    dir: PathBuf,
    relpath: String,
    name: String,
    task: Option<Task>,
    priority: u8,
    en: &mut Enumeration,
) {
    let missing: Vec<&str> = FILES
        .iter()
        .copied()
        .filter(|f| !dir.join(f).is_file())
        .collect();
    if missing.is_empty() {
        en.candidates.push(Candidate {
            relpath,
            dir,
            name,
            task,
            expected: None,
            priority,
        });
    } else {
        en.rejects.push(RejectRecord {
            path: relpath,
            reason: "incomplete".into(),
            detail: format!("missing {}", missing.join(",")),
        });
    }
}

/// `remote_super/quant`: class directories `regression`, `binary`, `multiclass`, `multi_label`,
/// `timeseries`, `synthetic/binary`. Root-level `<hash>` / `synth_<hash>` directories that also
/// exist under `regression/` and `binary/<id>` directories that also exist under
/// `synthetic/binary/` are duplicates by path rule and are not hashed.
pub fn enumerate_quant(root: &Path) -> Result<Enumeration> {
    let mut en = Enumeration::default();
    let classes: [(&str, Task, u8); 6] = [
        ("regression", Task::Regression, 1),
        ("binary", Task::Binary, 2),
        ("multiclass", Task::Multiclass, 3),
        ("multi_label", Task::MultiLabel, 4),
        ("timeseries", Task::Timeseries, 5),
        ("synthetic/binary", Task::Binary, 6),
    ];
    for (class, task, priority) in classes {
        let dir = root.join(class);
        if !is_dir(&dir) {
            continue;
        }
        quarantine(&dir, &format!("{class}/"), &mut en)?;
        for name in sorted_names(&dir)? {
            let Some(hash) = crate::dataset_dir_hash(&name) else {
                continue;
            };
            let sub = dir.join(&name);
            if !is_dir(&sub) {
                continue;
            }
            if class == "binary" && is_dir(&root.join("synthetic/binary").join(&name)) {
                en.duplicates.push(Duplicate {
                    path: format!("{class}/{name}"),
                    rule: "binary-synthetic-duplicate".into(),
                    canonical: Some(format!("synthetic/binary/{name}")),
                    dataset_hash: Some(hash.to_string()),
                });
                continue;
            }
            candidate_or_reject(
                sub,
                format!("{class}/{name}"),
                name.clone(),
                Some(task),
                priority,
                &mut en,
            );
        }
    }
    quarantine(root, "", &mut en)?;
    for name in sorted_names(root)? {
        let Some(hash) = crate::dataset_dir_hash(&name) else {
            continue;
        };
        let sub = root.join(&name);
        if !is_dir(&sub) {
            continue;
        }
        if is_dir(&root.join("regression").join(&name)) {
            en.duplicates.push(Duplicate {
                path: name.clone(),
                rule: "root-level-duplicate".into(),
                canonical: Some(format!("regression/{name}")),
                dataset_hash: Some(hash.to_string()),
            });
            continue;
        }
        candidate_or_reject(sub, name.clone(), name.clone(), None, 9, &mut en);
    }
    Ok(en)
}

/// The node-archive mirror: `index/dataset.json` maps `home/ubuntu/quant/<class>/<id>/<file>` to
/// its sha256; the bytes live under `dataset/<path>`. A computed sha256 that differs from the
/// index is a reject (`mirror_sha_mismatch`).
pub fn enumerate_mirror(root: &Path) -> Result<Enumeration> {
    let index_path = root.join("index").join("dataset.json");
    let text = std::fs::read_to_string(&index_path)
        .with_context(|| format!("read {}", index_path.display()))?;
    let index: BTreeMap<String, String> =
        serde_json::from_str(&text).context("parse index/dataset.json")?;
    let mut groups: BTreeMap<String, BTreeMap<String, String>> = BTreeMap::new();
    for (path, sha) in index {
        if let Some((dir, file)) = path.rsplit_once('/') {
            groups
                .entry(dir.to_string())
                .or_default()
                .insert(file.to_string(), sha);
        }
    }
    let mut en = Enumeration::default();
    for (dir, files) in groups {
        let name = dir.rsplit('/').next().unwrap_or_default().to_string();
        if crate::dataset_dir_hash(&name).is_none() {
            continue;
        }
        let rel = dir
            .split_once("quant/")
            .map_or(dir.clone(), |(_, r)| r.to_string());
        if rel.split('/').any(|c| c == QUARANTINE) {
            en.duplicates.push(Duplicate {
                path: rel,
                rule: "quarantine".into(),
                canonical: None,
                dataset_hash: crate::dataset_dir_hash(&name).map(str::to_string),
            });
            continue;
        }
        let class = rel.rsplit_once('/').map_or("", |(c, _)| c);
        let task = match class {
            "regression" => Some(Task::Regression),
            "binary" | "synthetic/binary" => Some(Task::Binary),
            "multiclass" => Some(Task::Multiclass),
            "multi_label" => Some(Task::MultiLabel),
            "timeseries" => Some(Task::Timeseries),
            _ => None,
        };
        let missing: Vec<&str> = FILES
            .iter()
            .copied()
            .filter(|f| !files.contains_key(*f))
            .collect();
        if !missing.is_empty() {
            en.rejects.push(RejectRecord {
                path: rel,
                reason: "incomplete".into(),
                detail: format!("index lacks {}", missing.join(",")),
            });
            continue;
        }
        let expected = [
            files["train.csv"].clone(),
            files["test.csv"].clone(),
            files["meta.json"].clone(),
        ];
        en.candidates.push(Candidate {
            relpath: rel,
            dir: root.join("dataset").join(&dir),
            name,
            task,
            expected: Some(expected),
            priority: 0,
        });
    }
    Ok(en)
}

pub fn enumerate(source: Source, root: &Path) -> Result<Enumeration> {
    match source {
        Source::Quant => enumerate_quant(root),
        Source::Mirror => enumerate_mirror(root),
    }
}

/// Hash every candidate's three files on `jobs` threads; results are indexed like `candidates`.
pub fn hash_all(candidates: &[Candidate], jobs: usize) -> Vec<std::io::Result<Hashed>> {
    let next = AtomicUsize::new(0);
    let results: Vec<Mutex<Option<std::io::Result<Hashed>>>> =
        (0..candidates.len()).map(|_| Mutex::new(None)).collect();
    let started = Instant::now();
    std::thread::scope(|s| {
        for _ in 0..jobs.max(1) {
            s.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= candidates.len() {
                    break;
                }
                let c = &candidates[i];
                let r = (|| -> std::io::Result<Hashed> {
                    let d = |f: &str| digest_file(&c.dir.join(f));
                    Ok(Hashed {
                        train: d("train.csv")?,
                        test: d("test.csv")?,
                        meta: d("meta.json")?,
                    })
                })();
                *results[i]
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(r);
                if (i + 1).is_multiple_of(5000) {
                    eprintln!(
                        "hashed {}/{} ({:.0}s)",
                        i + 1,
                        candidates.len(),
                        started.elapsed().as_secs_f64()
                    );
                }
            });
        }
    });
    results
        .into_iter()
        .map(|m| {
            m.into_inner()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .unwrap_or_else(|| Err(std::io::Error::other("not hashed")))
        })
        .collect()
}

pub struct Deduped {
    pub rows: Vec<CatalogRow>,
    pub duplicates: Vec<Duplicate>,
    pub rejects: Vec<RejectRecord>,
    pub blobs: HashMap<String, u64>,
}

/// Build rows in priority order, folding same-`dataset_hash` copies into `aliases`.
pub fn dedupe(
    source: Source,
    mut en: Enumeration,
    hashed: Vec<std::io::Result<Hashed>>,
) -> Deduped {
    let mut order: Vec<usize> = (0..en.candidates.len()).collect();
    order.sort_by(|&a, &b| {
        let (ca, cb) = (&en.candidates[a], &en.candidates[b]);
        (ca.priority, &ca.relpath).cmp(&(cb.priority, &cb.relpath))
    });
    let mut hashed: Vec<Option<std::io::Result<Hashed>>> = hashed.into_iter().map(Some).collect();
    let mut rows: Vec<CatalogRow> = Vec::new();
    let mut by_hash: HashMap<String, usize> = HashMap::new();
    let mut blobs: HashMap<String, u64> = HashMap::new();
    let ingested_at = now_rfc3339();
    for i in order {
        let c = &en.candidates[i];
        let h = match hashed[i]
            .take()
            .unwrap_or_else(|| Err(std::io::Error::other("not hashed")))
        {
            Ok(h) => h,
            Err(e) => {
                en.rejects.push(RejectRecord {
                    path: c.relpath.clone(),
                    reason: "io_error".into(),
                    detail: e.to_string(),
                });
                continue;
            }
        };
        if let Some(exp) = &c.expected {
            let got = [&h.train.sha256, &h.test.sha256, &h.meta.sha256];
            if let Some((f, e, g)) = FILES
                .iter()
                .zip(exp)
                .zip(got)
                .find(|((_, e), g)| e != g)
                .map(|((f, e), g)| (f, e, g))
            {
                en.rejects.push(RejectRecord {
                    path: c.relpath.clone(),
                    reason: "mirror_sha_mismatch".into(),
                    detail: format!("{f}: index {e} computed {g}"),
                });
                continue;
            }
        }
        let meta: serde_json::Value = match std::fs::read(c.dir.join("meta.json"))
            .map_err(|e| e.to_string())
            .and_then(|b| serde_json::from_slice(&b).map_err(|e| e.to_string()))
        {
            Ok(v) => v,
            Err(e) => {
                en.rejects.push(RejectRecord {
                    path: c.relpath.clone(),
                    reason: "meta_unparsable".into(),
                    detail: e,
                });
                continue;
            }
        };
        let Some(task) = c.task.or_else(|| task_from_meta(&meta)) else {
            en.rejects.push(RejectRecord {
                path: c.relpath.clone(),
                reason: "task_unknown".into(),
                detail: "no class directory and no meta.task_type".into(),
            });
            continue;
        };
        let ingest = Ingest {
            source: source.ingest_source(),
            relpath: c.relpath.clone(),
            ingested_at: ingested_at.clone(),
            tool_version: crate::TOOL_VERSION.into(),
        };
        let row = match build_row(&c.name, task, &meta, &h, ingest) {
            Ok(r) => r,
            Err(r) => {
                en.rejects.push(RejectRecord {
                    path: c.relpath.clone(),
                    reason: r.reason.into(),
                    detail: r.detail,
                });
                continue;
            }
        };
        if let Some(&idx) = by_hash.get(&row.dataset_hash) {
            let canon = &rows[idx];
            let same = canon.files.train.sha256 == h.train.sha256
                && canon.files.test.sha256 == h.test.sha256;
            let rule = if same {
                "dataset-hash-duplicate"
            } else {
                "dataset-hash-conflict"
            };
            en.duplicates.push(Duplicate {
                path: c.relpath.clone(),
                rule: rule.into(),
                canonical: canon.ingest.as_ref().map(|i| i.relpath.clone()),
                dataset_hash: Some(row.dataset_hash.clone()),
            });
            if same {
                rows[idx]
                    .aliases
                    .extend(aliases_for(&c.relpath, &h, source.ingest_source()));
            }
            continue;
        }
        for (_, f) in row.files.entries() {
            blobs.insert(f.sha256.clone(), f.bytes);
        }
        by_hash.insert(row.dataset_hash.clone(), rows.len());
        rows.push(row);
    }
    Deduped {
        rows,
        duplicates: en.duplicates,
        rejects: en.rejects,
        blobs,
    }
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    std::fs::write(path, serde_json::to_vec_pretty(value)?)
        .with_context(|| format!("write {}", path.display()))
}

fn read_ledger(path: &Path) -> HashSet<String> {
    let mut set = HashSet::new();
    if let Ok(f) = std::fs::File::open(path) {
        for line in std::io::BufReader::new(f).lines().map_while(Result::ok) {
            if let Ok(l) = serde_json::from_str::<LedgerLine>(&line) {
                if l.verified {
                    set.insert(l.sha256);
                }
            }
        }
    }
    set
}

async fn upload(
    store: &Store,
    rows: &[CatalogRow],
    en: &Enumeration,
    ledger_path: &Path,
    resume: HashSet<String>,
    summary: &mut Summary,
    ts: &str,
) -> Result<()> {
    use futures_util::stream::{self, StreamExt, TryStreamExt};
    let ledger = Mutex::new(
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(ledger_path)?,
    );
    let mut done: HashSet<String> = resume.clone();
    let mut jobs: Vec<(String, u64, PathBuf)> = Vec::new();
    for row in rows {
        let Some(ing) = &row.ingest else { continue };
        let _ = ing;
        for (name, f) in row.files.entries() {
            if done.insert(f.sha256.clone()) {
                let dir = en
                    .candidates
                    .iter()
                    .find(|c| c.name == row.id)
                    .map(|c| c.dir.clone())
                    .ok_or_else(|| anyhow!("no candidate for {}", row.id))?;
                jobs.push((f.sha256.clone(), f.bytes, dir.join(name)));
            } else if resume.contains(&f.sha256) {
                summary.skipped_by_ledger += 1;
            }
        }
    }
    let results: Vec<Result<PutOutcome>> =
        stream::iter(jobs.into_iter().map(|(sha, bytes, path)| {
            let ledger = &ledger;
            async move {
                let data = tokio::fs::read(&path)
                    .await
                    .with_context(|| format!("read {}", path.display()))?;
                let key = format!("blobs/sha256/{sha}");
                let outcome = store.put_if_absent(&key, data).await?;
                let size = store
                    .size(&key)
                    .await?
                    .ok_or_else(|| anyhow!("{key}: missing after put"))?;
                if size != bytes {
                    return Err(anyhow!("{key}: size {size} != local {bytes}"));
                }
                let line = LedgerLine {
                    sha256: sha,
                    bytes,
                    key,
                    outcome: format!("{outcome:?}").to_lowercase(),
                    verified: true,
                    at: now_rfc3339(),
                };
                let mut l = ledger
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                writeln!(l, "{}", serde_json::to_string(&line)?)?;
                Ok(outcome)
            }
        }))
        .buffer_unordered(32)
        .collect()
        .await;
    for r in results {
        match r? {
            PutOutcome::Created => summary.uploaded += 1,
            PutOutcome::Exists => summary.already_present += 1,
        }
    }
    // One PUT per dataset manifest (3,859 for the T1 tier), up to PARALLEL_GETS in flight instead
    // of one round trip at a time. Each manifest is its own key; `buffered` keeps input order, so
    // the error reported is still the first failing row's, and the catalog below is written only
    // after every manifest is in place, as before.
    stream::iter(rows.iter().map(|row| async move {
        store
            .put(
                &format!("datasets/{}/manifest.json", row.dataset_hash),
                serde_json::to_vec_pretty(row)?,
            )
            .await
    }))
    .buffered(crate::blobs::PARALLEL_GETS)
    .try_collect::<Vec<()>>()
    .await?;
    let next = match store.get_string("catalog/LATEST").await? {
        Some(v) => v.trim().trim_start_matches('v').parse::<u64>().unwrap_or(0) + 1,
        None => 1,
    };
    let mut jsonl = Vec::new();
    for row in rows {
        jsonl.extend(serde_json::to_vec(row)?);
        jsonl.push(b'\n');
    }
    store
        .put(&format!("catalog/v{next}/catalog.jsonl"), jsonl)
        .await?;
    store
        .put("catalog/LATEST", format!("v{next}\n").into_bytes())
        .await?;
    for name in [
        "duplicates.json",
        "rejects.json",
        "ledger.jsonl",
        "summary.json",
    ] {
        if let Ok(b) = std::fs::read(ledger_path.with_file_name(name)) {
            store.put(&format!("ingest/reports/{ts}/{name}"), b).await?;
        }
    }
    Ok(())
}

/// Run an ingest end to end and return the summary (also written to `<out>/summary.json`).
pub fn run(opts: &Options) -> Result<Summary> {
    let started = Instant::now();
    let ts = now_compact();
    std::fs::create_dir_all(&opts.out).with_context(|| format!("create {}", opts.out.display()))?;
    let mut en = enumerate(opts.source, &opts.root)?;
    if let Some(limit) = opts.limit {
        en.candidates.truncate(limit);
    }
    eprintln!(
        "enumerated {} candidates, {} duplicates by path rule, {} rejects ({:.1}s)",
        en.candidates.len(),
        en.duplicates.len(),
        en.rejects.len(),
        started.elapsed().as_secs_f64()
    );
    let candidates_n = en.candidates.len();
    let hashed = hash_all(&en.candidates, opts.jobs);
    let candidates_snapshot = Enumeration {
        candidates: en.candidates.clone(),
        duplicates: Vec::new(),
        rejects: Vec::new(),
    };
    let deduped = dedupe(opts.source, en, hashed);
    let mut summary = Summary {
        source: format!("{:?}", opts.source).to_lowercase(),
        root: opts.root.display().to_string(),
        dry_run: opts.dry_run,
        candidates: candidates_n,
        rows: deduped.rows.len(),
        distinct_blobs: deduped.blobs.len(),
        blob_bytes: deduped.blobs.values().sum(),
        duplicates: deduped.duplicates.len(),
        rejects: deduped.rejects.len(),
        out: opts.out.display().to_string(),
        ..Default::default()
    };
    for r in &deduped.rows {
        *summary
            .rows_by_task
            .entry(r.task.as_str().to_string())
            .or_default() += 1;
    }
    for d in &deduped.duplicates {
        *summary
            .duplicates_by_rule
            .entry(d.rule.clone())
            .or_default() += 1;
    }
    for r in &deduped.rejects {
        *summary
            .rejects_by_reason
            .entry(r.reason.clone())
            .or_default() += 1;
    }
    let mut jsonl = std::io::BufWriter::new(std::fs::File::create(opts.out.join("catalog.jsonl"))?);
    for row in &deduped.rows {
        serde_json::to_writer(&mut jsonl, row)?;
        jsonl.write_all(b"\n")?;
    }
    jsonl.flush()?;
    write_json(&opts.out.join("duplicates.json"), &deduped.duplicates)?;
    write_json(&opts.out.join("rejects.json"), &deduped.rejects)?;
    let ledger_path = opts.out.join("ledger.jsonl");
    let resume = if opts.resume {
        read_ledger(&ledger_path)
    } else {
        HashSet::new()
    };
    if !opts.resume && !ledger_path.exists() {
        std::fs::File::create(&ledger_path)?;
    }
    if !opts.dry_run {
        let url = opts
            .store_url
            .clone()
            .unwrap_or_else(crate::store::datasets_url);
        let store = Store::open(&url)?;
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?;
        summary.elapsed_s = started.elapsed().as_secs_f64();
        write_json(&opts.out.join("summary.json"), &summary)?;
        rt.block_on(upload(
            &store,
            &deduped.rows,
            &candidates_snapshot,
            &ledger_path,
            resume,
            &mut summary,
            &ts,
        ))?;
    }
    summary.elapsed_s = started.elapsed().as_secs_f64();
    write_json(&opts.out.join("summary.json"), &summary)?;
    Ok(summary)
}

/// Human-readable summary for the terminal.
pub fn describe(s: &Summary) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "source={} root={} dry_run={} elapsed={:.1}s\n",
        s.source, s.root, s.dry_run, s.elapsed_s
    ));
    out.push_str(&format!(
        "candidates={} rows={} distinct_blobs={} blob_bytes={}\n",
        s.candidates, s.rows, s.distinct_blobs, s.blob_bytes
    ));
    out.push_str(&format!("rows_by_task={:?}\n", s.rows_by_task));
    out.push_str(&format!(
        "duplicates={} by_rule={:?}\n",
        s.duplicates, s.duplicates_by_rule
    ));
    out.push_str(&format!(
        "rejects={} by_reason={:?}\n",
        s.rejects, s.rejects_by_reason
    ));
    if !s.dry_run {
        out.push_str(&format!(
            "uploaded={} already_present={} skipped_by_ledger={}\n",
            s.uploaded, s.already_present, s.skipped_by_ledger
        ));
    }
    out.push_str(&format!("reports: {}\n", s.out));
    out
}

/// Relative path helper re-exported for callers building alias paths.
pub fn rel(root: &Path, p: &Path) -> String {
    relpath(root, p)
}

#[allow(dead_code)]
fn _digest_type_check(_: FileDigest) {}
