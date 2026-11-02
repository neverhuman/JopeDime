use super::*;

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
