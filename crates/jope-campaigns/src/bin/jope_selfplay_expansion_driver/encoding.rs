use super::*;

/// Ids of encode jobs scq still considers alive. `None` means scq could not be
/// read — callers must then assume a job MIGHT be live (never submit a
/// duplicate, never reclaim a marker) rather than guess.
pub(crate) fn live_encode_jobs() -> Option<HashSet<String>> {
    let out = sh(scq_str(), &["--json", "ls"]).ok()?;
    let value = serde_json::from_str::<Value>(&out).ok()?;
    let binding = Vec::new();
    Some(
        value
            .pointer("/Jobs/jobs")
            .and_then(Value::as_array)
            .unwrap_or(&binding)
            .iter()
            .filter(|job| {
                matches!(
                    job.get("state").and_then(Value::as_str),
                    Some("queued" | "starting" | "dispatching" | "running" | "preempting" | "held")
                )
            })
            .filter_map(|job| job.get("id").and_then(Value::as_str))
            .filter(|id| {
                id.starts_with("jope-selfplay-breadth-encode-") || id.starts_with("jope-binexp-")
            })
            .map(str::to_owned)
            .collect(),
    )
}

pub(crate) fn live_encode_job(live: &Option<HashSet<String>>) -> bool {
    // Unknown scq state is treated as "a job is live": refusing to submit is
    // recoverable next cycle, double-submitting the same chunk is not.
    live.as_ref().map(|ids| !ids.is_empty()).unwrap_or(true)
}

/// Return `listed:*` markers to NULL so `submit_encode_chunk` can re-list them.
///
/// A dataset is stranded when its chunk job is gone from scq but the encode
/// never landed (sidecars_complete still 0). That is the steady state after any
/// encode failure, and nothing in the driver ever cleared a marker: when the
/// b2c05..b2c16 series died on the hyperion CPU-fallback timeout and then aged
/// out of scq, all 22,747 remaining binary datasets kept their markers, the
/// candidate pool became permanently empty, and the binary lane went silent
/// while regression kept flowing. Clearing markers here is what makes "new
/// datasets always being added" hold for binary without a human.
pub(crate) fn reclaim_stale_encode_markers(
    conn: &Connection,
    live: &Option<HashSet<String>>,
    log: &mut Vec<Value>,
) -> Result<()> {
    let Some(live_ids) = live else { return Ok(()) };
    // Outputs already sitting in the fetch dir encoded fine and are merely
    // waiting for install_batch to flip sidecars_complete. They must not burn
    // retry budget in that window (submit_encode_chunk skips fetched ids
    // anyway, so re-listing them would be a no-op with a cost).
    let fetched: HashSet<String> = std::fs::read_dir(Path::new(fetch_binary_str()).join("targets"))
        .map(|entries| {
            entries
                .flatten()
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .collect()
        })
        .unwrap_or_default();
    let mut statement = conn.prepare(
        "SELECT dataset_hash, encode_job_state, encode_attempts FROM datasets
          WHERE task='binary' AND encode_job_state LIKE 'listed:%'
            AND sidecars_complete=0 AND catalog_active=0",
    )?;
    let rows: Vec<(String, String, i64)> = statement
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?
        .filter_map(|r| r.ok())
        .collect();
    drop(statement);
    let mut reclaimed = 0_usize;
    let mut exhausted = 0_usize;
    let tx = conn.unchecked_transaction()?;
    for (hash, marker, attempts) in rows {
        // Marker bodies have varied across driver generations ("b2c4",
        // "binexp-b2c05"); match by containment against live job ids instead of
        // reconstructing an exact id, so a live chunk is never reclaimed.
        let body = marker.trim_start_matches("listed:");
        if live_ids.iter().any(|id| id.contains(body)) || fetched.contains(&hash) {
            continue;
        }
        if attempts + 1 >= MAX_ENCODE_ATTEMPTS {
            tx.execute(
                "UPDATE datasets SET encode_job_state='exhausted', encode_attempts=encode_attempts+1
                  WHERE task='binary' AND dataset_hash=?",
                [&hash],
            )?;
            exhausted += 1;
        } else {
            tx.execute(
                "UPDATE datasets SET encode_job_state=NULL, encode_attempts=encode_attempts+1
                  WHERE task='binary' AND dataset_hash=?",
                [&hash],
            )?;
            reclaimed += 1;
        }
    }
    tx.commit()?;
    if reclaimed > 0 || exhausted > 0 {
        log.push(json!({
            "step": "encode_markers_reclaimed",
            "reclaimed": reclaimed,
            "exhausted": exhausted,
        }));
    }
    Ok(())
}

pub(crate) fn submit_encode_chunk(conn: &Connection) -> Result<Option<EncodeChunkSubmission>> {
    // Un-encoded = binary datasets with data but incomplete sidecars, not active,
    // not already present in the fetch dir.
    let fetched: HashSet<String> = std::fs::read_dir(Path::new(fetch_binary_str()).join("targets"))
        .map(|entries| {
            entries
                .flatten()
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .collect()
        })
        .unwrap_or_default();
    let mut statement = conn.prepare(
        "SELECT dataset_hash, encode_attempts FROM datasets
          WHERE task='binary' AND has_train_test_meta=1 AND sidecars_complete=0
            AND catalog_active=0 AND sealed=0 AND encode_job_state IS NULL ORDER BY dataset_hash",
    )?;
    let pending: Vec<(String, i64)> = statement
        .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)))?
        .filter_map(|r| r.ok())
        .filter(|(id, _)| !fetched.contains(id))
        .take(ENCODE_CHUNK)
        .collect();
    if pending.is_empty() {
        return Ok(None);
    }
    let ids = pending.iter().map(|(id, _)| id.clone()).collect::<Vec<_>>();
    let attempt = pending
        .iter()
        .map(|(_, attempts)| attempts + 1)
        .max()
        .unwrap_or(1);
    // Rows carry content pins; sha256 via coreutils in batches to stay dependency-free.
    let registry = Path::new(registry_str()).join("binary");
    let mut digests: BTreeMap<String, String> = BTreeMap::new();
    for chunk in ids.chunks(400) {
        let mut args: Vec<String> = Vec::with_capacity(chunk.len() * 2);
        for id in chunk {
            args.push(
                registry
                    .join(id)
                    .join("train.csv")
                    .to_string_lossy()
                    .into_owned(),
            );
            args.push(
                registry
                    .join(id)
                    .join("test.csv")
                    .to_string_lossy()
                    .into_owned(),
            );
        }
        let refs: Vec<&str> = args.iter().map(String::as_str).collect();
        for line in sh("sha256sum", &refs)?.lines() {
            if let Some((digest, path)) = line.split_once("  ") {
                digests.insert(path.trim().to_owned(), digest.trim().to_owned());
            }
        }
    }
    let rows: Vec<Value> = ids
        .iter()
        .map(|id| {
            let train = registry.join(id).join("train.csv");
            let test = registry.join(id).join("test.csv");
            json!({
                "id": id,
                "train_sha256": digests.get(&train.to_string_lossy().into_owned()),
                "test_sha256": digests.get(&test.to_string_lossy().into_owned()),
            })
        })
        .collect();
    let mut out = json!({
        "schema": "EncodingChunk.v2",
        "group": "binary-expansion",
        "src_root": format!("{registry_str}/binary", registry_str = registry_str()),
        "out_subdir": "binary-expansion",
        "count": ids.len(),
        "ids": rows,
        "lite": false,
    });
    let runner = bundled_encoding_asset("runner.sh")?;
    let driver = bundled_encoding_asset("encode_batch.py")?;
    let encode_external = bundled_encoding_asset("encode_external.py")?;
    let encoder_identity = sha256_bytes(
        format!(
            "runtime-bundle-v3-encoder|{}|{}|{}",
            sha256_file(&runner)?,
            sha256_file(&driver)?,
            sha256_file(&encode_external)?,
        )
        .as_bytes(),
    );
    let chunk_sha256 = sha256_bytes(&serde_json::to_vec(&json!({
        "schema": "EncodingChunk.v2",
        "encoder_identity_sha256": encoder_identity,
        "datasets": rows,
    }))?);
    let job_id = format!(
        "jope-selfplay-breadth-encode-{}-a{attempt}",
        &chunk_sha256[..16]
    );
    out["schema"] = json!("EncodingChunk.v2");
    out["chunk_sha256"] = json!(chunk_sha256);
    out["encoder_identity_sha256"] = json!(encoder_identity);
    out["attempt_number"] = json!(attempt);
    out["job_identity"] = json!(job_id);
    out["project"] = json!("jope-selfplay-breadth-encode");
    out["required_node"] = json!("xbabe3");
    out["preemptible"] = json!(true);
    out["reserved_cpu_cores"] = json!(4);
    let listing = format!(
        "{}/_batch_manifests/breadth-{}-a{attempt}.v2.json",
        encodings_str(),
        &chunk_sha256[..16]
    );
    let mut listing_bytes = serde_json::to_vec(&out)?;
    listing_bytes.push(b'\n');
    publish_immutable(Path::new(&listing), &listing_bytes)?;
    let spec_directory = Path::new(root_str()).join("specs/breadth-encoding");
    std::fs::create_dir_all(&spec_directory)?;
    let spec = spec_directory.join(format!("{job_id}.toml"));
    let pythonpath = runtime_bundle_root()?.join("pytorch");
    // The bundled runtime FIRST so a bundled module always wins, then the
    // foundation-expert tree. See `foundation_expert_root_str`.
    let job_pythonpath = format!(
        "{}:{foundation_expert_root_str}",
        pythonpath.to_string_lossy(),
        foundation_expert_root_str = foundation_expert_root_str()
    );
    let mut lines = vec![
        "[job]".to_owned(),
        format!("id = \"{job_id}\""),
        "project = \"jope-selfplay-breadth-encode\"".to_owned(),
        "tier = \"normal\"".to_owned(),
        "command = \"/bin/sh\"".to_owned(),
        "args = [\"{input:runner}\", \"{input:driver}\", \"{input:listing}\", \"{output:receipt}\", \"{input:encode_external}\"]".to_owned(),
        String::new(),
        "[resource]".to_owned(),
        "gpus_per_task = 1".to_owned(),
        "cpus_reserved = 4".to_owned(),
        "omp_threads = 4".to_owned(),
        "ram_mb = 16384".to_owned(),
        "gang_size = 1".to_owned(),
        "placement = \"single_node\"".to_owned(),
        String::new(),
        "[[dependencies.smoke]]".to_owned(),
        "name = \"xbabe3 prime encode environment\"".to_owned(),
        "command = \"/usr/bin/test\"".to_owned(),
        format!(
            "args = [\"-f\", \"{}\"]",
            paths::legacy_tree("jope-scratch/binary-expansion/prime_xbabe3.toml").display()
        ),
        "timeout_secs = 5".to_owned(),
        String::new(),
        "[[dependencies.smoke]]".to_owned(),
        "name = \"xbabe3 RuntimeBundle Python package\"".to_owned(),
        "command = \"/usr/bin/env\"".to_owned(),
        format!(
            "args = [{}, \"/usr/bin/python3\", \"-c\", \"import jope; import python.elo_finetune.model_io\"]",
            serde_json::to_string(&format!("PYTHONPATH={job_pythonpath}"))?
        ),
        "timeout_secs = 30".to_owned(),
        String::new(),
        "[env]".to_owned(),
        "OMP_NUM_THREADS = \"4\"".to_owned(),
        format!("JOPE_RUNTIME_PYTHONPATH = \"{job_pythonpath}\""),
        String::new(),
        "[preempt]".to_owned(),
        "preemptible = true".to_owned(),
        "checkpoint = \"none\"".to_owned(),
        "resume = \"rerun\"".to_owned(),
        String::new(),
        "[limits]".to_owned(),
        "max_runtime_hours = 12".to_owned(),
        "max_restarts = 2".to_owned(),
        String::new(),
    ];
    append_spec_input(&mut lines, "runner", &runner, "runner.sh")?;
    append_spec_input(&mut lines, "driver", &driver, "encode_batch.py")?;
    append_spec_input(
        &mut lines,
        "encode_external",
        &encode_external,
        "encode_external.py",
    )?;
    append_spec_input(&mut lines, "listing", Path::new(&listing), "listing.json")?;
    lines.extend([
        "[[output]]".to_owned(),
        "name = \"receipt\"".to_owned(),
        "source = \"receipt\"".to_owned(),
        "directory = true".to_owned(),
        String::new(),
    ]);
    let spec_bytes = lines.join("\n").into_bytes();
    publish_equivalent_bundle_spec(&spec, &spec_bytes)?;
    let submission = scq_check_submit(&spec, &job_id)?;
    // Each id is listed at most once; ids whose source is absent on the encode
    // node PREP-FAIL there and must not be re-listed forever. Successful ones
    // flip sidecars_complete via the fetched outputs regardless of this marker.
    let tx = conn.unchecked_transaction()?;
    for id in &ids {
        tx.execute(
            "UPDATE datasets SET encode_job_state=? WHERE task='binary' AND dataset_hash=?",
            rusqlite::params![format!("listed:{job_id}"), id],
        )?;
    }
    tx.commit()?;
    Ok(Some(submission))
}

/// Exhaustion must be a BACKOFF, never a death sentence. Datasets get parked
/// after MAX_ENCODE_ATTEMPTS, but placement failures can burn the whole budget
/// without a dataset ever reaching a node that has the encode environment
/// (scq picks the node and the spec's hostname dependency does not constrain
/// placement — b2c17 attempt 1 landed on xbabe1, which has no prime config).
/// Once nothing submittable is left, un-park everything and start over: broken
/// datasets are only ever retried after the healthy backlog is exhausted, so
/// this cannot starve real work, and the binary lane can never permanently die.
pub(crate) fn unpark_exhausted_if_starved(conn: &Connection, log: &mut Vec<Value>) -> Result<()> {
    let submittable: i64 = conn.query_row(
        "SELECT COUNT(*) FROM datasets WHERE task='binary' AND has_train_test_meta=1
           AND sidecars_complete=0 AND catalog_active=0 AND encode_job_state IS NULL",
        [],
        |r| r.get(0),
    )?;
    if submittable > 0 {
        return Ok(());
    }
    let unparked = conn.execute(
        "UPDATE datasets SET encode_job_state=NULL, encode_attempts=0
          WHERE task='binary' AND encode_job_state='exhausted'",
        [],
    )?;
    if unparked > 0 {
        log.push(json!({"step": "encode_exhausted_unparked", "datasets": unparked}));
    }
    Ok(())
}

pub(crate) fn fetch_root(task: &str) -> PathBuf {
    if task == "binary" {
        PathBuf::from(fetch_binary_str())
    } else {
        paths::flywheel_encodings_root().join("_fetch/regression-expansion")
    }
}

pub(crate) fn hash_directory(path: &Path) -> Result<String> {
    let mut files = Vec::new();
    let mut stack = vec![path.to_path_buf()];
    while let Some(directory) = stack.pop() {
        for entry in std::fs::read_dir(directory)?.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.is_file() {
                files.push(path);
            }
        }
    }
    files.sort();
    let mut identity = Vec::new();
    for file in files {
        identity.extend_from_slice(
            file.strip_prefix(path)
                .unwrap_or(&file)
                .to_string_lossy()
                .as_bytes(),
        );
        identity.push(0);
        identity.extend_from_slice(sha256_file(&file)?.as_bytes());
        identity.push(b'\n');
    }
    Ok(sha256_bytes(&identity))
}

pub(crate) fn encoding_chunk_listing(job_id: &str) -> Result<Option<Value>> {
    let directory = paths::flywheel_encodings_root().join("_batch_manifests");
    for entry in std::fs::read_dir(directory)?.flatten() {
        let path = entry.path();
        if path.extension().and_then(|value| value.to_str()) != Some("json") {
            continue;
        }
        let Ok(value) = read_json(&path) else {
            continue;
        };
        if value.get("schema").and_then(Value::as_str) == Some("EncodingChunk.v2")
            && value.get("job_identity").and_then(Value::as_str) == Some(job_id)
        {
            return Ok(Some(value));
        }
    }
    Ok(None)
}

pub(crate) fn publish_encoding_receipt(receipt: &Value) -> Result<()> {
    let dataset_content = receipt
        .pointer("/dataset/content_sha256")
        .and_then(Value::as_str)
        .context("encoding receipt lacks dataset content identity")?;
    let attempt = receipt
        .get("attempt_number")
        .and_then(Value::as_u64)
        .context("encoding receipt lacks attempt number")?;
    let job = receipt
        .get("job_identity")
        .and_then(Value::as_str)
        .context("encoding receipt lacks job identity")?;
    let directory = Path::new(root_str()).join("ops/encoding-adoption-v2");
    std::fs::create_dir_all(&directory)?;
    let path = directory.join(format!(
        "{}-a{}-{}.json",
        dataset_content,
        attempt,
        &sha256_bytes(job.as_bytes())[..16]
    ));
    let mut bytes = serde_json::to_vec(receipt)?;
    bytes.push(b'\n');
    match std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
    {
        Ok(mut file) => {
            use std::io::Write;
            file.write_all(&bytes)?;
            file.sync_data()?;
            let mut events = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(Path::new(root_str()).join("ops/encoding-adoption-v2.jsonl"))?;
            events.write_all(&bytes)?;
            events.sync_data()?;
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            if std::fs::read(&path)? != bytes {
                bail!(
                    "immutable encoding adoption receipt drift: {}",
                    path.display()
                );
            }
        }
        Err(error) => return Err(error.into()),
    }
    Ok(())
}

pub(crate) fn emit_chunk_adoption_receipts(
    job_id: &str,
    job_state: &str,
    task: &str,
    root: &Path,
) -> Result<usize> {
    let Some(listing) = encoding_chunk_listing(job_id)? else {
        return Ok(0);
    };
    let encoder = listing
        .get("encoder_identity_sha256")
        .and_then(Value::as_str)
        .context("EncodingChunk.v2 lacks encoder identity")?;
    let attempt = listing
        .get("attempt_number")
        .and_then(Value::as_u64)
        .context("EncodingChunk.v2 lacks attempt number")?;
    let rows = listing
        .get("ids")
        .and_then(Value::as_array)
        .context("EncodingChunk.v2 lacks dataset identities")?;
    let registry = Path::new(registry_str()).join(task);
    let mut adopted = 0;
    for row in rows {
        let dataset_id = row
            .get("id")
            .and_then(Value::as_str)
            .context("EncodingChunk.v2 dataset lacks id")?;
        let train_sha256 = row
            .get("train_sha256")
            .and_then(Value::as_str)
            .context("EncodingChunk.v2 dataset lacks train hash")?;
        let test_sha256 = row
            .get("test_sha256")
            .and_then(Value::as_str)
            .context("EncodingChunk.v2 dataset lacks test hash")?;
        let train = registry.join(dataset_id).join("train.csv");
        let test = registry.join(dataset_id).join("test.csv");
        if sha256_file(&train)? != train_sha256 || sha256_file(&test)? != test_sha256 {
            bail!("{job_id}: registry source content changed for {dataset_id}");
        }
        let dataset_content_sha256 =
            sha256_bytes(format!("{train_sha256}:{test_sha256}").as_bytes());
        let target = root.join("targets").join(dataset_id);
        let foundation = root.join("foundation").join(format!("{dataset_id}.npy"));
        let disposition = if target.is_dir() {
            adopted += 1;
            json!({
                "kind": "succeeded",
                "encoded_train_sha256": hash_directory(&target)?,
                "encoded_test_sha256": if foundation.is_file() {
                    sha256_file(&foundation)?
                } else {
                    hash_directory(&target)?
                },
            })
        } else {
            json!({
                "kind": "failed",
                "error_class": format!("{job_state}:missing_dataset_output"),
            })
        };
        publish_encoding_receipt(&json!({
            "schema": "EncodingAdoptionReceipt.v2",
            "dataset": {
                "dataset_id": dataset_id,
                "content_sha256": dataset_content_sha256,
            },
            "attempt_number": attempt,
            "job_identity": job_id,
            "encoder_identity_sha256": encoder,
            "disposition": disposition,
            "recorded_at_ms": now_ms(),
        }))?;
    }
    Ok(adopted)
}

/// Pull outputs of completed encode chunks into the per-task fetch dirs.
/// Encoding and fetching are deliberately OUTSIDE the guardrail/valve — the
/// valve paces catalog admission only, never encode throughput.
pub(crate) fn reconcile_encode_fetches(log: &mut Vec<Value>) -> Result<()> {
    let Ok(out) = sh(scq_str(), &["--json", "ls"]) else {
        return Ok(());
    };
    let Ok(value) = serde_json::from_str::<Value>(&out) else {
        return Ok(());
    };
    let binding = Vec::new();
    for job in value
        .pointer("/Jobs/jobs")
        .and_then(Value::as_array)
        .unwrap_or(&binding)
    {
        let id = job.get("id").and_then(Value::as_str).unwrap_or("");
        let job_state = job
            .get("state")
            .and_then(Value::as_str)
            .unwrap_or("unknown");
        if !(id.starts_with("jope-selfplay-breadth-encode-")
            || id.starts_with("jope-binexp-")
            || id.starts_with("jope-regexp-"))
        {
            continue;
        }
        if !matches!(job_state, "completed" | "failed" | "cancelled") {
            continue;
        }
        let task =
            if id.starts_with("jope-binexp-") || id.starts_with("jope-selfplay-breadth-encode-") {
                "binary"
            } else {
                "regression"
            };
        let root = fetch_root(task);
        let marker = root.join(format!(".fetched-{id}"));
        if marker.exists() {
            continue;
        }
        std::fs::create_dir_all(root.join("targets"))?;
        std::fs::create_dir_all(root.join("foundation"))?;
        let tmp = root.join(format!(".fetch-tmp-{id}"));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp)?;
        if sh(
            scq_str(),
            &["fetch", id, "--output", &tmp.to_string_lossy()],
        )
        .is_err()
        {
            let _ = std::fs::remove_dir_all(&tmp);
            continue;
        }
        let mut moved = 0_usize;
        let mut stack = vec![tmp.clone()];
        while let Some(dir) = stack.pop() {
            for entry in std::fs::read_dir(&dir)?.flatten() {
                let path = entry.path();
                let name = entry.file_name().to_string_lossy().into_owned();
                if path.is_dir() && name == "targets" {
                    for ds in std::fs::read_dir(&path)?.flatten() {
                        let to = root.join("targets").join(ds.file_name());
                        if !to.exists() && std::fs::rename(ds.path(), &to).is_ok() {
                            moved += 1;
                        }
                    }
                } else if path.is_dir() && name == "foundation" {
                    for f in std::fs::read_dir(&path)?.flatten() {
                        let to = root.join("foundation").join(f.file_name());
                        if !to.exists() {
                            let _ = std::fs::rename(f.path(), &to);
                        }
                    }
                } else if path.is_dir() {
                    stack.push(path);
                }
            }
        }
        let _ = std::fs::remove_dir_all(&tmp);
        let adopted = emit_chunk_adoption_receipts(id, job_state, task, &root)?;
        std::fs::write(&marker, b"ok")?;
        log.push(json!({
            "step": "encode_fetched",
            "job": id,
            "terminal_state": job_state,
            "datasets_moved": moved,
            "datasets_adopted": adopted,
        }));
    }
    Ok(())
}
