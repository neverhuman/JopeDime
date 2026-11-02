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
