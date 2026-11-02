use super::*;

pub(crate) fn installable(conn: &Connection, task: &str) -> Result<Vec<String>> {
    let fetched: HashSet<String> = std::fs::read_dir(fetch_root(task).join("targets"))
        .map(|entries| {
            entries
                .flatten()
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .collect()
        })
        .unwrap_or_default();
    let mut statement = conn.prepare(
        "SELECT dataset_hash FROM datasets
          WHERE task=? AND catalog_active=0 AND sealed=0 AND sidecars_complete=0
          ORDER BY dataset_hash",
    )?;
    let ids: Vec<String> = statement
        .query_map([task], |r| r.get::<_, String>(0))?
        .filter_map(|r| r.ok())
        .filter(|id| fetched.contains(id))
        .collect();
    Ok(ids)
}

pub(crate) fn curate_tranche(
    root: &Path,
    conn: &Connection,
    task: &str,
    tranche_file: &Path,
) -> Result<usize> {
    let inv = read_json(&root.join("catalog/v2/inventory-adaptive-v1.json"))?;
    let binding = Vec::new();
    let entries = inv
        .get("entries")
        .and_then(Value::as_array)
        .unwrap_or(&binding);
    // Regression foundation .npy files are produced by foundation_backfill.py
    // AFTER curation (run_tranche), and prune_tranche.py drops any entry whose
    // .npy is still missing before activation. Requiring foundation_npy=1 here
    // therefore starved curation to zero the moment the pre-encoded pool was
    // fully activated (the silent 0-tranche stall of 2026-07-27/28: 6,412
    // fully-encoded = 6,412 active, 17,557 sidecar-complete datasets waiting).
    // Binary has no backfill lane — its .npy arrives with the encode fetch —
    // so the gate stays for binary.
    let sql = if task == "regression" {
        "SELECT 1 FROM datasets WHERE task=? AND dataset_hash=? AND sidecars_complete=1
            AND catalog_active=0 AND sealed=0"
    } else {
        "SELECT 1 FROM datasets WHERE task=? AND dataset_hash=? AND sidecars_complete=1
            AND foundation_npy=1 AND catalog_active=0 AND sealed=0"
    };
    let mut statement = conn.prepare(sql)?;
    let mut selected = Vec::new();
    for e in entries {
        if e.get("task").and_then(Value::as_str) != Some(task) {
            continue;
        }
        let hash = e.get("dataset_hash").and_then(Value::as_str).unwrap_or("");
        if statement.query_row([task, hash], |_| Ok(())).is_ok() {
            selected.push(e.clone());
            if selected.len() >= TRANCHE_CAP {
                break;
            }
        }
    }
    std::fs::write(tranche_file, serde_json::to_vec(&selected)?)?;
    Ok(selected.len())
}

pub(crate) fn prune_tranche(tranche_file: &Path, task: &str) -> Result<Value> {
    let entries = read_json(tranche_file)?;
    let entries = entries
        .as_array()
        .context("tranche inventory must be a JSON array")?;
    let mut kept = Vec::with_capacity(entries.len());
    let mut dropped = Vec::new();
    for entry in entries {
        let dataset = entry
            .get("dataset_id")
            .and_then(Value::as_str)
            .context("tranche entry lacks dataset_id")?;
        let cache = Path::new(encodings_str())
            .join(format!("{task}-expanded"))
            .join(dataset);
        let mut reason = None;
        let foundation = Path::new(foundation_str()).join(format!("{dataset}.npy"));
        if !foundation.is_file() || foundation.metadata()?.len() == 0 {
            reason = Some("missing_or_empty_foundation_npy".to_owned());
        } else {
            for sidecar in SIDECARS {
                let path = cache.join(sidecar);
                if !path.is_file() || path.metadata()?.len() == 0 {
                    reason = Some(format!("empty_or_missing:{sidecar}"));
                    break;
                }
            }
        }
        if let Some(reason) = reason {
            dropped.push(json!({"dataset_id": dataset, "reason": reason}));
        } else {
            kept.push(entry.clone());
        }
    }
    let temporary = tranche_file.with_extension(format!("tmp-{}", std::process::id()));
    std::fs::write(&temporary, serde_json::to_vec(&kept)?)?;
    std::fs::rename(temporary, tranche_file)?;
    Ok(json!({"kept": kept.len(), "dropped": dropped}))
}

pub(crate) fn run_tranche(
    root: &Path,
    conn: &Connection,
    task: &str,
    log: &mut Vec<Value>,
) -> Result<Option<String>> {
    let acts = activations(root)?;
    let prefix = if task == "regression" { 'r' } else { 'b' };
    let tranche_id = next_tranche_id(&acts, prefix);
    let tranche_file = root.join(format!(
        "catalog/v2/tranche-{}.inventory.json",
        &tranche_id["expanded-".len()..]
    ));
    let selected = curate_tranche(root, conn, task, &tranche_file)?;
    if selected == 0 {
        // Log the empty selection: four clean-guardrail ticks stalled silently
        // (log: []) before this receipt line existed.
        log.push(json!({"step": "tranche_empty", "task": task, "tranche": tranche_id}));
        return Ok(None);
    }
    log.push(json!({"step": "curated", "tranche": tranche_id, "selected": selected}));
    if task == "regression" {
        // Foundation backfill (torch — the sanctioned Python boundary).
        let foundation_backfill = bundled_encoding_asset("foundation_backfill.py")?;
        sh(
            "nice",
            &[
                "-n",
                "15",
                "python3",
                &foundation_backfill.to_string_lossy(),
                &tranche_file.to_string_lossy(),
            ],
        )?;
    }
    let tool = bundled_binary("jope_expanded_catalog")?;
    let tool = tool.to_string_lossy();
    sh(
        &tool,
        &[
            "materialize",
            "--inventory",
            &tranche_file.to_string_lossy(),
            "--task",
            task,
        ],
    )?;
    let pruned = prune_tranche(&tranche_file, task)?;
    log.push(json!({"step": "pruned", "detail": pruned}));
    // Curation no longer pre-requires foundation_npy for regression, so a
    // wholesale backfill failure can prune the tranche to zero — never hand an
    // empty inventory to activate.
    let remaining = read_json(&tranche_file)?
        .as_array()
        .map(Vec::len)
        .unwrap_or(0);
    if remaining == 0 {
        log.push(json!({"step": "tranche_all_pruned", "task": task, "tranche": tranche_id}));
        return Ok(None);
    }
    sh(
        &tool,
        &[
            "activate",
            "--inventory",
            &tranche_file.to_string_lossy(),
            "--tranche-id",
            &tranche_id,
            "--limit",
            "512",
        ],
    )?;
    let merged = sh(&tool, &["merge-live"])?;
    log.push(json!({"step": "merged", "detail": merged.trim()}));
    Ok(Some(tranche_id))
}

pub(crate) fn install_batch(
    root: &Path,
    task: &str,
    ids: &[String],
    log: &mut Vec<Value>,
) -> Result<()> {
    let fetch = fetch_root(task);
    let fetched_foundation = fetch.join("foundation");
    let mut verified = 0_usize;
    for id in ids {
        let from = fetched_foundation.join(format!("{id}.npy"));
        if !from.is_file() {
            continue;
        }
        let published = Path::new(foundation_str()).join(format!("{id}.npy"));
        if !published.is_file() {
            bail!(
                "foundation V1 is immutable and has no object for {id}; keep expansion disabled until Encoder V2 publishes a namespaced object"
            );
        }
        if sha256_file(&from)? != sha256_file(&published)? {
            bail!("foundation V1 identity collision for {id}");
        }
        verified += 1;
    }
    let receipts = root.join("ops/sidecar-install-driver.v1.jsonl");
    let tool = bundled_binary("jope_expanded_catalog")?;
    sh(
        &tool.to_string_lossy(),
        &[
            "install-sidecars",
            "--from",
            &fetch.join("targets").to_string_lossy(),
            "--task",
            task,
            "--receipts",
            &receipts.to_string_lossy(),
        ],
    )?;
    log.push(json!({"step": "install", "task": task, "sidecar_batch": ids.len(), "foundation_v1_verified": verified}));
    Ok(())
}

// ---------------------------------------------------------------------------
// Cycle
// ---------------------------------------------------------------------------

pub(crate) fn run_cycle(root: &Path) -> Result<Value> {
    reject_v1_quarantine_window(Path::new(encodings_str()))?;
    let conn = open_state_db(root)?;
    let (scanned, _) = refresh_filesystem_state(&conn, root)?;
    let exclusion_receipts_created = emit_encoding_exclusion_receipts(&conn, root)?;
    let obs_rows = refresh_observation_state(&conn, root).unwrap_or(0);
    let acts = activations(root)?;
    let coverage_for = |prefix: char| -> Result<Option<f64>> {
        Ok(match latest_tranche_for(&acts, prefix) {
            Some((_, catalog)) if catalog.is_file() => Some(tranche_coverage(&conn, &catalog)?),
            _ => None,
        })
    };
    let coverage_regression = coverage_for('r')?;
    let coverage_binary = coverage_for('b')?;
    let mut log: Vec<Value> = Vec::new();
    let disk_paths = vec![paths::jope_state(), PathBuf::from(encodings_str())];
    let disk_block = disk_hysteresis_block(root, &disk_paths)?;
    // Encoding is valve-free by design: fetch completed chunk outputs and keep
    // the encode series topped up BEFORE any guardrail/valve consideration.
    //
    // The whole encode lane is NON-FATAL to the cycle. It used to propagate
    // with `?`, which meant one bad `scq` call took the regression lane down
    // with it: run_cycle returned Err before ever reaching the guardrail /
    // tranche / activation block below, so an unrelated binary-side hiccup
    // silently cost that hour's dataset activation. That is now impossible.
    // It also has to be non-fatal because the encode spec declares a real
    // dependency: scq accepts a dependency-declared job ONLY while a
    // satisfying node has free capacity, so `scq check` exiting 1 is a routine
    // "xbabe3 is busy, try next hour" state, not an error worth losing a
    // tranche over.
    let mut submitted_chunk = None;
    let encode_lane = if let Some(reason) = &disk_block {
        log.push(json!({"step": "encode_lane_disk_paused", "reason": reason}));
        Ok(())
    } else {
        (|| -> Result<()> {
            reconcile_encode_fetches(&mut log)?;
            let live = live_encode_jobs();
            // Return stranded listings before deciding whether to submit, so a dead
            // chunk is re-listed on the very same cycle that noticed it died.
            reclaim_stale_encode_markers(&conn, &live, &mut log)?;
            unpark_exhausted_if_starved(&conn, &mut log)?;
            // Rule (d): top-up submission — with the full series queued upfront this
            // only fires if the queue ever drains (e.g. after failures).
            if !live_encode_job(&live) {
                submitted_chunk = match pending_breadth_admission(root)? {
                    Some(pending) => Some(pending),
                    None => submit_encode_chunk(&conn)?,
                };
                if let Some(submission) = &submitted_chunk {
                    log.push(json!({
                        "step": if submission.disposition == "submitted" {
                            "encode_chunk_submitted"
                        } else {
                            "encode_chunk_capacity_backoff"
                        },
                        "job": submission.job_id,
                        "disposition": submission.disposition,
                        "reason": submission.backoff_reason,
                    }));
                }
            }
            Ok(())
        })()
    };
    if let Err(err) = encode_lane {
        // Recorded, never raised: the receipt is the only place this is
        // visible, so it must say plainly what failed.
        log.push(json!({
            "step": "encode_lane_deferred",
            "detail": format!("{err:#}"),
        }));
    }
    let guards = Guardrails {
        alarms_path: root.join("ops/health-alarms.v1.json"),
        keeper_path: root.join("ops/queue-keeper-v3.json"),
        waves_dir: root.join("waves"),
        disk_paths,
    };
    // Shared guards (alarms/keeper/waves/disk) gate everything; the valve is
    // evaluated per task against that task's own last tranche.
    let blocked = disk_block.or_else(|| guardrail_block(&guards, None));
    let valve_shut = |cov: Option<f64>| -> Option<String> {
        cov.filter(|c| *c < COVERAGE_VALVE)
            .map(|c| format!("coverage valve: previous tranche at {c:.2} < {COVERAGE_VALVE}"))
    };
    let mut tranche = None;
    if let Some(reason) = &blocked {
        log.push(json!({"step": "tranche_skipped", "reason": reason}));
    } else {
        // Rule (b): binary install+tranche when enough fetched encodes await.
        // Binary is preferred but must never STARVE regression: with per-task
        // valves a shut binary valve is a routine state, and falling through
        // keeps the regression lane widening meanwhile. Still at most one
        // tranche per cycle.
        let binary_ready = installable(&conn, "binary")?;
        let binary_valve = valve_shut(coverage_binary);
        if binary_ready.len() >= BINARY_INSTALL_MIN && binary_valve.is_none() {
            install_batch(root, "binary", &binary_ready, &mut log)?;
            refresh_filesystem_state(&conn, root)?;
            tranche = run_tranche(root, &conn, "binary", &mut log)?;
        } else {
            if let (true, Some(reason)) = (binary_ready.len() >= BINARY_INSTALL_MIN, &binary_valve)
            {
                log.push(json!({"step": "tranche_skipped", "task": "binary", "reason": reason}));
            }
            // Rule (c): regression — install any fetched full-encode outputs
            // first (they join the sidecar-ready pool), then tranche.
            match valve_shut(coverage_regression) {
                Some(reason) => log.push(
                    json!({"step": "tranche_skipped", "task": "regression", "reason": reason}),
                ),
                None => {
                    let regression_ready = installable(&conn, "regression")?;
                    if !regression_ready.is_empty() {
                        install_batch(root, "regression", &regression_ready, &mut log)?;
                        refresh_filesystem_state(&conn, root)?;
                    }
                    tranche = run_tranche(root, &conn, "regression", &mut log)?;
                }
            }
        }
    }
    let summary = stage_summary(&conn)?;
    let report = json!({
        "schema": "JopeSelfplayExpansionDriver.v1",
        "generated_at_ms": now_ms(),
        "registry_scanned": scanned,
        "encoding_exclusion_receipts_created": exclusion_receipts_created,
        "obs_rows_ingested": obs_rows,
        // Kept for receipt continuity with earlier cycles; regression is the
        // lane that paces catalog growth today.
        "previous_tranche_coverage": coverage_regression,
        "coverage_regression": coverage_regression,
        "coverage_binary": coverage_binary,
        "guardrail_block": blocked,
        "tranche_completed": tranche,
        "encode_chunk_submitted": &submitted_chunk,
        "encode_chunk_backoff_reason": submitted_chunk
            .as_ref()
            .and_then(|submission| submission.backoff_reason.clone()),
        "stage_summary": summary,
        "log": log,
    });
    let target = root.join("ops/expansion-driver-v1.json");
    let tmp = root.join(format!(
        "ops/.expansion-driver-v1.json.{}",
        std::process::id()
    ));
    std::fs::write(&tmp, serde_json::to_vec_pretty(&report)?)?;
    std::fs::rename(&tmp, &target)?;
    let mut line = serde_json::to_vec(&report)?;
    line.push(b'\n');
    use std::io::Write;
    std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(root.join("ops/expansion-events.v1.jsonl"))?
        .write_all(&line)?;
    Ok(report)
}

pub(crate) fn query(root: &Path, dataset: Option<&str>, summary: bool) -> Result<()> {
    let conn = open_state_db(root)?;
    if let Some(id) = dataset {
        let mut statement = conn.prepare("SELECT * FROM datasets WHERE dataset_hash=?")?;
        let names: Vec<String> = statement
            .column_names()
            .iter()
            .map(|s| s.to_string())
            .collect();
        let rows = statement.query_map([id], |r| {
            let mut object = serde_json::Map::new();
            for (index, name) in names.iter().enumerate() {
                let value: Value = match r.get_ref(index)? {
                    rusqlite::types::ValueRef::Null => Value::Null,
                    rusqlite::types::ValueRef::Integer(v) => json!(v),
                    rusqlite::types::ValueRef::Real(v) => json!(v),
                    rusqlite::types::ValueRef::Text(v) => json!(String::from_utf8_lossy(v)),
                    rusqlite::types::ValueRef::Blob(_) => json!("<blob>"),
                };
                object.insert(name.clone(), value);
            }
            Ok(Value::Object(object))
        })?;
        for row in rows {
            println!("{}", serde_json::to_string(&row?)?);
        }
    }
    if summary {
        println!("{}", serde_json::to_string_pretty(&stage_summary(&conn)?)?);
    }
    Ok(())
}
