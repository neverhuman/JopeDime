use super::*;

pub(crate) fn accepted_parents(root: &Path) -> Result<(BTreeMap<Task, Sha256Digest>, bool)> {
    let pointer = read_json(&root.join("training/accepted-by-task.json"))?;
    ensure!(
        pointer.get("schema").and_then(Value::as_str) == Some("JopeSelfplayAcceptedByTask.v1"),
        "accepted-by-task pointer schema is not live v1"
    );
    let mut output = BTreeMap::new();
    let mut valid = true;
    for task in Task::ALL {
        let accepted_row = pointer
            .get("tasks")
            .and_then(|value| value.get(task.as_str()));
        let anchor;
        let row = if let Some(row) = accepted_row {
            row
        } else if matches!(task, Task::Multiclass | Task::Timeseries) {
            let path = root
                .join("training/bootstrap-evaluation-anchors")
                .join(format!("{}.v1.json", task.as_str()));
            if !path.is_file() {
                continue;
            }
            anchor = read_json(&path)?;
            ensure!(
                anchor.get("schema").and_then(Value::as_str)
                    == Some("TaskBootstrapEvaluationAnchor.v1")
                    && anchor.get("task").and_then(Value::as_str) == Some(task.as_str())
                    && anchor.get("accepted").and_then(Value::as_bool) == Some(false),
                "{task} bootstrap evaluation anchor is not an unaccepted v1 anchor"
            );
            &anchor
        } else {
            anyhow::bail!("accepted pointer lacks {task}");
        };
        let digest = row
            .get("checkpoint_sha256")
            .and_then(Value::as_str)
            .with_context(|| format!("{task} accepted pointer lacks checkpoint digest"))?;
        let path = Path::new(
            row.get("checkpoint")
                .and_then(Value::as_str)
                .with_context(|| format!("{task} accepted pointer lacks checkpoint path"))?,
        );
        if !path.is_file() || sha256_file(path)? != digest {
            valid = false;
        }
        output.insert(task, Sha256Digest::parse(digest.to_owned())?);
    }
    Ok((output, valid))
}

pub(crate) fn wave_states(root: &Path) -> Result<BTreeMap<u64, WaveState>> {
    let mut output = BTreeMap::new();
    let waves = root.join("waves");
    if !waves.is_dir() {
        return Ok(output);
    }
    for entry in std::fs::read_dir(waves)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        let Some(wave) = parse_wave(&name) else {
            continue;
        };
        if wave < 191 {
            continue;
        }
        let summary_path = entry.path().join("summary.json");
        let cohort_path = entry.path().join("cohort.json");
        if !summary_path.is_file() || !cohort_path.is_file() {
            continue;
        }
        let summary = read_json(&summary_path)?;
        let eligibility = summary.get("wave_summary").and_then(|value| {
            value
                .get("evidence_eligibility")
                .or_else(|| summary.get("evidence_eligibility"))
        });
        let completed = summary.get("immutable").and_then(Value::as_bool) == Some(true)
            && eligibility
                .and_then(|value| value.get("promotion"))
                .and_then(Value::as_bool)
                == Some(true)
            && eligibility
                .and_then(|value| value.get("recovery_accounting_only"))
                .and_then(Value::as_bool)
                != Some(true);
        let zero_backfill = summary
            .get("cross_context_backfill_rows")
            .and_then(Value::as_u64)
            == Some(0);
        let checkpoint_sha256 = summary
            .get("checkpoint_sha256")
            .and_then(Value::as_str)
            .context("wave summary lacks checkpoint identity")?
            .to_owned();
        let cohort = read_json(&cohort_path)?;
        let mut requested_slots = Task::ALL
            .into_iter()
            .map(|task| (task, 0))
            .collect::<BTreeMap<_, _>>();
        for row in cohort
            .get("entries")
            .and_then(Value::as_array)
            .context("wave cohort lacks entries")?
        {
            let task: Task = row
                .get("task")
                .and_then(Value::as_str)
                .context("cohort entry lacks task")?
                .parse()?;
            *requested_slots.entry(task).or_default() += 8;
        }
        output.insert(
            wave,
            WaveState {
                checkpoint_sha256,
                completed,
                zero_backfill,
                requested_slots,
            },
        );
    }
    Ok(output)
}

pub(crate) fn receipt_rows(
    connection: &Connection,
    accepted: &BTreeMap<Task, Sha256Digest>,
    accepted_lineage_valid: bool,
    waves: &BTreeMap<u64, WaveState>,
) -> Result<Vec<CausalEvaluationReceiptV2>> {
    let exists: bool = connection.query_row(
        "SELECT EXISTS(
           SELECT 1 FROM sqlite_master
           WHERE type='table' AND name='causal_evaluation_receipts_v2'
         )",
        [],
        |row| row.get(0),
    )?;
    if !exists {
        return Ok(Vec::new());
    }
    let mut statement = connection.prepare(
        "SELECT receipt_sha256,manifest_sha256,candidate_sha256,task,
                context_identity,accepted_parent_checkpoint_sha256,wave,lane,
                pair_index,pair_id,score,success,proposal_valid,runtime_seconds,
                ancestry,cross_context_backfill,valid_lineage,canary
           FROM causal_evaluation_receipts_v2
          WHERE wave>=191
          ORDER BY wave,task,context_identity,pair_id,lane,receipt_sha256",
    )?;
    let raw = statement
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, String>(5)?,
                row.get::<_, u64>(6)?,
                row.get::<_, String>(7)?,
                row.get::<_, usize>(8)?,
                row.get::<_, String>(9)?,
                row.get::<_, Option<f64>>(10)?,
                row.get::<_, bool>(11)?,
                row.get::<_, bool>(12)?,
                row.get::<_, f64>(13)?,
                row.get::<_, String>(14)?,
                row.get::<_, bool>(15)?,
                row.get::<_, bool>(16)?,
                row.get::<_, bool>(17)?,
            ))
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    let mut output = Vec::with_capacity(raw.len());
    for (
        receipt,
        manifest,
        candidate,
        task,
        context,
        parent,
        wave_number,
        lane,
        pair_index,
        pair_id,
        score,
        success,
        proposal_valid,
        runtime_seconds,
        ancestry,
        cross_context_backfill,
        valid_lineage,
        canary,
    ) in raw
    {
        let task: Task = task.parse()?;
        let Some(wave) = waves.get(&wave_number) else {
            continue;
        };
        let lane = match lane.as_str() {
            "native_dl_direct" => CausalLane::NativeDlDirect,
            "matched_gp_comparator" => CausalLane::MatchedGpComparator,
            _ => continue,
        };
        let ancestry = match ancestry.as_str() {
            "none" => DlAncestryKind::None,
            "accepted_dl" => DlAncestryKind::AcceptedDl,
            "shadow_dl" => DlAncestryKind::ShadowDl,
            _ => DlAncestryKind::Mixed,
        };
        output.push(CausalEvaluationReceiptV2 {
            schema: CAUSAL_EVALUATION_RECEIPT_SCHEMA.to_owned(),
            receipt_sha256: Sha256Digest::parse(receipt)?,
            manifest_sha256: Sha256Digest::parse(manifest)?,
            candidate_sha256: Sha256Digest::parse(candidate)?,
            task,
            context_identity: context,
            accepted_parent_checkpoint_sha256: Sha256Digest::parse(parent.clone())?,
            wave: wave_number,
            lane,
            pair_index,
            pair_id: Sha256Digest::parse(pair_id)?,
            score,
            success,
            proposal_valid,
            runtime_ms: (runtime_seconds.max(0.0) * 1_000.0).round() as u64,
            ancestry,
            cross_context_backfill: cross_context_backfill || !wave.zero_backfill,
            completed_causal_wave: wave.completed,
            valid_lineage: valid_lineage
                && accepted_lineage_valid
                && accepted.get(&task).map(Sha256Digest::as_str) == Some(parent.as_str())
                && wave.checkpoint_sha256 == parent,
            canary,
        });
    }
    Ok(output)
}

pub(crate) fn backfill_raw_receipts(root: &Path) -> Result<Value> {
    let corpus = root.join("corpus/strict-test-exact.sqlite");
    let mut connection = open_corpus(&corpus)?;
    // Receipt replay is idempotent and each row is protected by its immutable
    // triple key. WAL + NORMAL remains crash-consistent while avoiding one
    // durable fsync for every historical job; a lost tail is replayed on the
    // next run.
    connection.pragma_update(None, "synchronous", "NORMAL")?;
    let backlog = Connection::open_with_flags(
        root.join("ops/backlog.sqlite"),
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_URI,
    )?;
    backlog.busy_timeout(Duration::from_secs(60))?;
    let mut statement = backlog
        .prepare("SELECT job_id,manifest FROM claims WHERE state='ingested' ORDER BY job_id")?;
    let claims = statement
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                PathBuf::from(row.get::<_, String>(1)?),
            ))
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    drop(statement);
    drop(backlog);
    let already = connection.query_row(
        "SELECT COUNT(*) FROM causal_evaluation_receipts_v2",
        [],
        |row| row.get::<_, u64>(0),
    )?;
    let mut eligible_jobs = 0_u64;
    let mut backfilled_jobs = 0_u64;
    let mut rows = 0_u64;
    for (job_id, manifest) in claims {
        let Some(wave) = manifest
            .components()
            .filter_map(|part| part.as_os_str().to_str())
            .find_map(parse_wave)
        else {
            continue;
        };
        if wave < 191 {
            continue;
        }
        eligible_jobs += 1;
        let output = root.join("receipts").join(&job_id);
        if !output.join("SHARD_OK").is_file() {
            continue;
        }
        let summary =
            ingest_receipt_with_manifest_connection(&mut connection, &job_id, &output, &manifest)?;
        backfilled_jobs += 1;
        rows += summary.rows as u64;
    }
    let after = connection.query_row(
        "SELECT COUNT(*) FROM causal_evaluation_receipts_v2",
        [],
        |row| row.get::<_, u64>(0),
    )?;
    Ok(json!({
        "schema": "CausalReceiptBackfill.v2",
        "generated_at_ms": now_ms(),
        "eligible_jobs": eligible_jobs,
        "backfilled_jobs": backfilled_jobs,
        "raw_rows_replayed": rows,
        "receipt_rows_before": already,
        "receipt_rows_after": after,
        "append_only_rows_added": after.saturating_sub(already),
    }))
}

pub(crate) fn parse_wave(value: &str) -> Option<u64> {
    value.strip_prefix("strict-wave-")?.parse().ok()
}
