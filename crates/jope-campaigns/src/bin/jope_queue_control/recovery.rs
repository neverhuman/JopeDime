use super::*;

mod retirement;
pub(super) use retirement::{
    apply_retired_binding_repair, build_retired_binding_repair_plan, recover_retirement_candidates,
};

#[allow(clippy::too_many_arguments)]
pub(super) fn build_missing_receipt_recovery_plan(
    root: &Path,
    wave_dir: &Path,
    pair_registry: &Path,
    backlog: &Path,
    scheduler: &HashMap<String, (String, String)>,
    receipt_dir: &Path,
    scq_events: &Path,
    runs_root: &Path,
    scq: &Path,
    socket: &Path,
) -> Result<MissingReceiptRecoveryPlan> {
    let claims = load_claims(backlog)?;
    // Reuse the strict blocker audit to validate the complete wave, current
    // checkpoint, immutable manifests, and exact pair/claim bindings.
    let _validated = build_wave_blocker_plan(
        root,
        wave_dir,
        pair_registry,
        &claims,
        scheduler,
        receipt_dir,
        scq_events,
    )?;
    let (wave, desired, current_manifest_hashes) = load_wave_desired_pairs(wave_dir)?;
    let pair = Connection::open_with_flags(
        pair_registry,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )?;
    pair.busy_timeout(Duration::from_secs(60))?;
    let mut lookup = pair.prepare(
        "SELECT state,manifest_path,manifest_sha256,job_id,receipt_sha256,failure_class
         FROM pairs
         WHERE task=?1 AND dataset_hash=?2 AND semantic_sha256=?3 AND protocol_sha256=?4",
    )?;
    let mut blockers = BTreeMap::<String, (String, String, Vec<RegistryPairIdentity>, bool)>::new();
    for identity in &desired {
        type PairRecoveryRow = (
            String,
            Option<String>,
            Option<String>,
            Option<String>,
            Option<String>,
            Option<String>,
        );
        let row: Option<PairRecoveryRow> = lookup
            .query_row(
                params![
                    identity.task,
                    identity.dataset_hash,
                    identity.semantic_sha256,
                    identity.protocol_sha256
                ],
                |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                        row.get(5)?,
                    ))
                },
            )
            .optional()?;
        let Some((state, path, manifest, job, receipt, failure)) = row else {
            continue;
        };
        if !matches!(state.as_str(), "reserved" | "submitted" | "held") || receipt.is_some() {
            continue;
        }
        let (Some(path), Some(manifest), Some(job)) = (path, manifest, job) else {
            continue;
        };
        if current_manifest_hashes.contains(&manifest) {
            continue;
        }
        let entry = blockers
            .entry(manifest.clone())
            .or_insert_with(|| (job.clone(), path.clone(), Vec::new(), false));
        if entry.0 != job || entry.1 != path {
            bail!("missing-receipt blocker manifest binding is inconsistent");
        }
        entry.2.push(identity.clone());
        entry.3 |= failure.is_some();
    }
    drop(lookup);
    let mut manifest_pairs_lookup = pair.prepare(
        "SELECT task,dataset_hash,semantic_sha256,protocol_sha256,
                state,manifest_path,job_id,receipt_sha256,failure_class
         FROM pairs INDEXED BY pair_manifest
         WHERE manifest_sha256=?1
         ORDER BY task,dataset_hash,semantic_sha256,protocol_sha256",
    )?;

    let jobs = blockers
        .values()
        .map(|(job, _, _, _)| job.clone())
        .collect::<BTreeSet<_>>();
    let start_evidence = load_scq_start_evidence(scq_events, &jobs)?;
    let corpus = Connection::open_with_flags(
        root.join("corpus/strict-test-exact.sqlite"),
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )?;
    let mut reason_counts = BTreeMap::new();
    let mut exclusions = Vec::new();
    let mut candidates = Vec::new();
    for (manifest_sha256, (job_id, manifest_path, pairs, has_failure)) in blockers.iter() {
        let exclude = |reason: String,
                       reason_counts: &mut BTreeMap<String, usize>,
                       exclusions: &mut Vec<String>| {
            let class = reason.split(':').next().unwrap_or("unknown").to_owned();
            *reason_counts.entry(class).or_insert(0) += 1;
            exclusions.push(format!("{job_id}:{reason}"));
        };
        let Some(claim) = claims.get(job_id) else {
            bail!("missing-receipt blocker claim disappeared");
        };
        if claim.manifest_sha256 != *manifest_sha256
            || claim.manifest.to_string_lossy() != *manifest_path
        {
            bail!("missing-receipt blocker claim hash/path mismatch");
        }
        if !(claim.state == "submitted" || claim.state.starts_with("ingest_failed:")) {
            exclude(
                format!("claim_state_not_retryable:{}", claim.state),
                &mut reason_counts,
                &mut exclusions,
            );
            continue;
        }
        let old_manifest = match validate_manifest(claim) {
            Ok(manifest) => manifest,
            Err(error) => {
                exclude(
                    format!("manifest_provenance_invalid:{error:#}"),
                    &mut reason_counts,
                    &mut exclusions,
                );
                continue;
            }
        };
        if strict_wave(&old_manifest)
            .is_some_and(|old_wave| wave_number(&old_wave) >= wave_number(&wave))
        {
            exclude(
                "current_or_future_wave_binding".to_owned(),
                &mut reason_counts,
                &mut exclusions,
            );
            continue;
        }
        if *has_failure {
            exclude(
                "semantic_or_terminal_pair_failure".to_owned(),
                &mut reason_counts,
                &mut exclusions,
            );
            continue;
        }
        let Some(evidence) = start_evidence.get(job_id) else {
            exclude(
                "no_durable_launch_or_completion_evidence".to_owned(),
                &mut reason_counts,
                &mut exclusions,
            );
            continue;
        };
        if scheduler.get(job_id).is_some_and(|(_, state)| {
            matches!(
                state.as_str(),
                "queued" | "starting" | "held" | "dispatching" | "running"
            )
        }) {
            exclude(
                "current_active_scheduler_state".to_owned(),
                &mut reason_counts,
                &mut exclusions,
            );
            continue;
        }
        let ingested: bool = corpus.query_row(
            "SELECT EXISTS(SELECT 1 FROM ingested_jobs WHERE job_id=?1)",
            [job_id],
            |row| row.get(0),
        )?;
        if ingested {
            exclude(
                "authoritative_ingested_receipt".to_owned(),
                &mut reason_counts,
                &mut exclusions,
            );
            continue;
        }
        let receipt_root = root.join("receipts").join(job_id);
        if tree_contains_named_file(&receipt_root, "SHARD_OK")? {
            exclude(
                "authoritative_local_receipt".to_owned(),
                &mut reason_counts,
                &mut exclusions,
            );
            continue;
        }
        if tree_contains_named_file(&receipt_root, "SHARD_FAIL.json")?
            || tree_contains_named_file(&runs_root.join(job_id), "SHARD_FAIL.json")?
        {
            exclude(
                "semantic_terminal_failure_output".to_owned(),
                &mut reason_counts,
                &mut exclusions,
            );
            continue;
        }
        if tree_contains_named_file(&runs_root.join(job_id), "SHARD_OK")? {
            exclude(
                "recoverable_run_receipt_exists".to_owned(),
                &mut reason_counts,
                &mut exclusions,
            );
            continue;
        }
        match probe_scq_output(scq, socket, job_id)? {
            OutputProbe::AuthoritativeReceipt => {
                exclude(
                    "recoverable_scq_receipt_exists".to_owned(),
                    &mut reason_counts,
                    &mut exclusions,
                );
                continue;
            }
            OutputProbe::SemanticFailure => {
                exclude(
                    "semantic_terminal_failure_output".to_owned(),
                    &mut reason_counts,
                    &mut exclusions,
                );
                continue;
            }
            OutputProbe::Unverified(error) => {
                exclude(
                    format!("output_unavailability_unverified:{error}"),
                    &mut reason_counts,
                    &mut exclusions,
                );
                continue;
            }
            OutputProbe::Unavailable(reason) => {
                let rows = manifest_pairs_lookup
                    .query_map([manifest_sha256], |row| {
                        Ok(ManifestPairBinding {
                            identity: RegistryPairIdentity {
                                task: row.get(0)?,
                                dataset_hash: row.get(1)?,
                                semantic_sha256: row.get(2)?,
                                protocol_sha256: row.get(3)?,
                            },
                            state: row.get(4)?,
                            manifest_path: row.get(5)?,
                            job_id: row.get(6)?,
                            receipt_sha256: row.get(7)?,
                            failure_class: row.get(8)?,
                        })
                    })?
                    .collect::<rusqlite::Result<Vec<_>>>()?;
                let Some(all_pairs) =
                    validate_missing_receipt_manifest_bindings(rows, manifest_path, job_id)
                else {
                    exclude(
                        "manifest_contains_authoritative_or_nonactive_pair".to_owned(),
                        &mut reason_counts,
                        &mut exclusions,
                    );
                    continue;
                };
                *reason_counts
                    .entry("eligible_infrastructure_retry_missing_receipt".to_owned())
                    .or_insert(0) += 1;
                candidates.push(MissingReceiptRecoveryCandidate {
                    job_id: job_id.clone(),
                    manifest: manifest_path.clone(),
                    manifest_sha256: manifest_sha256.clone(),
                    claim_state: claim.state.clone(),
                    scq_evidence: format!("{evidence}; {reason}"),
                    desired_pair_count: pairs.len(),
                    pair_count: all_pairs.len(),
                    pairs: all_pairs,
                });
            }
        }
    }
    drop(manifest_pairs_lookup);
    drop(pair);
    candidates.sort_by(|left, right| left.job_id.cmp(&right.job_id));
    exclusions.sort();
    let blocker_pairs = blockers.values().map(|entry| entry.2.len()).sum();
    let eligible_desired_pairs = candidates
        .iter()
        .map(|candidate| candidate.desired_pair_count)
        .sum();
    let eligible_total_pairs = candidates
        .iter()
        .map(|candidate| candidate.pair_count)
        .sum();
    Ok(MissingReceiptRecoveryPlan {
        wave,
        wave_dir: fs::canonicalize(wave_dir)?.to_string_lossy().into_owned(),
        desired_pairs: desired.len(),
        blocker_pairs,
        blocker_manifests: blockers.len(),
        eligible_desired_pairs,
        eligible_total_pairs,
        eligible_manifests: candidates.len(),
        reason_counts,
        exclusions,
        candidates,
    })
}

pub(super) fn validate_missing_receipt_manifest_bindings(
    rows: Vec<ManifestPairBinding>,
    manifest_path: &str,
    job_id: &str,
) -> Option<Vec<RegistryPairIdentity>> {
    if rows.is_empty()
        || rows.iter().any(|row| {
            !matches!(row.state.as_str(), "reserved" | "submitted" | "held")
                || row.manifest_path.as_deref() != Some(manifest_path)
                || row.job_id.as_deref() != Some(job_id)
                || row.receipt_sha256.is_some()
                || row.failure_class.is_some()
        })
    {
        return None;
    }
    Some(rows.into_iter().map(|row| row.identity).collect())
}

pub(super) fn load_wave_desired_pairs(
    wave_dir: &Path,
) -> Result<(String, BTreeSet<RegistryPairIdentity>, BTreeSet<String>)> {
    let wave_dir = fs::canonicalize(wave_dir)?;
    let wave = wave_dir
        .file_name()
        .and_then(|name| name.to_str())
        .context("wave directory name is invalid")?
        .to_owned();
    let summary: Value = serde_json::from_slice(&fs::read(wave_dir.join("summary.json"))?)?;
    let entries = summary
        .get("manifests")
        .and_then(Value::as_array)
        .context("wave summary manifests are absent")?;
    let mut desired = BTreeSet::new();
    let mut hashes = BTreeSet::new();
    for entry in entries {
        let path = wave_dir.join(required_json_str(entry, "relative_path")?);
        let expected_hash = required_json_str(entry, "sha256")?;
        if sha256_file(&path)? != expected_hash {
            bail!("wave desired manifest hash drift");
        }
        let manifest: Value = serde_json::from_slice(&fs::read(path)?)?;
        let task = required_json_str(&manifest, "task")?;
        let dataset = required_json_str(&manifest, "dataset_hash")?;
        let protocol = required_json_str(&manifest, "protocol_sha256")?;
        for candidate in manifest
            .get("candidates")
            .and_then(Value::as_array)
            .context("wave desired candidates are absent")?
        {
            desired.insert(RegistryPairIdentity {
                task: task.to_owned(),
                dataset_hash: dataset.to_owned(),
                semantic_sha256: required_json_str(candidate, "semantic_sha256")?.to_owned(),
                protocol_sha256: protocol.to_owned(),
            });
        }
        hashes.insert(expected_hash.to_owned());
    }
    if desired.len() % 256 != 0 || !matches!(desired.len() / 256, 128 | 192 | 256) {
        bail!(
            "wave desired identity count {} is not a legal micro-wave total",
            desired.len()
        );
    }
    Ok((wave, desired, hashes))
}

pub(super) enum OutputProbe {
    AuthoritativeReceipt,
    SemanticFailure,
    Unavailable(String),
    Unverified(String),
}

pub(super) fn probe_scq_output(scq: &Path, socket: &Path, job_id: &str) -> Result<OutputProbe> {
    let temporary = ProbeDirectory::create()?;
    let output = Command::new(scq)
        .arg("fetch")
        .arg(job_id)
        .arg("--output")
        .arg(&temporary.path)
        .arg("--socket")
        .arg(socket)
        .output()
        .with_context(|| format!("probe SCQ output for {job_id}"))?;
    if output.status.success() {
        if tree_contains_named_file(&temporary.path, "SHARD_OK")? {
            return Ok(OutputProbe::AuthoritativeReceipt);
        }
        if tree_contains_named_file(&temporary.path, "SHARD_FAIL.json")? {
            return Ok(OutputProbe::SemanticFailure);
        }
        return Ok(OutputProbe::Unavailable(
            "SCQ fetch returned no authoritative marker".to_owned(),
        ));
    }
    let error = String::from_utf8_lossy(if output.stderr.is_empty() {
        &output.stdout
    } else {
        &output.stderr
    })
    .trim()
    .to_owned();
    if error.contains("no such job")
        || error.contains("output is unavailable")
        || error.contains("no outputs")
    {
        Ok(OutputProbe::Unavailable(error))
    } else {
        Ok(OutputProbe::Unverified(error))
    }
}

pub(super) struct ProbeDirectory {
    path: PathBuf,
}

impl ProbeDirectory {
    fn create() -> Result<Self> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "jope-scq-output-probe-{}-{}-{}",
            std::process::id(),
            now_ms(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path)?;
        Ok(Self { path })
    }
}

impl Drop for ProbeDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

pub(super) fn tree_contains_named_file(root: &Path, name: &str) -> Result<bool> {
    if !root.exists() {
        return Ok(false);
    }
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let kind = entry.file_type()?;
        if kind.is_symlink() {
            bail!("output proof tree contains a symlink");
        }
        if kind.is_dir() {
            if tree_contains_named_file(&entry.path(), name)? {
                return Ok(true);
            }
        } else if entry.file_name() == name {
            return Ok(true);
        }
    }
    Ok(false)
}
