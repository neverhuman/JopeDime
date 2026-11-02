use super::*;

pub(super) fn pair_identity(root: &Path, request: &PairRepairRequestV1) -> Result<PairIdentity> {
    let connection = Connection::open_with_flags(
        root.join("corpus/strict-test-exact.sqlite"),
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_URI,
    )?;
    connection.busy_timeout(Duration::from_secs(60))?;
    let mut statement = connection.prepare(
        "SELECT task,context_identity,accepted_parent_checkpoint_sha256,
                wave,pair_index,receipt_sha256
           FROM causal_evaluation_receipts_v2
          WHERE pair_id=?1 AND canary=0
          ORDER BY created_at_ms,receipt_sha256",
    )?;
    let rows = statement
        .query_map([request.pair_id.as_str()], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, u64>(3)?,
                row.get::<_, usize>(4)?,
                row.get::<_, String>(5)?,
            ))
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    let first = rows.first().context("repair pair has no raw receipt")?;
    let task: Task = first.0.parse()?;
    ensure!(task == request.task, "repair pair task drift");
    ensure!(
        rows.iter().all(|row| {
            row.0 == first.0
                && row.1 == first.1
                && row.2 == first.2
                && row.3 == first.3
                && row.4 == first.4
        }),
        "repair pair receipt identity is ambiguous"
    );
    Ok(PairIdentity {
        task,
        context_identity: first.1.clone(),
        accepted_parent_checkpoint_sha256: first.2.clone(),
        wave: first.3,
        pair_index: first.4,
        receipt_sha256s: rows.iter().map(|row| row.5.clone()).collect(),
    })
}

pub(super) fn target_candidate(
    root: &Path,
    request: &PairRepairRequestV1,
    identity: &PairIdentity,
) -> Result<TargetCandidate> {
    let directory = root
        .join("waves")
        .join(format!("strict-wave-{:06}", identity.wave))
        .join("manifests");
    ensure!(
        directory.is_dir(),
        "repair wave manifest directory is absent"
    );
    let mut matches = Vec::new();
    for entry in fs::read_dir(&directory)? {
        let path = entry?.path();
        if path.extension().and_then(|value| value.to_str()) != Some("json")
            || !path
                .file_name()
                .and_then(|value| value.to_str())
                .is_some_and(|name| name.contains(&identity.context_identity))
        {
            continue;
        }
        let manifest = read_json(&path)?;
        if manifest.get("task").and_then(Value::as_str) != Some(identity.task.as_str())
            || manifest.get("dataset_hash").and_then(Value::as_str)
                != Some(identity.context_identity.as_str())
            || manifest
                .pointer("/label_authority/model_checkpoint_sha256")
                .and_then(Value::as_str)
                != Some(identity.accepted_parent_checkpoint_sha256.as_str())
        {
            continue;
        }
        for candidate in manifest
            .get("candidates")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let evidence = candidate.get("evidence").unwrap_or(&Value::Null);
            if evidence.get("evaluation_lane").and_then(Value::as_str)
                == Some(lane_name(request.missing_lane))
                && evidence.get("matched_pair_index").and_then(Value::as_u64)
                    == Some(identity.pair_index as u64)
            {
                if let Some(pair_id) = evidence.get("pair_id").and_then(Value::as_str) {
                    ensure!(
                        pair_id == request.pair_id.as_str(),
                        "target manifest carries a conflicting pair id"
                    );
                }
                matches.push(TargetCandidate {
                    manifest_path: path.clone(),
                    manifest_sha256: sha256_file(&path)?,
                    candidate_sha256: candidate
                        .get("semantic_sha256")
                        .and_then(Value::as_str)
                        .context("repair candidate lacks semantic hash")?
                        .to_owned(),
                    candidate: candidate.clone(),
                    manifest: manifest.clone(),
                });
            }
        }
    }
    ensure!(
        matches.len() == 1,
        "repair target member count is {}, expected exactly one",
        matches.len()
    );
    Ok(matches.remove(0))
}
