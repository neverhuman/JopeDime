use super::*;

/// Acceptance 2 validator: classify every finding in `source_findings` as `confirmed`, `rejected`,
/// or `unknown` and separate `current_ci`, `historical_run`, and `proposed_protocol` categories.
pub fn verify_finding_classifications(
    repo: &Path,
    findings_val: &Value,
) -> Result<(usize, usize, usize)> {
    let mut snapshot = BaselineSnapshot::new(repo, RECEIVING_MAIN_HEAD_SHA)?;
    verify_finding_classifications_with_snapshot(&mut snapshot, findings_val)
}

pub(super) fn verify_finding_classifications_with_snapshot(
    snapshot: &mut BaselineSnapshot<'_>,
    findings_val: &Value,
) -> Result<(usize, usize, usize)> {
    let entries = list_of(findings_val, "source_findings")?;
    require!(!entries.is_empty(), "source_findings must not be empty");

    let mut confirmed_count = 0usize;
    let mut rejected_count = 0usize;
    let mut unknown_count = 0usize;
    let mut seen_ids = BTreeSet::new();

    for (idx, entry_val) in entries.iter().enumerate() {
        let ectx = format!("source_findings[{idx}]");
        let entry = dict_of(entry_val, &ectx)?;
        let fid = req_str(entry, "id", &ectx)?;
        require!(
            seen_ids.insert(fid.to_owned()),
            "{ectx}: duplicate finding id `{fid}`"
        );
        let classification = req_str(entry, "classification", &ectx)?;
        let category = req_str(entry, "category", &ectx)?;
        require!(
            matches!(
                category,
                "current_ci" | "historical_run" | "proposed_protocol"
            ),
            "{ectx} (`{fid}`): invalid category `{category}` (expected current_ci, historical_run, or proposed_protocol)"
        );
        let statement = req_str(entry, "statement", &ectx)?;
        require!(
            !statement.trim().is_empty(),
            "{ectx} (`{fid}`): empty statement"
        );

        match entry.get("live_qualification_active") {
            Some(Value::Bool(false)) => {}
            Some(Value::Bool(true)) | Some(Value::Str(_)) => {
                return Err(Error::Invalid(format!(
                    "{ectx} (`{fid}`): historical, skipped, or proposed protocol records must never claim live_qualification_active=true (must be boolean false)"
                )));
            }
            _ => {
                return Err(Error::Invalid(format!(
                    "{ectx} (`{fid}`): missing or non-boolean `live_qualification_active`"
                )));
            }
        }

        let commit_sha = req_str(entry, "source_sha", &ectx)?;
        require!(
            is_lower_hex_sha40(commit_sha) && commit_sha == RECEIVING_MAIN_HEAD_SHA,
            "{ectx} (`{fid}`): finding must bind 40-hex source_sha `{RECEIVING_MAIN_HEAD_SHA}`, got `{commit_sha}`"
        );
        let path = req_str(entry, "path", &ectx)?;
        validate_repo_relative_path(path)?;
        let witness_line = req_u64(entry, "witness_line", &ectx)? as usize;
        require!(
            witness_line >= 1,
            "{ectx} (`{fid}`): witness_line must be >= 1"
        );
        let witness_sub = req_str(entry, "witness_substring", &ectx)?;
        require!(
            !witness_sub.trim().is_empty(),
            "{ectx} (`{fid}`): witness_substring must not be empty"
        );
        let evidence_sha = req_str(entry, "evidence_sha256", &ectx)?;
        require!(
            is_lower_hex_sha64(evidence_sha),
            "{ectx} (`{fid}`): finding must bind a 64-hex `evidence_sha256`, got `{evidence_sha}`"
        );
        if snapshot.has_git_commit() {
            require!(
                snapshot.file_exists(path),
                "{ectx} (`{fid}`): path `{path}` does not exist in baseline commit"
            );
            let file_bytes = snapshot.blob_bytes(path)?;
            let file_text = String::from_utf8(file_bytes.clone())
                .map_err(|e| Error::Invalid(format!("{ectx}: non-UTF-8 `{path}`: {e}")))?;
            let lines: Vec<&str> = file_text.lines().collect();
            let line_count = lines.len().max(1);
            require!(
                witness_line <= line_count,
                "{ectx} (`{fid}`): witness_line {witness_line} out of range (1..={line_count}) for `{path}`"
            );
            let actual_line = lines.get(witness_line - 1).copied().unwrap_or("");
            require!(
                actual_line.contains(witness_sub),
                "{ectx} (`{fid}`): `{path}:{witness_line}` does not contain witness_substring `{witness_sub}` (found `{actual_line}`)"
            );
            let actual_sha = sha256_hex(&file_bytes);
            require!(
                actual_sha == evidence_sha,
                "{ectx} (`{fid}`): evidence_sha256 mismatch for `{path}`: expected {actual_sha}, got {evidence_sha}"
            );
        }

        match classification {
            "confirmed" => {
                require!(
                    category != "proposed_protocol",
                    "{ectx} (`{fid}`): a `proposed_protocol` finding cannot be classified as `confirmed`"
                );
                require!(
                    entry.get("resolution_gate").is_none()
                        && entry.get("coerced_to_confirmed").is_none(),
                    "{ectx} (`{fid}`): unknown finding with resolution_gate must never be coerced to `confirmed`"
                );
                confirmed_count += 1;
            }
            "rejected" => {
                let reason = req_str(entry, "rejection_reason", &ectx)?;
                require!(
                    !reason.trim().is_empty(),
                    "{ectx} (`{fid}`): rejected finding must include a non-empty `rejection_reason`"
                );
                rejected_count += 1;
            }
            "unknown" => {
                let gate = req_str(entry, "resolution_gate", &ectx)?;
                require!(
                    !gate.trim().is_empty(),
                    "{ectx} (`{fid}`): unknown finding must specify a non-empty `resolution_gate`"
                );
                require!(
                    entry.get("coerced_to_confirmed").is_none(),
                    "{ectx} (`{fid}`): unknown finding must never set coerced_to_confirmed"
                );
                unknown_count += 1;
            }
            other => {
                return Err(Error::Invalid(format!(
                    "{ectx} (`{fid}`): invalid classification `{other}` (expected confirmed, rejected, or unknown)"
                )));
            }
        }
    }

    require!(
        confirmed_count > 0 && rejected_count > 0 && unknown_count > 0,
        "source_findings must include at least one confirmed ({confirmed_count}), rejected ({rejected_count}), and unknown ({unknown_count}) finding"
    );
    Ok((confirmed_count, rejected_count, unknown_count))
}
