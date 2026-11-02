fn context_counts(tx: &rusqlite::Transaction<'_>, key: &str) -> Result<Counts> {
    Ok(
        tx.query_row(CONTEXT_COUNTS_SQL, rusqlite::params![key], |r| {
            let attempted: u64 = r.get(1)?;
            let reserved: u64 = r.get(2)?;
            let pending_plans: u64 = r.get(5)?;
            Ok(Counts {
                unique_strict_commits: r.get(0)?,
                attempted,
                selection_exposure: attempted.saturating_add(pending_plans),
                in_flight: reserved.saturating_add(pending_plans),
                unknown: r.get(3)?,
                ready_at: r.get(4)?,
                selected_in_population: false,
            })
        })?,
    )
}

/// Read and bind a frozen selection on both controller and rank paths.
pub fn selected_indices(path: &Path, corpus: &Path, contexts: &[ContextRow]) -> Result<Vec<usize>> {
    let selection: serde_json::Value = read_json(path)?;
    if selection["schema"] == "jope.native_generation_selection.v1" {
        let request = &selection["request"];
        ensure!(
            request["protocol"] == "all_eligible_training_contexts_v1"
                && request["corpus_sha256"] == file_hash(&corpus.join("index.json"))?,
            "native selection corpus/protocol changed"
        );
        let root = std::path::PathBuf::from(request["root"].as_str().context("selection root")?);
        ensure!(
            root.is_absolute() && root.canonicalize()? == root && corpus.starts_with(&root),
            "selection physical root differs"
        );
        let policy: crate::native_loop::ScientificPolicy = crate::native_genome::load(
            &root,
            &serde_json::from_value(request["scientific_policy"].clone())?,
        )?;
        ensure!(
            policy.schema == "jope.native_learning_policy.v1"
                && policy.sealed_partition == "meta_test",
            "native scientific selection policy differs"
        );
        let start = request["cut_start"]
            .as_u64()
            .context("selection cut start")?;
        let cuts = request["cuts"].as_u64().context("selection cuts")?;
        ensure!(cuts > 0, "empty selection cut budget");
        let overlay: Option<crate::native_graph::Artifact> =
            serde_json::from_value(request["preparation_overlay"].clone())?;
        let views = overlay
            .as_ref()
            .map(|a| {
                crate::preparation_overlay::views(
                    &root,
                    a,
                    request["run"].as_str().context("overlay selection run")?,
                )
            })
            .transpose()?;
        let expected: Vec<_> = contexts
            .iter()
            .enumerate()
            .filter(|(_, c)| {
                views
                    .as_ref()
                    .is_none_or(|v| v.contains(&(c.dataset.clone(), c.cut)))
                    && c.partition == "meta_train"
                    && !policy.excluded_families.contains(&c.family)
                    && c.cut as u64 >= start
                    && (c.cut as u64) - start < cuts
            })
            .map(|(i, c)| serde_json::json!({"context_index":i,"identity":Context::from_row(c)}))
            .collect();
        ensure!(
            !expected.is_empty() && selection["contexts"] == serde_json::to_value(&expected)?,
            "native selection omitted or added eligible contexts"
        );
        return expected
            .iter()
            .map(|v| Ok(v["context_index"].as_u64().context("context index")? as usize))
            .collect();
    }

    ensure!(
        selection["schema"] == "parallel.coverage_selection.v1"
            && selection["request"]["protocol"] == PROTOCOL
            && selection["request"]["implementation"] == IMPLEMENTATION
            && selection["request"]["corpus_sha256"] == file_hash(&corpus.join("index.json"))?,
        "coverage selection corpus or protocol changed"
    );
    let mut seen = std::collections::BTreeSet::new();
    selection["contexts"]
        .as_array()
        .context("coverage selection identities absent")?
        .iter()
        .map(|entry| {
            let i = usize::try_from(
                entry["context_index"]
                    .as_u64()
                    .context("coverage index absent")?,
            )?;
            let row = contexts.get(i).context("coverage index out of range")?;
            ensure!(
                seen.insert(i)
                    && row.partition == "meta_train"
                    && entry["identity"] == serde_json::to_value(Context::from_row(row))?,
                "coverage selection identity changed"
            );
            Ok(i)
        })
        .collect()
}

fn old_is_untracked(tx: &rusqlite::Transaction<'_>, job: &GpJob) -> Result<bool> {
    Ok(!tx.query_row(
        "SELECT EXISTS(SELECT 1 FROM coverage_requests WHERE output=?1)",
        [job.output.to_string_lossy()],
        |r| r.get::<_, bool>(0),
    )?)
}

/// Artifact locations are aliases, not new computation identities. Distinct
/// seeded/generation/parent requests may share a context; renamed copies may not.
fn request_identity(job: &GpJob, context: &Context) -> Result<String> {
    let mut binding = serde_json::to_value(job)?;
    for key in ["output", "prepared", "seeds", "parent_checkpoint"] {
        binding.as_object_mut().unwrap().remove(key);
    }
    Ok(hash(&serde_json::to_vec(&(
        IMPLEMENTATION,
        context,
        binding,
    ))?))
}

#[cfg(test)]
mod tests;
