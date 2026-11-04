use super::*;

fn read_json_or_jsonl(path: &Path) -> Result<Vec<Value>> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    if let Ok(value) = serde_json::from_slice::<Value>(&bytes) {
        return match value {
            Value::Array(rows) => Ok(rows),
            Value::Object(mut object) => {
                for key in ["lineage", "records", "members", "population"] {
                    if let Some(Value::Array(rows)) = object.remove(key) {
                        return Ok(rows);
                    }
                }
                Ok(vec![Value::Object(object)])
            }
            _ => bail!("lineage JSON must be an object or array"),
        };
    }
    let mut rows = Vec::new();
    for (index, line) in bytes.split(|byte| *byte == b'\n').enumerate() {
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        rows.push(
            serde_json::from_slice(line)
                .with_context(|| format!("parse lineage JSONL line {}", index + 1))?,
        );
    }
    Ok(rows)
}

pub(crate) fn lineage_value(
    path: &Path,
    champion: &str,
    task: Task,
    dataset: &str,
    run_id: &str,
) -> Result<Value> {
    validate_sha256(champion, "final champion semantic SHA-256")?;
    ensure!(!run_id.trim().is_empty(), "run identity must not be empty");
    let mut rows = Vec::new();
    let mut semantics = HashSet::new();
    for (index, value) in read_json_or_jsonl(path)?.into_iter().enumerate() {
        let row: LineageRow = serde_json::from_value(value)
            .with_context(|| format!("parse lineage record {index}"))?;
        validate_sha256(
            &row.semantic_sha256,
            &format!("lineage record {index} semantic SHA-256"),
        )?;
        if let Some(genome) = &row.canonical_genome {
            ensure!(
                sha256_bytes(genome.as_bytes()) == row.semantic_sha256,
                "lineage semantic identity drift at record {index}"
            );
        }
        ensure!(
            semantics.insert(row.semantic_sha256.clone()),
            "lineage repeats a semantic identity at record {index}"
        );
        for parent in row.all_parents() {
            validate_sha256(parent, "lineage parent semantic SHA-256")?;
        }
        rows.push(row);
    }
    ensure!(
        semantics.contains(champion),
        "final champion is absent from lineage"
    );
    let origins = rows
        .iter()
        .filter(|row| row.dl_origin)
        .map(|row| row.semantic_sha256.clone())
        .collect::<BTreeSet<_>>();
    let mut contributed = origins.clone();
    loop {
        let before = contributed.len();
        for row in &rows {
            if row.all_parents().any(|parent| contributed.contains(parent)) {
                contributed.insert(row.semantic_sha256.clone());
            }
        }
        if contributed.len() == before {
            break;
        }
    }
    let descendants = contributed
        .difference(&origins)
        .cloned()
        .collect::<BTreeSet<_>>();
    let strict_winners = rows
        .iter()
        .filter(|row| row.strict_win && contributed.contains(&row.semantic_sha256))
        .map(|row| row.semantic_sha256.clone())
        .collect::<BTreeSet<_>>();
    Ok(json!({
        "schema": LINEAGE_SCHEMA,
        "task": task.as_str(),
        "dataset_hash": dataset,
        "run_id": run_id,
        "lineage_sha256": sha256_file(path)?,
        "final_champion_semantic_sha256": champion,
        "dl_origin_parents": origins,
        "dl_origin_parent_count": origins.len(),
        "dl_origin_descendants": descendants,
        "dl_origin_descendant_count": descendants.len(),
        "dl_origin_strict_wins": strict_winners,
        "dl_origin_strict_win_count": strict_winners.len(),
        "final_champion_has_dl_contribution": contributed.contains(champion),
    }))
}
