use super::*;

pub(crate) fn sqlite_table_exists(connection: &Connection, table: &str) -> Result<bool> {
    Ok(connection.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name=?1)",
        [table],
        |row| row.get(0),
    )?)
}

pub(crate) fn sqlite_column_exists(
    connection: &Connection,
    table: &str,
    column: &str,
) -> Result<bool> {
    let count: i64 = connection.query_row(
        "SELECT COUNT(*) FROM pragma_table_info(?1) WHERE name=?2",
        params![table, column],
        |row| row.get(0),
    )?;
    Ok(count != 0)
}

pub(crate) fn write_content_addressed(
    directory: &Path,
    prefix: &str,
    value: &Value,
) -> Result<(PathBuf, String)> {
    fs::create_dir_all(directory)?;
    let mut bytes = canonical_json_bytes(value)?;
    bytes.push(b'\n');
    let digest = sha256_bytes(&bytes);
    let path = directory.join(format!("{prefix}-{digest}.json"));
    match OpenOptions::new().write(true).create_new(true).open(&path) {
        Ok(mut file) => {
            file.write_all(&bytes)?;
            file.sync_all()?;
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            if sha256_file(&path)? != digest {
                bail!("content-addressed receipt collision at {}", path.display());
            }
        }
        Err(error) => return Err(error.into()),
    }
    Ok((path, digest))
}

pub(crate) fn write_json_atomic(path: &Path, value: &Value) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension(format!("tmp.{}", std::process::id()));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)?;
    file.write_all(&canonical_json_bytes(value)?)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    fs::rename(&temporary, path)?;
    Ok(())
}

/// The exact ranking key the seed-bank SQL uses (`ORDER BY mean_percentile
/// DESC, worst_percentile DESC, strict_wins DESC, dataset_count DESC,
/// semantic_sha256`). Both source pools are sorted by this key, so merging
/// preserves the global merit order.
pub(crate) fn seed_rank_order(left: &SeedRow, right: &SeedRow) -> std::cmp::Ordering {
    right
        .mean_within_dataset_percentile
        .total_cmp(&left.mean_within_dataset_percentile)
        .then_with(|| {
            right
                .worst_within_dataset_percentile
                .total_cmp(&left.worst_within_dataset_percentile)
        })
        .then_with(|| right.strict_win_count.cmp(&left.strict_win_count))
        .then_with(|| right.dataset_count.cmp(&left.dataset_count))
        .then_with(|| left.semantic_sha256.cmp(&right.semantic_sha256))
}

/// Merge one task's non-DL and DL seed pools (each already in rank order)
/// keeping global merit order, with DL-origin entries capped at `dl_cap`.
/// A semantic present in both pools earned its slot without DL evidence and
/// is kept as the non-DL entry only, so it never consumes DL cap.
pub(crate) fn merge_seed_pools(
    non_dl: Vec<SeedRow>,
    dl: Vec<SeedRow>,
    limit: usize,
    dl_cap: usize,
) -> Vec<SeedRow> {
    let non_dl_semantics: BTreeSet<String> = non_dl
        .iter()
        .map(|row| row.semantic_sha256.clone())
        .collect();
    let mut dl: Vec<SeedRow> = dl
        .into_iter()
        .filter(|row| !non_dl_semantics.contains(&row.semantic_sha256))
        .collect();
    dl.truncate(dl_cap);
    let mut merged = Vec::new();
    let mut non_dl = non_dl.into_iter().peekable();
    let mut dl = dl.into_iter().peekable();
    while merged.len() < limit {
        let take_dl = match (non_dl.peek(), dl.peek()) {
            (Some(left), Some(right)) => seed_rank_order(right, left) == std::cmp::Ordering::Less,
            (None, Some(_)) => true,
            (Some(_), None) => false,
            (None, None) => break,
        };
        let row = if take_dl { dl.next() } else { non_dl.next() };
        merged.push(row.expect("peeked branch is non-empty"));
    }
    merged
}

pub(crate) fn query_seed_rows(
    connection: &Connection,
    minimum_datasets: usize,
    limit: usize,
    dl_origin: bool,
) -> Result<Vec<SeedRow>> {
    let source_predicate = if dl_origin {
        "candidate_source IN ('accepted_dl_checkpoint','shadow_dl_direct')"
    } else {
        "candidate_source NOT IN ('accepted_dl_checkpoint','shadow_dl_direct')"
    };
    let sql = format!(
        r#"
WITH ranked AS (
  SELECT task,dataset_hash,semantic_sha256,canonical_genome,score,
         ROW_NUMBER() OVER (
           PARTITION BY task,dataset_hash ORDER BY score,semantic_sha256
         ) AS ascending_rank,
         COUNT(*) OVER (PARTITION BY task,dataset_hash) AS dataset_size,
         RANK() OVER (
           PARTITION BY task,dataset_hash ORDER BY score DESC
         ) AS descending_rank
  FROM observations
  WHERE task IN ('regression','binary')
    AND {source_predicate}
),
aggregated AS (
  SELECT task,semantic_sha256,MAX(canonical_genome) AS canonical_genome,
         COUNT(*) AS dataset_count,
         AVG(CASE WHEN dataset_size=1 THEN 1.0
                  ELSE 1.0*(ascending_rank-1)/(dataset_size-1) END) AS mean_percentile,
         MIN(CASE WHEN dataset_size=1 THEN 1.0
                  ELSE 1.0*(ascending_rank-1)/(dataset_size-1) END) AS worst_percentile,
         SUM(CASE WHEN descending_rank=1 THEN 1 ELSE 0 END) AS strict_wins
  FROM ranked
  GROUP BY task,semantic_sha256
  HAVING COUNT(*) >= ?1
),
numbered AS (
  SELECT *,
         ROW_NUMBER() OVER (
           PARTITION BY task
           ORDER BY mean_percentile DESC,worst_percentile DESC,
                    strict_wins DESC,dataset_count DESC,semantic_sha256
         ) AS robust_rank
  FROM aggregated
)
SELECT task,semantic_sha256,canonical_genome,dataset_count,
       mean_percentile,worst_percentile,strict_wins
FROM numbered
WHERE robust_rank <= ?2
ORDER BY task,robust_rank
"#
    );
    let mut statement = connection.prepare(&sql)?;
    let rows = statement.query_map(
        rusqlite::params![minimum_datasets as i64, limit as i64],
        |row| {
            Ok(SeedRow {
                task: row.get(0)?,
                semantic_sha256: row.get(1)?,
                canonical_genome: row.get(2)?,
                dataset_count: row.get(3)?,
                mean_within_dataset_percentile: row.get(4)?,
                worst_within_dataset_percentile: row.get(5)?,
                strict_win_count: row.get(6)?,
                dl_origin,
            })
        },
    )?;
    let mut seeds = Vec::new();
    for row in rows {
        let row = row?;
        if sha256_bytes(row.canonical_genome.as_bytes()) != row.semantic_sha256 {
            bail!("corpus semantic hash drift for {}", row.semantic_sha256);
        }
        seeds.push(row);
    }
    Ok(seeds)
}

pub(crate) fn export_seed_bank(
    path: &Path,
    limit: usize,
    minimum_datasets: usize,
) -> Result<Value> {
    let connection = Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .with_context(|| format!("open corpus {}", path.display()))?;
    connection.busy_timeout(std::time::Duration::from_secs(60))?;
    connection.execute_batch("PRAGMA temp_store=FILE; PRAGMA cache_size=-262144;")?;
    let dl_cap = limit / 3;
    let non_dl_rows = query_seed_rows(&connection, minimum_datasets, limit, false)?;
    // Fetch the DL pool at full limit so semantics deduped into the non-DL
    // pool do not underfill the cap; merge_seed_pools truncates to dl_cap
    // after dedupe.
    let dl_rows = query_seed_rows(&connection, minimum_datasets, limit, true)?;
    let mut non_dl_by_task: BTreeMap<String, Vec<SeedRow>> = BTreeMap::new();
    for row in non_dl_rows {
        non_dl_by_task
            .entry(row.task.clone())
            .or_default()
            .push(row);
    }
    let mut dl_by_task: BTreeMap<String, Vec<SeedRow>> = BTreeMap::new();
    for row in dl_rows {
        dl_by_task.entry(row.task.clone()).or_default().push(row);
    }
    let mut by_task: BTreeMap<String, Vec<SeedRow>> = BTreeMap::new();
    let tasks: BTreeSet<String> = non_dl_by_task
        .keys()
        .chain(dl_by_task.keys())
        .cloned()
        .collect();
    let mut dl_origin_counts = serde_json::Map::new();
    for task in tasks {
        let merged = merge_seed_pools(
            non_dl_by_task.remove(&task).unwrap_or_default(),
            dl_by_task.remove(&task).unwrap_or_default(),
            limit,
            dl_cap,
        );
        let dl_count = merged.iter().filter(|row| row.dl_origin).count();
        dl_origin_counts.insert(
            task.clone(),
            json!({
                "total": merged.len(),
                "dl_origin": dl_count,
                "non_dl": merged.len() - dl_count,
            }),
        );
        by_task.insert(task, merged);
    }
    Ok(json!({
        "schema": "JopeRobustGpSeedBank.v1",
        "created_at_ms": now_ms(),
        "source": path,
        "selection": {
            "task_separated": true,
            "minimum_repeat_datasets": minimum_datasets,
            "limit_per_task": limit,
            "within_dataset_rank": "percentile; higher is better",
            "order": [
                "mean_within_dataset_percentile_desc",
                "worst_within_dataset_percentile_desc",
                "strict_win_count_desc",
                "dataset_count_desc",
                "semantic_sha256_asc"
            ],
            "dl_origin_source": ["accepted_dl_checkpoint", "shadow_dl_direct"],
            "dl_origin_cap_per_task": dl_cap,
            "dl_origin_policy": "non-DL and accepted/shadow DL pools ranked separately by the same key, then merged in rank order; their combined DL-origin entries are capped at floor(limit/3); a semantic present in both pools counts as non-DL"
        },
        "dl_origin_counts": dl_origin_counts,
        "tasks": by_task,
    }))
}
