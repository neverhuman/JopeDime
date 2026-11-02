use super::*;

pub(super) fn build_transfer_rankings(
    run_id: &str,
    observation_db: &Path,
    observation_db_sha256: &str,
    catalog: &Catalog,
    catalog_sha256: &str,
    target_entries: &[CatalogEntry],
) -> Result<BTreeMap<String, GlobalRankingReceipt>> {
    let targets = target_entries.to_vec();
    let family_by_dataset = catalog
        .entries
        .iter()
        .map(|entry| (entry.dataset_hash.clone(), entry.source_family.clone()))
        .collect::<HashMap<_, _>>();
    let connection = Connection::open_with_flags(
        observation_db,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )?;
    connection.execute_batch(
        "CREATE TEMP TABLE dataset_family(
            dataset_hash TEXT PRIMARY KEY,
            source_family TEXT NOT NULL
        ) WITHOUT ROWID;",
    )?;
    {
        let mut insert = connection
            .prepare("INSERT INTO dataset_family(dataset_hash, source_family) VALUES (?1, ?2)")?;
        for entry in &catalog.entries {
            insert.execute([&entry.dataset_hash, &entry.source_family])?;
        }
    }
    connection.execute_batch(
        "CREATE TEMP TABLE transfer_targets(
            dataset_hash TEXT PRIMARY KEY,
            source_family TEXT NOT NULL
        ) WITHOUT ROWID;",
    )?;
    {
        let mut insert = connection
            .prepare("INSERT INTO transfer_targets(dataset_hash, source_family) VALUES (?1, ?2)")?;
        for target in &targets {
            insert.execute([&target.dataset_hash, &target.source_family])?;
        }
    }

    let mut ranked = targets
        .iter()
        .map(|target| {
            (
                target.dataset_hash.clone(),
                Vec::<GlobalRankedGenome>::new(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let mut statement =
        connection.prepare(include_str!("../../sql/train_only_transfer_rankings.sql"))?;
    let mut rows = statement.query([])?;
    while let Some(row) = rows.next()? {
        let target_dataset_hash: String = row.get(0)?;
        let semantic_sha256: String = row.get(1)?;
        let canonical_genome: String = row.get(2)?;
        let source_ids = row
            .get::<_, String>(3)?
            .split(',')
            .map(str::to_string)
            .collect::<Vec<_>>();
        let mean_percentile: Option<f64> = row.get(4)?;
        let worst_percentile: Option<f64> = row.get(5)?;
        let strict_win_count = row.get::<_, i64>(6)? as usize;
        let dataset_count = row.get::<_, i64>(7)? as usize;
        let (Some(mean_percentile), Some(worst_percentile)) = (mean_percentile, worst_percentile)
        else {
            continue;
        };
        let mut included_ids = source_ids;
        included_ids.sort();
        included_ids.dedup();
        let source_families = included_ids
            .iter()
            .map(|dataset| {
                family_by_dataset
                    .get(dataset)
                    .cloned()
                    .unwrap_or_else(|| format!("dataset:{dataset}"))
            })
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let Some(target_rankings) = ranked.get_mut(&target_dataset_hash) else {
            bail!("transfer ranking query returned unexpected target {target_dataset_hash}");
        };
        target_rankings.push(GlobalRankedGenome {
            rank: 0,
            canonical_genome: canonical_genome.clone(),
            semantic_sha256: semantic_sha256.clone(),
            mean_percentile,
            worst_percentile,
            strict_win_count,
            dataset_count,
            source_dataset_ids: included_ids,
            source_families,
        });
    }

    let mut receipts = BTreeMap::new();
    for target in targets {
        let mut genomes = ranked.remove(&target.dataset_hash).unwrap();
        genomes.sort_by(|left, right| {
            right
                .mean_percentile
                .total_cmp(&left.mean_percentile)
                .then_with(|| right.worst_percentile.total_cmp(&left.worst_percentile))
                .then_with(|| right.strict_win_count.cmp(&left.strict_win_count))
                .then_with(|| right.dataset_count.cmp(&left.dataset_count))
                .then_with(|| left.semantic_sha256.cmp(&right.semantic_sha256))
        });
        genomes.truncate(50_000);
        for (rank, genome) in genomes.iter_mut().enumerate() {
            genome.rank = rank;
        }
        if genomes.len() < GLOBAL_QUOTA {
            bail!(
                "{} has fewer than {GLOBAL_QUOTA} target-clean transfer genomes",
                target.dataset_hash
            );
        }
        let excluded_dataset_ids = catalog
            .entries
            .iter()
            .filter(|entry| {
                entry.dataset_hash == target.dataset_hash
                    || entry.source_family == target.source_family
            })
            .map(|entry| entry.dataset_hash.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        receipts.insert(
            target.dataset_hash.clone(),
            GlobalRankingReceipt {
                schema: "TrainOnlyTransferGenomeRanking.v2".to_string(),
                run_id: run_id.to_string(),
                target_dataset_id: target.dataset_hash,
                target_source_family: target.source_family.clone(),
                observation_db_sha256: observation_db_sha256.to_string(),
                source_catalog_sha256: catalog_sha256.to_string(),
                excluded_dataset_ids,
                excluded_source_family: target.source_family,
                direct_dl_sources_excluded: true,
                ranking_policy:
                    "target and matching source family excluded; mean within-dataset percentile desc; worst percentile desc; strict win count desc; dataset count desc; semantic SHA-256 asc"
                        .to_string(),
                genomes,
            },
        );
    }
    Ok(receipts)
}
