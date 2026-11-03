use super::prelude::*;
use super::{
    cache::validate_cache,
    context::encode_context,
    inventory::{cache_path, compare_mirror, valid_dataset_id},
    model::{
        load_v1_quarantine, v1_identity_denied, ExpansionPaths, FileLock, InventoryEntry,
        SealedCohort, CONTEXT_PROTOCOL, CONTEXT_STRIDE, DATASET_DIM, OPTIONAL_ENCODINGS,
        REQUIRED_ENCODINGS, TARGET_DIM, TASKS, TRANCHE_MAX,
    },
    sealed::{load_sealed_exclusions, validate_sealed_cohort},
};

pub(crate) fn activated_identities(catalog_root: &Path) -> Result<HashSet<(String, String)>> {
    let ledger = catalog_root.join("activations.jsonl");
    if !ledger.is_file() {
        return Ok(HashSet::new());
    }
    let mut identities = HashSet::new();
    for line in BufReader::new(File::open(ledger)?).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let receipt: Value = serde_json::from_str(&line)?;
        let catalog = receipt["catalog"]
            .as_str()
            .context("activation receipt lacks catalog")?;
        let value = read_json(Path::new(catalog))?;
        for row in value["entries"]
            .as_array()
            .context("catalog entries are absent")?
        {
            identities.insert((
                row["dataset"]["task"]
                    .as_str()
                    .context("catalog task is absent")?
                    .to_owned(),
                row["dataset"]["dataset_hash"]
                    .as_str()
                    .context("catalog dataset hash is absent")?
                    .to_owned(),
            ));
        }
    }
    Ok(identities)
}

pub(crate) fn existing_activation(catalog_root: &Path, tranche_id: &str) -> Result<Option<Value>> {
    let ledger = catalog_root.join("activations.jsonl");
    if !ledger.is_file() {
        return Ok(None);
    }
    for line in BufReader::new(File::open(ledger)?).lines() {
        let value: Value = serde_json::from_str(&line?)?;
        if value["tranche_id"].as_str() != Some(tranche_id) {
            continue;
        }
        let catalog = Path::new(
            value["catalog"]
                .as_str()
                .context("activation receipt lacks catalog")?,
        );
        let data = Path::new(
            value["data"]
                .as_str()
                .context("activation receipt lacks data")?,
        );
        ensure!(
            sha256_file(catalog)?
                == value["catalog_sha256"]
                    .as_str()
                    .context("activation receipt lacks catalog hash")?
                && sha256_file(data)?
                    == value["data_sha256"]
                        .as_str()
                        .context("activation receipt lacks data hash")?,
            "activation receipt content drift"
        );
        return Ok(Some(value));
    }
    Ok(None)
}

pub(crate) fn atomic_bytes(path: &Path, bytes: &[u8]) -> Result<()> {
    fs::create_dir_all(path.parent().context("output path has no parent")?)?;
    let temporary = path.with_file_name(format!(
        ".{}.tmp-{}-{}",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("data"),
        std::process::id(),
        now_ms()
    ));
    let mut output = File::create(&temporary)?;
    output.write_all(bytes)?;
    output.sync_all()?;
    fs::rename(temporary, path)?;
    Ok(())
}

pub fn activate_tranche(
    paths: &ExpansionPaths,
    tranche_id: &str,
    requested: &[InventoryEntry],
) -> Result<Value> {
    ensure!(!tranche_id.is_empty(), "tranche id is absent");
    ensure!(
        !requested.is_empty() && requested.len() <= TRANCHE_MAX,
        "adaptive tranche must contain 1..={TRANCHE_MAX} datasets"
    );
    let _lock = FileLock::exclusive(&paths.catalog_root.join(".activation.lock"))?;
    if let Some(receipt) = existing_activation(&paths.catalog_root, tranche_id)? {
        return Ok(receipt);
    }
    let quarantine = load_v1_quarantine(paths)?;
    let (sealed_identities, sealed_families) = load_sealed_exclusions(paths)?;
    let activated = activated_identities(&paths.catalog_root)?;
    let mut seen = HashSet::new();
    let mut contexts = Vec::with_capacity(requested.len());
    let mut entries = Vec::with_capacity(requested.len());
    for (index, row) in requested.iter().enumerate() {
        ensure!(
            !v1_identity_denied(quarantine.as_ref(), &row.dataset_id),
            "catalog adoption denied for quarantined V1 identity {}",
            row.dataset_id
        );
        let identity = (row.task.clone(), row.dataset_hash.clone());
        ensure!(seen.insert(identity.clone()), "tranche repeats a dataset");
        ensure!(
            !sealed_identities.contains(&identity) && !sealed_families.contains(&row.source_family),
            "adaptive tranche intersects the sealed cohort"
        );
        ensure!(
            !activated.contains(&identity),
            "dataset was already activated in an earlier tranche"
        );
        compare_mirror(paths, &row.task, &row.dataset_id, &row.source)?;
        let cache = cache_path(paths, &row.task, &row.dataset_id);
        let manifest = validate_cache(&cache, Some(row))?;
        let context = encode_context(&cache, &row.dataset_hash, &paths.standardization)?;
        let context_sha256 = sha256_bytes(&context);
        let cache_manifest_sha256 = sha256_file(&cache.join("cache-manifest.v1.json"))?;
        let encoding_hashes = manifest
            .files
            .iter()
            .filter(|(name, _)| {
                REQUIRED_ENCODINGS.contains(&name.as_str())
                    || OPTIONAL_ENCODINGS.contains(&name.as_str())
            })
            .map(|(name, digest)| (name.clone(), digest.clone()))
            .collect::<BTreeMap<_, _>>();
        entries.push(CatalogEntryIdentity {
            dataset: row.dataset_identity(),
            source_family: row.source_family.clone(),
            cache_manifest_sha256,
            encoder_protocol_sha256: manifest.encoder_protocol_sha256,
            context_protocol_sha256: canonical_json_sha256(&json!({"protocol": CONTEXT_PROTOCOL}))?,
            context_sha256,
            encoding_hashes,
            catalog_index: index,
            split: jope_datasets::family_contracts::source_family_campaign_split(
                &row.source_family,
            )
            .to_owned(),
        });
        contexts.push(context);
    }
    let blob = contexts.concat();
    let data_sha256 = sha256_bytes(&blob);
    let task_counts = TASKS
        .into_iter()
        .filter_map(|task| {
            let count = entries
                .iter()
                .filter(|entry| entry.dataset.task == task)
                .count();
            (count != 0).then(|| (task.to_owned(), count))
        })
        .collect::<BTreeMap<_, _>>();
    let identity = CatalogIdentity {
        row_count: entries.len(),
        task_counts: task_counts.clone(),
        data_sha256: data_sha256.clone(),
        entries: entries.clone(),
    };
    identity.validate()?;
    fs::create_dir_all(&paths.catalog_root)?;
    let data_path = paths.catalog_root.join(format!("{data_sha256}.f32.bin"));
    if data_path.is_file() {
        ensure!(
            sha256_file(&data_path)? == data_sha256,
            "expanded context blob content drift"
        );
    } else {
        atomic_bytes(&data_path, &blob)?;
        ensure!(
            sha256_file(&data_path)? == data_sha256,
            "expanded context blob write validation failed"
        );
    }
    let catalog_path = paths.catalog_root.join(format!("{tranche_id}.json"));
    let existing_catalog = catalog_path
        .is_file()
        .then(|| read_json(&catalog_path))
        .transpose()?;
    let activated_at_ms = existing_catalog
        .as_ref()
        .and_then(|value| value["activated_at_ms"].as_u64())
        .map(u128::from)
        .unwrap_or_else(now_ms);
    let mut payload = json!({
        "schema": "JopeExpandedCatalog.v2",
        "immutable": true,
        "append_only": true,
        "tranche_id": tranche_id,
        "task_counts": task_counts,
        "row_count": entries.len(),
        "dataset_dimensions": DATASET_DIM,
        "target_dimensions": TARGET_DIM,
        "record_stride_bytes": CONTEXT_STRIDE,
        "context_protocol": CONTEXT_PROTOCOL,
        "data_sha256": data_sha256,
        "source_family_disjoint": true,
        "entries": entries,
        "activated_at_ms": activated_at_ms,
    });
    payload["catalog_content_sha256"] = Value::String(canonical_json_sha256(&payload)?);
    if let Some(existing) = existing_catalog {
        ensure!(
            existing == payload,
            "refusing to rewrite immutable expanded tranche"
        );
    } else {
        atomic_json(&catalog_path, &payload)?;
    }
    let receipt = json!({
        "schema": "JopeExpandedCatalogActivation.v2",
        "tranche_id": tranche_id,
        "catalog": catalog_path,
        "catalog_sha256": sha256_file(&catalog_path)?,
        "data": data_path,
        "data_sha256": data_sha256,
        "row_count": requested.len(),
        "task_counts": task_counts,
        "activated_at_ms": payload["activated_at_ms"],
    });
    let ledger_path = paths.catalog_root.join("activations.jsonl");
    let mut ledger = OpenOptions::new()
        .create(true)
        .append(true)
        .mode(0o644)
        .open(&ledger_path)?;
    serde_json::to_writer(&mut ledger, &receipt)?;
    ledger.write_all(b"\n")?;
    ledger.sync_all()?;
    atomic_json(
        &paths.catalog_root.join("current.json"),
        &json!({
            "schema": "JopeExpandedCatalogPointer.v2",
            "catalog": catalog_path,
            "catalog_sha256": sha256_file(&catalog_path)?,
            "data": data_path,
            "data_sha256": data_sha256,
            "task_counts": task_counts,
            "row_count": requested.len(),
            "tranche_id": tranche_id,
        }),
    )?;
    Ok(receipt)
}

pub fn validate_catalog_pointer(paths: &ExpansionPaths) -> Result<Value> {
    let pointer_path = paths.catalog_root.join("current.json");
    let pointer = read_json(&pointer_path)?;
    ensure!(
        pointer["schema"].as_str() == Some("JopeExpandedCatalogPointer.v2"),
        "expanded catalog pointer schema drift"
    );
    let catalog_path = Path::new(
        pointer["catalog"]
            .as_str()
            .context("expanded catalog pointer lacks catalog")?,
    );
    let data_path = Path::new(
        pointer["data"]
            .as_str()
            .context("expanded catalog pointer lacks data")?,
    );
    ensure!(
        sha256_file(catalog_path)?
            == pointer["catalog_sha256"]
                .as_str()
                .context("pointer lacks catalog hash")?,
        "expanded catalog pointer hash drift"
    );
    ensure!(
        sha256_file(data_path)?
            == pointer["data_sha256"]
                .as_str()
                .context("pointer lacks data hash")?,
        "expanded catalog data hash drift"
    );
    let payload = read_json(catalog_path)?;
    let entries: Vec<CatalogEntryIdentity> = serde_json::from_value(payload["entries"].clone())?;
    let identity = CatalogIdentity {
        row_count: payload["row_count"]
            .as_u64()
            .context("catalog lacks row count")? as usize,
        task_counts: serde_json::from_value(payload["task_counts"].clone())?,
        data_sha256: payload["data_sha256"]
            .as_str()
            .context("catalog lacks data hash")?
            .to_owned(),
        entries,
    };
    identity.validate()?;
    ensure!(
        fs::metadata(data_path)?.len() as usize == identity.row_count * CONTEXT_STRIDE,
        "expanded catalog context stride drift"
    );
    let mut input = BufReader::new(File::open(data_path)?);
    let mut raw = vec![0_u8; CONTEXT_STRIDE];
    for entry in &identity.entries {
        input.read_exact(&mut raw)?;
        ensure!(
            sha256_bytes(&raw) == entry.context_sha256,
            "expanded catalog context slice drift"
        );
    }
    let mut trailing = [0_u8; 1];
    ensure!(
        input.read(&mut trailing)? == 0,
        "expanded catalog data has trailing bytes"
    );
    Ok(json!({
        "healthy": true,
        "tranche_id": pointer["tranche_id"],
        "row_count": identity.row_count,
        "task_counts": identity.task_counts,
        "data_sha256": identity.data_sha256,
    }))
}

pub fn status(paths: &ExpansionPaths) -> Result<Value> {
    let mut counts = BTreeMap::new();
    let mut valid = BTreeMap::new();
    let mut invalid = BTreeMap::new();
    for task in TASKS {
        let root = paths.cache_root.join(format!("{task}-expanded"));
        let directories = if root.is_dir() {
            fs::read_dir(&root)?
                .filter_map(std::result::Result::ok)
                .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_dir()))
                .filter(|entry| valid_dataset_id(&entry.file_name().to_string_lossy()))
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        let mut good = 0;
        let mut bad = 0;
        for directory in &directories {
            if validate_cache(&directory.path(), None).is_ok() {
                good += 1;
            } else {
                bad += 1;
            }
        }
        counts.insert(task.to_owned(), directories.len());
        valid.insert(task.to_owned(), good);
        invalid.insert(task.to_owned(), bad);
    }
    let sealed = if paths.sealed_path.is_file() {
        let cohort: SealedCohort = serde_json::from_value(read_json(&paths.sealed_path)?)?;
        validate_sealed_cohort(&cohort)?;
        json!({"frozen": true, "cohort_sha256": cohort.cohort_sha256, "counts": cohort.counts})
    } else {
        json!({"frozen": false})
    };
    let catalog = if paths.catalog_root.join("current.json").is_file() {
        validate_catalog_pointer(paths)?
    } else {
        json!({"healthy": false, "active": false})
    };
    let activation_count = if paths.catalog_root.join("activations.jsonl").is_file() {
        BufReader::new(File::open(paths.catalog_root.join("activations.jsonl"))?)
            .lines()
            .collect::<std::io::Result<Vec<_>>>()?
            .into_iter()
            .filter(|line| !line.trim().is_empty())
            .count()
    } else {
        0
    };
    Ok(json!({
        "schema": "JopeExpandedDatasetStatus.v2",
        "cache_directories": counts,
        "valid_caches": valid,
        "invalid_caches": invalid,
        "sealed": sealed,
        "active_catalog": catalog,
        "activation_count": activation_count,
    }))
}

pub fn load_inventory_file(path: &Path) -> Result<Vec<InventoryEntry>> {
    let value = read_json(path)?;
    let entries = if value.is_array() {
        value
    } else {
        value["entries"].clone()
    };
    serde_json::from_value(entries).context("inventory file has invalid entries")
}

pub fn write_inventory_file(path: &Path, entries: &[InventoryEntry]) -> Result<Value> {
    let counts = TASKS
        .into_iter()
        .map(|task| {
            (
                task.to_owned(),
                entries.iter().filter(|entry| entry.task == task).count(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let mut payload = json!({
        "schema": "JopeExpandedInventory.v2",
        "authoritative_root": jope_paths::to_string(&paths::quant_root()),
        "mirror_root": jope_paths::to_string(&paths::quant_mirror_root()),
        "row_count": entries.len(),
        "task_counts": counts,
        "entries": entries,
        "created_at_ms": now_ms(),
    });
    payload["inventory_sha256"] = Value::String(canonical_json_sha256(&payload)?);
    atomic_json(path, &payload)?;
    Ok(json!({
        "inventory": path,
        "inventory_sha256": payload["inventory_sha256"],
        "row_count": entries.len(),
        "task_counts": counts,
    }))
}

pub fn select_unactivated_cached(
    paths: &ExpansionPaths,
    inventory: &[InventoryEntry],
    limit: usize,
) -> Result<Vec<InventoryEntry>> {
    ensure!(
        (1..=TRANCHE_MAX).contains(&limit),
        "limit must be 1..={TRANCHE_MAX}"
    );
    let (sealed_identities, sealed_families) = load_sealed_exclusions(paths)?;
    let activated = activated_identities(&paths.catalog_root)?;
    let quarantine = load_v1_quarantine(paths)?;
    let mut selected = Vec::new();
    for row in inventory {
        if v1_identity_denied(quarantine.as_ref(), &row.dataset_id) {
            continue;
        }
        let identity = (row.task.clone(), row.dataset_hash.clone());
        if sealed_identities.contains(&identity)
            || sealed_families.contains(&row.source_family)
            || activated.contains(&identity)
        {
            continue;
        }
        let cache = cache_path(paths, &row.task, &row.dataset_id);
        if validate_cache(&cache, Some(row)).is_ok()
            && paths.standardization.is_file()
            && read_json(&paths.standardization)?
                .get("foundation")
                .and_then(|value| value.get("cache_dir"))
                .and_then(Value::as_str)
                .is_some_and(|root| {
                    Path::new(root)
                        .join(format!("{}.npy", row.dataset_hash))
                        .is_file()
                })
        {
            selected.push(row.clone());
            if selected.len() == limit {
                break;
            }
        }
    }
    ensure!(
        !selected.is_empty(),
        "no fully encoded adaptive datasets are ready"
    );
    Ok(selected)
}
