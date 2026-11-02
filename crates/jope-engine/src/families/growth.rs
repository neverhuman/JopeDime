use super::*;

pub fn binding(
    path: &Path,
    dataset_id: &str,
    source_key: &str,
    audit_family: &str,
) -> Result<RegistryBinding> {
    let registry: Registry = read_json(path)?;
    let family = registry.assignment(dataset_id, source_key, audit_family)?;
    let dataset = family
        .datasets
        .iter()
        .find(|d| d.dataset == dataset_id && d.source_key == source_key)
        .unwrap();
    ensure!(
        dataset.preparation_eligible,
        "dataset has no verified reconstruction/preparation eligibility in this registry"
    );
    Ok(RegistryBinding {
        registry_sha256: file_hash(path)?,
        registry_id: registry.registry_id.clone(),
        family: family.family.clone(),
        partition: family.partition.clone(),
        metadata_sha256: dataset.metadata_sha256.clone(),
        qualification: true,
        full_inventory_complete: false,
        entity_review_sha256: dataset.entity_review_sha256.clone(),
    })
}

/// Publish membership only. New components have no prepared datasets, context
/// permissions or dispatch authority. The development reserve is deliberately
/// distinct from the existing eight-family meta_validation panel.
pub fn freeze_growth(
    parent_path: &Path,
    assignment_path: &Path,
    output: &Path,
) -> Result<Registry> {
    let directory = output
        .parent()
        .context("growth output requires directory")?;
    fs::create_dir_all(directory)?;
    let lock = OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(output.with_extension("lock"))?;
    ensure!(lock.try_lock().is_ok(), "growth output locked");
    ensure!(!output.exists(), "immutable growth registry already exists");
    let binding = FileBinding::new(assignment_path)?;
    let receipt = verified_growth(&binding)?;
    let mut registry: Registry = read_json(parent_path)?;
    registry.validate()?;
    let parent = FileBinding::new(parent_path)?;
    ensure!(
        receipt["parent_registry"]["sha256"] == parent.sha256,
        "growth uses a stale parent registry"
    );
    ensure!(
        receipt["execution_eligible"] == false && receipt["runtime_closure_qualified"] == false,
        "growth cannot authorize execution"
    );
    let root = Path::new(field(&receipt, "root")?);
    let graph_ref: FileBinding = serde_json::from_value(receipt["graph"].clone())?;
    let graph: Value = read_json(&root.join(graph_ref.path))?;
    let quarantined: BTreeSet<String> =
        serde_json::from_value(receipt["quarantined_components"].clone())?;
    for component in graph["components"]
        .as_array()
        .context("growth graph components missing")?
    {
        let cid = field(component, "component_id")?;
        if quarantined.contains(cid) {
            let aliases: BTreeSet<String> = serde_json::from_value(component["aliases"].clone())?;
            for family in registry.families.values_mut() {
                if aliases.contains(&family.family) || !aliases.is_disjoint(&family.source_keys) {
                    family
                        .holds
                        .insert(format!("alias_merge_crosses_frozen_partitions:{cid}"));
                }
            }
        }
    }
    for row in receipt["rows"].as_array().context("growth rows missing")? {
        let name = field(row, "component_id")?.to_owned();
        ensure!(
            !registry.families.contains_key(&name),
            "growth would replace an existing component"
        );
        let (digest, bucket, partition) = stable_component_assignment(&name);
        registry.families.insert(
            name.clone(),
            FamilyAssignment {
                family: name,
                partition,
                assignment_method: "stable_sha256_component_v2".into(),
                assignment_hash: Some(digest),
                bucket: Some(bucket),
                source_keys: serde_json::from_value(row["aliases"].clone())?,
                datasets: vec![],
                holds: BTreeSet::new(),
            },
        );
    }
    registry.parent = Some(parent);
    registry.growth.push(binding);
    registry.validate()?;
    write_json(output, &registry)?;
    File::open(directory)?.sync_all()?;
    Ok(registry)
}
