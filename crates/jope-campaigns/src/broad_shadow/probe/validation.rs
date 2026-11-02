use super::*;

fn validate_protocol(value: &Value, expected: &str) -> Result<()> {
    exact_fields(value, PROTOCOL_FIELDS, &format!("{expected} protocol"))?;
    ensure!(
        field(value, "fidelity")?.as_str() == Some(expected)
            && field(value, "train_only")? == &Value::Bool(true)
            && integer(field(value, "omp_threads")?, "omp_threads")? == 1,
        "{expected} protocol violates train-only/OMP authority"
    );
    if expected == "screening" {
        ensure!(
            integer(field(value, "folds")?, "folds")? == 3
                && integer(field(value, "row_cap")?, "row_cap")? == 768
                && integer(field(value, "timeout_seconds")?, "timeout_seconds")? == 30,
            "screening protocol must be CV3/768 rows/30 seconds"
        );
    } else {
        ensure!(
            integer(field(value, "folds")?, "folds")? == 5
                && field(value, "row_cap")?.is_null()
                && field(value, "timeout_seconds")?.is_null(),
            "exact protocol must be full-train CV5 without screening caps"
        );
    }
    Ok(())
}

pub(super) fn validate_probe_slate(slate: &Value, production: bool) -> Result<()> {
    exact_fields(slate, SLATE_FIELDS, "probe slate")?;
    ensure!(
        field(slate, "schema")?.as_str() == Some(PROBE_SLATE_SCHEMA),
        "unsupported probe slate schema"
    );
    let claimed = sha256_field(field(slate, "slate_sha256")?, "slate_sha256")?;
    let mut unhashed = slate.clone();
    unhashed.as_object_mut().unwrap().remove("slate_sha256");
    ensure!(
        claimed == canonical_hash(&unhashed)?,
        "probe slate content hash differs"
    );
    sha256_field(field(slate, "catalog_sha256")?, "catalog_sha256")?;
    positive_integer(field(slate, "created_at_ms")?, "created_at_ms")?;
    integer(field(slate, "seed")?, "probe seed")?;
    validate_protocol(field(slate, "screening_protocol")?, "screening")?;
    validate_protocol(field(slate, "exact_protocol")?, "exact")?;

    let anchors = field(slate, "anchors")?
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("probe slate requires exactly four anchors"))?;
    ensure!(
        anchors.len() == ANCHOR_COUNT,
        "probe slate requires exactly four anchors"
    );
    let mut anchor_semantics = Vec::new();
    let mut anchor_families = BTreeSet::new();
    for (index, anchor) in anchors.iter().enumerate() {
        exact_fields(anchor, ANCHOR_FIELDS, "probe anchor")?;
        ensure!(
            integer(field(anchor, "anchor_index")?, "anchor_index")? == index as i64
                && integer(
                    field(anchor, "performance_quartile")?,
                    "performance_quartile"
                )? == index as i64,
            "anchors must cover ordered performance quartiles 0..3"
        );
        ensure!(
            matches!(
                field(anchor, "engine")?.as_str(),
                Some("featmath" | "newmath")
            ),
            "probe anchor engine is unsupported"
        );
        let semantic =
            sha256_field(field(anchor, "semantic_sha256")?, "anchor semantic_sha256")?.to_owned();
        let genome = text(
            field(anchor, "canonical_genome")?,
            "anchor canonical genome",
        )?;
        ensure!(
            !genome.is_empty(),
            "anchor canonical genome must be nonempty"
        );
        ensure!(
            sha256_hex(genome.as_bytes()) == semantic,
            "anchor semantic hash does not bind its genome"
        );
        let family = slug(field(anchor, "algorithm_family")?, "algorithm_family")?.to_owned();
        ensure!(
            anchor_families.insert(family),
            "anchor top-level algorithm families must be distinct"
        );
        let coverage = field(anchor, "real_dataset_coverage")?
            .as_f64()
            .ok_or_else(|| anyhow::anyhow!("real_dataset_coverage must be finite"))?;
        ensure!(
            coverage >= 100.0,
            "anchor must cover at least 100 real datasets"
        );
        let runtime = field(anchor, "median_runtime_seconds")?
            .as_f64()
            .ok_or_else(|| anyhow::anyhow!("anchor median_runtime_seconds must be finite"))?;
        ensure!(
            runtime.is_finite() && runtime >= 0.0,
            "anchor median_runtime_seconds must be finite"
        );
        anchor_semantics.push(semantic);
    }
    ensure!(
        anchor_semantics.iter().collect::<BTreeSet<_>>().len() == ANCHOR_COUNT,
        "anchor semantic hashes must be distinct"
    );

    let datasets = field(slate, "datasets")?
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("probe slate datasets must be nonempty"))?;
    ensure!(
        !datasets.is_empty(),
        "probe slate datasets must be nonempty"
    );
    let mut previous: Option<&str> = None;
    for row in datasets {
        exact_fields(row, DATASET_FIELDS, "probe dataset")?;
        let dataset = dataset_hash(field(row, "dataset_hash")?, "dataset_hash")?;
        if let Some(previous) = previous {
            ensure!(
                previous < dataset,
                "probe datasets must be strictly sorted and unique"
            );
        }
        previous = Some(dataset);
        ensure!(
            matches!(
                field(row, "synthetic_split")?.as_str(),
                Some("train" | "validation" | "test")
            ),
            "probe dataset must bind a synthetic family split"
        );
        let slots = field(row, "slots")?
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("each probe dataset requires exactly ten slots"))?;
        ensure!(
            slots.len() == PILOT_GENOMES_PER_DATASET,
            "each probe dataset requires exactly ten slots"
        );
        let mut roles = std::collections::BTreeMap::<String, usize>::new();
        let mut semantics = BTreeSet::new();
        for (index, slot) in slots.iter().enumerate() {
            exact_fields(slot, SLOT_FIELDS, "probe slot")?;
            let role = text(field(slot, "role")?, "role")?;
            let role_count = SLOT_ROLES
                .iter()
                .find(|(name, _)| *name == role)
                .map(|(_, count)| *count);
            ensure!(
                integer(field(slot, "slot")?, "slot")? == index as i64 && role_count.is_some(),
                "probe slots must be ordered and use frozen roles"
            );
            ensure!(
                matches!(
                    field(slot, "engine")?.as_str(),
                    Some("featmath" | "newmath")
                ),
                "probe slot engine is unsupported"
            );
            let semantic =
                sha256_field(field(slot, "semantic_sha256")?, "slot semantic_sha256")?.to_owned();
            let genome = field(slot, "canonical_genome")?.as_str().unwrap_or("");
            ensure!(
                !genome.is_empty() && sha256_hex(genome.as_bytes()) == semantic,
                "probe slot semantic hash does not bind its genome"
            );
            ensure!(
                semantics.insert(semantic.clone()),
                "probe dataset contains a duplicate semantic genome"
            );
            *roles.entry(role.to_owned()).or_default() += 1;
            if role == "anchor" {
                let anchor_index = integer(field(slot, "anchor_index")?, "anchor_index")?;
                ensure!(
                    anchor_index == index as i64
                        && anchor_index >= 0
                        && (anchor_index as usize) < anchors.len()
                        && semantic == anchor_semantics[anchor_index as usize]
                        && field(slot, "engine")?
                            == field(&anchors[anchor_index as usize], "engine")?
                        && field(slot, "source_dataset_hash")?.is_null(),
                    "anchor slots must exactly match ProbeSlate anchors"
                );
            } else {
                ensure!(
                    field(slot, "anchor_index")?.is_null(),
                    "non-anchor slot cannot carry anchor_index"
                );
                let source = field(slot, "source_dataset_hash")?;
                if role == "transfer" || !source.is_null() {
                    dataset_hash(source, "source_dataset_hash")?;
                }
            }
        }
        ensure!(
            roles
                == SLOT_ROLES
                    .iter()
                    .map(|(name, count)| ((*name).to_owned(), *count))
                    .collect(),
            "probe dataset role counts must be 4/2/2/2"
        );
    }
    if production {
        ensure!(
            datasets.len() == PILOT_DATASET_COUNT,
            "production pilot requires exactly 2,000 synthetic datasets"
        );
    }
    Ok(())
}
