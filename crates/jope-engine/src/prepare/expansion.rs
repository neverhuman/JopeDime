use super::*;

/// Repeated, seeded five-fold partitions add teacher views without copying data.
/// Duplicate training memberships are skipped; source families do not multiply.
pub fn expand_cuts(prepared: &Path, output: &Path, rounds: usize) -> Result<Prepared> {
    ensure!(rounds > 0 && rounds <= 1000, "rounds must be in 1..=1000");
    ensure!(
        !output.exists(),
        "expanded destination exists; use an immutable new version"
    );
    let mut p: Prepared = read_json(&prepared.join("prepared.json"))?;
    ensure!(
        p.task != "timeseries",
        "time-series expansion requires distinct rolling origins"
    );
    let source = p
        .storage
        .as_ref()
        .context("expansion requires pointer views")?;
    let raw = load_raw_view(prepared, source, &p.selected_train_indices)?;
    reject_target_copies(&source.feature_columns, &raw)?;
    let entities = entity_values(&raw, &source.feature_columns, &p.entity_columns)?;
    fs::create_dir_all(output)?;
    let parent_sha = file_hash(&prepared.join("prepared.json"))?;
    let mut cuts = Vec::new();
    let mut seen = BTreeSet::new();
    let mut recipes = Vec::new();
    let mut skipped = 0;
    for round in 0..rounds {
        let seed = if round == 0 {
            p.seed
        } else {
            u64::from_str_radix(
                &hash(format!("{}:repeated-folds:{round}", p.seed).as_bytes())[..16],
                16,
            )?
        };
        for (fold, (tr, va)) in folds_with_seed(&raw, &p.task, 1, entities.as_deref(), seed)?
            .into_iter()
            .enumerate()
        {
            let membership = hash(&serde_json::to_vec(&tr)?);
            if !seen.insert(membership) {
                skipped += 1;
                continue;
            }
            let index = round * 5 + fold;
            let train_indices: Vec<usize> =
                tr.iter().map(|&i| p.selected_train_indices[i]).collect();
            let validation_indices: Vec<usize> =
                va.iter().map(|&i| p.selected_train_indices[i]).collect();
            check_cut_membership(
                &source.original_training_row_map,
                &train_indices,
                &validation_indices,
            )?;
            let training = tr.iter().map(|&i| raw[i].clone()).collect::<Vec<_>>();
            let heldout = va.iter().map(|&i| raw[i].clone()).collect::<Vec<_>>();
            let pp = Preprocessing::fit(&training, &p.task)?;
            let dir = output.join(format!("cut-{index}"));
            fs::create_dir_all(&dir)?;
            write_json(&dir.join("preprocessing.json"), &pp)?;
            let cut = Cut {
                index,
                train_indices,
                validation_indices,
                training_sha256: hash(&csv_bytes(&pp.transform(&training)?)?),
                validation_sha256: hash(&csv_bytes(&pp.transform(&heldout)?)?),
                preprocessing_sha256: file_hash(&dir.join("preprocessing.json"))?,
            };
            if round == 0 {
                let old = p
                    .cuts
                    .iter()
                    .find(|c| c.index == fold)
                    .context("base fold absent")?;
                ensure!(
                    old.training_sha256 == cut.training_sha256
                        && old.validation_sha256 == cut.validation_sha256,
                    "base fold changed during expansion"
                );
            }
            recipes.push(
                serde_json::json!({"cut":index,"round":round,"fold":fold,"partition_seed":seed}),
            );
            cuts.push(cut);
        }
    }
    p.cuts = cuts;
    write_json(&output.join("prepared.json"), &p)?;
    write_json(
        &output.join("expansion.json"),
        &serde_json::json!({"schema":"parallel.repeated_folds.v1","parent_prepared_sha256":parent_sha,"rounds":rounds,"cuts":recipes,"duplicate_cuts_skipped":skipped,"source_family":p.family,"independent_families":1,"copied_dataset_bytes":0,"heldout_role":"adaptive GP teacher objective; not an unbiased final test","final_test_targets_used":false}),
    )?;
    Ok(p)
}

/// Adapt an already verified cut to pointers and prove exact byte parity before
/// it is used. Original cut manifests remain as historical evidence.
pub fn index_existing(
    prepared: &Path,
    backing_manifest: &Path,
    metadata: &Path,
    output: &Path,
) -> Result<()> {
    ensure!(!output.exists(), "indexed destination exists");
    let mut p: Prepared = read_json(&prepared.join("prepared.json"))?;
    let reconstruction: Value = read_json(&prepared.join("reconstruction.json"))?;
    let receipt = &reconstruction["receipt"];
    let m: Value = read_json(metadata)?;
    let transfer: TransferManifest = read_json(backing_manifest)?;
    let sha = receipt["source_sha256"]
        .as_str()
        .context("raw source identity missing")?;
    let entry = transfer
        .files
        .iter()
        .find(|f| f.sha256 == sha)
        .context("backing source not allowlisted")?;
    p.storage = Some(ViewSource {
        source: entry.destination.clone(),
        source_sha256: sha.into(),
        feature_columns: serde_json::from_value(m["feature_columns"].clone())?,
        target_column: m["target_column"]
            .as_str()
            .or_else(|| m["source_target_column_original"].as_str())
            .context("target name missing")?
            .into(),
        original_training_row_map: serde_json::from_value(receipt["source_row_indices"].clone())?,
    });
    validate_view_source(p.storage.as_ref().context("indexed storage absent")?)?;
    fs::create_dir_all(output)?;
    for c in &p.cuts {
        let dir = output.join(format!("cut-{}", c.index));
        fs::create_dir_all(&dir)?;
        atomic_write(
            &dir.join("preprocessing.json"),
            &fs::read(prepared.join(format!("cut-{}/preprocessing.json", c.index)))?,
        )?;
    }
    write_json(&output.join("prepared.json"), &p)?;
    write_json(&output.join("reconstruction.json"), &reconstruction)?;
    for c in &p.cuts {
        load_view(output, c.index, true)?;
        load_view(output, c.index, false)?;
    }
    write_json(
        &output.join("pointer-parity.json"),
        &serde_json::json!({"passed":true,"source_prepared_sha256":file_hash(&prepared.join("prepared.json"))?,"views_checked":p.cuts.len()*2,"method":"exact SHA-256 of on-demand transformed view bytes","copied_dataset_bytes":0}),
    )
}
