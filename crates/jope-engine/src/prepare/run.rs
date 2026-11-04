use super::*;

pub fn run(
    audit: &Path,
    dataset: &str,
    output: &Path,
    allow_legacy: bool,
    qualification: bool,
    backing_manifest: &Path,
) -> Result<Prepared> {
    run_with_admission(
        audit,
        dataset,
        output,
        allow_legacy,
        qualification,
        backing_manifest,
        None,
        None,
        None,
    )
}

/// `authoritative_inventory_root`: the dataset tree the caller declares
/// authoritative. Without a frozen family registry, the family partition is
/// frozen only when the audit is complete and was taken over exactly that
/// root; with neither, it is not frozen. No environment variable decides it.
#[allow(clippy::too_many_arguments)]
pub fn run_with_admission(
    audit: &Path,
    dataset: &str,
    output: &Path,
    allow_legacy: bool,
    qualification: bool,
    backing_manifest: &Path,
    family_registry: Option<&Path>,
    entity_review: Option<&Path>,
    authoritative_inventory_root: Option<&Path>,
) -> Result<Prepared> {
    let report: AuditReport = read_json(audit)?;
    let candidates = report
        .datasets
        .iter()
        .filter(|d| d.id == dataset)
        .collect::<Vec<_>>();
    let mut matching = Vec::new();
    for d in candidates {
        if d.admission == Admission::Excluded {
            continue;
        }
        if let Some(path) = family_registry {
            if let Ok(binding) = crate::families::binding(path, &d.id, &d.source_key, &d.family) {
                if binding.metadata_sha256 == d.metadata_sha256 {
                    matching.push((d, Some(binding)));
                }
            }
        } else if d.admission == Admission::Approved {
            matching.push((d, None));
        }
    }
    // The full inventory includes acquisition stubs and release copies. Prefer
    // the canonical task/id directory, after any registry metadata constraint.
    let canonical = |d: &Dataset| {
        d.source_dir.file_name().and_then(|s| s.to_str()) == Some(d.id.as_str())
            && d.source_dir
                .parent()
                .and_then(|p| p.file_name())
                .and_then(|s| s.to_str())
                == Some(if d.task == "multiclass" {
                    "multi_label"
                } else {
                    d.task.as_str()
                })
    };
    if matching.iter().any(|(d, _)| canonical(d)) {
        matching.retain(|(d, _)| canonical(d));
    }
    ensure!(
        matching.len() == 1,
        "dataset needs exactly one approved or registry-bound canonical audit identity"
    );
    let (d, family_binding) = matching.pop().unwrap();
    ensure!(
        d.admission == Admission::Approved
            || family_binding
                .as_ref()
                .is_some_and(|b| b.entity_review_sha256.is_some()),
        "dataset needs an approved audit or frozen reviewed preparation admission"
    );
    if let Some(binding) = &family_binding {
        ensure!(
            !binding.qualification || qualification,
            "family registry requires explicit qualification admission"
        );
        ensure!(
            binding.entity_review_sha256.is_some() == entity_review.is_some(),
            "provide exactly the entity review required by the frozen registry"
        );
    } else {
        ensure!(
            entity_review.is_none(),
            "entity review requires a frozen family registry"
        );
    }
    // W1-T06: the audited files are read only as regular files of the audited
    // directory, never through a symlink that may resolve elsewhere.
    let audited = |name: &str| -> Result<std::path::PathBuf> {
        let path = d.source_dir.join(name);
        let meta = fs::symlink_metadata(&path).with_context(|| format!("audited {name} absent"))?;
        ensure!(
            !meta.file_type().is_symlink() && meta.is_file(),
            "audited {name} is a symlink or not a regular file; it is not followed"
        );
        Ok(path)
    };
    let meta_path = audited("meta.json")?;
    let train_path = audited("train.csv")?;
    // Route R8: parse exactly the metadata bytes the audit bound.
    let meta_bytes = read_verified(&meta_path, &d.metadata_sha256, "audited metadata")
        .context("metadata changed since audit")?;
    let m: Value = serde_json::from_slice(&meta_bytes)
        .with_context(|| format!("parse {}", meta_path.display()))?;
    if let Some(binding) = &family_binding {
        ensure!(
            binding.metadata_sha256 == d.metadata_sha256,
            "metadata differs from frozen family registry"
        );
    }
    let audited_transfer: TransferManifest = read_json(
        &audit
            .parent()
            .context("audit parent absent")?
            .join("transfer.json"),
    )?;
    // Route R8: the historical training split is read once. When the audit
    // bound its bytes, exactly those bytes are consumed; otherwise the digest
    // of the consumed bytes is recorded as unaudited.
    let train_identity = audited_transfer
        .files
        .iter()
        .find(|f| f.source == d.source_dir.join("train.csv"));
    let train_bytes = match train_identity {
        Some(entry) => read_verified(&train_path, &entry.sha256, "audited training split")?,
        None => fs::read(&train_path)?,
    };
    let original = Table::parse(&train_bytes, &m)?;
    let test = d
        .test
        .as_ref()
        .context("audited final-test dimensions absent")?;
    ensure!(original.width == test.columns, "split schema mismatch");
    let test_identity = audited_transfer
        .files
        .iter()
        .find(|f| f.source == d.source_dir.join("test.csv"));
    ensure!(
        test_identity.is_some() || (qualification && d.task == "binary"),
        "audited final-test hash absent; final targets remain sealed"
    );
    // Route R1: a missing recorded backing path may be relocated only to a
    // location the explicit backing allowlist names.
    let transfer: TransferManifest = read_json(backing_manifest)?;
    let reconstruction = if matches!(d.task.as_str(), "binary" | "multiclass") {
        reconstruct_classification_with(&original, &m, &transfer)
    } else {
        reconstruct_recorded(&original, &m, false, Some(&transfer))
    };
    let (train, reconstruction_receipt, reconstruction_error) = match reconstruction {
        Ok((table, receipt)) => (table, Some(receipt), None),
        Err(e) => (original, None, Some(format!("{e:#}"))),
    };
    // The reconstructed cut can be used for training. Final-test preprocessing
    // must be reconstructed and qualified separately before a score is claimed.
    let clean =
        reconstruction_receipt.is_some() && d.split_feature_overlap == 0 && test_identity.is_some();
    ensure!(
        clean || allow_legacy || (qualification && reconstruction_receipt.is_some()),
        "backing-source row mapping/preprocessing reconstruction required; original-test overlap requires explicit qualification (reconstruction: {})",
        reconstruction_error.as_deref().unwrap_or("succeeded")
    );
    let (ti, ei) = cap_indices(train.rows.len(), test.rows, &d.task, &train.rows)?;
    let receipt = reconstruction_receipt
        .as_ref()
        .context("pointer cuts require reconstructed source rows")?;
    let source_hash = receipt["source_sha256"]
        .as_str()
        .context("source hash absent")?;
    let backing = transfer
        .files
        .iter()
        .find(|f| f.sha256 == source_hash)
        .context("backing source is not in the transfer allowlist")?;
    let storage = ViewSource {
        source: backing.destination.clone(),
        source_sha256: source_hash.into(),
        feature_columns: m["feature_columns"]
            .as_array()
            .context("feature order absent")?
            .iter()
            .map(|v| v.as_str().unwrap_or("").to_string())
            .collect(),
        target_column: receipt["target_column"]
            .as_str()
            .context("target column absent")?
            .into(),
        original_training_row_map: serde_json::from_value(receipt["source_row_indices"].clone())?,
    };
    validate_view_source(&storage)?;
    let rows = ti
        .iter()
        .map(|&i| train.rows[i].clone())
        .collect::<Vec<_>>();
    reject_target_copies(&storage.feature_columns, &rows)?;
    let horizon = m
        .get("horizon")
        .and_then(Value::as_u64)
        .unwrap_or(test.rows as u64) as usize;
    let (entity_columns, grouping_binding) = if let Some(path) = entity_review {
        let expected = family_binding
            .as_ref()
            .and_then(|b| b.entity_review_sha256.as_deref())
            .context("frozen entity review identity absent")?;
        let (columns, binding) = reviewed_entity_columns(path, expected, d, &m, receipt)?;
        (columns, Some(binding))
    } else {
        (documented_entity_columns(&m), None)
    };
    let entities = entity_values(&rows, &storage.feature_columns, &entity_columns)?;
    let indices = folds(&rows, &d.task, horizon, entities.as_deref())?;
    for (tr, va) in &indices {
        check_cut_membership(
            &storage.original_training_row_map,
            &tr.iter().map(|&i| ti[i]).collect::<Vec<_>>(),
            &va.iter().map(|&i| ti[i]).collect::<Vec<_>>(),
        )?;
    }
    // Route R11: whether the family partition is frozen is decided only by an
    // explicit authority - a frozen family registry, or an inventory root the
    // caller declares - never by the environment (`JOPE_REGISTRY`).
    let (frozen, frozen_authority) = match (&family_binding, authoritative_inventory_root) {
        (Some(binding), _) => (
            binding.full_inventory_complete,
            serde_json::json!({"kind":"frozen_family_registry"}),
        ),
        (None, Some(root)) => (
            report.complete && report.roots.iter().any(|r| r == root),
            serde_json::json!({"kind":"declared_inventory_root","root":root,
                "audit_complete":report.complete}),
        ),
        (None, None) => (false, serde_json::json!({"kind":"none"})),
    };
    ensure!(
        !output.exists(),
        "prepared destination already exists; use a new immutable version"
    );
    fs::create_dir_all(output)?;
    write_json(
        &output.join("reconstruction.json"),
        &serde_json::json!({"receipt":reconstruction_receipt,"error":reconstruction_error,"sealed_test_reconstruction":"not_yet_qualified","original_test_feature_overlap":d.split_feature_overlap,"original_test_targets_opened":false,"test_identity_source":if test_identity.is_some(){"existing audited transfer manifest"}else{"unavailable_unopened_binary_qualification_only"},"training_split_identity_source":if train_identity.is_some(){"audited transfer manifest (consumed bytes verified)"}else{"unaudited: digest of the consumed bytes"},"family_partition_frozen_authority":frozen_authority,"entity_columns":entity_columns,"grouping_source_dataset_id":m["source_dataset_id"],"grouping_documentation":entity_documentation(&m),"grouping_binding":grouping_binding,"family_registry_binding":family_binding}),
    )?;
    let mut cuts = Vec::new();
    for (index, (tr, va)) in indices.into_iter().enumerate() {
        let training = tr.iter().map(|&i| rows[i].clone()).collect::<Vec<_>>();
        let validation = va.iter().map(|&i| rows[i].clone()).collect::<Vec<_>>();
        let pp = Preprocessing::fit(&training, &d.task)?;
        let dir = output.join(format!("cut-{index}"));
        fs::create_dir_all(&dir)?;
        let tb = csv_bytes(&pp.transform(&training)?)?;
        let vb = csv_bytes(&pp.transform(&validation)?)?;
        write_json(&dir.join("preprocessing.json"), &pp)?;
        cuts.push(Cut {
            index,
            train_indices: tr.iter().map(|&i| ti[i]).collect(),
            validation_indices: va.iter().map(|&i| ti[i]).collect(),
            training_sha256: hash(&tb),
            validation_sha256: hash(&vb),
            preprocessing_sha256: file_hash(&dir.join("preprocessing.json"))?,
        });
    }
    let p = Prepared {
        storage: Some(storage),
        schema: "parallel.prepared.v1".into(),
        dataset: d.id.clone(),
        family: family_binding
            .as_ref()
            .map(|b| b.family.clone())
            .unwrap_or_else(|| d.family.clone()),
        task: d.task.clone(),
        meta_partition: family_binding
            .as_ref()
            .map(|b| b.partition.clone())
            .or_else(|| d.meta_partition.clone())
            .context("missing family partition")?,
        seed: SEED,
        family_partition_frozen: frozen,
        clean_evaluation: clean,
        clean_status: if clean {
            "raw_training_reconstructed_final_test_unqualified"
        } else if qualification && reconstruction_receipt.is_some() {
            "raw_training_reconstructed_qualification_only"
        } else {
            "legacy_preprocessing_not_reconstructed"
        }
        .into(),
        qualification_only: qualification && !clean,
        evaluation_limitations: if d.split_feature_overlap > 0 {
            vec![format!(
                "original_test_feature_overlap:{}; final evaluation remains ineligible",
                d.split_feature_overlap
            )]
        } else {
            vec![]
        },
        entity_columns,
        family_registry_binding: family_binding,
        original_train_sha256: hash(&train_bytes),
        original_test_sha256: test_identity.map(|f| f.sha256.clone()).unwrap_or_default(),
        selected_train_indices: ti,
        selected_test_indices: ei.clone(),
        target_last: true,
        rows_total: rows.len() + ei.len(),
        cuts,
    };
    // Seal metadata binds original evaluation bytes; no final targets enter cut files.
    write_json(&output.join("prepared.json"), &p)?;
    Ok(p)
}
