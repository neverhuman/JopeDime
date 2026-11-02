mod selection_finalize;
mod selection_math;
pub(crate) use selection_finalize::finalize_selector;

fn arm_name(arm: Arm) -> &'static str {
    match arm {
        Arm::G20Extension => "g20-extension",
        Arm::SmartV3 => "smart-v3",
    }
}

pub(crate) fn run_arm(
    manifest_path: &Path,
    arm: Arm,
    selection_mode: SelectionMode,
    resume: bool,
    workers: usize,
    selected_dataset: Option<&str>,
) -> Result<PathBuf> {
    if workers == 0 {
        bail!("--workers must be positive");
    }
    if selection_mode == SelectionMode::SmartOnly && arm != Arm::SmartV3 {
        bail!("smart-only selection mode runs only the Smart-v3 arm");
    }
    let (manifest, manifest_digest) = validate_manifest(manifest_path)?;
    let root = manifest_path
        .parent()
        .context("red-team manifest has no run root")?;
    let datasets = manifest
        .datasets
        .iter()
        .filter(|dataset| selected_dataset.is_none_or(|selected| selected == dataset.dataset_hash))
        .collect::<Vec<_>>();
    if datasets.is_empty() {
        bail!("selected dataset is absent from the red-team manifest");
    }
    let aggregate_only = selected_dataset.is_none()
        && std::env::var("REDTEAM_AGGREGATE_ONLY").as_deref() == Ok("true");
    let mut receipt_sha256 = BTreeMap::new();
    let mut selected_output = None;
    for dataset in datasets {
        let (path, digest) = if aggregate_only {
            match arm {
                Arm::G20Extension => {
                    let path = root
                        .join("datasets")
                        .join(&dataset.dataset_hash)
                        .join("g20-extension-complete.v1.json");
                    let (receipt, digest): (G20Complete, _) = read_receipt(&path)?;
                    if receipt.schema != "TrainOnlyRedteamG20Complete.v1"
                        || receipt.run_id != manifest.run_id
                        || receipt.dataset_hash != dataset.dataset_hash
                        || !receipt.development_only
                        || receipt.confirmatory_valid
                    {
                        bail!("aggregate-only G20 receipt drift");
                    }
                    (path, digest)
                }
                Arm::SmartV3 => {
                    let path = root
                        .join("datasets")
                        .join(&dataset.dataset_hash)
                        .join("smart-v3-complete.v1.json");
                    let (receipt, digest): (SmartComplete, _) = read_receipt(&path)?;
                    if receipt.schema != "TrainOnlyRedteamSmartComplete.v1"
                        || receipt.run_id != manifest.run_id
                        || receipt.dataset_hash != dataset.dataset_hash
                        || !receipt.development_only
                        || receipt.confirmatory_valid
                    {
                        bail!("aggregate-only smart receipt drift");
                    }
                    (path, digest)
                }
            }
        } else {
            match arm {
                Arm::G20Extension => run_g20_dataset(root, &manifest, dataset, resume, workers)?,
                Arm::SmartV3 => run_smart_dataset(root, &manifest, dataset, resume, workers)?,
            }
        };
        receipt_sha256.insert(dataset.dataset_hash.clone(), digest);
        selected_output = Some(path);
        if !aggregate_only {
            let _ = finalize_selector(root, &manifest, dataset, selection_mode, workers)?;
        }
    }
    if selected_dataset.is_some() {
        return selected_output.context("arm produced no output");
    }
    if receipt_sha256.len() != manifest.datasets.len() {
        bail!("global arm completion requires every configured dataset");
    }
    let receipt = ArmComplete {
        schema: "TrainOnlyRedteamArmComplete.v1".to_string(),
        run_id: manifest.run_id.clone(),
        arm: arm_name(arm).to_string(),
        manifest_sha256: manifest_digest.clone(),
        dataset_receipt_sha256: receipt_sha256,
        scq_execution: scq_execution_provenance(&manifest, None, workers),
        development_only: true,
        confirmatory_valid: false,
    };
    let path = root.join(format!("{}-complete.v1.json", arm_name(arm)));
    write_receipt(&path, &receipt)?;

    let selector_filename = match selection_mode {
        SelectionMode::Combined => "selector-complete.v1.json",
        SelectionMode::SmartOnly => "smart-v3-selector-complete.v1.json",
    };
    if manifest.datasets.iter().all(|dataset| {
        root.join("datasets")
            .join(&dataset.dataset_hash)
            .join(selector_filename)
            .exists()
    }) {
        let selector_sha256 = manifest
            .datasets
            .iter()
            .map(|dataset| {
                let path = root
                    .join("datasets")
                    .join(&dataset.dataset_hash)
                    .join(selector_filename);
                let digest = match selection_mode {
                    SelectionMode::Combined => read_receipt::<SelectorComplete>(&path)?.1,
                    SelectionMode::SmartOnly => read_receipt::<SmartV3SelectorComplete>(&path)?.1,
                };
                Ok((dataset.dataset_hash.clone(), digest))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        let (cohort_path, schema) = match selection_mode {
            SelectionMode::Combined => (
                root.join("selector-complete.v1.json"),
                "TrainOnlyRedteamSelectorCohortComplete.v1",
            ),
            SelectionMode::SmartOnly => (
                root.join("smart-v3-selector-complete.v1.json"),
                "TrainOnlyRedteamSmartV3SelectorCohortComplete.v1",
            ),
        };
        write_receipt(
            &cohort_path,
            &json!({
                "schema": schema,
                "run_id": manifest.run_id,
                "manifest_sha256": manifest_digest,
                "selector_sha256": selector_sha256,
                "scq_execution": scq_execution_provenance(&manifest, None, workers),
                "selection_frozen_before_test_access": true,
                "development_only": true,
                "confirmatory_valid": false,
            }),
        )?;
    }
    Ok(path)
}
