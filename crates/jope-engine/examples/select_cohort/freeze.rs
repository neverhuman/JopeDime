use super::*;

pub(super) fn binding(root: &Path, path: &Path) -> Result<Binding> {
    let full = resolve(root, path)?;
    Ok(Binding {
        path: path.into(),
        sha256: file_hash(&full)?,
    })
}

pub(crate) fn freeze(
    root: &Path,
    registry_path: &Path,
    paths: &[PathBuf],
    output: &Path,
) -> Result<()> {
    let output = resolve(root, output)?;
    ensure!(!output.exists(), "selection receipt already exists");
    let registry: Registry = read_json(&resolve(root, registry_path)?)?;
    registry.validate()?;
    let mut manifests = paths
        .iter()
        .map(|p| binding(root, p))
        .collect::<Result<Vec<_>>>()?;
    manifests.sort_by(|a, b| a.path.cmp(&b.path));
    ensure!(
        manifests.windows(2).all(|w| w[0].path != w[1].path),
        "duplicate input job manifest"
    );
    let mut by_dataset: BTreeMap<String, Candidate> = BTreeMap::new();
    let mut prepared = BTreeMap::<PathBuf, Prepared>::new();
    let mut prepared_hashes = BTreeMap::<PathBuf, String>::new();
    let mut vocabulary: Option<Binding> = None;
    let mut cuts = BTreeMap::<(String, usize), String>::new();
    for input in &manifests {
        let jobs: GpJobs = read_json(&resolve(root, &input.path)?)?;
        ensure!(
            jobs.schema == "parallel.gp_jobs.v1" && jobs.qualification,
            "expected qualification jobs"
        );
        let vocab = binding(
            root,
            jobs.vocabulary.as_ref().context("job vocabulary absent")?,
        )?;
        if let Some(old) = &vocabulary {
            ensure!(old.sha256 == vocab.sha256, "job vocabularies differ");
        } else {
            vocabulary = Some(vocab);
        }
        for job in jobs.jobs {
            let manifest = job.prepared.join("prepared.json");
            if !prepared.contains_key(&job.prepared) {
                ensure!(
                    file_hash(&resolve(root, &manifest)?)? == job.prepared_sha256,
                    "prepared input changed"
                );
                let p: Prepared = read_json(&resolve(root, &manifest)?)?;
                ensure!(p.schema == "parallel.prepared.v1" && p.task == "regression" && p.target_last && p.clean_evaluation && !p.qualification_only,
                    "cohort requires reconstructed regression cuts without known original split overlap");
                let storage = p
                    .storage
                    .as_ref()
                    .context("cohort requires pointer-only preparations")?;
                let dataset_id = &p.dataset;
                let matches = registry
                    .families
                    .values()
                    .flat_map(|f| {
                        f.datasets
                            .iter()
                            .filter(move |d| &d.dataset == dataset_id)
                            .map(move |d| (f, d))
                    })
                    .collect::<Vec<_>>();
                ensure!(
                    matches.len() == 1,
                    "dataset registry identity missing or ambiguous: {}",
                    p.dataset
                );
                let (family, dataset) = matches[0];
                ensure!(
                    family.holds.is_empty() && dataset.preparation_eligible,
                    "held or unadmitted registry family"
                );
                ensure!(
                    family.family == p.family && family.partition == p.meta_partition,
                    "prepared family partition changed"
                );
                if let Some(b) = &p.family_registry_binding {
                    ensure!(
                        b.registry_id == registry.registry_id
                            && b.family == p.family
                            && b.partition == p.meta_partition
                            && b.metadata_sha256 == dataset.metadata_sha256
                            && b.entity_review_sha256 == dataset.entity_review_sha256,
                        "prepared registry lineage mismatch"
                    );
                }
                let candidate = Candidate {
                    dataset: p.dataset.clone(),
                    family: p.family.clone(),
                    partition: p.meta_partition.clone(),
                    source_key: dataset.source_key.clone(),
                    metadata_sha256: dataset.metadata_sha256.clone(),
                    source_identity_sha256: hash(&serde_json::to_vec(
                        &json!({"storage":p.storage,"original_train_sha256":p.original_train_sha256,"original_test_sha256":p.original_test_sha256,"selected_train_indices":p.selected_train_indices,"selected_test_indices":p.selected_test_indices,"entity_columns":p.entity_columns}),
                    )?),
                    rows: p.rows_total,
                    features: storage.feature_columns.len(),
                    eligible: true,
                    holds: vec![],
                    prepared_versions: vec![Binding {
                        path: manifest,
                        sha256: job.prepared_sha256.clone(),
                    }],
                };
                if let Some(previous) = by_dataset.get_mut(&p.dataset) {
                    ensure!(
                        previous.family == candidate.family
                            && previous.partition == candidate.partition
                            && previous.rows == candidate.rows
                            && previous.features == candidate.features
                            && previous.source_identity_sha256 == candidate.source_identity_sha256,
                        "prepared versions disagree on source identity, cohort dimensions, or family"
                    );
                    previous
                        .prepared_versions
                        .extend(candidate.prepared_versions);
                } else {
                    by_dataset.insert(p.dataset.clone(), candidate);
                }
                prepared.insert(job.prepared.clone(), p);
                prepared_hashes.insert(job.prepared.clone(), job.prepared_sha256.clone());
            }
            ensure!(
                prepared_hashes[&job.prepared] == job.prepared_sha256,
                "job prepared hash changes within manifest"
            );
            let p = &prepared[&job.prepared];
            let cut = p
                .cuts
                .iter()
                .find(|c| c.index == job.cut)
                .context("job cut absent")?;
            ensure!(
                cut.train_indices
                    .iter()
                    .all(|i| !cut.validation_indices.contains(i)),
                "cut row overlap"
            );
            let key = (p.dataset.clone(), job.cut);
            if let Some(old) = cuts.insert(key, job.prepared_sha256.clone()) {
                ensure!(
                    old == job.prepared_sha256,
                    "conflicting prepared versions for a cut"
                );
            }
        }
    }
    let mut candidates = by_dataset.into_values().collect::<Vec<_>>();
    for c in &mut candidates {
        c.prepared_versions.sort_by(|a, b| a.path.cmp(&b.path));
    }
    let selected = select(&candidates)?;
    let receipt=Selection {schema:"parallel.qualification_cohort.v1".into(),seed:SEED,qualification_only:true,independent_selection_eligible:false,
        registry:binding(root,registry_path)?,job_manifests:manifests,vocabulary:vocabulary.context("no vocabulary")?,
        method:"one dataset per family chosen by seeded SHA256; per partition round-robin fixed row bins [<256,<1024,<4096,<16384,larger] and feature bins [<=4,<=16,<=64,larger], with seeded stratum/family ordering; no outcomes or scores".into(),
        projection_limit:"Projection v2 was fitted on 55 meta-training families, exceeding the selected 48; this cohort remains qualification-only and does not establish fresh independent pilot evaluation.".into(),
        quotas:quotas(),candidates,selected,original_final_test_targets_opened:false,model_scores_consulted:false};
    write_json(&output, &receipt)?;
    println!(
        "{}",
        json!({"selected_families":64,"qualification_only":true,"receipt":output,"sha256":file_hash(&output)?})
    );
    Ok(())
}
