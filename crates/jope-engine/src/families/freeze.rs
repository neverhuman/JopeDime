use super::*;

pub(crate) fn field<'a>(v: &'a Value, name: &str) -> Result<&'a str> {
    v[name]
        .as_str()
        .with_context(|| format!("missing review field {name}"))
}

fn prepared_inputs(paths: &[PathBuf]) -> Result<Vec<(FileBinding, Value)>> {
    let mut discovered = BTreeSet::new();
    for path in paths {
        if path.is_file() {
            discovered.insert(path.canonicalize()?);
        } else {
            ensure!(
                path.is_dir(),
                "prepared input does not exist: {}",
                path.display()
            );
            for e in walkdir::WalkDir::new(path) {
                let e = e?;
                if e.file_type().is_file() && e.file_name() == "prepared.json" {
                    discovered.insert(e.path().canonicalize()?);
                }
            }
        }
    }
    discovered
        .iter()
        .map(|p| Ok((FileBinding::new(p)?, read_json(p)?)))
        .collect()
}

/// Create a new immutable revision. The output must not exist; active pointers are never updated.
pub fn freeze(
    audit_path: &Path,
    prepared_paths: &[PathBuf],
    review_path: &Path,
    decisions_path: &Path,
    parent_path: Option<&Path>,
    output: &Path,
) -> Result<Registry> {
    let directory = output
        .parent()
        .context("registry output requires a parent directory")?;
    fs::create_dir_all(directory)?;
    let lock = OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(output.with_extension("lock"))?;
    ensure!(
        lock.try_lock().is_ok(),
        "registry output is locked by another writer"
    );
    ensure!(!output.exists(), "immutable registry output already exists");
    let audit: AuditReport = read_json(audit_path)?;
    ensure!(audit.seed == SEED, "audit seed differs from registry seed");
    let audit_binding = FileBinding::new(audit_path)?;
    let prepared = prepared_inputs(prepared_paths)?;
    let prior = parent_path.map(read_json::<Registry>).transpose()?;
    if let Some(p) = &prior {
        p.validate()?;
    }
    let mut registry = prior.clone().unwrap_or(Registry {
        schema: SCHEMA.into(),
        seed: SEED,
        registry_id: hash(
            format!(
                "parallel.family_registry.v1:{SEED}:{}",
                audit_binding.sha256
            )
            .as_bytes(),
        ),
        qualification: true,
        full_inventory_complete: false,
        independent_selection_eligible: false,
        parent: None,
        audit: audit_binding.clone(),
        decisions: vec![],
        reviews: vec![],
        prepared_inputs: vec![],
        families: BTreeMap::new(),
        held_review_datasets: BTreeMap::new(),
        growth: vec![],
    });
    if let Some(path) = parent_path {
        registry.parent = Some(FileBinding::new(path)?);
    }
    ensure!(registry.audit.sha256 == audit_binding.sha256, "append must use the original audited identity snapshot; new admissions require reviewed evidence");
    // Full inventories include nested release copies and failed acquisition stubs.
    // Prefer the explicit task/id directory, then a stable path order; never let
    // scan order select an obsolete metadata identity for the same source.
    let mut approved: Vec<_> = audit
        .datasets
        .iter()
        .filter(|d| d.admission == Admission::Approved)
        .collect();
    approved.sort_by_key(|d| {
        let canonical = d.source_dir.file_name().is_some_and(|n| n == d.id.as_str())
            && d.source_dir
                .parent()
                .and_then(Path::file_name)
                .is_some_and(|n| n == d.task.as_str());
        (
            !canonical,
            d.source_dir.components().count(),
            d.source_dir.clone(),
        )
    });
    for d in approved {
        let partition = d
            .meta_partition
            .as_ref()
            .context("approved audit family lacks partition")?;
        let family = registry
            .families
            .entry(d.family.clone())
            .or_insert_with(|| FamilyAssignment {
                family: d.family.clone(),
                partition: partition.clone(),
                assignment_method: "preserved_audit_assignment".into(),
                assignment_hash: None,
                bucket: None,
                source_keys: BTreeSet::new(),
                datasets: vec![],
                holds: BTreeSet::new(),
            });
        ensure!(
            &family.partition == partition,
            "audit would reassign existing family {}",
            d.family
        );
        family.source_keys.insert(d.source_key.clone());
        if !family
            .datasets
            .iter()
            .any(|r| r.dataset == d.id && r.source_key == d.source_key)
        {
            for evidence in &d.evidence {
                evidence.verify()?;
            }
            family.datasets.push(RegisteredDataset {
                dataset: d.id.clone(),
                source_key: d.source_key.clone(),
                audit_family: d.family.clone(),
                metadata_sha256: d.metadata_sha256.clone(),
                entity_review_sha256: None,
                preparation_eligible: false,
                eligibility_reason: "no verified prepared view supplied".into(),
                prepared: vec![],
                evidence: d.evidence.clone(),
            });
        }
    }
    for (receipt, p) in prepared {
        let id = field(&p, "dataset")?;
        let family = registry
            .families
            .get_mut(field(&p, "family")?)
            .context("prepared family absent from audited registry")?;
        ensure!(
            field(&p, "meta_partition")? == family.partition,
            "prepared partition contradicts frozen audit assignment for {id}"
        );
        let record = family
            .datasets
            .iter_mut()
            .find(|d| d.dataset == id)
            .context("prepared dataset absent from audited family")?;
        let source = audit
            .datasets
            .iter()
            .find(|d| {
                d.id == id
                    && d.source_key == record.source_key
                    && d.metadata_sha256 == record.metadata_sha256
            })
            .context("prepared source missing from audit")?;
        if !record.prepared.iter().any(|p| p.sha256 == receipt.sha256) {
            record.prepared.push(receipt.clone());
        }
        if !registry
            .prepared_inputs
            .iter()
            .any(|p| p.sha256 == receipt.sha256)
        {
            registry.prepared_inputs.push(receipt);
        }
        if p["clean_evaluation"] == true
            && source.task == "regression"
            && source.split_feature_overlap == 0
            && p["storage"].is_object()
        {
            record.preparation_eligible = true;
            record.eligibility_reason = "existing raw reconstruction with zero audited original split feature overlap; final test remains unqualified".into();
        }
    }
    let review: Value = read_json(review_path)?;
    ensure!(
        review["schema"] == "parallel.provenance_review.v2" && review["active"] == false,
        "unsupported or already active provenance review"
    );
    ensure!(
        review["original_final_test_csv_opened"] == false,
        "review accessed original final tests"
    );
    let review_binding = FileBinding::new(review_path)?;
    let probe = if review["datasets"].as_array().is_some_and(|r| !r.is_empty()) {
        let path = Path::new(field(&review["reconstruction_probe"], "path")?);
        ensure!(
            file_hash(path)? == field(&review["reconstruction_probe"], "sha256")?,
            "review reconstruction probe changed"
        );
        Some(read_json::<Value>(path)?)
    } else {
        None
    };
    let decisions: Vec<Decision> = read_json(decisions_path)?;
    ensure!(
        decisions
            .iter()
            .map(|d| &d.key)
            .collect::<BTreeSet<_>>()
            .len()
            == decisions.len(),
        "duplicate provenance decisions"
    );
    if !registry
        .reviews
        .iter()
        .any(|r| r.sha256 == review_binding.sha256)
    {
        registry.reviews.push(review_binding.clone());
    }
    let decision_binding = FileBinding::new(decisions_path)?;
    if !registry
        .decisions
        .iter()
        .any(|r| r.sha256 == decision_binding.sha256)
    {
        registry.decisions.push(decision_binding);
    }
    for r in review["datasets"]
        .as_array()
        .context("missing reviewed datasets")?
    {
        let id = field(r, "id")?;
        let source_key = field(r, "source_key")?;
        if r["pilot_candidate_status"] != "eligible_for_fresh_cohort_preparation" {
            registry.held_review_datasets.insert(
                id.into(),
                r["pilot_hold_reason"]
                    .as_str()
                    .unwrap_or("not eligible")
                    .into(),
            );
            continue;
        }
        let reviewed_family = field(r, "proposed_family")?;
        let decision = decisions
            .iter()
            .find(|d| d.key == source_key)
            .context("review lacks matching provenance decision")?;
        ensure!(
            decision.admission == Admission::Approved && decision.family == reviewed_family,
            "review/decision admission mismatch"
        );
        for evidence in &decision.evidence {
            evidence.verify()?;
        }
        let source = audit
            .datasets
            .iter()
            .find(|d| {
                d.id == id
                    && d.source_key == source_key
                    && d.task == "regression"
                    && r["metadata"]["sha256"].as_str() == Some(d.metadata_sha256.as_str())
            })
            .context("reviewed dataset identity missing from full audit")?;
        ensure!(
            source.split_feature_overlap == 0,
            "reviewed source has audited split overlap"
        );
        ensure!(
            r["metadata"]["sha256"].as_str() == Some(&source.metadata_sha256),
            "reviewed metadata differs from audit"
        );
        ensure!(
            r["metadata"]["target_generation"] == "original_numeric_column",
            "review target is not original numeric column"
        );
        let reconstruction = &r["reconstruction"];
        ensure!(
            probe
                .as_ref()
                .and_then(|p| p["datasets"].as_array())
                .is_some_and(|rows| rows.iter().any(|p| p["id"] == id && p == reconstruction)),
            "review reconstruction is not probe-bound"
        );
        ensure!(
            reconstruction["reconstructed"] == true
                && reconstruction["metadata_sha256"].as_str() == Some(&source.metadata_sha256),
            "unverified reconstruction"
        );
        ensure!(
            reconstruction["max_match_abs_error"]
                .as_f64()
                .is_some_and(|e| e.is_finite() && e <= 1e-7),
            "reconstruction tolerance exceeded"
        );
        ensure!(
            file_hash(Path::new(field(&r["metadata"], "path")?))? == source.metadata_sha256,
            "review metadata payload changed"
        );
        ensure!(
            file_hash(Path::new(field(reconstruction, "source")?))?
                == field(reconstruction, "source_sha256")?,
            "review backing source changed"
        );
        ensure!(
            file_hash(&source.source_dir.join("train.csv"))?
                == field(reconstruction, "training_sha256")?,
            "review original training payload changed"
        );
        let documentation: Evidence = serde_json::from_value(r["documentation"].clone())?;
        documentation.verify()?;
        ensure!(
            decision
                .evidence
                .iter()
                .any(|e| e.sha256 == documentation.sha256),
            "review documentation is not decision-bound"
        );
        let mut sources = BTreeSet::from([source_key.to_string()]);
        let mut evidence = decision.evidence.clone();
        for alias in r["related_source_keys"]
            .as_array()
            .context("review aliases missing")?
        {
            if alias["relationship"] == "same_source_family" {
                let e: Evidence = serde_json::from_value(alias["evidence"].clone())?;
                e.verify()?;
                evidence.push(e);
                sources.insert(field(alias, "source_key")?.into());
            }
        }
        let owners: Vec<String> = registry
            .families
            .iter()
            .filter(|(name, f)| {
                name.as_str() == reviewed_family || !f.source_keys.is_disjoint(&sources)
            })
            .map(|(name, _)| name.clone())
            .collect();
        let name = owners
            .first()
            .map(String::as_str)
            .unwrap_or(reviewed_family)
            .to_string();
        if owners.len() > 1 {
            let partitions: BTreeSet<_> = owners
                .iter()
                .map(|o| registry.families[o].partition.clone())
                .collect();
            let hold = if partitions.len() > 1 {
                "alias_merge_crosses_frozen_partitions"
            } else {
                "alias_merge_of_existing_families_requires_review"
            };
            for owner in &owners {
                registry
                    .families
                    .get_mut(owner)
                    .unwrap()
                    .holds
                    .insert(format!("{hold}:{source_key}"));
            }
            registry.held_review_datasets.insert(id.into(), hold.into());
        }
        let family = registry.families.entry(name.clone()).or_insert_with(|| {
            let (digest, bucket, partition) = stable_assignment(&name);
            FamilyAssignment {
                family: name,
                partition,
                assignment_method: "stable_sha256_bucket_v1".into(),
                assignment_hash: Some(digest),
                bucket: Some(bucket),
                source_keys: BTreeSet::new(),
                datasets: vec![],
                holds: BTreeSet::new(),
            }
        });
        family.source_keys.extend(sources);
        if let Some(old) = family
            .datasets
            .iter()
            .find(|d| d.dataset == id && d.source_key == source_key)
        {
            ensure!(
                old.metadata_sha256 == source.metadata_sha256
                    && old.entity_review_sha256.as_ref() == Some(&review_binding.sha256),
                "cannot change an existing dataset review binding"
            );
        } else {
            family.datasets.push(RegisteredDataset { dataset: id.into(), source_key: source_key.into(), audit_family: source.family.clone(),
                metadata_sha256: source.metadata_sha256.clone(), entity_review_sha256: Some(review_binding.sha256.clone()), preparation_eligible: true,
                eligibility_reason: "documented real original target; exact reconstruction; zero audited split overlap; grouping review required before preparation".into(),
                prepared: vec![], evidence });
        }
    }
    if let Some(parent) = &prior {
        for (name, previous) in &parent.families {
            let new = registry
                .families
                .get(name)
                .context("append removed an existing family")?;
            ensure!(
                new.partition == previous.partition
                    && new.assignment_hash == previous.assignment_hash
                    && new.bucket == previous.bucket,
                "append changed frozen assignment for {name}"
            );
            for previous_dataset in &previous.datasets {
                ensure!(
                    new.datasets
                        .iter()
                        .any(|d| d.dataset == previous_dataset.dataset
                            && d.source_key == previous_dataset.source_key
                            && d.metadata_sha256 == previous_dataset.metadata_sha256
                            && d.entity_review_sha256 == previous_dataset.entity_review_sha256),
                    "append removed or changed the preceding dataset identity"
                );
            }
        }
    }
    registry.validate()?;
    write_json(output, &registry)?;
    File::open(directory)?.sync_all()?;
    Ok(registry)
}
