use super::*;

/// The growth receipt, recomputed from its own recorded inputs by the compiled release
/// (`crate::family_growth`, the native port of the retired `family_growth_policy.py`). No
/// helper in the workspace is consulted, so a replaced or forged one cannot authorize a
/// component. The receipt's bytes are hashed and parsed once.
pub(crate) fn verified_growth(binding: &FileBinding) -> Result<Value> {
    let bytes = fs::read(&binding.path)?;
    ensure!(hash(&bytes) == binding.sha256, "growth receipt changed");
    let request: Value = serde_json::from_slice(&bytes)?;
    let verified = crate::family_growth::verify_assignment(&request)
        .map_err(|e| anyhow::anyhow!("original growth evidence verification failed: {e}"))?;
    ensure!(verified == request, "growth verifier result changed");
    Ok(verified)
}

impl Registry {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == SCHEMA && self.seed == SEED,
            "unsupported family registry"
        );
        ensure!(self.qualification && !self.full_inventory_complete && !self.independent_selection_eligible,
            "this registry cannot promote qualification data or checkpoints to independent evaluation");
        let mut grown = BTreeMap::new();
        for binding in &self.growth {
            let receipt = verified_growth(binding)?;
            let prior: FileBinding = serde_json::from_value(receipt["parent_registry"].clone())?;
            let root = Path::new(field(&receipt, "root")?);
            let path = if prior.path.is_absolute() {
                prior.path.clone()
            } else {
                root.join(&prior.path)
            };
            ensure!(file_hash(&path)? == prior.sha256, "growth parent changed");
            let parent: Registry = read_json(&path)?;
            for (name, previous) in &parent.families {
                let new = self
                    .families
                    .get(name)
                    .context("growth removed historical family")?;
                ensure!(
                    new.partition == previous.partition
                        && new.assignment_method == previous.assignment_method
                        && new.assignment_hash == previous.assignment_hash
                        && new.bucket == previous.bucket
                        && previous.source_keys.is_subset(&new.source_keys)
                        && previous.holds.is_subset(&new.holds),
                    "growth changed protected historical membership"
                );
                for dataset in &previous.datasets {
                    let previous_dataset_json = serde_json::to_value(dataset)?;
                    ensure!(
                        new.datasets
                            .iter()
                            .any(|d| serde_json::to_value(d).ok().as_ref()
                                == Some(&previous_dataset_json)),
                        "growth changed historical dataset evidence"
                    );
                }
            }
            for row in receipt["rows"].as_array().context("growth rows missing")? {
                let family = field(row, "family")?.to_owned();
                ensure!(
                    grown.insert(family, row.clone()).is_none(),
                    "duplicate component growth assignment"
                );
            }
        }
        let mut datasets = BTreeSet::new();
        let mut source_owners: BTreeMap<&str, &FamilyAssignment> = BTreeMap::new();
        for (key, f) in &self.families {
            ensure!(key == &f.family && !key.is_empty(), "family key mismatch");
            ensure!(
                [
                    "meta_train",
                    "meta_validation",
                    "meta_test",
                    "development_validation"
                ]
                .contains(&f.partition.as_str()),
                "invalid family partition"
            );
            if f.assignment_method == "stable_sha256_bucket_v1" {
                let (digest, bucket, partition) = stable_assignment(key);
                ensure!(
                    f.assignment_hash.as_ref() == Some(&digest)
                        && f.bucket == Some(bucket)
                        && f.partition == partition,
                    "changed stable family assignment"
                );
            } else if f.assignment_method == "stable_sha256_component_v2" {
                ensure!(
                    f.datasets.is_empty(),
                    "growth membership has not qualified preparation"
                );
                let row = grown
                    .remove(key)
                    .context("component lacks authenticated growth evidence")?;
                let (digest, bucket, partition) = stable_component_assignment(key);
                let aliases: BTreeSet<String> = serde_json::from_value(row["aliases"].clone())?;
                ensure!(
                    f.assignment_hash.as_ref() == Some(&digest)
                        && f.bucket == Some(bucket)
                        && f.partition == partition
                        && row["partition"] == partition
                        && row["assignment_hash"] == digest
                        && row["bucket"] == bucket
                        && aliases == f.source_keys,
                    "changed component assignment or aliases"
                );
            } else {
                ensure!(
                    f.assignment_method == "preserved_audit_assignment"
                        && f.partition != "development_validation",
                    "unknown assignment method"
                );
            }
            for d in &f.datasets {
                ensure!(
                    f.source_keys.contains(&d.source_key),
                    "dataset source missing from family"
                );
                ensure!(
                    datasets.insert((&d.dataset, &d.source_key)),
                    "duplicate registered dataset identity"
                );
                if let Some(review) = &d.entity_review_sha256 {
                    ensure!(
                        self.reviews.iter().any(|r| &r.sha256 == review),
                        "dataset review is not registry-bound"
                    );
                }
            }
            for source in &f.source_keys {
                if let Some(previous) = source_owners.insert(source, f) {
                    ensure!(
                        !f.holds.is_empty() && !previous.holds.is_empty(),
                        "alias spans unquarantined families"
                    );
                }
            }
        }
        ensure!(grown.is_empty(), "authenticated growth membership removed");
        Ok(())
    }

    pub fn assignment(
        &self,
        dataset: &str,
        source_key: &str,
        audit_family: &str,
    ) -> Result<&FamilyAssignment> {
        self.validate()?;
        let found = self
            .families
            .values()
            .find(|f| {
                f.datasets
                    .iter()
                    .any(|d| d.dataset == dataset && d.source_key == source_key)
            })
            .context("dataset/source identity is not admitted by the registry")?;
        ensure!(
            found.holds.is_empty(),
            "family is quarantined: {:?}",
            found.holds
        );
        let d = found
            .datasets
            .iter()
            .find(|d| d.dataset == dataset && d.source_key == source_key)
            .unwrap();
        ensure!(
            audit_family == d.audit_family || audit_family == found.family,
            "audit family identity changed"
        );
        Ok(found)
    }

    pub fn quota_report(&self) -> Value {
        let mut all: BTreeMap<&str, usize> = BTreeMap::new();
        let mut eligible: BTreeMap<&str, usize> = BTreeMap::new();
        let mut prepared: BTreeMap<&str, usize> = BTreeMap::new();
        for f in self.families.values() {
            *all.entry(&f.partition).or_default() += 1;
            if f.holds.is_empty() && f.datasets.iter().any(|d| d.preparation_eligible) {
                *eligible.entry(&f.partition).or_default() += 1;
            }
            if f.holds.is_empty()
                && f.datasets
                    .iter()
                    .any(|d| d.preparation_eligible && !d.prepared.is_empty())
            {
                *prepared.entry(&f.partition).or_default() += 1;
            }
        }
        let required = BTreeMap::from([
            ("meta_train", 48_usize),
            ("meta_validation", 8),
            ("meta_test", 8),
        ]);
        let missing: BTreeMap<_, _> = required
            .iter()
            .map(|(p, n)| (*p, n.saturating_sub(*eligible.get(p).unwrap_or(&0))))
            .collect();
        json!({"qualification":true,"independent_selection_eligible":false,"full_inventory_complete":false,
            "all_registered_families":all,"families_with_verified_preparation_candidates":eligible,
            "families_with_existing_reconstructed_preparations":prepared,"required_pilot_families":required,
            "remaining_families_by_partition":missing,"quota_ready":missing.values().all(|n| *n == 0),
            "held_families":self.families.values().filter(|f| !f.holds.is_empty()).count(),
            "held_review_datasets":self.held_review_datasets.len(),
            "eligibility_limit":"Preparation candidates only; original final tests, full source inventory and fresh checkpoint eligibility remain unqualified."})
    }
}
