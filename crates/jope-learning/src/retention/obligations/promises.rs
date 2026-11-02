use super::*;

/// An active promise protects the complete declared dependency set until a tested migration or
/// an explicit generation-bound release is present.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RecoveryPromise {
    /// Stable identifier for this value.
    pub id: String,
    /// Record kind or category represented by this value.
    pub kind: PromiseKind,
    /// Identifier for the associated root.
    pub root_id: String,
    /// Active recorded for this recovery promise.
    pub active: bool,
    /// Collection of dependencies values carried by this recovery promise.
    pub dependencies: Vec<StrongPhysicalRef>,
    /// Collection of restore proofs values carried by this recovery promise.
    pub restore_proofs: Vec<RuntimeRestoreProof>,
    /// Optional released by transition value; `None` means it is not available.
    pub released_by_transition: Option<String>,
    /// Optional released at generation value; `None` means it is not available.
    pub released_at_generation: Option<String>,
}

impl RecoveryPromise {
    fn validate(&self) -> Result<()> {
        ensure!(
            !self.id.is_empty() && !self.root_id.is_empty(),
            "promise identity is missing"
        );
        ensure!(
            !self.dependencies.is_empty(),
            "promise dependency closure is empty"
        );
        for dep in &self.dependencies {
            dep.validate()?;
        }
        for proof in &self.restore_proofs {
            proof.source.validate()?;
            proof.replacement.validate()?;
            ensure!(
                valid_digest(&proof.receipt_sha256),
                "invalid restore receipt digest"
            );
        }
        if !self.active {
            ensure!(
                self.released_by_transition
                    .as_deref()
                    .is_some_and(|s| !s.is_empty())
                    && self
                        .released_at_generation
                        .as_deref()
                        .is_some_and(|s| !s.is_empty()),
                "inactive promise requires a recorded root transition"
            );
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
/// `RetiredVersion` is a record for the retired version contract in this module: Offline model of exact-version dependencies and retention promises.
pub struct RetiredVersion {
    /// Object recorded for this retired version.
    pub object: StrongPhysicalRef,
    /// Retired at pin generation recorded for this retired version.
    pub retired_at_pin_generation: String,
}

/// Atomic view of root and pin authorities used to fence stale publishers.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PinGeneration {
    /// Root generation recorded for this pin generation.
    pub root_generation: String,
    /// Pin generation recorded for this pin generation.
    pub pin_generation: String,
    /// Observed root generation recorded for this pin generation.
    pub observed_root_generation: String,
    /// Observed pin generation recorded for this pin generation.
    pub observed_pin_generation: String,
    /// Closure complete recorded for this pin generation.
    pub closure_complete: bool,
    /// Registered publishers recorded for this pin generation.
    pub registered_publishers: BTreeSet<String>,
    /// Observed publishers recorded for this pin generation.
    pub observed_publishers: BTreeSet<String>,
    /// Collection of unresolved values carried by this pin generation.
    pub unresolved: Vec<String>,
    /// Mapping of strong dependencies values carried by this pin generation.
    pub strong_dependencies: BTreeMap<StrongPhysicalRef, BTreeSet<String>>,
    /// Mapping of provenance values carried by this pin generation.
    pub provenance: BTreeMap<StrongPhysicalRef, BTreeSet<ProvenanceRef>>,
    /// Collection of promises values carried by this pin generation.
    pub promises: Vec<RecoveryPromise>,
    /// Mapping of retired versions values carried by this pin generation.
    pub retired_versions: BTreeMap<StrongPhysicalRef, RetiredVersion>,
}

impl PinGeneration {
    fn gaps(&self) -> Vec<String> {
        let mut gaps = self.unresolved.clone();
        if self.root_generation.is_empty() || self.pin_generation.is_empty() {
            gaps.push("root or pin generation identity is unknown".into());
        }
        if self.observed_root_generation != self.root_generation
            || self.observed_pin_generation != self.pin_generation
        {
            gaps.push("observed root/pin generation changed".into());
        }
        if !self.closure_complete {
            gaps.push("authoritative root closure is incomplete".into());
        }
        for publisher in self
            .registered_publishers
            .difference(&self.observed_publishers)
        {
            gaps.push(format!(
                "registered publisher is not quiesced or observed: {publisher}"
            ));
        }
        gaps.sort();
        gaps.dedup();
        gaps
    }

    /// Performs the validate publish operation and returns an error if it cannot be completed.
    pub fn validate_publish(
        &self,
        publisher_id: &str,
        publisher_pin_generation: &str,
        refs: &[StrongPhysicalRef],
    ) -> Result<()> {
        ensure!(
            self.gaps().is_empty(),
            "cannot validate publisher against an incomplete pin generation"
        );
        ensure!(
            publisher_pin_generation == self.pin_generation,
            "stale publisher pin generation"
        );
        ensure!(
            self.registered_publishers.contains(publisher_id)
                && self.observed_publishers.contains(publisher_id),
            "publisher is not registered and observed in this pin generation"
        );
        for reference in refs {
            reference.validate()?;
            ensure!(
                !self.retired_versions.contains_key(reference),
                "publisher cannot restore a retired physical version"
            );
        }
        Ok(())
    }

    fn promise_pins(&self) -> Result<BTreeMap<StrongPhysicalRef, BTreeSet<String>>> {
        let mut pinned = BTreeMap::<StrongPhysicalRef, BTreeSet<String>>::new();
        for promise in &self.promises {
            promise.validate()?;
            if !promise.active {
                ensure!(
                    promise.released_at_generation.as_deref()
                        == Some(self.root_generation.as_str()),
                    "released promise does not name the current root generation"
                );
                continue;
            }
            let mut replaced = BTreeSet::new();
            for dependency in &promise.dependencies {
                if let Some(proof) = promise
                    .restore_proofs
                    .iter()
                    .find(|p| p.validates_for(promise.kind, dependency, &self.pin_generation))
                {
                    replaced.insert(dependency.clone());
                    pinned
                        .entry(proof.replacement.clone())
                        .or_default()
                        .insert(promise.id.clone());
                }
            }
            for dependency in promise
                .dependencies
                .iter()
                .filter(|dep| !replaced.contains(*dep))
            {
                pinned
                    .entry(dependency.clone())
                    .or_default()
                    .insert(promise.id.clone());
            }
        }
        Ok(pinned)
    }
}
