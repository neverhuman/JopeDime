use super::*;

/// Isolated namespace used only to exercise recovery before an approved mock run.
#[derive(Clone, Debug, Default)]
pub struct MockRecoveryNamespace {
    pub namespace_id: String,
    objects: BTreeSet<StrongPhysicalRef>,
}

impl MockRecoveryNamespace {
    pub fn with_objects(
        namespace_id: impl Into<String>,
        objects: impl IntoIterator<Item = StrongPhysicalRef>,
    ) -> Self {
        Self {
            namespace_id: namespace_id.into(),
            objects: objects.into_iter().collect(),
        }
    }

    pub fn contains(&self, object: &StrongPhysicalRef) -> bool {
        self.objects.contains(object)
    }

    /// Exercise readback into an isolated namespace while the original exact version is absent.
    pub fn restore_exact(
        &mut self,
        original: &StrongPhysicalRef,
        replacement: &StrongPhysicalRef,
        plan_target_set_sha256: &str,
    ) -> Result<IndependentRestoreProof> {
        ensure!(
            !self.objects.contains(original),
            "mock recovery namespace unexpectedly contains the original"
        );
        ensure!(
            self.objects.contains(replacement),
            "mock recovery replacement is unavailable"
        );
        ensure!(
            original.sha256 == replacement.sha256 && original.bytes == replacement.bytes,
            "mock recovery replacement bytes do not match the requested source"
        );
        self.objects.insert(original.clone());
        let receipt_sha256 = canonical_sha256(&(
            &self.namespace_id,
            original,
            replacement,
            plan_target_set_sha256,
        ))?;
        Ok(IndependentRestoreProof {
            plan_target_set_sha256: plan_target_set_sha256.into(),
            namespace_id: self.namespace_id.clone(),
            original_absent_during_restore: true,
            replacement_readback_verified: self.objects.contains(original),
            runtime_restore_tested: true,
            receipt_sha256,
        })
    }
}

#[derive(Clone, Debug, Default)]
pub struct MockNamespace {
    objects: BTreeSet<StrongPhysicalRef>,
    pub(super) requests: BTreeMap<String, StrongPhysicalRef>,
    pub exact_request_count: u64,
    pub simulated_effect_count: u64,
}

impl MockNamespace {
    pub fn with_objects(objects: impl IntoIterator<Item = StrongPhysicalRef>) -> Self {
        Self {
            objects: objects.into_iter().collect(),
            ..Self::default()
        }
    }

    pub fn contains(&self, object: &StrongPhysicalRef) -> bool {
        self.objects.contains(object)
    }

    pub fn request_count_for(&self, request_id: &str) -> u64 {
        u64::from(self.requests.contains_key(request_id))
    }

    pub(super) fn request_exact(
        &mut self,
        request_id: &str,
        target: &StrongPhysicalRef,
    ) -> Result<()> {
        if let Some(recorded) = self.requests.get(request_id) {
            ensure!(
                recorded == target,
                "request identity was reused for another exact version"
            );
            return Ok(());
        }
        ensure!(
            self.objects.remove(target),
            "exact version is already absent from mock namespace"
        );
        self.requests.insert(request_id.into(), target.clone());
        self.exact_request_count += 1;
        self.simulated_effect_count += 1;
        Ok(())
    }
}
