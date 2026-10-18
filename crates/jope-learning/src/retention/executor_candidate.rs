//! Disabled, in-memory model for exact-version retirement review.
//!
//! This candidate has no adapter for real storage. Every effect below mutates only a mock set.

use crate::canon::canonical_sha256;
use crate::retention::obligations::{StrongPhysicalRef, VersionId};
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const EXECUTOR_PLAN_SCHEMA: &str = "jope.retirement-candidate/v1";

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AuthoritySnapshot {
    pub root_generation: String,
    pub pin_generation: String,
    pub lifecycle_generation: String,
    pub closure_complete: bool,
    pub registered_publishers: BTreeSet<String>,
    pub observed_publishers: BTreeSet<String>,
    pub active_pins: BTreeSet<StrongPhysicalRef>,
    pub retired: BTreeSet<StrongPhysicalRef>,
}

impl AuthoritySnapshot {
    fn validate(&self) -> Result<()> {
        ensure!(
            !self.root_generation.is_empty()
                && !self.pin_generation.is_empty()
                && !self.lifecycle_generation.is_empty(),
            "authority generations must be present"
        );
        ensure!(self.closure_complete, "authority closure is incomplete");
        ensure!(
            self.registered_publishers == self.observed_publishers,
            "publisher registry is not fully observed"
        );
        for reference in self.active_pins.iter().chain(self.retired.iter()) {
            reference.validate()?;
        }
        Ok(())
    }

    fn generation_matches(&self, root: &str, pin: &str, lifecycle: &str) -> bool {
        self.root_generation == root
            && self.pin_generation == pin
            && self.lifecycle_generation == lifecycle
            && self.validate().is_ok()
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RetirementPlan {
    pub schema: String,
    pub planner: String,
    pub root_generation: String,
    pub pin_generation: String,
    pub lifecycle_generation: String,
    pub exact_targets: Vec<StrongPhysicalRef>,
    pub target_set_sha256: String,
    pub plan_sha256: String,
}

impl RetirementPlan {
    pub fn propose(
        authority: &AuthoritySnapshot,
        planner: &str,
        mut exact_targets: Vec<StrongPhysicalRef>,
    ) -> Result<Self> {
        authority.validate()?;
        ensure!(!planner.is_empty(), "planner identity is missing");
        ensure!(!exact_targets.is_empty(), "retirement target set is empty");
        exact_targets.sort();
        let mut unique = BTreeSet::new();
        for target in &exact_targets {
            target.validate()?;
            ensure!(
                matches!(target.version_id, VersionId::Explicit(_)),
                "candidate requires one explicit version"
            );
            ensure!(
                !authority.active_pins.contains(target),
                "candidate intersects an active pin"
            );
            ensure!(
                !authority.retired.contains(target),
                "candidate is already retired"
            );
            ensure!(unique.insert(target.clone()), "duplicate exact target");
        }
        let target_set_sha256 = canonical_sha256(&exact_targets)?;
        let mut plan = Self {
            schema: EXECUTOR_PLAN_SCHEMA.into(),
            planner: planner.into(),
            root_generation: authority.root_generation.clone(),
            pin_generation: authority.pin_generation.clone(),
            lifecycle_generation: authority.lifecycle_generation.clone(),
            exact_targets,
            target_set_sha256,
            plan_sha256: String::new(),
        };
        plan.plan_sha256 = canonical_sha256(&plan)?;
        Ok(plan)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct IndependentRestoreProof {
    pub plan_target_set_sha256: String,
    pub namespace_id: String,
    pub original_absent_during_restore: bool,
    pub replacement_readback_verified: bool,
    pub runtime_restore_tested: bool,
    pub receipt_sha256: String,
}

impl IndependentRestoreProof {
    fn validate_for(&self, plan: &RetirementPlan) -> Result<()> {
        ensure!(
            self.plan_target_set_sha256 == plan.target_set_sha256,
            "restore proof covers a different target set"
        );
        ensure!(
            !self.namespace_id.is_empty(),
            "independent restore namespace is missing"
        );
        ensure!(
            self.original_absent_during_restore,
            "restore proof still has access to the original target"
        );
        ensure!(
            self.replacement_readback_verified && self.runtime_restore_tested,
            "independent runtime restoration is incomplete"
        );
        ensure!(
            self.receipt_sha256.len() == 64
                && self.receipt_sha256.bytes().all(|b| b.is_ascii_hexdigit()),
            "restore receipt digest is invalid"
        );
        Ok(())
    }
}

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

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct OwnerApproval {
    pub signed_plan_sha256: String,
    pub signer: String,
    /// Set only after a separate verifier has checked the signature against the owner key.
    pub signature_verified: bool,
    pub independent_reviewer: bool,
    pub restore: IndependentRestoreProof,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RetirementReservation {
    pub reservation_id: String,
    pub plan_sha256: String,
    pub root_generation: String,
    pub pin_generation: String,
    pub lifecycle_generation: String,
    pub exact_targets: Vec<StrongPhysicalRef>,
    pub approval_signer: String,
}

impl RetirementReservation {
    pub fn reserve(
        plan: &RetirementPlan,
        authority: &AuthoritySnapshot,
        approval: &OwnerApproval,
    ) -> Result<Self> {
        ensure!(
            authority.generation_matches(
                &plan.root_generation,
                &plan.pin_generation,
                &plan.lifecycle_generation
            ),
            "authority changed since planning"
        );
        ensure!(
            approval.signed_plan_sha256 == plan.plan_sha256 && approval.signature_verified,
            "approval does not authenticate this exact plan"
        );
        ensure!(
            approval.independent_reviewer && approval.signer != plan.planner,
            "approval is not independent of the planner"
        );
        approval.restore.validate_for(plan)?;
        for target in &plan.exact_targets {
            ensure!(
                !authority.active_pins.contains(target),
                "a current publisher pin conflicts with the reservation"
            );
            ensure!(
                !authority.retired.contains(target),
                "target is already retired"
            );
        }
        let reservation_id = canonical_sha256(&(
            &plan.plan_sha256,
            &authority.root_generation,
            &authority.pin_generation,
            &authority.lifecycle_generation,
        ))?;
        Ok(Self {
            reservation_id,
            plan_sha256: plan.plan_sha256.clone(),
            root_generation: plan.root_generation.clone(),
            pin_generation: plan.pin_generation.clone(),
            lifecycle_generation: plan.lifecycle_generation.clone(),
            exact_targets: plan.exact_targets.clone(),
            approval_signer: approval.signer.clone(),
        })
    }

    fn validate_current(&self, authority: &AuthoritySnapshot) -> Result<()> {
        ensure!(
            authority.generation_matches(
                &self.root_generation,
                &self.pin_generation,
                &self.lifecycle_generation
            ),
            "authority changed after reservation"
        );
        for target in &self.exact_targets {
            ensure!(
                !authority.active_pins.contains(target),
                "publisher added a pin after reservation"
            );
            ensure!(
                !authority.retired.contains(target),
                "target retired through another reservation"
            );
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TombstoneState {
    Prepared,
    Acknowledged,
    ReconciledAbsent,
    AbortedStale,
    AbsentBeforeRequest,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Tombstone {
    pub schema: String,
    pub plan_sha256: String,
    pub reservation_id: String,
    pub request_id: String,
    pub exact_target: StrongPhysicalRef,
    pub state: TombstoneState,
    pub detail: String,
    pub tombstone_sha256: String,
}

impl Tombstone {
    fn new(reservation: &RetirementReservation, target: &StrongPhysicalRef) -> Result<Self> {
        let request_id = canonical_sha256(&(
            &reservation.plan_sha256,
            &reservation.reservation_id,
            target,
        ))?;
        let mut tombstone = Self {
            schema: "jope.retirement-tombstone/v1".into(),
            plan_sha256: reservation.plan_sha256.clone(),
            reservation_id: reservation.reservation_id.clone(),
            request_id,
            exact_target: target.clone(),
            state: TombstoneState::Prepared,
            detail: "durable intent represented in the mock journal".into(),
            tombstone_sha256: String::new(),
        };
        tombstone.tombstone_sha256 = canonical_sha256(&tombstone)?;
        Ok(tombstone)
    }

    fn reseal(&mut self) -> Result<()> {
        self.tombstone_sha256.clear();
        self.tombstone_sha256 = canonical_sha256(self)?;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SimulatedCrash {
    None,
    BeforeExactRequest,
    AfterExactRequestBeforeAcknowledgement,
}

#[derive(Clone, Debug, Default)]
pub struct MockNamespace {
    objects: BTreeSet<StrongPhysicalRef>,
    requests: BTreeMap<String, StrongPhysicalRef>,
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

    fn request_exact(&mut self, request_id: &str, target: &StrongPhysicalRef) -> Result<()> {
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

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MockExecutionReceipt {
    pub tombstones: Vec<Tombstone>,
    pub crashed: bool,
    pub live_removal_count: u64,
    pub simulated_effect_count: u64,
}

#[derive(Clone, Debug, Default)]
pub struct ExactVersionExecutor {
    enabled: bool,
    journal: BTreeMap<String, Tombstone>,
}

impl ExactVersionExecutor {
    pub fn disabled() -> Self {
        Self::default()
    }

    /// Constructs the mock-only state machine used by bounded fixture tests.
    pub fn mock_candidate() -> Self {
        Self {
            enabled: true,
            journal: BTreeMap::new(),
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn tombstones(&self) -> Vec<Tombstone> {
        self.journal.values().cloned().collect()
    }

    pub fn execute_mock(
        &mut self,
        reservation: &RetirementReservation,
        authority: &AuthoritySnapshot,
        namespace: &mut MockNamespace,
        crash: SimulatedCrash,
    ) -> Result<MockExecutionReceipt> {
        ensure!(self.enabled, "executor candidate is disabled");
        reservation.validate_current(authority)?;
        let mut crashed = false;
        for (index, target) in reservation.exact_targets.iter().enumerate() {
            reservation.validate_current(authority)?;
            let mut tombstone = Tombstone::new(reservation, target)?;
            if let Some(existing) = self.journal.get(&tombstone.request_id) {
                ensure!(
                    existing.exact_target == *target,
                    "tombstone request identity collision"
                );
                continue;
            }
            self.journal
                .insert(tombstone.request_id.clone(), tombstone.clone());
            if crash == SimulatedCrash::BeforeExactRequest && index == 0 {
                crashed = true;
                break;
            }
            namespace.request_exact(&tombstone.request_id, target)?;
            if crash == SimulatedCrash::AfterExactRequestBeforeAcknowledgement && index == 0 {
                crashed = true;
                break;
            }
            tombstone.state = TombstoneState::Acknowledged;
            tombstone.detail = "mock exact-version request acknowledged".into();
            tombstone.reseal()?;
            self.journal.insert(tombstone.request_id.clone(), tombstone);
        }
        Ok(self.receipt(namespace, crashed))
    }

    /// Reconcile only by the same exact request identity and physical version.
    pub fn reconcile_mock(
        &mut self,
        reservation: &RetirementReservation,
        authority: &AuthoritySnapshot,
        namespace: &mut MockNamespace,
    ) -> Result<MockExecutionReceipt> {
        ensure!(self.enabled, "executor candidate is disabled");
        for target in &reservation.exact_targets {
            let intent = Tombstone::new(reservation, target)?;
            let Some(mut tombstone) = self.journal.get(&intent.request_id).cloned() else {
                continue;
            };
            if namespace.requests.contains_key(&tombstone.request_id) {
                ensure!(
                    namespace.requests.get(&tombstone.request_id) == Some(target),
                    "mock request record names a different exact version"
                );
                if !namespace.contains(target) {
                    tombstone.state = TombstoneState::ReconciledAbsent;
                    tombstone.detail =
                        "exact version absent after recorded request; no retry issued".into();
                }
            } else if !namespace.contains(target) {
                tombstone.state = TombstoneState::AbsentBeforeRequest;
                tombstone.detail =
                    "exact version absent without a recorded request; no retry issued".into();
            } else {
                reservation.validate_current(authority)?;
                namespace.request_exact(&tombstone.request_id, target)?;
                tombstone.state = TombstoneState::Acknowledged;
                tombstone.detail = "same exact request reconciled in mock namespace".into();
            }
            tombstone.reseal()?;
            self.journal.insert(tombstone.request_id.clone(), tombstone);
        }
        Ok(self.receipt(namespace, false))
    }

    fn receipt(&self, namespace: &MockNamespace, crashed: bool) -> MockExecutionReceipt {
        MockExecutionReceipt {
            tombstones: self.tombstones(),
            crashed,
            live_removal_count: 0,
            simulated_effect_count: namespace.simulated_effect_count,
        }
    }
}
