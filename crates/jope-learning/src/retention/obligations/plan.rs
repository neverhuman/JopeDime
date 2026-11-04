use super::*;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(tag = "state", rename_all = "snake_case", deny_unknown_fields)]
/// `TagAssessment` is an enum for the tag assessment contract in this module: Offline model of exact-version dependencies and retention promises.
pub enum TagAssessment {
    /// The `Verified` case of `TagAssessment`.
    Verified {
        /// Required tags present recorded for this outcome.
        required_tags_present: bool,
    },
    /// The `Unknown` case of `TagAssessment`.
    Unknown {
        /// Explanation accompanying this outcome.
        reason: String,
    },
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
/// `LifecycleAssessment` is a record for the lifecycle assessment contract in this module: Offline model of exact-version dependencies and retention promises.
pub struct LifecycleAssessment {
    /// SHA-256 digest identifying the associated configuration.
    pub configuration_sha256: Option<String>,
    /// Complete recorded for this lifecycle assessment.
    pub complete: bool,
    /// Collection of matching current expiration rules values carried by this lifecycle assessment.
    pub matching_current_expiration_rules: Vec<String>,
    /// Collection of matching noncurrent expiration rules values carried by this lifecycle assessment.
    pub matching_noncurrent_expiration_rules: Vec<String>,
    /// Collection of unsupported fields values carried by this lifecycle assessment.
    pub unsupported_fields: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
/// `Candidate` is a record for the candidate contract in this module: Offline model of exact-version dependencies and retention promises.
pub struct Candidate {
    /// Object recorded for this candidate.
    pub object: StrongPhysicalRef,
    /// Tags recorded for this candidate.
    pub tags: TagAssessment,
    /// Lifecycle recorded for this candidate.
    pub lifecycle: LifecycleAssessment,
    /// Collection of unknowns values carried by this candidate.
    pub unknowns: Vec<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
/// `RetentionObligationDisposition` is an enum for the disposition contract in this module: Offline model of exact-version dependencies and retention promises.
pub enum RetentionObligationDisposition {
    /// The `Keep` case of `RetentionObligationDisposition`.
    Keep,
    /// The `Hold` case of `RetentionObligationDisposition`.
    Hold,
    /// The `ReviewAfterRootTransition` case of `RetentionObligationDisposition`.
    ReviewAfterRootTransition,
}

/// Compatibility alias for the domain-specific `RetentionObligationDisposition` type.
pub type Disposition = RetentionObligationDisposition;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
/// `ObligationRow` is a record for the obligation row contract in this module: Offline model of exact-version dependencies and retention promises.
pub struct ObligationRow {
    /// Object recorded for this obligation row.
    pub object: StrongPhysicalRef,
    /// RetentionObligationDisposition recorded for this obligation row.
    pub disposition: RetentionObligationDisposition,
    /// Collection of strong roots values carried by this obligation row.
    pub strong_roots: Vec<String>,
    /// Collection of promises values carried by this obligation row.
    pub promises: Vec<String>,
    /// Collection of provenance values carried by this obligation row.
    pub provenance: Vec<ProvenanceRef>,
    /// Collection of unresolved values carried by this obligation row.
    pub unresolved: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
/// `ObligationPlanV2` is a record for the obligation plan v2 contract in this module: Offline model of exact-version dependencies and retention promises.
pub struct ObligationPlanV2 {
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// Root generation recorded for this obligation plan v2.
    pub root_generation: String,
    /// Pin generation recorded for this obligation plan v2.
    pub pin_generation: String,
    /// Number of rows represented by this value.
    pub rows: Vec<ObligationRow>,
    /// SHA-256 digest identifying the associated plan.
    pub plan_sha256: String,
}

impl ObligationPlanV2 {
    /// Evaluate supplied exact-version candidates against one pinned authority snapshot.
    pub fn build(snapshot: &PinGeneration, mut candidates: Vec<Candidate>) -> Result<Self> {
        let promise_pins = snapshot.promise_pins()?;
        for (object, owners) in &snapshot.strong_dependencies {
            object.validate()?;
            ensure!(!owners.is_empty(), "strong dependency has no owning root");
        }
        for (object, references) in &snapshot.provenance {
            object.validate()?;
            for reference in references {
                reference.validate()?;
            }
        }
        for (object, retired) in &snapshot.retired_versions {
            object.validate()?;
            ensure!(
                &retired.object == object && object.exact_version(),
                "retired registry key must identify one exact physical version"
            );
            ensure!(
                !retired.retired_at_pin_generation.is_empty(),
                "retired registry entry has no fencing generation"
            );
        }
        candidates.sort_by(|a, b| a.object.cmp(&b.object));
        let mut seen = BTreeSet::new();
        let generation_gaps = snapshot.gaps();
        let mut rows = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            candidate.object.validate()?;
            ensure!(
                seen.insert(candidate.object.clone()),
                "duplicate candidate physical version"
            );
            let mut unresolved = generation_gaps.clone();
            unresolved.extend(
                candidate
                    .unknowns
                    .iter()
                    .map(|u| format!("unknown configuration: {u}")),
            );
            if !candidate.object.exact_version() {
                unresolved.push("physical version is null or unknown".into());
            }
            match &candidate.tags {
                TagAssessment::Verified {
                    required_tags_present: false,
                } => {
                    unresolved.push(
                        "required object tags are missing; tag state cannot establish safety"
                            .into(),
                    );
                }
                TagAssessment::Unknown { reason } => {
                    unresolved.push(format!("tag configuration is unknown: {reason}"));
                }
                TagAssessment::Verified {
                    required_tags_present: true,
                } => {}
            }
            if !candidate.lifecycle.complete {
                unresolved.push("lifecycle configuration is incomplete".into());
            }
            if candidate
                .lifecycle
                .configuration_sha256
                .as_deref()
                .is_none_or(|s| !valid_digest(s))
            {
                unresolved.push("lifecycle configuration identity is unknown".into());
            }
            unresolved.extend(
                candidate
                    .lifecycle
                    .unsupported_fields
                    .iter()
                    .map(|f| format!("unsupported lifecycle field: {f}")),
            );
            unresolved.extend(
                candidate
                    .lifecycle
                    .matching_current_expiration_rules
                    .iter()
                    .map(|r| format!("matching current-version lifecycle expiration: {r}")),
            );
            unresolved.extend(
                candidate
                    .lifecycle
                    .matching_noncurrent_expiration_rules
                    .iter()
                    .map(|r| format!("matching noncurrent-version lifecycle expiration: {r}")),
            );
            let mut strong_roots = snapshot
                .strong_dependencies
                .get(&candidate.object)
                .cloned()
                .unwrap_or_default();
            let promises = promise_pins
                .get(&candidate.object)
                .cloned()
                .unwrap_or_default();
            for promise in snapshot
                .promises
                .iter()
                .filter(|p| promises.contains(&p.id))
            {
                strong_roots.insert(promise.root_id.clone());
            }
            let retired = snapshot.retired_versions.contains_key(&candidate.object);
            if retired {
                unresolved.push("candidate is already retired in the pinned authority".into());
            }
            unresolved.sort();
            unresolved.dedup();
            let disposition = if !unresolved.is_empty() {
                RetentionObligationDisposition::Hold
            } else if !strong_roots.is_empty() || !promises.is_empty() {
                RetentionObligationDisposition::Keep
            } else {
                RetentionObligationDisposition::ReviewAfterRootTransition
            };
            rows.push(ObligationRow {
                provenance: snapshot
                    .provenance
                    .get(&candidate.object)
                    .cloned()
                    .unwrap_or_default()
                    .into_iter()
                    .collect(),
                object: candidate.object,
                disposition,
                strong_roots: strong_roots.into_iter().collect(),
                promises: promises.into_iter().collect(),
                unresolved,
            });
        }
        let mut plan = Self {
            schema: OBLIGATION_PLAN_SCHEMA.into(),
            root_generation: snapshot.root_generation.clone(),
            pin_generation: snapshot.pin_generation.clone(),
            rows,
            plan_sha256: String::new(),
        };
        plan.plan_sha256 = canonical_sha256(&plan)?;
        Ok(plan)
    }
}
