use super::*;

/// A complete evaluator identity. Every field participates in equality and table joins.
#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MeasurementScope {
    /// SHA-256 digest identifying the associated context.
    pub context_sha256: String,
    /// SHA-256 digest of the checkpoint bound to this evidence.
    pub checkpoint_sha256: String,
    /// SHA-256 digest identifying the associated policy.
    pub policy_sha256: String,
    /// SHA-256 digest identifying the evaluator used for this evidence.
    pub evaluator_sha256: String,
    /// SHA-256 digest identifying the associated target.
    pub target_sha256: String,
    /// SHA-256 digest identifying the associated profile.
    pub profile_sha256: String,
    /// Fidelity recorded for this measurement scope.
    pub fidelity: String,
    /// Data or authority role assigned to this value.
    pub role: String,
    /// Metric name recorded for this measurement scope.
    pub metric_name: String,
}

impl MeasurementScope {
    fn validate(&self) -> Result<()> {
        for (name, value) in [
            ("context", &self.context_sha256),
            ("checkpoint", &self.checkpoint_sha256),
            ("policy", &self.policy_sha256),
            ("evaluator", &self.evaluator_sha256),
            ("target", &self.target_sha256),
            ("profile", &self.profile_sha256),
        ] {
            ensure!(
                is_hex64(value),
                "{name} identity must be a lowercase sha256"
            );
        }
        ensure!(!self.fidelity.trim().is_empty(), "fidelity is required");
        ensure!(!self.role.trim().is_empty(), "measurement role is required");
        ensure!(
            self.metric_name == METRIC_NAME,
            "unsupported metric identity"
        );
        Ok(())
    }

    fn same_subject(&self, other: &Self) -> bool {
        self.context_sha256 == other.context_sha256
            && self.checkpoint_sha256 == other.checkpoint_sha256
            && self.policy_sha256 == other.policy_sha256
    }

    fn mismatches(&self, other: &Self) -> Vec<String> {
        let mut out = Vec::new();
        macro_rules! mismatch {
            ($field:ident) => {
                if self.$field != other.$field {
                    out.push(stringify!($field).to_owned());
                }
            };
        }
        mismatch!(context_sha256);
        mismatch!(checkpoint_sha256);
        mismatch!(policy_sha256);
        mismatch!(evaluator_sha256);
        mismatch!(target_sha256);
        mismatch!(profile_sha256);
        mismatch!(fidelity);
        mismatch!(role);
        mismatch!(metric_name);
        out
    }
}

/// One exact requested cell. Replicates are independent only when the source contract gives
/// them distinct stable IDs; `None` is a single unreplicated key, never an invitation to average.
#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(deny_unknown_fields)]
pub struct OutcomeKey {
    /// Scope label describing the population covered by this result.
    pub scope: MeasurementScope,
    /// SHA-256 digest of the canonical program source.
    pub program_sha256: String,
    /// Identifier for the associated replicate.
    pub replicate_id: Option<String>,
}

impl OutcomeKey {
    fn validate(&self) -> Result<()> {
        self.scope.validate()?;
        ensure!(
            is_hex64(&self.program_sha256),
            "program identity must be a lowercase sha256"
        );
        if let Some(id) = &self.replicate_id {
            ensure!(!id.trim().is_empty(), "replicate id cannot be empty");
        }
        Ok(())
    }

    fn same_subject(&self, other: &Self) -> bool {
        self.scope.same_subject(&other.scope) && self.program_sha256 == other.program_sha256
    }
}

/// Statuses remain distinct so failure, censoring and unknown evidence retain their denominator.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OutcomeStatus {
    /// The `Complete` case of `OutcomeStatus`.
    Complete,
    /// The `Failed` case of `OutcomeStatus`.
    Failed,
    /// The `Timeout` case of `OutcomeStatus`.
    Timeout,
    /// The `Rejected` case of `OutcomeStatus`.
    Rejected,
    /// The `Unknown` case of `OutcomeStatus`.
    Unknown,
}

/// An immutable source locator and its declared outcome. Complete values use exact I02 f64 bits.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ObservedOutcome {
    /// Key recorded for this observed outcome.
    pub key: OutcomeKey,
    /// Outcome or lifecycle status recorded for this value.
    pub status: OutcomeStatus,
    /// Optional metric f64 bits value; `None` means it is not available.
    pub metric_f64_bits: Option<String>,
    /// SHA-256 digest identifying the associated evidence receipt.
    pub evidence_receipt_sha256: String,
    /// SHA-256 digest identifying the associated source locator.
    pub source_locator_sha256: String,
}

impl ObservedOutcome {
    fn validate(&self) -> Result<()> {
        self.key.validate()?;
        ensure!(
            is_hex64(&self.evidence_receipt_sha256)
                && self.evidence_receipt_sha256.bytes().any(|b| b != b'0'),
            "nonzero evidence receipt sha256 required"
        );
        ensure!(
            is_hex64(&self.source_locator_sha256),
            "source locator must be a lowercase sha256"
        );
        match (self.status, self.metric_f64_bits.as_deref()) {
            (OutcomeStatus::Complete, Some(bits)) => {
                let exact =
                    jope_records::wire_contract::ExactF64::from_hex("/metric_f64_bits", bits)?;
                ensure!(exact.to_hex() == bits, "metric bits must be canonical");
            }
            (OutcomeStatus::Complete, None) => anyhow::bail!("complete outcome has no metric"),
            (_, None) => (),
            (_, Some(_)) => anyhow::bail!("unavailable outcome cannot carry a metric"),
        }
        Ok(())
    }
}

/// Source view pinned by I27/I28. Its sorted locator inventory binds every observation admitted
/// by this build; the verifier must authenticate the view and prove that none of its inputs were
/// revoked. This local type is a draft adapter until the upstream accepted view DTO is integrated.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ImmutableSourceView {
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// Source commit recorded for this immutable source view.
    pub source_commit: String,
    /// SHA-256 digest identifying the associated source receipt.
    pub source_receipt_sha256: String,
    /// SHA-256 digest identifying the associated revocation snapshot.
    pub revocation_snapshot_sha256: String,
    /// SHA-256 digest identifying the associated observation locators.
    pub observation_locators_sha256: Vec<String>,
    /// SHA-256 digest identifying the associated view.
    pub view_sha256: String,
}

impl ImmutableSourceView {
    /// Performs the seal operation and returns an error if it cannot be completed.
    pub fn seal(&mut self) -> Result<()> {
        self.view_sha256.clear();
        self.view_sha256 = canonical_sha256(self)?;
        Ok(())
    }

    fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == SOURCE_VIEW_SCHEMA,
            "unsupported source view schema"
        );
        ensure!(
            (self.source_commit.len() == 40 || self.source_commit.len() == 64)
                && self
                    .source_commit
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
            "source commit must be a lowercase Git object id"
        );
        for (name, value) in [
            ("source receipt", &self.source_receipt_sha256),
            ("revocation snapshot", &self.revocation_snapshot_sha256),
            ("view", &self.view_sha256),
        ] {
            ensure!(is_hex64(value), "{name} must be a lowercase sha256");
        }
        ensure!(
            self.observation_locators_sha256
                .windows(2)
                .all(|pair| pair[0] < pair[1])
                && self
                    .observation_locators_sha256
                    .iter()
                    .all(|id| is_hex64(id)),
            "source locators must be unique, sorted sha256 identities"
        );
        let mut unsealed = self.clone();
        let expected = unsealed.view_sha256.clone();
        unsealed.view_sha256.clear();
        ensure!(
            canonical_sha256(&unsealed)? == expected,
            "source view digest mismatch"
        );
        Ok(())
    }
}

/// Trusted adapter boundary. An accepted I27/I28 implementation must verify the actual immutable
/// receipts, exact source closure and revocation snapshot. Complete outcomes must return I02's
/// sealed `VerifiedMeasuredOutcome`; unavailable outcomes must be authenticated and return None.
pub trait UniverseEvidenceVerifier {
    /// Performs the verify view operation and returns an error if it cannot be completed.
    fn verify_view(&self, view: &ImmutableSourceView) -> Result<()>;

    /// Performs the verify observation operation and returns an error if it cannot be completed.
    fn verify_observation(
        &self,
        view: &ImmutableSourceView,
        observation: &ObservedOutcome,
    ) -> Result<Option<VerifiedMeasuredOutcome>>;
}
