use super::*;

/// Explicit draft bounds. The defaults are prospective worker limits, not adopted product caps.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PendingLimits {
    /// Upper bound on requests.
    pub max_requests: u64,
    /// Upper bound on pending.
    pub max_pending: u64,
    /// Upper bound on total bytes.
    pub max_total_bytes: u64,
    /// Upper bound on startup requests.
    pub max_startup_requests: u64,
    /// Upper bound on draws per request.
    pub max_draws_per_request: u64,
    /// Upper bound on attempts.
    pub max_attempts: u64,
    /// Upper bound on record bytes.
    pub max_record_bytes: u64,
    /// Upper bound on status bytes.
    pub max_status_bytes: u64,
}

impl PendingLimits {
    /// Constant identifying the draft.
    pub const DRAFT: Self = Self {
        max_requests: 10_000,
        max_pending: 1_000,
        max_total_bytes: 1 << 30,
        max_startup_requests: 10_000,
        max_draws_per_request: 10_000,
        max_attempts: 4,
        max_record_bytes: 1 << 20,
        max_status_bytes: 16 << 10,
    };

    /// Validates this value against its invariants and returns an error when validation fails.
    pub fn validate(self) -> Result<()> {
        ensure!(
            self.max_requests > 0 && self.max_pending > 0,
            "request limits must be positive"
        );
        ensure!(
            self.max_pending <= self.max_requests,
            "pending limit exceeds total requests"
        );
        ensure!(
            self.max_startup_requests > 0 && self.max_startup_requests <= self.max_requests,
            "invalid startup request bound"
        );
        ensure!(
            self.max_draws_per_request > 0 && self.max_attempts > 0,
            "draw and attempt limits must be positive"
        );
        ensure!(
            self.max_record_bytes > 0 && self.max_status_bytes > 0 && self.max_total_bytes > 0,
            "byte limits must be positive"
        );
        Ok(())
    }
}

/// I05/I16 draft identity inputs. The physical key intentionally excludes draw IDs and ledger
/// multiplicity: several logical draws may name the same physical evaluation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RequestContext {
    /// SHA-256 digest identifying the associated context.
    pub context_sha256: String,
    /// SHA-256 digest identifying the associated recipe.
    pub recipe_sha256: String,
    /// SHA-256 digest identifying the associated fidelity.
    pub fidelity_sha256: String,
    /// SHA-256 digest identifying the associated source.
    pub source_sha256: String,
    /// SHA-256 digest identifying the associated original window.
    pub original_window_sha256: String,
    /// Identifier for the associated reservation.
    pub reservation_id: String,
    /// Deadline unix ms recorded for this request context.
    pub deadline_unix_ms: u64,
}

impl RequestContext {
    /// Validates this value against its invariants and returns an error when validation fails.
    pub fn validate(&self) -> Result<()> {
        for (name, digest) in [
            ("context", &self.context_sha256),
            ("recipe", &self.recipe_sha256),
            ("fidelity", &self.fidelity_sha256),
            ("source", &self.source_sha256),
            ("original window", &self.original_window_sha256),
        ] {
            ensure!(is_hex64(digest), "{name} identity must be lowercase sha256");
        }
        ensure!(
            !self.reservation_id.is_empty(),
            "reservation ID is required"
        );
        ensure!(self.deadline_unix_ms > 0, "original deadline is required");
        Ok(())
    }

    /// Performs the physical request key operation and returns an error if it cannot be completed.
    pub fn physical_request_key(&self) -> Result<String> {
        self.validate()?;
        canonical_sha256(self)
    }
}

/// Immutable reference to one original logical draw. `multiplicity` preserves the ledger's
/// denominator contribution even when canonical recipes are equal.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DrawRef {
    /// Identifier for the associated draw.
    pub draw_id: String,
    /// SHA-256 digest identifying the associated slot record.
    pub slot_record_sha256: String,
    /// SHA-256 digest identifying the associated canonical draw.
    pub canonical_draw_sha256: String,
    /// Multiplicity recorded for this draw ref.
    pub multiplicity: u64,
    /// Shared denominator used to interpret probability numerators in this law.
    pub denominator: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `ProposalBundle` is a record for the proposal bundle contract in this module: Bounded durable proposal delivery state in the draft proposals/delivery-v1 namespace.
pub struct ProposalBundle {
    /// SHA-256 digest identifying the associated ledger.
    pub ledger_sha256: String,
    /// Number of draws included in the calculation.
    pub draws: Vec<DrawRef>,
}

impl ProposalBundle {
    /// Validates this value against its invariants and returns an error when validation fails.
    pub fn validate(&self, limits: PendingLimits) -> Result<()> {
        ensure!(
            is_hex64(&self.ledger_sha256),
            "draw ledger digest is invalid"
        );
        ensure!(
            !self.draws.is_empty(),
            "proposal must retain at least one original draw"
        );
        ensure!(
            self.draws.len() as u64 <= limits.max_draws_per_request,
            "draw count exceeds admitted per-request bound"
        );
        let mut ids = std::collections::BTreeSet::new();
        for draw in &self.draws {
            ensure!(
                !draw.draw_id.is_empty() && ids.insert(&draw.draw_id),
                "draw IDs must be present and unique"
            );
            ensure!(
                is_hex64(&draw.slot_record_sha256) && is_hex64(&draw.canonical_draw_sha256),
                "draw reference digest is invalid"
            );
            ensure!(
                draw.multiplicity > 0 && draw.denominator >= draw.multiplicity,
                "draw multiplicity is outside its denominator"
            );
        }
        Ok(())
    }
}

/// Coordinator adapter that proves IDs and immutable references came from the accepted I05 draw
/// ledger and I16 reservation/window. It checks existing authority; it issues none.
pub trait AcceptedProposalInputs {
    /// Performs the verify request context operation and returns an error if it cannot be completed.
    fn verify_request_context(&self, context: &RequestContext) -> Result<()>;
    /// Performs the verify draw bundle operation and returns an error if it cannot be completed.
    fn verify_draw_bundle(&self, bundle: &ProposalBundle) -> Result<()>;
}

/// Coordinator-supplied verifier for the I16 terminal/reap receipt that resolves an unknown start.
pub trait TerminalProofVerifier {
    /// Performs the verify terminal or reap operation and returns an error if it cannot be completed.
    fn verify_terminal_or_reap(
        &self,
        request_key: &str,
        generation: u64,
        receipt_sha256: &str,
    ) -> Result<()>;
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `ProposalRecord` is a record for the proposal record contract in this module: Bounded durable proposal delivery state in the draft proposals/delivery-v1 namespace.
pub struct ProposalRecord {
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// Request key recorded for this proposal record.
    pub request_key: String,
    /// Context recorded for this proposal record.
    pub context: RequestContext,
    /// Bundle recorded for this proposal record.
    pub bundle: ProposalBundle,
    /// SHA-256 digest identifying the associated record.
    pub record_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
/// `TerminalKind` is an enum for the terminal kind contract in this module: Bounded durable proposal delivery state in the draft proposals/delivery-v1 namespace.
pub enum TerminalKind {
    /// The `Released` case of `TerminalKind`.
    Released {
        /// Identifier for the associated physical attempt.
        physical_attempt_id: String,
    },
    /// The `Refused` case of `TerminalKind`.
    Refused {
        /// SHA-256 digest identifying the associated reason.
        reason_sha256: String,
    },
    /// The `Failed` case of `TerminalKind`.
    Failed {
        /// SHA-256 digest of the receipt that authenticates this result.
        receipt_sha256: String,
        /// Retryable recorded for this terminal kind.
        retryable: bool,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `PhysicalAttempt` is a record for the physical attempt contract in this module: Bounded durable proposal delivery state in the draft proposals/delivery-v1 namespace.
pub struct PhysicalAttempt {
    /// Generation recorded for this physical attempt.
    pub generation: u64,
    /// Owner recorded for this physical attempt.
    pub owner: String,
    /// Started unix ms recorded for this physical attempt.
    pub started_unix_ms: u64,
    /// Unknown recorded for this physical attempt.
    pub unknown: bool,
    /// Optional terminal value; `None` means it is not available.
    pub terminal: Option<TerminalKind>,
    /// SHA-256 digest identifying the associated ack.
    pub ack_sha256: Option<String>,
    /// Effort only; this field never grants committed optimizer/update exposure.
    pub physical_cost_units: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `Lease` is a record for the lease contract in this module: Bounded durable proposal delivery state in the draft proposals/delivery-v1 namespace.
pub struct Lease {
    /// Request key recorded for this lease.
    pub request_key: String,
    /// Generation recorded for this lease.
    pub generation: u64,
    /// Owner recorded for this lease.
    pub owner: String,
    /// Expires unix ms recorded for this lease.
    pub expires_unix_ms: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `DeliveryStatus` is a record for the delivery status contract in this module: Bounded durable proposal delivery state in the draft proposals/delivery-v1 namespace.
pub struct DeliveryStatus {
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// Request key recorded for this delivery status.
    pub request_key: String,
    /// Generation recorded for this delivery status.
    pub generation: u64,
    /// Optional lease value; `None` means it is not available.
    pub lease: Option<Lease>,
    /// Collection of attempts values carried by this delivery status.
    pub attempts: Vec<PhysicalAttempt>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// `Inserted` is an enum for the inserted contract in this module: Bounded durable proposal delivery state in the draft proposals/delivery-v1 namespace.
pub enum Inserted {
    /// The `Added` case of `Inserted`.
    Added {
        /// Request key recorded for this outcome.
        request_key: String,
    },
    /// The `Existing` case of `Inserted`.
    Existing {
        /// Request key recorded for this outcome.
        request_key: String,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// `Recovery` is a record for the recovery contract in this module: Bounded durable proposal delivery state in the draft proposals/delivery-v1 namespace.
pub struct Recovery {
    /// Requests recorded for this recovery.
    pub requests: u64,
    /// Pending recorded for this recovery.
    pub pending: u64,
    /// Byte count for the associated durable.
    pub durable_bytes: u64,
}
