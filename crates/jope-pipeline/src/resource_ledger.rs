//! Additive, opt-in physical effort accounting. Persistence is supplied by the
//! existing caller's journal; every transition is persisted before it takes effect.
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const SCHEMA: &str = "jope.resource-ledger/v1";

/// I02's length-prefixed domain separation over canonical JSON. This local
/// adapter only admits exact interoperable integers (no floats); it needs no
/// new Cargo dependency or change to any historical receipt hash.
fn identity(schema: &str, value: &impl Serialize) -> Result<String> {
    fn canonical(value: &serde_json::Value, out: &mut Vec<u8>) -> Result<()> {
        match value {
            serde_json::Value::Object(map) => {
                let mut keys: Vec<_> = map.keys().collect();
                keys.sort_by_key(|key| key.encode_utf16().collect::<Vec<_>>());
                out.push(b'{');
                for (index, key) in keys.into_iter().enumerate() {
                    if index > 0 {
                        out.push(b',');
                    }
                    out.extend(serde_json::to_vec(key)?);
                    out.push(b':');
                    canonical(&map[key], out)?;
                }
                out.push(b'}');
            }
            serde_json::Value::Array(array) => {
                out.push(b'[');
                for (index, value) in array.iter().enumerate() {
                    if index > 0 {
                        out.push(b',');
                    }
                    canonical(value, out)?;
                }
                out.push(b']');
            }
            serde_json::Value::Number(number) => {
                const MAX: u64 = (1_u64 << 53) - 1;
                ensure!(
                    number.as_u64().is_some_and(|v| v <= MAX)
                        || number.as_i64().is_some_and(|v| v.unsigned_abs() <= MAX),
                    "resource identity requires exact interoperable integer"
                );
                out.extend(number.to_string().bytes());
            }
            _ => out.extend(serde_json::to_vec(value)?),
        }
        Ok(())
    }
    let mut payload = Vec::new();
    canonical(&serde_json::to_value(value)?, &mut payload)?;
    let mut bytes = schema.as_bytes().to_vec();
    bytes.push(0);
    bytes.extend(u64::try_from(payload.len())?.to_be_bytes());
    bytes.extend(payload);
    Ok(crate::sha256_hex(&bytes))
}

/// Integer physical units, never interchangeable. One ledger has at most one
/// exact GPU device/MIG allocation (named in its policy).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Unit {
    CpuCoreMicros,
    GpuDeviceMicros,
    AttemptWallMicros,
    PeakMemoryBytes,
    MemoryByteSeconds,
    StorageBytes,
    StorageOperations,
    Evaluations,
    Compilations,
    CustodyOperations,
    Retries,
}

/// A sparse vector; an absent cap does not authorize that resource.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Amounts(pub BTreeMap<Unit, u64>);
impl Amounts {
    pub fn new(values: impl IntoIterator<Item = (Unit, u64)>) -> Self {
        Self(values.into_iter().collect())
    }
    pub fn get(&self, unit: Unit) -> u64 {
        self.0.get(&unit).copied().unwrap_or(0)
    }
    fn add(&mut self, unit: Unit, amount: u64) -> Result<()> {
        let value = self
            .get(unit)
            .checked_add(amount)
            .context("resource overflow")?;
        self.0.insert(unit, value);
        Ok(())
    }
}

/// Convert measured seconds without accepting negative, nonfinite or overflowing
/// counters. Round upward so sub-microsecond positive work is never free.
pub fn seconds_to_micros(seconds: f64) -> Result<u64> {
    let micros = (seconds * 1_000_000.).ceil();
    ensure!(
        seconds.is_finite() && seconds >= 0. && micros < u64::MAX as f64,
        "invalid or overflowing seconds"
    );
    Ok(micros as u64)
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JoinIds {
    pub experiment_sha256: String,
    /// I05 identifiers are opaque SHA-256 joins, not a second event authority.
    pub draw_id: String,
    pub request_id: String,
    pub attempt_id: String,
    pub logical_id: String,
}
impl JoinIds {
    fn validate(&self) -> Result<()> {
        for id in [
            &self.experiment_sha256,
            &self.draw_id,
            &self.request_id,
            &self.attempt_id,
            &self.logical_id,
        ] {
            ensure!(
                crate::experiment::sha256(id),
                "invalid effort join identity"
            );
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Stage {
    Prepare,
    Encode,
    Compile,
    Fit,
    Evaluate,
    Checkpoint,
    Publication,
    Custody,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case", deny_unknown_fields)]
pub enum CacheState {
    Cold {
        lifetime_id: String,
    },
    Warm {
        lifetime_id: String,
        preparation_cost_sha256: String,
    },
    Unavailable {
        reason: String,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Policy {
    pub schema: String,
    pub experiment_sha256: String,
    /// Immutable original window, shared by retries and every accounting stage.
    pub opened_unix_us: u64,
    pub deadline_unix_us: u64,
    pub terminal_headroom_us: u64,
    pub cap: Amounts,
    /// Withheld from work reservations for checkpoint/publication/cleanup.
    pub terminal_reserve: Amounts,
    pub gpu_allocation: Option<String>,
    pub conversion: Option<CreditConversion>,
}
impl Policy {
    pub fn identity(&self) -> Result<String> {
        identity("jope.resource-policy/v1", self)
    }
    pub fn work_deadline(&self) -> Result<u64> {
        self.deadline_unix_us
            .checked_sub(self.terminal_headroom_us)
            .context("headroom exceeds original window")
    }
    fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == SCHEMA && crate::experiment::sha256(&self.experiment_sha256),
            "invalid resource policy identity"
        );
        ensure!(
            self.opened_unix_us < self.work_deadline()? && self.terminal_headroom_us > 0,
            "empty original work window or missing terminal headroom"
        );
        ensure!(!self.cap.0.is_empty(), "empty resource cap");
        for (&unit, &amount) in &self.terminal_reserve.0 {
            ensure!(
                self.cap.0.contains_key(&unit) && amount <= self.cap.get(unit),
                "terminal reserve exceeds cap"
            );
        }
        if self.cap.0.contains_key(&Unit::GpuDeviceMicros) {
            ensure!(
                self.gpu_allocation
                    .as_ref()
                    .is_some_and(|v| !v.trim().is_empty()),
                "GPU cap lacks exact device/MIG allocation"
            );
        }
        if let Some(conversion) = &self.conversion {
            conversion.validate(self.opened_unix_us)?;
        }
        self.identity()?;
        Ok(())
    }
}

/// A prospective, versioned integer credit per physical microsecond. Raw units
/// remain present. The caller supplies the preregistration's immutable digest.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CreditConversion {
    pub version: String,
    pub preregistration_sha256: String,
    pub registered_unix_us: u64,
    pub cpu_credits_per_micro: u64,
    pub gpu_credits_per_micro: u64,
}
impl CreditConversion {
    fn validate(&self, opened: u64) -> Result<()> {
        ensure!(
            !self.version.is_empty()
                && crate::experiment::sha256(&self.preregistration_sha256)
                && self.registered_unix_us <= opened
                && self.cpu_credits_per_micro > 0
                && self.gpu_credits_per_micro > 0,
            "invalid or retrospective credit conversion"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReservationRequest {
    pub ids: JoinIds,
    pub stage: Stage,
    pub requested: Amounts,
    pub retry_of: Option<String>,
    pub cache: CacheState,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Reservation {
    pub schema: String,
    pub policy_sha256: String,
    pub request: ReservationRequest,
    pub reserved_unix_us: u64,
    pub deadline_unix_us: u64,
}
impl Reservation {
    pub fn identity(&self) -> Result<String> {
        identity("jope.resource-reservation/v1", self)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case", deny_unknown_fields)]
pub enum Reading {
    Known { value: u64, uncertainty: u64 },
    Unknown { reason: String },
}
impl Reading {
    pub fn exact(value: u64) -> Self {
        Self::Known {
            value,
            uncertainty: 0,
        }
    }
    pub fn unknown(reason: impl Into<String>) -> Self {
        Self::Unknown {
            reason: reason.into(),
        }
    }
}
/// Paths identify containment in the same physical accounting hierarchy. Parent
/// inclusive counters and descendant counters with overlapping intervals cannot
/// both be charged. Disjoint child cgroups may run concurrently.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Measurement {
    pub reading: Reading,
    pub collector: String,
    pub clock: String,
    pub allocation: String,
    pub scope: Vec<String>,
    pub from_unix_us: u64,
    pub to_unix_us: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Disposition {
    Completed,
    Failed,
    Killed,
    Cancelled,
    Interrupted,
    ProvenNotStarted,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AttemptCost {
    pub schema: String,
    pub reservation_sha256: String,
    pub ids: JoinIds,
    pub disposition: Disposition,
    pub ended_unix_us: u64,
    pub exit_code: Option<i32>,
    pub signal: Option<i32>,
    pub measurements: BTreeMap<Unit, Measurement>,
    /// Raw worker/log/partial receipts survive every terminal disposition.
    pub evidence_sha256: BTreeSet<String>,
    pub retained_sha256: BTreeSet<String>,
    /// Retained physical bytes require their own measurement; references and
    /// successful computation alone do not establish a storage byte count.
    pub retained_bytes: Reading,
    pub reason: String,
}
impl AttemptCost {
    pub fn identity(&self) -> Result<String> {
        identity("jope.resource-attempt-cost/v1", self)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "transition", rename_all = "snake_case", deny_unknown_fields)]
pub enum Event {
    Reserved { reservation: Reservation },
    Started { attempt_id: String, at_unix_us: u64 },
    Finished { cost: AttemptCost },
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct Reconciliation {
    pub requested: Amounts,
    pub active_reserved: Amounts,
    pub started: u64,
    pub terminal: u64,
    pub retained: u64,
    pub retained_bytes_known: u64,
    pub retained_bytes_unknown_attempts: u64,
    pub computed_known: Amounts,
    pub unknown_attempts: BTreeMap<Unit, u64>,
    /// Unknown cumulative consumption conservatively keeps its full reservation.
    pub charged: Amounts,
    pub released: Amounts,
    pub over_cap: bool,
}

/// One writer owns the existing journal. Sharing this value requires the same
/// mutex as its persistence callback; mutable access serializes admission.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResourceLedger {
    policy: Policy,
    reservations: BTreeMap<String, Reservation>,
    starts: BTreeMap<String, u64>,
    costs: BTreeMap<String, AttemptCost>,
    last_unix_us: u64,
}
impl ResourceLedger {
    pub fn new(policy: Policy) -> Result<Self> {
        policy.validate()?;
        let last_unix_us = policy.opened_unix_us;
        Ok(Self {
            policy,
            reservations: BTreeMap::new(),
            starts: BTreeMap::new(),
            costs: BTreeMap::new(),
            last_unix_us,
        })
    }
    pub fn policy(&self) -> &Policy {
        &self.policy
    }
    pub fn reservation(&self, id: &str) -> Option<&Reservation> {
        self.reservations.get(id)
    }
    pub fn cost(&self, id: &str) -> Option<&AttemptCost> {
        self.costs.get(id)
    }
    pub fn started_at(&self, id: &str) -> Option<u64> {
        self.starts.get(id).copied()
    }

    /// Replay journal order, including duplicates, without opening a new window.
    pub fn restore(policy: Policy, events: &[Event]) -> Result<Self> {
        let mut ledger = Self::new(policy)?;
        for event in events {
            match event {
                Event::Reserved { reservation } => {
                    let actual = ledger.reserve(
                        reservation.request.clone(),
                        reservation.reserved_unix_us,
                        |_| Ok(()),
                    )?;
                    ensure!(
                        &actual == reservation,
                        "changed reservation on journal replay"
                    );
                }
                Event::Started {
                    attempt_id,
                    at_unix_us,
                } => ledger.start(attempt_id, *at_unix_us, |_| Ok(()))?,
                Event::Finished { cost } => ledger.finish(cost.clone(), |_| Ok(()))?,
            }
        }
        Ok(ledger)
    }

    pub fn reserve(
        &mut self,
        request: ReservationRequest,
        now: u64,
        persist: impl FnOnce(&Event) -> Result<()>,
    ) -> Result<Reservation> {
        request.ids.validate()?;
        if let Some(prior) = self.reservations.get(&request.ids.attempt_id) {
            ensure!(prior.request == request, "conflicting replayed reservation");
            return Ok(prior.clone());
        }
        ensure!(now >= self.last_unix_us, "accounting clock moved backwards");
        ensure!(
            request.ids.experiment_sha256 == self.policy.experiment_sha256,
            "foreign experiment reservation"
        );
        let terminal_stage = matches!(
            request.stage,
            Stage::Checkpoint | Stage::Publication | Stage::Custody
        );
        let deadline = if terminal_stage {
            self.policy.deadline_unix_us
        } else {
            self.policy.work_deadline()?
        };
        let available_us = deadline
            .checked_sub(now)
            .context("original deadline expired")?;
        ensure!(
            available_us > 0
                && request.requested.get(Unit::AttemptWallMicros) > 0
                && request.requested.get(Unit::AttemptWallMicros) <= available_us,
            "work exceeds original deadline/headroom"
        );
        let count = match request.stage {
            Stage::Compile => Some(Unit::Compilations),
            Stage::Fit | Stage::Evaluate => Some(Unit::Evaluations),
            Stage::Custody | Stage::Publication | Stage::Checkpoint => {
                Some(Unit::CustodyOperations)
            }
            _ => None,
        };
        if let Some(unit) = count {
            ensure!(
                request.requested.get(unit) == 1,
                "stage must reserve one physical operation"
            );
        }
        if let Some(prior) = &request.retry_of {
            let cost = self
                .costs
                .get(prior)
                .context("retry lacks terminal predecessor")?;
            ensure!(
                !matches!(
                    cost.disposition,
                    Disposition::Completed | Disposition::ProvenNotStarted
                ) && cost.ids.logical_id == request.ids.logical_id
                    && cost.ids.request_id == request.ids.request_id
                    && cost.ids.draw_id == request.ids.draw_id
                    && request.requested.get(Unit::Retries) == 1,
                "retry must retain failed physical attempt and original joins"
            );
        } else {
            ensure!(
                request.requested.get(Unit::Retries) == 0,
                "retry count without predecessor"
            );
        }
        match &request.cache {
            CacheState::Warm {
                lifetime_id,
                preparation_cost_sha256,
            } => {
                ensure!(
                    !lifetime_id.is_empty() && crate::experiment::sha256(preparation_cost_sha256),
                    "warm cache lacks original lifetime preparation cost"
                );
            }
            CacheState::Cold { lifetime_id } => {
                ensure!(!lifetime_id.is_empty(), "empty cache lifetime")
            }
            CacheState::Unavailable { reason } => {
                ensure!(!reason.is_empty(), "missing cache reason")
            }
        }
        let totals = self.reconcile()?;
        ensure!(!totals.over_cap, "prior overrun prevents more admission");
        for (&unit, &amount) in &request.requested.0 {
            ensure!(
                self.policy.cap.0.contains_key(&unit),
                "resource not authorized in cap"
            );
            let withheld = if terminal_stage {
                0
            } else {
                self.policy.terminal_reserve.get(unit)
            };
            let used = if unit == Unit::PeakMemoryBytes {
                0
            } else {
                totals.charged.get(unit)
            };
            let total = used
                .checked_add(totals.active_reserved.get(unit))
                .and_then(|v| v.checked_add(amount))
                .and_then(|v| v.checked_add(withheld))
                .context("reservation overflow")?;
            ensure!(
                total <= self.policy.cap.get(unit),
                "resource cap/terminal reserve exceeded: {unit:?}"
            );
        }
        let reservation = Reservation {
            schema: SCHEMA.into(),
            policy_sha256: self.policy.identity()?,
            request,
            reserved_unix_us: now,
            deadline_unix_us: deadline,
        };
        reservation.identity()?;
        persist(&Event::Reserved {
            reservation: reservation.clone(),
        })?;
        self.last_unix_us = now;
        self.reservations.insert(
            reservation.request.ids.attempt_id.clone(),
            reservation.clone(),
        );
        Ok(reservation)
    }

    pub fn start(
        &mut self,
        id: &str,
        now: u64,
        persist: impl FnOnce(&Event) -> Result<()>,
    ) -> Result<()> {
        if let Some(prior) = self.starts.get(id) {
            ensure!(*prior == now, "conflicting attempt start");
            return Ok(());
        }
        let reservation = self
            .reservations
            .get(id)
            .context("start without durable reservation")?;
        ensure!(
            !self.costs.contains_key(id)
                && now >= reservation.reserved_unix_us
                && now
                    .checked_add(reservation.request.requested.get(Unit::AttemptWallMicros))
                    .is_some_and(|end| end <= reservation.deadline_unix_us),
            "start exceeds original reservation window"
        );
        persist(&Event::Started {
            attempt_id: id.into(),
            at_unix_us: now,
        })?;
        self.starts.insert(id.into(), now);
        self.last_unix_us = self.last_unix_us.max(now);
        Ok(())
    }

    /// Overruns are retained, then block further admission. Unknowns keep their
    /// allowance charged; they never return unmeasured capacity to the worker.
    pub fn finish(
        &mut self,
        cost: AttemptCost,
        persist: impl FnOnce(&Event) -> Result<()>,
    ) -> Result<()> {
        if let Some(prior) = self.costs.get(&cost.ids.attempt_id) {
            ensure!(prior == &cost, "conflicting terminal attempt receipt");
            return Ok(());
        }
        let reservation = self
            .reservations
            .get(&cost.ids.attempt_id)
            .context("cost without reservation")?;
        ensure!(
            cost.schema == SCHEMA
                && cost.ids == reservation.request.ids
                && cost.reservation_sha256 == reservation.identity()?
                && !cost.reason.is_empty(),
            "cost identity/measurement reason mismatch"
        );
        let start = self.starts.get(&cost.ids.attempt_id).copied();
        ensure!(
            cost.disposition == Disposition::ProvenNotStarted || start.is_some(),
            "cost before start"
        );
        ensure!(
            cost.ended_unix_us >= start.unwrap_or(reservation.reserved_unix_us),
            "cost clock moved backwards"
        );
        ensure!(
            cost.measurements
                .keys()
                .eq(reservation.request.requested.0.keys()),
            "every reserved dimension needs known or explicitly unknown telemetry"
        );
        ensure!(
            !(cost.exit_code.is_some() && cost.signal.is_some()),
            "exit and signal are mutually exclusive"
        );
        ensure!(
            cost.retained_sha256
                .iter()
                .chain(&cost.evidence_sha256)
                .all(|v| crate::experiment::sha256(v))
                && !cost.evidence_sha256.is_empty(),
            "missing or invalid evidence digest"
        );
        ensure!(
            cost.retained_sha256.is_empty() || cost.disposition == Disposition::Completed,
            "failed physical work cannot claim a retained result"
        );
        if let Reading::Unknown { reason } = &cost.retained_bytes {
            ensure!(
                !reason.is_empty(),
                "unknown retained storage requires reason"
            );
        }
        for (&unit, measurement) in &cost.measurements {
            ensure!(
                !measurement.collector.is_empty()
                    && !measurement.clock.is_empty()
                    && !measurement.allocation.is_empty()
                    && !measurement.scope.is_empty()
                    && measurement
                        .scope
                        .iter()
                        .all(|v| !v.is_empty() && v != "." && v != ".." && !v.contains('/'))
                    && measurement.from_unix_us >= reservation.reserved_unix_us
                    && measurement.from_unix_us <= measurement.to_unix_us
                    && measurement.to_unix_us <= cost.ended_unix_us,
                "invalid measurement provenance/interval"
            );
            if unit == Unit::GpuDeviceMicros {
                ensure!(
                    self.policy.gpu_allocation.as_ref() == Some(&measurement.allocation),
                    "wrong GPU/MIG allocation"
                );
            }
            if let Reading::Unknown { reason } = &measurement.reading {
                ensure!(!reason.is_empty(), "unknown telemetry requires reason");
            }
            if cost.disposition != Disposition::ProvenNotStarted
                && matches!(
                    unit,
                    Unit::Evaluations
                        | Unit::Compilations
                        | Unit::CustodyOperations
                        | Unit::Retries
                )
            {
                if let Reading::Known { value, uncertainty } = measurement.reading {
                    ensure!(
                        value == 1 && uncertainty == 0,
                        "started physical operation must be counted once or explicitly unknown"
                    );
                }
            }
            if cost.disposition == Disposition::ProvenNotStarted {
                ensure!(
                    measurement.reading == Reading::exact(0)
                        && cost.reason.starts_with("not_started: gate never released")
                        && cost.exit_code.is_none()
                        && cost.signal.is_none()
                        && start.is_none(),
                    "unstarted zero needs a proven unopened gate"
                );
            }
            for prior in self
                .costs
                .values()
                .filter_map(|c| c.measurements.get(&unit))
            {
                if matches!(measurement.reading, Reading::Known { value, .. } if value > 0)
                    && matches!(prior.reading, Reading::Known { value, .. } if value > 0)
                {
                    ensure!(
                        measurement.allocation != prior.allocation
                            || !(measurement.scope.starts_with(&prior.scope)
                                || prior.scope.starts_with(&measurement.scope))
                            || measurement.to_unix_us <= prior.from_unix_us
                            || prior.to_unix_us <= measurement.from_unix_us,
                        "overlapping inclusive measurement domains would double charge"
                    );
                }
            }
        }
        // Validate the full aggregate before journaling, without discarding any
        // valid overrun just because it exceeded a requested allowance.
        let mut candidate = self.clone();
        cost.identity()?;
        candidate
            .costs
            .insert(cost.ids.attempt_id.clone(), cost.clone());
        candidate.reconcile()?;
        persist(&Event::Finished { cost: cost.clone() })?;
        self.costs.insert(cost.ids.attempt_id.clone(), cost.clone());
        self.last_unix_us = self.last_unix_us.max(cost.ended_unix_us);
        Ok(())
    }

    pub fn reconcile(&self) -> Result<Reconciliation> {
        let mut out = Reconciliation {
            started: self.starts.len() as u64,
            terminal: self.costs.len() as u64,
            retained: self
                .costs
                .values()
                .filter(|cost| !cost.retained_sha256.is_empty())
                .count() as u64,
            ..Default::default()
        };
        for cost in self.costs.values() {
            match cost.retained_bytes {
                Reading::Known { value, .. } => {
                    out.retained_bytes_known = out
                        .retained_bytes_known
                        .checked_add(value)
                        .context("retained storage overflow")?
                }
                Reading::Unknown { .. } => out.retained_bytes_unknown_attempts += 1,
            }
        }
        for (id, reservation) in &self.reservations {
            for (&unit, &amount) in &reservation.request.requested.0 {
                out.requested.add(unit, amount)?;
                if let Some(cost) = self.costs.get(id) {
                    let charged = match cost.measurements[&unit].reading {
                        Reading::Known { value, uncertainty } => {
                            if unit == Unit::PeakMemoryBytes {
                                out.computed_known
                                    .0
                                    .insert(unit, out.computed_known.get(unit).max(value));
                            } else {
                                out.computed_known.add(unit, value)?;
                            }
                            value
                                .checked_add(uncertainty)
                                .context("uncertainty overflow")?
                        }
                        Reading::Unknown { .. } => {
                            let count = out.unknown_attempts.entry(unit).or_default();
                            *count = count.checked_add(1).context("attempt count overflow")?;
                            amount
                        }
                    };
                    if unit == Unit::PeakMemoryBytes {
                        out.charged
                            .0
                            .insert(unit, out.charged.get(unit).max(charged));
                    } else {
                        out.charged.add(unit, charged)?;
                    }
                    out.released.add(unit, amount.saturating_sub(charged))?;
                    out.over_cap |= charged > amount;
                } else {
                    out.active_reserved.add(unit, amount)?;
                }
            }
        }
        out.over_cap |= out
            .charged
            .0
            .iter()
            .any(|(&unit, &amount)| amount > self.policy.cap.get(unit));
        Ok(out)
    }

    /// No implicit CPU+GPU total exists. Credits require the frozen prospective
    /// conversion and complete raw CPU/GPU telemetry.
    pub fn credits(&self) -> Result<u64> {
        let conversion = self
            .policy
            .conversion
            .as_ref()
            .context("no preregistered credit conversion")?;
        let totals = self.reconcile()?;
        ensure!(
            totals.active_reserved.0.is_empty()
                && totals
                    .unknown_attempts
                    .get(&Unit::CpuCoreMicros)
                    .copied()
                    .unwrap_or(0)
                    == 0
                && totals
                    .unknown_attempts
                    .get(&Unit::GpuDeviceMicros)
                    .copied()
                    .unwrap_or(0)
                    == 0,
            "credit total unavailable while consumption is unknown/in flight"
        );
        totals
            .computed_known
            .get(Unit::CpuCoreMicros)
            .checked_mul(conversion.cpu_credits_per_micro)
            .and_then(|cpu| {
                totals
                    .computed_known
                    .get(Unit::GpuDeviceMicros)
                    .checked_mul(conversion.gpu_credits_per_micro)
                    .and_then(|gpu| cpu.checked_add(gpu))
            })
            .context("credit conversion overflow")
    }
}
