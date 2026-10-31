//! Lossless archival observations. Retention does not issue replay capabilities.

use super::index_producer::{
    self, HistoricalFixture, HistoricalStatus, ProducerClosure, ProducerInput, RowIdentity,
};
use crate::canon::is_hex64;
use crate::retention::parquet::{self, Cell, ColumnSpec, ColumnType, Table, WriteOptions};
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
/// `ObservedOutcome` is an enum for the observed outcome contract in this module: Lossless archival observations. Retention does not issue replay capabilities.
pub enum ObservedOutcome {
    /// The `Ok` case of `ObservedOutcome`.
    Ok,
    /// The `Failed` case of `ObservedOutcome`.
    Failed,
    /// The `Timeout` case of `ObservedOutcome`.
    Timeout,
    /// Only part of the outcome mass is supported by measured outcomes.
    Partial,
    /// The `Unknown` case of `ObservedOutcome`.
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "availability", rename_all = "snake_case")]
/// `Measurement` is an enum for the measurement contract in this module: Lossless archival observations. Retention does not issue replay capabilities.
pub enum Measurement {
    /// The `Missing` case of `Measurement`.
    Missing,
    /// The `Finite` case of `Measurement`.
    Finite {
        /// F64 bits recorded for this measurement.
        f64_bits: String,
        /// Optional score name value; `None` means it is not available.
        score_name: Option<String>,
    },
    /// The `Rejected` case of `Measurement`.
    Rejected {
        /// Explanation associated with the recorded disposition.
        reason: String,
    },
}
impl Measurement {
    /// Performs the value operation.
    pub fn value(&self) -> Option<f64> {
        match self {
            Self::Finite { f64_bits, .. } => {
                u64::from_str_radix(f64_bits, 16).ok().map(f64::from_bits)
            }
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "observation_type", rename_all = "snake_case")]
/// `DurationObservation` is an enum for the duration observation contract in this module: Lossless archival observations. Retention does not issue replay capabilities.
pub enum DurationObservation {
    /// Field value recorded by this public API type.
    /// All supported outcome mass is evaluated exactly.
    Exact {
        /// Milliseconds recorded for this outcome.
        milliseconds: u64,
    },
    /// The `RightCensored` case of `DurationObservation`.
    RightCensored {
        /// Lower bound ms recorded for this outcome.
        lower_bound_ms: u64,
    },
    /// The `Unknown` case of `DurationObservation`.
    Unknown,
}

/// Serialized I04/I05 evidence is retained without reinterpreting its authority. The draft
/// interfaces are integrated by the controller; neither JSON nor a copied role issues a grant.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct DependencyEvidence {
    /// Optional i04 admission value; `None` means it is not available.
    pub i04_admission: Option<Value>,
    /// Optional i05 events value; `None` means it is not available.
    pub i05_events: Option<Value>,
}
impl DependencyEvidence {
    /// Performs the from records operation and returns an error if it cannot be completed.
    pub fn from_records<A: Serialize, E: Serialize>(
        admission: Option<&A>,
        events: Option<&E>,
    ) -> Result<Self> {
        Ok(Self {
            i04_admission: admission.map(serde_json::to_value).transpose()?,
            i05_events: events.map(serde_json::to_value).transpose()?,
        })
    }
}

/// Availability is distinct from eligibility. Until exact I04/I05 closure is integrated, every
/// per-label global mask remains held. This type cannot be passed to any training export.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LabelHold {
    /// Observed recorded for this label hold.
    pub observed: bool,
    /// Collection of reasons values carried by this label hold.
    pub reasons: Vec<String>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
/// `LabelMasks` is a record for the label masks contract in this module: Lossless archival observations. Retention does not issue replay capabilities.
pub struct LabelMasks {
    /// Quality recorded for this label masks.
    pub quality: LabelHold,
    /// Runtime recorded for this label masks.
    pub runtime: LabelHold,
    /// Failure recorded for this label masks.
    pub failure: LabelHold,
    /// Policy recorded for this label masks.
    pub policy: LabelHold,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
/// `Disposition` is an enum for the disposition contract in this module: Lossless archival observations. Retention does not issue replay capabilities.
pub enum Disposition {
    /// The `Scored` case of `Disposition`.
    Scored,
    /// The `Failed` case of `Disposition`.
    Failed,
    /// The `Timeout` case of `Disposition`.
    Timeout,
    /// Only part of the outcome mass is supported by measured outcomes.
    Partial,
    /// The `Unscored` case of `Disposition`.
    Unscored,
    /// The `Unknown` case of `Disposition`.
    Unknown,
    /// The `RejectedMeasurement` case of `Disposition`.
    RejectedMeasurement,
    /// The `DuplicateDelivery` case of `Disposition`.
    DuplicateDelivery,
    /// The `Conflict` case of `Disposition`.
    Conflict,
}
impl Disposition {
    /// Returns the stable string representation of this value.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Scored => "scored",
            Self::Failed => "failed",
            Self::Timeout => "timeout",
            Self::Partial => "partial",
            Self::Unscored => "unscored",
            Self::Unknown => "unknown",
            Self::RejectedMeasurement => "rejected_measurement",
            Self::DuplicateDelivery => "duplicate_delivery",
            Self::Conflict => "conflict",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
/// `Observation` is a record for the observation contract in this module: Lossless archival observations. Retention does not issue replay capabilities.
pub struct Observation {
    /// Archive version recorded for this observation.
    pub archive_version: u32,
    /// Identity recorded for this observation.
    pub identity: RowIdentity,
    /// Original row recorded for this observation.
    pub original_row: Value,
    /// Outcome recorded for this observation.
    pub outcome: ObservedOutcome,
    /// Optional failure reason value; `None` means it is not available.
    pub failure_reason: Option<String>,
    /// Measurement recorded for this observation.
    pub measurement: Measurement,
    /// Duration recorded for this observation.
    pub duration: DurationObservation,
    /// Optional original role value; `None` means it is not available.
    pub original_role: Option<String>,
    /// Collection of provenance holds values carried by this observation.
    pub provenance_holds: Vec<String>,
    /// Disposition recorded for this observation.
    pub disposition: Disposition,
    /// Collection of related occurrences values carried by this observation.
    pub related_occurrences: Vec<String>,
    /// References are opaque: row counts are never physical-attempt or logical-success rates.
    pub event_references: BTreeMap<String, Value>,
    /// Dependencies recorded for this observation.
    pub dependencies: DependencyEvidence,
    /// Labels recorded for this observation.
    pub labels: LabelMasks,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
/// `ObservationArchive` is a record for the observation archive contract in this module: Lossless archival observations. Retention does not issue replay capabilities.
pub struct ObservationArchive {
    /// Archive version recorded for this observation archive.
    pub archive_version: u32,
    /// SHA-256 digest identifying the associated input.
    pub input_sha256: String,
    /// Exact original UTF-8 bytes, including numeric spelling, malformed measurement strings,
    /// whitespace and unknown fields. Row locators refer to this immutable input.
    pub original_index: String,
    /// Optional producer manifest value; `None` means it is not available.
    pub producer_manifest: Option<Vec<u8>>,
    /// Producer closure recorded for this observation archive.
    pub producer_closure: ProducerClosure,
    /// Collection of observations values carried by this observation archive.
    pub observations: Vec<Observation>,
    /// Mapping of primary counts values carried by this observation archive.
    pub primary_counts: BTreeMap<String, usize>,
    /// Historical recorded for this observation archive.
    pub historical: HistoricalStatus,
}

fn text(row: &Value, key: &str) -> Option<String> {
    row.get(key).and_then(Value::as_str).map(str::to_owned)
}
fn outcome(status: Option<&str>) -> ObservedOutcome {
    match status {
        Some("ok" | "completed" | "success") => ObservedOutcome::Ok,
        Some("failed" | "failure") => ObservedOutcome::Failed,
        Some("timeout" | "timed_out") => ObservedOutcome::Timeout,
        Some("partial") => ObservedOutcome::Partial,
        _ => ObservedOutcome::Unknown,
    }
}

fn observation(
    row: &Value,
    identity: RowIdentity,
    known: &BTreeSet<String>,
    dependencies: DependencyEvidence,
    bound: bool,
) -> Observation {
    let mut holds = Vec::new();
    for key in [
        "bucket",
        "key",
        "dataset",
        "program_sha",
        "context_sha256",
        "object_sha256",
    ] {
        let value = text(row, key);
        if value.as_ref().is_none_or(|s| s.trim().is_empty()) {
            holds.push(format!("missing_{key}"));
        } else if (key.ends_with("sha256") || key == "program_sha")
            && !value.as_deref().is_some_and(is_hex64)
        {
            holds.push(format!("invalid_{key}"));
        }
    }
    let role = text(row, "split_role").or_else(|| text(row, "role"));
    if role.is_none() {
        holds.push("missing_role".into());
    }
    if text(row, "split_role")
        .zip(text(row, "role"))
        .is_some_and(|(a, b)| a != b)
    {
        holds.push("conflicting_role_claims".into());
    }
    if !bound {
        holds.push("producer_closure_held".into());
    }
    let observed_outcome = outcome(
        text(row, "outcome")
            .or_else(|| text(row, "status"))
            .as_deref(),
    );
    if observed_outcome == ObservedOutcome::Unknown {
        holds.push("unknown_outcome".into());
    }
    if text(row, "outcome")
        .zip(text(row, "status"))
        .is_some_and(|(a, b)| outcome(Some(&a)) != outcome(Some(&b)))
    {
        holds.push("conflicting_outcome_claims".into());
    }
    let score_name = text(row, "score_name");
    let measurement = match row.get("score") {
        None | Some(Value::Null) => Measurement::Missing,
        Some(v) => match v.as_f64().filter(|v| v.is_finite()) {
            Some(v) => Measurement::Finite {
                f64_bits: format!("{:016x}", v.to_bits()),
                score_name: score_name.clone(),
            },
            None => Measurement::Rejected {
                reason: "malformed_or_nonfinite_score".into(),
            },
        },
    };
    if measurement.value().is_some() && !score_name.as_ref().is_some_and(|s| known.contains(s)) {
        holds.push("missing_or_unknown_score_identity".into());
    }
    let duration = match observed_outcome {
        ObservedOutcome::Timeout => row
            .get("lower_bound_ms")
            .and_then(Value::as_u64)
            .map(|lower_bound_ms| DurationObservation::RightCensored { lower_bound_ms })
            .unwrap_or(DurationObservation::Unknown),
        ObservedOutcome::Ok | ObservedOutcome::Failed | ObservedOutcome::Partial => row
            .get("elapsed_ms")
            .and_then(Value::as_u64)
            .map(|milliseconds| DurationObservation::Exact { milliseconds })
            .unwrap_or(DurationObservation::Unknown),
        ObservedOutcome::Unknown => DurationObservation::Unknown,
    };
    let disposition = match (&measurement, observed_outcome) {
        (Measurement::Rejected { .. }, _) => Disposition::RejectedMeasurement,
        (_, ObservedOutcome::Failed) => Disposition::Failed,
        (_, ObservedOutcome::Timeout) => Disposition::Timeout,
        (_, ObservedOutcome::Partial) => Disposition::Partial,
        (Measurement::Finite { .. }, _) => Disposition::Scored,
        (_, ObservedOutcome::Ok) => Disposition::Unscored,
        _ => Disposition::Unknown,
    };
    // A referenced or serialized capability must still be revalidated by I04 at consumption.
    let hold = |observed| LabelHold {
        observed,
        reasons: vec![
            "i04_verified_per_purpose_admission_required".into(),
            "i05_verified_event_closure_required".into(),
        ],
    };
    let labels = LabelMasks {
        quality: hold(measurement.value().is_some()),
        runtime: hold(!matches!(duration, DurationObservation::Unknown)),
        failure: hold(matches!(
            observed_outcome,
            ObservedOutcome::Failed | ObservedOutcome::Timeout
        )),
        policy: hold(false),
    };
    let event_references = [
        "draw_id",
        "request_id",
        "attempt_id",
        "retry_of",
        "event_sha256",
        "cost_ref",
    ]
    .iter()
    .filter_map(|k| row.get(*k).map(|v| (k.to_string(), v.clone())))
    .collect();
    Observation {
        archive_version: 2,
        identity,
        original_row: row.clone(),
        outcome: observed_outcome,
        failure_reason: text(row, "failure_reason").or_else(|| text(row, "reason")),
        measurement,
        duration,
        original_role: role,
        provenance_holds: holds,
        disposition,
        related_occurrences: Vec::new(),
        event_references,
        dependencies,
        labels,
    }
}

/// Every runs-array occurrence survives, including scalar/unresolved entries and duplicates.
/// Missing producer authentication holds completeness while retaining the supplied denominator.
pub fn import_index(
    bytes: &[u8],
    producer: Option<ProducerInput<'_>>,
    known: &BTreeSet<String>,
    dependencies: &BTreeMap<String, DependencyEvidence>,
    historical: Option<&HistoricalFixture>,
) -> Result<ObservationArchive> {
    let index = index_producer::parse_index(bytes)?;
    let input_sha256 = crate::canon::sha256_hex(bytes);
    let identities = index_producer::row_identities(&index, &input_sha256)?;
    let producer_closure = index_producer::verify_producer(bytes, producer);
    let bound = matches!(producer_closure, ProducerClosure::Bound { .. });
    let mut observations: Vec<_> = index["runs"]
        .as_array()
        .expect("parsed runs")
        .iter()
        .zip(identities)
        .map(|(row, id)| {
            let deps = dependencies
                .get(&id.occurrence_id)
                .cloned()
                .unwrap_or_default();
            observation(row, id, known, deps, bound)
        })
        .collect();
    let mut deliveries: BTreeMap<(String, String, Option<String>), Vec<usize>> = BTreeMap::new();
    for (i, obs) in observations.iter().enumerate() {
        if let (Some(bucket), Some(key)) = (
            text(&obs.original_row, "bucket"),
            text(&obs.original_row, "key"),
        ) {
            deliveries
                .entry((bucket, key, text(&obs.original_row, "version_id")))
                .or_default()
                .push(i);
        }
    }
    for occurrences in deliveries.values().filter(|v| v.len() > 1) {
        let first = occurrences[0];
        let conflict = occurrences.iter().any(|i| {
            observations[*i].identity.row_sha256 != observations[first].identity.row_sha256
        });
        for (n, &i) in occurrences.iter().enumerate() {
            observations[i].related_occurrences = occurrences
                .iter()
                .filter(|j| **j != i)
                .map(|j| observations[*j].identity.occurrence_id.clone())
                .collect();
            if conflict {
                observations[i].disposition = Disposition::Conflict;
                observations[i]
                    .provenance_holds
                    .push("conflicting_delivery".into());
            } else if n > 0 {
                observations[i].disposition = Disposition::DuplicateDelivery;
            }
        }
    }
    let mut primary_counts = BTreeMap::new();
    for obs in &observations {
        *primary_counts
            .entry(obs.disposition.as_str().into())
            .or_insert(0) += 1;
    }
    ensure!(
        primary_counts.values().sum::<usize>() == observations.len(),
        "observation denominator mismatch"
    );
    let historical = index_producer::verify_historical(
        historical,
        &input_sha256,
        &producer_closure,
        &primary_counts,
    );
    Ok(ObservationArchive {
        archive_version: 2,
        input_sha256,
        original_index: String::from_utf8(bytes.to_vec())?,
        producer_manifest: producer.map(|p| p.manifest_bytes.to_vec()),
        producer_closure,
        observations,
        primary_counts,
        historical,
    })
}

impl ObservationArchive {
    /// Local archival companion. The JSON column retains orthogonal masks, references and raw
    /// rows; the original input and producer bytes remain in the archive, not shared statistics.
    pub fn columnar(&self, opts: &WriteOptions) -> Result<Vec<u8>> {
        let column = |name: &str, ty, required| ColumnSpec {
            name: name.into(),
            ty,
            required,
        };
        let table = Table {
            columns: vec![
                column("occurrence_id", ColumnType::Utf8, true),
                column("score", ColumnType::Double, false),
                column("observation", ColumnType::Utf8, true),
            ],
            rows: self
                .observations
                .iter()
                .map(|o| {
                    Ok(vec![
                        Cell::Str(o.identity.occurrence_id.clone()),
                        o.measurement.value().map(Cell::F64).unwrap_or(Cell::Null),
                        Cell::Str(serde_json::to_string(o)?),
                    ])
                })
                .collect::<Result<_>>()?,
            metadata: vec![
                ("archive_version".into(), "2".into()),
                ("input_sha256".into(), self.input_sha256.clone()),
            ],
        };
        parquet::write_with_statistics(&table, opts, "jope-observation-archive/v2")
    }
}
