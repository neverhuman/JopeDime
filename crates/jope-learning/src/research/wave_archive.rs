//! Content-addressed local archive for typed wave telemetry. This module does not publish remotely.
use crate::canon::{canonical_json, sha256_hex};
use crate::retention::parquet::{self, Cell, ColumnSpec, ColumnType, Table};
use crate::state::State;
use anyhow::Context as _;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;

/// Schema identifier for  values.
pub const SCHEMA: &str = "jope.wave-evidence/v1";
/// Configured upper or lower bound for max events.
pub const MAX_EVENTS: usize = 16_384;
/// Configured upper or lower bound for max shard rows.
pub const MAX_SHARD_ROWS: usize = 512;
/// Configured upper or lower bound for max decoded bytes.
pub const MAX_DECODED_BYTES: usize = 8 * 1024 * 1024;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "state", content = "value", rename_all = "snake_case")]
/// `Known` is an enum for the known contract in this module: Content-addressed local archive for typed wave telemetry. This module does not publish remotely.
pub enum Known<T> {
    /// The `Known` case of `Known`.
    Known(T),
    /// The `Missing` case of `Known`.
    Missing {
        /// Explanation accompanying this outcome.
        reason: String,
    },
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
/// `Disposition` is an enum for the disposition contract in this module: Content-addressed local archive for typed wave telemetry. This module does not publish remotely.
pub enum Disposition {
    /// The `Success` case of `Disposition`.
    Success,
    /// The `RejectedChild` case of `Disposition`.
    RejectedChild,
    /// The `Aborted` case of `Disposition`.
    Aborted,
    /// The `TimeoutOnly` case of `Disposition`.
    TimeoutOnly,
    /// The `Inconclusive` case of `Disposition`.
    Inconclusive,
    /// The `InterruptedResumed` case of `Disposition`.
    InterruptedResumed,
    /// The `ZeroUpdate` case of `Disposition`.
    ZeroUpdate,
    /// The `HistoricalPartial` case of `Disposition`.
    HistoricalPartial,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
/// `Metric` is a record for the metric contract in this module: Content-addressed local archive for typed wave telemetry. This module does not publish remotely.
pub struct Metric {
    /// Measured or derived value carried by this result.
    pub value: Option<f64>,
    /// Unit recorded for this metric.
    pub unit: String,
    /// Shared denominator used to interpret probability numerators in this law.
    pub denominator: Option<u64>,
    /// Outcome or lifecycle status recorded for this value.
    pub status: String,
    /// Source field recorded for this metric.
    pub source_field: String,
    /// Optional missing reason value; `None` means it is not available.
    pub missing_reason: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
/// `Event` is a record for the event contract in this module: Content-addressed local archive for typed wave telemetry. This module does not publish remotely.
pub struct Event {
    /// Sequence recorded for this event.
    pub sequence: u64,
    /// Attempt recorded for this event.
    pub attempt: u32,
    /// Rank recorded for this event.
    pub rank: u32,
    /// Record kind or category represented by this value.
    pub kind: String,
    /// Mapping of fields values carried by this event.
    pub fields: BTreeMap<String, Known<String>>,
    /// Mapping of counters values carried by this event.
    pub counters: BTreeMap<String, Known<u64>>,
    /// Mapping of metrics values carried by this event.
    pub metrics: BTreeMap<String, Metric>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
/// `WaveInput` is a record for the wave input contract in this module: Content-addressed local archive for typed wave telemetry. This module does not publish remotely.
pub struct WaveInput {
    /// Identifier for the associated wave.
    pub wave_id: String,
    /// Version recorded for this wave input.
    pub version: String,
    /// Disposition recorded for this wave input.
    pub disposition: Disposition,
    /// New wave recorded for this wave input.
    pub new_wave: bool,
    /// Mapping of identity values carried by this wave input.
    pub identity: BTreeMap<String, Known<String>>,
    /// Collection of events values carried by this wave input.
    pub events: Vec<Event>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
/// `ShardRef` is a record for the shard ref contract in this module: Content-addressed local archive for typed wave telemetry. This module does not publish remotely.
pub struct ShardRef {
    /// Ordinal recorded for this shard ref.
    pub ordinal: u32,
    /// Sha-256 recorded for this shard ref.
    pub sha256: String,
    /// Blob key recorded for this shard ref.
    pub blob_key: String,
    /// Codec recorded for this shard ref.
    pub codec: String,
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// Number of rows represented by this value.
    pub rows: u64,
    /// Byte count for the associated decoded.
    pub decoded_bytes: u64,
    /// Byte count for the associated compressed.
    pub compressed_bytes: u64,
    /// SHA-256 digest identifying the associated member.
    pub member_sha256: String,
    /// Optional physical version value; `None` means it is not available.
    pub physical_version: Option<String>,
    /// First sequence recorded for this shard ref.
    pub first_sequence: u64,
    /// Last sequence recorded for this shard ref.
    pub last_sequence: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
/// `Manifest` is a record for the manifest contract in this module: Content-addressed local archive for typed wave telemetry. This module does not publish remotely.
pub struct Manifest {
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// Identifier for the associated wave.
    pub wave_id: String,
    /// Version recorded for this manifest.
    pub version: String,
    /// Disposition recorded for this manifest.
    pub disposition: Disposition,
    /// Archive status recorded for this manifest.
    pub archive_status: String,
    /// Capture status recorded for this manifest.
    pub capture_status: String,
    /// Mapping of identity values carried by this manifest.
    pub identity: BTreeMap<String, Known<String>>,
    /// Mapping of missing fields values carried by this manifest.
    pub missing_fields: BTreeMap<String, String>,
    /// Expected events recorded for this manifest.
    pub expected_events: u64,
    /// Collection of shards values carried by this manifest.
    pub shards: Vec<ShardRef>,
    /// SHA-256 digest identifying the associated manifest.
    pub manifest_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// `ArchiveReceipt` is a record for the archive receipt contract in this module: Content-addressed local archive for typed wave telemetry. This module does not publish remotely.
pub struct ArchiveReceipt {
    /// Manifest recorded for this archive receipt.
    pub manifest: Manifest,
    /// Manifest blob recorded for this archive receipt.
    pub manifest_blob: String,
    /// Byte count for the associated manifest.
    pub manifest_bytes: Vec<u8>,
}

/// Performs the archive operation and returns an error if it cannot be completed.
pub fn archive(state: &State, input: &WaveInput) -> anyhow::Result<ArchiveReceipt> {
    validate(input)?;
    let mut events = input.events.clone();
    events.sort_by_key(|e| (e.sequence, e.rank, e.attempt));
    let mut shards = Vec::new();
    for (ordinal, chunk) in events.chunks(MAX_SHARD_ROWS).enumerate() {
        let estimated_bytes = chunk.iter().try_fold(0usize, |sum, event| {
            sum.checked_add(estimated_event_bytes(event))
                .context("decoded shard size overflow")
        })?;
        anyhow::ensure!(
            estimated_bytes <= MAX_DECODED_BYTES,
            "estimated decoded shard exceeds limit"
        );
        let decoded = canonical_json(&chunk.to_vec())?;
        anyhow::ensure!(
            decoded.len() <= MAX_DECODED_BYTES,
            "decoded shard exceeds limit"
        );
        let rows = chunk
            .iter()
            .map(|event| {
                let json =
                    String::from_utf8(canonical_json(event)?).map_err(anyhow::Error::from)?;
                Ok(vec![
                    Cell::I64(i64::try_from(event.sequence)?),
                    Cell::I64(i64::from(event.rank)),
                    Cell::I64(i64::from(event.attempt)),
                    Cell::Str(json),
                ])
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let table = Table {
            columns: vec![
                ColumnSpec {
                    name: "sequence".into(),
                    ty: ColumnType::Int64,
                    required: true,
                },
                ColumnSpec {
                    name: "rank".into(),
                    ty: ColumnType::Int64,
                    required: true,
                },
                ColumnSpec {
                    name: "attempt".into(),
                    ty: ColumnType::Int64,
                    required: true,
                },
                ColumnSpec {
                    name: "event_json".into(),
                    ty: ColumnType::Utf8,
                    required: true,
                },
            ],
            rows,
            metadata: vec![
                ("schema".into(), SCHEMA.into()),
                ("wave_id".into(), input.wave_id.clone()),
                ("version".into(), input.version.clone()),
                ("ordinal".into(), ordinal.to_string()),
            ],
        };
        let bytes = parquet::write(&table, true, "jope-wave-evidence/v1")?;
        let digest = sha256_hex(&bytes);
        let key = state.put_blob(&bytes)?;
        shards.push(ShardRef {
            ordinal: ordinal as u32,
            sha256: digest.clone(),
            blob_key: format!("blobs/sha256/{key}"),
            codec: "parquet-zstd".into(),
            schema: SCHEMA.into(),
            rows: chunk.len() as u64,
            decoded_bytes: decoded.len() as u64,
            compressed_bytes: bytes.len() as u64,
            member_sha256: sha256_hex(&decoded),
            physical_version: None,
            first_sequence: chunk.first().unwrap().sequence,
            last_sequence: chunk.last().unwrap().sequence,
        });
    }
    let mut missing_fields = BTreeMap::new();
    for (key, state) in &input.identity {
        if let Known::Missing { reason } = state {
            missing_fields.insert(key.clone(), reason.clone());
        }
    }
    for event in &events {
        for (key, state) in &event.fields {
            if let Known::Missing { reason } = state {
                missing_fields.insert(format!("event:{}:{key}", event.sequence), reason.clone());
            }
        }
        for (key, state) in &event.counters {
            if let Known::Missing { reason } = state {
                missing_fields.insert(
                    format!("event:{}:counter:{key}", event.sequence),
                    reason.clone(),
                );
            }
        }
        for (key, metric) in &event.metrics {
            if let Some(reason) = &metric.missing_reason {
                missing_fields.insert(
                    format!("event:{}:metric:{key}", event.sequence),
                    reason.clone(),
                );
            }
        }
    }
    let historical_partial = !input.new_wave
        && (input.disposition == Disposition::HistoricalPartial || !missing_fields.is_empty());
    let archive_status = if historical_partial {
        "HISTORICAL_PARTIAL"
    } else {
        "LOCAL_COMMITTED_REMOTE_PENDING"
    };
    let capture_status = if historical_partial {
        "HISTORICAL_PARTIAL"
    } else if input.new_wave && !missing_fields.is_empty() {
        "INCOMPLETE_BLOCKED"
    } else {
        "COMPLETE"
    };
    let mut manifest = Manifest {
        schema: SCHEMA.into(),
        wave_id: input.wave_id.clone(),
        version: input.version.clone(),
        disposition: input.disposition,
        archive_status: archive_status.into(),
        capture_status: capture_status.into(),
        identity: input.identity.clone(),
        missing_fields,
        expected_events: events.len() as u64,
        shards,
        manifest_sha256: String::new(),
    };
    let payload = canonical_json(&manifest)?;
    manifest.manifest_sha256 = sha256_hex(&payload);
    let manifest_bytes = canonical_json(&manifest)?;
    // Manifest is added only after every immutable shard blob has been stored.
    let manifest_blob = state.put_blob(&manifest_bytes)?;
    Ok(ArchiveReceipt {
        manifest,
        manifest_blob,
        manifest_bytes,
    })
}

/// Performs the reconstruct operation and returns an error if it cannot be completed.
pub fn reconstruct(state: &State, manifest: &Manifest) -> anyhow::Result<Vec<Event>> {
    anyhow::ensure!(
        manifest.schema == SCHEMA,
        "unsupported wave evidence schema"
    );
    let mut unsigned = manifest.clone();
    let expected_manifest_hash = unsigned.manifest_sha256.clone();
    unsigned.manifest_sha256.clear();
    anyhow::ensure!(
        sha256_hex(&canonical_json(&unsigned)?) == expected_manifest_hash,
        "manifest digest mismatch"
    );
    let mut events = Vec::new();
    for (expected, shard) in manifest.shards.iter().enumerate() {
        anyhow::ensure!(
            shard.ordinal as usize == expected
                && shard.schema == SCHEMA
                && shard.codec == "parquet-zstd",
            "invalid shard ordering/identity"
        );
        anyhow::ensure!(
            shard.blob_key == format!("blobs/sha256/{}", shard.sha256),
            "shard blob key does not bind its digest"
        );
        let path = state.blob_path(&shard.sha256);
        let metadata = fs::symlink_metadata(&path)?;
        anyhow::ensure!(
            metadata.is_file() && !metadata.file_type().is_symlink(),
            "shard blob is not a regular file"
        );
        let bytes = fs::read(path)?;
        anyhow::ensure!(
            sha256_hex(&bytes) == shard.sha256 && bytes.len() as u64 == shard.compressed_bytes,
            "shard digest/length mismatch"
        );
        let table = parquet::read(&bytes)?;
        for (key, value) in [
            ("schema", SCHEMA.to_owned()),
            ("wave_id", manifest.wave_id.clone()),
            ("version", manifest.version.clone()),
            ("ordinal", expected.to_string()),
        ] {
            anyhow::ensure!(
                table.metadata.iter().any(|(k, v)| k == key && v == &value),
                "shard metadata mismatch for {key}"
            );
        }
        anyhow::ensure!(
            table.rows.len() as u64 == shard.rows,
            "shard row count mismatch"
        );
        let first_event = events.len();
        for row in table.rows {
            match row.get(3) {
                Some(Cell::Str(s)) => {
                    let event = serde_json::from_str::<Event>(s)?;
                    anyhow::ensure!(
                        row.first() == Some(&Cell::I64(i64::try_from(event.sequence)?))
                            && row.get(1) == Some(&Cell::I64(i64::from(event.rank)))
                            && row.get(2) == Some(&Cell::I64(i64::from(event.attempt))),
                        "typed row keys disagree with event payload"
                    );
                    events.push(event);
                }
                _ => anyhow::bail!("event row is malformed"),
            }
        }
        let chunk = &events[first_event..];
        anyhow::ensure!(
            chunk.first().map(|e| e.sequence) == Some(shard.first_sequence)
                && chunk.last().map(|e| e.sequence) == Some(shard.last_sequence),
            "shard sequence boundary mismatch"
        );
        let decoded = canonical_json(&chunk.to_vec())?;
        anyhow::ensure!(
            decoded.len() as u64 == shard.decoded_bytes
                && sha256_hex(&decoded) == shard.member_sha256
                && decoded.len() <= MAX_DECODED_BYTES,
            "decoded shard/member digest or length mismatch"
        );
    }
    events.sort_by_key(|e| (e.sequence, e.rank, e.attempt));
    anyhow::ensure!(
        events.len() as u64 == manifest.expected_events,
        "event census differs from manifest"
    );
    validate_event_order(&events)?;
    Ok(events)
}

fn validate(input: &WaveInput) -> anyhow::Result<()> {
    anyhow::ensure!(
        safe_segment(&input.wave_id) && !input.version.is_empty() && safe_segment(&input.version),
        "unsafe wave id/version"
    );
    anyhow::ensure!(
        input.events.len() <= MAX_EVENTS,
        "event count exceeds limit"
    );
    if input.new_wave {
        anyhow::ensure!(
            !input.events.is_empty(),
            "new-wave evidence has no producer events"
        );
        for key in [
            "source",
            "config",
            "grammar",
            "evaluator",
            "registry",
            "context",
            "support",
            "optimizer_recipe",
            "rng_sampler",
            "data_role",
        ] {
            anyhow::ensure!(
                matches!(input.identity.get(key), Some(Known::Known(value)) if !value.is_empty()),
                "new wave missing required identity {key}"
            );
        }
    }
    for event in &input.events {
        anyhow::ensure!(!event.kind.trim().is_empty(), "event kind is required");
        for metric in event.metrics.values() {
            anyhow::ensure!(
                !metric.unit.is_empty() && !metric.source_field.is_empty(),
                "metric unit/source are required"
            );
            if let Some(value) = metric.value {
                anyhow::ensure!(
                    value.is_finite() && metric.denominator.is_some_and(|n| n > 0),
                    "finite metric requires positive denominator"
                );
            } else {
                anyhow::ensure!(
                    metric
                        .missing_reason
                        .as_ref()
                        .is_some_and(|s| !s.trim().is_empty()),
                    "null metric requires missing reason"
                );
            }
        }
        if input.new_wave {
            for key in ["computed", "applied", "committed"] {
                anyhow::ensure!(
                    event.counters.contains_key(key),
                    "new-wave event lacks {key} counter"
                );
            }
        }
    }
    let mut sorted = input.events.clone();
    sorted.sort_by_key(|e| (e.sequence, e.rank, e.attempt));
    validate_event_order(&sorted)
}

fn validate_event_order(events: &[Event]) -> anyhow::Result<()> {
    let mut keys = BTreeSet::new();
    for event in events {
        anyhow::ensure!(
            keys.insert((event.sequence, event.rank, event.attempt)),
            "duplicate event identity"
        );
    }
    Ok(())
}
fn estimated_event_bytes(event: &Event) -> usize {
    let strings = event.kind.len()
        + event
            .fields
            .iter()
            .map(|(k, v)| {
                k.len()
                    + match v {
                        Known::Known(s) => s.len(),
                        Known::Missing { reason } => reason.len(),
                    }
            })
            .sum::<usize>()
        + event
            .counters
            .iter()
            .map(|(k, v)| {
                k.len()
                    + match v {
                        Known::Known(_) => 24,
                        Known::Missing { reason } => reason.len(),
                    }
            })
            .sum::<usize>()
        + event
            .metrics
            .iter()
            .map(|(k, m)| {
                k.len()
                    + m.unit.len()
                    + m.status.len()
                    + m.source_field.len()
                    + m.missing_reason.as_ref().map_or(0, String::len)
                    + 64
            })
            .sum::<usize>();
    // JSON escaping can expand each control character to six bytes; use a conservative bound.
    strings.saturating_mul(6).saturating_add(256)
}
fn safe_segment(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 128
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"-_.".contains(&b))
}
