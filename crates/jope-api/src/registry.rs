//! Canonical operation/surface metadata for spec §4.2, §6.3 and §7.2.
//! This is data, not an HTTP dispatcher or handler implementation. Named JSON
//! exchanges bind the actual DTO schema; opaque byte streams have no JSON schema.
use crate::{decimal::DecimalU64, ids::DatasetId, *};
use schemars::{JsonSchema, Schema};
use serde::{de, Deserialize, Deserializer, Serialize};

/// Canonical decimal u64, including the exact 20-digit upper boundary.
const DECIMAL_U64_PATTERN: &str = "^(0|[1-9][0-9]{0,18}|1[0-7][0-9]{18}|18[0-3][0-9]{17}|184[0-3][0-9]{16}|1844[0-5][0-9]{15}|18446[0-6][0-9]{14}|184467[0-3][0-9]{13}|1844674[0-3][0-9]{12}|184467440[0-6][0-9]{10}|1844674407[0-2][0-9]{9}|18446744073[0-6][0-9]{8}|1844674407370[0-8][0-9]{6}|18446744073709[0-4][0-9]{5}|184467440737095[0-4][0-9]{4}|18446744073709550[0-9]{3}|18446744073709551[0-5][0-9]{2}|1844674407370955160[0-9]{1}|1844674407370955161[0-4]|18446744073709551615)$(?![\\s\\S])";

/// JSON registration is exclusive. Raw CSV uploads use the separate CSV media type.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(untagged, deny_unknown_fields)]
pub enum DatasetRegistration {
    Path {
        #[serde(deserialize_with = "nonempty")]
        #[schemars(length(min = 1))]
        path: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    CsvText {
        /// UTF-8 bytes, not Unicode character count, must fit in 1 MiB.
        #[serde(deserialize_with = "csv_text")]
        #[schemars(length(max = 1048576), extend("x-jope-max-utf8-bytes" = 1048576))]
        csv_text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
}
fn nonempty<'de, D: Deserializer<'de>>(d: D) -> Result<String, D::Error> {
    let text = String::deserialize(d)?;
    if text.is_empty() {
        return Err(de::Error::custom("path must not be empty"));
    }
    Ok(text)
}
fn csv_text<'de, D: Deserializer<'de>>(d: D) -> Result<String, D::Error> {
    let text = String::deserialize(d)?;
    if text.len() > 1024 * 1024 {
        return Err(de::Error::custom("csv_text exceeds 1 MiB of UTF-8 bytes"));
    }
    Ok(text)
}
fn page_items<'de, D: Deserializer<'de>, T: Deserialize<'de>>(d: D) -> Result<Vec<T>, D::Error> {
    let items = Vec::<T>::deserialize(d)?;
    if items.len() > 256 {
        return Err(de::Error::custom("list page exceeds 256 items"));
    }
    Ok(items)
}
macro_rules! list_page {
    ($name:ident, $field:ident, $item:ty) => {
        #[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
        #[serde(deny_unknown_fields)]
        pub struct $name {
            #[serde(deserialize_with = "page_items")]
            #[schemars(length(max = 256))]
            pub $field: Vec<$item>,
            /// Opaque continuation token; null denotes the end of the list.
            pub next_after: Option<String>,
        }
    };
}
list_page!(DatasetPage, datasets, DatasetProfile);
list_page!(RunPage, runs, RunSnapshot);
list_page!(AlgorithmPage, algorithms, Algorithm);

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DeleteReceipt {
    /// A missing or protected resource is a Problem, never a false success receipt.
    #[serde(deserialize_with = "deleted")]
    #[schemars(extend("const" = true))]
    pub deleted: bool,
}
fn deleted<'de, D: Deserializer<'de>>(d: D) -> Result<bool, D::Error> {
    if !bool::deserialize(d)? {
        return Err(de::Error::custom("deletion did not succeed"));
    }
    Ok(true)
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ObjectiveValidationRequest {
    pub spec: ObjectiveSpec,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task: Option<Task>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ObjectiveValidationIssue {
    pub code: crate::error::ErrorCode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub field: Option<String>,
    pub message: String,
}
/// Objective grammar resolution belongs to the runtime. Errors are structured
/// diagnostics; malformed request JSON is INVALID_INPUT at the HTTP boundary.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ObjectiveValidation {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolved: Option<ObjectiveSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub digest: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub direction: Option<Direction>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub required_columns: Vec<String>,
    pub errors: Vec<ObjectiveValidationIssue>,
}

#[derive(Clone, Debug, PartialEq, Serialize, JsonSchema)]
#[serde(tag = "action", rename_all = "snake_case", deny_unknown_fields)]
#[schemars(extend("allOf" = [{"if":{"properties":{"action":{"const":"set_effort"}}},"then":{"anyOf":[{"required":["effort"],"properties":{"effort":{"not":{"type":"null"}}}},{"required":["budget"],"properties":{"budget":{"not":{"type":"null"},"minProperties":1}}}]}}]))]
pub enum RunControlRequest {
    Pause {
        #[serde(skip_serializing_if = "Option::is_none")]
        #[schemars(regex(pattern = DECIMAL_U64_PATTERN))]
        expected_revision: Option<DecimalU64>,
    },
    Continue {
        #[serde(skip_serializing_if = "Option::is_none")]
        #[schemars(regex(pattern = DECIMAL_U64_PATTERN))]
        expected_revision: Option<DecimalU64>,
    },
    Cancel {
        #[serde(skip_serializing_if = "Option::is_none")]
        #[schemars(regex(pattern = DECIMAL_U64_PATTERN))]
        expected_revision: Option<DecimalU64>,
    },
    SetEffort {
        #[serde(skip_serializing_if = "Option::is_none")]
        effort: Option<Effort>,
        #[serde(skip_serializing_if = "Option::is_none")]
        budget: Option<BudgetUpdate>,
        #[serde(skip_serializing_if = "Option::is_none")]
        #[schemars(regex(pattern = DECIMAL_U64_PATTERN))]
        expected_revision: Option<DecimalU64>,
    },
}
// Remote derive decodes the same enum, then the cross-field check below rejects
// an empty update. The generated schema records that same constraint above.
#[derive(Deserialize)]
#[serde(
    remote = "RunControlRequest",
    tag = "action",
    rename_all = "snake_case",
    deny_unknown_fields
)]
enum ControlWire {
    Pause {
        expected_revision: Option<DecimalU64>,
    },
    Continue {
        expected_revision: Option<DecimalU64>,
    },
    Cancel {
        expected_revision: Option<DecimalU64>,
    },
    SetEffort {
        effort: Option<Effort>,
        budget: Option<BudgetUpdate>,
        expected_revision: Option<DecimalU64>,
    },
}
impl<'de> Deserialize<'de> for RunControlRequest {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let value = ControlWire::deserialize(d)?;
        if let Self::SetEffort {
            effort: None,
            budget,
            ..
        } = &value
        {
            if budget
                .as_ref()
                .is_none_or(|budget| budget.evaluations.is_none() && budget.wall_seconds.is_none())
            {
                return Err(de::Error::custom(
                    "set_effort requires effort or a nonempty budget update",
                ));
            }
        }
        Ok(value)
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ForkSeed {
    Same,
    New,
}

/// Missing fields inherit; explicit null clears that requested budget override.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct BudgetUpdate {
    #[serde(
        default,
        deserialize_with = "present_nullable",
        skip_serializing_if = "Option::is_none"
    )]
    #[schemars(regex(pattern = DECIMAL_U64_PATTERN))]
    pub evaluations: Option<Option<DecimalU64>>,
    #[serde(
        default,
        deserialize_with = "present_nullable",
        skip_serializing_if = "Option::is_none"
    )]
    #[schemars(regex(pattern = DECIMAL_U64_PATTERN))]
    pub wall_seconds: Option<Option<DecimalU64>>,
}

/// Only fields supplied here change the parent RunSpec. Seed selection, dry-run
/// and plan binding are separate mechanisms, so they cannot be overridden here.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RunOverrides {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dataset: Option<DatasetId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task: Option<Task>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ignore_columns: Option<Vec<String>>,
    /// Missing inherits; explicit null removes the parent's external test dataset.
    #[serde(
        default,
        deserialize_with = "present_nullable",
        skip_serializing_if = "Option::is_none"
    )]
    pub test_dataset: Option<Option<DatasetId>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub split: Option<SplitSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub objective: Option<ObjectiveSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effort: Option<Effort>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub budget: Option<BudgetUpdate>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub landmarks: Option<LandmarkMode>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub guidance: Option<Guidance>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub languages: Option<Vec<Language>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub require_all_languages: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device: Option<String>,
}
fn present_nullable<'de, D: Deserializer<'de>, T: Deserialize<'de>>(
    d: D,
) -> Result<Option<Option<T>>, D::Error> {
    Option::<T>::deserialize(d).map(Some)
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RunForkRequest {
    pub seed: ForkSeed,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub warm_start: Option<bool>,
    /// With warm_start, the service must also verify unchanged ProblemDigest.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub overrides: Option<RunOverrides>,
}

/// Durable events only. Cross-item ordering/cursor relations are checked by
/// Deserialize; JSON Schema cannot express those relational constraints.
#[derive(Clone, Debug, PartialEq, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct EventPage {
    #[schemars(length(max = 256))]
    pub events: Vec<PublicEvent>,
    #[schemars(regex(pattern = DECIMAL_U64_PATTERN))]
    pub next_after: DecimalU64,
}
impl<'de> Deserialize<'de> for EventPage {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Wire {
            #[serde(deserialize_with = "page_items")]
            events: Vec<PublicEvent>,
            next_after: DecimalU64,
        }
        let wire = Wire::deserialize(d)?;
        if wire
            .events
            .windows(2)
            .any(|pair| pair[0].sequence >= pair[1].sequence || pair[0].run_id != pair[1].run_id)
            || wire
                .events
                .last()
                .is_some_and(|event| event.sequence > wire.next_after)
        {
            return Err(de::Error::custom("event page must have one run, increasing sequences and a cursor at or beyond the last event"));
        }
        Ok(Self {
            events: wire.events,
            next_after: wire.next_after,
        })
    }
}
/// Opt-in internal evaluation data (§13.4), not a public event or raw data row.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FeedbackRecord {
    pub algorithm: String,
    pub features: Vec<f64>,
    pub scores: std::collections::BTreeMap<String, Option<f64>>,
    pub failure: Option<PublicFailure>,
    pub conditioning_tokens: [[f64; 32]; 10],
    pub conditioning_mask: [bool; 10],
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TrainingFeedback {
    pub records: Vec<FeedbackRecord>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct GcRequest {
    pub dry_run: bool,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct GcBlob {
    #[serde(deserialize_with = "sha256")]
    #[schemars(length(min = 64, max = 64), regex(pattern = "^[0-9a-f]{64}$"))]
    pub sha256: String,
    #[schemars(regex(pattern = DECIMAL_U64_PATTERN))]
    pub bytes: DecimalU64,
}
fn sha256<'de, D: Deserializer<'de>>(d: D) -> Result<String, D::Error> {
    let value = String::deserialize(d)?;
    if value.len() != 64
        || !value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(de::Error::custom(
            "blob SHA256 must be 64 lowercase hex digits",
        ));
    }
    Ok(value)
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct GcReport {
    pub dry_run: bool,
    pub blobs: Vec<GcBlob>,
    #[schemars(regex(pattern = DECIMAL_U64_PATTERN))]
    pub bytes: DecimalU64,
}

#[cfg(test)]
mod exchange_proofs;
mod operations;

pub use operations::*;
