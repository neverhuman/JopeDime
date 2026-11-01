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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HttpMethod {
    Get,
    Post,
    Delete,
}
impl HttpMethod {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Get => "GET",
            Self::Post => "POST",
            Self::Delete => "DELETE",
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CommandKind {
    Engine,
    Draft,
    Client,
    EngineAndDraft,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MediaType {
    Json,
    Csv,
    Artifact,
    EventStream,
}
impl MediaType {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Json => "application/json",
            Self::Csv => "text/csv",
            Self::Artifact => "application/octet-stream",
            Self::EventStream => "text/event-stream",
        }
    }
}

/// A real schema function when available, or an explicitly unmaterialized
/// exchange. OpenAPI consumers must handle the error, never substitute `{}`.
#[derive(Clone, Copy, Debug)]
pub struct PayloadTypeRef {
    pub name: &'static str,
    pub media_type: MediaType,
    pub json_schema: Option<fn() -> Schema>,
}
impl PayloadTypeRef {
    pub fn resolve_schema(&self) -> Result<Schema, &'static str> {
        self.json_schema.map(|f| f()).ok_or(self.name)
    }
}
fn schema<T: JsonSchema>() -> Schema {
    schemars::schema_for!(T)
}
const fn json_type<T: JsonSchema>(name: &'static str) -> PayloadTypeRef {
    PayloadTypeRef {
        name,
        media_type: MediaType::Json,
        json_schema: Some(schema::<T>),
    }
}
const fn stream_type<T: JsonSchema>(name: &'static str) -> PayloadTypeRef {
    PayloadTypeRef {
        name,
        media_type: MediaType::EventStream,
        json_schema: Some(schema::<T>),
    }
}
const CSV: PayloadTypeRef = PayloadTypeRef {
    name: "CSV",
    media_type: MediaType::Csv,
    json_schema: None,
};
const ARTIFACT: PayloadTypeRef = PayloadTypeRef {
    name: "ArtifactBytes",
    media_type: MediaType::Artifact,
    json_schema: None,
};

#[derive(Clone, Copy, Debug)]
pub struct TerminalCommand {
    pub name: &'static str,
    pub aliases: &'static [&'static str],
    pub arguments: &'static str,
    pub kind: CommandKind,
    pub engine_operations: &'static [u8],
    pub draft_plan_operation: Option<u8>,
    pub dry_run: bool,
}
#[derive(Clone, Copy, Debug)]
pub struct Operation {
    pub id: u8,
    pub name: &'static str,
    pub method: HttpMethod,
    /// Optional suffixes in §4.2 are expanded into concrete route templates.
    pub paths: &'static [&'static str],
    pub mcp_tool: Option<&'static str>,
    /// Subcommand paths exclude the executable name. Flags are owned by adapters.
    pub cli_subcommands: &'static [&'static str],
    /// HTTP body types, not path/query parameters or MCP wrapper envelopes.
    pub request_types: &'static [PayloadTypeRef],
    pub response_types: &'static [PayloadTypeRef],
    pub mutating: bool,
    pub requires_idempotency_key: bool,
    pub terminal_commands: &'static [TerminalCommand],
}
macro_rules! command {
    ($name:expr,$aliases:expr,$args:expr,$kind:ident,$ops:expr,$draft:expr,$dry:expr) => {
        TerminalCommand {
            name: $name,
            aliases: $aliases,
            arguments: $args,
            kind: CommandKind::$kind,
            engine_operations: $ops,
            draft_plan_operation: $draft,
            dry_run: $dry,
        }
    };
}
/// The sole engine-operation table. Read-only POST objective validation remains
/// distinct from mutations. Dry-run variants retain operation-level key safety.
pub static OPERATIONS: &[Operation] = &[
    Operation {
        id: 1,
        name: "system_info",
        method: HttpMethod::Get,
        paths: &["/v1/system"],
        mcp_tool: Some("system_info"),
        cli_subcommands: &["doctor"],
        request_types: &[],
        response_types: &[json_type::<crate::SystemInfo>("SystemInfo")],
        mutating: false,
        requires_idempotency_key: false,
        terminal_commands: &[
            command!("\\doctor", &[], "", Engine, &[1], None, false),
            command!("\\metrics", &[], "[--detailed]", Engine, &[1], None, false),
        ],
    },
    Operation {
        id: 2,
        name: "dataset_register",
        method: HttpMethod::Post,
        paths: &["/v1/datasets"],
        mcp_tool: Some("dataset_register"),
        cli_subcommands: &["dataset add"],
        request_types: &[json_type::<DatasetRegistration>("DatasetRegistration"), CSV],
        response_types: &[json_type::<crate::DatasetProfile>("DatasetProfile")],
        mutating: true,
        requires_idempotency_key: true,
        terminal_commands: &[command!(
            "\\open",
            &[],
            "PATH [--name N]",
            Engine,
            &[2],
            None,
            false
        )],
    },
    Operation {
        id: 3,
        name: "dataset_inspect",
        method: HttpMethod::Get,
        paths: &["/v1/datasets", "/v1/datasets/{id}"],
        mcp_tool: Some("dataset_inspect"),
        cli_subcommands: &["dataset list", "dataset show"],
        request_types: &[],
        response_types: &[
            json_type::<DatasetPage>("DatasetPage"),
            json_type::<crate::DatasetProfile>("DatasetProfile"),
        ],
        mutating: false,
        requires_idempotency_key: false,
        terminal_commands: &[],
    },
    Operation {
        id: 4,
        name: "dataset_delete",
        method: HttpMethod::Delete,
        paths: &["/v1/datasets/{id}"],
        mcp_tool: None,
        cli_subcommands: &["dataset delete"],
        request_types: &[],
        response_types: &[json_type::<DeleteReceipt>("DeleteReceipt")],
        mutating: true,
        requires_idempotency_key: true,
        terminal_commands: &[],
    },
    Operation {
        id: 5,
        name: "objective_validate",
        method: HttpMethod::Post,
        paths: &["/v1/objectives/validate"],
        mcp_tool: Some("objective_validate"),
        cli_subcommands: &["objective validate"],
        request_types: &[json_type::<ObjectiveValidationRequest>(
            "ObjectiveValidationRequest",
        )],
        response_types: &[json_type::<ObjectiveValidation>("ObjectiveValidation")],
        mutating: false,
        requires_idempotency_key: false,
        terminal_commands: &[command!(
            "\\objective",
            &["\\fitness"],
            "NAME | NAME:k=v | @file.json | {json}",
            EngineAndDraft,
            &[5],
            Some(6),
            false
        )],
    },
    Operation {
        id: 6,
        name: "run_start",
        method: HttpMethod::Post,
        paths: &["/v1/runs"],
        mcp_tool: Some("run_start"),
        cli_subcommands: &["run"],
        request_types: &[json_type::<crate::RunSpec>("RunSpec")],
        response_types: &[
            json_type::<crate::Plan>("Plan"),
            json_type::<crate::RunSnapshot>("RunSnapshot"),
        ],
        mutating: true,
        requires_idempotency_key: true,
        terminal_commands: &[
            command!("\\plan", &[], "", Engine, &[6], None, true),
            command!("\\run", &[], "[--seed S]", Engine, &[6], None, false),
        ],
    },
    Operation {
        id: 7,
        name: "run_status",
        method: HttpMethod::Get,
        paths: &["/v1/runs", "/v1/runs/{id}"],
        mcp_tool: Some("run_status"),
        cli_subcommands: &["runs", "status"],
        request_types: &[],
        response_types: &[
            json_type::<RunPage>("RunPage"),
            json_type::<crate::RunSnapshot>("RunSnapshot"),
        ],
        mutating: false,
        requires_idempotency_key: false,
        terminal_commands: &[
            command!("\\status", &[], "[RUN]", Engine, &[7], None, false),
            command!("\\runs", &[], "[--all]", Engine, &[7], None, false),
            command!("\\attach", &[], "RUN", Engine, &[7, 11], None, false),
        ],
    },
    Operation {
        id: 8,
        name: "run_control",
        method: HttpMethod::Post,
        paths: &["/v1/runs/{id}/control"],
        mcp_tool: Some("run_control"),
        cli_subcommands: &["control"],
        request_types: &[json_type::<RunControlRequest>("RunControlRequest")],
        response_types: &[json_type::<crate::RunSnapshot>("RunSnapshot")],
        mutating: true,
        requires_idempotency_key: true,
        terminal_commands: &[
            command!("\\pause", &[], "", Engine, &[8], None, false),
            command!("\\continue", &[], "", Engine, &[8], None, false),
            command!("\\cancel", &[], "", Engine, &[8], None, false),
            command!(
                "\\effort",
                &[],
                "fast|medium|high|ultra [--evals N] [--time DURATION]",
                EngineAndDraft,
                &[8],
                Some(6),
                false
            ),
        ],
    },
    Operation {
        id: 9,
        name: "run_fork",
        method: HttpMethod::Post,
        paths: &["/v1/runs/{id}/fork"],
        mcp_tool: Some("run_fork"),
        cli_subcommands: &["restart"],
        request_types: &[json_type::<RunForkRequest>("RunForkRequest")],
        response_types: &[json_type::<crate::RunSnapshot>("RunSnapshot")],
        mutating: true,
        requires_idempotency_key: true,
        terminal_commands: &[command!(
            "\\restart",
            &[],
            "[--same-seed|--new-seed] [--warm] [overrides…]",
            Engine,
            &[9],
            None,
            false
        )],
    },
    Operation {
        id: 10,
        name: "run_delete",
        method: HttpMethod::Delete,
        paths: &["/v1/runs/{id}"],
        mcp_tool: None,
        cli_subcommands: &["runs delete"],
        request_types: &[],
        response_types: &[json_type::<DeleteReceipt>("DeleteReceipt")],
        mutating: true,
        requires_idempotency_key: true,
        terminal_commands: &[],
    },
    Operation {
        id: 11,
        name: "run_events",
        method: HttpMethod::Get,
        paths: &["/v1/runs/{id}/events"],
        mcp_tool: Some("run_events"),
        cli_subcommands: &["events"],
        request_types: &[],
        response_types: &[
            json_type::<EventPage>("EventPage"),
            stream_type::<crate::PublicEvent>("PublicEvent"),
            stream_type::<crate::TelemetryEvent>("TelemetryEvent"),
        ],
        mutating: false,
        requires_idempotency_key: false,
        terminal_commands: &[],
    },
    Operation {
        id: 12,
        name: "algorithm_get",
        method: HttpMethod::Get,
        paths: &["/v1/runs/{id}/algorithms", "/v1/runs/{id}/algorithms/{alg}"],
        mcp_tool: Some("algorithm_get"),
        cli_subcommands: &["code"],
        request_types: &[],
        response_types: &[
            json_type::<AlgorithmPage>("AlgorithmPage"),
            json_type::<crate::Algorithm>("Algorithm"),
        ],
        mutating: false,
        requires_idempotency_key: false,
        terminal_commands: &[
            command!("\\code", &[], "[LANG] [ALG]", Engine, &[12], None, false),
            command!("\\diff", &[], "[ALG_A] [ALG_B]", Engine, &[12], None, false),
        ],
    },
    Operation {
        id: 13,
        name: "model_export",
        method: HttpMethod::Post,
        paths: &["/v1/runs/{id}/exports"],
        mcp_tool: Some("model_export"),
        cli_subcommands: &["export"],
        request_types: &[json_type::<crate::ExportRequest>("ExportRequest")],
        response_types: &[json_type::<crate::ExportManifest>("ExportManifest")],
        mutating: true,
        requires_idempotency_key: true,
        terminal_commands: &[command!(
            "\\export",
            &[],
            "[--kind bundle|model|source] [--lang …] [--alg ALG] --out DIR [--force]",
            Engine,
            &[13, 14],
            None,
            false
        )],
    },
    Operation {
        id: 14,
        name: "artifact_download",
        method: HttpMethod::Get,
        paths: &["/v1/artifacts/{id}"],
        mcp_tool: None,
        cli_subcommands: &[],
        request_types: &[],
        response_types: &[ARTIFACT],
        mutating: false,
        requires_idempotency_key: false,
        terminal_commands: &[],
    },
    Operation {
        id: 15,
        name: "model_predict",
        method: HttpMethod::Post,
        paths: &["/v1/predictions"],
        mcp_tool: Some("model_predict"),
        cli_subcommands: &["predict"],
        request_types: &[json_type::<crate::PredictRequest>("PredictRequest")],
        response_types: &[CSV],
        mutating: true,
        requires_idempotency_key: true,
        terminal_commands: &[command!(
            "\\predict",
            &[],
            "--input PATH --output PATH [--model ALG|PATH]",
            Engine,
            &[15],
            None,
            false
        )],
    },
    Operation {
        id: 16,
        name: "feedback_export",
        method: HttpMethod::Get,
        paths: &["/v1/runs/{id}/feedback"],
        mcp_tool: None,
        cli_subcommands: &["runs export-feedback"],
        request_types: &[],
        response_types: &[json_type::<TrainingFeedback>("TrainingFeedback")],
        mutating: false,
        requires_idempotency_key: false,
        terminal_commands: &[],
    },
    Operation {
        id: 17,
        name: "store_gc",
        method: HttpMethod::Post,
        paths: &["/v1/store/gc"],
        mcp_tool: None,
        cli_subcommands: &["gc"],
        request_types: &[json_type::<GcRequest>("GcRequest")],
        response_types: &[json_type::<GcReport>("GcReport")],
        mutating: true,
        requires_idempotency_key: true,
        terminal_commands: &[],
    },
];
/// These six commands are deliberately not separate engine operations (§4.2).
pub static LOCAL_COMMANDS: &[TerminalCommand] = &[
    command!("\\help", &[], "[command]", Client, &[], None, false),
    command!(
        "\\target",
        &[],
        "COLUMN [--task regression|binary|multiclass] [--ignore COL,…] [--group COL] [--time COL]",
        Draft,
        &[],
        Some(6),
        false
    ),
    command!("\\landmarks", &[], "full|none", Draft, &[], Some(6), false),
    command!(
        "\\lang",
        &["\\language"],
        "rust,python,javascript[,typescript] [--best-effort]",
        Draft,
        &[],
        Some(6),
        false
    ),
    command!(
        "\\reasoning",
        &[],
        "off|concise|detailed",
        Client,
        &[],
        None,
        false
    ),
    command!("\\quit", &[], "[--pause]", Client, &[], None, false),
];
pub fn operation(id: u8) -> Option<&'static Operation> {
    OPERATIONS.iter().find(|op| op.id == id)
}
pub fn operation_for_tool(name: &str) -> Option<&'static Operation> {
    OPERATIONS.iter().find(|op| op.mcp_tool == Some(name))
}
pub fn terminal_commands() -> impl Iterator<Item = &'static TerminalCommand> {
    OPERATIONS
        .iter()
        .flat_map(|op| op.terminal_commands)
        .chain(LOCAL_COMMANDS)
}
/// Backslash is canonical; slash is a silent prefix alias. Names are case-sensitive.
pub fn terminal_command(name: &str) -> Option<&'static TerminalCommand> {
    let bare = name.strip_prefix('\\').or_else(|| name.strip_prefix('/'))?;
    terminal_commands().find(|c| &c.name[1..] == bare || c.aliases.iter().any(|a| &a[1..] == bare))
}

#[cfg(test)]
mod exchange_proofs {
    use super::*;
    use serde::{de::DeserializeOwned, Serialize};
    use serde_json::{json, Value};

    fn exchange_snapshot_path() -> std::path::PathBuf {
        std::env::var_os("JOPE_EXCHANGE_SNAPSHOT_OUTPUT")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("tests/snapshots/exchanges.json")
            })
    }
    use std::collections::BTreeMap;

    fn positives() -> BTreeMap<&'static str, Vec<Value>> {
        let profile = json!({"dataset_id":"ds_00112233445566778899aabb","name":"training fixture","bytes":"42","rows":"3","columns":[],"target_suggestions":[],"warnings":[]});
        let snapshot: Value =
            serde_json::from_str(include_str!("../tests/fixtures/run_snapshot.json")).unwrap();
        let event: Value =
            serde_json::from_str(include_str!("../tests/fixtures/reasoning_event.json")).unwrap();
        let algorithm = json!({"algorithm_id":"alg_0001","summary":"Public fitted algorithm","metric":"rmse","cv_value":0.25,"model_artifact":null,"model_digest":"sha256:model","conditioning_digest":"sha256:conditioning","bundle_digest":"sha256:bundle","revision":"1","source":[]});
        let feedback = json!({"algorithm":"opt-in evaluation description","features":[0.5],"scores":{"rmse":0.25,"test":null},"failure":null,"conditioning_tokens":vec![vec![0.0;32];10],"conditioning_mask":vec![true;10]});
        BTreeMap::from([
            (
                "DatasetRegistration",
                vec![
                    json!({"path":"data/train.csv","name":"training"}),
                    json!({"csv_text":"x,y\n1,2\n"}),
                ],
            ),
            (
                "DatasetPage",
                vec![json!({"datasets":[profile],"next_after":null})],
            ),
            (
                "RunPage",
                vec![json!({"runs":[snapshot],"next_after":"opaque-continuation"})],
            ),
            (
                "AlgorithmPage",
                vec![json!({"algorithms":[algorithm],"next_after":null})],
            ),
            ("DeleteReceipt", vec![json!({"deleted":true})]),
            (
                "ObjectiveValidationRequest",
                vec![
                    json!({"spec":"rmse","task":"regression"}),
                    json!({"spec":{"name":"weighted_error","column":"weight"}}),
                ],
            ),
            (
                "ObjectiveValidation",
                vec![
                    json!({"resolved":{"name":"rmse"},"digest":"sha256:reviewed-objective","direction":"minimize","required_columns":["target"],"errors":[]}),
                    json!({"errors":[{"code":"INVALID_INPUT","field":"spec","message":"Unknown objective"}]}),
                ],
            ),
            (
                "RunControlRequest",
                vec![
                    json!({"action":"pause"}),
                    json!({"action":"continue","expected_revision":"7"}),
                    json!({"action":"cancel"}),
                    json!({"action":"set_effort","effort":"high"}),
                    json!({"action":"set_effort","budget":{"evaluations":"20"},"expected_revision":"9007199254740993"}),
                ],
            ),
            (
                "RunForkRequest",
                vec![
                    json!({"seed":"same"}),
                    json!({"seed":"new","warm_start":true,"overrides":{"effort":"fast","test_dataset":null,"budget":{"wall_seconds":null}}}),
                ],
            ),
            (
                "EventPage",
                vec![
                    json!({"events":[event.clone()],"next_after":event["sequence"]}),
                    json!({"events":[],"next_after":"0"}),
                ],
            ),
            (
                "TrainingFeedback",
                vec![json!({"records":[]}), json!({"records":[feedback]})],
            ),
            (
                "GcRequest",
                vec![json!({"dry_run":true}), json!({"dry_run":false})],
            ),
            (
                "GcReport",
                vec![
                    json!({"dry_run":true,"blobs":[{"sha256":"a".repeat(64),"bytes":"9007199254740993"}],"bytes":"9007199254740993"}),
                ],
            ),
        ])
    }
    fn negatives() -> BTreeMap<&'static str, Vec<Value>> {
        BTreeMap::from([
            (
                "DatasetRegistration",
                vec![
                    json!({}),
                    json!({"path":"a","csv_text":"b"}),
                    json!({"path":""}),
                ],
            ),
            (
                "DatasetPage",
                vec![json!({"datasets":{},"next_after":null})],
            ),
            ("RunPage", vec![json!({"runs":{},"next_after":null})]),
            (
                "AlgorithmPage",
                vec![json!({"algorithms":{},"next_after":null})],
            ),
            ("DeleteReceipt", vec![json!({"deleted":false})]),
            (
                "ObjectiveValidationRequest",
                vec![
                    json!({"spec":false}),
                    json!({"spec":"rmse","task":"unknown"}),
                ],
            ),
            (
                "ObjectiveValidation",
                vec![
                    json!({"errors":"bad"}),
                    json!({"errors":[{"code":"NEW_UNKNOWN_CODE","message":"bad"}]}),
                ],
            ),
            (
                "RunControlRequest",
                vec![
                    json!({"action":"stop"}),
                    json!({"action":"pause","effort":"fast"}),
                    json!({"action":"set_effort"}),
                    json!({"action":"set_effort","effort":null,"budget":null}),
                    json!({"action":"set_effort","budget":{}}),
                ],
            ),
            (
                "RunForkRequest",
                vec![
                    json!({"seed":"invalid"}),
                    json!({"seed":"same","overrides":{"seed":"5"}}),
                    json!({"seed":"same","overrides":{"dry_run":true}}),
                ],
            ),
            (
                "EventPage",
                vec![
                    json!({"events":[],"next_after":7}),
                    json!({"events":[],"next_after":"01"}),
                ],
            ),
            (
                "TrainingFeedback",
                vec![json!({"records":[],"rows":[[1,2]]})],
            ),
            (
                "GcRequest",
                vec![
                    json!({"dry_run":"true"}),
                    json!({"dry_run":true,"force":true}),
                ],
            ),
            (
                "GcReport",
                vec![
                    json!({"dry_run":true,"blobs":[],"bytes":42}),
                    json!({"dry_run":true,"blobs":[{"sha256":"A".repeat(64),"bytes":"1"}],"bytes":"1"}),
                ],
            ),
        ])
    }
    fn document() -> Value {
        let mut schemas = BTreeMap::new();
        for operation in OPERATIONS {
            for exchange in operation
                .request_types
                .iter()
                .chain(operation.response_types)
            {
                if let Some(schema) = exchange.json_schema {
                    let value = serde_json::to_value(schema()).unwrap();
                    if let Some(previous) = schemas.insert(exchange.name, value.clone()) {
                        assert_eq!(previous, value);
                    }
                }
            }
        }
        json!({"schemas":schemas,"positive":positives(),"negative":negatives()})
    }
    fn round_trip<T: DeserializeOwned + Serialize>(value: &Value) {
        let dto: T = serde_json::from_value(value.clone()).unwrap();
        assert_eq!(serde_json::to_value(dto).unwrap(), *value);
    }
    fn typed_cases<T: DeserializeOwned + Serialize>(name: &str) {
        for value in &positives()[name] {
            round_trip::<T>(value);
        }
        for value in &negatives()[name] {
            assert!(
                serde_json::from_value::<T>(value.clone()).is_err(),
                "{name} accepted {value}"
            );
        }
    }
    #[test]
    fn all_thirteen_exchange_examples_and_refusals_round_trip() {
        typed_cases::<DatasetRegistration>("DatasetRegistration");
        typed_cases::<DatasetPage>("DatasetPage");
        typed_cases::<RunPage>("RunPage");
        typed_cases::<AlgorithmPage>("AlgorithmPage");
        typed_cases::<DeleteReceipt>("DeleteReceipt");
        typed_cases::<ObjectiveValidationRequest>("ObjectiveValidationRequest");
        typed_cases::<ObjectiveValidation>("ObjectiveValidation");
        typed_cases::<RunControlRequest>("RunControlRequest");
        typed_cases::<RunForkRequest>("RunForkRequest");
        typed_cases::<EventPage>("EventPage");
        typed_cases::<TrainingFeedback>("TrainingFeedback");
        typed_cases::<GcRequest>("GcRequest");
        typed_cases::<GcReport>("GcReport");
    }
    #[test]
    fn csv_limit_counts_utf8_bytes_at_the_inclusive_boundary() {
        for (text, valid) in [
            ("x".repeat(1024 * 1024), true),
            ("x".repeat(1024 * 1024 + 1), false),
            ("é".repeat(512 * 1024), true),
            ("é".repeat(512 * 1024 + 1), false),
        ] {
            assert_eq!(
                serde_json::from_value::<DatasetRegistration>(json!({"csv_text":text})).is_ok(),
                valid
            );
        }
    }
    #[test]
    fn list_pages_and_event_pages_enforce_256_and_cursor_order() {
        let samples = positives();
        for (name, key) in [
            ("DatasetPage", "datasets"),
            ("RunPage", "runs"),
            ("AlgorithmPage", "algorithms"),
            ("EventPage", "events"),
        ] {
            let item = samples[name][0][key][0].clone();
            for count in [256, 257] {
                let mut page = samples[name][0].clone();
                page[key] = json!(vec![item.clone(); count]);
                if name == "EventPage" {
                    for (index, event) in page[key].as_array_mut().unwrap().iter_mut().enumerate() {
                        event["sequence"] = json!((index + 1).to_string());
                    }
                    page["next_after"] = json!(count.to_string());
                }
                let valid = match name {
                    "DatasetPage" => serde_json::from_value::<DatasetPage>(page).is_ok(),
                    "RunPage" => serde_json::from_value::<RunPage>(page).is_ok(),
                    "AlgorithmPage" => serde_json::from_value::<AlgorithmPage>(page).is_ok(),
                    _ => serde_json::from_value::<EventPage>(page).is_ok(),
                };
                assert_eq!(valid, count == 256, "{name}/{count}");
            }
        }
        let event = samples["EventPage"][0]["events"][0].clone();
        assert!(serde_json::from_value::<EventPage>(
            json!({"events":[event.clone(),event.clone()],"next_after":event["sequence"]})
        )
        .is_err());
        assert!(
            serde_json::from_value::<EventPage>(json!({"events":[event],"next_after":"0"}))
                .is_err()
        );
    }
    #[test]
    fn fork_patch_preserves_omission_null_and_partial_budgets() {
        let missing: RunOverrides = serde_json::from_value(json!({})).unwrap();
        let clear: RunOverrides = serde_json::from_value(json!({"test_dataset":null})).unwrap();
        assert_eq!(missing.test_dataset, None);
        assert_eq!(clear.test_dataset, Some(None));
        assert_eq!(serde_json::to_value(missing).unwrap(), json!({}));
        assert_eq!(
            serde_json::to_value(clear).unwrap(),
            json!({"test_dataset":null})
        );
        round_trip::<RunControlRequest>(
            &json!({"action":"set_effort","budget":{"wall_seconds":null}}),
        );
    }
    #[test]
    fn feedback_has_fixed_conditioning_and_no_raw_rows() {
        let valid = positives()["TrainingFeedback"][1].clone();
        for field in ["conditioning_tokens", "conditioning_mask"] {
            let mut bad = valid.clone();
            bad["records"][0][field].as_array_mut().unwrap().pop();
            assert!(serde_json::from_value::<TrainingFeedback>(bad).is_err());
        }
        let mut raw = valid;
        raw["records"][0]["rows"] = json!([[1, 2]]);
        assert!(serde_json::from_value::<TrainingFeedback>(raw).is_err());
    }
    #[test]
    fn generated_exchange_schemas_and_examples_are_current() {
        let path = exchange_snapshot_path();
        let expected: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        assert_eq!(
            document(),
            expected,
            "regenerate through the explicit owning proof test"
        );
    }
    #[test]
    #[ignore = "explicit owning generator for the exchange proof snapshot"]
    fn regenerate_exchange_proofs() {
        let path = exchange_snapshot_path();
        std::fs::write(
            path,
            format!("{}\n", serde_json::to_string_pretty(&document()).unwrap()),
        )
        .unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    #[test]
    fn every_json_exchange_has_a_materialized_schema() {
        let mut names = HashSet::new();
        for op in OPERATIONS {
            for ty in op.request_types.iter().chain(op.response_types) {
                if ty.media_type == MediaType::Json {
                    let schema = ty
                        .resolve_schema()
                        .unwrap_or_else(|name| panic!("unresolved JSON exchange: {name}"));
                    assert!(!schema.as_object().is_some_and(|value| value.is_empty()));
                    names.insert(ty.name);
                }
            }
        }
        assert!(names.contains("TrainingFeedback"));
    }
    #[test]
    fn exact_operations_tools_and_commands_match_spec_inventory() {
        let names = [
            "system_info",
            "dataset_register",
            "dataset_inspect",
            "dataset_delete",
            "objective_validate",
            "run_start",
            "run_status",
            "run_control",
            "run_fork",
            "run_delete",
            "run_events",
            "algorithm_get",
            "model_export",
            "artifact_download",
            "model_predict",
            "feedback_export",
            "store_gc",
        ];
        assert_eq!(OPERATIONS.len(), 17);
        for (index, (op, name)) in OPERATIONS.iter().zip(names).enumerate() {
            assert_eq!(op.id as usize, index + 1);
            assert_eq!(op.name, name);
        }
        let tools = [
            "system_info",
            "dataset_register",
            "dataset_inspect",
            "objective_validate",
            "run_start",
            "run_status",
            "run_control",
            "run_fork",
            "run_events",
            "algorithm_get",
            "model_export",
            "model_predict",
        ];
        assert_eq!(
            OPERATIONS
                .iter()
                .filter_map(|o| o.mcp_tool)
                .collect::<Vec<_>>(),
            tools
        );
        let mut actual = terminal_commands().map(|c| c.name).collect::<Vec<_>>();
        actual.sort_unstable();
        let mut expected = vec![
            "\\help",
            "\\open",
            "\\target",
            "\\objective",
            "\\effort",
            "\\landmarks",
            "\\lang",
            "\\plan",
            "\\run",
            "\\pause",
            "\\continue",
            "\\cancel",
            "\\restart",
            "\\status",
            "\\metrics",
            "\\reasoning",
            "\\code",
            "\\diff",
            "\\export",
            "\\predict",
            "\\runs",
            "\\attach",
            "\\doctor",
            "\\quit",
        ];
        expected.sort_unstable();
        assert_eq!(actual, expected);
    }
    #[test]
    fn api08_uniqueness_mutation_keys_and_unexposed_admin_operations() {
        let mut ids = HashSet::new();
        let mut names = HashSet::new();
        let mut tools = HashSet::new();
        let mut routes = HashSet::new();
        let mut cli = HashSet::new();
        for op in OPERATIONS {
            assert!(ids.insert(op.id));
            assert!(names.insert(op.name));
            if let Some(tool) = op.mcp_tool {
                assert!(tools.insert(tool));
            }
            for path in op.paths {
                assert!(routes.insert((op.method, *path)));
                assert!(path.starts_with("/v1/"));
            }
            for command in op.cli_subcommands {
                assert!(cli.insert(*command));
            }
            assert!(!op.mutating || op.requires_idempotency_key);
            if [4, 10, 16, 17].contains(&op.id) {
                assert!(op.mcp_tool.is_none());
            }
        }
        let mut command_names = HashSet::new();
        for c in terminal_commands() {
            assert!(command_names.insert(c.name));
            for alias in c.aliases {
                assert!(command_names.insert(*alias));
            }
        }
        assert_eq!(command_names.len(), 26);
    }
    #[test]
    fn api08_engine_mappings_aliases_and_context_roles() {
        for c in terminal_commands() {
            for id in c.engine_operations {
                assert!(operation(*id).is_some());
            }
            if let Some(id) = c.draft_plan_operation {
                assert_eq!(id, 6);
                assert!(operation(id).is_some());
            }
            match c.kind {
                CommandKind::Client => {
                    assert!(c.engine_operations.is_empty());
                    assert!(c.draft_plan_operation.is_none());
                }
                CommandKind::Draft => {
                    assert!(c.engine_operations.is_empty());
                    assert_eq!(c.draft_plan_operation, Some(6));
                }
                CommandKind::Engine => {
                    assert!(!c.engine_operations.is_empty());
                    assert!(c.draft_plan_operation.is_none());
                }
                CommandKind::EngineAndDraft => {
                    assert!(!c.engine_operations.is_empty());
                    assert_eq!(c.draft_plan_operation, Some(6));
                }
            }
        }
        assert_eq!(terminal_command("\\fitness").unwrap().name, "\\objective");
        assert_eq!(terminal_command("/language").unwrap().name, "\\lang");
        assert_eq!(
            terminal_command("\\objective").unwrap().kind,
            CommandKind::EngineAndDraft
        );
        assert_eq!(
            terminal_command("\\effort").unwrap().kind,
            CommandKind::EngineAndDraft
        );
        assert_eq!(
            terminal_command("\\attach").unwrap().engine_operations,
            &[7, 11]
        );
        assert_eq!(
            terminal_command("\\export").unwrap().engine_operations,
            &[13, 14]
        );
        assert!(terminal_command("/plan").unwrap().dry_run);
        assert!(terminal_command("run").is_none());
        assert!(terminal_command("\\RUN").is_none());
        assert!(operation(0).is_none());
        assert!(operation_for_tool("store_gc").is_none());
    }
    #[test]
    fn http_paths_methods_and_key_semantics_are_not_guessed_from_method() {
        assert_eq!(
            operation(3).unwrap().paths,
            &["/v1/datasets", "/v1/datasets/{id}"]
        );
        assert_eq!(operation(7).unwrap().paths, &["/v1/runs", "/v1/runs/{id}"]);
        assert_eq!(
            operation(12).unwrap().paths,
            &["/v1/runs/{id}/algorithms", "/v1/runs/{id}/algorithms/{alg}"]
        );
        for id in [2, 6, 8, 9, 13, 15, 17] {
            assert_eq!(operation(id).unwrap().method, HttpMethod::Post);
            assert!(operation(id).unwrap().mutating);
        }
        assert_eq!(operation(5).unwrap().method, HttpMethod::Post);
        assert!(!operation(5).unwrap().mutating);
        assert!(!operation(5).unwrap().requires_idempotency_key);
        assert_eq!(operation(16).unwrap().method, HttpMethod::Get);
        assert!(!operation(16).unwrap().mutating);
        assert!(operation(14).unwrap().mcp_tool.is_none());
    }
    #[test]
    fn schema_bindings_and_opaque_media_are_explicit() {
        let mut bound = 0;
        let mut pending = 0;
        for op in OPERATIONS {
            for ty in op.request_types.iter().chain(op.response_types) {
                assert!(!ty.name.is_empty());
                if ty.json_schema.is_some() {
                    assert!(ty.resolve_schema().is_ok());
                    bound += 1;
                } else {
                    assert_eq!(ty.resolve_schema().unwrap_err(), ty.name);
                    if ty.media_type == MediaType::Json {
                        pending += 1;
                    }
                }
            }
        }
        assert!(bound > 0);
        assert_eq!(pending, 0);
        assert_eq!(operation(6).unwrap().request_types[0].name, "RunSpec");
        assert_eq!(
            operation(6)
                .unwrap()
                .response_types
                .iter()
                .map(|x| x.name)
                .collect::<Vec<_>>(),
            ["Plan", "RunSnapshot"]
        );
        assert_eq!(
            operation(11).unwrap().response_types[1].media_type,
            MediaType::EventStream
        );
        assert_eq!(
            operation(15).unwrap().response_types[0].media_type,
            MediaType::Csv
        );
    }
}
