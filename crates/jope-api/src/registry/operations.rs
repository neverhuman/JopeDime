use super::*;

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
mod tests;
