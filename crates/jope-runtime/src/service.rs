//! Local JSON-RPC and MCP adapters over the same runtime API. No file/network access.
//!
//! One dispatcher serves two framings over JSON lines on stdin/stdout: plain JSON-RPC 2.0
//! (`jopedime api`, method = tool name) and the Model Context Protocol (`jopedime mcp`,
//! `initialize` / `tools/list` / `tools/call`). Both run exactly the Rust functions below, so a
//! score or a selection is identical whichever adapter asked for it.
use crate::{
    api, budget::Budget, fitness::Fitness, fitted::CreatedBy, limits::Limits, table,
    task::TaskChoice,
};
use anyhow::{ensure, Result};
use serde::Deserialize;
use serde_json::{json, Value};
use std::time::Duration;

pub const MAX_MESSAGE_BYTES: usize = 8 * 1024 * 1024;
/// The newest MCP revision this server speaks; `initialize` echoes a supported older request.
pub const MCP_VERSION: &str = "2025-11-25";
/// 2025-03-26 is left out on purpose: it obliges servers to accept JSON-RPC batches, which this
/// server refuses, so a client asking for it is offered the newest revision instead.
pub const MCP_SUPPORTED_VERSIONS: [&str; 3] = ["2025-11-25", "2025-06-18", "2024-11-05"];

const INSTRUCTIONS: &str = "Regression objectives for JopeDime. Call fitness_list to see the \
     built-in objectives (pass a name, `name:param=value`, or a JSON specification), \
     fitness_validate to check a custom specification and get its identity, fitness_evaluate to \
     score predictions, and regression_fit_predict to select, calibrate and refit native models \
     for an objective on inline CSV. Custom objectives are JSON row expressions over y, pred and \
     numeric training columns (op column), aggregated by mean/sum/max/quantile/tail_mean/\
     root_mean; weighted and composite forms combine them. Order-dependent objectives (returns, \
     drawdown, MASE) need options.validation = {kind: forward, gap: N} on chronological rows.";

/// A specification given as a JSON object, a catalog name or `name:key=value,…`.
pub fn fitness_from_value(value: &Value) -> Result<Fitness> {
    match value {
        Value::String(text) => Fitness::parse(text),
        Value::Object(_) => {
            let fitness: Fitness = serde_json::from_value(value.clone())?;
            fitness.validate()?;
            Ok(fitness)
        }
        _ => anyhow::bail!("fitness must be a JSON specification or a catalog name"),
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Evaluation {
    pub fitness: Value,
    pub targets: Vec<f64>,
    pub predictions: Vec<f64>,
    /// Auxiliary columns the specification reads, aligned with `targets`.
    #[serde(default)]
    pub columns: crate::fitness::Columns,
    /// Candidate resource readings for objectives that read `ResourceVar` values.
    #[serde(default)]
    pub resources: crate::fitness::Resources,
}
pub fn evaluate(request: Evaluation) -> Result<Value> {
    let fitness = fitness_from_value(&request.fitness)?;
    Ok(serde_json::to_value(fitness.evaluate_with_resources(
        &request.targets,
        &request.predictions,
        &request.columns,
        &request.resources,
    )?)?)
}

/// What a specification is, without scoring anything: its resolved form and identity.
pub fn describe(fitness: &Fitness) -> Result<Value> {
    Ok(json!({
        "schema": crate::fitness::FITNESS_VERSION,
        "name": fitness.name(),
        "direction": fitness.direction(),
        "fitness_sha256": fitness.identity()?,
        "columns": fitness.columns(),
        "resources": fitness.resources(),
        "requires_measured_resources": fitness.requires_measured_resources(),
        "requires_chronological_validation": fitness.requires_order(),
        "specification": fitness,
    }))
}

pub fn catalog() -> Value {
    use crate::fitness::ResourceVar;
    json!({"schema": crate::fitness::FITNESS_VERSION,
        "fitness": Fitness::catalog().into_iter().map(|entry| json!({
            "name": entry.name,
            "direction": entry.fitness.direction(),
            "requires_chronological_validation": entry.fitness.requires_order(),
            "description": entry.description,
            "specification": entry.fitness})).collect::<Vec<_>>(),
        "resources": ResourceVar::ALL.into_iter().map(|var| json!({
            "name": var,
            "direction": if var.maximized() { "maximize" } else { "minimize" },
            "measured": var.is_measured(),
            "description": var.description()})).collect::<Vec<_>>(),
        "custom": {
            "expression_operators": ["y", "pred", "constant", "column", "resource", "add", "sub",
                "mul", "div", "min", "max", "pow", "abs", "neg", "square", "sqrt", "ln", "log1p",
                "exp", "step", "clip", "where"],
            "aggregations": ["mean", "sum", "max", "quantile", "tail_mean", "root_mean"],
            "forms": ["resource", "custom", "weighted", "composite", "constrained"],
            "shorthand": "name:key=value,... overrides a built-in's numeric defaults, e.g. pinball:quantile=0.9"
        },
        "calibration": ["auto", "none", "shift", "affine"]})
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FitPredict {
    train_csv: String,
    input_csv: String,
    target: String,
    #[serde(default)]
    options: Value,
}
fn fit_predict(request: FitPredict, created_by: CreatedBy) -> Result<Value> {
    ensure!(
        request.train_csv.len() + request.input_csv.len() <= MAX_MESSAGE_BYTES,
        "CSV input is too large"
    );
    // Accept a catalog name or shorthand wherever a specification object is accepted.
    let mut options = match request.options {
        Value::Null => json!({}),
        other => other,
    };
    if let Some(spec) = options.get("fitness").filter(|v| !v.is_null()) {
        let fitness = fitness_from_value(spec)?;
        options["fitness"] = serde_json::to_value(fitness)?;
    }
    let options: api::FitOptions = serde_json::from_value(options)?;
    ensure!(
        (2..=20).contains(&options.folds),
        "service folds must be in 2..=20"
    );
    ensure!(
        options.candidates.len() <= 32,
        "service supports at most 32 candidate selectors"
    );
    let limits = Limits {
        max_input_bytes: MAX_MESSAGE_BYTES as u64,
        max_rows: 100_000,
        max_columns: 1000,
        ..Limits::default()
    };
    let auxiliary = options
        .fitness
        .as_ref()
        .map(Fitness::columns)
        .unwrap_or_default();
    let csv = table::parse_csv(request.train_csv.as_bytes(), &limits)?;
    let train = table::training_with_auxiliary(
        &csv,
        &request.target,
        TaskChoice::parse("regression").expect("known task"),
        &limits,
        &auxiliary,
    )?;
    let input = table::parse_csv(request.input_csv.as_bytes(), &limits)?;
    // Validate prediction input before paying for selection.
    let auxiliary: Vec<String> = auxiliary.into_iter().collect();
    let (rows, ignored_columns) =
        table::prediction_ignoring(&input, &train.schema, &train.target, &auxiliary)?;
    let budget = Budget::new(Some(Duration::from_secs(30)), 1);
    let outcome = api::fit(&train, &options, &budget, &csv.sha256, created_by)?;
    let predictions = api::predict(&outcome.fitted, &rows, &budget)?;
    Ok(
        json!({"schema":"jope.regression-fit-predict/v1", "model_header":outcome.fitted.header,
        "selection":outcome.selection,"predictions_csv":String::from_utf8(api::predictions_csv(&predictions)?)?,
        "ignored_columns":ignored_columns,"budget":budget.record()}),
    )
}

fn tools() -> Value {
    let definitions = crate::fitness::schema()["$defs"].clone();
    let fitness = json!({"oneOf":[{"$ref":"#/$defs/fitness"},
        {"type":"string","description":"A catalog name or name:key=value,... shorthand, e.g. pinball:quantile=0.9"}]});
    let read_only = json!({"readOnlyHint":true,"openWorldHint":false});
    json!({"tools":[
        {"name":"fitness_list","title":"List regression objectives",
            "description":"List built-in regression objectives with defaults, directions, descriptions and temporal requirements, plus the custom expression grammar.",
            "inputSchema":{"type":"object","properties":{},"additionalProperties":false},"annotations":read_only},
        {"name":"fitness_validate","title":"Validate an objective",
            "description":"Check a regression objective specification and return its resolved form, direction, auxiliary columns and sha256 identity.",
            "inputSchema":{"$defs":definitions,"type":"object","required":["fitness"],"additionalProperties":false,
            "properties":{"fitness":fitness}},"annotations":read_only},
        {"name":"fitness_evaluate","title":"Score predictions",
            "description":"Evaluate a regression objective on equal, nonempty target/prediction arrays. Supply columns for objectives that read auxiliary columns.",
            "inputSchema":{"$defs":definitions,"type":"object","required":["fitness","targets","predictions"],"additionalProperties":false,
            "properties":{"fitness":fitness,"targets":{"type":"array","items":{"type":"number"},"minItems":1},"predictions":{"type":"array","items":{"type":"number"},"minItems":1},
                "columns":{"type":"object","additionalProperties":{"type":"array","items":{"type":"number"}}}}},
            "annotations":read_only},
        {"name":"regression_fit_predict","title":"Fit for an objective and predict",
            "description":"Select native regression models by held-out objective, calibrate their outputs to it, refit the winner and predict an inline CSV. Columns the objective reads are taken from train_csv and never used as features. 30s cooperative selection budget, one CPU, no file access. Order-dependent objectives require options.validation={kind:forward,gap:N}; input rows must be chronological.",
            "inputSchema":{"$defs":definitions,"type":"object","required":["train_csv","input_csv","target"],"additionalProperties":false,
            "properties":{"train_csv":{"type":"string"},"input_csv":{"type":"string"},"target":{"type":"string"},
            "options":{"type":"object","additionalProperties":false,"properties":{
                "method":{"type":"string","default":"auto"},"candidates":{"type":"array","items":{"type":"string"},"maxItems":32},
                "folds":{"type":"integer","minimum":2,"maximum":20,"default":5},"fitness":fitness,
                "validation":{"oneOf":[{"type":"object","required":["kind"],"additionalProperties":false,"properties":{"kind":{"const":"iid"}}},
                    {"type":"object","required":["kind","gap"],"additionalProperties":false,"properties":{"kind":{"const":"forward"},"gap":{"type":"integer","minimum":0}}}]},
                "calibration":{"enum":["auto","none","shift","affine"],"default":"auto"}
            }}}},"annotations":read_only}
    ]})
}

#[derive(Default)]
pub struct Server {
    initialized: bool,
    ready: bool,
    pub created_by: CreatedBy,
}
impl Server {
    pub fn new(created_by: CreatedBy) -> Self {
        Self {
            created_by,
            ..Self::default()
        }
    }
    fn call(&self, name: &str, arguments: Value) -> Result<Value> {
        match name {
            "fitness_list" => {
                ensure!(
                    arguments == json!({}) || arguments.is_null(),
                    "fitness_list takes no arguments"
                );
                Ok(catalog())
            }
            "fitness_validate" => {
                #[derive(Deserialize)]
                #[serde(deny_unknown_fields)]
                struct Validate {
                    fitness: Value,
                }
                let request: Validate = serde_json::from_value(arguments)?;
                describe(&fitness_from_value(&request.fitness)?)
            }
            "fitness_evaluate" => evaluate(serde_json::from_value(arguments)?),
            "regression_fit_predict" => {
                fit_predict(serde_json::from_value(arguments)?, self.created_by.clone())
            }
            _ => anyhow::bail!("unknown tool {name}"),
        }
    }
    pub fn dispatch(&mut self, request: Value, mcp: bool) -> Option<Value> {
        let id = request.get("id").cloned();
        let error = |code, message: String| json!({"jsonrpc":"2.0","id":id,"error":{"code":code,"message":message}});
        if !request.is_object()
            || request["jsonrpc"] != "2.0"
            || !request["method"].is_string()
            || id
                .as_ref()
                .is_some_and(|v| !v.is_string() && !v.is_i64() && !v.is_u64())
        {
            return Some(
                json!({"jsonrpc":"2.0","id":null,"error":{"code":-32600,"message":"invalid JSON-RPC request"}}),
            );
        }
        let method = request["method"].as_str().expect("checked method");
        if id.is_none() {
            if mcp && method == "notifications/initialized" && self.initialized {
                self.ready = true;
            }
            return None;
        }
        let params = request.get("params").cloned().unwrap_or(json!({}));
        if mcp && method == "initialize" {
            if self.initialized {
                return Some(error(-32600, "already initialized".into()));
            }
            if !params["protocolVersion"].is_string()
                || !params["clientInfo"].is_object()
                || !params["capabilities"].is_object()
            {
                return Some(error(
                    -32602,
                    "initialize needs protocolVersion, capabilities and clientInfo".into(),
                ));
            }
            let requested = params["protocolVersion"].as_str().expect("checked");
            let version = MCP_SUPPORTED_VERSIONS
                .into_iter()
                .find(|v| *v == requested)
                .unwrap_or(MCP_VERSION);
            self.initialized = true;
            return Some(
                json!({"jsonrpc":"2.0","id":id,"result":{"protocolVersion":version,"capabilities":{"tools":{"listChanged":false}},
                "serverInfo":{"name":"jopedime-regression","title":"JopeDime regression objectives","version":env!("CARGO_PKG_VERSION")},
                "instructions":INSTRUCTIONS}}),
            );
        }
        if mcp && method != "ping" && !self.ready {
            return Some(error(
                -32600,
                "initialize and notifications/initialized are required".into(),
            ));
        }
        if method == "ping" {
            return Some(json!({"jsonrpc":"2.0","id":id,"result":{}}));
        }
        if mcp && method == "tools/list" {
            return Some(json!({"jsonrpc":"2.0","id":id,"result":tools()}));
        }
        let (name, arguments) = if mcp {
            if method != "tools/call" {
                return Some(error(-32601, "method not found".into()));
            }
            let Some(name) = params["name"].as_str() else {
                return Some(error(-32602, "tool name is required".into()));
            };
            (name, params.get("arguments").cloned().unwrap_or(json!({})))
        } else {
            (method, params.clone())
        };
        let known = tools()["tools"]
            .as_array()
            .expect("tool list")
            .iter()
            .any(|t| t["name"] == name);
        if !known {
            // An unknown tool is a protocol error; a failing known tool is a tool result.
            return Some(error(
                if mcp { -32602 } else { -32601 },
                format!("unknown tool {name}"),
            ));
        }
        let result = self.call(name, arguments);
        Some(match result {
            Ok(value) if mcp => {
                json!({"jsonrpc":"2.0","id":id,"result":{"content":[{"type":"text","text":value.to_string()}],"structuredContent":value,"isError":false}})
            }
            Ok(value) => json!({"jsonrpc":"2.0","id":id,"result":value}),
            Err(e) if mcp => {
                json!({"jsonrpc":"2.0","id":id,"result":{"content":[{"type":"text","text":format!("{e:#}")}],"isError":true}})
            }
            Err(e) => error(-32602, format!("{e:#}")),
        })
    }
}

#[cfg(test)]
mod tests;
