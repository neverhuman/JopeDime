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
pub const MCP_SUPPORTED_VERSIONS: [&str; 4] =
    ["2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"];

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
}
pub fn evaluate(request: Evaluation) -> Result<Value> {
    let fitness = fitness_from_value(&request.fitness)?;
    Ok(serde_json::to_value(fitness.evaluate_with(
        &request.targets,
        &request.predictions,
        &request.columns,
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
        "requires_chronological_validation": fitness.requires_order(),
        "specification": fitness,
    }))
}

pub fn catalog() -> Value {
    json!({"schema": crate::fitness::FITNESS_VERSION,
        "fitness": Fitness::catalog().into_iter().map(|entry| json!({
            "name": entry.name,
            "direction": entry.fitness.direction(),
            "requires_chronological_validation": entry.fitness.requires_order(),
            "description": entry.description,
            "specification": entry.fitness})).collect::<Vec<_>>(),
        "custom": {
            "expression_operators": ["y", "pred", "constant", "column", "add", "sub", "mul",
                "div", "min", "max", "pow", "abs", "neg", "square", "sqrt", "ln", "log1p", "exp",
                "step", "clip", "where"],
            "aggregations": ["mean", "sum", "max", "quantile", "tail_mean", "root_mean"],
            "forms": ["custom", "weighted", "composite"],
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
mod tests {
    use super::*;
    fn params() -> Value {
        json!({"fitness":{"metric":"mae"},"targets":[1,2,3],"predictions":[2,2,2]})
    }
    fn ready() -> Server {
        let mut server = Server::default();
        assert!(server
            .dispatch(json!({"jsonrpc":"2.0","id":0,"method":"tools/list"}), true)
            .unwrap()
            .get("error")
            .is_some());
        let init=server.dispatch(json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":MCP_VERSION,"clientInfo":{"name":"test","version":"1"},"capabilities":{}}}),true).unwrap();
        assert_eq!(init["result"]["protocolVersion"], MCP_VERSION);
        assert!(init["result"]["instructions"].is_string());
        assert!(server
            .dispatch(
                json!({"jsonrpc":"2.0","method":"notifications/initialized"}),
                true
            )
            .is_none());
        server
    }
    #[test]
    fn api_mcp_and_rust_evaluation_have_identical_scores() {
        let rust = evaluate(serde_json::from_value(params()).unwrap()).unwrap();
        let api = Server::default()
            .dispatch(
                json!({"jsonrpc":"2.0","id":4,"method":"fitness_evaluate","params":params()}),
                false,
            )
            .unwrap();
        let mcp=ready().dispatch(json!({"jsonrpc":"2.0","id":"x","method":"tools/call","params":{"name":"fitness_evaluate","arguments":params()}}),true).unwrap();
        assert_eq!(rust, api["result"]);
        assert_eq!(rust, mcp["result"]["structuredContent"]);
        assert_eq!(mcp["id"], "x");
        assert_eq!(mcp["result"]["isError"], false);
        // A name, shorthand or object resolve to the same specification and identity.
        let by_name = evaluate(
            serde_json::from_value(
                json!({"fitness":"pinball:quantile=0.9","targets":[1,2],"predictions":[2,2]}),
            )
            .unwrap(),
        )
        .unwrap();
        let by_object = evaluate(
            serde_json::from_value(json!({"fitness":{"metric":"pinball","quantile":0.9},"targets":[1,2],"predictions":[2,2]}))
                .unwrap(),
        )
        .unwrap();
        assert_eq!(by_name, by_object);
    }
    #[test]
    fn rpc_lifecycle_discovery_errors_and_notifications_are_explicit() {
        let mut server = ready();
        let list = server
            .dispatch(json!({"jsonrpc":"2.0","id":2,"method":"tools/list"}), true)
            .unwrap();
        assert_eq!(list["result"]["tools"].as_array().unwrap().len(), 4);
        assert!(server
            .dispatch(json!([]), false)
            .unwrap()
            .get("error")
            .is_some());
        assert!(server
            .dispatch(json!({"jsonrpc":"2.0","method":"ping"}), true)
            .is_none());
        let bad=server.dispatch(json!({"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"fitness_evaluate","arguments":{"fitness":{"metric":"wat"}}}}),true).unwrap();
        assert_eq!(bad["result"]["isError"], true);
        let unknown=server.dispatch(json!({"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"nope","arguments":{}}}),true).unwrap();
        assert_eq!(unknown["error"]["code"], -32602);
        assert!(server
            .dispatch(
                json!({"jsonrpc":"2.0","id":5,"method":"fitness_list","params":{"unexpected":1}}),
                false
            )
            .unwrap()
            .get("error")
            .is_some());
        // An older client revision is echoed; an unknown one gets the newest.
        for (requested, expected) in [("2025-06-18", "2025-06-18"), ("1999-01-01", MCP_VERSION)] {
            let init = Server::default().dispatch(json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":requested,"clientInfo":{"name":"t","version":"1"},"capabilities":{}}}),true).unwrap();
            assert_eq!(init["result"]["protocolVersion"], expected);
        }
    }
    #[test]
    fn validate_reports_identity_columns_and_temporal_requirements() {
        let described = Server::default()
            .dispatch(
                json!({"jsonrpc":"2.0","id":1,"method":"fitness_validate","params":{"fitness":
                    {"metric":"weighted","weights":{"op":"column","name":"exposure"},"fitness":{"metric":"poisson_deviance"}}}}),
                false,
            )
            .unwrap();
        let result = &described["result"];
        assert_eq!(result["name"], "weighted_poisson_deviance");
        assert_eq!(result["columns"], json!(["exposure"]));
        assert_eq!(result["requires_chronological_validation"], false);
        assert_eq!(result["fitness_sha256"].as_str().unwrap().len(), 64);
        let sortino = describe(&Fitness::parse("sortino").unwrap()).unwrap();
        assert_eq!(sortino["requires_chronological_validation"], true);
        assert_eq!(sortino["direction"], "maximize");
    }
    #[test]
    fn fit_predict_transport_runs_the_custom_objective_and_binds_it_to_the_model() {
        let train = String::from("x,y\n")
            + &(0..32)
                .map(|i| format!("{i},{}\n", i * 2 + 1))
                .collect::<String>();
        let args = json!({"train_csv":train,"input_csv":"x\n1\n3\n","target":"y","options":{"candidates":["ridge"],"folds":3,"fitness":{"metric":"mae"}}});
        let a = Server::default()
            .dispatch(
                json!({"jsonrpc":"2.0","id":1,"method":"regression_fit_predict","params":args}),
                false,
            )
            .unwrap();
        assert!(a.get("error").is_none(), "{a}");
        assert_eq!(a["result"]["selection"]["metric"], "mae");
        assert_eq!(a["result"]["selection"]["fitness"]["metric"], "mae");
        assert_eq!(
            a["result"]["model_header"]["training_procedure"]["selection"]["fitness_sha256"],
            a["result"]["selection"]["fitness_sha256"]
        );
        let b=ready().dispatch(json!({"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"regression_fit_predict","arguments":args}}),true).unwrap();
        assert_eq!(
            a["result"]["predictions_csv"],
            b["result"]["structuredContent"]["predictions_csv"]
        );
    }
    #[test]
    fn fit_predict_reads_objective_columns_from_training_and_ignores_them_in_input() {
        let train = String::from("x,price,y\n")
            + &(0..36)
                .map(|i| format!("{i},{},{}\n", 1 + i % 4, i * 3 + i % 7))
                .collect::<String>();
        let profit = json!({"metric":"custom","name":"margin","direction":"maximize",
            "expression":{"op":"sub",
                "left":{"op":"mul","left":{"op":"column","name":"price"},"right":{"op":"min","left":{"op":"y"},"right":{"op":"pred"}}},
                "right":{"op":"mul","left":{"op":"constant","value":0.5},"right":{"op":"pred"}}},
            "aggregation":{"kind":"mean"}});
        let args = json!({"train_csv":train,"input_csv":"price,x\n2,1\n3,5\n","target":"y",
            "options":{"candidates":["ridge"],"folds":3,"fitness":profit,"calibration":"shift"}});
        let a = Server::default()
            .dispatch(
                json!({"jsonrpc":"2.0","id":1,"method":"regression_fit_predict","params":args}),
                false,
            )
            .unwrap();
        assert!(a.get("error").is_none(), "{a}");
        let result = &a["result"];
        assert_eq!(result["ignored_columns"], json!(["price"]));
        assert_eq!(
            result["model_header"]["auxiliary_columns"],
            json!(["price"])
        );
        assert_eq!(
            result["model_header"]["input_schema"]
                .as_array()
                .unwrap()
                .len(),
            1
        );
        assert_eq!(result["selection"]["calibration"]["family"], "shift");
        // Asking for calibration without an objective is refused.
        let refused = Server::default()
            .dispatch(
                json!({"jsonrpc":"2.0","id":2,"method":"regression_fit_predict","params":
                    {"train_csv":"x,y\n1,2\n2,3\n3,4\n4,5\n","input_csv":"x\n1\n","target":"y","options":{"calibration":"affine"}}}),
                false,
            )
            .unwrap();
        assert!(refused.get("error").is_some());
    }
}
