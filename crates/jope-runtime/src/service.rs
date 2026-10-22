//! Local JSON-RPC and MCP adapters over the same runtime API. No file/network access.
use crate::{
    api, budget::Budget, fitness::Fitness, fitted::CreatedBy, limits::Limits, table,
    task::TaskChoice,
};
use anyhow::{ensure, Result};
use serde::Deserialize;
use serde_json::{json, Value};
use std::time::Duration;

pub const MAX_MESSAGE_BYTES: usize = 8 * 1024 * 1024;
pub const MCP_VERSION: &str = "2025-11-25";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Evaluation {
    pub fitness: Fitness,
    pub targets: Vec<f64>,
    pub predictions: Vec<f64>,
}
pub fn evaluate(request: Evaluation) -> Result<Value> {
    Ok(serde_json::to_value(
        request
            .fitness
            .evaluate(&request.targets, &request.predictions)?,
    )?)
}
pub fn catalog() -> Value {
    json!({"schema": crate::fitness::FITNESS_VERSION,
        "fitness": Fitness::catalog().into_iter().map(|(name,fitness)| json!({
            "name":name,"direction":fitness.direction(),"requires_chronological_validation":fitness.requires_order(),
            "specification":fitness})).collect::<Vec<_>>(),
        "custom": "bounded JSON expression over y and pred; mean/sum/max/quantile aggregation; positive scaled composites"})
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FitPredict {
    train_csv: String,
    input_csv: String,
    target: String,
    #[serde(default)]
    options: api::FitOptions,
}
fn fit_predict(request: FitPredict, created_by: CreatedBy) -> Result<Value> {
    ensure!(
        request.train_csv.len() + request.input_csv.len() <= MAX_MESSAGE_BYTES,
        "CSV input is too large"
    );
    ensure!(
        (2..=20).contains(&request.options.folds),
        "service folds must be in 2..=20"
    );
    ensure!(
        request.options.candidates.len() <= 32,
        "service supports at most 32 candidate selectors"
    );
    let limits = Limits {
        max_input_bytes: MAX_MESSAGE_BYTES as u64,
        max_rows: 100_000,
        max_columns: 1000,
        ..Limits::default()
    };
    let csv = table::parse_csv(request.train_csv.as_bytes(), &limits)?;
    let train = table::training(
        &csv,
        &request.target,
        TaskChoice::parse("regression").expect("known task"),
        &limits,
    )?;
    let input = table::parse_csv(request.input_csv.as_bytes(), &limits)?;
    // Validate prediction input before paying for selection.
    let (rows, ignored_columns) = table::prediction(&input, &train.schema, &train.target)?;
    let budget = Budget::new(Some(Duration::from_secs(30)), 1);
    let outcome = api::fit(&train, &request.options, &budget, &csv.sha256, created_by)?;
    let predictions = api::predict(&outcome.fitted, &rows, &budget)?;
    Ok(
        json!({"schema":"jope.regression-fit-predict/v1", "model_header":outcome.fitted.header,
        "selection":outcome.selection,"predictions_csv":String::from_utf8(api::predictions_csv(&predictions)?)?,
        "ignored_columns":ignored_columns,"budget":budget.record()}),
    )
}

fn tools() -> Value {
    let definitions = crate::fitness::schema()["$defs"].clone();
    let fitness = json!({"$ref":"#/$defs/fitness"});
    json!({"tools":[
        {"name":"fitness_list","description":"List regression objectives, defaults, directions and temporal requirements.",
            "inputSchema":{"type":"object","properties":{},"additionalProperties":false},"annotations":{"readOnlyHint":true,"openWorldHint":false}},
        {"name":"fitness_evaluate","description":"Evaluate a regression fitness specification on equal, nonempty target/prediction arrays.",
            "inputSchema":{"$defs":definitions,"type":"object","required":["fitness","targets","predictions"],"additionalProperties":false,
            "properties":{"fitness":fitness,"targets":{"type":"array","items":{"type":"number"},"minItems":1},"predictions":{"type":"array","items":{"type":"number"},"minItems":1}}},
            "annotations":{"readOnlyHint":true,"openWorldHint":false}},
        {"name":"regression_fit_predict","description":"Select and refit native regression models using held-out fitness, then predict an inline CSV. 30s cooperative selection budget, one CPU, no file access. Temporal objectives require options.validation={kind:forward,gap:N}; input rows must be chronological.",
            "inputSchema":{"$defs":definitions,"type":"object","required":["train_csv","input_csv","target"],"additionalProperties":false,
            "properties":{"train_csv":{"type":"string"},"input_csv":{"type":"string"},"target":{"type":"string"},
            "options":{"type":"object","additionalProperties":false,"properties":{
                "method":{"type":"string","default":"auto"},"candidates":{"type":"array","items":{"type":"string"},"maxItems":32},
                "folds":{"type":"integer","minimum":2,"maximum":20,"default":5},"fitness":fitness,
                "validation":{"oneOf":[{"type":"object","required":["kind"],"additionalProperties":false,"properties":{"kind":{"const":"iid"}}},
                    {"type":"object","required":["kind","gap"],"additionalProperties":false,"properties":{"kind":{"const":"forward"},"gap":{"type":"integer","minimum":0}}}]}
            }}}},"annotations":{"readOnlyHint":true,"openWorldHint":false}}
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
            self.initialized = true;
            return Some(
                json!({"jsonrpc":"2.0","id":id,"result":{"protocolVersion":MCP_VERSION,"capabilities":{"tools":{}},
                "serverInfo":{"name":"jopedime-regression","version":env!("CARGO_PKG_VERSION")}}}),
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
        let result: Result<Value> = (|| match name {
            "fitness_list" => {
                ensure!(arguments == json!({}), "fitness_list takes no arguments");
                Ok(catalog())
            }
            "fitness_evaluate" => evaluate(serde_json::from_value(arguments)?),
            "regression_fit_predict" => {
                fit_predict(serde_json::from_value(arguments)?, self.created_by.clone())
            }
            _ => anyhow::bail!("unknown tool {name}"),
        })();
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
    }
    #[test]
    fn rpc_lifecycle_discovery_errors_and_notifications_are_explicit() {
        let mut server = ready();
        let list = server
            .dispatch(json!({"jsonrpc":"2.0","id":2,"method":"tools/list"}), true)
            .unwrap();
        assert_eq!(list["result"]["tools"].as_array().unwrap().len(), 3);
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
        assert!(server
            .dispatch(
                json!({"jsonrpc":"2.0","id":5,"method":"fitness_list","params":{"unexpected":1}}),
                false
            )
            .unwrap()
            .get("error")
            .is_some());
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
}
