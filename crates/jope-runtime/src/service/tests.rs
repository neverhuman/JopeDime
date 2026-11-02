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
    for (requested, expected) in [
        ("2025-06-18", "2025-06-18"),
        ("2025-03-26", MCP_VERSION),
        ("1999-01-01", MCP_VERSION),
    ] {
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
