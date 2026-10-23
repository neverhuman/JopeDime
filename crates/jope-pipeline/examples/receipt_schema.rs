//! Regenerate with `cargo run -q -p jope-pipeline --example receipt_schema [objective]`.
fn main() {
    let schema = match std::env::args().nth(1).as_deref() {
        Some("experiment") => {
            serde_json::to_value(schemars::schema_for!(jope_pipeline::experiment::Experiment))
                .unwrap()
        }
        Some("panel") => {
            serde_json::to_value(schemars::schema_for!(jope_pipeline::experiment::Panel)).unwrap()
        }
        Some("request") => serde_json::to_value(schemars::schema_for!(
            jope_pipeline::experiment::EvaluationRequest
        ))
        .unwrap(),
        Some("result") => serde_json::to_value(schemars::schema_for!(
            jope_pipeline::experiment::EvaluationResult
        ))
        .unwrap(),
        Some("result-v2") => serde_json::to_value(schemars::schema_for!(
            jope_pipeline::experiment_v2::EvaluationResult
        ))
        .unwrap(),
        None => jope_pipeline::receipt::derived_schema(),
        Some("objective") => jope_pipeline::receipt::derived_objective_schema(),
        Some(other) => panic!("unknown schema {other}"),
    };
    println!("{}", serde_json::to_string_pretty(&schema).unwrap());
}
