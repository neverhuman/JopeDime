//! Small front door. Report the measured scope; unsupported capabilities remain
//! explicit until their own implementation and qualification have landed.
pub fn doctor() -> serde_json::Value {
    let heads: Vec<_> = feat_math::registry::operator_registry()
        .iter()
        .filter(|op| op.role == feat_math::registry::OperatorRole::Head)
        .map(|op| {
            serde_json::json!({"name":op.name, "algorithm":op.semantic_key,
            "lifecycle":op.lifecycle, "declared_tasks":op.tasks})
        })
        .collect();
    serde_json::json!({
        "schema":"jope.doctor/v1", "name":"Algorithm Synthesis Model (ASM)",
        "description":"learns to generate and optimize executable algorithms from data",
        "tool_version":crate::TOOL_VERSION, "panel_ready":false,
        "capabilities":[
            {"name":"regression pipeline", "status":"implemented", "command":"jope pipeline run", "limit":"generic GP seeds; numerical finite inputs; recipe source emission"},
            {"name":"binary and multiclass pipeline", "status":"not_integrated", "limit":"task-aware kernels exist; the regression pipeline refuses these tasks"},
            {"name":"bounded 30-dataset benchmark", "status":"execution_implemented", "limit":"frozen inputs, typed fits, shared isolated workers and resumable comparator run/report exist; dated synthetic host isolation receipts exist; current host admission and real neural comparison remain required"},
            {"name":"neural generation-zero proposals", "status":if cfg!(feature = "neural-search") {"execution_implemented"} else {"feature_not_enabled"}, "limit":"bounded proposal-to-fit and training-side selection source exists; accepted-model consumption requires current regression-consumption proof; historical receipts are not current qualification"},
            {"name":"CPU context production", "status":if cfg!(feature = "qualified-context") {"qualification_required"} else {"feature_not_enabled"}, "limit":"eight retained native reference cases plus fresh encoding require original bounded execution receipts; dated parity receipts exist; current source and asset bindings must be requalified"},
            {"name":"policy training", "status":"not_qualified", "limit":"experimental category trainer supports admitted projected inputs, strict optimizer resume and controlled siblings; live learning remains unqualified; legacy local train-step updates a centroid router"},
            {"name":"fitted predictor export across five backends", "status":"not_qualified", "limit":"private fitted export and independent parity verifier are implemented; each winner needs measured backend parity before export qualification"}
        ],
        "native_head_inventory":heads,
        "inventory_note":"Governed registry metadata, including aliases and unavailable entries; not a runnable or qualified benchmark roster. Advanced-v3 heads are inventoried separately in docs/asm.md.",
        "legacy_fixture_limit":"fit/search/router use numeric R2 diagnostics; non-regression outputs are not classification evidence and are excluded from the panel",
        "serving_state_changed":false
    })
}
