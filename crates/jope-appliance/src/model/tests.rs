use super::*;

#[test]
fn mcp_wrapper_preserves_session_show_result() {
    let root = tempfile::tempdir().unwrap();
    let result = call_mcp_legacy_tool(
        root.path(),
        "model_session_show",
        json!({"name":"baseline"}),
    )
    .unwrap();
    assert_eq!(result["schema"], sessions::SCHEMA);
    assert_eq!(result["session"], "baseline");
    assert_eq!(result["head"], "empty");
    assert_eq!(result["versions"], json!([]));
}
