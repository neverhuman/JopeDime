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
