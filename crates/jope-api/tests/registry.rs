use std::collections::HashSet;

use jope_api::registry::{terminal_command, terminal_commands, CommandKind, OPERATIONS};

#[test]
fn operation_names_are_unique() {
    let mut names = HashSet::new();

    for operation in OPERATIONS {
        assert!(
            names.insert(operation.name),
            "duplicate operation name: {}",
            operation.name
        );
    }
}

#[test]
fn mcp_tool_names_are_unique() {
    let mut names = HashSet::new();

    for operation in OPERATIONS {
        if let Some(tool) = operation.mcp_tool {
            assert!(names.insert(tool), "duplicate MCP tool name: {tool}");
        }
    }
}

#[test]
fn terminal_command_names_and_aliases_are_unique() {
    let mut names = HashSet::new();

    for command in terminal_commands() {
        assert!(
            names.insert(command.name),
            "duplicate terminal command name: {}",
            command.name
        );
        for alias in command.aliases {
            assert!(
                names.insert(*alias),
                "duplicate terminal command alias: {alias}"
            );
        }
    }
}

#[test]
fn mutating_operations_require_idempotency_and_admin_operations_have_no_tool() {
    for operation in OPERATIONS {
        assert!(
            !operation.mutating || operation.requires_idempotency_key,
            "mutating operation {} must require an idempotency key",
            operation.id
        );
    }

    for id in [4, 10, 16, 17] {
        let operation = OPERATIONS
            .iter()
            .find(|operation| operation.id == id)
            .unwrap_or_else(|| panic!("operation {id} is missing"));
        assert!(
            operation.mcp_tool.is_none(),
            "operation {id} must not expose an MCP tool"
        );
    }
}

#[test]
fn engine_commands_map_to_existing_operations() {
    for command in terminal_commands() {
        let is_engine_command = matches!(
            command.kind,
            CommandKind::Engine | CommandKind::EngineAndDraft
        );
        if is_engine_command {
            assert!(
                !command.engine_operations.is_empty(),
                "engine command {} must map to an operation",
                command.name
            );
        }

        for operation_id in command.engine_operations {
            assert!(
                OPERATIONS
                    .iter()
                    .any(|operation| operation.id == *operation_id),
                "command {} maps to missing operation {operation_id}",
                command.name
            );
        }
    }
}

#[test]
fn fitness_and_language_are_aliases_of_objective_and_lang() {
    assert_eq!(
        terminal_command("\\fitness").map(|command| command.name),
        Some("\\objective")
    );
    assert_eq!(
        terminal_command("\\language").map(|command| command.name),
        Some("\\lang")
    );
}
