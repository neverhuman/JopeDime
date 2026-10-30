# Connecting agent clients

JopeDime exposes its agent tools through the `jopedime mcp` stdio client. The MCP process attaches to the background engine and starts it when needed; runs continue after the agent session ends. Set the command to the absolute path of the `jopedime` executable on your machine.

## Claude Code

Tested client: Claude Code `2.1.314` (locally verified with the mock scenario; the Claude Code process was not launched).

Register the server for your user:

```bash
claude mcp add --transport stdio --scope user jopedime -- /absolute/path/to/jopedime mcp
```

## Gemini CLI and generic `mcpServers`

The generic JSON configuration below is suitable for clients that use the `mcpServers` shape. For Gemini CLI, add the entry to `~/.gemini/settings.json`:

```json
{
  "mcpServers": {
    "jopedime": {
      "command": "/absolute/path/to/jopedime",
      "args": ["mcp"]
    }
  }
}
```

Gemini CLI version: not recorded. The configuration follows its documented settings format but is **not locally verified** (`verified: false` in [client-versions.json](../../deploy/qualification/agents/client-versions.json)). The mock scenario validates the JSON configuration and MCP golden path without launching Gemini CLI.

## Codex CLI

Add this server entry to `~/.codex/config.toml`:

```toml
[mcp_servers.jopedime]
command = "/absolute/path/to/jopedime"
args = ["mcp"]
```

Codex CLI version: not recorded. The configuration follows its documented TOML format but is **not locally verified** (`verified: false` in [client-versions.json](../../deploy/qualification/agents/client-versions.json)). The mock scenario validates the TOML configuration and MCP golden path without launching Codex CLI.

## Golden path and retry behavior

The qualification scenarios use an in-process mock MCP client. They register a small CSV dataset, request a dry-run plan, start a run, retry the same `run_start` mutation after a simulated lost response with the same `idempotency_key`, poll status, export a model, and predict on the registered dataset. The mock verifies the retry returns the original run and creates no duplicate. No real client or network connection is used.

Run all three scenarios from the repository root:

```bash
python3 deploy/qualification/agents/scenario_claude_code.py
python3 deploy/qualification/agents/scenario_gemini_cli.py
python3 deploy/qualification/agents/scenario_codex_cli.py
```

Claude Code `2.1.314` is marked locally verified because its mock scenario passes; this does not qualify a live Claude Code connection. Gemini CLI and Codex CLI live-client verification is tracked by [MCP-04-F01](../../deploy/qualification/agents/MCP-04-F01.md) and remains pending until their binaries are available.
