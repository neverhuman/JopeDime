# LUNA handoff cards

These refine existing queued tasks; they do not authorize parallel changes to the
MCP-02 implementation or create a second MCP implementation. Claim each task only
after its dependencies merge. OWNER ruled OD9 as rmcp 3.8.0 in the Jepson-approved
t526u scope nudge; full product qualification remains with MCP-02.

## MCP-03 · LUNA · S · W3 · deps MCP-02

**Do:** Implement the one-minor-version compatibility aliases specified in §7.1
using MCP-02's chosen dispatcher and `jope-client` calls. Check that each old name
is callable, maps to its designated v1 operation, and has the same structured
result and protocol/tool error distinction. Mark aliases deprecated in their
descriptions. Do not add a separate legacy server.

**Owns:** `crates/jope-appliance/src/mcp/aliases.rs` (as in the existing MCP-03 card).

**Done when:** Existing alias tests pass against the new adapter and `tools/list`
marks the aliases deprecated. Alias calls through the existing MCP-02 harness pass;
the suite negotiates all three legacy versions exactly. `2025-03-26` remains
excluded. Run the existing MCP-03 card's local checks and retain raw-wire evidence.

## MCP-04 · LUNA · S · W3 · deps MCP-02

**Do:** Write client setup docs and scenario scripts using the one adapter selected
by OWNER in OD9. Explain legacy initialize/initialized and, only if qualified by
MCP-02, handshake-free `2026-07-28` discovery/metadata. Use absolute executable
paths in user examples without committing machine-specific home paths. Include
the batch CLI fallback and bounded polling flow from spec §7.2–7.4.

**Owns:** `docs/product/agents.md`, `deploy/qualification/agents/*` (as in MCP-04).

**Done when:** The docs build and each client scenario script executes AG1 with a
forced retry against a mock client harness. Record tested client versions. Setup
success alone does not close A5; MCP-05 owns the real named-client scenarios and
wire receipts against MCP-02. Run the MCP-04 local checks.

MCP-02 is still COMPLEX: engine integration, schema/resource wiring, bounded IO,
real `run_status`/`run_control` p95 under fitting load and a policy for malformed
frames need its two-reviewer implementation review. Fixture echo timing is not
evidence for those product behaviors.
