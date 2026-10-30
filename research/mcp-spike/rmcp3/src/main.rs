use mcp_spike_tools::{arguments, tools, INSTRUCTIONS};
use rmcp::{model::*, service::RequestContext, ErrorData, RoleServer, ServerHandler, ServiceExt};
use serde_json::{json, Value};
use std::{borrow::Cow, time::Duration};

#[derive(Clone)]
struct Server;

impl ServerHandler for Server {
    fn get_info(&self) -> ServerConfig {
        serde_json::from_value(json!({"protocolVersion":"2025-11-25",
            "capabilities":{"tools":{"listChanged":false}},
            "serverInfo":{"name":"mcp-spike-rmcp3","version":env!("CARGO_PKG_VERSION")},
            "instructions":INSTRUCTIONS}))
        .expect("static server info")
    }

    fn supported_protocol_versions(&self) -> Cow<'static, [ProtocolVersion]> {
        Cow::Owned(vec![
            ProtocolVersion::V_2024_11_05,
            ProtocolVersion::V_2025_06_18,
            ProtocolVersion::V_2025_11_25,
            ProtocolVersion::V_2026_07_28,
        ])
    }

    async fn list_tools(
        &self,
        _: Option<PaginatedRequestParams>,
        _: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, ErrorData> {
        let definitions =
            serde_json::from_value(tools()["tools"].clone()).expect("static tool catalog");
        Ok(ListToolsResult::with_all_items(definitions))
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        _: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, ErrorData> {
        let name = request.name.as_ref();
        if name != "echo" && name != "slow_echo" {
            return Err(ErrorData::invalid_params("unknown tool", None));
        }
        let args = Value::Object(request.arguments.unwrap_or_default());
        let result = match arguments(&args) {
            Ok((text, delay)) => {
                if name == "slow_echo" {
                    tokio::time::sleep(Duration::from_millis(delay)).await;
                }
                // Include both a short text summary and structured content (§7.2).
                CallToolResult::structured(json!({"text":text}))
            }
            Err(message) => CallToolResult::error(vec![ContentBlock::text(message)]),
        };
        Ok(result.into())
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    Server
        .serve(rmcp::transport::stdio())
        .await?
        .waiting()
        .await?;
    Ok(())
}
