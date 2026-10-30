//! Extraction of service.rs::Server::dispatch at main@5c8a16711632b4b8c588c9a16679a585abc0f135.
//! Only the tool catalog/call and server identity differ. Serial dispatch is intentional.
use mcp_spike_tools::{arguments, result, tool_error, tools, INSTRUCTIONS};
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};
use std::time::Duration;

const VERSIONS: [&str; 3] = ["2025-11-25", "2025-06-18", "2024-11-05"];

#[derive(Default)]
struct Server {
    initialized: bool,
    ready: bool,
}

impl Server {
    fn dispatch(&mut self, request: Value) -> Option<Value> {
        let id = request.get("id").cloned();
        let error = |code, message: &str| json!({"jsonrpc":"2.0","id":id,"error":{"code":code,"message":message}});
        if !request.is_object()
            || request["jsonrpc"] != "2.0"
            || !request["method"].is_string()
            || id
                .as_ref()
                .is_some_and(|v| !v.is_string() && !v.is_i64() && !v.is_u64())
        {
            return Some(json!({"jsonrpc":"2.0","id":null,
                "error":{"code":-32600,"message":"invalid JSON-RPC request"}}));
        }
        let method = request["method"].as_str().expect("checked method");
        if id.is_none() {
            if method == "notifications/initialized" && self.initialized {
                self.ready = true;
            }
            return None;
        }
        let params = request.get("params").cloned().unwrap_or(json!({}));
        if method == "initialize" {
            if self.initialized {
                return Some(error(-32600, "already initialized"));
            }
            if !params["protocolVersion"].is_string()
                || !params["clientInfo"].is_object()
                || !params["capabilities"].is_object()
            {
                return Some(error(
                    -32602,
                    "initialize needs protocolVersion, capabilities and clientInfo",
                ));
            }
            let requested = params["protocolVersion"].as_str().expect("checked");
            let version = VERSIONS
                .into_iter()
                .find(|v| *v == requested)
                .unwrap_or(VERSIONS[0]);
            self.initialized = true;
            return Some(json!({"jsonrpc":"2.0","id":id,"result":{
                "protocolVersion":version,"capabilities":{"tools":{"listChanged":false}},
                "serverInfo":{"name":"mcp-spike-dispatcher","version":env!("CARGO_PKG_VERSION")},
                "instructions":INSTRUCTIONS}}));
        }
        if method != "ping" && !self.ready {
            return Some(error(
                -32600,
                "initialize and notifications/initialized are required",
            ));
        }
        if method == "ping" {
            return Some(json!({"jsonrpc":"2.0","id":id,"result":{}}));
        }
        if method == "tools/list" {
            return Some(json!({"jsonrpc":"2.0","id":id,"result":tools()}));
        }
        if method != "tools/call" {
            return Some(error(-32601, "method not found"));
        }
        let Some(name) = params["name"].as_str() else {
            return Some(error(-32602, "tool name is required"));
        };
        if name != "echo" && name != "slow_echo" {
            return Some(error(-32602, "unknown tool"));
        }
        let args = params.get("arguments").cloned().unwrap_or(json!({}));
        let output = match arguments(&args) {
            Ok((text, delay)) => {
                if name == "slow_echo" {
                    std::thread::sleep(Duration::from_millis(delay));
                }
                result(text)
            }
            Err(message) => tool_error(&message),
        };
        Some(json!({"jsonrpc":"2.0","id":id,"result":output}))
    }
}

fn main() -> io::Result<()> {
    let mut server = Server::default();
    let mut stdout = io::stdout().lock();
    // The fixture client bounds input. Product byte-limit qualification is out of scope.
    for line in io::stdin().lock().lines() {
        let response = match serde_json::from_str(&line?) {
            Ok(request) => server.dispatch(request),
            Err(_) => Some(json!({"jsonrpc":"2.0","id":null,
                "error":{"code":-32700,"message":"parse error"}})),
        };
        if let Some(response) = response {
            writeln!(stdout, "{response}")?;
            stdout.flush()?;
        }
    }
    Ok(())
}
