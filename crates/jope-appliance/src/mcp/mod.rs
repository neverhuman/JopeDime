//! One rmcp 3.8.0 adapter over the shared v1 client. It owns no engine or run state.
//! CLI discovery/auto-start, aliases and canonical schema completion retain their owning tasks.
mod aliases;
pub mod catalog;
mod dispatch;
mod transport;
use jope_api::registry::OPERATIONS;
use jope_client::Transport;
use rmcp::{model::*, service::RequestContext, ErrorData, RoleServer, ServerHandler};
use serde_json::{json, Value};
use std::{borrow::Cow, sync::Arc, time::Duration};
use tokio::{
    sync::Semaphore,
    time::{timeout_at, Instant},
};

pub const MAX_FRAME: usize = 8 * 1024 * 1024;
pub struct Adapter<T> {
    client: Arc<T>,
    resources: Arc<dispatch::Resources>,
    fast: Arc<Semaphore>,
    slow: Arc<Semaphore>,
    control: Arc<Semaphore>,
}
impl<T> Clone for Adapter<T> {
    fn clone(&self) -> Self {
        Self {
            client: self.client.clone(),
            resources: self.resources.clone(),
            fast: self.fast.clone(),
            slow: self.slow.clone(),
            control: self.control.clone(),
        }
    }
}
impl<T: Transport + Send + Sync + 'static> Adapter<T> {
    pub fn new(client: T) -> Self {
        Self {
            client: Arc::new(client),
            resources: Arc::new(dispatch::Resources::default()),
            fast: Arc::new(Semaphore::new(6)),
            slow: Arc::new(Semaphore::new(8)),
            control: Arc::new(Semaphore::new(2)),
        }
    }
    pub async fn serve_io<R, W>(self, input: R, output: W) -> anyhow::Result<()>
    where
        R: tokio::io::AsyncRead + Send + Unpin + 'static,
        W: tokio::io::AsyncWrite + Send + Unpin + 'static,
    {
        transport::serve(self, input, output).await
    }
    pub fn serve_stdio(self) -> anyhow::Result<()> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .max_blocking_threads(16)
            .enable_all()
            .build()?;
        let result = runtime.block_on(self.serve_io(tokio::io::stdin(), tokio::io::stdout()));
        runtime.shutdown_timeout(Duration::from_secs(1));
        result
    }
    async fn call(
        &self,
        request: CallToolRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, ErrorData> {
        if aliases::contains(&request.name) {
            let args = Value::Object(request.arguments.unwrap_or_default());
            let name = request.name.to_string();
            let result = tokio::select! {
                _ = context.ct.cancelled() => return Err(ErrorData::internal_error("request cancelled", None)),
                result = tokio::task::spawn_blocking(move || aliases::call(&name, args)) => {
                    result.map_err(|_| ErrorData::internal_error("legacy tool worker failed", None))?
                }
            };
            return Ok(result);
        }
        let op = OPERATIONS
            .iter()
            .find(|op| op.mcp_tool == Some(request.name.as_ref()))
            .ok_or_else(|| ErrorData::invalid_params("unknown tool", None))?;
        let args = Value::Object(request.arguments.unwrap_or_default());
        catalog::validate_arguments(op, &args).map_err(|e| ErrorData::invalid_params(e, None))?;
        // Validate keys before allocating a worker or doing any engine read/mutation.
        if op.requires_idempotency_key {
            jope_client::Request::new(
                op.id,
                0,
                &if op.paths[0].contains("{id}") {
                    vec![("id", "validation")]
                } else {
                    vec![]
                },
            )
            .and_then(|r| r.key(args["idempotency_key"].as_str()))
            .map_err(|_| ErrorData::invalid_params("invalid idempotency_key", None))?;
        }
        let wait = if op.id == 7 {
            args["wait_seconds"].as_u64().unwrap_or(0)
        } else {
            0
        };
        let immediate = op.id == 8 || op.id == 7 && wait == 0;
        let permit = Arc::new(
            (if op.id == 8 {
                self.control.clone()
            } else if immediate {
                self.fast.clone()
            } else {
                self.slow.clone()
            })
            .try_acquire_owned()
            .map_err(|_| {
                ErrorData::internal_error(
                    "request capacity reached; retry with the same idempotency key",
                    None,
                )
            })?,
        );
        let deadline = Instant::now() + Duration::from_secs(if wait > 0 { wait } else { 65 });
        let mut last = None;
        loop {
            let mut wire_args = args.clone();
            if wait > 0 {
                let remaining = deadline
                    .saturating_duration_since(Instant::now())
                    .as_secs()
                    .clamp(1, 20);
                wire_args["wait_seconds"] = json!(remaining);
            }
            let client = self.client.clone();
            let resources = self.resources.clone();
            let held = permit.clone();
            let worker = tokio::task::spawn_blocking(move || {
                let _held = held;
                dispatch::execute(client.as_ref(), resources.as_ref(), op, wire_args)
            });
            let outcome = tokio::select! {
                _=context.ct.cancelled()=>return Err(ErrorData::internal_error("request cancelled",None)),
                result=timeout_at(deadline,worker)=>match result{
                    Ok(Ok(value))=>value,
                    Ok(Err(_))=>return Err(ErrorData::internal_error("engine request worker failed",None)),
                    Err(_)=>return Ok(last.unwrap_or_else(||CallToolResult::structured_error(serde_json::to_value(dispatch::problem(jope_api::error::ErrorCode::IoFailure,"Engine response deadline reached. Retry mutations with the same idempotency key.")).expect("Problem JSON")))),
                }
            };
            let result = match outcome {
                dispatch::Outcome::Success(value) => {
                    let mut links = Vec::new();
                    if op.id == 12 {
                        let prefix = format!(
                            "jope://runs/{}/algorithms/{}/source/",
                            args["run_id"].as_str().unwrap_or_default(),
                            value["algorithm_id"].as_str().unwrap_or_default()
                        );
                        links = self
                            .resources
                            .list()
                            .map_err(|e| ErrorData::internal_error(e, None))?
                            .into_iter()
                            .filter(|(uri, _, _)| uri.starts_with(&prefix))
                            .collect();
                    } else if op.id == 15 {
                        if let Some(uri) = value["resource"].as_str() {
                            links.push((uri.to_owned(), "text/csv".into(), 0));
                        }
                    }
                    let mut result = CallToolResult::structured(value);
                    // Keep the SDK JSON text fallback for clients that do not expose
                    // structuredContent to their model. Both carry the same public value.
                    for (uri, mime, _) in links {
                        if context.protocol_version() == Some(ProtocolVersion::V_2024_11_05) {
                            result
                                .content
                                .push(ContentBlock::text(format!("Full content: {uri}")));
                        } else {
                            result.content.push(ContentBlock::resource_link(
                                Resource::new(uri.clone(), uri).with_mime_type(mime),
                            ));
                        }
                    }
                    result
                }
                dispatch::Outcome::Failure(value) => CallToolResult::structured_error(value),
            };
            if serde_json::to_vec(&result)
                .map_err(|_| ErrorData::internal_error("result serialization failed", None))?
                .len()
                > MAX_FRAME / 2
            {
                return Ok(CallToolResult::structured_error(serde_json::to_value(dispatch::problem(jope_api::error::ErrorCode::InvalidInput,"Result exceeds the bounded MCP response limit; use a resource or a smaller request.")).expect("Problem JSON")));
            }
            // The API long-poll is split into <=20s requests because Http has a 30s receive
            // timeout. A recorded after_event controls whether another slice is necessary.
            let should_poll = wait > 0
                && result.is_error != Some(true)
                && args.get("after_event").is_some_and(|after| {
                    let after =
                        serde_json::from_value::<jope_api::decimal::DecimalU64>(after.clone()).ok();
                    let value = result.structured_content.as_ref();
                    let current = value.and_then(|v| {
                        serde_json::from_value::<jope_api::decimal::DecimalU64>(
                            v["event_watermark"].clone(),
                        )
                        .ok()
                    });
                    after.zip(current).is_some_and(|(a, b)| b.0 <= a.0)
                        && value
                            .and_then(|v| v["state"].as_str())
                            .is_some_and(|state| {
                                !matches!(state, "completed" | "cancelled" | "failed")
                            })
                });
            if !should_poll || Instant::now() >= deadline {
                return Ok(result);
            }
            last = Some(result);
            tokio::select! {_=context.ct.cancelled()=>return Err(ErrorData::internal_error("request cancelled",None)),_=tokio::time::sleep_until((Instant::now()+Duration::from_millis(50)).min(deadline))=>{}}
            if Instant::now() >= deadline {
                return Ok(last.expect("a previous snapshot"));
            }
        }
    }
}

impl<T: Transport + Send + Sync + 'static> ServerHandler for Adapter<T> {
    fn get_info(&self) -> ServerConfig {
        serde_json::from_value(json!({"protocolVersion":"2025-11-25","capabilities":{"tools":{"listChanged":false},"resources":{"subscribe":false,"listChanged":false},"experimental":{"jope/apiSchemas":{"pending":catalog::pending_exchanges(),"qualified":false}}},"serverInfo":{"name":"jopedime","version":env!("CARGO_PKG_VERSION")},"instructions":catalog::INSTRUCTIONS})).expect("static server info")
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
        request: Option<PaginatedRequestParams>,
        _: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, ErrorData> {
        if request.is_some_and(|r| r.cursor.is_some()) {
            return Err(ErrorData::invalid_params("unknown tool cursor", None));
        }
        let mut tools: Vec<Tool> =
            serde_json::from_value(catalog::tools()["tools"].clone()).expect("tool definitions");
        tools.extend(
            serde_json::from_value::<Vec<Tool>>(Value::Array(aliases::tools()))
                .expect("legacy tool definitions"),
        );
        Ok(ListToolsResult::with_all_items(tools))
    }
    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, ErrorData> {
        Ok(self.call(request, context).await?.into())
    }
    async fn list_resources(
        &self,
        _: Option<PaginatedRequestParams>,
        _: RequestContext<RoleServer>,
    ) -> Result<ListResourcesResult, ErrorData> {
        let resources = self
            .resources
            .list()
            .map_err(|e| ErrorData::internal_error(e, None))?
            .into_iter()
            .map(|(uri, mime, size)| {
                Resource::new(uri.clone(), uri)
                    .with_mime_type(mime)
                    .with_size(size as u64)
            })
            .collect();
        Ok(ListResourcesResult::with_all_items(resources))
    }
    async fn list_resource_templates(
        &self,
        _: Option<PaginatedRequestParams>,
        _: RequestContext<RoleServer>,
    ) -> Result<ListResourceTemplatesResult, ErrorData> {
        let templates = [
            ("jope://runs/{run_id}", "Run snapshot"),
            (
                "jope://runs/{run_id}/algorithms/{algorithm_id}/source/{language}",
                "Algorithm source",
            ),
            ("jope://artifacts/{artifact_id}/manifest", "Export manifest"),
            ("jope://datasets/{dataset_id}/profile", "Dataset profile"),
        ];
        Ok(ListResourceTemplatesResult::with_all_items(
            templates
                .into_iter()
                .map(|(uri, name)| {
                    serde_json::from_value(json!({"uriTemplate":uri,"name":name}))
                        .expect("resource template")
                })
                .collect(),
        ))
    }
    async fn read_resource(
        &self,
        request: ReadResourceRequestParams,
        _: RequestContext<RoleServer>,
    ) -> Result<ReadResourceResponse, ErrorData> {
        let permit =
            self.slow.clone().try_acquire_owned().map_err(|_| {
                ErrorData::internal_error("resource request capacity reached", None)
            })?;
        let client = self.client.clone();
        let resources = self.resources.clone();
        let uri = request.uri;
        let result = tokio::task::spawn_blocking(move || {
            let _permit = permit;
            dispatch::read_resource(client.as_ref(), &resources, &uri)
                .map(|(mime, text)| (uri, mime, text))
        })
        .await
        .map_err(|_| ErrorData::internal_error("resource worker failed", None))?
        .map_err(|e| ErrorData::invalid_params(e, None))?;
        Ok(ReadResourceResult::new(vec![
            ResourceContents::text(result.2, result.0).with_mime_type(result.1)
        ])
        .into())
    }
}

#[cfg(test)]
mod tests;
