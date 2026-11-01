//! Bounded stdout relay around the SDK's independent read/dispatch/write implementation.
use super::{Adapter, MAX_FRAME};
use jope_client::Transport;
use rmcp::{
    model::{
        ClientJsonRpcMessage, ClientNotification, ClientRequest, ProtocolVersion,
        ServerJsonRpcMessage,
    },
    transport::{async_rw::AsyncRwTransport, Transport as SdkTransport},
    RoleServer, ServiceExt,
};
use std::future::Future;
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, BufReader};

/// Discovery is a probe, not a selected lifecycle. Some clients publish a legacy
/// roots hint immediately after a valid modern discovery request. This adapter
/// never consumes client roots: ignore only that hint during bootstrap, while
/// leaving lifecycle selection and all request validation with rmcp.
struct BootstrapTransport<T> {
    inner: T,
    discovery_seen: bool,
    bootstrapping: bool,
}

impl<T> BootstrapTransport<T> {
    fn new(inner: T) -> Self {
        Self {
            inner,
            discovery_seen: false,
            bootstrapping: true,
        }
    }
}

impl<T: SdkTransport<RoleServer>> SdkTransport<RoleServer> for BootstrapTransport<T> {
    type Error = T::Error;

    fn send(
        &mut self,
        item: ServerJsonRpcMessage,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send + 'static {
        self.inner.send(item)
    }

    async fn receive(&mut self) -> Option<ClientJsonRpcMessage> {
        loop {
            let message = self.inner.receive().await?;
            if !self.bootstrapping {
                return Some(message);
            }
            match &message {
                ClientJsonRpcMessage::Request(request) => {
                    let meta = request.request.get_meta();
                    let modern = meta
                        .protocol_version()
                        .is_some_and(|version| version.as_str() == "2026-07-28")
                        && meta
                            .missing_required_keys(&ProtocolVersion::V_2026_07_28)
                            .is_empty();
                    match &request.request {
                        ClientRequest::DiscoverRequest(_) if modern => self.discovery_seen = true,
                        ClientRequest::InitializeRequest(_) => self.bootstrapping = false,
                        ClientRequest::PingRequest(_) | ClientRequest::DiscoverRequest(_) => {}
                        _ if modern => self.bootstrapping = false,
                        _ => {}
                    }
                }
                ClientJsonRpcMessage::Notification(
                    ClientNotification::RootsListChangedNotification(_),
                ) if self.bootstrapping && self.discovery_seen => continue,
                _ => {}
            }
            return Some(message);
        }
    }

    fn close(&mut self) -> impl Future<Output = Result<(), Self::Error>> + Send {
        self.inner.close()
    }
}

pub async fn serve<T, R, W>(adapter: Adapter<T>, input: R, output: W) -> anyhow::Result<()>
where
    T: Transport + Send + Sync + 'static,
    R: AsyncRead + Send + Unpin + 'static,
    W: AsyncWrite + Send + Unpin + 'static,
{
    // SDK writes may apply backpressure, but the input and dispatch futures remain independent.
    let (sdk_output, wire_output) = tokio::io::duplex(64 * 1024);
    let service = async move {
        let service = adapter
            .serve(BootstrapTransport::new(
                AsyncRwTransport::<RoleServer, _, _>::new(input, sdk_output)
                    .with_max_line_length(MAX_FRAME),
            ))
            .await?;
        service.waiting().await?;
        Ok::<(), anyhow::Error>(())
    };
    tokio::try_join!(service, relay(wire_output, output))?;
    Ok(())
}

pub(super) async fn relay<R: AsyncRead + Unpin, W: AsyncWrite + Unpin>(
    input: R,
    mut output: W,
) -> anyhow::Result<()> {
    let mut input = BufReader::new(input);
    loop {
        let mut frame = Vec::new();
        (&mut input)
            .take((MAX_FRAME + 1) as u64)
            .read_until(b'\n', &mut frame)
            .await?;
        if frame.is_empty() {
            return Ok(());
        }
        if frame.len() > MAX_FRAME || !frame.ends_with(b"\n") {
            anyhow::bail!("outgoing MCP frame exceeds limit or is incomplete");
        }
        let value: serde_json::Value = serde_json::from_slice(&frame)?;
        if !value.is_object() || value["jsonrpc"] != "2.0" {
            anyhow::bail!("non-JSON-RPC bytes refused on MCP stdout");
        }
        output.write_all(&frame).await?;
        output.flush().await?;
    }
}
