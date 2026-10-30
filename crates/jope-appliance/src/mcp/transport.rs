//! Bounded stdout relay around the SDK's independent read/dispatch/write implementation.
use super::{Adapter, MAX_FRAME};
use jope_client::Transport;
use rmcp::{transport::async_rw::AsyncRwTransport, RoleServer, ServiceExt};
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, BufReader};

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
            .serve(
                AsyncRwTransport::<RoleServer, _, _>::new(input, sdk_output)
                    .with_max_line_length(MAX_FRAME),
            )
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
