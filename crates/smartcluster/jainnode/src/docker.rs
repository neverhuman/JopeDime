use std::{collections::BTreeMap, path::PathBuf, time::Duration};

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use thiserror::Error;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::UnixStream,
    sync::mpsc,
};

const MAX_DOCKER_RESPONSE: usize = 8 * 1024 * 1024;
const MAX_HEADER_BYTES: usize = 32 * 1024;

mod engine;
mod protocol;

pub use engine::*;
pub(crate) use protocol::percent_encode;
pub(super) use protocol::{
    decode_http_body, drain_chunked, drain_content_length, find_subsequence,
    parse_http_body_framing, take_multiplexed_logs, CreateResponse, HttpBodyFraming,
    InspectResponse, WaitResponse,
};

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::net::UnixListener;

    #[test]
    fn create_body_is_non_privileged_readonly_and_device_exact() {
        let spec = DockerRunSpec {
            name: "jain-run-1".to_owned(),
            image: "registry.invalid/x@sha256:00".to_owned(),
            argv: vec!["/app/run".to_owned()],
            working_directory: "work".to_owned(),
            environment: BTreeMap::new(),
            readonly_binds: vec![(PathBuf::from("/staging/input"), "/inputs/data".to_owned())],
            output_staging: PathBuf::from("/staging/output"),
            scratch_bytes: 1_000,
            cpu_millicores: 500,
            memory_bytes: 2_000,
            pids: 32,
            read_bytes: 1_000,
            write_bytes: 1_000,
            io_operations: 100,
            gpu_uuids: vec!["GPU-exact".to_owned()],
            network_name: "jain-net-1".to_owned(),
            seccomp_profile: "/etc/jain/seccomp.json".to_owned(),
            apparmor_profile: "jain-governed-rd".to_owned(),
        };
        let body = spec.create_body();
        assert_eq!(body.pointer("/HostConfig/Privileged"), Some(&json!(false)));
        assert_eq!(
            body.pointer("/HostConfig/ReadonlyRootfs"),
            Some(&json!(true))
        );
        assert_eq!(
            body.pointer("/HostConfig/NetworkMode"),
            Some(&json!("jain-net-1"))
        );
        assert_eq!(
            body.pointer("/HostConfig/DeviceRequests/0/DeviceIDs/0"),
            Some(&json!("GPU-exact"))
        );
        let encoded = body.to_string();
        assert!(!encoded.contains("docker.sock"));
        assert!(!encoded.contains("\"PidMode\":\"host\""));
        assert!(!encoded.contains("\"NetworkMode\":\"host\""));
    }

    #[test]
    fn multiplexed_frames_are_bounded_and_ordered() {
        let mut bytes = vec![1, 0, 0, 0, 0, 0, 0, 3];
        bytes.extend_from_slice(b"out");
        bytes.extend_from_slice(&[2, 0, 0, 0, 0, 0, 0, 3]);
        bytes.extend_from_slice(b"err");
        let chunks = take_multiplexed_logs(&mut bytes).unwrap();
        assert_eq!(
            chunks[0],
            DockerLogChunk {
                stderr: false,
                bytes: b"out".to_vec()
            }
        );
        assert_eq!(
            chunks[1],
            DockerLogChunk {
                stderr: true,
                bytes: b"err".to_vec()
            }
        );
        let mut oversized = vec![1, 0, 0, 0, 0, 0, 0, 0];
        oversized[4..8].copy_from_slice(&(65_u32 * 1024).to_be_bytes());
        assert!(take_multiplexed_logs(&mut oversized).is_err());
    }

    #[test]
    fn streaming_chunk_and_multiplex_decoders_preserve_partial_frames() {
        let frame = [1_u8, 0, 0, 0, 0, 0, 0, 5, b'h', b'e', b'l', b'l', b'o'];
        let mut wire = format!("{:X}\r\n", frame.len()).into_bytes();
        wire.extend_from_slice(&frame[..6]);
        let mut payload = Vec::new();
        assert!(!drain_chunked(&mut wire, &mut payload).unwrap());
        assert!(payload.is_empty());
        wire.extend_from_slice(&frame[6..]);
        wire.extend_from_slice(b"\r\n0\r\n\r\n");
        assert!(drain_chunked(&mut wire, &mut payload).unwrap());
        let first = payload.drain(..6).collect::<Vec<_>>();
        let mut partial = first.clone();
        assert!(take_multiplexed_logs(&mut partial).unwrap().is_empty());
        let mut joined = first;
        joined.extend_from_slice(&payload);
        let decoded = take_multiplexed_logs(&mut joined).unwrap();
        assert_eq!(decoded[0].bytes, b"hello");
        assert!(joined.is_empty());

        let mut trailing = b"0\r\n\r\nextra".to_vec();
        assert!(drain_chunked(&mut trailing, &mut Vec::new()).is_err());
    }

    #[test]
    fn ordinary_responses_require_exact_unambiguous_framing() {
        let exact = "HTTP/1.1 200 OK\r\nContent-Length: 2";
        assert_eq!(decode_http_body(exact, b"{}", 200).unwrap(), b"{}");
        assert!(decode_http_body(exact, b"{", 200).is_err());
        assert!(decode_http_body(exact, b"{}x", 200).is_err());
        assert!(decode_http_body("HTTP/1.1 200 OK", b"{}", 200).is_err());
        assert!(decode_http_body(
            "HTTP/1.1 200 OK\r\nContent-Length: 2\r\nContent-Length: 2",
            b"{}",
            200,
        )
        .is_err());
        assert!(decode_http_body(
            "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nContent-Length: 2",
            b"0\r\n\r\n",
            200,
        )
        .is_err());
        assert!(decode_http_body("HTTP/1.1 204 No Content", b"", 204).is_ok());
    }

    #[test]
    fn chunked_responses_require_complete_zero_terminator_and_no_trailing_bytes() {
        let header = "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked";
        assert_eq!(
            decode_http_body(header, b"2\r\n{}\r\n0\r\n\r\n", 200).unwrap(),
            b"{}"
        );
        assert!(decode_http_body(header, b"2\r\n{}\r\n0\r\n", 200).is_err());
        assert!(decode_http_body(header, b"2\r\n{}\r\n0\r\n\r\nextra", 200).is_err());
    }

    async fn log_events_for_response(
        response: Vec<u8>,
    ) -> Vec<Result<DockerLogEvent, DockerError>> {
        let root = tempfile::tempdir().unwrap();
        let socket = root.path().join("docker.sock");
        let listener = UnixListener::bind(&socket).unwrap();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = [0_u8; 4096];
            stream.read(&mut request).await.unwrap();
            stream.write_all(&response).await.unwrap();
            stream.shutdown().await.unwrap();
        });
        let engine = DockerUnixEngine::new(socket, Duration::from_secs(1)).unwrap();
        let mut receiver = engine.logs("container-1").await.unwrap();
        let mut events = Vec::new();
        while let Some(event) = receiver.recv().await {
            events.push(event);
        }
        server.await.unwrap();
        events
    }

    #[tokio::test]
    async fn live_log_transport_requires_exact_unambiguous_framing() {
        for response in [
            b"HTTP/1.1 200 OK\r\nContent-Length: 8\r\n\r\n".to_vec(),
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nContent-Length: 0\r\n\r\n".to_vec(),
            b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nContent-Length: 0\r\n\r\n".to_vec(),
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: gzip\r\n\r\n".to_vec(),
            b"HTTP/1.1 200 OK\r\n\r\n".to_vec(),
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n".to_vec(),
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\nextra".to_vec(),
        ] {
            let events = log_events_for_response(response).await;
            assert!(events.iter().any(Result::is_err), "events={events:?}");
            assert!(!events
                .iter()
                .any(|event| matches!(event, Ok(DockerLogEvent::Complete))));
        }

        let multiplexed = [1, 0, 0, 0, 0, 0, 0, 2, b'o', b'k'];
        let mut exact = b"HTTP/1.1 200 OK\r\nContent-Length: 10\r\n\r\n".to_vec();
        exact.extend_from_slice(&multiplexed);
        let events = log_events_for_response(exact).await;
        assert!(matches!(
            events.as_slice(),
            [
                Ok(DockerLogEvent::Chunk(DockerLogChunk { stderr: false, bytes })),
                Ok(DockerLogEvent::Complete)
            ] if bytes == b"ok"
        ));
    }
}
