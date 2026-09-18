use std::{collections::BTreeMap, sync::Arc};

use async_trait::async_trait;
use jain_smartcluster_fabric::{sha256, valid_identifier, Digest32};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::{
    docker::{percent_encode, DockerError, DockerUnixEngine},
    ExecutionError, NetworkLease, NetworkPolicyPort,
};

const CLOSED_NETWORK_SCHEMA: &str = "jain.node.closed-network/v1";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DockerNetworkReceipt {
    pub id: String,
    pub name: String,
    pub driver: String,
    pub scope: String,
    pub internal: bool,
    pub attachable: bool,
    pub ingress: bool,
    pub config_only: bool,
    pub enable_ipv6: bool,
    pub labels: BTreeMap<String, String>,
}

#[async_trait]
pub trait DockerNetworkEnginePort: Send + Sync {
    async fn create_internal_network(
        &self,
        name: &str,
        labels: &BTreeMap<String, String>,
    ) -> Result<DockerNetworkReceipt, DockerError>;

    async fn remove_network(&self, network_id: &str) -> Result<(), DockerError>;
}

#[async_trait]
impl DockerNetworkEnginePort for DockerUnixEngine {
    async fn create_internal_network(
        &self,
        name: &str,
        labels: &BTreeMap<String, String>,
    ) -> Result<DockerNetworkReceipt, DockerError> {
        if !valid_docker_identifier(name) || labels.is_empty() {
            return Err(DockerError::MalformedResponse);
        }
        let payload = self
            .json_request(
                "POST",
                "/v1.47/networks/create",
                Some(&internal_network_create_body(name, labels)),
                &[201],
            )
            .await?;
        let created: CreateNetworkResponse =
            serde_json::from_slice(&payload).map_err(|_| DockerError::MalformedResponse)?;
        if !valid_docker_identifier(&created.id) {
            return Err(DockerError::MalformedResponse);
        }
        if created
            .warning
            .as_deref()
            .is_some_and(|warning| !warning.is_empty())
        {
            self.remove_network(&created.id).await?;
            return Err(DockerError::MalformedResponse);
        }
        let inspected = self
            .json_request(
                "GET",
                &format!("/v1.47/networks/{}", percent_encode(&created.id)),
                None,
                &[200],
            )
            .await
            .and_then(|payload| {
                serde_json::from_slice::<InspectNetworkResponse>(&payload)
                    .map_err(|_| DockerError::MalformedResponse)
            });
        let inspected = match inspected {
            Ok(inspected) => inspected,
            Err(error) => {
                self.remove_network(&created.id).await?;
                return Err(error);
            }
        };
        let receipt = DockerNetworkReceipt {
            id: inspected.id,
            name: inspected.name,
            driver: inspected.driver,
            scope: inspected.scope,
            internal: inspected.internal,
            attachable: inspected.attachable,
            ingress: inspected.ingress,
            config_only: inspected.config_only,
            enable_ipv6: inspected.enable_ipv6,
            labels: inspected.labels,
        };
        if receipt.id != created.id || validate_network_shape(&receipt, name, labels).is_err() {
            self.remove_network(&created.id).await?;
            return Err(DockerError::MalformedResponse);
        }
        Ok(receipt)
    }

    async fn remove_network(&self, network_id: &str) -> Result<(), DockerError> {
        if !valid_docker_identifier(network_id) {
            return Err(DockerError::MalformedResponse);
        }
        self.json_request(
            "DELETE",
            &format!("/v1.47/networks/{}", percent_encode(network_id)),
            None,
            &[204, 404],
        )
        .await
        .map(|_| ())
    }
}

fn internal_network_create_body(name: &str, labels: &BTreeMap<String, String>) -> Value {
    json!({
        "Name": name,
        "CheckDuplicate": true,
        "Driver": "bridge",
        "Internal": true,
        "Attachable": false,
        "Ingress": false,
        "EnableIPv6": false,
        "IPAM": {"Driver": "default", "Config": []},
        "Options": {},
        "Labels": labels
    })
}

fn valid_docker_identifier(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 256
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b':'))
}

#[derive(Deserialize)]
struct CreateNetworkResponse {
    #[serde(rename = "Id")]
    id: String,
    #[serde(rename = "Warning", default)]
    warning: Option<String>,
}

#[derive(Deserialize)]
struct InspectNetworkResponse {
    #[serde(rename = "Id")]
    id: String,
    #[serde(rename = "Name")]
    name: String,
    #[serde(rename = "Driver")]
    driver: String,
    #[serde(rename = "Scope")]
    scope: String,
    #[serde(rename = "Internal")]
    internal: bool,
    #[serde(rename = "Attachable")]
    attachable: bool,
    #[serde(rename = "Ingress")]
    ingress: bool,
    #[serde(rename = "ConfigOnly")]
    config_only: bool,
    #[serde(rename = "EnableIPv6")]
    enable_ipv6: bool,
    #[serde(rename = "Labels", default)]
    labels: BTreeMap<String, String>,
}

#[derive(Clone, Debug)]
pub struct DockerNetworkPolicy<D: DockerNetworkEnginePort> {
    engine: Arc<D>,
}

impl<D: DockerNetworkEnginePort> DockerNetworkPolicy<D> {
    pub fn new(engine: Arc<D>) -> Self {
        Self { engine }
    }
}

#[derive(Serialize)]
struct ClosedNetworkBinding<'a> {
    schema: &'static str,
    run_id: &'a str,
    network_name: &'a str,
    open_outbound: bool,
    byte_limit: u64,
    connection_limit: u32,
    deadline_ms: u64,
}

#[async_trait]
impl<D> NetworkPolicyPort for DockerNetworkPolicy<D>
where
    D: DockerNetworkEnginePort + 'static,
{
    async fn allocate(
        &self,
        run_id: &str,
        open_outbound: bool,
        byte_limit: u64,
        connection_limit: u32,
        deadline_ms: u64,
    ) -> Result<NetworkLease, ExecutionError> {
        if !valid_identifier(run_id)
            || open_outbound
            || byte_limit != 0
            || connection_limit != 0
            || deadline_ms != 0
        {
            return Err(ExecutionError::NetworkPolicyUnavailable);
        }
        let name = format!("jain-{run_id}-{}", ulid::Ulid::new());
        if !valid_identifier(&name) {
            return Err(ExecutionError::NetworkPolicyUnavailable);
        }
        let policy_digest = closed_policy_digest(run_id, &name)?;
        let labels = BTreeMap::from([
            ("org.jain.fabric".to_owned(), "governed-rd-v1".to_owned()),
            ("org.jain.run-id".to_owned(), run_id.to_owned()),
            ("org.jain.egress".to_owned(), "denied".to_owned()),
            (
                "org.jain.network-policy-digest".to_owned(),
                hex_digest(policy_digest),
            ),
        ]);
        let receipt = self.engine.create_internal_network(&name, &labels).await?;
        if validate_receipt(&receipt, &name, &labels, policy_digest).is_err() {
            if valid_identifier(&receipt.id) {
                self.engine.remove_network(&receipt.id).await?;
            }
            return Err(ExecutionError::NetworkPolicyUnavailable);
        }
        Ok(NetworkLease {
            run_id: run_id.to_owned(),
            name,
            network_id: receipt.id,
            policy_digest,
        })
    }

    async fn revoke(&self, lease: &NetworkLease) -> Result<(), ExecutionError> {
        if !valid_identifier(&lease.run_id)
            || !valid_identifier(&lease.name)
            || !valid_identifier(&lease.network_id)
            || lease.policy_digest == [0; 32]
            || !lease.name.starts_with(&format!("jain-{}-", lease.run_id))
            || closed_policy_digest(&lease.run_id, &lease.name)? != lease.policy_digest
        {
            return Err(ExecutionError::NetworkPolicyUnavailable);
        }
        self.engine.remove_network(&lease.network_id).await?;
        Ok(())
    }
}

fn closed_policy_digest(run_id: &str, network_name: &str) -> Result<Digest32, ExecutionError> {
    let bytes = postcard::to_stdvec(&ClosedNetworkBinding {
        schema: CLOSED_NETWORK_SCHEMA,
        run_id,
        network_name,
        open_outbound: false,
        byte_limit: 0,
        connection_limit: 0,
        deadline_ms: 0,
    })
    .map_err(|_| ExecutionError::NetworkPolicyUnavailable)?;
    Ok(sha256(&bytes))
}

fn validate_receipt(
    receipt: &DockerNetworkReceipt,
    expected_name: &str,
    expected_labels: &BTreeMap<String, String>,
    expected_digest: Digest32,
) -> Result<(), ExecutionError> {
    if !valid_identifier(&receipt.id)
        || validate_network_shape(receipt, expected_name, expected_labels).is_err()
        || expected_digest == [0; 32]
    {
        return Err(ExecutionError::NetworkPolicyUnavailable);
    }
    Ok(())
}

fn validate_network_shape(
    receipt: &DockerNetworkReceipt,
    expected_name: &str,
    expected_labels: &BTreeMap<String, String>,
) -> Result<(), DockerError> {
    if receipt.name != expected_name
        || receipt.driver != "bridge"
        || receipt.scope != "local"
        || !receipt.internal
        || receipt.attachable
        || receipt.ingress
        || receipt.config_only
        || receipt.enable_ipv6
        || receipt.labels != *expected_labels
    {
        return Err(DockerError::MalformedResponse);
    }
    Ok(())
}

fn hex_digest(digest: Digest32) -> String {
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use std::{path::Path, sync::Mutex, time::Duration};

    use crate::docker::DockerError;
    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        net::{UnixListener, UnixStream},
    };

    use super::*;

    #[derive(Default)]
    struct NetworkEngineMock {
        create_calls: Mutex<usize>,
        removals: Mutex<Vec<String>>,
        corrupt_readback: bool,
    }

    #[async_trait]
    impl DockerNetworkEnginePort for NetworkEngineMock {
        async fn create_internal_network(
            &self,
            name: &str,
            labels: &BTreeMap<String, String>,
        ) -> Result<DockerNetworkReceipt, DockerError> {
            *self.create_calls.lock().unwrap() += 1;
            Ok(DockerNetworkReceipt {
                id: "network-id-1".to_owned(),
                name: name.to_owned(),
                driver: "bridge".to_owned(),
                scope: "local".to_owned(),
                internal: !self.corrupt_readback,
                attachable: false,
                ingress: false,
                config_only: false,
                enable_ipv6: false,
                labels: labels.clone(),
            })
        }

        async fn remove_network(&self, network_id: &str) -> Result<(), DockerError> {
            self.removals.lock().unwrap().push(network_id.to_owned());
            Ok(())
        }
    }

    #[test]
    fn internal_network_request_has_no_ingress_or_attachment_escape() {
        let labels = BTreeMap::from([
            ("org.jain.egress".to_owned(), "denied".to_owned()),
            ("org.jain.run-id".to_owned(), "run-1".to_owned()),
        ]);
        let body = internal_network_create_body("jain-run-1", &labels);
        assert_eq!(body.pointer("/Name"), Some(&json!("jain-run-1")));
        assert_eq!(body.pointer("/Driver"), Some(&json!("bridge")));
        assert_eq!(body.pointer("/Internal"), Some(&json!(true)));
        assert_eq!(body.pointer("/Attachable"), Some(&json!(false)));
        assert_eq!(body.pointer("/Ingress"), Some(&json!(false)));
        assert_eq!(body.pointer("/EnableIPv6"), Some(&json!(false)));
        assert_eq!(
            body.pointer("/Labels/org.jain.egress"),
            Some(&json!("denied"))
        );
        assert!(body.get("EnableIPv4").is_none());
    }

    async fn read_request(stream: &mut UnixStream) -> Vec<u8> {
        let mut request = Vec::new();
        let mut buffer = [0_u8; 4096];
        loop {
            let count = stream.read(&mut buffer).await.unwrap();
            assert_ne!(count, 0, "request ended before its declared body");
            request.extend_from_slice(&buffer[..count]);
            let Some(header_end) = request.windows(4).position(|part| part == b"\r\n\r\n") else {
                continue;
            };
            let header = std::str::from_utf8(&request[..header_end]).unwrap();
            let content_length = header
                .lines()
                .find_map(|line| line.strip_prefix("Content-Length: "))
                .unwrap()
                .parse::<usize>()
                .unwrap();
            if request.len() == header_end + 4 + content_length {
                return request;
            }
            assert!(request.len() < header_end + 4 + content_length);
        }
    }

    async fn serve_network_lifecycle(socket: &Path) -> Vec<Vec<u8>> {
        let listener = UnixListener::bind(socket).unwrap();
        let mut requests = Vec::new();

        let (mut create_stream, _) = listener.accept().await.unwrap();
        let create = read_request(&mut create_stream).await;
        let create_body = &create[create
            .windows(4)
            .position(|part| part == b"\r\n\r\n")
            .unwrap()
            + 4..];
        let create_json: Value = serde_json::from_slice(create_body).unwrap();
        let network_name = create_json["Name"].as_str().unwrap().to_owned();
        let labels = create_json["Labels"].clone();
        let create_response = r#"{"Id":"network-id-1","Warning":""}"#;
        create_stream
            .write_all(
                format!(
                    "HTTP/1.1 201 Created\r\nContent-Length: {}\r\n\r\n{create_response}",
                    create_response.len()
                )
                .as_bytes(),
            )
            .await
            .unwrap();
        create_stream.shutdown().await.unwrap();
        requests.push(create);

        let (mut inspect_stream, _) = listener.accept().await.unwrap();
        let inspect = read_request(&mut inspect_stream).await;
        let inspect_body = json!({
            "Id": "network-id-1",
            "Name": network_name,
            "Driver": "bridge",
            "Scope": "local",
            "Internal": true,
            "Attachable": false,
            "Ingress": false,
            "ConfigOnly": false,
            "EnableIPv6": false,
            "Labels": labels
        })
        .to_string();
        inspect_stream
            .write_all(
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n{inspect_body}",
                    inspect_body.len()
                )
                .as_bytes(),
            )
            .await
            .unwrap();
        inspect_stream.shutdown().await.unwrap();
        requests.push(inspect);

        let (mut delete_stream, _) = listener.accept().await.unwrap();
        let delete = read_request(&mut delete_stream).await;
        delete_stream
            .write_all(b"HTTP/1.1 204 No Content\r\nContent-Length: 0\r\n\r\n")
            .await
            .unwrap();
        delete_stream.shutdown().await.unwrap();
        requests.push(delete);
        requests
    }

    #[tokio::test]
    async fn unix_engine_creates_inspects_and_deletes_exact_internal_network() {
        let root = tempfile::tempdir().unwrap();
        let socket = root.path().join("docker.sock");
        let server_socket = socket.clone();
        let server = tokio::spawn(async move { serve_network_lifecycle(&server_socket).await });
        while !socket.exists() {
            tokio::task::yield_now().await;
        }
        let engine = DockerUnixEngine::new(socket, Duration::from_secs(1)).unwrap();
        let policy = DockerNetworkPolicy::new(Arc::new(engine));
        let lease = policy.allocate("01HZXNODE", false, 0, 0, 0).await.unwrap();
        policy.revoke(&lease).await.unwrap();

        let requests = server.await.unwrap();
        assert!(requests[0].starts_with(b"POST /v1.47/networks/create HTTP/1.1\r\n"));
        assert!(requests[1].starts_with(b"GET /v1.47/networks/network-id-1 HTTP/1.1\r\n"));
        assert!(requests[2].starts_with(b"DELETE /v1.47/networks/network-id-1 HTTP/1.1\r\n"));
    }

    #[tokio::test]
    async fn closed_egress_is_exactly_bound_and_revoked_by_engine_id() {
        let engine = Arc::new(NetworkEngineMock::default());
        let policy = DockerNetworkPolicy::new(Arc::clone(&engine));
        let lease = policy.allocate("01HZXNODE", false, 0, 0, 0).await.unwrap();
        assert_eq!(lease.run_id, "01HZXNODE");
        assert!(lease.name.starts_with("jain-01HZXNODE-"));
        assert_eq!(lease.network_id, "network-id-1");
        assert_ne!(lease.policy_digest, [0; 32]);
        policy.revoke(&lease).await.unwrap();
        assert_eq!(*engine.create_calls.lock().unwrap(), 1);
        assert_eq!(
            *engine.removals.lock().unwrap(),
            vec!["network-id-1".to_owned()]
        );

        let mut tampered = lease;
        tampered.policy_digest = [8; 32];
        assert!(policy.revoke(&tampered).await.is_err());
        assert_eq!(engine.removals.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn open_or_nonzero_closed_budget_fails_before_docker() {
        let engine = Arc::new(NetworkEngineMock::default());
        let policy = DockerNetworkPolicy::new(Arc::clone(&engine));
        for request in [
            (true, 1, 1, 1),
            (false, 1, 0, 0),
            (false, 0, 1, 0),
            (false, 0, 0, 1),
        ] {
            assert!(policy
                .allocate("01HZXNODE", request.0, request.1, request.2, request.3)
                .await
                .is_err());
        }
        assert_eq!(*engine.create_calls.lock().unwrap(), 0);
    }

    #[tokio::test]
    async fn mismatched_readback_is_deleted_before_failure() {
        let engine = Arc::new(NetworkEngineMock {
            corrupt_readback: true,
            ..NetworkEngineMock::default()
        });
        let policy = DockerNetworkPolicy::new(Arc::clone(&engine));
        assert!(policy.allocate("01HZXNODE", false, 0, 0, 0).await.is_err());
        assert_eq!(
            *engine.removals.lock().unwrap(),
            vec!["network-id-1".to_owned()]
        );
    }
}
