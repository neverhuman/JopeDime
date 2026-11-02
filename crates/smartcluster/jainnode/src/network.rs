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
mod tests;
