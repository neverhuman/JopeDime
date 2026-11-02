use crate::task::Task;
use anyhow::{ensure, Context, Result};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const TASK_HEAD_POLICY_SCHEMA: &str = "JopeTaskHeadPolicy.v1";
pub const TASK_HEAD_POLICY_JSON: &str =
    include_str!("../manifests/epoch6/JopeTaskHeadPolicy.v1.json");
pub const TASK_HEAD_POLICY_SHA256: &str =
    "fa52320db524d316ef41d878a8c7dd2930cb1f8d989f05dda6a2c4e29a0927eb";

#[derive(Clone, Debug, Deserialize)]
struct GeneratedRegistryDocument {
    registry_version: String,
    registry_hash: String,
    maximum_output_columns: u32,
    operators: Vec<GeneratedOperator>,
}

#[derive(Clone, Debug, Deserialize)]
pub(super) struct GeneratedOperator {
    id: u32,
    pub(super) name: String,
    pub(super) role: String,
    #[serde(default)]
    pub(super) target_side: bool,
    #[serde(default)]
    pub(super) regression: bool,
    #[serde(default)]
    pub(super) estimated_cost: u64,
    pub(super) lifecycle: GeneratedOperatorLifecycle,
}

#[derive(Clone, Debug, Deserialize)]
pub(super) struct GeneratedOperatorLifecycle {
    pub(super) parseable: bool,
    #[serde(default)]
    pub(super) quarantined: bool,
    #[serde(default)]
    pub(super) deprecated: bool,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GeneratedTaskHeadPolicyDocument {
    schema: String,
    heads: Vec<GeneratedTaskHeadPolicy>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct GeneratedTaskHeadPolicy {
    operator: String,
    pub(super) tasks: Vec<Task>,
    pub(super) estimated_cost: u64,
}

#[derive(Clone, Debug)]
pub struct GeneratedOperatorRegistry {
    pub(super) sha256: String,
    operators: BTreeMap<String, GeneratedOperator>,
    heads: BTreeMap<String, GeneratedTaskHeadPolicy>,
}

impl GeneratedOperatorRegistry {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let document: GeneratedRegistryDocument =
            serde_json::from_slice(bytes).context("decode generated operator registry")?;
        ensure!(
            !document.registry_version.trim().is_empty()
                && is_sha256(&document.registry_hash)
                && document.maximum_output_columns > 0
                && !document.operators.is_empty(),
            "generated operator registry header is invalid"
        );
        let mut ids = BTreeSet::new();
        let mut operators = BTreeMap::new();
        for operator in document.operators {
            ensure!(
                ids.insert(operator.id)
                    && !operator.name.trim().is_empty()
                    && !operator.role.trim().is_empty()
                    && operators.insert(operator.name.clone(), operator).is_none(),
                "generated operator registry repeats an operator identity"
            );
        }
        let head_document: GeneratedTaskHeadPolicyDocument =
            serde_json::from_str(TASK_HEAD_POLICY_JSON)
                .context("decode generated task/head policy")?;
        ensure!(
            hex::encode(Sha256::digest(TASK_HEAD_POLICY_JSON.as_bytes()))
                == TASK_HEAD_POLICY_SHA256,
            "generated task/head policy digest drift"
        );
        ensure!(
            head_document.schema == TASK_HEAD_POLICY_SCHEMA && !head_document.heads.is_empty(),
            "generated task/head policy header is invalid"
        );
        let mut heads = BTreeMap::new();
        for head in head_document.heads {
            ensure!(
                !head.operator.trim().is_empty()
                    && !head.tasks.is_empty()
                    && head.estimated_cost > 0
                    && head.tasks.iter().copied().collect::<BTreeSet<_>>().len()
                        == head.tasks.len()
                    && heads.insert(head.operator.clone(), head).is_none(),
                "generated task/head policy repeats or invalidates a head"
            );
        }
        for operator in operators
            .values()
            .filter(|operator| operator.role == "head")
        {
            ensure!(
                heads.contains_key(&operator.name),
                "generated operator head {} lacks task policy",
                operator.name
            );
        }
        Ok(Self {
            sha256: hex::encode(Sha256::digest(bytes)),
            operators,
            heads,
        })
    }

    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    pub(super) fn get(&self, name: &str) -> Option<&GeneratedOperator> {
        self.operators.get(name)
    }

    pub(super) fn head(&self, name: &str) -> Option<&GeneratedTaskHeadPolicy> {
        self.heads.get(name)
    }
}
