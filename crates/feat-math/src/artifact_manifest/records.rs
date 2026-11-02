use super::*;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileBinding {
    pub path: String,
    pub bytes: u64,
    pub sha256: String,
}
impl FileBinding {
    pub fn new(path: &str, bytes: &[u8]) -> Self {
        Self {
            path: path.into(),
            bytes: bytes.len() as u64,
            sha256: sha256(bytes),
        }
    }
}

/// A recipe only. No compiler version or executable parity is asserted here.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FittedIdentity {
    pub state_sha256: String,
    pub input_sha256: String,
    pub task_sha256: String,
    pub schema_sha256: String,
    pub vocabulary_sha256: String,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum TargetState {
    Supported {
        product: String,
        emitter: String,
        files: Vec<FileBinding>,
        build: BuildIdentity,
        fitted: Option<Box<FittedIdentity>>,
    },
    Unsupported {
        reason: String,
        registry_sha256: String,
    },
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TargetRecord {
    pub target: Target,
    pub state: TargetState,
}
impl TargetRecord {
    pub fn unsupported(target: Target, reason: String) -> Self {
        Self {
            target,
            state: TargetState::Unsupported {
                reason,
                registry_sha256: crate::kernel::registry_v2::source_sha256(),
            },
        }
    }
    pub fn supported(
        target: Target,
        dialect: &str,
        emitter: &str,
        sources: &[(&str, &str)],
        fitted: Option<FittedIdentity>,
    ) -> Self {
        Self {
            target,
            state: TargetState::Supported {
                product: target.product(dialect).into(),
                emitter: emitter.into(),
                files: sources
                    .iter()
                    .map(|(p, s)| FileBinding::new(p, s.as_bytes()))
                    .collect(),
                build: BuildIdentity::declared(target),
                fitted: fitted.map(Box::new),
            },
        }
    }
}
