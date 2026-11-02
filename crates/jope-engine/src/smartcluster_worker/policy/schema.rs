use super::*;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerPolicy {
    pub root: PathBuf,
    pub state_root: PathBuf,
    pub daemon_socket: PathBuf,
    pub cgroup_root: PathBuf,
    pub daemon_uid: u32,
    pub daemon_sha256: String,
    pub worker_sha256: String,
    pub allowed_runtime_sha256: std::collections::BTreeSet<String>,
    pub cpu_evaluator: Option<PathBuf>,
    #[serde(default)]
    pub datasets_executable: Option<PathBuf>,
    #[serde(default)]
    pub datasets_sha256: Option<String>,
    #[serde(default)]
    pub approved_dataset_packers: BTreeMap<PathBuf, String>,
    #[serde(default)]
    pub dataset_verification_timeout_ms: Option<u64>,
    pub fit_authority: Option<PathBuf>,
    pub fit_service_socket: Option<PathBuf>,
    pub fit_scope: Option<String>,
    pub fit_witness: Option<PathBuf>,
    pub minimum_available_memory_bytes: u64,
    pub storage_reserves: BTreeMap<PathBuf, u64>,
    pub storage_devices: BTreeMap<PathBuf, PathBuf>,
    pub scratch_root: PathBuf,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputManifest {
    pub schema: String,
    pub root: PathBuf,
    pub operation: Operation,
    #[serde(default)]
    pub input_artifacts: BTreeMap<PathBuf, String>,
    #[serde(default)]
    pub closure: Option<ClosureManifest>,
    pub output_receipt: PathBuf,
}

/// Every input frozen before continuation declarations existed. It never
/// carries a declaration.
pub const INPUT_SCHEMA_V1: &str = "jope.smartcluster_input.v1";
/// A training input that declares its lineage continuation mode
/// (W6 follow-up). Only `Operation::Train` uses it, and it always declares.
pub const INPUT_SCHEMA_V2: &str = "jope.smartcluster_input.v2";

impl InputManifest {
    /// The schema/declaration pairing:
    /// - v1 carries no declaration;
    /// - v2 is a training input declaring a lineage mode, never `exact_resume`,
    ///   which is an attempt property.
    pub fn validate_version(&self) -> Result<()> {
        let declared = match &self.operation {
            Operation::Train {
                continuation_mode, ..
            } => *continuation_mode,
            _ => None,
        };
        match self.schema.as_str() {
            INPUT_SCHEMA_V1 => ensure!(
                declared.is_none(),
                "a continuation declaration requires {INPUT_SCHEMA_V2}"
            ),
            INPUT_SCHEMA_V2 => {
                ensure!(
                    matches!(self.operation, Operation::Train { .. }),
                    "{INPUT_SCHEMA_V2} is a training input"
                );
                let mode = declared.with_context(|| {
                    format!("{INPUT_SCHEMA_V2} must declare its continuation mode")
                })?;
                ensure!(
                    mode != crate::task_contract::ContinuationMode::ExactResume,
                    "exact resume is an attempt property; declare the lineage mode"
                );
            }
            other => anyhow::bail!("unknown input manifest schema {other}"),
        }
        Ok(())
    }

    /// The training input to freeze for a request, under the version it was
    /// first frozen with. `existing` holds the bytes already frozen at this
    /// input's path, if any:
    /// - absent: a new input, frozen as v2 with the declared `mode`;
    /// - a v1 input: re-materialized as v1, with no declaration, so its bytes
    ///   are unchanged;
    /// - a v2 input: re-materialized as v2 with the same declaration.
    ///
    /// `self` is the undeclared v1 form the materializer always built.
    pub fn declare_training(
        mut self,
        existing: Option<&[u8]>,
        mode: crate::task_contract::ContinuationMode,
    ) -> Result<Self> {
        ensure!(
            self.schema == INPUT_SCHEMA_V1,
            "a training input is declared from its undeclared form"
        );
        let Operation::Train {
            continuation_mode, ..
        } = &mut self.operation
        else {
            anyhow::bail!("only a training input declares a continuation mode")
        };
        ensure!(
            continuation_mode.is_none(),
            "a training input is declared from its undeclared form"
        );
        let version = match existing {
            None => INPUT_SCHEMA_V2.to_string(),
            Some(bytes) => {
                #[derive(Deserialize)]
                struct Version {
                    schema: String,
                }
                serde_json::from_slice::<Version>(bytes)
                    .context("frozen input manifest unreadable")?
                    .schema
            }
        };
        match version.as_str() {
            INPUT_SCHEMA_V1 => {}
            INPUT_SCHEMA_V2 => {
                *continuation_mode = Some(mode);
                self.schema = INPUT_SCHEMA_V2.into();
            }
            other => anyhow::bail!("unknown frozen input manifest schema {other}"),
        }
        self.validate_version()?;
        Ok(self)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClosureManifest {
    pub path: PathBuf,
    pub sha256: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ClosureEntry {
    path: PathBuf,
    sha256: String,
}

pub(super) fn read_closure(
    root: &Path,
    closure: &ClosureManifest,
) -> Result<BTreeMap<PathBuf, String>> {
    use std::io::BufRead;
    let path = resolve(root, &closure.path)?;
    let file = fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(&path)?;
    ensure!(
        file.metadata()?.is_file() && file.metadata()?.len() <= 256 * 1024 * 1024,
        "closure size/type invalid"
    );
    let mut reader = std::io::BufReader::new(file);
    let mut digest = sha2::Sha256::new();
    use sha2::Digest;
    let mut entries = BTreeMap::new();
    let mut total = 0usize;
    loop {
        let mut line = Vec::new();
        let bytes = reader.by_ref().take(8193).read_until(b'\n', &mut line)?;
        if bytes == 0 {
            break;
        }
        total += bytes;
        ensure!(
            bytes <= 8192 && total <= 256 * 1024 * 1024 && entries.len() < 1_000_000,
            "closure record/count/byte limit exceeded"
        );
        digest.update(&line);
        let entry: ClosureEntry = serde_json::from_slice(&line)?;
        relative(&entry.path)?;
        ensure!(
            entry.sha256.len() == 64
                && entry
                    .sha256
                    .bytes()
                    .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()),
            "closure digest invalid"
        );
        ensure!(
            entries.insert(entry.path, entry.sha256).is_none(),
            "duplicate closure path"
        );
    }
    ensure!(
        format!("{:x}", digest.finalize()) == closure.sha256,
        "closure bytes changed"
    );
    Ok(entries)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", deny_unknown_fields)]
pub enum Operation {
    Resident,
    DirectEvaluation {
        manifest: PathBuf,
        sha256: String,
    },
    SeededContinuation {
        manifest: PathBuf,
        sha256: String,
    },
    Search {
        prepared: PathBuf,
        cut: usize,
        output: PathBuf,
        evaluations: usize,
        seconds: u64,
        qualification: bool,
        seeds: Option<PathBuf>,
        vocabulary: Option<PathBuf>,
        rng_seed: u64,
    },
    Pack {
        contexts: PathBuf,
        vocabulary: PathBuf,
        observations: Vec<PathBuf>,
        output: PathBuf,
        qualification: bool,
    },
    DatasetPublication {
        manifest: PathBuf,
        sha256: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        logical_attempt: Option<u64>,
    },
    PackAndPublish {
        manifest: PathBuf,
        sha256: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        logical_attempt: Option<u64>,
    },
    Verify,
    Train {
        corpus: PathBuf,
        run: PathBuf,
        launch: PathBuf,
        micro: usize,
        checkpoint_steps: usize,
        initialize: PathBuf,
        replay_policy: PathBuf,
        initial_optimizer_step: u64,
        requested_step_increment: u64,
        target_optimizer_step: u64,
        /// Explicit lineage-mode declaration; absent on requests frozen before
        /// it, which keep their implicit meaning (W6 follow-up).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        continuation_mode: Option<crate::task_contract::ContinuationMode>,
    },
    Proposal {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        preparation_overlay: Option<crate::native_graph::Artifact>,
        champion_snapshot: PathBuf,
        corpus: PathBuf,
        checkpoint: PathBuf,
        output: PathBuf,
        cuts: usize,
        cut_start: usize,
        shuffled: bool,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SubtaskCommand {
    pub manifest: PathBuf,
    pub manifest_sha256: String,
}

impl Operation {
    pub(crate) fn dataset_logical_attempt(&self) -> Option<u64> {
        match self {
            Self::DatasetPublication {
                logical_attempt, ..
            }
            | Self::PackAndPublish {
                logical_attempt, ..
            } => *logical_attempt,
            _ => None,
        }
    }
    pub(crate) fn dataset_attempt(&self, native_number: u64) -> Result<u64> {
        let attempt = self.dataset_logical_attempt().unwrap_or(native_number);
        ensure!(attempt > 0, "dataset logical attempt must be positive");
        Ok(attempt)
    }
}

/// Opaque process-local proof; only verified native allocation dispatch constructs it.
pub struct VerifiedTask {
    #[cfg(feature = "neural")]
    pub(super) attempt: FencedAttempt,
    #[cfg(feature = "neural")]
    pub(super) manifest: TaskManifest,
    #[cfg(feature = "neural")]
    pub(super) root: PathBuf,
    pub(super) operation: Operation,
    pub(super) stop: Arc<AtomicBool>,
    #[cfg(feature = "neural")]
    pub(super) recovering: bool,
}
impl VerifiedTask {
    #[cfg(feature = "neural")]
    pub(crate) fn attempt(&self) -> FencedAttempt {
        self.attempt
    }
    #[cfg(feature = "neural")]
    pub(crate) fn manifest(&self) -> &TaskManifest {
        &self.manifest
    }
    #[cfg(feature = "neural")]
    pub(crate) fn root(&self) -> &Path {
        &self.root
    }
    pub(crate) fn operation(&self) -> &Operation {
        &self.operation
    }
    #[cfg(feature = "neural")]
    pub(crate) fn stop(&self) -> &AtomicBool {
        &self.stop
    }
    #[cfg(feature = "neural")]
    pub(crate) fn recovering(&self) -> bool {
        self.recovering
    }
}
