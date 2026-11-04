const MAX_FRAME: usize = 65536;
const SEALS: i32 = libc::F_SEAL_WRITE | libc::F_SEAL_GROW | libc::F_SEAL_SHRINK | libc::F_SEAL_SEAL;
fn sha(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}
fn lock<T>(value: &Mutex<T>) -> Result<std::sync::MutexGuard<'_, T>> {
    value
        .lock()
        .map_err(|_| anyhow::anyhow!("authority service mutex poisoned"))
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServiceInputs {
    pub program: String,
    pub x: Vec<Vec<f64>>,
    pub y: Vec<f64>,
    pub preprocessing: Vec<u8>,
    pub feature_view: Vec<u8>,
}
impl ServiceInputs {
    fn verify(
        &self,
        executable: &str,
        runtime: &str,
        config: &ServiceConfig,
    ) -> Result<VerifiedFit> {
        VerifiedFit::from_verified_runtime(
            &self.program,
            crate::eval_registry::FitInputs {
                x: self.x.clone(),
                y: self.y.clone(),
                preprocessing: self.preprocessing.clone(),
                feature_view: self.feature_view.clone(),
            },
            config.fit_seed,
            executable.into(),
            runtime.into(),
            &config.contract,
        )
    }
}

#[cfg(test)]
#[derive(Default)]
pub(super) struct TestPause {
    reached: AtomicBool,
    release: AtomicBool,
}

#[derive(Clone)]
pub struct ServiceConfig {
    pub authority_root: PathBuf,
    pub witness: PathBuf,
    pub socket: PathBuf,
    pub worker_executable: PathBuf,
    pub worker_arguments: Vec<String>,
    pub contract: NumericalContract,
    pub fit_seed: u64,
    pub max_workers: usize,
    pub fit_deadline: Duration,
    pub handshake_timeout: Duration,
    pub max_descriptor_bytes: u64,
    pub event_witness: Option<super::witness::WitnessClientConfig>,
    pub funding_config: Option<PathBuf>,
    pub native_fit_policy: Option<PathBuf>,
    pub sandbox: WorkerSandbox,
    #[cfg(test)]
    pub(super) funding_fixture: Option<super::funding_config::InstalledFunding>,
    #[cfg(test)]
    pub(super) drop_funding_reply: bool,
    #[cfg(test)]
    pub(super) drop_start_reply: bool,
    #[cfg(test)]
    pub(super) before_authorize: Option<Arc<TestPause>>,
    #[cfg(test)]
    pub(super) synthetic_query_access: std::collections::BTreeSet<String>,
    #[cfg(test)]
    pub(super) query_fail_after_accept: bool,
    #[cfg(test)]
    pub(super) query_worker_arguments: Vec<String>,
    #[cfg(test)]
    pub(super) query_worker_sleep_ms: Option<u64>,
}
impl ServiceConfig {
    pub fn regression(
        authority_root: PathBuf,
        witness: PathBuf,
        socket: PathBuf,
        worker_executable: PathBuf,
        max_workers: usize,
    ) -> Self {
        Self {
            authority_root,
            witness,
            socket,
            worker_executable,
            worker_arguments: vec!["registry-service-fit".into()],
            contract: NumericalContract {
                protocol: "featmath_fixed_seed0_f64_single_thread_v1".into(),
                precision: "f64".into(),
                omp_threads: 1,
                rayon_threads: 1,
                blas_threads: 1,
                mkl_threads: 1,
            },
            fit_seed: 0,
            max_workers,
            fit_deadline: Duration::from_secs(30),
            handshake_timeout: Duration::from_secs(30),
            max_descriptor_bytes: 2 * 1024 * 1024 * 1024,
            event_witness: None,
            funding_config: None,
            native_fit_policy: None,
            sandbox: WorkerSandbox::default(),
            #[cfg(test)]
            funding_fixture: None,
            #[cfg(test)]
            drop_funding_reply: false,
            #[cfg(test)]
            drop_start_reply: false,
            #[cfg(test)]
            before_authorize: None,
            #[cfg(test)]
            synthetic_query_access: Default::default(),
            #[cfg(test)]
            query_fail_after_accept: false,
            #[cfg(test)]
            query_worker_arguments: Vec::new(),
            #[cfg(test)]
            query_worker_sleep_ms: None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServiceResult {
    pub fit_key: String,
    pub authority_instance: String,
    pub completion_sha256: String,
    pub completion_path: PathBuf,
    pub model_sha256: Option<String>,
    pub model_path: Option<PathBuf>,
    pub receipt: CompletionReceipt,
    pub reused: bool,
}
impl ServiceResult {
    /// Verify immutable bytes without opening the database in a client process.
    pub fn model_bytes(&self) -> Result<Vec<u8>> {
        ensure!(
            self.receipt.outcome == ExecutionOutcome::Success,
            "original fit did not succeed"
        );
        read_content(
            self.model_path
                .as_deref()
                .context("original model unavailable")?,
            self.model_sha256
                .as_deref()
                .context("original model identity absent")?,
        )
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServiceQueryResult {
    pub authority_instance: String,
    pub record: QueryRecord,
    pub request_path: PathBuf,
    pub completion_path: PathBuf,
    pub model_path: PathBuf,
    pub predictions_path: PathBuf,
    pub metrics_path: PathBuf,
    pub receipt_path: PathBuf,
    pub receipt: query::QueryReceipt,
    pub reused: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ServiceStats {
    pub authority_opens: usize,
    pub requests: usize,
    pub worker_spawns: usize,
    pub committed: usize,
    pub reused: usize,
    pub active: usize,
}
