use super::*;

pub trait SearchSpace {
    fn canonical(&self, dsl: &str) -> Result<String>;
    fn seeds(&self) -> Result<Vec<String>>;
    fn propose(
        &self,
        parents: &[String],
        count: usize,
        rng: &mut ChaCha12Rng,
    ) -> Result<Vec<String>>;
}

/// Pure numerical/protocol boundary. Implementations must return the echoed
/// attempt exactly and must not publish store effects.
pub trait Executor: Sync {
    fn evaluate(&self, request: &Request) -> Response;
    fn fitted_model(&self, request: &Request) -> Result<Vec<u8>>;
    fn revoke(&self, _attempt: &Attempt) {}
}

#[derive(Debug)]
pub(super) struct RefitWorkerError(pub(super) Response);
impl std::fmt::Display for RefitWorkerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("bounded promoted fit failed")
    }
}
impl std::error::Error for RefitWorkerError {}

#[derive(Debug)]
pub(super) struct RefitIoError(std::io::Error);
impl std::fmt::Display for RefitIoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "private fit handoff: {}", self.0)
    }
}
impl std::error::Error for RefitIoError {}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RefitEnvelope {
    attempt: Attempt,
    model_json: String,
    model_digest: String,
    predictions_digest: String,
}

/// Linux parent-death containment is tied to the *creating thread*. Keep pool
/// callers alive for the entire run, including between waves and promotions;
/// short-lived scoped callers would kill idle reusable workers on thread exit.
pub struct ProcessExecutor {
    sender: Option<std::sync::mpsc::Sender<ProcessJob>>,
    threads: Vec<std::thread::JoinHandle<()>>,
    pool: Arc<WorkerPool>,
    scratch_root: PathBuf,
}
type ProcessJob = (Request, std::sync::mpsc::SyncSender<Response>);

impl ProcessExecutor {
    pub fn new(executable: &Path, workers: usize, scratch_root: &Path) -> Result<Self> {
        ensure!(workers > 0 && workers <= 1024, "invalid live workers");
        let mut builder = fs::DirBuilder::new();
        builder.recursive(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            builder.mode(0o700);
        }
        builder.create(scratch_root)?;
        ensure!(
            !fs::symlink_metadata(scratch_root)?.file_type().is_symlink(),
            "private fit scratch root must not be a symlink"
        );
        let scratch_root = fs::canonicalize(scratch_root)?;
        let pool = Arc::new(WorkerPool::new(worker::PoolConfig::new(
            executable.to_path_buf(),
            workers,
        ))?);
        let (sender, receiver) = std::sync::mpsc::channel::<ProcessJob>();
        let receiver = Arc::new(Mutex::new(receiver));
        let mut threads = Vec::new();
        for _ in 0..workers {
            let pool = Arc::clone(&pool);
            let receiver = Arc::clone(&receiver);
            threads.push(std::thread::spawn(move || loop {
                let job = receiver.lock().expect("dispatch mutex poisoned").recv();
                let Ok((request, result)) = job else {
                    break;
                };
                let response = pool.evaluate(&request);
                let _ = result.send(response);
            }));
        }
        Ok(Self {
            sender: Some(sender),
            threads,
            pool,
            scratch_root,
        })
    }
}

impl Drop for ProcessExecutor {
    fn drop(&mut self) {
        self.sender.take();
        for thread in self.threads.drain(..) {
            let _ = thread.join();
        }
    }
}

impl Executor for ProcessExecutor {
    fn evaluate(&self, request: &Request) -> Response {
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        if self
            .sender
            .as_ref()
            .is_none_or(|dispatch| dispatch.send((request.clone(), sender)).is_err())
        {
            return Response::failed(request.attempt.clone(), WorkerFailure::Unavailable, 0);
        }
        receiver.recv().unwrap_or_else(|_| {
            Response::failed(request.attempt.clone(), WorkerFailure::Unavailable, 0)
        })
    }
    fn fitted_model(&self, request: &Request) -> Result<Vec<u8>> {
        let directory = tempfile::Builder::new()
            .prefix("promoted-fit-")
            .tempdir_in(&self.scratch_root)
            .map_err(RefitIoError)?;
        let output = directory.path().join("model.json");
        let mut request = request.clone();
        let fields = request
            .fold_spec
            .as_object_mut()
            .context("fit request needs an object fold specification")?;
        fields.insert("refit_output".into(), json!(output));
        let maximum = REFIT_HANDOFF_BYTES.min(request.limits.rss_bytes);
        fields.insert("refit_max_bytes".into(), json!(maximum));
        // The unchanged pool supplies the same hard wall/RSS limits and fence.
        // Returned training scores are deliberately never used for selection.
        let response = checked_response(&request, self.evaluate(&request));
        if response.failure.is_some() {
            return Err(RefitWorkerError(response).into());
        }
        let invalid = || {
            RefitWorkerError(Response::failed(
                request.attempt.clone(),
                WorkerFailure::Protocol,
                response.wall_ms,
            ))
        };
        let metadata = fs::symlink_metadata(&output).map_err(RefitIoError)?;
        if !metadata.is_file() || metadata.len() > maximum {
            return Err(invalid().into());
        }
        let mut bytes = vec![];
        fs::File::open(&output)
            .map_err(RefitIoError)?
            .take(maximum + 1)
            .read_to_end(&mut bytes)
            .map_err(RefitIoError)?;
        if bytes.len() as u64 > maximum {
            return Err(invalid().into());
        }
        let envelope: RefitEnvelope = serde_json::from_slice(&bytes).map_err(|_| invalid())?;
        if envelope.attempt != request.attempt
            || envelope.predictions_digest != response.predictions_digest
            || digest(envelope.model_json.as_bytes()) != envelope.model_digest
        {
            return Err(invalid().into());
        }
        Ok(envelope.model_json.into_bytes())
    }
    fn revoke(&self, attempt: &Attempt) {
        self.pool.revoke(attempt);
    }
}
