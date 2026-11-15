//! Carbon serverless cell execution adapters (`ADR-0007` Phase C3).
//!
//! Provides `CarbonSeededSpace` (Wave-0 neural / landmark-conditioned starter ordering),
//! `CarbonInProcessExecutor` (`SoloSingleLambda` in-memory FIT evaluation),
//! `CarbonRemoteEvalExecutor` (`HybridFanOut` shard delegation),
//! `CarbonStoreArchive` (stateless pause/resume store snapshotting with FIT-path rebinding),
//! separated outer-holdout evaluation (`evaluate_outer_holdout_only`), and champion-only
//! inference (`predict_with_champion_bytes`).

use super::{
    executor::{RefitIoError, RefitWorkerError},
    *,
};
use crate::run::learned::{AdmittedRanker, LearnedAdmission};
use jope_prep::conditioning::FitLandmarkBundle;
use jope_runtime::backend::DeviceRequest;

const CSNP_MAGIC: &[u8; 8] = b"CSNP\x01\x00\x00\x00";
const MAX_CSNP_FILES: usize = 256;
const MAX_CSNP_FILE_BYTES: usize = 32 * 1024 * 1024;
const MAX_CSNP_TOTAL_BYTES: usize = 64 * 1024 * 1024;
const DEFAULT_LAMBDA_R882_LAYER_DIR: &str = "/opt/jope-models/r882";

/// `CarbonNeuralGuidancePolicy` is an enum selecting how an authenticated `r882` neural fusion
/// bundle guides search waves for a product tier (`Wave0SeedOnly` for Free; `MultiWaveLearned`
/// for Pro and Enterprise).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CarbonNeuralGuidancePolicy {
    /// Condition `r882` on client `FIT` landmarks to rank Wave-0 starter seeds, then run heuristic
    /// evolutionary waves (`Free` tier).
    Wave0SeedOnly,
    /// Condition `r882` on client `FIT` landmarks to rank Wave-0 starter seeds AND invoke
    /// `LearnedAdmission` surrogate cost re-ranking across every evolutionary wave (`Pro` / `Enterprise`).
    MultiWaveLearned,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CarbonRefitEnvelope {
    attempt: Attempt,
    model_json: String,
    model_digest: String,
    predictions_digest: String,
}

/// `CarbonSeededSpace` is a struct wrapping `NativeSpace` to prepend Wave-0 starter candidates
/// ordered by neural conditioning (when an `AdmittedRanker` is installed) or deterministic
/// FIT-landmark digest ordering.
pub struct CarbonSeededSpace {
    seeded_starters: Vec<String>,
    seen_digests: std::sync::Mutex<BTreeSet<String>>,
}

impl CarbonSeededSpace {
    /// Construct a `CarbonSeededSpace` from explicit starter expressions, canonicalizing and
    /// deduplicating ahead of `NativeSpace::seeds()`.
    pub fn from_starters(starters: Vec<String>) -> Result<Self> {
        let base = NativeSpace.seeds()?;
        let mut seen = BTreeSet::new();
        let mut seen_digests = BTreeSet::new();
        let mut ordered = Vec::new();
        for item in starters.into_iter().chain(base) {
            if let Ok(canon) = NativeSpace.canonical(&item) {
                if seen.insert(canon.clone()) {
                    seen_digests.insert(digest(canon.as_bytes()));
                    ordered.push(canon);
                }
            }
        }
        ensure!(
            !ordered.is_empty(),
            "CarbonSeededSpace produced zero valid starters"
        );
        Ok(Self {
            seeded_starters: ordered,
            seen_digests: std::sync::Mutex::new(seen_digests),
        })
    }

    fn sync_checkpoint_memo(&self, memo: &BTreeSet<String>) {
        if let Ok(mut guard) = self.seen_digests.lock() {
            for entry in memo {
                guard.insert(entry.clone());
            }
        }
    }

    /// Build Wave-0 starters conditioned on the client's `FIT`-only `LandmarkBundle`.
    /// When `ranker` is present, runs `LearnedAdmission::bind_client_landmarks` +
    /// `rank_wave0_seeds` (`neural_seed_applied = true`); otherwise deterministically
    /// permutes the starter slate from the landmark vector digest (`neural_seed_applied = false`).
    pub(crate) fn conditioned_from_landmarks(
        lm: &FitLandmarkBundle,
        seed: u64,
        ranker: Option<&AdmittedRanker>,
        plan: &Plan,
    ) -> Result<(Self, Option<LearnedAdmission>, bool)> {
        let mut candidate_pool = NativeSpace.seeds()?;
        let mut seen: BTreeSet<String> = candidate_pool.iter().cloned().collect();
        let mut rng = ChaCha12Rng::seed_from_u64(seed ^ 0xCA4B_095E_ED00_0001);
        for _ in 0..16 {
            let extra = feat_math::evolve::random_canonical_pipeline(rng.next_u64(), 5);
            if let Ok(canon) = NativeSpace.canonical(&extra) {
                if seen.insert(canon.clone()) {
                    candidate_pool.push(canon);
                }
            }
        }

        if let Some(admitted) = ranker {
            let mut learned = admitted.instantiate(DeviceRequest::Cpu, 1)?;
            learned.bind_client_landmarks(lm, plan)?;
            let ranked = learned.rank_wave0_seeds(&candidate_pool)?;
            let space = Self::from_starters(ranked)?;
            return Ok((space, Some(learned), true));
        }

        let mut lm_hasher = Sha256::new();
        lm_hasher.update(seed.to_le_bytes());
        for &val in lm.vector256() {
            lm_hasher.update(val.to_bits().to_le_bytes());
        }
        let lm_digest: [u8; 32] = lm_hasher.finalize().into();
        let mut keyed: Vec<([u8; 32], String)> = candidate_pool
            .into_iter()
            .map(|dsl| {
                let mut h = Sha256::new();
                h.update(lm_digest);
                h.update(dsl.as_bytes());
                (h.finalize().into(), dsl)
            })
            .collect();
        keyed.sort_by_key(|a| a.0);
        let ordered = keyed.into_iter().map(|(_, dsl)| dsl).collect();
        Ok((Self::from_starters(ordered)?, None, false))
    }
}

impl SearchSpace for CarbonSeededSpace {
    fn canonical(&self, dsl: &str) -> Result<String> {
        NativeSpace.canonical(dsl)
    }

    fn seeds(&self) -> Result<Vec<String>> {
        Ok(self.seeded_starters.clone())
    }

    fn propose(
        &self,
        parents: &[String],
        count: usize,
        rng: &mut ChaCha12Rng,
    ) -> Result<Vec<String>> {
        let mut guard = self
            .seen_digests
            .lock()
            .map_err(|_| anyhow::anyhow!("CarbonSeededSpace mutex poisoned"))?;
        for parent in parents {
            guard.insert(digest(parent.as_bytes()));
        }
        let mut output = Vec::with_capacity(count);
        for _ in 0..64 {
            for raw in NativeSpace.propose(parents, count, rng)? {
                let canon = NativeSpace.canonical(&raw)?;
                if guard.insert(digest(canon.as_bytes())) {
                    output.push(canon);
                    if output.len() == count {
                        return Ok(output);
                    }
                }
            }
        }
        anyhow::bail!("unable to build complete fresh Carbon proposal slate of {count} candidates")
    }
}

struct CarbonLearnedExecutorAdapter<'a, E> {
    inner: &'a E,
    learned: bool,
    calibration_mode: Option<&'a str>,
}

impl<E: Executor> Executor for CarbonLearnedExecutorAdapter<'_, E> {
    fn evaluate(&self, request: &Request) -> Response {
        let needs_learned = self.learned
            && request.attempt.slot < 65_536
            && request.fold_spec.get("refit_output").is_none()
            && request.fold_spec.get("refit_max_bytes").is_none();
        if needs_learned || self.calibration_mode.is_some() {
            let mut request = request.clone();
            if needs_learned {
                request.fold_spec["version"] = serde_json::json!(2);
            }
            if let Some(mode) = self.calibration_mode {
                request.fold_spec["calibration_mode"] = serde_json::json!(mode);
            }
            self.inner.evaluate(&request)
        } else {
            self.inner.evaluate(request)
        }
    }

    fn fitted_model(&self, request: &Request) -> Result<Vec<u8>> {
        if let Some(mode) = self.calibration_mode {
            let mut request = request.clone();
            request.fold_spec["calibration_mode"] = serde_json::json!(mode);
            self.inner.fitted_model(&request)
        } else {
            self.inner.fitted_model(request)
        }
    }

    fn revoke(&self, attempt: &crate::worker::Attempt) {
        self.inner.revoke(attempt)
    }
}

struct ScrubbedFitBytes(Vec<u8>);

impl AsRef<[u8]> for ScrubbedFitBytes {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl Drop for ScrubbedFitBytes {
    fn drop(&mut self) {
        self.0.fill(0);
        std::hint::black_box(&mut self.0);
    }
}

/// `CarbonInProcessExecutor` is a struct evaluating candidates directly in-memory against an
/// immutable `FIT` `Matrix` (whose backing buffer scrubs itself on drop) for `SoloSingleLambda`
/// and cached `CarbonEvalWorker` shard execution without child processes.
pub struct CarbonInProcessExecutor {
    fit_matrix: Arc<Matrix<ScrubbedFitBytes>>,
    scratch_root: PathBuf,
}

impl CarbonInProcessExecutor {
    /// Open `fit_bytes`, verify `expected_fit_sha256`, and initialize a private refit scratch dir.
    pub fn open_fit(
        fit_bytes: Vec<u8>,
        expected_fit_sha256: &str,
        scratch_root: PathBuf,
    ) -> Result<Self> {
        fs::create_dir_all(&scratch_root)?;
        let scratch_root = fs::canonicalize(&scratch_root)?;
        let matrix = Matrix::open(ScrubbedFitBytes(fit_bytes))?;
        ensure!(
            matrix.sha256_hex() == expected_fit_sha256,
            "CarbonInProcessExecutor FIT matrix SHA-256 mismatch"
        );
        Ok(Self {
            fit_matrix: Arc::new(matrix),
            scratch_root,
        })
    }

    /// Validated lowercase SHA-256 hex digest of the opened `FIT` matrix.
    pub fn fit_sha256_hex(&self) -> &str {
        self.fit_matrix.sha256_hex()
    }

    /// Run private refit and return both the checked `Response` and serialized `NativeModel` bytes
    /// in a single fit pass without re-evaluating all CV folds.
    pub fn fitted_model_with_response(&self, request: &Request) -> Result<(Response, Vec<u8>)> {
        run_private_refit_with_response(&self.scratch_root, request, |req| self.evaluate(req))
    }
}

impl Executor for CarbonInProcessExecutor {
    fn evaluate(&self, request: &Request) -> Response {
        let t0 = Instant::now();
        let outcome = if request.fold_spec.get("version") == Some(&serde_json::json!(2)) {
            crate::run::surrogate::native(request, &self.fit_matrix)
        } else {
            evaluate_native(request, &self.fit_matrix)
        };
        let wall_ms = t0.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
        match outcome {
            Ok(scores) => Response {
                attempt: request.attempt.clone(),
                scores: scores.values,
                fold_scores: scores.fold_scores,
                predictions_digest: scores.predictions_digest,
                wall_ms,
                failure: None,
            },
            Err(failure) => Response::failed(request.attempt.clone(), failure, wall_ms),
        }
    }

    fn fitted_model(&self, request: &Request) -> Result<Vec<u8>> {
        run_private_refit(&self.scratch_root, request, |req| self.evaluate(req))
    }
}

/// `CarbonShardEvalOutcome` is a struct carrying one remote `eval` shard result (`Response` plus
/// optional promoted champion bytes when `fit_champion` was requested).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CarbonShardEvalOutcome {
    pub response: Response,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fitted_model_bytes: Option<Vec<u8>>,
}

/// `CarbonRemoteEvalExecutor` is a struct routing candidate evaluations to stateless `carbon eval`
/// workers for `HybridFanOut` execution while preserving deterministic wave commits.
pub struct CarbonRemoteEvalExecutor<F>
where
    F: Fn(&Request, bool) -> Result<CarbonShardEvalOutcome> + Send + Sync,
{
    dispatch_fn: F,
}

impl<F> CarbonRemoteEvalExecutor<F>
where
    F: Fn(&Request, bool) -> Result<CarbonShardEvalOutcome> + Send + Sync,
{
    pub fn new(dispatch_fn: F) -> Self {
        Self { dispatch_fn }
    }
}

impl<F> Executor for CarbonRemoteEvalExecutor<F>
where
    F: Fn(&Request, bool) -> Result<CarbonShardEvalOutcome> + Send + Sync,
{
    fn evaluate(&self, request: &Request) -> Response {
        match (self.dispatch_fn)(request, false) {
            Ok(outcome) => outcome.response,
            Err(_) => Response::failed(request.attempt.clone(), WorkerFailure::Unavailable, 0),
        }
    }

    fn fitted_model(&self, request: &Request) -> Result<Vec<u8>> {
        let outcome = (self.dispatch_fn)(request, true)?;
        let checked = checked_response(request, outcome.response);
        if checked.failure.is_some() {
            return Err(RefitWorkerError(checked).into());
        }
        let model_bytes = outcome
            .fitted_model_bytes
            .context("remote eval shard omitted promoted model bytes")?;
        let _: super::native::NativeModel = serde_json::from_slice(&model_bytes).map_err(|_| {
            RefitWorkerError(Response::failed(
                request.attempt.clone(),
                WorkerFailure::Protocol,
                checked.wall_ms,
            ))
        })?;
        Ok(model_bytes)
    }
}

/// Stateless `carbon eval` worker entrypoint: evaluates one `Request` (and optional promoted refit)
/// against verified `fit_bytes`.
pub fn evaluate_carbon_eval_shard(
    fit_bytes: &[u8],
    expected_fit_sha256: &str,
    scratch_root: &Path,
    request: &Request,
    fit_champion: bool,
) -> Result<CarbonShardEvalOutcome> {
    let exec = CarbonInProcessExecutor::open_fit(
        fit_bytes.to_vec(),
        expected_fit_sha256,
        scratch_root.to_path_buf(),
    )?;
    evaluate_carbon_eval_shard_with_executor(&exec, request, fit_champion)
}

/// Evaluate one `Request` (and optional single-pass promoted refit) against a pre-validated
/// [`CarbonInProcessExecutor`], avoiding per-candidate `fit.jmat` copies, SHA-256 re-hashing,
/// `Matrix::open` scans, and duplicate 5-fold CV refit passes.
pub fn evaluate_carbon_eval_shard_with_executor(
    exec: &CarbonInProcessExecutor,
    request: &Request,
    fit_champion: bool,
) -> Result<CarbonShardEvalOutcome> {
    if fit_champion {
        let t0 = Instant::now();
        match exec.fitted_model_with_response(request) {
            Ok((refit_resp, model_bytes)) => Ok(CarbonShardEvalOutcome {
                response: refit_resp,
                fitted_model_bytes: Some(model_bytes),
            }),
            Err(err) => {
                let wall_ms = t0.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
                let resp = err
                    .downcast_ref::<RefitWorkerError>()
                    .map(|w| w.0.clone())
                    .unwrap_or_else(|| {
                        Response::failed(request.attempt.clone(), WorkerFailure::NonFinite, wall_ms)
                    });
                Ok(CarbonShardEvalOutcome {
                    response: resp,
                    fitted_model_bytes: None,
                })
            }
        }
    } else {
        Ok(CarbonShardEvalOutcome {
            response: exec.evaluate(request),
            fitted_model_bytes: None,
        })
    }
}

fn run_private_refit(
    scratch_root: &Path,
    request: &Request,
    eval_fn: impl FnOnce(&Request) -> Response,
) -> Result<Vec<u8>> {
    let (_response, model_bytes) = run_private_refit_with_response(scratch_root, request, eval_fn)?;
    Ok(model_bytes)
}

fn run_private_refit_with_response(
    scratch_root: &Path,
    request: &Request,
    eval_fn: impl FnOnce(&Request) -> Response,
) -> Result<(Response, Vec<u8>)> {
    let directory = tempfile::Builder::new()
        .prefix("carbon-fit-")
        .tempdir_in(scratch_root)
        .map_err(RefitIoError)?;
    let output = directory.path().join("model.json");
    let mut refit_req = request.clone();
    let fields = refit_req
        .fold_spec
        .as_object_mut()
        .context("fit request needs an object fold specification")?;
    fields.insert("refit_output".into(), json!(output));
    let maximum = REFIT_HANDOFF_BYTES.min(refit_req.limits.rss_bytes);
    fields.insert("refit_max_bytes".into(), json!(maximum));
    let response = checked_response(&refit_req, eval_fn(&refit_req));
    if response.failure.is_some() {
        return Err(RefitWorkerError(response).into());
    }
    let invalid = || {
        RefitWorkerError(Response::failed(
            refit_req.attempt.clone(),
            WorkerFailure::Protocol,
            response.wall_ms,
        ))
    };
    let metadata = fs::symlink_metadata(&output).map_err(RefitIoError)?;
    if !metadata.is_file() || metadata.len() > maximum {
        return Err(invalid().into());
    }
    let bytes = fs::read(&output).map_err(RefitIoError)?;
    if bytes.len() as u64 > maximum {
        return Err(invalid().into());
    }
    let envelope: CarbonRefitEnvelope = serde_json::from_slice(&bytes).map_err(|_| invalid())?;
    if envelope.attempt != refit_req.attempt
        || envelope.predictions_digest != response.predictions_digest
        || digest(envelope.model_json.as_bytes()) != envelope.model_digest
    {
        return Err(invalid().into());
    }
    Ok((response, envelope.model_json.into_bytes()))
}

/// `CarbonWaveTelemetry` is a struct emitted after each committed wave with strictly scalar metrics.
#[derive(Clone, Debug)]
pub struct CarbonWaveTelemetry {
    pub wave_index: u32,
    pub candidates_evaluated: u64,
    pub best_cv_loss: Option<f64>,
    pub best_cv_metric: Option<f64>,
    pub paused: bool,
    pub completed_reason: Option<CompletionReason>,
}

/// `CarbonWaveRunOutcome` is a struct returned when `run_carbon_waves` pauses or completes.
#[derive(Clone, Debug)]
pub struct CarbonWaveRunOutcome {
    pub wave_index: u32,
    pub candidates_evaluated: u64,
    pub best_cv_loss: Option<f64>,
    pub best_cv_metric: Option<f64>,
    pub neural_seed_applied: bool,
    pub neural_guidance_mode: &'static str,
    pub bundle_digest: Option<String>,
    pub paused: bool,
    pub completed_reason: Option<CompletionReason>,
    pub champion_model_bytes: Option<Vec<u8>>,
    pub store_archive_bytes: Vec<u8>,
}

/// Attempt to load the authenticated `r882` DOPE + JopeDime neural fusion ranker from:
/// 1. `CARBON_MODEL_R882_DIR` (explicit directory override; fails closed on any tamper),
/// 2. `/opt/jope-models/r882` (read-only AWS Lambda Layer mount; fails closed if present & tampered),
/// 3. Self-embedded signed appliance package (`Ok(None)` in unbundled CI/dev builds).
pub fn try_open_embedded_ranker() -> Result<Option<AdmittedRanker>> {
    if let Some(explicit_dir) = std::env::var_os("CARBON_MODEL_R882_DIR") {
        let dir_path = Path::new(&explicit_dir);
        let verified =
            crate::ranker::cuda::release::load_r882_release(dir_path).with_context(|| {
                format!(
                    "CARBON_MODEL_R882_DIR ({}) failed cryptographic r882 release verification",
                    dir_path.display()
                )
            })?;
        return Ok(Some(AdmittedRanker::r882(verified)?));
    }

    let layer_path = Path::new(DEFAULT_LAMBDA_R882_LAYER_DIR);
    if layer_path.join("release-catalog.json").exists() {
        let verified =
            crate::ranker::cuda::release::load_r882_release(layer_path).with_context(|| {
                format!(
                    "Lambda layer {} failed cryptographic r882 release verification",
                    layer_path.display()
                )
            })?;
        return Ok(Some(AdmittedRanker::r882(verified)?));
    }

    #[cfg(unix)]
    {
        crate::proposal::from_self()
    }
    #[cfg(not(unix))]
    {
        Ok(None)
    }
}

/// Cryptographically verify an extracted `r882` Lambda Layer bundle directory (cosign P-256
/// detached signature, `release-catalog.json`, all 56 asset digests, and `AdmittedRanker::r882`
/// admission) before packaging, returning the canonical `sha256:...` bundle digest.
pub fn verify_r882_release_dir(dir: &Path) -> Result<String> {
    let verified = crate::ranker::cuda::release::load_r882_release(dir).with_context(|| {
        format!(
            "r882 bundle directory {} failed cryptographic release verification",
            dir.display()
        )
    })?;
    let _admitted = AdmittedRanker::r882(verified)?;
    Ok(crate::ranker::cuda::release::R882.to_owned())
}

/// `CarbonWaveSessionOptions` is a struct bundling the per-session inputs for `run_carbon_waves`.
pub struct CarbonWaveSessionOptions<'a> {
    pub store_dir: &'a Path,
    pub plan: Plan,
    pub lm_bundle: &'a FitLandmarkBundle,
    pub ranker: Option<&'a AdmittedRanker>,
    pub guidance_policy: CarbonNeuralGuidancePolicy,
    pub live_workers: usize,
    pub max_waves_this_session: u32,
}

/// Run or resume deterministic wave search on `store_dir` using `plan` (`FIT`-only!), invoking
/// `on_wave` after each wave and honoring cooperative `should_pause` ("see it now" or Lambda
/// deadline headroom), defaulting post-hoc output calibration to `"auto"`.
pub fn run_carbon_waves(
    session: CarbonWaveSessionOptions<'_>,
    executor: &impl Executor,
    should_pause: &dyn Fn(u32) -> bool,
    on_wave: &mut dyn FnMut(&CarbonWaveTelemetry),
) -> Result<CarbonWaveRunOutcome> {
    run_carbon_waves_with_calibration(session, Some("auto"), executor, should_pause, on_wave)
}

/// Run or resume deterministic wave search with an explicit `calibration_mode` (`"auto"`,
/// `"none"`, `"shift"`, or `"affine"`).
pub fn run_carbon_waves_with_calibration(
    session: CarbonWaveSessionOptions<'_>,
    calibration_mode: Option<&str>,
    executor: &impl Executor,
    should_pause: &dyn Fn(u32) -> bool,
    on_wave: &mut dyn FnMut(&CarbonWaveTelemetry),
) -> Result<CarbonWaveRunOutcome> {
    let CarbonWaveSessionOptions {
        store_dir,
        plan,
        lm_bundle,
        ranker,
        guidance_policy,
        live_workers,
        max_waves_this_session,
    } = session;
    ensure!(
        max_waves_this_session > 0,
        "max_waves_this_session must be >= 1"
    );
    let store = Arc::new(Store::open(store_dir)?);
    let mut discovery = Discovery::open(Arc::clone(&store), plan)?;
    let active_plan = discovery.checkpoint.plan.clone();
    let (seeded_space, learned_admission, neural_seed_applied) =
        CarbonSeededSpace::conditioned_from_landmarks(
            lm_bundle,
            active_plan.seed,
            ranker,
            &active_plan,
        )?;
    seeded_space.sync_checkpoint_memo(&discovery.checkpoint.memo);
    let bundle_digest = ranker.map(|r| r.digest());
    let fit_rows_admissible = fs::read(&active_plan.matrix_path)
        .ok()
        .and_then(|b| Matrix::open(b).ok())
        .is_some_and(|m| (4..=20_000).contains(&m.rows()));
    let use_multi_wave_learned = learned_admission.is_some()
        && matches!(
            guidance_policy,
            CarbonNeuralGuidancePolicy::MultiWaveLearned
        )
        && fit_rows_admissible
        && active_plan.times.is_none()
        && active_plan.groups.is_none()
        && crate::objective::regression::RegressionObjective::parse(&active_plan.objective).is_ok();
    let adapted_executor = CarbonLearnedExecutorAdapter {
        inner: executor,
        learned: use_multi_wave_learned,
        calibration_mode,
    };
    let neural_guidance_mode = if !neural_seed_applied {
        "landmark_digest_fallback"
    } else if use_multi_wave_learned {
        "multi_wave_learned"
    } else {
        "wave0_seed_only"
    };

    let mut waves_executed = 0u32;
    let mut paused = false;
    while discovery.completed().is_none() && waves_executed < max_waves_this_session {
        seeded_space.sync_checkpoint_memo(&discovery.checkpoint.memo);
        if use_multi_wave_learned {
            if let Some(learned) = &learned_admission {
                discovery.advance(&seeded_space, learned, &adapted_executor, live_workers, 1)?;
            }
        } else {
            discovery.advance(
                &seeded_space,
                &HeuristicAdmission,
                &adapted_executor,
                live_workers,
                1,
            )?;
        }
        waves_executed = waves_executed.saturating_add(1);
        let current_wave = u32::try_from(discovery.checkpoint.next_round).unwrap_or(u32::MAX);
        let interrupted_early = discovery.completed().is_none() && should_pause(current_wave);
        let reached_session_cap =
            discovery.completed().is_none() && waves_executed >= max_waves_this_session;
        if interrupted_early || reached_session_cap {
            discovery.pause()?;
        }
        paused = interrupted_early;
        let telemetry = CarbonWaveTelemetry {
            wave_index: current_wave,
            candidates_evaluated: discovery.snapshot().budget_used.evaluations.0,
            best_cv_loss: discovery.champion().map(|c| c.loss),
            best_cv_metric: discovery.champion().map(|c| c.value),
            paused,
            completed_reason: discovery.completed(),
        };
        on_wave(&telemetry);
        if paused {
            break;
        }
    }

    let wave_index = u32::try_from(discovery.checkpoint.next_round).unwrap_or(u32::MAX);
    let candidates_evaluated = discovery.snapshot().budget_used.evaluations.0;
    let best_cv_loss = discovery.champion().map(|c| c.loss);
    let best_cv_metric = discovery.champion().map(|c| c.value);
    let completed_reason = discovery.completed();
    let champion_model_bytes = discovery
        .champion()
        .map(|champ| store.read_blob(&champ.model))
        .transpose()?;

    drop(discovery);
    drop(store);
    let store_archive_bytes = CarbonStoreArchive::pack_store_dir(store_dir)?;

    Ok(CarbonWaveRunOutcome {
        wave_index,
        candidates_evaluated,
        best_cv_loss,
        best_cv_metric,
        neural_seed_applied,
        neural_guidance_mode,
        bundle_digest,
        paused,
        completed_reason,
        champion_model_bytes,
        store_archive_bytes,
    })
}

/// Evaluate the promoted champion strictly on `holdout.jmat` AFTER search pauses or finishes,
/// returning `(outer_holdout_loss, outer_holdout_metric)` without ever exposing `holdout.jmat`
/// to candidate generation, fitting, or selection.
pub fn evaluate_outer_holdout_only(
    champion_model_bytes: &[u8],
    holdout_jmat_bytes: &[u8],
    expected_holdout_sha256: &str,
    target: &str,
    features: &[String],
    objective_spec: &str,
) -> Result<(f64, f64)> {
    ensure!(
        digest(holdout_jmat_bytes) == expected_holdout_sha256,
        "HOLDOUT matrix SHA-256 mismatch"
    );
    let fitted: super::native::NativeModel =
        serde_json::from_slice(champion_model_bytes).context("invalid champion model payload")?;
    let holdout_matrix = Matrix::open(holdout_jmat_bytes)?;
    ensure!(holdout_matrix.rows() > 0, "HOLDOUT matrix is empty");
    let find_col = |name: &str| {
        holdout_matrix
            .columns()
            .iter()
            .position(|c| c.name == name)
            .with_context(|| format!("unknown HOLDOUT column '{name}'"))
    };
    let target_col = find_col(target)?;
    let feat_cols: Vec<usize> = features
        .iter()
        .map(|f| find_col(f))
        .collect::<Result<_>>()?;
    let mut hx = Vec::with_capacity(holdout_matrix.rows());
    let mut hy = Vec::with_capacity(holdout_matrix.rows());
    for row in 0..holdout_matrix.rows() {
        let Some(JmatCell::Numeric(label)) = holdout_matrix.cell(row, target_col) else {
            bail!("invalid numeric target in HOLDOUT row {row}");
        };
        ensure!(label.is_finite(), "non-finite target in HOLDOUT row {row}");
        hy.push(label);
        let row_vals = feat_cols
            .iter()
            .map(|&col| match holdout_matrix.cell(row, col) {
                Some(JmatCell::Numeric(v)) if v.is_finite() => Ok(v),
                Some(JmatCell::Missing) => Ok(f64::NAN),
                _ => bail!("invalid numeric feature in HOLDOUT row {row}"),
            })
            .collect::<Result<Vec<f64>>>()?;
        hx.push(row_vals);
    }
    let preds = fitted.predict(&hx)?;
    let objective = RegressionObjective::parse(objective_spec)?;
    let holdout_rows: Vec<usize> = (0..holdout_matrix.rows()).collect();
    let aux_cols =
        super::native::aux_columns_for_rows(&holdout_matrix, &objective.columns(), &holdout_rows)?;
    let score = objective.evaluate_with(&hy, &preds, &aux_cols)?;
    ensure!(
        score.loss.is_finite() && score.value.is_finite(),
        "non-finite HOLDOUT score"
    );
    Ok((score.loss, score.value))
}

/// Compute E2EE permutation feature importances (`delta_loss = permuted_loss - baseline_loss`)
/// strictly on `holdout.jmat` using the actual promoted GP `NativeModel` champion.
#[allow(clippy::too_many_arguments)]
pub fn explain_champion_on_holdout(
    champion_model_bytes: &[u8],
    holdout_jmat_bytes: &[u8],
    expected_holdout_sha256: &str,
    target: &str,
    features: &[String],
    objective_spec: &str,
    top_k: Option<usize>,
    seed: u64,
) -> Result<(f64, Vec<jope_api::carbon::CarbonFeatureImportanceWire>)> {
    ensure!(
        digest(holdout_jmat_bytes) == expected_holdout_sha256,
        "HOLDOUT matrix SHA-256 mismatch"
    );
    let fitted: super::native::NativeModel =
        serde_json::from_slice(champion_model_bytes).context("invalid champion model payload")?;
    ensure!(
        fitted.features == features,
        "explain feature list does not match fitted champion features"
    );
    let holdout_matrix = Matrix::open(holdout_jmat_bytes)?;
    ensure!(holdout_matrix.rows() > 0, "HOLDOUT matrix is empty");
    let all_rows: Vec<usize> = (0..holdout_matrix.rows()).collect();
    let (hx, hy) = super::native::numeric_rows(&holdout_matrix, target, features, &all_rows)?;
    let objective = RegressionObjective::parse(objective_spec)?;
    let aux_cols =
        super::native::aux_columns_for_rows(&holdout_matrix, &objective.columns(), &all_rows)?;
    let base_preds = fitted.predict(&hx)?;
    let baseline_loss = objective.evaluate_with(&hy, &base_preds, &aux_cols)?.loss;
    ensure!(
        baseline_loss.is_finite(),
        "non-finite baseline HOLDOUT loss"
    );

    let repeats = if hx.len() >= 2 { 3usize } else { 1usize };
    let mut raw_deltas = Vec::with_capacity(features.len());
    for (col_idx, name) in features.iter().enumerate() {
        let mut delta_sum = 0.0_f64;
        for rep in 0..repeats {
            let mut permuted = hx.clone();
            if permuted.len() >= 2 {
                let mut rng = ChaCha12Rng::seed_from_u64(
                    seed ^ ((col_idx as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15))
                        ^ ((rep as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9)),
                );
                for i in (1..permuted.len()).rev() {
                    let j = (rng.next_u64() as usize) % (i + 1);
                    let tmp = permuted[i][col_idx];
                    permuted[i][col_idx] = permuted[j][col_idx];
                    permuted[j][col_idx] = tmp;
                }
            }
            let perm_loss = fitted
                .predict(&permuted)
                .and_then(|p| objective.evaluate_with(&hy, &p, &aux_cols))
                .map(|s| s.loss)
                .unwrap_or_else(|_| baseline_loss + baseline_loss.abs().max(1.0));
            let delta = if perm_loss.is_finite() {
                (perm_loss - baseline_loss).max(0.0)
            } else {
                baseline_loss.abs().max(1.0)
            };
            delta_sum += delta;
        }
        raw_deltas.push((name.clone(), delta_sum / repeats as f64));
    }

    let total_delta: f64 = raw_deltas.iter().map(|(_, d)| *d).sum();
    let mut items: Vec<jope_api::carbon::CarbonFeatureImportanceWire> = raw_deltas
        .into_iter()
        .map(
            |(feature, delta_loss)| jope_api::carbon::CarbonFeatureImportanceWire {
                feature,
                importance: if total_delta > 0.0 {
                    delta_loss / total_delta
                } else {
                    0.0
                },
                delta_loss,
            },
        )
        .collect();
    items.sort_by(|a, b| {
        b.delta_loss
            .total_cmp(&a.delta_loss)
            .then_with(|| a.feature.cmp(&b.feature))
    });
    if let Some(k) = top_k {
        items.truncate(k.max(1));
    }
    Ok((baseline_loss, items))
}

/// Export a decrypted GP `NativeModel` champion to standalone, zero-dependency source code in
/// `Python`, `Rust`, `RustRefittable`, `Js`, `C`, or `Cpp`.
pub fn export_champion_standalone_source(
    champion_model_bytes: &[u8],
    target: &str,
    lang: jope_api::carbon::CarbonExportLangWire,
) -> Result<String> {
    use jope_api::carbon::CarbonExportLangWire;
    use jope_runtime::math_pipeline::{
        InputContract, MathPipeline, NumericColumn, PipelineState, TargetScale, INPUT_PART,
        PIPELINE_PART,
    };

    let fitted: super::native::NativeModel =
        serde_json::from_slice(champion_model_bytes).context("invalid champion model payload")?;
    ensure!(
        fitted.features.len() == fitted.min.len()
            && fitted.features.len() == fitted.max.len()
            && fitted.features.len() == fitted.fill.len(),
        "corrupt NativeModel scaler metadata"
    );
    let genome = feat_math::GenomeIr::parse(&fitted.dsl)?;
    match lang {
        CarbonExportLangWire::C => {
            let emitted =
                feat_math::c17::emit(&genome, &feat_math::transpile::EmitOptions::default())?;
            return Ok(serde_json::to_string_pretty(&serde_json::json!({
                "schema": "carbon.c17_bundle.v1",
                "artifact_kind": "gp_feature_basis",
                "note": "C17 bundle emits the discovered GP feature-basis kernel; apply feature_min/feature_max/feature_fill scaling before calling and use --lang python, rust, rust-refittable, or js for full end-to-end fitted head + calibration export.",
                "build": feat_math::c17::BUILD,
                "target": target,
                "features": fitted.features,
                "feature_min": fitted.min,
                "feature_max": fitted.max,
                "feature_fill": fitted.fill,
                "target_min": fitted.target_min,
                "target_scale": fitted.target_scale,
                "calibration": fitted.calibration,
                "files": emitted.files,
            }))?);
        }
        CarbonExportLangWire::Cpp => {
            let mut cpp = feat_math::transpile::emit_cpp17(
                &genome,
                &feat_math::transpile::EmitOptions::default(),
            )?;
            cpp.insert_str(
                0,
                "// NOTE: C++17 export emits the discovered GP feature-basis kernel (normalized [0,1] inputs).\n// Use `--lang python`, `--lang rust`, `--lang rust-refittable`, or `--lang js` for full fitted head + output calibration.\n",
            );
            return Ok(cpp);
        }
        _ => {}
    }

    let (eff_min, eff_scale, lower_clamp) = match &fitted.calibration {
        Some(cal) => (
            cal.intercept + cal.slope * fitted.target_min,
            cal.slope * fitted.target_scale,
            cal.lower,
        ),
        None => (fitted.target_min, fitted.target_scale, None),
    };
    let input = InputContract::new(
        fitted
            .features
            .iter()
            .enumerate()
            .map(|(i, name)| NumericColumn {
                name: name.clone(),
                min: fitted.min[i],
                max: fitted.max[i],
                constant: fitted.min[i] == fitted.max[i],
            })
            .collect(),
        (0..fitted.features.len()).collect(),
        TargetScale {
            name: target.to_owned(),
            min: eff_min,
            scale: eff_scale,
        },
    )?;
    let parts = BTreeMap::from([
        (
            PIPELINE_PART.to_owned(),
            serde_json::to_vec(&PipelineState {
                dsl: fitted.dsl,
                model: fitted.model,
            })?,
        ),
        (INPUT_PART.to_owned(), serde_json::to_vec(&input)?),
    ]);
    let pipeline = MathPipeline::decode(&parts, PIPELINE_PART)?;
    match lang {
        CarbonExportLangWire::Python => {
            let mut src =
                String::from_utf8(jope_runtime::fitted_py::emit_math_pipeline(&pipeline)?)?;
            if let Some(lb) = lower_clamp {
                src.push_str(&format!(
                    "\n_carbon_unclamped_predict_rows = predict_rows\ndef predict_rows(rows):\n    return [max(v, {lb:?}) for v in _carbon_unclamped_predict_rows(rows)]\n"
                ));
            }
            Ok(src)
        }
        CarbonExportLangWire::Rust => {
            let mut src = jope_runtime::fitted_rs::emit_pipeline(&pipeline)?;
            if let Some(lb) = lower_clamp {
                const TARGET_SNIPPET: &str = "if !value.is_finite() { return Err(\"prediction failed: non-finite target prediction\".into()); }\n        Ok(value)";
                ensure!(
                    src.contains(TARGET_SNIPPET),
                    "fitted Rust emitter missing expected target return snippet for lower_clamp"
                );
                let clamped = format!(
                    "if !value.is_finite() {{ return Err(\"prediction failed: non-finite target prediction\".into()); }}\n        let value = value.max({lb:?});\n        Ok(value)"
                );
                src = src.replacen(TARGET_SNIPPET, &clamped, 1);
            }
            Ok(src)
        }
        CarbonExportLangWire::RustRefittable => pipeline.export_refittable_rust(),
        CarbonExportLangWire::Js => {
            let mut src = String::from_utf8(pipeline.export_javascript()?)?;
            if let Some(lb) = lower_clamp {
                src.push_str(&format!(
                    "\nconst _carbonUnclampedPredictRows = predictRows;\npredictRows = function(rows) {{ return _carbonUnclampedPredictRows(rows).map(v => Math.max(v, {lb:?})); }};\nif (typeof module !== \"undefined\" && module.exports) module.exports = {{ MODEL_COLUMNS, predictRows }};\n"
                ));
            }
            Ok(src)
        }
        CarbonExportLangWire::C | CarbonExportLangWire::Cpp => unreachable!(),
    }
}

/// `CarbonParsedChampion` is an opaque parsed wrapper around `NativeModel` allowing warm
/// `CarbonEngineWorker` instances to skip repeated JSON deserializations on `/predict` after
/// authenticating `sealed_champion.enc`.
pub struct CarbonParsedChampion {
    inner: super::native::NativeModel,
}

impl CarbonParsedChampion {
    /// Parse a decrypted champion JSON payload once into an in-memory `CarbonParsedChampion`.
    pub fn from_json_bytes(champion_model_bytes: &[u8]) -> Result<Self> {
        let inner: super::native::NativeModel = serde_json::from_slice(champion_model_bytes)
            .context("invalid champion model payload")?;
        Ok(Self { inner })
    }

    /// Run inference on `rows` aligned by `feature_names`, returning scalar `f64` predictions.
    pub fn predict(&self, feature_names: &[String], rows: &[Vec<Option<f64>>]) -> Result<Vec<f64>> {
        ensure!(
            feature_names.len() == self.inner.features.len(),
            "predict feature count {} does not match fitted model feature count {}",
            feature_names.len(),
            self.inner.features.len()
        );
        let col_map: Vec<usize> = self
            .inner
            .features
            .iter()
            .map(|expected| {
                feature_names
                    .iter()
                    .position(|provided| provided == expected)
                    .with_context(|| format!("missing required feature column '{expected}'"))
            })
            .collect::<Result<_>>()?;

        let mut aligned: feat_math::Rows = Vec::with_capacity(rows.len());
        for row in rows {
            ensure!(
                row.len() == feature_names.len(),
                "ragged predict row width {} (expected {})",
                row.len(),
                feature_names.len()
            );
            let mapped: Vec<f64> = col_map
                .iter()
                .map(|&idx| match row[idx] {
                    Some(v) if v.is_finite() => Ok(v),
                    Some(_) => bail!("non-finite numeric input in predict row"),
                    None => Ok(f64::NAN),
                })
                .collect::<Result<_>>()?;
            aligned.push(mapped);
        }
        if aligned.is_empty() {
            return Ok(vec![]);
        }
        self.inner.predict(&aligned)
    }
}

/// Run inference on feature `rows` using a decrypted champion model buffer, returning ONLY scalar
/// `f64` predictions.
pub fn predict_with_champion_bytes(
    champion_model_bytes: &[u8],
    feature_names: &[String],
    rows: &[Vec<Option<f64>>],
) -> Result<Vec<f64>> {
    let parsed = CarbonParsedChampion::from_json_bytes(champion_model_bytes)?;
    parsed.predict(feature_names, rows)
}

/// `CarbonStoreArchive` is a struct providing deterministic binary packing and cross-container
/// unpacking of a `jope-service` `Store` directory for `AES-256-GCM` checkpoint sealing.
pub struct CarbonStoreArchive;

impl CarbonStoreArchive {
    /// Pack all regular files under `store_dir` into a deterministic `CSNP` v1 byte vector.
    pub fn pack_store_dir(store_dir: &Path) -> Result<Vec<u8>> {
        let mut entries: Vec<(String, Vec<u8>)> = Vec::new();
        collect_dir_files(store_dir, store_dir, &mut entries)?;
        entries.sort_by(|a, b| a.0.cmp(&b.0));

        let mut out = Vec::new();
        out.extend_from_slice(CSNP_MAGIC);
        out.extend_from_slice(&(entries.len() as u32).to_le_bytes());
        for (rel, data) in entries {
            let rel_bytes = rel.as_bytes();
            out.extend_from_slice(&(rel_bytes.len() as u32).to_le_bytes());
            out.extend_from_slice(rel_bytes);
            out.extend_from_slice(&(data.len() as u64).to_le_bytes());
            out.extend_from_slice(&data);
        }
        Ok(out)
    }

    /// Unpack a `CSNP` v1 byte vector into `target_store_dir` and rebind the stored `Plan`'s
    /// `matrix_path` to `new_fit_path` after verifying `sha256(new_fit_path) == plan.matrix_sha256`.
    pub fn unpack_store_dir_and_rebind_fit(
        archive_bytes: &[u8],
        target_store_dir: &Path,
        run_id: &RunId,
        new_fit_path: &Path,
    ) -> Result<Plan> {
        ensure!(
            archive_bytes.len() >= 12
                && archive_bytes.len() <= MAX_CSNP_TOTAL_BYTES
                && &archive_bytes[..8] == CSNP_MAGIC,
            "invalid or oversized CSNP store archive header"
        );
        fs::create_dir_all(target_store_dir)?;
        let count = u32::from_le_bytes(archive_bytes[8..12].try_into()?) as usize;
        ensure!(
            count <= MAX_CSNP_FILES,
            "CSNP archive file count {count} exceeds cap {MAX_CSNP_FILES}"
        );
        let mut cursor = 12usize;
        let mut unpacked_bytes = 0usize;
        let mut seen_paths = BTreeSet::new();
        for _ in 0..count {
            ensure!(cursor + 4 <= archive_bytes.len(), "truncated CSNP path len");
            let path_len =
                u32::from_le_bytes(archive_bytes[cursor..cursor + 4].try_into()?) as usize;
            cursor += 4;
            ensure!(
                path_len > 0 && path_len <= 256 && cursor + path_len + 8 <= archive_bytes.len(),
                "truncated or invalid CSNP entry path"
            );
            let rel = std::str::from_utf8(&archive_bytes[cursor..cursor + path_len])?;
            cursor += path_len;
            let rel_path = Path::new(rel);
            let safe_components = !rel.contains('\\')
                && !rel.contains('\0')
                && !rel_path.is_absolute()
                && rel_path
                    .components()
                    .all(|c| matches!(c, std::path::Component::Normal(_)));
            ensure!(
                safe_components && seen_paths.insert(rel.to_owned()),
                "unsafe or duplicate relative path in CSNP archive"
            );
            let data_len_u64 = u64::from_le_bytes(archive_bytes[cursor..cursor + 8].try_into()?);
            let data_len = usize::try_from(data_len_u64)?;
            cursor += 8;
            ensure!(
                data_len <= MAX_CSNP_FILE_BYTES && cursor + data_len <= archive_bytes.len(),
                "CSNP file payload exceeds bound or truncates archive"
            );
            unpacked_bytes = unpacked_bytes
                .checked_add(data_len)
                .filter(|&tot| tot <= MAX_CSNP_TOTAL_BYTES)
                .context("CSNP unpacked byte total exceeds bound")?;
            let dest = target_store_dir.join(rel_path);
            if let Some(parent) = dest.parent() {
                fs::create_dir_all(parent)?;
            }
            if let Ok(existing_meta) = fs::symlink_metadata(&dest) {
                ensure!(
                    !existing_meta.file_type().is_symlink(),
                    "refusing symlink target in CSNP unpack"
                );
            }
            fs::write(&dest, &archive_bytes[cursor..cursor + data_len])?;
            cursor += data_len;
        }
        ensure!(
            cursor == archive_bytes.len(),
            "trailing bytes in CSNP archive"
        );

        let fit_bytes = fs::read(new_fit_path)?;
        let fit_sha = digest(&fit_bytes);

        let store = Store::open(target_store_dir)?;
        let key = RecordKey::Run(run_id.clone());
        let mut record = store.get(&key)?.context("archived run record missing")?;
        let mut saved: Checkpoint = serde_json::from_value(record.data["checkpoint"].clone())?;
        ensure!(
            saved.plan.matrix_sha256 == fit_sha,
            "rebound FIT matrix SHA-256 mismatch on resume"
        );
        saved.plan.matrix_path = new_fit_path.to_path_buf();
        let updated_plan = saved.plan.clone();
        record.data["checkpoint"] = serde_json::to_value(&saved)?;
        store.transaction(|tx| tx.put(&key, &record))?;
        Ok(updated_plan)
    }
}

fn collect_dir_files(root: &Path, current: &Path, out: &mut Vec<(String, Vec<u8>)>) -> Result<()> {
    for entry in fs::read_dir(current)? {
        let entry = entry?;
        let meta = entry.metadata()?;
        let path = entry.path();
        if meta.is_dir() {
            collect_dir_files(root, &path, out)?;
        } else if meta.is_file() {
            let rel = path
                .strip_prefix(root)?
                .to_str()
                .context("non-UTF8 store file path")?
                .replace('\\', "/");
            out.push((rel, fs::read(&path)?));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use jope_prep::{
        conditioning::FitLandmarkBundle,
        jmat_lite::{split_and_encode_jmat_v1, PrepTable},
        split::SplitPlan,
    };

    #[test]
    fn carbon_neural_seeded_waves_pause_cross_host_resume_holdout_and_predict() {
        let total_rows = 40usize;
        let features = 3usize;
        let mut x_f32 = Vec::with_capacity(total_rows * features);
        let mut y_f32 = Vec::with_capacity(total_rows);
        let mut rows_opt = Vec::with_capacity(total_rows);
        for r in 0..total_rows {
            let x0 = (r as f64) * 0.25 - 3.0;
            let x1 = ((r * 7 % 13) as f64) * 0.4;
            let x2 = ((r % 5) as f64) - 2.0;
            let y = 2.5 * x0 - 1.2 * x1 + 0.5 * x2 + 1.0;
            x_f32.extend_from_slice(&[x0 as f32, x1 as f32, x2 as f32]);
            y_f32.push(y as f32);
            rows_opt.push(vec![
                Some(x0.to_string()),
                Some(x1.to_string()),
                Some(x2.to_string()),
                Some(y.to_string()),
            ]);
        }
        let split = SplitPlan::deterministic(total_rows, 19, 0.25).expect("split");
        let lm_bundle =
            FitLandmarkBundle::extract_fit_only(&x_f32, total_rows, features, &y_f32, &split)
                .expect("landmarks");
        let table = PrepTable {
            headers: vec!["x0".into(), "x1".into(), "x2".into(), "y".into()],
            rows: rows_opt,
        };
        let split_jmat = split_and_encode_jmat_v1(&table, &split).expect("jmat");
        let fit_sha_hex = digest(&split_jmat.fit_jmat);
        let holdout_sha_hex = digest(&split_jmat.holdout_jmat);

        let host1 = tempfile::tempdir().expect("host1");
        let fit_path_1 = host1.path().join("fit.jmat");
        fs::write(&fit_path_1, &split_jmat.fit_jmat).expect("write fit 1");
        let store_1 = host1.path().join("store");
        let scratch_1 = host1.path().join("scratch");

        let run_id: RunId = "run_01JCCARB0N0000000000000002".parse().expect("run_id");
        let dataset_id: DatasetId = "ds_222222222222222222222222".parse().expect("dataset_id");
        let plan_1 = Plan {
            run_id: run_id.clone(),
            dataset_id,
            problem_digest: "3".repeat(64),
            bundle_digest: "4".repeat(64),
            binary_version: "1.0.0-test".into(),
            matrix_path: fit_path_1,
            matrix_sha256: fit_sha_hex.clone(),
            target: "y".into(),
            features: vec!["x0".into(), "x1".into(), "x2".into()],
            objective: "rmse".into(),
            seed: 19,
            planned_workers: 2,
            effort: Effort::Fast,
            evaluations: 64,
            wall_seconds: 60,
            evaluation_wall_ms: 5_000,
            evaluation_rss_bytes: 256 * 1024 * 1024,
            groups: None,
            times: None,
            gap_rows: 0,
        };

        let exec_1 =
            CarbonInProcessExecutor::open_fit(split_jmat.fit_jmat.clone(), &fit_sha_hex, scratch_1)
                .expect("exec_1");
        let mut seen_waves = 0u32;
        let out_1 = run_carbon_waves(
            CarbonWaveSessionOptions {
                store_dir: &store_1,
                plan: plan_1,
                lm_bundle: &lm_bundle,
                ranker: None,
                guidance_policy: CarbonNeuralGuidancePolicy::Wave0SeedOnly,
                live_workers: 2,
                max_waves_this_session: 1,
            },
            &exec_1,
            &|_| true,
            &mut |tel| {
                seen_waves = tel.wave_index;
            },
        )
        .expect("session 1");
        assert!(!out_1.neural_seed_applied);
        assert_eq!(out_1.neural_guidance_mode, "landmark_digest_fallback");
        assert!(out_1.paused);
        assert_eq!(seen_waves, 1);
        assert!(out_1.champion_model_bytes.is_some());

        // Unpack into a completely separate directory (`host2`) to prove cross-container Lambda resume!
        let host2 = tempfile::tempdir().expect("host2");
        let fit_path_2 = host2.path().join("rebound-fit.jmat");
        fs::write(&fit_path_2, &split_jmat.fit_jmat).expect("write fit 2");
        let store_2 = host2.path().join("store");
        let scratch_2 = host2.path().join("scratch");
        let rebound_plan = CarbonStoreArchive::unpack_store_dir_and_rebind_fit(
            &out_1.store_archive_bytes,
            &store_2,
            &run_id,
            &fit_path_2,
        )
        .expect("rebind on host2");

        let exec_2 = CarbonRemoteEvalExecutor::new(|req, fit_champ| {
            evaluate_carbon_eval_shard(
                &split_jmat.fit_jmat,
                &fit_sha_hex,
                &scratch_2,
                req,
                fit_champ,
            )
        });
        let out_2 = run_carbon_waves(
            CarbonWaveSessionOptions {
                store_dir: &store_2,
                plan: rebound_plan,
                lm_bundle: &lm_bundle,
                ranker: None,
                guidance_policy: CarbonNeuralGuidancePolicy::MultiWaveLearned,
                live_workers: 2,
                max_waves_this_session: 2,
            },
            &exec_2,
            &|_| false,
            &mut |_| {},
        )
        .expect("session 2 resume via remote eval shard");
        assert!(out_2.wave_index >= 2);
        assert!(out_2.candidates_evaluated > out_1.candidates_evaluated);
        let champ_bytes = out_2.champion_model_bytes.expect("champion present");

        let (holdout_loss, holdout_metric) = evaluate_outer_holdout_only(
            &champ_bytes,
            &split_jmat.holdout_jmat,
            &holdout_sha_hex,
            "y",
            &["x0".into(), "x1".into(), "x2".into()],
            "rmse",
        )
        .expect("outer holdout eval");
        assert!(holdout_loss.is_finite() && holdout_metric.is_finite());

        let preds = predict_with_champion_bytes(
            &champ_bytes,
            &["x2".into(), "x0".into(), "x1".into()],
            &[
                vec![Some(0.0), Some(1.0), Some(2.0)],
                vec![None, Some(-1.0), Some(0.5)],
            ],
        )
        .expect("predict with reordered columns + missing cell");
        assert_eq!(preds.len(), 2);
        assert!(preds.iter().all(|p| p.is_finite()));
    }

    #[test]
    fn carbon_csnp_bounds_and_tampered_r882_dir_refusal() {
        let tmp = tempfile::tempdir().expect("tmp");
        let fit_path = tmp.path().join("fit.jmat");
        fs::write(&fit_path, b"dummy").expect("write dummy");

        // 1. CSNP archive path-traversal refusal (`RT-09`).
        let mut bad_archive = Vec::new();
        bad_archive.extend_from_slice(CSNP_MAGIC);
        bad_archive.extend_from_slice(&1u32.to_le_bytes());
        let evil_rel = b"../escape.txt";
        bad_archive.extend_from_slice(&(evil_rel.len() as u32).to_le_bytes());
        bad_archive.extend_from_slice(evil_rel);
        bad_archive.extend_from_slice(&4u64.to_le_bytes());
        bad_archive.extend_from_slice(b"boom");
        let unpack_err = CarbonStoreArchive::unpack_store_dir_and_rebind_fit(
            &bad_archive,
            &tmp.path().join("bad"),
            &"run_01JCCARB0N0000000000000012".parse().expect("id"),
            &fit_path,
        )
        .unwrap_err();
        assert!(unpack_err.to_string().contains("unsafe"));

        // 2. CSNP archive file-count cap refusal (`> 256` entries).
        let mut too_many = Vec::new();
        too_many.extend_from_slice(CSNP_MAGIC);
        too_many.extend_from_slice(&300u32.to_le_bytes());
        let count_err = CarbonStoreArchive::unpack_store_dir_and_rebind_fit(
            &too_many,
            &tmp.path().join("many"),
            &"run_01JCCARB0N0000000000000012".parse().expect("id"),
            &fit_path,
        )
        .unwrap_err();
        assert!(count_err.to_string().contains("exceeds cap"));

        // 3. Tampered / unsigned r882 directory fails closed (`RT-13`).
        let fake_r882 = tmp.path().join("fake_r882");
        fs::create_dir_all(&fake_r882).expect("mkdir");
        fs::write(fake_r882.join("release-catalog.json"), b"{}").expect("write fake catalog");
        let Err(load_err) = crate::ranker::cuda::release::load_r882_release(&fake_r882) else {
            panic!("unsigned r882 must fail closed");
        };
        assert!(!load_err.to_string().is_empty());

        // 4. Optional live r882 verification when `JOPE_TEST_R882_DIR` is supplied.
        if let Some(real_dir) = std::env::var_os("JOPE_TEST_R882_DIR") {
            let verified = crate::ranker::cuda::release::load_r882_release(Path::new(&real_dir))
                .expect("verified r882 release");
            let ranker = AdmittedRanker::r882(verified).expect("admitted r882");
            let ranker_hex = ranker.digest().trim_start_matches("sha256:").to_owned();

            let total_rows = 40usize;
            let features = 2usize;
            let mut x_f32 = Vec::with_capacity(total_rows * features);
            let mut y_f32 = Vec::with_capacity(total_rows);
            let mut rows_opt = Vec::with_capacity(total_rows);
            for r in 0..total_rows {
                let x0 = (r as f64) * 0.2 - 2.0;
                let x1 = ((r * 5 % 11) as f64) * 0.3;
                let y = 1.8 * x0 - 0.9 * x1 + 0.25;
                x_f32.extend_from_slice(&[x0 as f32, x1 as f32]);
                y_f32.push(y as f32);
                rows_opt.push(vec![
                    Some(x0.to_string()),
                    Some(x1.to_string()),
                    Some(y.to_string()),
                ]);
            }
            let split = SplitPlan::deterministic(total_rows, 7, 0.25).expect("split");
            let lm_bundle =
                FitLandmarkBundle::extract_fit_only(&x_f32, total_rows, features, &y_f32, &split)
                    .expect("landmarks");
            let table = PrepTable {
                headers: vec!["x0".into(), "x1".into(), "y".into()],
                rows: rows_opt,
            };
            let split_jmat = split_and_encode_jmat_v1(&table, &split).expect("jmat");
            let fit_sha_hex = digest(&split_jmat.fit_jmat);

            let free_dir = tempfile::tempdir().expect("free_dir");
            let free_fit = free_dir.path().join("fit.jmat");
            fs::write(&free_fit, &split_jmat.fit_jmat).expect("write free fit");
            let free_exec = CarbonInProcessExecutor::open_fit(
                split_jmat.fit_jmat.clone(),
                &fit_sha_hex,
                free_dir.path().join("scratch"),
            )
            .expect("free_exec");
            let free_plan = Plan {
                run_id: "run_01JCCARB0N0000000000000011".parse().expect("run_id"),
                dataset_id: "ds_333333333333333333333333".parse().expect("ds"),
                problem_digest: "5".repeat(64),
                bundle_digest: ranker_hex.clone(),
                binary_version: "1.0.0-test".into(),
                matrix_path: free_fit,
                matrix_sha256: fit_sha_hex.clone(),
                target: "y".into(),
                features: vec!["x0".into(), "x1".into()],
                objective: "rmse".into(),
                seed: 7,
                planned_workers: 2,
                effort: Effort::Fast,
                evaluations: 48,
                wall_seconds: 60,
                evaluation_wall_ms: 5_000,
                evaluation_rss_bytes: 256 * 1024 * 1024,
                groups: None,
                times: None,
                gap_rows: 0,
            };
            let free_out = run_carbon_waves(
                CarbonWaveSessionOptions {
                    store_dir: &free_dir.path().join("store"),
                    plan: free_plan,
                    lm_bundle: &lm_bundle,
                    ranker: Some(&ranker),
                    guidance_policy: CarbonNeuralGuidancePolicy::Wave0SeedOnly,
                    live_workers: 2,
                    max_waves_this_session: 3,
                },
                &free_exec,
                &|_| false,
                &mut |_| {},
            )
            .expect("free run");
            assert!(free_out.neural_seed_applied);
            assert_eq!(free_out.neural_guidance_mode, "wave0_seed_only");
            let forwards_after_free = ranker.execution_counter().load(Ordering::Relaxed);
            assert!(forwards_after_free > 0);

            let pro_dir = tempfile::tempdir().expect("pro_dir");
            let pro_fit = pro_dir.path().join("fit.jmat");
            fs::write(&pro_fit, &split_jmat.fit_jmat).expect("write pro fit");
            let pro_exec = CarbonInProcessExecutor::open_fit(
                split_jmat.fit_jmat.clone(),
                &fit_sha_hex,
                pro_dir.path().join("scratch"),
            )
            .expect("pro_exec");
            let pro_plan = Plan {
                run_id: "run_01JCCARB0N0000000000000012".parse().expect("run_id"),
                dataset_id: "ds_333333333333333333333333".parse().expect("ds"),
                problem_digest: "5".repeat(64),
                bundle_digest: ranker_hex,
                binary_version: "1.0.0-test".into(),
                matrix_path: pro_fit,
                matrix_sha256: fit_sha_hex,
                target: "y".into(),
                features: vec!["x0".into(), "x1".into()],
                objective: "rmse".into(),
                seed: 7,
                planned_workers: 2,
                effort: Effort::Fast,
                evaluations: 48,
                wall_seconds: 60,
                evaluation_wall_ms: 5_000,
                evaluation_rss_bytes: 256 * 1024 * 1024,
                groups: None,
                times: None,
                gap_rows: 0,
            };
            let pro_out = run_carbon_waves(
                CarbonWaveSessionOptions {
                    store_dir: &pro_dir.path().join("store"),
                    plan: pro_plan,
                    lm_bundle: &lm_bundle,
                    ranker: Some(&ranker),
                    guidance_policy: CarbonNeuralGuidancePolicy::MultiWaveLearned,
                    live_workers: 2,
                    max_waves_this_session: 3,
                },
                &pro_exec,
                &|_| false,
                &mut |_| {},
            )
            .expect("pro run");
            assert!(pro_out.neural_seed_applied);
            assert_eq!(pro_out.neural_guidance_mode, "multi_wave_learned");
            let pro_forwards =
                ranker.execution_counter().load(Ordering::Relaxed) - forwards_after_free;
            assert!(pro_forwards > forwards_after_free);
        }
    }

    #[test]
    fn prep_objective_spec_matches_runtime_fitness_across_all_53_catalog_entries_and_examples() {
        let prep_cat = jope_prep::objective::ObjectiveSpec::catalog();
        let rt_cat = jope_runtime::fitness::Fitness::catalog();
        assert_eq!(prep_cat.len(), 53);
        assert_eq!(rt_cat.len(), 53);
        for (p, r) in prep_cat.iter().zip(rt_cat.iter()) {
            assert_eq!(p.name, r.name);
            assert_eq!(
                p.direction == jope_prep::objective::ObjectiveDirection::Minimize,
                r.fitness.direction() == jope_runtime::fitness::Direction::Minimize,
                "direction mismatch on {}",
                p.name
            );
            assert_eq!(
                p.requires_order,
                r.fitness.requires_order(),
                "chronological flag mismatch on {}",
                p.name
            );
            let p_spec = jope_prep::objective::ObjectiveSpec::parse(p.name).unwrap();
            let r_spec = jope_runtime::fitness::Fitness::parse(r.name).unwrap();
            assert_eq!(
                p_spec.identity_sha256(),
                r_spec.identity().unwrap(),
                "SHA-256 identity mismatch on {}",
                p.name
            );
        }

        let y = [1.5, 2.2, 4.0, 3.1, 5.5, 6.2, 4.8, 7.1];
        let pred = [1.3, 2.5, 3.6, 3.3, 5.0, 6.6, 4.5, 6.9];
        for spec_str in [
            "rmse",
            "mae",
            "pinball:quantile=0.9",
            "asymmetric_absolute:under_cost=4,over_cost=1",
            "poisson_deviance",
            "spearman",
            "normalized_gini",
            "mase:season=2",
            include_str!("../../../../../examples/objectives/grid_underforecast.json"),
            include_str!("../../../../../examples/objectives/inventory_profit.json"),
        ] {
            let p_spec = jope_prep::objective::ObjectiveSpec::parse(spec_str).unwrap();
            let r_spec = jope_runtime::fitness::Fitness::parse(spec_str).unwrap();
            assert_eq!(p_spec.identity_sha256(), r_spec.identity().unwrap());
            let p_score = p_spec.evaluate(&y, &pred, &BTreeMap::new()).unwrap();
            let r_score = r_spec.evaluate(&y, &pred).unwrap();
            assert!(
                (p_score.loss - r_score.loss).abs() < 1e-11,
                "loss mismatch on {spec_str}: {} vs {}",
                p_score.loss,
                r_score.loss
            );
            assert!(
                (p_score.value - r_score.value).abs() < 1e-11,
                "value mismatch on {spec_str}: {} vs {}",
                p_score.value,
                r_score.value
            );
        }
    }

    #[test]
    fn custom_aux_columns_calibration_chronological_explain_and_export_work_end_to_end() {
        let n = 48usize;
        let headers = vec![
            "x0".to_string(),
            "x1".to_string(),
            "unit_margin".to_string(),
            "holding_cost".to_string(),
            "y".to_string(),
        ];
        let mut rows = Vec::with_capacity(n);
        let mut x_f32 = Vec::with_capacity(n * 2);
        let mut y_f32 = Vec::with_capacity(n);
        for i in 0..n {
            let x0 = ((i as f64) * 0.41).sin() * 4.0 + 5.0;
            let x1 = ((i * 7) % 11) as f64 * 0.1;
            let margin = 8.0 + (i % 3) as f64;
            let hold = 1.0;
            let asym_noise = ((i % 5) as f64) * 0.5;
            let y = 10.0 + 3.0 * x0 - 0.1 * x1 + asym_noise;
            x_f32.extend_from_slice(&[x0 as f32, x1 as f32]);
            y_f32.push(y as f32);
            rows.push(vec![
                Some(x0.to_string()),
                Some(x1.to_string()),
                Some(margin.to_string()),
                Some(hold.to_string()),
                Some(y.to_string()),
            ]);
        }
        let table = PrepTable { headers, rows };
        let split = SplitPlan::chronological(n, 42, 0.25).unwrap();
        let encoded = split_and_encode_jmat_v1(&table, &split).unwrap();
        let fit_sha_hex = digest(&encoded.fit_jmat);
        let hold_sha_hex = digest(&encoded.holdout_jmat);

        let dir = tempfile::tempdir().unwrap();
        let fit_path = dir.path().join("fit.jmat");
        fs::write(&fit_path, &encoded.fit_jmat).unwrap();
        let exec = CarbonInProcessExecutor::open_fit(
            encoded.fit_jmat.clone(),
            &fit_sha_hex,
            dir.path().join("scratch"),
        )
        .unwrap();

        let lm_bundle = FitLandmarkBundle::extract_fit_only(&x_f32, n, 2, &y_f32, &split).unwrap();

        let aux_spec = r#"{"metric":"custom","name":"newsvendor","direction":"minimize","expression":{"op":"where","condition":{"op":"sub","left":{"op":"y"},"right":{"op":"pred"}},"then":{"op":"mul","left":{"op":"column","name":"unit_margin"},"right":{"op":"sub","left":{"op":"y"},"right":{"op":"pred"}}},"else":{"op":"mul","left":{"op":"column","name":"holding_cost"},"right":{"op":"sub","left":{"op":"pred"},"right":{"op":"y"}}}},"aggregation":{"kind":"mean"}}"#;
        let fit_len = split.fit_rows().len();
        let make_plan = |obj: &str, suffix: u8| Plan {
            run_id: format!("run_01JCCARB0N000000000000001{suffix}")
                .parse()
                .unwrap(),
            dataset_id: format!("ds_33333333333333333333333{suffix}")
                .parse()
                .unwrap(),
            problem_digest: "6".repeat(64),
            bundle_digest: "7".repeat(64),
            binary_version: "1.0.0-test".into(),
            matrix_path: fit_path.clone(),
            matrix_sha256: fit_sha_hex.clone(),
            target: "y".into(),
            features: vec!["x0".into(), "x1".into()],
            objective: obj.into(),
            seed: 42,
            planned_workers: 2,
            effort: Effort::Fast,
            evaluations: 24,
            wall_seconds: 30,
            evaluation_wall_ms: 5_000,
            evaluation_rss_bytes: 256 * 1024 * 1024,
            groups: None,
            times: Some((0..fit_len).map(|i| i as f64).collect()),
            gap_rows: 0,
        };

        let uncal_out = run_carbon_waves_with_calibration(
            CarbonWaveSessionOptions {
                store_dir: &dir.path().join("store_uncal"),
                plan: make_plan(aux_spec, 1),
                lm_bundle: &lm_bundle,
                ranker: None,
                guidance_policy: CarbonNeuralGuidancePolicy::Wave0SeedOnly,
                live_workers: 2,
                max_waves_this_session: 2,
            },
            Some("none"),
            &exec,
            &|_| false,
            &mut |_| {},
        )
        .unwrap();
        let cal_out = run_carbon_waves_with_calibration(
            CarbonWaveSessionOptions {
                store_dir: &dir.path().join("store_cal"),
                plan: make_plan(aux_spec, 2),
                lm_bundle: &lm_bundle,
                ranker: None,
                guidance_policy: CarbonNeuralGuidancePolicy::Wave0SeedOnly,
                live_workers: 2,
                max_waves_this_session: 2,
            },
            Some("auto"),
            &exec,
            &|_| false,
            &mut |_| {},
        )
        .unwrap();
        let champ_cal = cal_out.champion_model_bytes.as_ref().unwrap();
        let champ_uncal = uncal_out.champion_model_bytes.as_ref().unwrap();
        assert!(cal_out.best_cv_loss.unwrap() <= uncal_out.best_cv_loss.unwrap() + 1e-6);

        let (hold_loss_cal, _) = evaluate_outer_holdout_only(
            champ_cal,
            &encoded.holdout_jmat,
            &hold_sha_hex,
            "y",
            &["x0".into(), "x1".into()],
            aux_spec,
        )
        .unwrap();
        let (hold_loss_uncal, _) = evaluate_outer_holdout_only(
            champ_uncal,
            &encoded.holdout_jmat,
            &hold_sha_hex,
            "y",
            &["x0".into(), "x1".into()],
            aux_spec,
        )
        .unwrap();
        assert!(hold_loss_cal.is_finite() && hold_loss_uncal.is_finite());

        let (base_loss, importances) = explain_champion_on_holdout(
            champ_cal,
            &encoded.holdout_jmat,
            &hold_sha_hex,
            "y",
            &["x0".into(), "x1".into()],
            aux_spec,
            None,
            42,
        )
        .unwrap();
        assert!((base_loss - hold_loss_cal).abs() < 1e-12);
        assert_eq!(importances.len(), 2);
        assert_eq!(importances[0].feature, "x0");
        assert!(importances[0].importance > importances[1].importance);

        let py_src = export_champion_standalone_source(
            champ_cal,
            "y",
            jope_api::carbon::CarbonExportLangWire::Python,
        )
        .unwrap();
        let rs_src = export_champion_standalone_source(
            champ_cal,
            "y",
            jope_api::carbon::CarbonExportLangWire::Rust,
        )
        .unwrap();
        let js_src = export_champion_standalone_source(
            champ_cal,
            "y",
            jope_api::carbon::CarbonExportLangWire::Js,
        )
        .unwrap();
        assert!(py_src.contains("def predict_rows("));
        assert!(rs_src.contains("MODEL_COLUMNS"));
        assert!(js_src.contains("predictRows"));

        let py_file = dir.path().join("exported_model.py");
        fs::write(&py_file, &py_src).unwrap();
        let py_out = std::process::Command::new("python3")
            .arg("-c")
            .arg(format!(
                "import importlib.util, json; spec=importlib.util.spec_from_file_location('m', {:?}); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); print(json.dumps(m.predict_rows([[5.5, 0.3], [3.2, 0.7]])))",
                py_file
            ))
            .output()
            .unwrap();
        assert!(
            py_out.status.success(),
            "python3 stderr: {}",
            String::from_utf8_lossy(&py_out.stderr)
        );
        let py_preds: Vec<f64> = serde_json::from_slice(&py_out.stdout).unwrap();
        let native_preds = predict_with_champion_bytes(
            champ_cal,
            &["x0".into(), "x1".into()],
            &[vec![Some(5.5), Some(0.3)], vec![Some(3.2), Some(0.7)]],
        )
        .unwrap();
        for (a, b) in py_preds.iter().zip(native_preds.iter()) {
            assert!(
                (a - b).abs() < 1e-9,
                "exported python pred {a} != native {b}"
            );
        }
    }
}
