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
}

impl<E: Executor> Executor for CarbonLearnedExecutorAdapter<'_, E> {
    fn evaluate(&self, request: &Request) -> Response {
        if self.learned
            && request.attempt.slot < 65_536
            && request.fold_spec.get("refit_output").is_none()
            && request.fold_spec.get("refit_max_bytes").is_none()
        {
            let mut request = request.clone();
            request.fold_spec["version"] = serde_json::json!(2);
            self.inner.evaluate(&request)
        } else {
            self.inner.evaluate(request)
        }
    }

    fn fitted_model(&self, request: &Request) -> Result<Vec<u8>> {
        self.inner.fitted_model(request)
    }

    fn revoke(&self, attempt: &crate::worker::Attempt) {
        self.inner.revoke(attempt)
    }
}

/// `CarbonInProcessExecutor` is a struct evaluating candidates directly in-memory against an
/// immutable `FIT` `Matrix<Vec<u8>>` for `SoloSingleLambda` execution without child processes.
pub struct CarbonInProcessExecutor {
    fit_matrix: Arc<Matrix<Vec<u8>>>,
    scratch_root: PathBuf,
}

impl CarbonInProcessExecutor {
    /// Open `fit_bytes`, verify `expected_fit_sha256`, and initialize a private refit scratch dir.
    pub fn open_fit(
        fit_bytes: Vec<u8>,
        expected_fit_sha256: &str,
        scratch_root: PathBuf,
    ) -> Result<Self> {
        ensure!(
            digest(&fit_bytes) == expected_fit_sha256,
            "CarbonInProcessExecutor FIT matrix SHA-256 mismatch"
        );
        fs::create_dir_all(&scratch_root)?;
        let scratch_root = fs::canonicalize(&scratch_root)?;
        let matrix = Matrix::open(fit_bytes)?;
        Ok(Self {
            fit_matrix: Arc::new(matrix),
            scratch_root,
        })
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
    if fit_champion {
        let t0 = Instant::now();
        match exec.fitted_model(request) {
            Ok(model_bytes) => {
                let eval_resp = exec.evaluate(request);
                Ok(CarbonShardEvalOutcome {
                    response: eval_resp,
                    fitted_model_bytes: Some(model_bytes),
                })
            }
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
    Ok(envelope.model_json.into_bytes())
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
/// deadline headroom).
pub fn run_carbon_waves(
    session: CarbonWaveSessionOptions<'_>,
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
        && crate::objective::regression::RegressionObjective::parse(&active_plan.objective).is_ok();
    let adapted_executor = CarbonLearnedExecutorAdapter {
        inner: executor,
        learned: use_multi_wave_learned,
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
    let score = objective.evaluate(&hy, &preds)?;
    ensure!(
        score.loss.is_finite() && score.value.is_finite(),
        "non-finite HOLDOUT score"
    );
    Ok((score.loss, score.value))
}

/// Run inference on feature `rows` using a decrypted champion model buffer, returning ONLY scalar
/// `f64` predictions.
pub fn predict_with_champion_bytes(
    champion_model_bytes: &[u8],
    feature_names: &[String],
    rows: &[Vec<Option<f64>>],
) -> Result<Vec<f64>> {
    #[derive(Deserialize)]
    struct ChampionHeader {
        features: Vec<String>,
    }
    let header: ChampionHeader =
        serde_json::from_slice(champion_model_bytes).context("invalid champion model header")?;
    let fitted: super::native::NativeModel =
        serde_json::from_slice(champion_model_bytes).context("invalid champion model payload")?;
    ensure!(
        feature_names.len() == header.features.len(),
        "predict feature count {} does not match fitted model feature count {}",
        feature_names.len(),
        header.features.len()
    );
    let col_map: Vec<usize> = header
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
    fitted.predict(&aligned)
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
}
