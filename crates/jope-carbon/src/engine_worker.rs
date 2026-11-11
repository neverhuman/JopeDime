//! Sealed `carbon engine` orchestrator (`3,008 MB` Graviton2 container, `ADR-0007` Phase C5).
//!
//! Authenticates client `JLMB` v1 `FitLandmarkBundle` bytes, plans `SoloSingleLambda` vs
//! `HybridFanOut`, runs wave-by-wave search (`CarbonSeededSpace` + cooperative `stop.json` +
//! Lambda deadline headroom drain), encrypts checkpoints & champion models with `AES-256-GCM`
//! (`SealedCheckpointEnvelope`), computes separated outer-holdout scalar metrics, and serves
//! sealed server-side `predict` requests.

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use base64::Engine as _;
use jope_api::{
    carbon::{
        CarbonCreateRunRequest, CarbonExecutionModeWire, CarbonPredictRequest,
        CarbonPredictResponse, CarbonPublicWaveProgress, CarbonResumeRunRequest,
        CarbonRunStateWire, CarbonTierWire,
    },
    decimal::DecimalU64,
};
use jope_prep::conditioning::FitLandmarkBundle;
use jope_service::discover::r#loop::{
    carbon::{
        evaluate_outer_holdout_only, predict_with_champion_bytes, run_carbon_waves,
        try_open_embedded_ranker, CarbonInProcessExecutor, CarbonRemoteEvalExecutor,
        CarbonStoreArchive, CarbonWaveRunOutcome, CarbonWaveSessionOptions,
    },
    Plan,
};

use crate::{
    cas::{sha256_lower_hex, CarbonCasStore},
    edge_router::{
        map_api_effort, now_epoch_ms, state_key, CarbonEngineInvoker, CarbonEngineWireCommand,
    },
    eval_worker::{CarbonEvalInvoker, CarbonEvalWireRequest},
    planner::{plan_carbon_execution, CarbonExecutionMode, CarbonPlanDecision, CarbonPlanInput},
    role::{CarbonEffortTier, CarbonProductTier},
    seal::{CarbonSealKey, SealedCheckpointEnvelope},
};

const DEADLINE_DRAIN_MARGIN_MS: u64 = 15_000;

struct EngineSessionDir {
    path: PathBuf,
}

impl EngineSessionDir {
    fn create(parent: &std::path::Path, prefix: &str) -> Result<Self, String> {
        let dir = parent.join(format!("{prefix}{}", ulid::Ulid::new()));
        fs::create_dir_all(&dir)
            .map_err(|e| format!("create session dir {}: {e}", dir.display()))?;
        Ok(Self { path: dir })
    }

    fn path(&self) -> &std::path::Path {
        &self.path
    }
}

impl Drop for EngineSessionDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

/// `CarbonEngineWorker` is a struct implementing the sealed `carbon engine` cell.
pub struct CarbonEngineWorker {
    cas: Arc<dyn CarbonCasStore>,
    eval_invoker: Option<Arc<dyn CarbonEvalInvoker>>,
    seal_key: CarbonSealKey,
    work_root: PathBuf,
    force_hybrid_fanout: bool,
}

impl CarbonEngineWorker {
    pub fn new(
        cas: Arc<dyn CarbonCasStore>,
        eval_invoker: Option<Arc<dyn CarbonEvalInvoker>>,
        seal_key: CarbonSealKey,
        work_root: PathBuf,
    ) -> Result<Self, String> {
        fs::create_dir_all(&work_root).map_err(|e| format!("create engine work_root: {e}"))?;
        Ok(Self {
            cas,
            eval_invoker,
            seal_key,
            work_root,
            force_hybrid_fanout: false,
        })
    }

    pub fn with_forced_hybrid_fanout(mut self, force: bool) -> Self {
        self.force_hybrid_fanout = force;
        self
    }

    pub fn handle_invocation_bytes(
        &self,
        payload: &[u8],
        deadline_epoch_ms: Option<u64>,
    ) -> Result<Vec<u8>, String> {
        let cmd: CarbonEngineWireCommand =
            serde_json::from_slice(payload).map_err(|e| format!("decode engine command: {e}"))?;
        self.execute_command(&cmd, deadline_epoch_ms)
    }

    pub fn execute_command(
        &self,
        cmd: &CarbonEngineWireCommand,
        deadline_epoch_ms: Option<u64>,
    ) -> Result<Vec<u8>, String> {
        match cmd {
            CarbonEngineWireCommand::StartRun { request, region } => {
                let progress = self.execute_start_run(request, region, deadline_epoch_ms)?;
                serde_json::to_vec(&progress).map_err(|e| format!("encode start progress: {e}"))
            }
            CarbonEngineWireCommand::ResumeRun {
                run_id,
                resume,
                region,
            } => {
                let progress =
                    self.execute_resume_run(run_id, resume, region, deadline_epoch_ms)?;
                serde_json::to_vec(&progress).map_err(|e| format!("encode resume progress: {e}"))
            }
            CarbonEngineWireCommand::Predict { run_id, predict } => {
                let pred = self.execute_predict(run_id, predict)?;
                serde_json::to_vec(&pred).map_err(|e| format!("encode predict response: {e}"))
            }
        }
    }

    fn execute_start_run(
        &self,
        create: &CarbonCreateRunRequest,
        region: &str,
        deadline_epoch_ms: Option<u64>,
    ) -> Result<CarbonPublicWaveProgress, String> {
        let t0 = Instant::now();
        let run_str = create.run_id.to_string();
        let lm_raw = base64::engine::general_purpose::STANDARD
            .decode(&create.landmark_bundle_b64)
            .map_err(|e| format!("invalid landmark_bundle_b64: {e}"))?;
        let lm_bundle = FitLandmarkBundle::from_wire_bytes(&lm_raw)
            .map_err(|e| format!("invalid FitLandmarkBundle: {e}"))?;

        let fit_bytes = self
            .cas
            .get_blob_verified(&create.fit_jmat_sha256.to_ascii_lowercase())?;
        let holdout_bytes = self
            .cas
            .get_blob_verified(&create.holdout_jmat_sha256.to_ascii_lowercase())?;

        let effort = map_api_effort(create.effort);
        let decision = self.compute_plan_decision(create, effort)?;
        let use_hybrid = (decision.mode == CarbonExecutionMode::HybridFanOut
            || self.force_hybrid_fanout)
            && self.eval_invoker.is_some();

        let session_dir = EngineSessionDir::create(&self.work_root, "engine-run-")?;
        let fit_path = session_dir.path().join("fit.jmat");
        fs::write(&fit_path, &fit_bytes).map_err(|e| format!("stage fit.jmat: {e}"))?;
        let store_dir = session_dir.path().join("store");
        let scratch_dir = session_dir.path().join("scratch");

        let ranker = try_open_embedded_ranker().ok().flatten();
        let bundle_digest = ranker
            .as_ref()
            .and_then(|r| r.digest().strip_prefix("sha256:").map(str::to_owned))
            .unwrap_or_else(|| "0".repeat(64));

        let problem_hash = sha256_lower_hex(
            format!(
                "{}:{}:{}:{}",
                create.fit_jmat_sha256,
                create.target_name,
                create.feature_names.join(","),
                create.objective
            )
            .as_bytes(),
        );
        let total_evals = u64::from(effort.planned_waves())
            .saturating_mul(u64::from(effort.candidates_per_wave()));
        let plan = Plan {
            run_id: create.run_id.clone(),
            dataset_id: create.dataset_id.clone(),
            problem_digest: problem_hash,
            bundle_digest,
            binary_version: env!("CARGO_PKG_VERSION").to_owned(),
            matrix_path: fit_path,
            matrix_sha256: create.fit_jmat_sha256.to_ascii_lowercase(),
            target: create.target_name.clone(),
            features: create.feature_names.clone(),
            objective: create.objective.clone(),
            seed: create.seed.0,
            planned_workers: decision.shards_per_wave.max(2),
            effort: create.effort,
            evaluations: total_evals.max(16),
            wall_seconds: u64::from(effort.lease_ceiling_secs()).max(30),
            evaluation_wall_ms: 15_000,
            evaluation_rss_bytes: (decision.eval_class.memory_mb() as u64) * 1024 * 1024,
            groups: None,
            times: None,
            gap_rows: 0,
        };

        let outcome = self.drive_waves(
            create,
            &run_str,
            region,
            &decision,
            use_hybrid,
            &store_dir,
            &scratch_dir,
            &fit_bytes,
            plan,
            &lm_bundle,
            ranker.as_ref(),
            effort.planned_waves(),
            deadline_epoch_ms,
            t0,
        )?;

        self.finalize_and_persist_progress(
            create,
            &run_str,
            region,
            &decision,
            use_hybrid,
            &holdout_bytes,
            outcome,
            effort.planned_waves(),
            t0.elapsed().as_millis() as u64,
        )
    }

    fn execute_resume_run(
        &self,
        run_id_str: &str,
        resume: &CarbonResumeRunRequest,
        region: &str,
        deadline_epoch_ms: Option<u64>,
    ) -> Result<CarbonPublicWaveProgress, String> {
        let t0 = Instant::now();
        let req_bytes = self
            .cas
            .get_state_object(&state_key(&resume.tenant_id, run_id_str, "request.json"))?
            .ok_or_else(|| format!("missing stored request.json for run '{run_id_str}'"))?;
        let mut create: CarbonCreateRunRequest =
            serde_json::from_slice(&req_bytes).map_err(|e| format!("corrupt request.json: {e}"))?;
        if let Some(new_effort) = resume.effort {
            create.effort = new_effort;
        }
        let sealed_ckp_bytes = self
            .cas
            .get_state_object(&state_key(
                &resume.tenant_id,
                run_id_str,
                "sealed_checkpoint.enc",
            ))?
            .ok_or_else(|| format!("run '{run_id_str}' has no sealed_checkpoint.enc to resume"))?;

        let (_saved_wave, decrypted_archive) = SealedCheckpointEnvelope::open(
            &self.seal_key,
            &resume.tenant_id,
            run_id_str,
            &sealed_ckp_bytes,
        )
        .map_err(|e| format!("sealed checkpoint authentication failed: {e}"))?;

        // Clear any prior cooperative early-stop signal before resuming.
        let _ =
            self.cas
                .delete_state_object(&state_key(&resume.tenant_id, run_id_str, "stop.json"));

        let lm_raw = base64::engine::general_purpose::STANDARD
            .decode(&create.landmark_bundle_b64)
            .map_err(|e| format!("invalid landmark_bundle_b64: {e}"))?;
        let lm_bundle = FitLandmarkBundle::from_wire_bytes(&lm_raw)
            .map_err(|e| format!("invalid FitLandmarkBundle: {e}"))?;
        let fit_bytes = self
            .cas
            .get_blob_verified(&create.fit_jmat_sha256.to_ascii_lowercase())?;
        let holdout_bytes = self
            .cas
            .get_blob_verified(&create.holdout_jmat_sha256.to_ascii_lowercase())?;

        let effort = map_api_effort(create.effort);
        let decision = self.compute_plan_decision(&create, effort)?;
        let use_hybrid = (decision.mode == CarbonExecutionMode::HybridFanOut
            || self.force_hybrid_fanout)
            && self.eval_invoker.is_some();

        let session_dir = EngineSessionDir::create(&self.work_root, "engine-resume-")?;
        let fit_path = session_dir.path().join("fit.jmat");
        fs::write(&fit_path, &fit_bytes).map_err(|e| format!("stage fit.jmat: {e}"))?;
        let store_dir = session_dir.path().join("store");
        let scratch_dir = session_dir.path().join("scratch");

        let rebound_plan = CarbonStoreArchive::unpack_store_dir_and_rebind_fit(
            &decrypted_archive,
            &store_dir,
            &create.run_id,
            &fit_path,
        )
        .map_err(|e| format!("unpack & rebind paused checkpoint failed: {e}"))?;

        let ranker = try_open_embedded_ranker().ok().flatten();
        let extra_waves = resume
            .additional_waves
            .unwrap_or_else(|| effort.planned_waves())
            .max(1);
        let waves_target = effort.planned_waves().max(extra_waves);

        let outcome = self.drive_waves(
            &create,
            run_id_str,
            region,
            &decision,
            use_hybrid,
            &store_dir,
            &scratch_dir,
            &fit_bytes,
            rebound_plan,
            &lm_bundle,
            ranker.as_ref(),
            extra_waves,
            deadline_epoch_ms,
            t0,
        )?;

        self.finalize_and_persist_progress(
            &create,
            run_id_str,
            region,
            &decision,
            use_hybrid,
            &holdout_bytes,
            outcome,
            waves_target,
            t0.elapsed().as_millis() as u64,
        )
    }

    fn execute_predict(
        &self,
        run_id_str: &str,
        pred_req: &CarbonPredictRequest,
    ) -> Result<CarbonPredictResponse, String> {
        let run_id = run_id_str
            .parse()
            .map_err(|_| format!("invalid RunId '{run_id_str}'"))?;
        let champ_enc = self
            .cas
            .get_state_object(&state_key(
                &pred_req.tenant_id,
                run_id_str,
                "sealed_champion.enc",
            ))?
            .ok_or_else(|| format!("run '{run_id_str}' has no sealed champion model"))?;
        let (_wave, decrypted_champ) = SealedCheckpointEnvelope::open(
            &self.seal_key,
            &pred_req.tenant_id,
            run_id_str,
            &champ_enc,
        )
        .map_err(|e| format!("sealed champion authentication failed: {e}"))?;

        let predictions =
            predict_with_champion_bytes(&decrypted_champ, &pred_req.feature_names, &pred_req.rows)
                .map_err(|e| format!("champion predict failed: {e}"))?;

        let mut pred_le = Vec::with_capacity(predictions.len() * 8);
        for &val in &predictions {
            pred_le.extend_from_slice(&val.to_le_bytes());
        }
        Ok(CarbonPredictResponse {
            run_id,
            row_count: predictions.len() as u32,
            predictions,
            predictions_sha256: sha256_lower_hex(&pred_le),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn drive_waves(
        &self,
        create: &CarbonCreateRunRequest,
        run_str: &str,
        region: &str,
        decision: &CarbonPlanDecision,
        use_hybrid: bool,
        store_dir: &std::path::Path,
        scratch_dir: &std::path::Path,
        fit_bytes: &[u8],
        plan: Plan,
        lm_bundle: &FitLandmarkBundle,
        ranker: Option<&jope_service::run::learned::AdmittedRanker>,
        max_waves: u32,
        deadline_epoch_ms: Option<u64>,
        t0: Instant,
    ) -> Result<CarbonWaveRunOutcome, String> {
        let stop_key = state_key(&create.tenant_id, run_str, "stop.json");
        let status_key = state_key(&create.tenant_id, run_str, "status.json");
        let should_pause = |_wave: u32| -> bool {
            if let Some(deadline) = deadline_epoch_ms {
                if now_epoch_ms().saturating_add(DEADLINE_DRAIN_MARGIN_MS) >= deadline {
                    return true;
                }
            }
            matches!(self.cas.get_state_object(&stop_key), Ok(Some(_)))
        };

        let mode_wire = if use_hybrid {
            CarbonExecutionModeWire::HybridFanOut
        } else {
            CarbonExecutionModeWire::SoloSingleLambda
        };
        let mut on_wave = |tel: &jope_service::discover::r#loop::carbon::CarbonWaveTelemetry| {
            let live_progress = CarbonPublicWaveProgress {
                run_id: create.run_id.clone(),
                state: if tel.paused {
                    CarbonRunStateWire::Paused
                } else {
                    CarbonRunStateWire::Running
                },
                execution_mode: mode_wire,
                eval_memory_mb: decision.eval_class.memory_mb() as u32,
                eval_shards: if use_hybrid {
                    decision.shards_per_wave.max(2)
                } else {
                    1
                },
                wave_index: tel.wave_index,
                waves_target: max_waves,
                candidates_evaluated: DecimalU64(tel.candidates_evaluated),
                best_cv_loss: tel.best_cv_loss,
                best_cv_metric: tel.best_cv_metric,
                outer_holdout_loss: None,
                outer_holdout_metric: None,
                neural_seed_applied: ranker.is_some(),
                can_resume: tel.paused,
                sealed_checkpoint_sha256: None,
                sealed_champion_sha256: None,
                elapsed_ms: DecimalU64(t0.elapsed().as_millis() as u64),
                region: region.to_owned(),
                failure_message: None,
            };
            if let Ok(bytes) = serde_json::to_vec(&live_progress) {
                let _ = self.cas.put_state_object(&status_key, &bytes);
            }
        };

        let workers = if use_hybrid {
            decision.shards_per_wave.max(2) as usize
        } else {
            2usize
        };
        let session_opts = CarbonWaveSessionOptions {
            store_dir,
            plan,
            lm_bundle,
            ranker,
            live_workers: workers,
            max_waves_this_session: max_waves,
        };
        if use_hybrid {
            let eval_invoker = Arc::clone(
                self.eval_invoker
                    .as_ref()
                    .expect("use_hybrid verified eval_invoker.is_some()"),
            );
            let fit_sha = create.fit_jmat_sha256.to_ascii_lowercase();
            let remote_exec = CarbonRemoteEvalExecutor::new(move |req, fit_champ| {
                eval_invoker
                    .invoke_eval(&CarbonEvalWireRequest {
                        fit_jmat_sha256: fit_sha.clone(),
                        request: req.clone(),
                        fit_champion: fit_champ,
                    })
                    .map_err(|e| anyhow::anyhow!("{e}"))
            });
            run_carbon_waves(session_opts, &remote_exec, &should_pause, &mut on_wave)
                .map_err(|e| format!("hybrid wave execution failed: {e}"))
        } else {
            let inproc_exec = CarbonInProcessExecutor::open_fit(
                fit_bytes.to_vec(),
                &create.fit_jmat_sha256.to_ascii_lowercase(),
                scratch_dir.to_path_buf(),
            )
            .map_err(|e| format!("open fit in-process failed: {e}"))?;
            run_carbon_waves(session_opts, &inproc_exec, &should_pause, &mut on_wave)
                .map_err(|e| format!("solo wave execution failed: {e}"))
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn finalize_and_persist_progress(
        &self,
        create: &CarbonCreateRunRequest,
        run_str: &str,
        region: &str,
        decision: &CarbonPlanDecision,
        use_hybrid: bool,
        holdout_bytes: &[u8],
        outcome: CarbonWaveRunOutcome,
        waves_target: u32,
        elapsed_ms: u64,
    ) -> Result<CarbonPublicWaveProgress, String> {
        let sealed_ckp = SealedCheckpointEnvelope::seal(
            &self.seal_key,
            &create.tenant_id,
            run_str,
            u64::from(outcome.wave_index),
            &outcome.store_archive_bytes,
        )
        .map_err(|e| format!("seal checkpoint failed: {e}"))?;
        let ckp_sha = sha256_lower_hex(sealed_ckp.wire_bytes());
        self.cas.put_state_object(
            &state_key(&create.tenant_id, run_str, "sealed_checkpoint.enc"),
            sealed_ckp.wire_bytes(),
        )?;

        let mut champ_sha = None;
        let mut outer_loss = None;
        let mut outer_metric = None;
        if let Some(champ_bytes) = &outcome.champion_model_bytes {
            let sealed_champ = SealedCheckpointEnvelope::seal(
                &self.seal_key,
                &create.tenant_id,
                run_str,
                u64::from(outcome.wave_index),
                champ_bytes,
            )
            .map_err(|e| format!("seal champion failed: {e}"))?;
            champ_sha = Some(sha256_lower_hex(sealed_champ.wire_bytes()));
            self.cas.put_state_object(
                &state_key(&create.tenant_id, run_str, "sealed_champion.enc"),
                sealed_champ.wire_bytes(),
            )?;
            let (h_loss, h_val) = evaluate_outer_holdout_only(
                champ_bytes,
                holdout_bytes,
                &create.holdout_jmat_sha256.to_ascii_lowercase(),
                &create.target_name,
                &create.feature_names,
                &create.objective,
            )
            .map_err(|e| format!("outer holdout evaluation failed: {e}"))?;
            outer_loss = Some(h_loss);
            outer_metric = Some(h_val);
        }

        let final_state = if outcome.paused {
            CarbonRunStateWire::Paused
        } else {
            CarbonRunStateWire::Completed
        };
        let progress = CarbonPublicWaveProgress {
            run_id: create.run_id.clone(),
            state: final_state,
            execution_mode: if use_hybrid {
                CarbonExecutionModeWire::HybridFanOut
            } else {
                CarbonExecutionModeWire::SoloSingleLambda
            },
            eval_memory_mb: decision.eval_class.memory_mb() as u32,
            eval_shards: if use_hybrid {
                decision.shards_per_wave.max(2)
            } else {
                1
            },
            wave_index: outcome.wave_index,
            waves_target,
            candidates_evaluated: DecimalU64(outcome.candidates_evaluated),
            best_cv_loss: outcome.best_cv_loss,
            best_cv_metric: outcome.best_cv_metric,
            outer_holdout_loss: outer_loss,
            outer_holdout_metric: outer_metric,
            neural_seed_applied: outcome.neural_seed_applied,
            can_resume: final_state == CarbonRunStateWire::Paused,
            sealed_checkpoint_sha256: Some(ckp_sha),
            sealed_champion_sha256: champ_sha,
            elapsed_ms: DecimalU64(elapsed_ms),
            region: region.to_owned(),
            failure_message: None,
        };
        let progress_bytes =
            serde_json::to_vec(&progress).map_err(|e| format!("encode final progress: {e}"))?;
        self.cas.put_state_object(
            &state_key(&create.tenant_id, run_str, "status.json"),
            &progress_bytes,
        )?;
        Ok(progress)
    }

    fn compute_plan_decision(
        &self,
        create: &CarbonCreateRunRequest,
        effort: CarbonEffortTier,
    ) -> Result<CarbonPlanDecision, String> {
        let tier = match create.tier {
            CarbonTierWire::Free => CarbonProductTier::Free,
            CarbonTierWire::Pro => CarbonProductTier::Pro,
            CarbonTierWire::Enterprise => CarbonProductTier::Enterprise,
        };
        let total_rows = (create.fit_rows as usize).saturating_add(create.holdout_rows as usize);
        plan_carbon_execution(&CarbonPlanInput {
            product_tier: tier,
            effort,
            total_rows,
            fit_rows: create.fit_rows as usize,
            features: create.feature_names.len(),
            est_ms_per_candidate_1vcpu: 12.0,
        })
        .map_err(|e| e.to_string())
    }
}

impl CarbonEngineInvoker for CarbonEngineWorker {
    fn invoke_engine(&self, command: &CarbonEngineWireCommand) -> Result<Vec<u8>, String> {
        self.execute_command(command, None)
    }
}
