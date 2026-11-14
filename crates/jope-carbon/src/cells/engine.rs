//! Sealed `carbon engine` orchestrator (`3,008 MB` Graviton2 container, `ADR-0007`).
//!
//! Unwraps per-user `CUDK\x01` `U_DEK` envelopes via `ECDH P-256`, decrypts `CUEN\x01` artifacts in
//! `Zeroizing` memory, cryptographically verifies the `r882` DOPE + JopeDime neural fusion bundle
//! fail-closed, enforces Free (`Wave0SeedOnly`) vs Pro/Enterprise (`MultiWaveLearned`) neural
//! guidance, issues 60-second `FitJmat`-only slot subkeys to stateless `eval` shards (`RT-07`),
//! domain-seals `CheckpointArchive` vs `ChampionModel` double-bound to `U_DEK` (`RT-06`), and
//! returns E2E encrypted `/predict` outputs.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};
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
use jope_prep::{conditioning::FitLandmarkBundle, cuen::CuenPayloadKind};
use jope_service::discover::r#loop::{
    carbon::{
        evaluate_carbon_eval_shard, evaluate_outer_holdout_only, predict_with_champion_bytes,
        run_carbon_waves, try_open_embedded_ranker, CarbonInProcessExecutor,
        CarbonNeuralGuidancePolicy, CarbonRemoteEvalExecutor, CarbonStoreArchive,
        CarbonWaveRunOutcome, CarbonWaveSessionOptions,
    },
    Plan,
};
use zeroize::Zeroizing;

use crate::{
    cells::{
        edge::{map_api_effort, wire_to_product_tier},
        engine_invoker::{now_epoch_ms, CarbonEngineInvoker, CarbonEngineWireCommand},
        eval::{CarbonEvalDecryptionGrant, CarbonEvalInvoker, CarbonEvalWireRequest},
    },
    modes::planner::{
        estimate_ms_per_candidate_1vcpu, plan_carbon_execution, CarbonExecutionMode,
        CarbonPlanDecision, CarbonPlanInput,
    },
    role::CarbonEffortTier,
    security::{
        seal::{CarbonSealDomain, CarbonSealKey, SealedCheckpointEnvelope},
        user_envelope::{
            decode_p256_pubkey_b64, decrypt_cuen_payload, derive_cuen_slot_subkey,
            encrypt_cuen_payload, generate_random_u_dek, unwrap_u_dek_with_kek,
            wrap_u_dek_to_pubkey, CarbonEncryptedPredictPayload, CarbonEngineKek,
            CAS_BLOB_CUEN_SCOPE,
        },
    },
    storage::cas::{
        canonical_blob_scope, canonical_run_state_key, sha256_lower_hex, CarbonCasStore,
    },
};

const DEADLINE_DRAIN_MARGIN_MS: u64 = 20_000;
const MAX_INLINE_PREDICT_CUEN_BYTES: usize = 1_572_864;

struct EngineSessionDir {
    path: PathBuf,
}

impl EngineSessionDir {
    fn create(parent: &Path, prefix: &str) -> Result<Self, String> {
        let dir = parent.join(format!("{prefix}{}", ulid::Ulid::new()));
        fs::create_dir_all(&dir)
            .map_err(|e| format!("create session dir {}: {e}", dir.display()))?;
        Ok(Self { path: dir })
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for EngineSessionDir {
    fn drop(&mut self) {
        scrub_and_remove_dir(&self.path);
    }
}

fn scrub_and_remove_dir(dir: &Path) {
    if let Ok(entries) = fs::read_dir(dir) {
        let zeros = vec![0u8; 65_536];
        for entry in entries.flatten() {
            let p = entry.path();
            if let Ok(meta) = fs::symlink_metadata(&p) {
                if meta.is_dir() {
                    scrub_and_remove_dir(&p);
                } else if meta.is_file() {
                    let mut rem = meta.len();
                    if rem > 0 {
                        if let Ok(mut f) = fs::OpenOptions::new().write(true).open(&p) {
                            use std::io::Write as _;
                            while rem > 0 {
                                let chunk = (rem.min(zeros.len() as u64)) as usize;
                                if f.write_all(&zeros[..chunk]).is_err() {
                                    break;
                                }
                                rem -= chunk as u64;
                            }
                            let _ = f.sync_data();
                        }
                    }
                    let _ = fs::remove_file(&p);
                }
            }
        }
    }
    let _ = fs::remove_dir_all(dir);
}

fn select_neural_guidance_policy(tier: CarbonTierWire) -> CarbonNeuralGuidancePolicy {
    match tier {
        CarbonTierWire::Free => CarbonNeuralGuidancePolicy::Wave0SeedOnly,
        CarbonTierWire::Pro | CarbonTierWire::Enterprise => {
            CarbonNeuralGuidancePolicy::MultiWaveLearned
        }
    }
}

/// `CarbonEngineWorker` is a struct implementing the sealed `carbon engine` cell.
pub struct CarbonEngineWorker {
    cas: Arc<dyn CarbonCasStore>,
    eval_invoker: Option<Arc<dyn CarbonEvalInvoker>>,
    seal_key: CarbonSealKey,
    engine_kek: Option<Arc<CarbonEngineKek>>,
    work_root: PathBuf,
    force_hybrid_fanout: bool,
}

struct OpenedEngineRunInputs {
    landmark_bundle: FitLandmarkBundle,
    fit_bytes: Zeroizing<Vec<u8>>,
    fit_plaintext_sha256: String,
    holdout_bytes: Zeroizing<Vec<u8>>,
    holdout_plaintext_sha256: String,
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
            engine_kek: None,
            work_root,
            force_hybrid_fanout: false,
        })
    }

    pub fn with_engine_kek(mut self, kek: Option<Arc<CarbonEngineKek>>) -> Self {
        self.engine_kek = kek;
        self
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
                match self.execute_start_run(request, region, deadline_epoch_ms) {
                    Ok(progress) => serde_json::to_vec(&progress)
                        .map_err(|e| format!("encode start progress: {e}")),
                    Err(err) => {
                        self.record_failed_status(
                            &request.tenant_id,
                            &request.run_id.to_string(),
                            &err,
                        );
                        Err(err)
                    }
                }
            }
            CarbonEngineWireCommand::ResumeRun {
                run_id,
                resume,
                region,
            } => match self.execute_resume_run(run_id, resume, region, deadline_epoch_ms) {
                Ok(progress) => serde_json::to_vec(&progress)
                    .map_err(|e| format!("encode resume progress: {e}")),
                Err(err) => {
                    self.record_failed_status(&resume.tenant_id, run_id, &err);
                    Err(err)
                }
            },
            CarbonEngineWireCommand::Predict { run_id, predict } => {
                let pred = self.execute_predict(run_id, predict)?;
                serde_json::to_vec(&pred).map_err(|e| format!("encode predict response: {e}"))
            }
        }
    }

    fn record_failed_status(&self, tenant_id: &str, run_id_str: &str, err_msg: &str) {
        let Ok(status_key) = canonical_run_state_key(tenant_id, run_id_str, "status.json") else {
            return;
        };
        if let Ok(Some(cur)) = self.cas.get_state_object(&status_key) {
            if let Ok(mut prog) = serde_json::from_slice::<CarbonPublicWaveProgress>(&cur) {
                prog.state = CarbonRunStateWire::Failed;
                prog.can_resume = false;
                prog.failure_message = Some(err_msg.to_owned());
                if let Ok(bytes) = serde_json::to_vec(&prog) {
                    let _ = self.cas.put_state_object(&status_key, &bytes);
                }
            }
        }
    }

    fn unwrap_optional_udek(
        &self,
        encrypted_cuen: bool,
        wrapped_u_dek_b64: Option<&str>,
        tenant_id: &str,
        run_id_str: &str,
    ) -> Result<Option<Zeroizing<[u8; 32]>>, String> {
        if !encrypted_cuen {
            return Ok(None);
        }
        let kek = self.engine_kek.as_ref().ok_or_else(|| {
            "carbon-engine has no P-256 KEK configured for encrypted run".to_owned()
        })?;
        let b64 = wrapped_u_dek_b64
            .ok_or_else(|| "missing wrapped_u_dek_b64 for encrypted run".to_owned())?;
        let cudk_bytes = base64::engine::general_purpose::STANDARD
            .decode(b64)
            .map_err(|e| format!("invalid wrapped_u_dek_b64: {e}"))?;
        let dek = unwrap_u_dek_with_kek(kek, &cudk_bytes, tenant_id, run_id_str)
            .map_err(|e| e.to_string())?;
        Ok(Some(dek))
    }

    fn load_and_open_inputs(
        &self,
        create: &CarbonCreateRunRequest,
        run_str: &str,
        u_dek: Option<&[u8; 32]>,
    ) -> Result<OpenedEngineRunInputs, String> {
        let lm_wire = base64::engine::general_purpose::STANDARD
            .decode(&create.landmark_bundle_b64)
            .map_err(|e| format!("invalid landmark_bundle_b64: {e}"))?;
        let lm_bundle = if let Some(dek) = u_dek {
            let pt = decrypt_cuen_payload(
                dek,
                CuenPayloadKind::LandmarkJlmb,
                &create.tenant_id,
                run_str,
                &lm_wire,
            )
            .map_err(|e| format!("decrypt landmark_bundle failed: {e}"))?;
            FitLandmarkBundle::from_wire_bytes(&pt)
                .map_err(|e| format!("invalid decrypted FitLandmarkBundle: {e}"))?
        } else {
            FitLandmarkBundle::from_wire_bytes(&lm_wire)
                .map_err(|e| format!("invalid FitLandmarkBundle: {e}"))?
        };
        if lm_bundle.feature_count() != create.feature_names.len() {
            return Err("FitLandmarkBundle feature_count mismatch with feature_names".to_owned());
        }

        let blob_scope = match create.upload_id.as_deref() {
            Some(uid) => Some(canonical_blob_scope(&create.tenant_id, uid)?),
            None => None,
        };
        let raw_fit = self.cas.get_scoped_blob_verified(
            blob_scope.as_deref(),
            &create.fit_jmat_sha256.to_ascii_lowercase(),
        )?;
        let raw_holdout = self.cas.get_scoped_blob_verified(
            blob_scope.as_deref(),
            &create.holdout_jmat_sha256.to_ascii_lowercase(),
        )?;

        if let Some(dek) = u_dek {
            let fit_pt = decrypt_cuen_payload(
                dek,
                CuenPayloadKind::FitJmat,
                &create.tenant_id,
                CAS_BLOB_CUEN_SCOPE,
                &raw_fit,
            )
            .map_err(|e| format!("decrypt fit.jmat CUEN failed: {e}"))?;
            let holdout_pt = decrypt_cuen_payload(
                dek,
                CuenPayloadKind::HoldoutJmat,
                &create.tenant_id,
                CAS_BLOB_CUEN_SCOPE,
                &raw_holdout,
            )
            .map_err(|e| format!("decrypt holdout.jmat CUEN failed: {e}"))?;
            let fit_sha = sha256_lower_hex(&fit_pt);
            let holdout_sha = sha256_lower_hex(&holdout_pt);
            Ok(OpenedEngineRunInputs {
                landmark_bundle: lm_bundle,
                fit_bytes: fit_pt,
                fit_plaintext_sha256: fit_sha,
                holdout_bytes: holdout_pt,
                holdout_plaintext_sha256: holdout_sha,
            })
        } else {
            let fit_sha = create.fit_jmat_sha256.to_ascii_lowercase();
            let holdout_sha = create.holdout_jmat_sha256.to_ascii_lowercase();
            Ok(OpenedEngineRunInputs {
                landmark_bundle: lm_bundle,
                fit_bytes: Zeroizing::new(raw_fit),
                fit_plaintext_sha256: fit_sha,
                holdout_bytes: Zeroizing::new(raw_holdout),
                holdout_plaintext_sha256: holdout_sha,
            })
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
        let u_dek = self.unwrap_optional_udek(
            create.encrypted_cuen,
            create.wrapped_u_dek_b64.as_deref(),
            &create.tenant_id,
            &run_str,
        )?;
        let OpenedEngineRunInputs {
            landmark_bundle: lm_bundle,
            fit_bytes,
            fit_plaintext_sha256: fit_pt_sha,
            holdout_bytes,
            holdout_plaintext_sha256: holdout_pt_sha,
        } = self.load_and_open_inputs(create, &run_str, u_dek.as_deref())?;

        let effort = map_api_effort(create.effort);
        let decision = self.compute_plan_decision(create, effort)?;
        let use_hybrid = (decision.mode == CarbonExecutionMode::HybridFanOut
            || self.force_hybrid_fanout)
            && self.eval_invoker.is_some();

        let session_dir = EngineSessionDir::create(&self.work_root, "engine-run-")?;
        let fit_path = session_dir.path().join("fit.jmat");
        fs::write(&fit_path, &*fit_bytes).map_err(|e| format!("stage fit.jmat: {e}"))?;
        let store_dir = session_dir.path().join("store");
        let scratch_dir = session_dir.path().join("scratch");

        let ranker = try_open_embedded_ranker()
            .map_err(|e| format!("r882 neural ranker verification failed: {e}"))?;
        let bundle_digest = ranker
            .as_ref()
            .and_then(|r| r.digest().strip_prefix("sha256:").map(str::to_owned))
            .unwrap_or_else(|| "0".repeat(64));

        let problem_hash = sha256_lower_hex(
            format!(
                "{}:{}:{}:{}",
                fit_pt_sha,
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
            matrix_sha256: fit_pt_sha.clone(),
            target: create.target_name.clone(),
            features: create.feature_names.clone(),
            objective: create.objective.clone(),
            seed: create.seed.0,
            planned_workers: decision.shards_per_wave.max(2),
            effort: create.effort,
            evaluations: total_evals.max(16),
            wall_seconds: u64::from(effort.lease_ceiling_secs()).max(60),
            evaluation_wall_ms: 30_000,
            evaluation_rss_bytes: (decision.eval_class.memory_mb() as u64) * 1_048_576,
            groups: None,
            times: None,
            gap_rows: 0,
        };

        let (outcome, eval_invocations, eval_total_wall_ms) = self.drive_waves(
            create,
            &run_str,
            region,
            &decision,
            use_hybrid,
            &store_dir,
            &scratch_dir,
            &fit_bytes,
            &fit_pt_sha,
            u_dek.as_deref(),
            plan,
            &lm_bundle,
            ranker.as_ref(),
            effort.planned_waves(),
            deadline_epoch_ms,
            t0,
            0,
            0,
        )?;

        self.finalize_and_persist_progress(
            create,
            &run_str,
            region,
            &decision,
            use_hybrid,
            &holdout_bytes,
            &holdout_pt_sha,
            u_dek.as_deref(),
            outcome,
            effort.planned_waves(),
            t0.elapsed().as_millis() as u64,
            eval_invocations,
            eval_total_wall_ms,
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
        let req_key = canonical_run_state_key(&resume.tenant_id, run_id_str, "request.json")?;
        let req_bytes = self
            .cas
            .get_state_object(&req_key)?
            .ok_or_else(|| format!("missing stored request.json for run '{run_id_str}'"))?;
        let mut create: CarbonCreateRunRequest =
            serde_json::from_slice(&req_bytes).map_err(|e| format!("corrupt request.json: {e}"))?;
        if let Some(new_effort) = resume.effort {
            create.effort = new_effort;
        }
        let u_dek = self.unwrap_optional_udek(
            create.encrypted_cuen,
            resume.wrapped_u_dek_b64.as_deref(),
            &resume.tenant_id,
            run_id_str,
        )?;

        let ckp_key =
            canonical_run_state_key(&resume.tenant_id, run_id_str, "sealed_checkpoint.enc")?;
        let sealed_ckp_bytes = self
            .cas
            .get_state_object(&ckp_key)?
            .ok_or_else(|| format!("run '{run_id_str}' has no sealed_checkpoint.enc to resume"))?;

        let (_saved_wave, decrypted_archive) = SealedCheckpointEnvelope::open_with_domain(
            &self.seal_key,
            CarbonSealDomain::CheckpointArchive,
            &resume.tenant_id,
            run_id_str,
            u_dek.as_deref(),
            &sealed_ckp_bytes,
        )
        .map_err(|e| format!("sealed checkpoint authentication failed: {e}"))?;

        if let Ok(stop_key) = canonical_run_state_key(&resume.tenant_id, run_id_str, "stop.json") {
            let _ = self.cas.delete_state_object(&stop_key);
        }

        let OpenedEngineRunInputs {
            landmark_bundle: lm_bundle,
            fit_bytes,
            fit_plaintext_sha256: fit_pt_sha,
            holdout_bytes,
            holdout_plaintext_sha256: holdout_pt_sha,
        } = self.load_and_open_inputs(&create, run_id_str, u_dek.as_deref())?;

        let effort = map_api_effort(create.effort);
        let decision = self.compute_plan_decision(&create, effort)?;
        let use_hybrid = (decision.mode == CarbonExecutionMode::HybridFanOut
            || self.force_hybrid_fanout)
            && self.eval_invoker.is_some();

        let session_dir = EngineSessionDir::create(&self.work_root, "engine-resume-")?;
        let fit_path = session_dir.path().join("fit.jmat");
        fs::write(&fit_path, &*fit_bytes).map_err(|e| format!("stage fit.jmat: {e}"))?;
        let store_dir = session_dir.path().join("store");
        let scratch_dir = session_dir.path().join("scratch");

        let rebound_plan = CarbonStoreArchive::unpack_store_dir_and_rebind_fit(
            &decrypted_archive,
            &store_dir,
            &create.run_id,
            &fit_path,
        )
        .map_err(|e| format!("unpack & rebind paused checkpoint failed: {e}"))?;

        let ranker = try_open_embedded_ranker()
            .map_err(|e| format!("r882 neural ranker verification failed: {e}"))?;
        let extra_waves = resume
            .additional_waves
            .unwrap_or_else(|| effort.planned_waves())
            .max(1);
        let waves_target = effort.planned_waves().max(extra_waves);

        let (prior_eval_invocations, prior_eval_wall_ms, prior_elapsed_ms) =
            canonical_run_state_key(&resume.tenant_id, run_id_str, "status.json")
                .ok()
                .and_then(|k| self.cas.get_state_object(&k).ok().flatten())
                .and_then(|b| serde_json::from_slice::<CarbonPublicWaveProgress>(&b).ok())
                .map(|p| (p.eval_invocations.0, p.eval_total_wall_ms.0, p.elapsed_ms.0))
                .unwrap_or((0, 0, 0));

        let (outcome, eval_invocations, eval_total_wall_ms) = self.drive_waves(
            &create,
            run_id_str,
            region,
            &decision,
            use_hybrid,
            &store_dir,
            &scratch_dir,
            &fit_bytes,
            &fit_pt_sha,
            u_dek.as_deref(),
            rebound_plan,
            &lm_bundle,
            ranker.as_ref(),
            extra_waves,
            deadline_epoch_ms,
            t0,
            prior_eval_invocations,
            prior_eval_wall_ms,
        )?;

        self.finalize_and_persist_progress(
            &create,
            run_id_str,
            region,
            &decision,
            use_hybrid,
            &holdout_bytes,
            &holdout_pt_sha,
            u_dek.as_deref(),
            outcome,
            waves_target,
            prior_elapsed_ms.saturating_add(t0.elapsed().as_millis() as u64),
            eval_invocations,
            eval_total_wall_ms,
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
        let req_key = canonical_run_state_key(&pred_req.tenant_id, run_id_str, "request.json")?;
        let req_bytes = self
            .cas
            .get_state_object(&req_key)?
            .ok_or_else(|| format!("missing request.json for run '{run_id_str}'"))?;
        let stored_create: CarbonCreateRunRequest =
            serde_json::from_slice(&req_bytes).map_err(|e| format!("corrupt request.json: {e}"))?;
        let tier = wire_to_product_tier(stored_create.tier);

        let u_dek = self.unwrap_optional_udek(
            stored_create.encrypted_cuen,
            pred_req.wrapped_u_dek_b64.as_deref(),
            &pred_req.tenant_id,
            run_id_str,
        )?;

        let champ_key =
            canonical_run_state_key(&pred_req.tenant_id, run_id_str, "sealed_champion.enc")?;
        let champ_enc = self
            .cas
            .get_state_object(&champ_key)?
            .ok_or_else(|| format!("run '{run_id_str}' has no sealed champion model"))?;
        let (_wave, decrypted_champ) = SealedCheckpointEnvelope::open_with_domain(
            &self.seal_key,
            CarbonSealDomain::ChampionModel,
            &pred_req.tenant_id,
            run_id_str,
            u_dek.as_deref(),
            &champ_enc,
        )
        .map_err(|e| format!("sealed champion authentication failed: {e}"))?;

        let (feat_names, rows) = if let Some(dek) = u_dek.as_ref() {
            let enc_wire = if let Some(cas_sha) = pred_req.encrypted_rows_cas_sha256.as_deref() {
                let in_scope = match pred_req.predict_upload_id.as_deref() {
                    Some(puid) => Some(canonical_blob_scope(&pred_req.tenant_id, puid)?),
                    None => None,
                };
                let sha_lc = cas_sha.to_ascii_lowercase();
                let bytes = self
                    .cas
                    .get_scoped_blob_verified(in_scope.as_deref(), &sha_lc)?;
                let _ = self.cas.delete_scoped_blob(in_scope.as_deref(), &sha_lc);
                bytes
            } else {
                let enc_b64 = pred_req
                    .encrypted_rows_cuen_b64
                    .as_deref()
                    .ok_or_else(|| "missing encrypted_rows_cuen_b64".to_owned())?;
                base64::engine::general_purpose::STANDARD
                    .decode(enc_b64)
                    .map_err(|e| format!("invalid encrypted_rows_cuen_b64: {e}"))?
            };
            let pt = decrypt_cuen_payload(
                dek,
                CuenPayloadKind::PredictFeatures,
                &pred_req.tenant_id,
                run_id_str,
                &enc_wire,
            )
            .map_err(|e| format!("decrypt predict features failed: {e}"))?;
            let decoded: CarbonEncryptedPredictPayload = serde_json::from_slice(&pt)
                .map_err(|e| format!("invalid encrypted predict JSON: {e}"))?;
            (decoded.feature_names, decoded.rows)
        } else {
            (pred_req.feature_names.clone(), pred_req.rows.clone())
        };

        if let Some(max_r) = tier.max_predict_rows_per_call() {
            if rows.len() > max_r {
                return Err(format!(
                    "{} tier admits at most {max_r} rows per /predict call",
                    tier.as_str()
                ));
            }
        }
        if let Some(max_f) = tier.max_features() {
            if feat_names.len() > max_f {
                return Err(format!(
                    "{} tier admits at most {max_f} features per /predict call",
                    tier.as_str()
                ));
            }
        }

        let predictions = predict_with_champion_bytes(&decrypted_champ, &feat_names, &rows)
            .map_err(|e| format!("champion predict failed: {e}"))?;

        let mut pred_le = Zeroizing::new(Vec::with_capacity(predictions.len() * 8));
        for &val in &predictions {
            pred_le.extend_from_slice(&val.to_le_bytes());
        }
        let predictions_sha256 = sha256_lower_hex(&pred_le);
        let row_count = predictions.len() as u32;

        if stored_create.encrypted_cuen {
            let client_pub_b64 = pred_req
                .client_response_pubkey_b64
                .as_deref()
                .ok_or_else(|| "missing client_response_pubkey_b64".to_owned())?;
            let client_pub = decode_p256_pubkey_b64(client_pub_b64).map_err(|e| e.to_string())?;
            let r_dek = generate_random_u_dek().map_err(|e| e.to_string())?;
            let resp_cuen = encrypt_cuen_payload(
                &r_dek,
                CuenPayloadKind::PredictOutputs,
                &pred_req.tenant_id,
                run_id_str,
                &pred_le,
            )
            .map_err(|e| e.to_string())?;
            let resp_cudk =
                wrap_u_dek_to_pubkey(&r_dek, &client_pub, &pred_req.tenant_id, run_id_str)
                    .map_err(|e| e.to_string())?;
            let resp_cudk_b64 = base64::engine::general_purpose::STANDARD.encode(resp_cudk);
            if pred_req.prefer_cas_output || resp_cuen.len() > MAX_INLINE_PREDICT_CUEN_BYTES {
                let out_sha = sha256_lower_hex(&resp_cuen);
                let out_scope = match pred_req.predict_upload_id.as_deref() {
                    Some(puid) => Some(canonical_blob_scope(&pred_req.tenant_id, puid)?),
                    None => Some(canonical_blob_scope(&pred_req.tenant_id, run_id_str)?),
                };
                self.cas
                    .put_scoped_blob_verified(out_scope.as_deref(), &out_sha, &resp_cuen)?;
                return Ok(CarbonPredictResponse {
                    run_id,
                    predictions: Vec::new(),
                    predictions_sha256,
                    row_count,
                    encrypted_predictions_cuen_b64: None,
                    encrypted_predictions_cas_url: None,
                    encrypted_predictions_cas_sha256: Some(out_sha),
                    response_cudk_b64: Some(resp_cudk_b64),
                });
            }
            return Ok(CarbonPredictResponse {
                run_id,
                predictions: Vec::new(),
                predictions_sha256,
                row_count,
                encrypted_predictions_cuen_b64: Some(
                    base64::engine::general_purpose::STANDARD.encode(resp_cuen),
                ),
                encrypted_predictions_cas_url: None,
                encrypted_predictions_cas_sha256: None,
                response_cudk_b64: Some(resp_cudk_b64),
            });
        }

        Ok(CarbonPredictResponse {
            run_id,
            row_count,
            predictions,
            predictions_sha256,
            encrypted_predictions_cuen_b64: None,
            encrypted_predictions_cas_url: None,
            encrypted_predictions_cas_sha256: None,
            response_cudk_b64: None,
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
        store_dir: &Path,
        scratch_dir: &Path,
        fit_bytes: &[u8],
        fit_pt_sha: &str,
        u_dek: Option<&[u8; 32]>,
        plan: Plan,
        lm_bundle: &FitLandmarkBundle,
        ranker: Option<&jope_service::run::learned::AdmittedRanker>,
        max_waves: u32,
        deadline_epoch_ms: Option<u64>,
        t0: Instant,
        initial_eval_invocations: u64,
        initial_eval_wall_ms: u64,
    ) -> Result<(CarbonWaveRunOutcome, u64, u64), String> {
        let stop_key = canonical_run_state_key(&create.tenant_id, run_str, "stop.json")?;
        let status_key = canonical_run_state_key(&create.tenant_id, run_str, "status.json")?;
        let guidance_policy = select_neural_guidance_policy(create.tier);
        let ranker_digest = ranker.map(|r| r.digest().to_owned());
        let eval_invocations = Arc::new(AtomicU64::new(initial_eval_invocations));
        let eval_total_wall_ms = Arc::new(AtomicU64::new(initial_eval_wall_ms));

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
        let guidance_label = if ranker.is_some() {
            Some(
                match guidance_policy {
                    CarbonNeuralGuidancePolicy::Wave0SeedOnly => "wave0_seed_only",
                    CarbonNeuralGuidancePolicy::MultiWaveLearned => "multi_wave_learned",
                }
                .to_owned(),
            )
        } else {
            Some("unseeded".to_owned())
        };
        let effective_hybrid_shards = if use_hybrid {
            resolve_effective_eval_shards(decision.shards_per_wave)
        } else {
            0
        };
        let eval_inv_wave = Arc::clone(&eval_invocations);
        let eval_ms_wave = Arc::clone(&eval_total_wall_ms);
        let mut on_wave = |tel: &jope_service::discover::r#loop::carbon::CarbonWaveTelemetry| {
            // Keep `status.json` in `Running` until `finalize_and_persist_progress` has durably
            // sealed `sealed_checkpoint.enc` + `sealed_champion.enc` and scored outer holdout!
            let live_progress = CarbonPublicWaveProgress {
                run_id: create.run_id.clone(),
                state: CarbonRunStateWire::Running,
                execution_mode: mode_wire,
                eval_memory_mb: decision.eval_class.memory_mb() as u32,
                eval_shards: effective_hybrid_shards,
                eval_invocations: DecimalU64(eval_inv_wave.load(Ordering::Relaxed)),
                eval_total_wall_ms: DecimalU64(eval_ms_wave.load(Ordering::Relaxed)),
                wave_index: tel.wave_index,
                waves_target: max_waves,
                candidates_evaluated: DecimalU64(tel.candidates_evaluated),
                best_cv_loss: tel.best_cv_loss,
                best_cv_metric: tel.best_cv_metric,
                outer_holdout_loss: None,
                outer_holdout_metric: None,
                neural_seed_applied: ranker.is_some(),
                neural_guidance_mode: guidance_label.clone(),
                bundle_digest: ranker_digest.clone(),
                tier_applied: Some(create.tier),
                encrypted_cuen: create.encrypted_cuen,
                can_resume: false,
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
            effective_hybrid_shards as usize
        } else {
            2usize
        };
        let session_opts = CarbonWaveSessionOptions {
            store_dir,
            plan,
            lm_bundle,
            ranker,
            guidance_policy,
            live_workers: workers,
            max_waves_this_session: max_waves,
        };
        let outcome =
            if let (true, Some(eval_invoker_ref)) = (use_hybrid, self.eval_invoker.as_ref()) {
                let eval_invoker = Arc::clone(eval_invoker_ref);
                let cas_fit_sha = create.fit_jmat_sha256.to_ascii_lowercase();
                let cas_blob_scope = match create.upload_id.as_deref() {
                    Some(uid) => Some(canonical_blob_scope(&create.tenant_id, uid)?),
                    None => None,
                };
                let grant_template = if let Some(dek) = u_dek {
                    let subkey = derive_cuen_slot_subkey(
                        dek,
                        CuenPayloadKind::FitJmat,
                        &create.tenant_id,
                        CAS_BLOB_CUEN_SCOPE,
                    )
                    .map_err(|e| e.to_string())?;
                    Some((
                        create.tenant_id.clone(),
                        base64::engine::general_purpose::STANDARD.encode(*subkey),
                        fit_pt_sha.to_owned(),
                    ))
                } else {
                    None
                };
                let fallback_fit_bytes = Arc::new(Zeroizing::new(fit_bytes.to_vec()));
                let fallback_fit_pt_sha = fit_pt_sha.to_owned();
                let fallback_scratch_dir = scratch_dir.to_path_buf();
                let eval_inv_exec = Arc::clone(&eval_invocations);
                let eval_ms_exec = Arc::clone(&eval_total_wall_ms);
                let remote_exec = CarbonRemoteEvalExecutor::new(move |req, fit_champ| {
                    let decryption_grant = grant_template.as_ref().map(|(tid, sub_b64, pt_sha)| {
                        CarbonEvalDecryptionGrant {
                            tenant_id: tid.clone(),
                            fit_slot_subkey_b64: sub_b64.clone(),
                            fit_plaintext_sha256: pt_sha.clone(),
                            exp_epoch_secs: (now_epoch_ms() / 1_000).saturating_add(60),
                        }
                    });
                    let shard_t0 = Instant::now();
                    let res = eval_invoker.invoke_eval(&CarbonEvalWireRequest {
                        fit_jmat_sha256: cas_fit_sha.clone(),
                        blob_scope: cas_blob_scope.clone(),
                        request: req.clone(),
                        fit_champion: fit_champ,
                        decryption_grant,
                    });
                    match res {
                        Ok(out) => {
                            let billed_ms = (shard_t0.elapsed().as_millis() as u64).max(1);
                            eval_inv_exec.fetch_add(1, Ordering::Relaxed);
                            eval_ms_exec.fetch_add(billed_ms, Ordering::Relaxed);
                            Ok(out)
                        }
                        Err(err) if is_retriable_eval_capacity_error(&err) => {
                            // Zero-drop saturation safety valve: if regional `carbon-eval` concurrency
                            // saturates under multi-hundred-user bursts, evaluate this candidate batch
                            // directly inside `carbon-engine`'s 3,008 MB Graviton2 container!
                            evaluate_carbon_eval_shard(
                                fallback_fit_bytes.as_slice(),
                                &fallback_fit_pt_sha,
                                &fallback_scratch_dir,
                                req,
                                fit_champ,
                            )
                        }
                        Err(err) => Err(anyhow::anyhow!("{err}")),
                    }
                });
                run_carbon_waves(session_opts, &remote_exec, &should_pause, &mut on_wave)
                    .map_err(|e| format!("hybrid wave execution failed: {e}"))?
            } else {
                let inproc_exec = CarbonInProcessExecutor::open_fit(
                    fit_bytes.to_vec(),
                    fit_pt_sha,
                    scratch_dir.to_path_buf(),
                )
                .map_err(|e| format!("open fit in-process failed: {e}"))?;
                run_carbon_waves(session_opts, &inproc_exec, &should_pause, &mut on_wave)
                    .map_err(|e| format!("solo wave execution failed: {e}"))?
            };
        Ok((
            outcome,
            eval_invocations.load(Ordering::Relaxed),
            eval_total_wall_ms.load(Ordering::Relaxed),
        ))
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
        holdout_pt_sha: &str,
        u_dek: Option<&[u8; 32]>,
        outcome: CarbonWaveRunOutcome,
        waves_target: u32,
        elapsed_ms: u64,
        eval_invocations: u64,
        eval_total_wall_ms: u64,
    ) -> Result<CarbonPublicWaveProgress, String> {
        let sealed_ckp = SealedCheckpointEnvelope::seal_with_domain(
            &self.seal_key,
            CarbonSealDomain::CheckpointArchive,
            &create.tenant_id,
            run_str,
            u64::from(outcome.wave_index),
            u_dek,
            &outcome.store_archive_bytes,
        )
        .map_err(|e| format!("seal checkpoint failed: {e}"))?;
        let ckp_sha = sha256_lower_hex(sealed_ckp.wire_bytes());
        self.cas.put_state_object(
            &canonical_run_state_key(&create.tenant_id, run_str, "sealed_checkpoint.enc")?,
            sealed_ckp.wire_bytes(),
        )?;

        let mut champ_sha = None;
        let mut outer_loss = None;
        let mut outer_metric = None;
        if let Some(champ_bytes) = &outcome.champion_model_bytes {
            let sealed_champ = SealedCheckpointEnvelope::seal_with_domain(
                &self.seal_key,
                CarbonSealDomain::ChampionModel,
                &create.tenant_id,
                run_str,
                u64::from(outcome.wave_index),
                u_dek,
                champ_bytes,
            )
            .map_err(|e| format!("seal champion failed: {e}"))?;
            champ_sha = Some(sha256_lower_hex(sealed_champ.wire_bytes()));
            self.cas.put_state_object(
                &canonical_run_state_key(&create.tenant_id, run_str, "sealed_champion.enc")?,
                sealed_champ.wire_bytes(),
            )?;
            let (h_loss, h_val) = evaluate_outer_holdout_only(
                champ_bytes,
                holdout_bytes,
                holdout_pt_sha,
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
                resolve_effective_eval_shards(decision.shards_per_wave)
            } else {
                0
            },
            eval_invocations: DecimalU64(eval_invocations),
            eval_total_wall_ms: DecimalU64(eval_total_wall_ms),
            wave_index: outcome.wave_index,
            waves_target,
            candidates_evaluated: DecimalU64(outcome.candidates_evaluated),
            best_cv_loss: outcome.best_cv_loss,
            best_cv_metric: outcome.best_cv_metric,
            outer_holdout_loss: outer_loss,
            outer_holdout_metric: outer_metric,
            neural_seed_applied: outcome.neural_seed_applied,
            neural_guidance_mode: Some(outcome.neural_guidance_mode.to_owned()),
            bundle_digest: outcome.bundle_digest,
            tier_applied: Some(create.tier),
            encrypted_cuen: create.encrypted_cuen,
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
            &canonical_run_state_key(&create.tenant_id, run_str, "status.json")?,
            &progress_bytes,
        )?;
        Ok(progress)
    }

    fn compute_plan_decision(
        &self,
        create: &CarbonCreateRunRequest,
        effort: CarbonEffortTier,
    ) -> Result<CarbonPlanDecision, String> {
        let tier = wire_to_product_tier(create.tier);
        let fit_rows = create.fit_rows as usize;
        let feature_count = create.feature_names.len();
        let total_rows = fit_rows.saturating_add(create.holdout_rows as usize);
        plan_carbon_execution(&CarbonPlanInput {
            product_tier: tier,
            effort,
            total_rows,
            fit_rows,
            features: feature_count,
            est_ms_per_candidate_1vcpu: estimate_ms_per_candidate_1vcpu(fit_rows, feature_count),
        })
        .map_err(|e| e.to_string())
    }
}

impl CarbonEngineInvoker for CarbonEngineWorker {
    fn invoke_engine(&self, command: &CarbonEngineWireCommand) -> Result<Vec<u8>, String> {
        self.execute_command(command, None)
    }

    fn invoke_engine_async(&self, command: &CarbonEngineWireCommand) -> Result<(), String> {
        let _ = self.execute_command(command, None)?;
        Ok(())
    }
}

fn resolve_effective_eval_shards(planned: u32) -> u32 {
    let base = planned.max(2);
    match std::env::var("CARBON_MAX_EVAL_SHARDS") {
        Ok(raw) => match raw.trim().parse::<u32>() {
            Ok(cap) if (1..=16).contains(&cap) => base.min(cap),
            _ => base,
        },
        Err(_) => base,
    }
}

fn is_retriable_eval_capacity_error(err: &str) -> bool {
    let lc = err.to_ascii_lowercase();
    lc.contains("http 429")
        || lc.contains("http 502")
        || lc.contains("http 503")
        || lc.contains("http 504")
        || lc.contains("toomanyrequests")
        || lc.contains("rate exceeded")
        || lc.contains("serviceexception")
        || lc.contains("aws lambda eval invoke")
}
