use std::{cell::Cell, rc::Rc};

/// W2-T04 G-11e: `context_cost.cached` relaxes the arm-charge check of a
/// qualified CPU context (`search_context::load_for_comparison`), yet no
/// producer issues a cached cost; only deserialized comparison JSON can
/// carry `true`. A cost claiming a cache is refused at its receiver.
#[test]
fn a_context_cost_cannot_claim_a_cache_nothing_issued() {
    let dir = tempfile::tempdir().unwrap();
    let receipt = dir.path().join("production-receipt.json");
    std::fs::write(&receipt, b"{}").unwrap();
    let cost = |cached: bool| ContextCost {
        cached,
        production_receipt: crate::execution::Artifact {
            path: receipt.clone(),
            sha256: sha256_hex(b"{}"),
        },
        original_wall_seconds: Some(1.),
        original_cpu_seconds: Some(1.),
        original_gpu_seconds: Some(0.),
        amortization_uses: None,
    };
    // Positive control: an uncached cost charged to its original arm.
    cost(false).validate().unwrap();
    let refused = cost(true).validate();
    assert!(
        matches!(&refused, Err(e) if e.to_string().contains("cached context cost")),
        "a cached context cost was admitted without an issuer: {refused:?}"
    );
}

struct FixtureBackend {
    time: u64,
    started: Rc<Cell<bool>>,
    deadlines: BTreeMap<String, u64>,
    cache: BTreeMap<String, Feedback>,
    journal: BTreeMap<String, serde_json::Value>,
    dispatched: Vec<Candidate>,
    unknown_counter: bool,
    admission_unavailable: bool,
    expire_during_admission: bool,
    cancelled: bool,
    interrupt_after_observed: bool,
}
impl Default for FixtureBackend {
    fn default() -> Self {
        Self {
            time: 1000,
            started: Rc::new(Cell::new(false)),
            deadlines: BTreeMap::new(),
            cache: BTreeMap::new(),
            journal: BTreeMap::new(),
            dispatched: vec![],
            unknown_counter: false,
            admission_unavailable: false,
            expire_during_admission: false,
            cancelled: false,
            interrupt_after_observed: false,
        }
    }
}
impl Backend for FixtureBackend {
    fn begin_arm(&mut self, arm: &ArmBudget) -> Result<u64> {
        self.started.set(true);
        Ok(*self
            .deadlines
            .entry(arm.id_sha256.clone())
            .or_insert(self.time + arm.seconds * 1000))
    }
    fn now_ms(&self) -> u64 {
        self.time
    }
    fn cancelled(&self) -> bool {
        self.cancelled
    }
    fn produce(&mut self, _job: WorkerJob) -> Result<WorkerOutcome> {
        anyhow::bail!("synthetic coordinator fixture never runs a producer")
    }
    fn evaluate(
        &mut self,
        arm: &ArmBudget,
        selection: &SelectionPlan,
        candidates: &[Candidate],
    ) -> Result<Vec<Feedback>> {
        if self.admission_unavailable {
            self.admission_unavailable = false;
            if self.expire_during_admission {
                self.time = self.deadlines[&arm.id_sha256];
            }
            return Ok(candidates
                .iter()
                .map(|c| {
                    self.cache
                        .get(&c.id_sha256)
                        .cloned()
                        .unwrap_or_else(|| Feedback {
                            candidate_sha256: c.id_sha256.clone(),
                            status: if self.cancelled {
                                EvaluationStatus::Cancelled
                            } else {
                                EvaluationStatus::Unavailable
                            },
                            loss: None,
                            physical_fit_starts: Some(0),
                            worker_receipt_sha256: None,
                            replay_evidence_sha256: None,
                            fitted_artifact_sha256: None,
                            reason: Some("synthetic missing dependency".into()),
                            oof: None,
                            retryable_inventory: true,
                        })
                })
                .collect());
        }
        candidates
            .iter()
            .map(|candidate| {
                if let Some(cached) = self.cache.get(&candidate.id_sha256) {
                    return Ok(cached.clone());
                }
                if self.time >= self.deadlines[&arm.id_sha256] {
                    return Ok(Feedback::exhausted(candidate));
                }
                self.dispatched.push(candidate.clone());
                // Synthetic callback evidence; never a policy/native fit claim.
                let oof = OofEvidence::for_candidate(
                    selection,
                    &candidate.program_sha256,
                    &candidate.id_sha256,
                    vec![1; 5],
                    vec![sha256_hex(candidate.genome.as_bytes()); 5],
                )?;
                let fitted_sha = oof.fitted_bundle_sha256()?;
                let feedback = Feedback {
                    candidate_sha256: candidate.id_sha256.clone(),
                    status: EvaluationStatus::Completed,
                    loss: Some(1. + candidate.initial_slot.unwrap_or(0) as f64 / 100.),
                    physical_fit_starts: if self.unknown_counter { None } else { Some(5) },
                    worker_receipt_sha256: Some(sha256_hex(b"synthetic callback receipt")),
                    replay_evidence_sha256: None,
                    fitted_artifact_sha256: Some(fitted_sha),
                    reason: None,
                    oof: Some(oof),
                    retryable_inventory: false,
                };
                self.cache
                    .insert(candidate.id_sha256.clone(), feedback.clone());
                Ok(feedback)
            })
            .collect()
    }
    fn persist(&mut self, arm: &ArmBudget, name: &str, value: &serde_json::Value) -> Result<()> {
        let key = format!("{}/{name}", arm.id_sha256);
        if let Some(old) = self.journal.get(&key) {
            ensure!(old == value, "fixture stale journal");
        } else {
            self.journal.insert(key, value.clone());
        }
        if self.interrupt_after_observed && name.ends_with("-observed") {
            self.interrupt_after_observed = false;
            anyhow::bail!("synthetic interruption after durable generation observation");
        }
        Ok(())
    }
    fn load(&self, arm: &ArmBudget, name: &str) -> Result<Option<serde_json::Value>> {
        Ok(self
            .journal
            .get(&format!("{}/{name}", arm.id_sha256))
            .cloned())
    }
}
struct FixtureVariation {
    started: Rc<Cell<bool>>,
}
impl Variation for FixtureVariation {
    fn identity(&self) -> &str {
        "synthetic variation/v1"
    }
    fn generic_policy(&self) -> &str {
        "synthetic uniform/v1"
    }
    fn generic(&mut self, seed: u64) -> Result<Option<String>> {
        ensure!(
            self.started.get(),
            "generation happened before original clock"
        );
        Ok(Some(format!(
            "(pipe X (ridge lambda={}))",
            1 + seed % 1_000_000_000
        )))
    }
    fn mutate(&mut self, _parent: &str, seed: u64) -> Result<String> {
        Ok(format!(
            "(pipe X (standardize) (ridge lambda={}))",
            1 + seed % 1_000_000_000
        ))
    }
    fn crossover(&mut self, _left: &str, _right: &str, seed: u64) -> Result<String> {
        Ok(format!(
            "(pipe X (robust_scale) (ridge lambda={}))",
            1 + seed % 1_000_000_000
        ))
    }
}
fn comparison(max_evaluations: usize) -> PairInputPlan {
    PairInputPlan {
        selection: crate::search_plan::fixture(),
        preprocessing_sha256: sha256_hex(b"synthetic preprocessing"),
        raw_input_policy: RAW_INPUT_POLICY.into(),
        fit_randomness_policy: FIT_RANDOMNESS_POLICY.into(),
        installation: InstallationPins {
            release_manifest_sha256: sha256_hex(b"synthetic release"),
            policy_program_sha256: sha256_hex(b"synthetic program"),
            source_manifest_sha256: sha256_hex(b"synthetic source"),
            checkpoint_receipt_sha256: sha256_hex(b"synthetic checkpoint"),
        },
        encoding: EncodingPlan {
            input_sha256: sha256_hex(b"synthetic encoding input"),
            source_sha40: "0".repeat(40),
            program_sha256: sha256_hex(b"synthetic encoder program"),
            source_manifest_sha256: sha256_hex(b"synthetic encoder source"),
            encoder_assets_manifest_sha256: sha256_hex(b"synthetic encoder assets"),
            qualification_sha256: None,
            reference_manifest_sha256: None,
            backend: "synthetic fixture".into(),
        },
        proposal: jope_policy::producer::RequestPlan {
            worker_source_sha256: sha256_hex(b"synthetic source"),
            prepared_sha256: sha256_hex(b"prepared"),
            training_sha256: sha256_hex(b"training"),
            cut: 0,
            checkpoint_sha256: jope_policy::ACCEPTED_STATE_SHA256.into(),
            policy_safetensors_sha256:
                "4b4b5a40f03c9226b4b15856fd35b90127c8b6a4d1435b815484cdd050744f1e".into(),
            vocabulary_sha256: jope_policy::PRODUCT_VOCABULARY_SHA256.into(),
            sampling_seed: 1,
            fallback_seed: 2,
            exploration_seed: 3,
            sampling_rounds: 1,
            sampling_batch: 8,
            sampling_temperature: 1.,
            challenger: None,
        }
        .into(),
        variation_identity: "synthetic variation/v1".into(),
        generic_sampling_policy: "synthetic uniform/v1".into(),
        checkpoint_training_exposure: "unknown".into(),
        exposure_evidence_sha256: None,
        prepared_sha256: sha256_hex(b"prepared"),
        training_sha256: sha256_hex(b"training"),
        cut: 0,
        vocabulary_sha256: jope_policy::PRODUCT_VOCABULARY_SHA256.into(),
        seconds: 120,
        max_evaluations,
        random_seed: 71,
        incumbents: None,
    }
}
fn synthetic_initial(arm: &ArmBudget) -> Vec<Candidate> {
    (0..INITIAL_SIZE)
        .map(|slot| {
            Candidate::new(
                arm,
                format!("(pipe X (ridge lambda={}))", slot + 1),
                0,
                slot,
                if slot < 60 {
                    Origin::NeuralPolicy
                } else if slot < 64 {
                    Origin::NeuralRejectionFallback
                } else {
                    Origin::RandomExploration
                },
                vec![],
                None,
            )
            .unwrap()
        })
        .collect()
}

#[test]
fn accepted_envelope_preserves_serialized_checkpoint_plan_and_evidence() -> Result<()> {
    let pair = comparison(80);
    let accepted = pair.proposal.accepted()?;
    let before = serde_json::to_vec(accepted)?;
    assert_eq!(serde_json::to_vec(&pair.proposal)?, before);
    let decoded: ProposalPlan = serde_json::from_slice(&before)?;
    assert_eq!(serde_json::to_vec(decoded.accepted()?)?, before);
    let evidence = AcceptedProposalEvidence {
        request: accepted.request(sha256_hex(b"context")),
        context_cost: ContextCost {
            cached: false,
            production_receipt: crate::execution::Artifact {
                path: "/private/fixture".into(),
                sha256: sha256_hex(b"receipt"),
            },
            original_wall_seconds: Some(1.),
            original_cpu_seconds: Some(1.),
            original_gpu_seconds: Some(0.),
            amortization_uses: Some(1),
        },
        producer_job_sha256: sha256_hex(b"job"),
        producer_output_sha256: sha256_hex(b"output"),
        population_sha256: sha256_hex(b"population"),
        context_sha256: sha256_hex(b"context"),
        checkpoint_sha256: accepted.checkpoint_sha256.clone(),
        ranking_backend: "fixture".into(),
        native_cuda_parity_qualified: false,
        proposal_ledger_sha256: None,
        proposal_draws: None,
    };
    let before = serde_json::to_vec(&evidence)?;
    // Evidence recorded before the ledger existed keeps its bytes; new evidence names it.
    assert!(!String::from_utf8(before.clone())?.contains("proposal_ledger_sha256"));
    assert!(!String::from_utf8(before.clone())?.contains("proposal_draws"));
    let wrapped = ProposalEvidence::Accepted(Box::new(evidence));
    assert_eq!(serde_json::to_vec(&wrapped)?, before);
    let decoded: ProposalEvidence = serde_json::from_slice(&before)?;
    assert_eq!(serde_json::to_vec(decoded.accepted()?)?, before);
    Ok(())
}
#[test]
fn initial_candidate_identity_rejects_slot_origin_and_genome_relabeling() -> Result<()> {
    let arm = comparison(80).arm(true)?;
    let candidate = synthetic_initial(&arm).remove(0);
    candidate.validate_initial(&arm)?;
    for changed in [
        {
            let mut c = candidate.clone();
            c.initial_slot = Some(1);
            c
        },
        {
            let mut c = candidate.clone();
            c.origin = Origin::Generic;
            c
        },
        {
            let mut c = candidate.clone();
            c.genome = "(pipe X (ridge lambda=99))".into();
            c
        },
    ] {
        assert!(changed.validate_initial(&arm).is_err());
    }
    let different = ArmBudget {
        id_sha256: sha256_hex(b"different"),
        seconds: arm.seconds,
    };
    assert!(candidate.validate_initial(&different).is_err());
    Ok(())
}
