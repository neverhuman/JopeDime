impl SearchReceipt {
    /// Validate retained population metadata in both live resume and read-only
    /// reports. This does not authenticate worker outputs or proposal origins;
    /// callers must still replay the pinned producer and completed OOF receipts.
    pub fn validate_population(&self, plan: &PairInputPlan) -> Result<()> {
        let neural_arm = plan.arm(true)?;
        let generic_arm = plan.arm(false)?;
        let neural = self.arm.id_sha256 == neural_arm.id_sha256;
        let seeds = plan.arm_incumbents(neural);
        let first_incumbent = INITIAL_SIZE - seeds.len();
        ensure!(
            self.schema == PROTOCOL
                && self.comparison_sha256 == sha256_hex(&serde_json::to_vec(plan)?)
                && self.selection_sha256 == plan.selection.identity()?
                && self.arm.seconds == plan.seconds
                && (neural || self.arm.id_sha256 == generic_arm.id_sha256),
            "search population binding changed"
        );
        ensure!(
            (INITIAL_SIZE..=plan.max_evaluations).contains(&self.observations.len()),
            "search population count changed"
        );
        let mut earlier = BTreeMap::<&str, &Observation>::new();
        let mut programs = BTreeSet::new();
        for (generation, group) in self
            .observations
            .chunk_by(|a, b| a.candidate.generation == b.candidate.generation)
            .enumerate()
        {
            ensure!(
                group[0].candidate.generation == generation
                    && if generation == 0 {
                        group.len() == INITIAL_SIZE
                    } else {
                        group.len() <= INITIAL_SIZE - ELITES
                    },
                "search generations or initial 80 slots changed"
            );
            for (slot, observation) in group.iter().enumerate() {
                let c = &observation.candidate;
                if generation == 0 {
                    let expected_origin = if slot >= first_incumbent {
                        c.origin == Origin::Incumbent
                            && c.genome == seeds[slot - first_incumbent].original_proposal
                    } else if !neural {
                        c.origin == Origin::Generic
                    } else if slot < NEURAL_SLOTS {
                        matches!(
                            c.origin,
                            Origin::NeuralPolicy | Origin::NeuralRejectionFallback
                        )
                    } else {
                        c.origin == Origin::RandomExploration
                    };
                    ensure!(
                        expected_origin && c.parents.is_empty(),
                        "initial origin/parents changed"
                    );
                    ensure!(
                        (neural && c.origin != Origin::Incumbent) || c.rank_log_weight.is_none(),
                        "generic or incumbent slot has policy rank"
                    );
                } else {
                    ensure!(
                        matches!(
                            (&c.origin, c.parents.len()),
                            (Origin::Mutation, 1) | (Origin::Crossover, 2)
                        ) && c.rank_log_weight.is_none(),
                        "offspring origin/parent count changed"
                    );
                    ensure!(
                        c.parents
                            .iter()
                            .all(|id| earlier.get(id.as_str()).is_some_and(|p| {
                                p.candidate.generation < generation
                                    && p.feedback.status == EvaluationStatus::Completed
                            })),
                        "offspring lacks an earlier completed parent"
                    );
                }
                let mut expected = Candidate::new(
                    &self.arm,
                    c.genome.clone(),
                    generation,
                    slot,
                    c.origin.clone(),
                    c.parents.clone(),
                    c.rank_log_weight,
                )?;
                c.check_proposal_record(neural && generation == 0 && slot < first_incumbent)?;
                expected
                    .proposal_record_sha256
                    .clone_from(&c.proposal_record_sha256);
                ensure!(
                    serde_json::to_value(c)? == serde_json::to_value(expected)?,
                    "candidate identity, recipe or slot changed"
                );
                ensure!(
                    observation.feedback.candidate_sha256 == c.id_sha256
                        && earlier.insert(c.id_sha256.as_str(), observation).is_none(),
                    "duplicate or mismatched candidate identity"
                );
                let unique = programs.insert(c.program_sha256.as_str());
                ensure!(
                    (neural && generation == 0) || unique,
                    "generic or offspring canonical recipe repeated"
                );
                if observation.feedback.status == EvaluationStatus::Completed {
                    let mut checked = observation.feedback.clone();
                    checked.enforce_selection(&plan.selection, c);
                    ensure!(
                        serde_json::to_value(checked)?
                            == serde_json::to_value(&observation.feedback)?,
                        "completed parent/candidate lacks exact OOF measurements"
                    );
                }
            }
        }
        Ok(())
    }

    /// Completed candidates keyed by id: (generation, on a completed path from a
    /// generation-zero NeuralPolicy root). Exploration, fallback and incumbent
    /// roots are never on such a path.
    fn neural_paths(&self) -> BTreeMap<&str, (usize, bool)> {
        let mut completed = BTreeMap::<&str, (usize, bool)>::new();
        for o in &self.observations {
            if o.feedback.status != EvaluationStatus::Completed {
                continue;
            }
            let c = &o.candidate;
            let neural_path = if c.generation == 0 {
                c.origin == Origin::NeuralPolicy
            } else {
                matches!(
                    (&c.origin, c.parents.len()),
                    (Origin::Mutation, 1) | (Origin::Crossover, 2)
                ) && c.parents.iter().all(|id| {
                    completed
                        .get(id.as_str())
                        .is_some_and(|(generation, _)| *generation < c.generation)
                }) && c.parents.iter().any(|id| {
                    completed
                        .get(id.as_str())
                        .is_some_and(|(_, neural)| *neural)
                })
            };
            completed.insert(c.id_sha256.as_str(), (c.generation, neural_path));
        }
        completed
    }
    /// After population and execution replay, prove an actual completed path
    /// from a generation-zero NeuralPolicy candidate to a completed offspring.
    /// Exploration and rejection fallback roots never establish neural ancestry.
    pub fn has_fitted_neural_descendant(&self) -> bool {
        self.neural_paths()
            .values()
            .any(|(generation, neural)| *generation > 0 && *neural)
    }
    /// Whether a completed candidate descends, through completed parents only,
    /// from a generation-zero NeuralPolicy root (or is such a root).
    pub fn neural_policy_lineage(&self, candidate_sha256: &str) -> bool {
        self.neural_paths()
            .get(candidate_sha256)
            .is_some_and(|(_, neural)| *neural)
    }
    pub fn winner(&self) -> Option<&Observation> {
        self.observations
            .iter()
            .filter(|o| {
                o.feedback.status == EvaluationStatus::Completed
                    && o.feedback.loss.is_some_and(f64::is_finite)
            })
            .min_by(|a, b| {
                a.feedback
                    .loss
                    .unwrap()
                    .total_cmp(&b.feedback.loss.unwrap())
                    .then_with(|| a.candidate.program_sha256.cmp(&b.candidate.program_sha256))
            })
    }
    pub fn neural_physical_fit_starts(&self) -> Option<u64> {
        self.observations
            .iter()
            .filter(|o| o.candidate.origin == Origin::NeuralPolicy)
            .try_fold(0u64, |sum, o| {
                sum.checked_add(o.feedback.physical_fit_starts?)
            })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AcceptedProposalEvidence {
    pub request: jope_policy::producer::Request,
    pub context_cost: ContextCost,
    pub producer_job_sha256: String,
    pub producer_output_sha256: String,
    pub population_sha256: String,
    pub context_sha256: String,
    pub checkpoint_sha256: String,
    pub ranking_backend: String,
    /// Accepted checkpoint bytes do not qualify this CPU implementation numerically.
    pub native_cuda_parity_qualified: bool,
    /// sha256 of the reconciled `jope.proposal-ledger/v1` in the producer artifact: every
    /// draw's terminal outcome and each seed slot's original tokens, scores, parser identity
    /// and adaptation edges. Absent in evidence recorded before the ledger existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proposal_ledger_sha256: Option<String>,
    /// W4-T03: every draw of that ledger with its terminal outcome (completed, surplus,
    /// duplicate, parser refusal, incomplete; fallback and exploration admissions and refusals),
    /// the slot each handed-off draw seeds, and the reconciled denominators. Absent in evidence
    /// recorded before the candidate ledger carried them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proposal_draws: Option<jope_policy::proposal_ledger::DrawOutcomes>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ProposalEvidence {
    Accepted(Box<AcceptedProposalEvidence>),
    #[cfg(feature = "qualified-context")]
    Experimental(Box<crate::experimental_producer::Evidence>),
}
impl ProposalEvidence {
    pub fn accepted(&self) -> Result<&AcceptedProposalEvidence> {
        match self {
            Self::Accepted(value) => Ok(value),
            #[cfg(feature = "qualified-context")]
            Self::Experimental(_) => {
                anyhow::bail!("experimental evidence cannot enter accepted receipt verifier")
            }
        }
    }
    pub fn checkpoint_sha256(&self) -> &str {
        match self {
            Self::Accepted(value) => &value.checkpoint_sha256,
            #[cfg(feature = "qualified-context")]
            Self::Experimental(value) => &value.checkpoint_sha256,
        }
    }
}

/// Narrow test seam. Production uses `PoolBackend` below. Implementations must
/// persist begin_arm once and enforce its original deadline on every child.
pub trait Backend {
    fn begin_arm(&mut self, arm: &ArmBudget) -> Result<u64>;
    fn now_ms(&self) -> u64;
    fn cancelled(&self) -> bool {
        false
    }
    fn evaluate(
        &mut self,
        arm: &ArmBudget,
        selection: &SelectionPlan,
        candidates: &[Candidate],
    ) -> Result<Vec<Feedback>>;
    fn produce(&mut self, job: WorkerJob) -> Result<WorkerOutcome>;
    fn persist(&mut self, arm: &ArmBudget, name: &str, value: &serde_json::Value) -> Result<()>;
    fn load(&self, arm: &ArmBudget, name: &str) -> Result<Option<serde_json::Value>>;
}

/// Grammar choice belongs to the task. Classification must supply classification
/// grammar here; the built-in `RegressionVariation` is deliberately explicit.
pub trait Variation {
    fn identity(&self) -> &str;
    fn generic_policy(&self) -> &str;
    fn generic(&mut self, seed: u64) -> Result<Option<String>>;
    fn mutate(&mut self, parent: &str, seed: u64) -> Result<String>;
    fn crossover(&mut self, left: &str, right: &str, seed: u64) -> Result<String>;
}
pub struct RegressionVariation<'a> {
    pub vocabulary: &'a jope_policy::grammar::BoundVocabulary,
}
impl Variation for RegressionVariation<'_> {
    fn identity(&self) -> &str {
        "feat-math.canonical-mutate-crossover/max-transforms4/v1"
    }
    fn generic_policy(&self) -> &str {
        "jope-policy.grammar-uniform-token-chacha8/v1"
    }
    fn generic(&mut self, seed: u64) -> Result<Option<String>> {
        jope_policy::producer::generic_program(self.vocabulary, "regression", seed)
    }
    fn mutate(&mut self, parent: &str, seed: u64) -> Result<String> {
        evolve::mutate_canonical_pipeline(parent, seed, 4).map_err(anyhow::Error::msg)
    }
    fn crossover(&mut self, left: &str, right: &str, seed: u64) -> Result<String> {
        evolve::crossover_canonical_pipelines(left, right, seed, 4).map_err(anyhow::Error::msg)
    }
}

/// A release-pinned policy command. Pins come from the verified installation
/// manifest, not the produced file. The child loads verified native context,
/// calls `producer::produce` with bound model, checkpoint and vocabulary. Context and
/// proposal work inside that command are subject to the same process/cgroup
/// gate, fit cap (normally30s) and original120s arm deadline as any worker.
/// Existing context files may be reused; this API never relabels an old encoder.
#[derive(Serialize, Deserialize)]
pub struct PolicyWorker {
    pub job: WorkerJob,
    pub expected_program_sha256: String,
    pub release_manifest: crate::execution::Artifact,
    pub source_manifest: crate::execution::Artifact,
    pub request_artifact: crate::execution::Artifact,
    pub checkpoint_receipt: crate::execution::Artifact,
    pub request: jope_policy::producer::Request,
    pub context: super::search_context::ContextFiles,
    pub output_name: PathBuf,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct NeuralRun {
    pub proposal_execution: WorkerOutcome,
    /// None means proposal failed/timed out/unavailable, not a generic substitute.
    pub search: Option<SearchReceipt>,
}

/// One immutable pair, fixed neural-then-generic execution order, same fit and
/// search budgets, same variation implementation and generic token policy.
/// Errors are retained as inventory; they are never converted to wins or losses.
#[derive(Debug, Serialize, Deserialize)]
pub struct PairedReceipt {
    pub comparison_sha256: String,
    pub neural: Option<NeuralRun>,
    pub neural_error: Option<String>,
    pub generic: Option<SearchReceipt>,
    pub generic_error: Option<String>,
}
fn freeze_pair<B: Backend, V: Variation>(
    backend: &mut B,
    comparison: &PairInputPlan,
    variation: &V,
) -> Result<()> {
    comparison.validate()?;
    ensure!(
        variation.identity() == comparison.variation_identity
            && variation.generic_policy() == comparison.generic_sampling_policy,
        "variation/generic policy differs from frozen pair"
    );
    let value = serde_json::json!({"schema":PROTOCOL,"comparison":comparison,"order":["neural","generic"],"seed_policy":"asm.candidate-request-seed/v1","selection":"five training-side OOF folds","refit":"freeze selection winner before development hold scoring"});
    let slot = ArmBudget {
        id_sha256: sha256_hex(&serde_json::to_vec(&(
            "asm.paired-slot/v1",
            comparison.selection.outer(),
        ))?),
        seconds: comparison.seconds,
    };
    backend.persist(&slot, "paired-registration", &value)?;
    for neural in [true, false] {
        backend.persist(&comparison.arm(neural)?, "paired-plan", &value)?;
    }
    Ok(())
}
/// Call before fresh encoding. Repeating it returns the original persisted
/// deadline. A cached context still retains its original production-cost receipt.
pub fn begin_pair<B: Backend, V: Variation>(
    backend: &mut B,
    plan: &PairInputPlan,
    variation: &V,
) -> Result<(ArmBudget, u64)> {
    freeze_pair(backend, plan, variation)?;
    let arm = plan.arm(true)?;
    let deadline = backend.begin_arm(&arm)?;
    Ok((arm, deadline))
}
