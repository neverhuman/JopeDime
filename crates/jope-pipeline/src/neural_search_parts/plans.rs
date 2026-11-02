/// Accepted serialization is unchanged. The experimental variant has a distinct
/// schema and cannot be passed to the accepted checkpoint/context loader.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ProposalPlan {
    Accepted(Box<jope_policy::producer::RequestPlan>),
    #[cfg(feature = "qualified-context")]
    Experimental(Box<crate::experimental_producer::RequestPlan>),
}
impl ProposalPlan {
    pub fn accepted(&self) -> Result<&jope_policy::producer::RequestPlan> {
        match self {
            Self::Accepted(value) => Ok(value),
            #[cfg(feature = "qualified-context")]
            Self::Experimental(_) => {
                anyhow::bail!("experimental proposal cannot enter accepted policy loader")
            }
        }
    }
    pub fn checkpoint_sha256(&self) -> &str {
        match self {
            Self::Accepted(value) => &value.checkpoint_sha256,
            #[cfg(feature = "qualified-context")]
            Self::Experimental(value) => &value.checkpoint.sha256,
        }
    }
    fn validate_pair(&self, pair: &PairInputPlan) -> Result<()> {
        match self {
            Self::Accepted(value) => {
                value.validate()?;
                ensure!(
                    value.prepared_sha256 == pair.prepared_sha256
                        && value.training_sha256 == pair.training_sha256
                        && value.cut == pair.cut
                        && value.vocabulary_sha256 == pair.vocabulary_sha256
                        && value.worker_source_sha256 == pair.installation.source_manifest_sha256,
                    "accepted producer differs from pair plan"
                );
            }
            #[cfg(feature = "qualified-context")]
            Self::Experimental(value) => {
                value.validate()?;
                ensure!(
                    value.task == pair.selection.outer().task
                        && value.training.prepared_sha256 == pair.prepared_sha256
                        && value.training.training_sha256 == pair.training_sha256
                        && value.training.cut == pair.cut
                        && value.training.selected_training_rows_sha256
                            == pair.selection.outer().fitting_rows_sha256
                        && value.training.preprocessing_sha256 == pair.preprocessing_sha256
                        && value.vocabulary.sha256 == pair.vocabulary_sha256
                        && value.runtime.program.sha256 == pair.installation.policy_program_sha256
                        && value.export_manifest.sha256
                            == pair.installation.checkpoint_receipt_sha256
                        && value.worker_source_sha256 == pair.installation.source_manifest_sha256
                        && value.generic_policy == pair.generic_sampling_policy,
                    "experimental producer differs from pair task/rows/installation"
                );
            }
        }
        Ok(())
    }
}
impl From<jope_policy::producer::RequestPlan> for ProposalPlan {
    fn from(value: jope_policy::producer::RequestPlan) -> Self {
        Self::Accepted(Box::new(value))
    }
}

/// The controller supplies the full comparison identity (task, dataset, cut,
/// evaluator, vocabulary, preprocessing and protocol), not just a dataset name.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PairInputPlan {
    pub selection: SelectionPlan,
    pub preprocessing_sha256: String,
    pub raw_input_policy: String,
    pub fit_randomness_policy: String,
    pub installation: InstallationPins,
    pub proposal: ProposalPlan,
    pub encoding: EncodingPlan,
    pub variation_identity: String,
    pub generic_sampling_policy: String,
    pub checkpoint_training_exposure: String,
    pub exposure_evidence_sha256: Option<String>,
    pub prepared_sha256: String,
    pub training_sha256: String,
    pub cut: usize,
    pub vocabulary_sha256: String,
    pub seconds: u64,
    pub max_evaluations: usize,
    pub random_seed: u64,
    /// Earlier arm winners seeded into generation zero of their own arm. Omitted
    /// when absent, so historical arm identities and comparison digests are unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub incumbents: Option<ArmIncumbents>,
}
impl PairInputPlan {
    fn validate(&self) -> Result<()> {
        for digest in [
            &self.preprocessing_sha256,
            &self.prepared_sha256,
            &self.training_sha256,
            &self.vocabulary_sha256,
        ] {
            ensure!(
                jope_policy::is_sha256_hex(digest),
                "invalid comparison digest"
            );
        }
        ensure!(
            self.raw_input_policy == RAW_INPUT_POLICY
                && self.fit_randomness_policy == FIT_RANDOMNESS_POLICY,
            "unsupported input/randomness policy"
        );
        self.selection.validate()?;
        ensure!(
            self.cut == self.selection.outer().fold as usize,
            "context cut differs from frozen outer fold"
        );
        self.installation.validate()?;
        self.encoding.validate()?;
        self.proposal.validate_pair(self)?;
        ensure!(
            !self.variation_identity.is_empty() && !self.generic_sampling_policy.is_empty(),
            "missing frozen variation/generic policy"
        );
        ensure!(
            ["known_seen", "known_unseen", "unknown"]
                .contains(&self.checkpoint_training_exposure.as_str()),
            "unknown exposure status"
        );
        ensure!(
            self.exposure_evidence_sha256
                .as_ref()
                .is_none_or(|d| jope_policy::is_sha256_hex(d))
                && (self.checkpoint_training_exposure == "unknown"
                    || self.exposure_evidence_sha256.is_some()),
            "known exposure needs evidence"
        );
        ensure!(
            (1..=120).contains(&self.seconds),
            "search arm must be at most 120 seconds"
        );
        ensure!(
            (INITIAL_SIZE..=1000).contains(&self.max_evaluations),
            "search must admit 80 initial slots and at most 1000 evaluations"
        );
        if let Some(incumbents) = &self.incumbents {
            // Incumbents come from the accepted parent's teaching ledger; a native challenger
            // comparison has none.
            ensure!(
                self.proposal
                    .accepted()
                    .is_ok_and(|p| p.challenger.is_none()),
                "incumbent seeding requires the accepted proposal plan"
            );
            ensure!(
                !(incumbents.neural.is_empty() && incumbents.generic.is_empty())
                    && incumbents.neural.len() <= MAX_INCUMBENTS
                    && incumbents.generic.len() <= MAX_INCUMBENTS,
                "incumbent count outside approved bounds"
            );
            for seeds in [&incumbents.neural, &incumbents.generic] {
                let mut programs = BTreeSet::new();
                for seed in seeds {
                    seed.validate()?;
                    ensure!(
                        programs.insert(&seed.program_sha256),
                        "incumbent recipe repeated within an arm"
                    );
                }
            }
        }
        Ok(())
    }
    pub fn arm_incumbents(&self, neural: bool) -> &[IncumbentSeed] {
        match &self.incumbents {
            Some(seeds) if neural => &seeds.neural,
            Some(seeds) => &seeds.generic,
            None => &[],
        }
    }
    pub fn arm(&self, neural: bool) -> Result<ArmBudget> {
        self.validate()?;
        Ok(ArmBudget {
            id_sha256: sha256_hex(&serde_json::to_vec(&(
                PROTOCOL,
                self,
                if neural { "neural" } else { "generic" },
            ))?),
            seconds: self.seconds,
        })
    }
}

/// Encoder inputs are frozen before the neural arm starts. Output digests are
/// absent here because a fresh, bounded encoding has not produced them yet.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EncodingPlan {
    pub input_sha256: String,
    pub source_sha40: String,
    pub program_sha256: String,
    pub source_manifest_sha256: String,
    pub encoder_assets_manifest_sha256: String,
    pub qualification_sha256: Option<String>,
    pub reference_manifest_sha256: Option<String>,
    pub backend: String,
}
impl EncodingPlan {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.source_sha40.len() == 40
                && self.source_sha40.bytes().all(|c| c.is_ascii_hexdigit()),
            "invalid encoder source identity"
        );
        ensure!(
            self.reference_manifest_sha256
                .as_ref()
                .is_none_or(|s| jope_policy::is_sha256_hex(s)),
            "invalid reference manifest identity"
        );
        for digest in [
            &self.input_sha256,
            &self.program_sha256,
            &self.source_manifest_sha256,
            &self.encoder_assets_manifest_sha256,
        ] {
            ensure!(
                jope_policy::is_sha256_hex(digest),
                "invalid encoding input pin"
            );
        }
        ensure!(
            !self.backend.is_empty()
                && self
                    .qualification_sha256
                    .as_ref()
                    .is_none_or(|d| jope_policy::is_sha256_hex(d)),
            "invalid encoding backend/qualification pin"
        );
        Ok(())
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContextCost {
    pub cached: bool,
    pub production_receipt: crate::execution::Artifact,
    pub original_wall_seconds: Option<f64>,
    pub original_cpu_seconds: Option<f64>,
    pub original_gpu_seconds: Option<f64>,
    /// None means no measured reuse count, not zero production cost.
    pub amortization_uses: Option<u64>,
}
impl ContextCost {
    pub(crate) fn validate(&self) -> Result<()> {
        self.production_receipt.verify()?;
        // W2-T04 G-11e: every producer charges its context to the original
        // arm (`cached: false`); a cached cost has no issuer and would relax
        // the arm-charge check of the qualified CPU context.
        ensure!(
            !self.cached,
            "a cached context cost has no issuer: the context must be charged to its original arm"
        );
        ensure!(
            [
                self.original_wall_seconds,
                self.original_cpu_seconds,
                self.original_gpu_seconds
            ]
            .into_iter()
            .flatten()
            .all(|v| v.is_finite() && v >= 0.),
            "invalid original encoding cost"
        );
        ensure!(
            self.amortization_uses.is_none_or(|n| n > 0),
            "invalid encoding reuse count"
        );
        Ok(())
    }
}
/// Outputs attach to a previously registered input plan. Dereferencing exposes
/// that immutable input plan; arm identity never depends on these output hashes.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Comparison {
    pub input_plan: PairInputPlan,
    pub producer_request: jope_policy::producer::Request,
    pub context_receipt_sha256: String,
    pub context_cost: ContextCost,
}
impl std::ops::Deref for Comparison {
    type Target = PairInputPlan;
    fn deref(&self) -> &Self::Target {
        &self.input_plan
    }
}
impl Comparison {
    fn validate(&self) -> Result<()> {
        self.input_plan.validate()?;
        self.producer_request.validate()?;
        // Only the encoding outputs (context digest and, for a qualified CPU context, the
        // controller-issued admission digest) are attached to the preregistered plan.
        ensure!(
            self.producer_request
                == self.proposal.accepted()?.request_with_admission(
                    self.producer_request.context_sha256.clone(),
                    self.producer_request.context_admission_sha256.clone()
                )
                && jope_policy::is_sha256_hex(&self.context_receipt_sha256),
            "context outputs changed preregistered producer inputs"
        );
        self.context_cost.validate()
    }
}

/// Installation provenance is frozen before either arm. The controller verifies
/// the private release manifest; producer output cannot choose these pins.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InstallationPins {
    pub release_manifest_sha256: String,
    pub policy_program_sha256: String,
    pub source_manifest_sha256: String,
    pub checkpoint_receipt_sha256: String,
}
impl InstallationPins {
    fn validate(&self) -> Result<()> {
        for digest in [
            &self.release_manifest_sha256,
            &self.policy_program_sha256,
            &self.source_manifest_sha256,
            &self.checkpoint_receipt_sha256,
        ] {
            ensure!(
                jope_policy::is_sha256_hex(digest),
                "invalid frozen installation pin"
            );
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CandidateOrigin {
    NeuralPolicy,
    NeuralRejectionFallback,
    RandomExploration,
    Generic,
    Mutation,
    Crossover,
    /// A prior round's arm winner, measured again in this arm. Never a new proposal.
    Incumbent,
}

/// Compatibility name for a candidate's search provenance.
pub type Origin = CandidateOrigin;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct IncumbentSeed {
    /// Content address of the retained FrozenRecipe in the teaching state.
    pub frozen_recipe_sha256: String,
    pub original_proposal: String,
    pub program_sha256: String,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ArmIncumbents {
    pub neural: Vec<IncumbentSeed>,
    pub generic: Vec<IncumbentSeed>,
}
