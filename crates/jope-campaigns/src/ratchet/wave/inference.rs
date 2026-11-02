use super::*;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct SeedInferenceRequestV2 {
    pub(crate) schema: String,
    pub(crate) learning_wave_sha256: String,
    pub(crate) dataset_encoding_sha256: String,
    pub(crate) checkpoint_sha256: String,
    pub(crate) corrector_sha256: String,
    pub(crate) registry_sha256: String,
    pub(crate) vocabulary_sha256: String,
    pub(crate) state_snapshot_sha256: String,
    pub(crate) draw_seed: u64,
    pub(crate) required_count: usize,
}

impl SeedInferenceRequestV2 {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.schema != SEED_REQUEST_V2_SCHEMA || self.required_count != MODEL_PROPOSAL_COUNT {
            bail!("promoted seed request is not the required v2/96 contract");
        }
        validate_hashes([
            ("seed learning wave", self.learning_wave_sha256.as_str()),
            (
                "seed dataset encoding",
                self.dataset_encoding_sha256.as_str(),
            ),
            ("seed checkpoint", self.checkpoint_sha256.as_str()),
            ("seed corrector", self.corrector_sha256.as_str()),
            ("seed registry", self.registry_sha256.as_str()),
            ("seed vocabulary", self.vocabulary_sha256.as_str()),
            ("seed state snapshot", self.state_snapshot_sha256.as_str()),
        ])
    }

    pub(crate) fn request_sha256(&self) -> Result<String> {
        self.validate()?;
        Ok(sha256_bytes(&canonical_json(self)?))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub(crate) struct ModelProposalV2 {
    pub(crate) candidate: CandidateIdentityV1,
    pub(crate) reward_prediction: f64,
    pub(crate) disagreement_prediction: f64,
    pub(crate) novelty_prediction: f64,
    pub(crate) temperature_logit: f64,
}

impl ModelProposalV2 {
    pub(in crate::ratchet) fn validate(&self) -> Result<()> {
        self.candidate.validate()?;
        if [
            self.reward_prediction,
            self.disagreement_prediction,
            self.novelty_prediction,
            self.temperature_logit,
        ]
        .iter()
        .any(|value| !value.is_finite())
        {
            bail!("model proposal contains a non-finite prediction");
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub(crate) struct SeedInferenceResponseV2 {
    pub(crate) schema: String,
    pub(crate) request_sha256: String,
    pub(crate) proposals: Vec<ModelProposalV2>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ModelRetentionBucketV2 {
    Reward,
    Disagreement,
    Novelty,
    SeededTemperature,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub(crate) struct RetainedModelProposalV2 {
    pub(crate) proposal: ModelProposalV2,
    pub(crate) bucket: ModelRetentionBucketV2,
    pub(crate) bucket_rank: usize,
}

impl SeedInferenceResponseV2 {
    pub(crate) fn validate_against(&self, request: &SeedInferenceRequestV2) -> Result<()> {
        request.validate()?;
        if self.schema != SEED_RESPONSE_V2_SCHEMA
            || self.request_sha256 != request.request_sha256()?
            || self.proposals.len() != MODEL_PROPOSAL_COUNT
        {
            bail!("Python sidecar response does not bind the v2 request or all 96 proposals");
        }
        validate_sha256("seed request receipt", &self.request_sha256)?;
        for proposal in &self.proposals {
            proposal.validate()?;
        }
        require_unique(
            "model proposals",
            self.proposals
                .iter()
                .map(|proposal| proposal.candidate.semantic_sha256.as_str()),
        )
    }

    pub(crate) fn retain_64(
        &self,
        request: &SeedInferenceRequestV2,
        excluded_semantics: &BTreeSet<String>,
    ) -> Result<Vec<RetainedModelProposalV2>> {
        self.validate_against(request)?;
        let available = self
            .proposals
            .iter()
            .enumerate()
            .filter(|(_, proposal)| {
                !excluded_semantics.contains(&proposal.candidate.semantic_sha256)
            })
            .map(|(index, proposal)| (index, proposal.clone()))
            .collect::<Vec<_>>();
        if available.len() < MODEL_RETAINED_COUNT {
            bail!("fewer than 64 unique valid non-archive model proposals remain");
        }

        let mut selected = BTreeSet::<usize>::new();
        let mut retained = Vec::with_capacity(MODEL_RETAINED_COUNT);
        select_ranked(
            &available,
            &mut selected,
            &mut retained,
            32,
            ModelRetentionBucketV2::Reward,
            |left, right| descending_f64(left.reward_prediction, right.reward_prediction),
        );
        select_ranked(
            &available,
            &mut selected,
            &mut retained,
            16,
            ModelRetentionBucketV2::Disagreement,
            |left, right| {
                descending_f64(left.disagreement_prediction, right.disagreement_prediction)
            },
        );
        select_ranked(
            &available,
            &mut selected,
            &mut retained,
            8,
            ModelRetentionBucketV2::Novelty,
            |left, right| descending_f64(left.novelty_prediction, right.novelty_prediction),
        );

        let mut temperature = available
            .iter()
            .filter(|(index, _)| !selected.contains(index))
            .map(|(index, proposal)| {
                (
                    *index,
                    proposal.clone(),
                    seeded_temperature_score(
                        request.draw_seed,
                        &proposal.candidate.semantic_sha256,
                        proposal.temperature_logit,
                    ),
                )
            })
            .collect::<Vec<_>>();
        temperature.sort_by(|left, right| {
            descending_f64(left.2, right.2).then_with(|| {
                left.1
                    .candidate
                    .semantic_sha256
                    .cmp(&right.1.candidate.semantic_sha256)
            })
        });
        for (rank, (index, proposal, _)) in temperature.into_iter().take(8).enumerate() {
            selected.insert(index);
            retained.push(RetainedModelProposalV2 {
                proposal,
                bucket: ModelRetentionBucketV2::SeededTemperature,
                bucket_rank: rank,
            });
        }
        if retained.len() != MODEL_RETAINED_COUNT {
            bail!("model retention subpool exhausted before exact 32/16/8/8 quotas");
        }
        Ok(retained)
    }
}

pub(super) fn descending_f64(left: f64, right: f64) -> Ordering {
    right.total_cmp(&left)
}

pub(super) fn select_ranked(
    available: &[(usize, ModelProposalV2)],
    selected: &mut BTreeSet<usize>,
    retained: &mut Vec<RetainedModelProposalV2>,
    count: usize,
    bucket: ModelRetentionBucketV2,
    comparison: impl Fn(&ModelProposalV2, &ModelProposalV2) -> Ordering,
) {
    let mut ranked = available
        .iter()
        .filter(|(index, _)| !selected.contains(index))
        .cloned()
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        comparison(&left.1, &right.1).then_with(|| {
            left.1
                .candidate
                .semantic_sha256
                .cmp(&right.1.candidate.semantic_sha256)
        })
    });
    for (rank, (index, proposal)) in ranked.into_iter().take(count).enumerate() {
        selected.insert(index);
        retained.push(RetainedModelProposalV2 {
            proposal,
            bucket: bucket.clone(),
            bucket_rank: rank,
        });
    }
}

pub(super) fn seeded_temperature_score(seed: u64, semantic_sha256: &str, logit: f64) -> f64 {
    let mut hash = Sha256::new();
    hash.update(b"jope-dime/seeded-temperature/v1\0");
    hash.update(seed.to_le_bytes());
    hash.update(semantic_sha256.as_bytes());
    let digest = hash.finalize();
    let raw = u64::from_le_bytes(digest[..8].try_into().expect("SHA-256 prefix"));
    let uniform = ((raw as f64) + 1.0) / ((u64::MAX as f64) + 2.0);
    logit - (-uniform.ln()).ln()
}

/// Read compatibility envelope. V1 remains deserializable for old receipts,
/// but callers must use `SeedInferenceResponseV2` for promoted authority.
#[derive(Clone, Debug, Deserialize)]
pub(crate) struct LegacySeedInferenceV1 {
    // Serde consumes this field when validating the v1 receipt shape.
    #[allow(
        dead_code,
        reason = "serde compatibility carrier: typed deserialization in validate_artifact_file enforces the schema; payload intentionally unread"
    )]
    pub(crate) schema: String,
    // Preserve the complete v1 payload accepted by the compatibility reader.
    #[allow(
        dead_code,
        reason = "serde compatibility carrier: typed deserialization in validate_artifact_file enforces the schema; payload intentionally unread"
    )]
    #[serde(flatten)]
    pub(crate) fields: BTreeMap<String, Value>,
}

pub(crate) fn read_seed_inference(bytes: &[u8]) -> Result<ReadableSeedInference> {
    let value: Value = serde_json::from_slice(bytes)?;
    match value.get("schema").and_then(Value::as_str) {
        Some(SEED_RESPONSE_V2_SCHEMA) => {
            Ok(ReadableSeedInference::V2(serde_json::from_value(value)?))
        }
        Some(schema) if schema.starts_with("SeedInference") && schema.ends_with(".v1") => {
            Ok(ReadableSeedInference::V1(serde_json::from_value(value)?))
        }
        _ => bail!("unknown seed inference receipt schema"),
    }
}

#[derive(Clone, Debug)]
pub(crate) enum ReadableSeedInference {
    // Retain the typed payload used by the compatibility artifact validator.
    V1(
        #[allow(
            dead_code,
            reason = "serde compatibility carrier: typed deserialization in validate_artifact_file enforces the schema; payload intentionally unread"
        )]
        LegacySeedInferenceV1,
    ),
    V2(SeedInferenceResponseV2),
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub(crate) struct ModelRetentionReceiptV2 {
    pub(crate) schema: String,
    pub(crate) request_sha256: String,
    pub(crate) response_sha256: String,
    pub(crate) excluded_semantics_sha256: String,
    pub(crate) retained: Vec<RetainedModelProposalV2>,
}

impl ModelRetentionReceiptV2 {
    pub(crate) fn build(
        request: &SeedInferenceRequestV2,
        response: &SeedInferenceResponseV2,
        excluded_semantics: &BTreeSet<String>,
    ) -> Result<Self> {
        for semantic in excluded_semantics {
            validate_sha256("model-retention exclusion", semantic)?;
        }
        let retained = response.retain_64(request, excluded_semantics)?;
        let response_sha256 = sha256_bytes(&canonical_json(response)?);
        let excluded_semantics_sha256 = sha256_bytes(&canonical_json(excluded_semantics)?);
        let receipt = Self {
            schema: "ModelRetentionReceipt.v2".into(),
            request_sha256: request.request_sha256()?,
            response_sha256,
            excluded_semantics_sha256,
            retained,
        };
        receipt.validate(request, response, excluded_semantics)?;
        Ok(receipt)
    }

    pub(crate) fn validate(
        &self,
        request: &SeedInferenceRequestV2,
        response: &SeedInferenceResponseV2,
        excluded_semantics: &BTreeSet<String>,
    ) -> Result<()> {
        if self.schema != "ModelRetentionReceipt.v2"
            || self.request_sha256 != request.request_sha256()?
            || self.response_sha256 != sha256_bytes(&canonical_json(response)?)
            || self.excluded_semantics_sha256 != sha256_bytes(&canonical_json(excluded_semantics)?)
            || self.retained != response.retain_64(request, excluded_semantics)?
        {
            bail!("model-retention receipt does not reconstruct from Rust authority");
        }
        Ok(())
    }
}
