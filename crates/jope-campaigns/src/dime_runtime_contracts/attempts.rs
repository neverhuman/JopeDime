use super::*;

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DimeRoleV1 {
    CohortProposer,
    AlternatingTrainer,
    BreadthEncoder,
    BudgetCampaign,
}

impl DimeRoleV1 {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::CohortProposer => "cohort_proposer",
            Self::AlternatingTrainer => "alternating_trainer",
            Self::BreadthEncoder => "breadth_encoder",
            Self::BudgetCampaign => "budget_campaign",
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DimeNamedDigestV1 {
    pub name: String,
    pub sha256: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DimeScientificPayloadV1 {
    pub parent_intent_sha256: String,
    pub runtime_bundle_sha256: String,
    pub binary_sha256: String,
    pub configuration_sha256: String,
    pub train_data_sha256: String,
    pub protocol_sha256: String,
    pub metric_sha256: String,
    pub seed: u64,
    pub inputs: Vec<DimeNamedDigestV1>,
}

impl DimeScientificPayloadV1 {
    pub fn validate(&self) -> Result<()> {
        for (field, digest) in [
            ("parent intent", &self.parent_intent_sha256),
            ("runtime bundle", &self.runtime_bundle_sha256),
            ("binary", &self.binary_sha256),
            ("configuration", &self.configuration_sha256),
            ("train data", &self.train_data_sha256),
            ("protocol", &self.protocol_sha256),
            ("metric", &self.metric_sha256),
        ] {
            require_sha256(digest, field)?;
        }
        ensure!(!self.inputs.is_empty(), "scientific payload has no inputs");
        let mut prior: Option<&str> = None;
        for input in &self.inputs {
            ensure!(
                !input.name.trim().is_empty(),
                "scientific input name is empty"
            );
            require_sha256(&input.sha256, "scientific input")?;
            if let Some(previous) = prior {
                ensure!(
                    previous < input.name.as_str(),
                    "scientific inputs are not sorted and unique"
                );
            }
            prior = Some(&input.name);
        }
        Ok(())
    }

    pub fn identity_sha256(&self) -> Result<String> {
        self.validate()?;
        canonical_sha256(self)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DimeTerminalClassV1 {
    InfrastructureFailure,
    ScientificFailure,
    IntentionalDrain,
    Completed,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DimeTerminalEvidenceV1 {
    pub terminal_state: String,
    pub class: DimeTerminalClassV1,
    pub evidence_sha256: String,
    pub observed_at_ms: u128,
    pub foreign_cancellation: bool,
}

impl DimeTerminalEvidenceV1 {
    fn validate(&self) -> Result<()> {
        ensure!(
            !self.terminal_state.trim().is_empty() && self.observed_at_ms > 0,
            "terminal evidence is incomplete"
        );
        require_sha256(&self.evidence_sha256, "terminal evidence")?;
        ensure!(
            !self.foreign_cancellation
                || (self.class == DimeTerminalClassV1::InfrastructureFailure
                    && self.terminal_state == "cancelled"),
            "foreign cancellation classification is inconsistent"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DimeInfrastructureAttemptV1 {
    pub schema: String,
    pub role: DimeRoleV1,
    pub scientific_payload: DimeScientificPayloadV1,
    pub scientific_payload_sha256: String,
    pub attempt_number: u64,
    pub job_identity: String,
    pub next_job_identity: Option<String>,
    pub prior_attempt_receipt_sha256: Option<String>,
    pub created_at_ms: u128,
    pub retry_not_before_ms: u128,
    pub terminal: DimeTerminalEvidenceV1,
    pub content_sha256: String,
}

impl DimeInfrastructureAttemptV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new_terminal(
        role: DimeRoleV1,
        scientific_payload: DimeScientificPayloadV1,
        attempt_number: u64,
        job_identity: String,
        next_job_identity: Option<String>,
        prior_attempt_receipt_sha256: Option<String>,
        created_at_ms: u128,
        retry_not_before_ms: u128,
        terminal: DimeTerminalEvidenceV1,
    ) -> Result<Self> {
        let scientific_payload_sha256 = scientific_payload.identity_sha256()?;
        let mut value = Self {
            schema: DIME_INFRASTRUCTURE_ATTEMPT_SCHEMA.to_owned(),
            role,
            scientific_payload,
            scientific_payload_sha256,
            attempt_number,
            job_identity,
            next_job_identity,
            prior_attempt_receipt_sha256,
            created_at_ms,
            retry_not_before_ms,
            terminal,
            content_sha256: String::new(),
        };
        value.content_sha256 = value.computed_content_sha256()?;
        value.validate()?;
        Ok(value)
    }

    fn computed_content_sha256(&self) -> Result<String> {
        let mut copy = self.clone();
        copy.content_sha256.clear();
        canonical_sha256(&copy)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == DIME_INFRASTRUCTURE_ATTEMPT_SCHEMA,
            "infrastructure attempt schema is not v1"
        );
        self.scientific_payload.validate()?;
        ensure!(
            self.scientific_payload.identity_sha256()? == self.scientific_payload_sha256,
            "infrastructure attempt scientific payload hash drift"
        );
        ensure!(
            self.attempt_number > 0
                && !self.job_identity.trim().is_empty()
                && self.created_at_ms > 0
                && self.retry_not_before_ms >= self.created_at_ms,
            "infrastructure attempt identity or time is invalid"
        );
        if let Some(next) = &self.next_job_identity {
            ensure!(
                !next.trim().is_empty() && next != &self.job_identity,
                "recovery did not allocate a fresh job identity"
            );
        }
        if let Some(prior) = &self.prior_attempt_receipt_sha256 {
            require_sha256(prior, "prior attempt receipt")?;
        }
        self.terminal.validate()?;
        require_sha256(&self.content_sha256, "attempt content")?;
        ensure!(
            self.computed_content_sha256()? == self.content_sha256,
            "infrastructure attempt content hash drift"
        );
        Ok(())
    }
}

fn attempt_directory(root: &Path, receipt: &DimeInfrastructureAttemptV1) -> PathBuf {
    root.join("ops/dime-infrastructure-attempts")
        .join(receipt.role.as_str())
        .join(&receipt.scientific_payload_sha256)
}

/// Atomically append one terminal attempt receipt. Replaying an identical
/// receipt is idempotent; reusing an attempt number with different bytes or
/// skipping a number is rejected.
pub fn append_infrastructure_attempt(
    root: &Path,
    receipt: &DimeInfrastructureAttemptV1,
) -> Result<PathBuf> {
    receipt.validate()?;
    let directory = attempt_directory(root, receipt);
    fs::create_dir_all(&directory)?;
    let prior =
        read_attempt_directory(&directory, receipt.role, &receipt.scientific_payload_sha256)?;
    if let Some((existing, path)) = prior
        .iter()
        .find(|(value, _)| value.attempt_number == receipt.attempt_number)
    {
        ensure!(
            existing == receipt,
            "attempt replay differs from its receipt"
        );
        return Ok(path.clone());
    }
    let expected = prior
        .last()
        .map_or(1, |(value, _)| value.attempt_number + 1);
    ensure!(
        receipt.attempt_number == expected,
        "infrastructure attempt number is not monotonic"
    );
    let expected_prior = prior.last().map(|(value, _)| value.content_sha256.as_str());
    ensure!(
        receipt.prior_attempt_receipt_sha256.as_deref() == expected_prior,
        "infrastructure attempt prior-receipt chain drift"
    );
    let path = directory.join(format!(
        "{:08}-{}.json",
        receipt.attempt_number, receipt.content_sha256
    ));
    atomic_json(&path, &serde_json::to_value(receipt)?)?;
    Ok(path)
}

fn read_attempt_directory(
    directory: &Path,
    role: DimeRoleV1,
    scientific_payload_sha256: &str,
) -> Result<Vec<(DimeInfrastructureAttemptV1, PathBuf)>> {
    if !directory.is_dir() {
        return Ok(Vec::new());
    }
    let mut paths = fs::read_dir(directory)?
        .collect::<std::io::Result<Vec<_>>>()?
        .into_iter()
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|value| value.to_str()) == Some("json"))
        .collect::<Vec<_>>();
    paths.sort();
    let mut values = Vec::with_capacity(paths.len());
    for path in paths {
        let value: DimeInfrastructureAttemptV1 = serde_json::from_slice(
            &fs::read(&path).with_context(|| format!("read attempt receipt {}", path.display()))?,
        )
        .with_context(|| format!("decode attempt receipt {}", path.display()))?;
        value.validate()?;
        ensure!(
            value.role == role && value.scientific_payload_sha256 == scientific_payload_sha256,
            "attempt directory contains a foreign receipt"
        );
        ensure!(
            path.file_name().and_then(|name| name.to_str())
                == Some(
                    format!("{:08}-{}.json", value.attempt_number, value.content_sha256).as_str()
                ),
            "attempt receipt filename does not bind its content"
        );
        values.push((value, path));
    }
    values.sort_by_key(|(value, _)| value.attempt_number);
    let mut prior = None;
    for (index, (value, _)) in values.iter().enumerate() {
        ensure!(
            value.attempt_number == index as u64 + 1,
            "attempt ledger is not contiguous"
        );
        ensure!(
            value.prior_attempt_receipt_sha256.as_deref() == prior,
            "attempt ledger prior-receipt chain drift"
        );
        prior = Some(value.content_sha256.as_str());
    }
    Ok(values)
}

pub fn latest_attempt_receipt_sha256(
    root: &Path,
    role: DimeRoleV1,
    scientific_payload_sha256: &str,
) -> Result<Option<String>> {
    require_sha256(scientific_payload_sha256, "scientific payload")?;
    Ok(
        infrastructure_attempts(root, role, scientific_payload_sha256)?
            .last()
            .map(|value| value.content_sha256.clone()),
    )
}

pub fn latest_infrastructure_attempt(
    root: &Path,
    role: DimeRoleV1,
    scientific_payload_sha256: &str,
) -> Result<Option<DimeInfrastructureAttemptV1>> {
    Ok(infrastructure_attempts(root, role, scientific_payload_sha256)?.pop())
}

pub fn infrastructure_attempts(
    root: &Path,
    role: DimeRoleV1,
    scientific_payload_sha256: &str,
) -> Result<Vec<DimeInfrastructureAttemptV1>> {
    require_sha256(scientific_payload_sha256, "scientific payload")?;
    let directory = root
        .join("ops/dime-infrastructure-attempts")
        .join(role.as_str())
        .join(scientific_payload_sha256);
    if !directory.is_dir() {
        return Ok(Vec::new());
    }
    Ok(
        read_attempt_directory(&directory, role, scientific_payload_sha256)?
            .into_iter()
            .map(|(value, _)| value)
            .collect(),
    )
}
pub fn digest_label(label: &str) -> String {
    sha256_bytes(label.as_bytes())
}

pub fn read_validated_attempt(path: &Path) -> Result<DimeInfrastructureAttemptV1> {
    let value: DimeInfrastructureAttemptV1 = serde_json::from_slice(
        &fs::read(path).with_context(|| format!("read {}", path.display()))?,
    )?;
    value.validate()?;
    Ok(value)
}

pub fn ensure_same_scientific_payload(
    expected: &DimeScientificPayloadV1,
    observed: &DimeScientificPayloadV1,
) -> Result<()> {
    expected.validate()?;
    observed.validate()?;
    if expected != observed {
        bail!("infrastructure retry changed its scientific payload");
    }
    Ok(())
}
