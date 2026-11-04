use super::*;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DimeDrainIntentV1 {
    pub schema: String,
    pub role: DimeRoleV1,
    pub scientific_payload_sha256: String,
    pub parent_intent_sha256: String,
    pub issued_at_ms: u128,
    pub expires_at_ms: u128,
    pub issuer: String,
    pub reason: String,
    pub content_sha256: String,
}

impl DimeDrainIntentV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        role: DimeRoleV1,
        scientific_payload_sha256: String,
        parent_intent_sha256: String,
        issued_at_ms: u128,
        expires_at_ms: u128,
        issuer: String,
        reason: String,
    ) -> Result<Self> {
        let mut value = Self {
            schema: DIME_DRAIN_INTENT_SCHEMA.to_owned(),
            role,
            scientific_payload_sha256,
            parent_intent_sha256,
            issued_at_ms,
            expires_at_ms,
            issuer,
            reason,
            content_sha256: String::new(),
        };
        value.content_sha256 = value.computed_content_sha256()?;
        value.validate_at(issued_at_ms)?;
        Ok(value)
    }

    fn computed_content_sha256(&self) -> Result<String> {
        let mut copy = self.clone();
        copy.content_sha256.clear();
        canonical_sha256(&copy)
    }

    pub fn validate_at(&self, now_ms: u128) -> Result<()> {
        ensure!(
            self.schema == DIME_DRAIN_INTENT_SCHEMA,
            "drain schema is not v1"
        );
        require_sha256(&self.scientific_payload_sha256, "drain scientific payload")?;
        require_sha256(&self.parent_intent_sha256, "drain parent intent")?;
        ensure!(
            self.issued_at_ms > 0
                && self.issued_at_ms <= now_ms
                && now_ms < self.expires_at_ms
                && self.expires_at_ms.saturating_sub(self.issued_at_ms) <= 24 * 60 * 60 * 1_000,
            "drain receipt is future-dated, expired, or exceeds 24 hours"
        );
        ensure!(
            !self.issuer.trim().is_empty() && !self.reason.trim().is_empty(),
            "drain issuer or reason is empty"
        );
        require_sha256(&self.content_sha256, "drain content")?;
        ensure!(
            self.computed_content_sha256()? == self.content_sha256,
            "drain content hash drift"
        );
        Ok(())
    }
}

pub fn write_drain_intent(root: &Path, intent: &DimeDrainIntentV1) -> Result<PathBuf> {
    intent.validate_at(intent.issued_at_ms)?;
    let path = root
        .join("ops/dime-drain-intents")
        .join(format!("{}.json", intent.content_sha256));
    if path.is_file() {
        let existing: DimeDrainIntentV1 = serde_json::from_slice(&fs::read(&path)?)?;
        ensure!(existing == *intent, "drain replay differs from its receipt");
        return Ok(path);
    }
    atomic_json(&path, &serde_json::to_value(intent)?)?;
    Ok(path)
}

pub fn matching_drain_intent(
    root: &Path,
    role: DimeRoleV1,
    scientific_payload: &DimeScientificPayloadV1,
    now_ms: u128,
) -> Result<Option<DimeDrainIntentV1>> {
    let directory = root.join("ops/dime-drain-intents");
    if !directory.is_dir() {
        return Ok(None);
    }
    let payload_sha256 = scientific_payload.identity_sha256()?;
    let mut matches = Vec::new();
    for entry in fs::read_dir(directory)?.filter_map(|entry| entry.ok()) {
        let Ok(value) = serde_json::from_slice::<DimeDrainIntentV1>(&fs::read(entry.path())?)
        else {
            continue;
        };
        if value.validate_at(now_ms).is_ok()
            && value.role == role
            && value.scientific_payload_sha256 == payload_sha256
            && value.parent_intent_sha256 == scientific_payload.parent_intent_sha256
        {
            matches.push(value);
        }
    }
    matches.sort_by_key(|value| value.issued_at_ms);
    Ok(matches.pop())
}
