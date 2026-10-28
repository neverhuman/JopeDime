//! FIT-scoped components admitted by the shared controller verifier, never by request claims.
use super::{sha256, FusionComponents, TrainingInput};
use anyhow::{ensure, Context, Result};
use dope_core::prepare::{ClassVocabPolicy, PreparedTable};
use jope_records::replay_admission::{AdmissionVerifier, FitAdmission};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

mod authority;
#[cfg(test)]
mod tests;
use authority::{read_input, ControllerStore};

const PREPARATION_SOURCE: &str = "15e12aa3dc33851655580f6aa7c9c66353143f23";
/// Already projected/normalized by the authenticated producer; fusion adds no normalizer.
pub const FUSION_COMPONENT_RECIPE: &str = "foundation1024-target512-supplied-normalized/v1";
/// Explicit product preparation; model weights and coordinate recipes remain unchanged.
pub const LOSSLESS_PREPARATION_RECIPE: &str = "jope.dope-product-preparation/lossless-v1";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FitScope {
    pub schema: String,
    pub preparation_source: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preparation_recipe: Option<String>,
    pub source_digest: String,
    pub physical_rows_sha256: String,
    pub rows: usize,
    pub columns: usize,
    pub task: String,
    pub fold: Option<String>,
}
impl FitScope {
    pub fn from_prepared(input: &TrainingInput, prepared: &PreparedTable) -> Result<Self> {
        ensure!(
            include_str!("../../Cargo.toml").contains(PREPARATION_SOURCE),
            "component preparation identity must follow the pinned Dope source"
        );
        Ok(Self {
            schema: "jope.dope-fit-scope/v1".into(),
            preparation_source: PREPARATION_SOURCE.into(),
            preparation_recipe: match prepared.transform.class_vocab_policy {
                ClassVocabPolicy::Cap32V1 => None,
                ClassVocabPolicy::LosslessV1 => Some(LOSSLESS_PREPARATION_RECIPE.into()),
            },
            source_digest: prepared.source_rows_digest(),
            physical_rows_sha256: sha256(&serde_json::to_vec(&input.ids)?),
            rows: input.ids.len(),
            columns: input.categories.len(),
            task: format!("{:?}", input.head),
            fold: input.fold.clone(),
        })
    }
    pub fn digest(&self) -> Result<String> {
        Ok(sha256(&serde_json::to_vec(self)?))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ComponentAdmissionIdentity {
    pub schema: String,
    pub subject_sha256: String,
    pub exposure_receipt_sha256: String,
    pub policy_sha256: String,
    pub authority_sha256: String,
    pub fit_scope_digest: String,
    pub episode: String,
    pub purpose: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Requests {
    schema: String,
    entries: Vec<Request>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    fit_scope_digest: String,
    subject_sha256: String,
    exposure_receipt_sha256: String,
    episode: String,
}

/// A retained subject, not an authority or an admission Boolean. Floats use exact LE bytes.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FusionComponentArtifact {
    pub schema: String,
    pub fit_scope: FitScope,
    pub episode: String,
    pub permitted_use: String,
    pub recipe: String,
    pub producer_inputs: BTreeMap<String, String>,
    pub foundation1024_le_f32: String,
    pub target512_le_f32: String,
}

/// Request references have no authority path, issuer override or caller-selected policy pin.
pub struct ComponentInputs {
    authority: Arc<ControllerStore>,
    requests: BTreeMap<String, Request>,
    path: PathBuf,
}
impl ComponentInputs {
    pub fn load(path: &Path) -> Result<Self> {
        Self::load_with_controller(path, &jope_paths::jope_state())
    }

    /// The host/controller selects this store; it must never come from request contents.
    pub fn load_with_controller(path: &Path, controller_state: &Path) -> Result<Self> {
        let request: Requests = serde_json::from_slice(&read_input(path, 1024 * 1024)?)?;
        ensure!(
            request.schema == "jope.fusion-component-requests/v1"
                && !request.entries.is_empty()
                && request.entries.len() <= 128,
            "invalid component request roster"
        );
        let mut requests = BTreeMap::new();
        for entry in request.entries {
            ensure!(
                !entry.episode.is_empty() && entry.episode.len() <= 256,
                "invalid component episode"
            );
            ensure!(
                [
                    &entry.fit_scope_digest,
                    &entry.subject_sha256,
                    &entry.exposure_receipt_sha256
                ]
                .into_iter()
                .all(|s| authority::is_digest(s)),
                "invalid component request digest"
            );
            ensure!(
                requests
                    .insert(entry.fit_scope_digest.clone(), entry)
                    .is_none(),
                "ambiguous component FIT scope"
            );
        }
        Ok(Self {
            authority: Arc::new(ControllerStore::load(controller_state)?),
            requests,
            path: path.to_owned(),
        })
    }

    pub fn protected_paths(&self) -> [&Path; 2] {
        [&self.path, self.authority.root()]
    }

    pub fn resolve(&self, expected: &FitScope) -> Result<AuthenticatedFusionComponents> {
        let fit_scope_digest = expected.digest()?;
        let request = self
            .requests
            .get(&fit_scope_digest)
            .context("no admitted components for this physical FIT/fold scope")?;
        // This selector grants nothing: the verifier compares every ancestor's episode.
        let episode = request.episode.clone();
        let verifier = AdmissionVerifier::from_adopted_policy(self.authority.as_ref())?;
        let admitted = verifier.admit_fitting_components(
            &request.subject_sha256,
            &request.exposure_receipt_sha256,
            &episode,
            authority::now_ms()?,
        )?;
        let bytes = self.authority.subject(&request.subject_sha256)?;
        let payload = jope_records::wire_contract::CanonicalPayload::from_canonical_bytes(&bytes)?;
        ensure!(
            payload.schema_id() == "jope.dope-fusion-components/v1",
            "invalid component subject schema"
        );
        let artifact: FusionComponentArtifact = serde_json::from_slice(&bytes)?;
        ensure!(
            artifact.fit_scope == *expected
                && artifact.episode == episode
                && artifact.permitted_use == "episode_fit"
                && artifact.recipe == FUSION_COMPONENT_RECIPE,
            "component FIT/preparation/episode/recipe differs"
        );
        ensure!(
            artifact.producer_inputs.contains_key("foundation")
                && artifact.producer_inputs.contains_key("target")
                && artifact
                    .producer_inputs
                    .values()
                    .all(|d| authority::is_digest(d)
                        && d != &request.subject_sha256
                        && admitted.admission().source_subjects().contains(d)),
            "component producer inputs are not in the admitted ancestry"
        );
        let foundation = decode_f32(&artifact.foundation1024_le_f32, 1024)?;
        let target = decode_f32(&artifact.target512_le_f32, 512)?;
        self.authority.require_active()?;
        let policy_sha256 = admitted.admission().policy_sha256().to_owned();
        Ok(AuthenticatedFusionComponents {
            admission: Arc::new(admitted),
            authority: self.authority.clone(),
            scope: expected.clone(),
            recipe: artifact.recipe,
            foundation,
            target,
            identity: ComponentAdmissionIdentity {
                schema: "jope.component-admission/v1".into(),
                subject_sha256: request.subject_sha256.clone(),
                exposure_receipt_sha256: request.exposure_receipt_sha256.clone(),
                policy_sha256,
                authority_sha256: self.authority.digest().into(),
                fit_scope_digest,
                episode,
                purpose: "episode_fit".into(),
            },
        })
    }
}

/// Opaque host capability. Deserializing a vector or receipt cannot construct it.
#[derive(Clone)]
pub struct AuthenticatedFusionComponents {
    admission: Arc<FitAdmission>,
    authority: Arc<ControllerStore>,
    scope: FitScope,
    recipe: String,
    foundation: Vec<f32>,
    target: Vec<f32>,
    identity: ComponentAdmissionIdentity,
}
impl AuthenticatedFusionComponents {
    pub fn identity(&self) -> &ComponentAdmissionIdentity {
        &self.identity
    }
    pub fn require_active(&self) -> Result<()> {
        self.require_active_at(authority::now_ms()?)
    }
    fn require_active_at(&self, at_ms: u64) -> Result<()> {
        // Recheck policy time, every retained proof/subject byte and the strict FIT ancestry.
        // Controller-file identity alone does not keep an earlier capability current.
        let verifier = AdmissionVerifier::from_adopted_policy(self.authority.as_ref())?;
        let admission = self.admission.admission();
        verifier.admit_fitting_components(
            admission.subject_sha256(),
            admission.receipt_sha256(),
            &self.identity.episode,
            at_ms,
        )?;
        self.authority.require_active()
    }
    pub fn bind(&self, expected: &FitScope, landmark_scope: &str) -> Result<FusionComponents> {
        ensure!(
            self.scope == *expected,
            "authenticated component FIT/fold scope differs"
        );
        self.require_active()?;
        Ok(FusionComponents {
            source_digest: landmark_scope.into(),
            recipe: self.recipe.clone(),
            foundation1024: self.foundation.clone(),
            target512: self.target.clone(),
        })
    }
}

fn decode_f32(value: &str, count: usize) -> Result<Vec<f32>> {
    ensure!(value.len() == count * 8, "component vector width differs");
    let mut result = Vec::with_capacity(count);
    for bytes in value.as_bytes().chunks_exact(8) {
        let mut raw = [0u8; 4];
        for (out, hex) in raw.iter_mut().zip(bytes.chunks_exact(2)) {
            let text = std::str::from_utf8(hex)?;
            ensure!(
                hex.iter()
                    .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(c)),
                "noncanonical component float bytes"
            );
            *out = u8::from_str_radix(text, 16)?;
        }
        let value = f32::from_le_bytes(raw);
        ensure!(value.is_finite(), "nonfinite component value");
        result.push(value);
    }
    Ok(result)
}
