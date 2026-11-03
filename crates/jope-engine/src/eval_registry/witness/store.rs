use super::*;

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RecordCore {
    schema: String,
    sequence: u64,
    previous: String,
    request: SignedRequest,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Record {
    core: RecordCore,
    ack: SignedAck,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Head {
    sequence: u64,
    sha256: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OriginalEvent {
    instance: String,
    seq: u64,
    previous: String,
    kind: String,
    payload: Value,
    effects: Vec<OriginalEffect>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OriginalEffect {
    table: String,
    key: String,
    row: Value,
}
pub(crate) struct EventIdentity {
    pub(crate) seq: u64,
    pub(crate) deadline: u64,
}
#[cfg(test)]
pub(crate) fn event_identity(policy: &WitnessPolicy, event: &[u8]) -> Result<EventIdentity> {
    event_identity_effective(policy, policy, event)
}
pub(crate) fn event_identity_effective(
    base: &WitnessPolicy,
    policy: &WitnessPolicy,
    event: &[u8],
) -> Result<EventIdentity> {
    ensure!(
        !event.is_empty() && event.len() < MAX_FRAME / 2,
        "invalid authorization event size"
    );
    let value: OriginalEvent = serde_json::from_slice(event)?;
    ensure!(
        value.instance == policy.authority_instance
            && value.seq > 0
            && valid_hash(&value.previous)
            && matches!(
                value.kind.as_str(),
                "authorize" | "funding_start" | "funding_participant"
            )
            && value.effects.len() == 1,
        "not an original authorization event"
    );
    let effect = &value.effects[0];
    let row = &effect.row;
    if value.kind != "authorize" {
        let trust = policy
            .funding_trust
            .as_ref()
            .context("funding witness trust not installed")?;
        let original: crate::eval_registry::funding::FundingEvent =
            serde_json::from_value(value.payload["original"].clone())?;
        let (event_key, deadline) = crate::eval_registry::funding::validate_witness_event(
            trust,
            &original,
            &policy.authority_instance,
            &policy.authority_epoch,
        )?;
        let admission_deadline = value.payload["admission_deadline_monotonic_ns"]
            .as_u64()
            .context("original funding admission deadline absent")?;
        ensure!(
            original.claimed_monotonic_ns < admission_deadline && admission_deadline <= deadline,
            "funding admission deadline is outside original task window"
        );
        ensure!(
            value.payload["event_key"] == event_key
                && value.payload["reservation_id"] == original.reservation_id(),
            "funding journal key mismatch"
        );
        match original.participant_slot {
            None => {
                let reservation: crate::eval_registry::funding::SignedReservation =
                    serde_json::from_str(
                        row["json"].as_str().context("signed reservation absent")?,
                    )?;
                let stored: crate::eval_registry::funding::FundingEvent = serde_json::from_str(
                    row["start_json"]
                        .as_str()
                        .context("original funding start absent")?,
                )?;
                ensure!(
                    value.kind == "funding_start"
                        && effect.table == "funding_reservations"
                        && effect.key == original.reservation_id()
                        && row["id"] == effect.key
                        && row["state"] == "started"
                        && row["task_sha256"] == original.demand.task_sha256
                        && reservation == original.reservation
                        && row["authorization_id"]
                            == crate::eval_registry::funding::digest(&original.authorization.body)?
                        && stored == original,
                    "funding start postimage differs from original"
                );
            }
            Some(slot) => {
                let id = original.participant_id()?;
                let stored: crate::eval_registry::funding::FundingEvent = serde_json::from_str(
                    row["json"]
                        .as_str()
                        .context("original funding participant absent")?,
                )?;
                ensure!(
                    value.kind == "funding_participant"
                        && effect.table == "funding_participants"
                        && effect.key == id
                        && row["id"] == id
                        && value.payload["participant_id"] == id
                        && row["reservation_id"] == original.reservation_id()
                        && row["slot"].as_str() == Some(slot.to_string().as_str())
                        && stored == original,
                    "funding participant postimage differs from original"
                );
            }
        }
        return Ok(EventIdentity {
            seq: value.seq,
            deadline: admission_deadline,
        });
    }
    ensure!(
        effect.table == "entries"
            && row["key"] == effect.key
            && row["state"] == "authorized"
            && row["epoch"] == policy.authority_epoch,
        "authorization epoch/state mismatch"
    );
    let start: crate::eval_registry::StartToken = serde_json::from_str(
        row["start_json"]
            .as_str()
            .context("original full start absent")?,
    )?;
    let identity: crate::eval_registry::FitIdentity = serde_json::from_str(
        row["identity_json"]
            .as_str()
            .context("original fit identity absent")?,
    )?;
    if let Some(native) = value.payload.get("native_fit") {
        let delegated;
        let trust = if let Some(value) = native.get("native_grant_proof").filter(|v| !v.is_null()) {
            let proof: NativeGrantProof = serde_json::from_value(value.clone())?;
            delegated = verify_native_grant_proof(base, &proof)?;
            ensure!(
                verify_epoch_chain(base, &proof.epoch_proofs)?.sha256()? == policy.sha256()?,
                "native grant epoch differs from event epoch"
            );
            ensure!(
                native["signed_grant"] == serde_json::to_value(&proof.request.body.signed_grant)?,
                "native event grant differs from witnessed delegation"
            );
            &delegated
        } else {
            policy
                .native_fit_trust
                .as_ref()
                .context("native witness trust unavailable")?
        };
        crate::eval_registry::native_scq::verify_witness_event(
            trust, native, &identity, &start, policy,
        )?;
    }
    ensure!(
        start.fit_key == effect.key
            && sha(&bytes(&identity)?) == effect.key
            && identity.executable_sha256 == policy.authority_executable_sha256
            && value.payload["fit_key"] == effect.key
            && value.payload["token"] == start.token
            && value.payload["executor"] == serde_json::to_value(&start.executor)?
            && value.payload["claim_token"] == row["claim_token"]
            && start.task_id == row["task_id"]
            && start.deadline_monotonic_ns > start.authorized_monotonic_ns,
        "authorization original identity/token/custody mismatch"
    );
    Ok(EventIdentity {
        seq: value.seq,
        deadline: start.deadline_monotonic_ns,
    })
}
fn verify_request(base: &WitnessPolicy, request: &SignedRequest) -> Result<EventIdentity> {
    let effective = verify_epoch_chain(base, &request.body.epoch_proofs)?;
    let policy = &effective;
    ensure!(
        request.body.schema == "parallel.fit_event_witness_request.v1"
            && request.body.policy_sha256 == policy.sha256()?,
        "witness request policy mismatch"
    );
    verify(
        &policy.authority_public_key,
        "parallel.fit_event_witness.request.v1",
        &request.body,
        &request.signature,
    )?;
    event_identity_effective(base, policy, &request.body.event)
}
impl SignedAck {
    pub fn verify(&self, policy: &WitnessPolicy, request: &SignedRequest) -> Result<()> {
        let event = verify_request(policy, request)?;
        let base = policy;
        let effective = verify_epoch_chain(base, &request.body.epoch_proofs)?;
        let policy = &effective;
        ensure!(
            self.body.schema == "parallel.fit_event_witness_ack.v1"
                && self.body.policy_sha256 == policy.sha256()?
                && self.body.witness_instance == policy.witness_instance
                && self.body.authority_instance == policy.authority_instance
                && self.body.authority_epoch == policy.authority_epoch
                && self.body.authority_sequence == event.seq
                && self.body.event_sha256 == sha(&request.body.event)
                && self.body.request_sha256 == sha(&bytes(request)?)
                && self.body.witness_sequence > 0
                && valid_hash(&self.body.previous_record_sha256)
                && valid_hash(&self.body.committed_record_core_sha256),
            "witness ACK does not bind exact original authorization"
        );
        let original: OriginalEvent = serde_json::from_slice(&request.body.event)?;
        if let Some(value) = original
            .payload
            .get("native_fit")
            .and_then(|v| v.get("native_grant_proof"))
            .filter(|v| !v.is_null())
        {
            let proof: NativeGrantProof = serde_json::from_value(value.clone())?;
            verify_native_grant_proof(base, &proof)?;
            ensure!(
                proof.ack.body.witness_sequence < self.body.witness_sequence,
                "native grant update does not precede authorization ACK"
            );
        }
        verify(
            &policy.witness_public_key,
            "parallel.fit_event_witness.ack.v1",
            &self.body,
            &self.signature,
        )
    }
}

/// Storage stays outside the authority database restore domain. No startup
/// creates storage, repairs missing committed tails, or erases unknown files.
pub struct WitnessStore {
    pub(crate) root: PathBuf,
    pub(crate) policy: WitnessPolicy,
    pub(crate) signing: Option<SigningKey>,
    pub(crate) _lease: File,
    pub(crate) records: BTreeMap<u64, (String, SignedAck, String)>,
    pub(super) native_grants: native_grants::NativeGrantState,
    pub(crate) epoch_proofs: Vec<WitnessEpochProof>,
    pub(crate) sequence: u64,
    pub(crate) tip: String,
    pub(crate) root_identity: (u64, u64),
    pub(crate) segment_identity: (u64, u64),
    pub(crate) lease_identity: (u64, u64),
    pub(crate) read_only: bool,
}
mod operations;
