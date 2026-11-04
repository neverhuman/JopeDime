use super::*;

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TombstoneState {
    Prepared,
    Acknowledged,
    ReconciledAbsent,
    AbortedStale,
    AbsentBeforeRequest,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Tombstone {
    pub schema: String,
    pub plan_sha256: String,
    pub reservation_id: String,
    pub request_id: String,
    pub exact_target: StrongPhysicalRef,
    pub state: TombstoneState,
    pub detail: String,
    pub tombstone_sha256: String,
}

impl Tombstone {
    pub(crate) fn new(
        reservation: &RetirementReservation,
        target: &StrongPhysicalRef,
    ) -> Result<Self> {
        let request_id = canonical_sha256(&(
            &reservation.plan_sha256,
            &reservation.reservation_id,
            target,
        ))?;
        let mut tombstone = Self {
            schema: "jope.retirement-tombstone/v1".into(),
            plan_sha256: reservation.plan_sha256.clone(),
            reservation_id: reservation.reservation_id.clone(),
            request_id,
            exact_target: target.clone(),
            state: TombstoneState::Prepared,
            detail: "durable intent represented in the mock journal".into(),
            tombstone_sha256: String::new(),
        };
        tombstone.tombstone_sha256 = canonical_sha256(&tombstone)?;
        Ok(tombstone)
    }

    pub(crate) fn reseal(&mut self) -> Result<()> {
        self.tombstone_sha256.clear();
        self.tombstone_sha256 = canonical_sha256(self)?;
        Ok(())
    }
}
