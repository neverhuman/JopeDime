use super::*;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case", deny_unknown_fields)]
pub enum FundingRequest {
    Register {
        authorization: f::SignedAuthorization,
    },
    Reserve {
        authorization: f::SignedAuthorization,
        reservation: f::SignedReservation,
        descriptor_json: Value,
    },
    Preflight {
        authorization: f::SignedAuthorization,
        reservation: f::SignedReservation,
        descriptor_json: Value,
    },
    Start {
        authorization: f::SignedAuthorization,
        reservation: f::SignedReservation,
        descriptor_json: Value,
    },
    Participant {
        authorization: f::SignedAuthorization,
        reservation: f::SignedReservation,
        descriptor_json: Value,
        participant_slot: u32,
    },
}

/// Opaque proof of one witnessed native admission. This is not a complete GPU
/// execution permit: device lease, hold revision, inputs and post-context
/// runtime qualification remain separate mandatory task-contract checks.
pub struct VerifiedFundingAdmission {
    receipt: f::FundingReceipt,
}
impl VerifiedFundingAdmission {
    pub fn receipt(&self) -> &f::FundingReceipt {
        &self.receipt
    }
    pub fn check_current(&self, descriptor: &Value, participant: Option<u32>) -> Result<()> {
        let event = &self.receipt.event;
        ensure!(
            event.execution.process == ProcessIdentity::current()?
                && event.execution.request_sha256 == sha(&serde_json::to_vec(descriptor)?)
                && event.participant_slot == participant,
            "witnessed funding admission belongs to another task or participant"
        );
        let now = f::FundingClock::current()?;
        ensure!(
            now.unix_ms < event.demand.deadline_unix_ms
                && now.monotonic_ns < event.demand.deadline_monotonic_ns,
            "witnessed funding admission expired"
        );
        Ok(())
    }
}

pub fn funding_admission(
    config: &Path,
    request: &FundingRequest,
) -> Result<VerifiedFundingAdmission> {
    let started = request_monotonic_ns()?;
    let installation = InstalledFunding::load(config)?;
    funding_admission_installed(&installation, request, started)
}

pub(super) fn funding_admission_installed(
    installation: &InstalledFunding,
    request: &FundingRequest,
    started: u64,
) -> Result<VerifiedFundingAdmission> {
    ensure!(
        matches!(
            request,
            FundingRequest::Start { .. } | FundingRequest::Participant { .. }
        ),
        "preflight/reservation is not numerical admission"
    );
    let original = funding_request_installed(installation, request, started)?;
    Ok(VerifiedFundingAdmission {
        receipt: serde_json::from_value(original["receipt"].clone())?,
    })
}
impl FundingRequest {
    pub(super) fn task(&self) -> Option<(&f::SignedAuthorization, &f::SignedReservation, &Value)> {
        match self {
            Self::Register { .. } => None,
            Self::Reserve {
                authorization,
                reservation,
                descriptor_json,
            }
            | Self::Preflight {
                authorization,
                reservation,
                descriptor_json,
            }
            | Self::Start {
                authorization,
                reservation,
                descriptor_json,
            }
            | Self::Participant {
                authorization,
                reservation,
                descriptor_json,
                ..
            } => Some((authorization, reservation, descriptor_json)),
        }
    }
    pub(super) fn validate(&self, trust: &f::FundingTrust) -> Result<()> {
        let clock = f::FundingClock::current()?;
        if let Some((authorization, reservation, descriptor)) = self.task() {
            ensure!(
                descriptor.is_object()
                    && sha(&serde_json::to_vec(descriptor)?) == reservation.body.demand.task_sha256,
                "sealed common task descriptor differs from approved signed task"
            );
            f::verify_reservation(
                trust,
                authorization,
                reservation,
                &reservation.body.demand,
                clock,
            )?;
            if descriptor["schema"] == "parallel.resident_common_task.v1" {
                crate::task_contract::verify_resident_funding_demand(
                    descriptor,
                    &reservation.body.demand,
                )?;
            }
        } else if let Self::Register { authorization } = self {
            f::verify_authorization(trust, authorization, clock)?;
        }
        Ok(())
    }
}
