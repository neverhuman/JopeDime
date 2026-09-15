use super::*;
use crate::eval_registry::{funding as f, funding_config::InstalledFunding};

impl Authority {
    pub fn attach_funding(&mut self, installation: InstalledFunding) -> Result<()> {
        self.require_funding_schema()?;
        let config = installation.revalidate()?;
        let trust = &config.trust;
        let epoch: String = self.db.query_row("SELECT COALESCE((SELECT epoch FROM recovery_epochs WHERE id='current'),instance) FROM meta", [], |r| r.get(0))?;
        ensure!(
            trust.authority_instance == self.manifest.instance
                && trust.authority_epoch == epoch
                && trust.authority_manifest_sha256 == self.manifest_sha
                && trust.authority_executable_sha256 == hash(&fs::read("/proc/self/exe")?)
                && trust.boot_id == ProcessIdentity::current()?.boot_id
                && trust.host_id == fs::read_to_string("/proc/sys/kernel/hostname")?.trim(),
            "funding installation does not bind this authority/epoch/executable/host"
        );
        ensure!(
            fs::canonicalize(&config.authority_root)? == self.root
                && fs::canonicalize(&config.witness_path)? == self.witness,
            "funding installation refers to another authority store"
        );
        let policy = self
            .event_witness
            .as_ref()
            .context("funding admission requires independent witness")?
            .policy();
        ensure!(
            policy.sha256()? == config.witness_policy_sha256
                && policy.funding_trust.as_ref() == Some(trust),
            "funding trust is not pinned by the independently installed witness policy"
        );
        self.funding_installation = Some(installation);
        Ok(())
    }

    fn funding_trust(&self) -> Result<f::FundingTrust> {
        self.require_funding_schema()?;
        Ok(self
            .funding_installation
            .as_ref()
            .context("installed funding authority unavailable")?
            .trust()?
            .clone())
    }

    pub(crate) fn funding_register(
        &mut self,
        authorization: &f::SignedAuthorization,
        deadline: u64,
    ) -> Result<String> {
        let trust = self.funding_trust()?;
        let id = f::verify_authorization(&trust, authorization, f::FundingClock::current()?)?;
        self.mutation(
            "funding_authorize",
            &json!({"authorization_id":id,"admission_deadline_monotonic_ns":deadline}),
            |db| {
                ensure!(
                    monotonic_ns()? < deadline,
                    "funding registration deadline elapsed"
                );
                f::register_authorization(db, &trust, authorization, f::FundingClock::current()?)
            },
        )
    }

    pub(crate) fn funding_reserve(
        &mut self,
        authorization: &f::SignedAuthorization,
        reservation: &f::SignedReservation,
        deadline: u64,
    ) -> Result<String> {
        let trust = self.funding_trust()?;
        self.mutation("funding_reserve", &json!({"reservation_id":reservation.body.reservation_id,"admission_deadline_monotonic_ns":deadline}), |db| {
            ensure!(monotonic_ns()? < deadline && deadline <= reservation.body.demand.deadline_monotonic_ns,
                "funding reservation deadline elapsed or exceeds original task window");
            let verified = f::verify_reservation(&trust, authorization, reservation, &reservation.body.demand, f::FundingClock::current()?)?;
            f::reserve(db, &verified)
        })
    }

    pub(crate) fn funding_preflight(
        &mut self,
        authorization: &f::SignedAuthorization,
        reservation: &f::SignedReservation,
        deadline: u64,
    ) -> Result<()> {
        let trust = self.funding_trust()?;
        ensure!(
            deadline <= reservation.body.demand.deadline_monotonic_ns,
            "funding preflight exceeds original task window"
        );
        let _lock = FileLock::acquire_until(
            &self.root.join("authority.lock"),
            true,
            false,
            Some(deadline),
        )?;
        self.verify_tail()?;
        let verified = f::verify_reservation(
            &trust,
            authorization,
            reservation,
            &reservation.body.demand,
            f::FundingClock::current()?,
        )?;
        ensure!(
            monotonic_ns()? < deadline,
            "funding preflight deadline elapsed"
        );
        f::preflight(&self.db, &verified)
    }

    /// Only the IPC layer supplies this independently observed peer execution.
    /// Losing the reply never makes its original numerical slot reusable.
    pub(crate) fn funding_admit(
        &mut self,
        authorization: &f::SignedAuthorization,
        reservation: &f::SignedReservation,
        execution: &f::ExecutionClaim,
        slot: Option<u32>,
        descriptor: &[u8],
        admission_deadline: u64,
    ) -> Result<Value> {
        let trust = self.funding_trust()?;
        let clock = f::FundingClock::current()?;
        ensure!(
            clock.monotonic_ns < admission_deadline
                && admission_deadline <= reservation.body.demand.deadline_monotonic_ns,
            "funding admission deadline elapsed or exceeds the original task window"
        );
        let verified = f::verify_reservation(
            &trust,
            authorization,
            reservation,
            &reservation.body.demand,
            clock,
        )?;
        ensure!(
            descriptor.len() <= 1024 * 1024 && hash(descriptor) == execution.request_sha256,
            "funding task descriptor custody mismatch or bound exceeded"
        );
        self.store_blob(descriptor)?;
        let original = f::event(&verified, execution, slot, clock)?;
        f::verify_event(&trust, &original, clock)?;
        let kind = if slot.is_some() {
            "funding_participant"
        } else {
            "funding_start"
        };
        let participant_id = slot
            .map(|slot| f::participant_id(original.reservation_id(), slot))
            .transpose()?;
        let payload = json!({"reservation_id":original.reservation_id(),"participant_id":participant_id,
            "event_key":original.event_key,"original":original,
            "admission_deadline_monotonic_ns":admission_deadline});
        self.mutation(kind, &payload, |db| {
            ensure!(
                monotonic_ns()? < admission_deadline,
                "funding admission expired before consumption"
            );
            f::verify_event(&trust, &original, f::FundingClock::current()?)?;
            let committed = match slot {
                Some(slot) => f::claim_participant(db, &verified, slot, execution, clock)?,
                None => f::consume_start(db, &verified, execution, clock)?,
            };
            ensure!(committed == original, "funding event changed before commit");
            Ok(())
        })?;
        ensure!(
            monotonic_ns()? < admission_deadline,
            "funding admission expired after durable commit; never replay"
        );
        let (event_sha, request_sha, ack_sha, policy_sha): (String, String, String, String) = self.db.query_row(
            "SELECT event_sha,request_sha,ack_sha,policy_sha FROM funding_witness WHERE event_key=?", [&original.event_key],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))?;
        Ok(
            json!({"schema":"parallel.funding_witness_proof.v1", "receipt":f::FundingReceipt {
            event:original, event_sha256:event_sha.clone(),request_sha256:request_sha.clone(),
            ack_sha256:ack_sha.clone(),policy_sha256:policy_sha.clone()},
            "event_bytes":self.read_blob(&event_sha)?,
            "request":serde_json::from_slice::<Value>(&self.read_blob(&request_sha)?)?,
            "ack":serde_json::from_slice::<Value>(&self.read_blob(&ack_sha)?)?,
            "policy":serde_json::from_slice::<Value>(&self.read_blob(&policy_sha)?)?}),
        )
    }
}
