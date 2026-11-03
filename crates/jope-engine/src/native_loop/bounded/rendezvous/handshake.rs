use super::*;

/// Coordinator-side rendezvous for one plan attempt. Fail-closed: the first invalid,
/// forged, stale or substituted message refuses the whole group.

pub struct Rendezvous {
    contract: BoundedRunContract,
    plan_sha256: String,
    closure_sha256: String,
    challenge: Challenge,
    hellos: BTreeMap<u32, (Vec<u8>, Hello)>,
    refused: Option<GroupRefusal>,
    admitted: Option<GroupAdmission>,
}
impl Rendezvous {
    /// `nonce` must be fresh per open; the id also binds the coordinator boot, so a hello
    /// captured before a coordinator restart can never join the new rendezvous.
    pub fn open(
        contract: &BoundedRunContract,
        plan_sha256: &str,
        coordinator: &SigningKey,
        coordinator_boot_id: &str,
        nonce: &[u8; 16],
        now_unix_ms: u64,
    ) -> Result<Self> {
        ensure!(
            hex::encode(coordinator.verifying_key().to_bytes()) == contract.coordinator_public_key,
            "coordinator key is not the planned coordinator"
        );
        ensure!(
            now_unix_ms >= contract.not_before_unix_ms
                && now_unix_ms < contract.original_deadline_unix_ms,
            "rendezvous outside the original run window"
        );
        let rendezvous_id = digest(&(
            "jope.native_rendezvous_id.v1",
            plan_sha256,
            contract.attempt,
            coordinator_boot_id,
            now_unix_ms,
            hex::encode(nonce),
        ))?;
        let grants_expire = contract
            .participants
            .iter()
            .map(|p| p.grant.body.expires_unix_ms)
            .min()
            .unwrap_or(0);
        let body = ChallengeBody {
            schema: CHALLENGE_SCHEMA.into(),
            run: contract.run.clone(),
            attempt: contract.attempt,
            plan_sha256: plan_sha256.into(),
            rendezvous_id,
            coordinator_boot_id: coordinator_boot_id.into(),
            opened_unix_ms: now_unix_ms,
            join_deadline_unix_ms: now_unix_ms
                .saturating_add(contract.rendezvous.join_timeout_ms)
                .min(contract.original_deadline_unix_ms)
                .min(grants_expire),
        };
        Ok(Self {
            closure_sha256: closure_digest(&contract.artifact_closure)?,
            contract: contract.clone(),
            plan_sha256: plan_sha256.into(),
            challenge: Challenge {
                signature: sign(coordinator, CHALLENGE_DOMAIN, &body)?,
                body,
            },
            hellos: BTreeMap::new(),
            refused: None,
            admitted: None,
        })
    }
    pub fn challenge(&self) -> &Challenge {
        &self.challenge
    }
    pub(super) fn contract(&self) -> &BoundedRunContract {
        &self.contract
    }
    pub fn rendezvous_id(&self) -> &str {
        &self.challenge.body.rendezvous_id
    }
    pub fn refusal(&self) -> Option<&GroupRefusal> {
        self.refused.as_ref()
    }
    pub fn joined(&self) -> Vec<u32> {
        self.hellos.keys().copied().collect()
    }
    pub(super) fn refuse(&mut self, cause: RefusalCause) -> GroupRefusal {
        let refusal = self.refused.get_or_insert_with(|| GroupRefusal {
            run: self.contract.run.clone(),
            attempt: self.contract.attempt,
            rendezvous_id: self.challenge.body.rendezvous_id.clone(),
            cause,
            started_ranks: 0,
            committed_updates: 0,
        });
        refusal.clone()
    }
    /// Offer one untrusted hello frame from a same-host fixture socket. Returns the rank it
    /// authenticated as.
    pub fn offer(&mut self, frame: &[u8], now_unix_ms: u64) -> Result<u32, GroupRefusal> {
        self.offer_from(frame, &PeerIdentity::SameHostSocket, now_unix_ms)
    }
    /// Offer one untrusted hello frame received from `peer`. A certificate-authenticated peer
    /// may only speak for a rank planned on the host that owns its certificate; any other
    /// rank is refused as a wrong participant, which refuses the whole group.
    pub fn offer_from(
        &mut self,
        frame: &[u8],
        peer: &PeerIdentity,
        now_unix_ms: u64,
    ) -> Result<u32, GroupRefusal> {
        if let Some(r) = &self.refused {
            return Err(r.clone());
        }
        if self.admitted.is_some() {
            return Err(self.refuse(RefusalCause::Malformed {
                detail: "hello after admission".into(),
            }));
        }
        let limit = u64::from(self.contract.rendezvous.max_message_bytes);
        if frame.len() as u64 > limit {
            return Err(self.refuse(RefusalCause::Oversized {
                bytes: frame.len() as u64,
                limit,
            }));
        }
        let hello: Hello = match serde_json::from_slice(frame) {
            Ok(h) => h,
            Err(e) => {
                return Err(self.refuse(RefusalCause::Malformed {
                    detail: e.to_string(),
                }))
            }
        };
        match self.check_hello(&hello, now_unix_ms) {
            Ok(()) => {}
            Err(cause) => return Err(self.refuse(cause)),
        }
        let rank = hello.body.global_rank;
        if let PeerIdentity::Certificate { sha256 } = peer {
            let planned = self
                .contract
                .participant(rank)
                .map(|p| p.host.certificate_sha256.as_str());
            if planned != Some(sha256.as_str()) {
                return Err(self.refuse(RefusalCause::WrongParticipant { rank }));
            }
        }
        if let Some((old, _)) = self.hellos.get(&rank) {
            if old.as_slice() == frame {
                return Ok(rank); // identical redelivery is idempotent
            }
            return Err(self.refuse(RefusalCause::Equivocation { rank }));
        }
        self.hellos.insert(rank, (frame.to_vec(), hello));
        Ok(rank)
    }
    fn check_hello(&self, hello: &Hello, now: u64) -> std::result::Result<(), RefusalCause> {
        let b = &hello.body;
        let c = &self.contract;
        let rank = b.global_rank;
        let Some(p) = c.participant(rank) else {
            return Err(RefusalCause::UnknownRank { rank });
        };
        // Authenticate before trusting any other field.
        if verify(&p.public_key, HELLO_DOMAIN, b, &hello.signature).is_err() {
            return Err(RefusalCause::ForgedPeer { claimed_rank: rank });
        }
        if b.schema != HELLO_SCHEMA || b.delivery_id == 0 {
            return Err(RefusalCause::Malformed {
                detail: "hello schema/delivery id".into(),
            });
        }
        if b.run != c.run {
            return Err(RefusalCause::WrongRun);
        }
        if b.attempt != c.attempt {
            return Err(RefusalCause::ReplayedAttempt { attempt: b.attempt });
        }
        if b.plan_sha256 != self.plan_sha256 {
            return Err(RefusalCause::WrongPlan);
        }
        if b.rendezvous_id != self.challenge.body.rendezvous_id {
            return Err(RefusalCause::StaleRendezvous);
        }
        if b.boot_id != p.host.boot_id {
            return Err(RefusalCause::ReplayedBoot { rank });
        }
        if b.host != p.host.host
            || b.machine_id != p.host.machine_id
            || b.device_uuid != p.device.as_ref().map(|d| d.uuid.clone())
        {
            return Err(RefusalCause::WrongParticipant { rank });
        }
        if b.source_commit != p.host.source_commit
            || b.executable_sha256 != p.host.executable_sha256
            || b.runtime != p.host.runtime
            || b.installed_policy_sha256 != p.host.installed_policy_sha256
        {
            return Err(RefusalCause::SubstitutedSource { rank });
        }
        if b.artifact_closure_sha256 != self.closure_sha256 {
            return Err(RefusalCause::SubstitutedInput { rank });
        }
        if digest(&p.grant).ok().as_deref() != Some(b.grant_sha256.as_str()) {
            return Err(RefusalCause::WrongGrant { rank });
        }
        let skew = c.telemetry.clock_skew_ms;
        if b.sent_unix_ms.saturating_add(skew) < self.challenge.body.opened_unix_ms
            || b.sent_unix_ms > now.saturating_add(skew)
        {
            return Err(RefusalCause::StaleMessage { rank });
        }
        if now >= c.original_deadline_unix_ms {
            return Err(RefusalCause::DeadlineElapsed);
        }
        if now >= p.grant.body.expires_unix_ms || now < p.grant.body.not_before_unix_ms {
            return Err(RefusalCause::GrantExpired { rank });
        }
        if now > self.challenge.body.join_deadline_unix_ms {
            return Err(RefusalCause::JoinTimeout {
                missing: self.missing(),
            });
        }
        let bootstrap_ok = match (&b.communicator_bootstrap, rank) {
            (Some(x), 0) => {
                x.len() <= 2 * MAX_BOOTSTRAP_BYTES && !x.is_empty() && hex::decode(x).is_ok()
            }
            (None, r) => r != 0,
            _ => false,
        };
        if !bootstrap_ok {
            return Err(RefusalCause::Malformed {
                detail: "communicator bootstrap is carried by rank 0 only".into(),
            });
        }
        Ok(())
    }
    fn missing(&self) -> Vec<u32> {
        (0..self.contract.world_size() as u32)
            .filter(|r| !self.hellos.contains_key(r))
            .collect()
    }
    /// `Ok(None)` while waiting inside the join window. Admission requires every planned
    /// rank, unexpired grants and the original deadline; after the join deadline a
    /// partial group is refused as a whole.
    pub fn admit(
        &mut self,
        coordinator: &SigningKey,
        now_unix_ms: u64,
    ) -> Result<Option<GroupAdmission>, GroupRefusal> {
        if let Some(r) = &self.refused {
            return Err(r.clone());
        }
        if let Some(a) = &self.admitted {
            return Ok(Some(a.clone()));
        }
        let missing = self.missing();
        if !missing.is_empty() {
            if now_unix_ms > self.challenge.body.join_deadline_unix_ms {
                return Err(self.refuse(RefusalCause::JoinTimeout { missing }));
            }
            return Ok(None);
        }
        if now_unix_ms >= self.contract.original_deadline_unix_ms {
            return Err(self.refuse(RefusalCause::DeadlineElapsed));
        }
        for p in &self.contract.participants {
            if now_unix_ms >= p.grant.body.expires_unix_ms {
                return Err(self.refuse(RefusalCause::GrantExpired {
                    rank: p.global_rank,
                }));
            }
        }
        let grants_sha256 = {
            let grants: Vec<_> = self
                .contract
                .participants
                .iter()
                .map(|p| &p.grant)
                .collect();
            digest(&grants)
        };
        let grants_sha256 = match grants_sha256 {
            Ok(g) => g,
            Err(e) => {
                return Err(self.refuse(RefusalCause::Malformed {
                    detail: e.to_string(),
                }))
            }
        };
        let members = self
            .hellos
            .iter()
            .map(|(rank, (bytes, h))| Member {
                global_rank: *rank,
                hello_sha256: crate::manifest::hash(bytes),
                host: h.body.host.clone(),
                boot_id: h.body.boot_id.clone(),
                device_uuid: h.body.device_uuid.clone(),
            })
            .collect();
        let body = AdmissionBody {
            schema: ADMISSION_SCHEMA.into(),
            run: self.contract.run.clone(),
            attempt: self.contract.attempt,
            plan_sha256: self.plan_sha256.clone(),
            rendezvous_id: self.challenge.body.rendezvous_id.clone(),
            admitted_unix_ms: now_unix_ms,
            original_deadline_unix_ms: self.contract.original_deadline_unix_ms,
            grants_sha256,
            members,
            communicator_bootstrap: self.hellos[&0]
                .1
                .body
                .communicator_bootstrap
                .clone()
                .unwrap_or_default(),
        };
        let signature = sign(coordinator, ADMISSION_DOMAIN, &body).map_err(|e| {
            self.refuse(RefusalCause::Malformed {
                detail: e.to_string(),
            })
        })?;
        let admission = GroupAdmission { body, signature };
        self.admitted = Some(admission.clone());
        Ok(Some(admission))
    }
}
