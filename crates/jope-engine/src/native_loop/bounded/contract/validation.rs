use super::*;

impl BoundedRunContract {
    /// Time-independent validation: identity, uniqueness, grants and funding arithmetic.
    pub fn validate_structure(&self) -> Check {
        self.validate_with(Signatures::Verify)
    }
    /// Every check of [`Self::validate_structure`] except the grant signatures: an authored
    /// contract before `grant sign` (W7-D6). An unsigned contract is never a plan.
    pub fn validate_unsigned(&self) -> Check {
        self.validate_with(Signatures::Skip)
    }
    fn validate_with(&self, signatures: Signatures) -> Check {
        need!(
            self.schema == CONTRACT_SCHEMA,
            Invalid,
            "unsupported contract schema `{}`",
            self.schema
        );
        need!(is_name(&self.run), Invalid, "run identity");
        need!(self.attempt >= 1, Invalid, "attempt numbers start at 1");
        need!(is_name(&self.coordinator_host), Invalid, "coordinator host");
        verifying_key(&self.coordinator_public_key).map_err(invalid)?;
        verifying_key(&self.grant_authority_public_key).map_err(invalid)?;
        need!(
            self.coordinator_public_key != self.grant_authority_public_key,
            Invalid,
            "coordinator cannot hold the grant authority key"
        );
        need!(
            self.not_before_unix_ms < self.original_deadline_unix_ms,
            Invalid,
            "empty run window"
        );
        let window = self.original_deadline_unix_ms - self.not_before_unix_ms;
        need!(window <= MAX_WINDOW_MS, Invalid, "run window exceeds bound");
        need!(
            (1..=600_000).contains(&self.telemetry.max_age_ms)
                && self.telemetry.clock_skew_ms <= 10_000,
            Invalid,
            "telemetry freshness policy"
        );
        let r = &self.rendezvous;
        need!(
            (1..=window).contains(&r.join_timeout_ms)
                && (1..=window).contains(&r.heartbeat_timeout_ms)
                && (MIN_MESSAGE_BYTES..=MAX_MESSAGE_BYTES).contains(&r.max_message_bytes),
            Invalid,
            "rendezvous timeout/message policy"
        );
        self.validate_closure()?;
        self.validate_participants(window, signatures)?;
        self.validate_work_census()?;
        need!(
            self.counters.planned_updates > 0
                && self
                    .counters
                    .committed_updates_at_start
                    .checked_add(self.counters.planned_updates)
                    .is_some(),
            Invalid,
            "update counters"
        );
        match (&self.predecessor, self.attempt) {
            (None, 1) => {}
            (Some(p), a) if a > 1 => {
                need!(
                    p.attempt + 1 == a
                        && is_digest(&p.plan_sha256)
                        && is_digest(&p.journal_tip_sha256)
                        && p.committed_updates == self.counters.committed_updates_at_start,
                    Invalid,
                    "predecessor attempt/counters do not continue exactly"
                );
            }
            _ => {
                return Err(PlanRefusal::Invalid(
                    "attempt 1 has no predecessor; later attempts require one".into(),
                ))
            }
        }
        self.validate_training()
    }
    /// A selected partition must be one the trainer's gate accepts for this world size.
    fn validate_training(&self) -> Check {
        let Some(t) = &self.training else {
            return Ok(());
        };
        let world = self.participants.len();
        need!(
            TRAINING_MICROBATCHES.contains(&t.microbatch)
                && TRAINING_GLOBAL_BATCHES.contains(&t.global_batch),
            Invalid,
            "training batch shape: microbatch {} and global batch {}",
            t.microbatch,
            t.global_batch
        );
        match t.execution_partition.as_str() {
            crate::task_contract::PARTITION_V1_SCHEMA => need!(
                [1, 2, 4, 8].contains(&world) && (t.global_batch as usize).is_multiple_of(world * t.microbatch as usize),
                Invalid,
                "partition v1 cannot divide the global batch over {world} ranks; select partition v2"
            ),
            crate::task_contract::PARTITION_V2_SCHEMA => need!(
                (1..=8).contains(&world),
                Invalid,
                "partition v2 supports one to eight ranks, not {world}"
            ),
            other => {
                return Err(PlanRefusal::Invalid(format!(
                    "unknown training execution partition `{other}`"
                )))
            }
        }
        self.execution_partition().map_err(invalid)?;
        // W7-D2: what the ranks train from is pinned by the plan.
        let in_closure =
            |path: &std::path::Path| self.artifact_closure.iter().any(|e| e.path == path);
        need!(
            is_relative(&t.corpus) && in_closure(&t.corpus.join("index.json")),
            Invalid,
            "the training corpus index `{}/index.json` is not in the plan's input closure",
            t.corpus.display()
        );
        need!(
            (1..=self.counters.planned_updates).contains(&t.checkpoint_steps),
            Invalid,
            "checkpoint cadence {} is outside 1..={} planned updates",
            t.checkpoint_steps,
            self.counters.planned_updates
        );
        match &t.initialize {
            Some(parent) => need!(
                is_relative(parent) && in_closure(&parent.join("state.json")),
                Invalid,
                "the parent checkpoint state `{}/state.json` is not in the plan's input closure",
                parent.display()
            ),
            None => need!(
                self.counters.committed_updates_at_start == 0,
                Invalid,
                "a fresh training run starts at 0 committed updates, not {}; name the parent checkpoint it continues",
                self.counters.committed_updates_at_start
            ),
        }
        for p in &self.participants {
            need!(
                p.device.is_some(),
                Invalid,
                "rank {} has no GPU; every rank of a training plan trains on a GPU",
                p.global_rank
            );
        }
        Ok(())
    }
    /// The execution-partition record this plan's ranks train under, derived from its
    /// selection for exactly its world size; `None` when the plan selects no training.
    pub fn execution_partition(&self) -> anyhow::Result<Option<Value>> {
        self.training
            .as_ref()
            .map(|t| {
                crate::task_contract::training_partition_of(
                    &t.execution_partition,
                    self.world_size(),
                    t.microbatch as usize,
                    t.global_batch as usize,
                )
            })
            .transpose()
    }
    fn validate_closure(&self) -> Check {
        need!(
            (1..=MAX_CLOSURE_ENTRIES).contains(&self.artifact_closure.len()),
            Invalid,
            "artifact closure size"
        );
        for pair in self.artifact_closure.windows(2) {
            need!(
                pair[0].path < pair[1].path,
                Invalid,
                "artifact closure must be sorted and unique"
            );
        }
        for e in &self.artifact_closure {
            need!(
                is_relative(&e.path)
                    && is_digest(&e.sha256)
                    && (1..=MAX_ARTIFACT_BYTES).contains(&e.bytes),
                Invalid,
                "artifact closure entry `{}`",
                e.path.display()
            );
        }
        Ok(())
    }
    fn validate_participants(&self, window: u64, signatures: Signatures) -> Check {
        let n = self.participants.len();
        need!(
            (1..=MAX_PARTICIPANTS).contains(&n),
            Invalid,
            "participant count {n}"
        );
        let closure: BTreeSet<_> = self.artifact_closure.iter().map(|e| &e.path).collect();
        let mut identities: BTreeMap<&str, &HostIdentity> = BTreeMap::new();
        let mut local_ranks: BTreeMap<&str, BTreeSet<u32>> = BTreeMap::new();
        let mut devices = BTreeSet::new();
        let mut indices = BTreeSet::new();
        let mut keys = BTreeSet::from([
            self.coordinator_public_key.as_str(),
            self.grant_authority_public_key.as_str(),
        ]);
        let mut destinations = BTreeSet::new();
        for (i, p) in self.participants.iter().enumerate() {
            need!(
                p.global_rank as usize == i,
                Invalid,
                "global ranks must be exactly 0..{n} in order"
            );
            validate_identity(&p.host).map_err(invalid)?;
            let host = p.host.host.as_str();
            if let Some(first) = identities.insert(host, &p.host) {
                need!(
                    *first == p.host,
                    Invalid,
                    "participants on host `{host}` disagree on its identity"
                );
            }
            // One machine and one boot are one host: two host names cannot share them.
            for (name, other) in &identities {
                need!(
                    *name == host
                        || (other.machine_id != p.host.machine_id
                            && other.boot_id != p.host.boot_id),
                    Invalid,
                    "hosts `{name}` and `{host}` share a machine or boot identity"
                );
            }
            need!(
                local_ranks.entry(host).or_default().insert(p.local_rank),
                Invalid,
                "duplicate local rank on host `{host}`"
            );
            match &p.device {
                Some(d) => {
                    need!(is_name(&d.uuid), Invalid, "device UUID");
                    need!(
                        devices.insert(d.uuid.clone()),
                        Invalid,
                        "device `{}` assigned to more than one worker",
                        d.uuid
                    );
                    need!(
                        indices.insert((host, d.local_index)),
                        Invalid,
                        "local device index reused on host `{host}`"
                    );
                    need!(
                        p.ceilings.gpu_vram_bytes.is_some_and(|v| v > 0),
                        Invalid,
                        "GPU participant {i} lacks a VRAM ceiling"
                    );
                }
                None => need!(
                    p.ceilings.gpu_vram_bytes.is_none(),
                    Invalid,
                    "CPU participant {i} declares a VRAM ceiling"
                ),
            }
            need!(
                p.ceilings.cpu_millicores > 0
                    && p.ceilings.ram_bytes > 0
                    && (1..=window).contains(&p.ceilings.wall_ms),
                Invalid,
                "participant {i} ceilings"
            );
            verifying_key(&p.public_key).map_err(invalid)?;
            need!(
                keys.insert(p.public_key.as_str()),
                Invalid,
                "participant {i} key is not unique"
            );
            let d = &p.result_destination;
            need!(
                is_relative(&d.path)
                    && (1..=MAX_RESULT_BYTES).contains(&d.max_bytes)
                    && !closure.contains(&d.path)
                    && destinations.insert(d.path.clone()),
                Invalid,
                "participant {i} result destination"
            );
            self.validate_grant(p, signatures)?;
        }
        for (host, ranks) in local_ranks {
            need!(
                ranks.iter().copied().eq(0..ranks.len() as u32),
                Invalid,
                "local ranks on host `{host}` must be 0..{}",
                ranks.len()
            );
        }
        Ok(())
    }
    fn validate_grant(&self, p: &Participant, signatures: Signatures) -> Check {
        let g = &p.grant.body;
        need!(
            g.schema == GRANT_SCHEMA
                && g.run == self.run
                && g.attempt == self.attempt
                && g.global_rank == p.global_rank
                && g.host == p.host.host
                && g.boot_id == p.host.boot_id
                && g.device_uuid == p.device.as_ref().map(|d| d.uuid.clone())
                && g.participant_public_key == p.public_key
                && g.not_before_unix_ms < g.expires_unix_ms,
            Invalid,
            "grant for rank {} does not bind this run/attempt/participant",
            p.global_rank
        );
        if signatures == Signatures::Verify {
            verify(
                &self.grant_authority_public_key,
                GRANT_DOMAIN,
                g,
                &p.grant.signature,
            )
            .map_err(|e| {
                PlanRefusal::Invalid(format!("grant signature for rank {}: {e:#}", p.global_rank))
            })?;
        }
        need!(
            g.expires_unix_ms <= self.original_deadline_unix_ms,
            Invalid,
            "grant for rank {} outlives the original deadline",
            p.global_rank
        );
        let c = &p.ceilings;
        need!(
            g.cpu_millicores >= c.cpu_millicores
                && g.ram_bytes >= c.ram_bytes
                && g.gpu_vram_bytes.unwrap_or(0) >= c.gpu_vram_bytes.unwrap_or(0)
                && g.wall_ms >= c.wall_ms,
            Underfunded,
            "grant for rank {} does not fund its ceilings",
            p.global_rank
        );
        let start = g.not_before_unix_ms.max(self.not_before_unix_ms);
        need!(
            g.expires_unix_ms.saturating_sub(start) >= c.wall_ms,
            Underfunded,
            "grant window for rank {} is shorter than its wall ceiling",
            p.global_rank
        );
        Ok(())
    }
    fn validate_work_census(&self) -> Check {
        let n = self.participants.len() as u32;
        need!(
            (1..=MAX_WORK_ITEMS).contains(&self.work_census.len()),
            Invalid,
            "work census size"
        );
        let mut ids = BTreeSet::new();
        #[derive(Default)]
        struct Sum {
            cpu: u128,
            gpu: u64,
            wall: u64,
            ram: u64,
            vram: u64,
            updates: u64,
            kinds: BTreeSet<WorkKind>,
        }
        let mut sums: BTreeMap<u32, Sum> = BTreeMap::new();
        for w in &self.work_census {
            need!(
                is_name(&w.id) && ids.insert(w.id.as_str()),
                Invalid,
                "work item id `{}`",
                w.id
            );
            need!(
                w.global_rank < n,
                Invalid,
                "work item `{}` names no participant",
                w.id
            );
            let p = &self.participants[w.global_rank as usize];
            need!(
                p.device.is_some() || (w.gpu_ms == 0 && w.peak_vram_bytes == 0),
                Invalid,
                "GPU work `{}` on CPU participant",
                w.id
            );
            need!(
                w.kind == WorkKind::Numerical || w.updates == 0,
                Invalid,
                "non-numerical work `{}` claims updates",
                w.id
            );
            let s = sums.entry(w.global_rank).or_default();
            s.cpu += u128::from(w.cpu_core_ms);
            s.gpu = s.gpu.saturating_add(w.gpu_ms);
            s.wall = s.wall.saturating_add(w.wall_ms);
            s.ram = s.ram.max(w.peak_ram_bytes);
            s.vram = s.vram.max(w.peak_vram_bytes);
            s.updates = s.updates.saturating_add(w.updates);
            s.kinds.insert(w.kind);
        }
        for p in &self.participants {
            let r = p.global_rank;
            let Some(s) = sums.get(&r) else {
                return Err(PlanRefusal::Invalid(format!(
                    "work census omits participant {r}"
                )));
            };
            need!(
                s.kinds.contains(&WorkKind::Verification) && s.kinds.contains(&WorkKind::Cleanup),
                Invalid,
                "work census for rank {r} omits verification or cleanup"
            );
            need!(
                s.updates == self.counters.planned_updates,
                Invalid,
                "rank {r} plans {} updates; the synchronous group plans {}",
                s.updates,
                self.counters.planned_updates
            );
            let c = &p.ceilings;
            let funded_cpu = u128::from(c.cpu_millicores) * u128::from(c.wall_ms) / 1000;
            need!(
                s.cpu <= funded_cpu,
                Underfunded,
                "rank {r} needs {} CPU core-ms; ceilings fund {funded_cpu}",
                s.cpu
            );
            need!(
                s.wall <= c.wall_ms && s.gpu <= c.wall_ms,
                Underfunded,
                "rank {r} work census exceeds its wall ceiling"
            );
            need!(
                s.ram <= c.ram_bytes && s.vram <= c.gpu_vram_bytes.unwrap_or(0),
                Underfunded,
                "rank {r} peak memory exceeds its ceilings"
            );
        }
        Ok(())
    }
}
