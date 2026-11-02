use super::*;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Slate {
    pub record: ResearchRound,
    pub dsls: Vec<String>,
    pub dispatch: Vec<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CapturedAttempt {
    pub proposal_index: usize,
    pub admitted_slot: Option<usize>,
    pub request: BlobRef,
    pub result: Option<BlobRef>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CapturedRound {
    pub slate: BlobRef,
    pub attempts: Vec<CapturedAttempt>,
    pub committed: Option<BlobRef>,
    pub aborted: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Measured {
    pub response: Response,
    pub outcome: RecordedOutcome,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct AuxiliaryAttempt {
    pub request: BlobRef,
    pub result: Option<BlobRef>,
    pub model: Option<BlobRef>,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum AuxiliaryResult {
    Evaluation { response: Response },
    Fit,
    Failure { worker: Option<Response> },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CaptureState {
    pub version: u32,
    pub binding: BlobRef,
    pub config: CaptureConfig,
    pub charged: u64,
    pub bytes: u64,
    pub label_counter: u32,
    pub tainted: bool,
    pub rounds: BTreeMap<u64, CapturedRound>,
    pub auxiliary: Vec<AuxiliaryAttempt>,
}

impl CaptureState {
    pub fn new(store: &Store, context: &CaptureContext) -> Result<Self> {
        let bytes = serde_json::to_vec(&context.binding)?;
        ensure!(
            bytes.len() as u64 <= context.binding.config.max_bytes,
            "capture binding exceeds byte budget"
        );
        let binding = store.publish_blob(Cursor::new(&bytes))?;
        Ok(Self {
            version: ENVELOPE_VERSION,
            binding,
            config: context.binding.config.clone(),
            charged: 0,
            bytes: bytes.len() as u64,
            label_counter: 0,
            tainted: false,
            rounds: BTreeMap::new(),
            auxiliary: vec![],
        })
    }

    pub fn check_binding(&self, store: &Store, context: &CaptureContext) -> Result<()> {
        ensure!(
            self.version == ENVELOPE_VERSION && self.config == context.binding.config,
            "capture version/config mismatch"
        );
        ensure!(
            store.read_blob(&self.binding)? == serde_json::to_vec(&context.binding)?,
            "capture input binding changed"
        );
        self.config.validate(context.binding.config.max_attempts)?;
        ensure!(
            self.charged <= self.config.max_attempts
                && self.bytes <= self.config.max_bytes
                && self.rounds.len() <= self.config.max_rounds,
            "capture accounting exceeds its bounds"
        );
        self.validate_storage(store)?;
        Ok(())
    }

    pub fn validate_storage(&self, store: &Store) -> Result<()> {
        ensure!(
            self.version == ENVELOPE_VERSION,
            "capture envelope version mismatch"
        );
        self.config.validate(self.config.max_attempts)?;
        let binding: Binding = Self::read(store, &self.binding)?;
        ensure!(self.config == binding.config, "capture config drift");
        let mut charged = 0_u64;
        for (index, (number, round)) in self.rounds.iter().enumerate() {
            ensure!(*number == index as u64, "capture round gap");
            let slate: Slate = Self::read(store, &round.slate)?;
            slate.record.validate()?;
            ensure!(
                slate.record.round == *number
                    && slate.record.identity == binding.identity
                    && slate.record.seed == binding.seed
                    && slate.record.tokens == binding.tokens
                    && slate.record.mask == binding.mask
                    && slate.record.evaluate_all_proposals == self.config.evaluate_all_proposals
                    && slate.dsls.len() == slate.record.proposals.len()
                    && slate
                        .dsls
                        .iter()
                        .zip(&slate.record.proposals)
                        .all(|(dsl, p)| digest(dsl.as_bytes()) == p.program_sha256),
                "capture slate binding mismatch"
            );
            let mut expected = slate.record.policy.admitted.clone();
            if slate.record.evaluate_all_proposals {
                expected.extend(
                    (0..slate.dsls.len()).filter(|i| !slate.record.policy.admitted.contains(i)),
                );
            }
            ensure!(
                expected.starts_with(&slate.dispatch),
                "capture dispatch mapping drift"
            );
            charged = charged
                .checked_add(slate.dispatch.len() as u64)
                .context("capture charge overflow")?;
            ensure!(
                round.attempts.len() == slate.dispatch.len(),
                "capture reservation inventory mismatch"
            );
            for (position, attempt) in round.attempts.iter().enumerate() {
                let request: Request = Self::read(store, &attempt.request)?;
                ensure!(
                    attempt.proposal_index == slate.dispatch[position]
                        && request.attempt.run.to_string() == binding.identity.run_id
                        && request.attempt.round == *number
                        && request.dsl == slate.dsls[attempt.proposal_index],
                    "capture request binding mismatch"
                );
                let admitted = slate
                    .record
                    .policy
                    .admitted
                    .iter()
                    .position(|i| *i == attempt.proposal_index);
                ensure!(
                    attempt.admitted_slot == admitted
                        && request.attempt.slot
                            == admitted.map_or(32 + attempt.proposal_index as u32, |s| s as u32),
                    "capture request slot mismatch"
                );
            }
            if let Some(reference) = &round.committed {
                let actual = self.completed_record(store, *number)?;
                ensure!(
                    store.read_blob(reference)? == serde_json::to_vec(&actual)?,
                    "committed capture differs from attempt journal"
                );
            }
        }
        for attempt in &self.auxiliary {
            let request: Request = Self::read(store, &attempt.request)?;
            ensure!(
                request.attempt.run.to_string() == binding.identity.run_id,
                "auxiliary request run mismatch"
            );
            if let Some(reference) = &attempt.result {
                let response = match Self::read(store, reference)? {
                    AuxiliaryResult::Evaluation { response } => Some(response),
                    AuxiliaryResult::Fit => {
                        ensure!(
                            attempt.model.as_ref().is_some_and(|model| model.bytes > 0),
                            "captured model root missing or empty"
                        );
                        None
                    }
                    AuxiliaryResult::Failure { worker } => {
                        ensure!(
                            worker
                                .as_ref()
                                .is_none_or(|response| response.failure.is_some()),
                            "auxiliary failure contains a successful response"
                        );
                        worker
                    }
                };
                if let Some(response) = response {
                    ensure!(
                        response.attempt == request.attempt
                            && response.valid()
                            && response.failure != Some(crate::worker::WorkerFailure::StaleAttempt),
                        "auxiliary response identity or outcome mismatch"
                    );
                }
            }
        }
        ensure!(
            charged == self.charged
                && charged <= self.config.max_attempts
                && self.bytes <= self.config.max_bytes
                && self.rounds.len() <= self.config.max_rounds
                && self.auxiliary.len() <= self.config.max_rounds.saturating_mul(2),
            "capture accounting mismatch"
        );
        let mut roots = BTreeMap::new();
        for root in self.roots() {
            store.verify_blob(&root)?;
            roots.insert(root.sha256, root.bytes);
        }
        ensure!(
            roots
                .values()
                .try_fold(0_u64, |sum, bytes| sum.checked_add(*bytes))
                .is_some_and(|total| total <= self.bytes),
            "capture byte accounting mismatch"
        );
        Ok(())
    }

    pub fn publish<T: Serialize>(&mut self, store: &Store, value: &T) -> Result<BlobRef> {
        let bytes = serde_json::to_vec(value)?;
        let total = self
            .bytes
            .checked_add(bytes.len() as u64)
            .context("capture byte counter exhausted")?;
        ensure!(
            total <= self.config.max_bytes,
            "capture byte budget exhausted"
        );
        let reference = store.publish_blob(Cursor::new(bytes))?;
        self.bytes = total;
        Ok(reference)
    }

    pub fn read<T: serde::de::DeserializeOwned>(store: &Store, reference: &BlobRef) -> Result<T> {
        ensure!(
            reference.bytes <= MAX_CAPTURE_BYTES,
            "capture blob exceeds ceiling"
        );
        Ok(serde_json::from_slice(&store.read_blob(reference)?)?)
    }

    pub fn roots(&self) -> Vec<BlobRef> {
        let mut roots = vec![self.binding.clone()];
        for round in self.rounds.values() {
            roots.push(round.slate.clone());
            roots.extend(round.committed.iter().cloned());
            for attempt in &round.attempts {
                roots.push(attempt.request.clone());
                roots.extend(attempt.result.iter().cloned());
            }
        }
        for attempt in &self.auxiliary {
            roots.push(attempt.request.clone());
            roots.extend(attempt.result.iter().cloned());
            roots.extend(attempt.model.iter().cloned());
        }
        roots
    }

    pub fn unknown(&self) -> bool {
        self.tainted
            || self.auxiliary.iter().any(|a| a.result.is_none())
            || self
                .rounds
                .values()
                .any(|r| r.aborted || r.attempts.iter().any(|a| a.result.is_none()))
    }

    pub fn measured_pool(&self, store: &Store, keys: &[String]) -> Result<Vec<RecordedReveal>> {
        let mut observed = BTreeMap::new();
        for round in self.rounds.values().filter(|r| r.committed.is_some()) {
            let record: ResearchRound = Self::read(store, round.committed.as_ref().unwrap())?;
            for outcome in record.outcomes {
                if !record.policy.admitted.contains(&outcome.proposal_index) {
                    continue;
                }
                if let Rung0Measurement::Success {
                    surrogate, gate, ..
                } = outcome.measurement
                {
                    let program = record.proposals[outcome.proposal_index].clone();
                    observed.insert(
                        program.program_sha256.clone(),
                        RecordedReveal {
                            program,
                            surrogate,
                            gate,
                        },
                    );
                }
            }
        }
        keys.iter()
            .map(|key| {
                observed
                    .get(key)
                    .cloned()
                    .context("elite lacks a committed measured gate")
            })
            .collect()
    }

    fn completed_record(&self, store: &Store, number: u64) -> Result<ResearchRound> {
        let round = self
            .rounds
            .get(&number)
            .context("capture round missing")?
            .clone();
        ensure!(!round.aborted, "capture round aborted");
        let mut slate: Slate = Self::read(store, &round.slate)?;
        ensure!(
            round.attempts.len() == slate.dispatch.len(),
            "incomplete capture reservations"
        );
        for attempt in &round.attempts {
            let measured: Measured = Self::read(
                store,
                attempt.result.as_ref().context("unknown capture outcome")?,
            )?;
            let request: Request = Self::read(store, &attempt.request)?;
            ensure!(
                measured.response.attempt == request.attempt
                    && measured.response.valid()
                    && measured.outcome.proposal_index == attempt.proposal_index,
                "captured attempt mismatch"
            );
            ensure!(
                match (&measured.response.failure, &measured.outcome.measurement) {
                    (Some(code), Rung0Measurement::Failed { code: recorded }) =>
                        code == recorded && *code != crate::worker::WorkerFailure::StaleAttempt,
                    (
                        None,
                        Rung0Measurement::Success {
                            surrogate,
                            objective,
                            ..
                        },
                    ) =>
                        measured.response.scores.len() == 3
                            && measured.response.scores[0] == *surrogate
                            && measured.response.scores[1] == *objective,
                    _ => false,
                },
                "capture measurement differs from response"
            );
            slate.record.dispatched.push(attempt.proposal_index);
            slate.record.outcomes.push(measured.outcome);
        }
        let expected = if slate.record.evaluate_all_proposals {
            slate.record.proposals.len()
        } else {
            slate.record.policy.admitted.len()
        };
        slate.record.capture_complete = slate.record.dispatched.len() == expected;
        slate.record.validate()?;
        ensure!(
            serde_json::to_vec(&slate.record)?.len() < MAX_LINE_BYTES,
            "capture record exceeds line bound"
        );
        Ok(slate.record)
    }

    pub fn finish_round(&mut self, store: &Store, number: u64) -> Result<()> {
        ensure!(
            self.rounds
                .get(&number)
                .is_some_and(|r| r.committed.is_none()),
            "capture round already committed"
        );
        let record = self.completed_record(store, number)?;
        let reference = self.publish(store, &record)?;
        self.rounds.get_mut(&number).unwrap().committed = Some(reference);
        Ok(())
    }
}
