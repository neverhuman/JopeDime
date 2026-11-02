use super::*;

/// A run manifest plus the work id of each `spec(i)` (manifest order is by work id, not `i`).
pub struct Run {
    pub manifest: Decoded<RunManifest>,
    pub work: Vec<String>,
}

pub fn run(id: &str, units: usize, max_attempts: u32) -> Run {
    let specs: Vec<WorkSpec> = (0..units).map(spec).collect();
    let work = specs.iter().map(|s| s.work_id(id).unwrap()).collect();
    let manifest = RunManifest::new(id, PUBLISHER, SOURCE_COMMIT, max_attempts, specs).unwrap();
    Run {
        manifest: Decoded::new(manifest).unwrap(),
        work,
    }
}

impl Run {
    pub fn id(&self) -> &str {
        &self.manifest.value().run_id
    }

    pub fn event(&self, unit: usize, attempt: u32, body: EventBody) -> Decoded<Event> {
        Decoded::new(Event::new(&self.manifest, &self.work[unit], attempt, body)).unwrap()
    }

    pub fn scheduled(&self, unit: usize) -> Decoded<Event> {
        self.event(unit, 1, EventBody::Scheduled)
    }

    pub fn started(&self, unit: usize, attempt: u32) -> Decoded<Event> {
        self.event(
            unit,
            attempt,
            EventBody::Started {
                worker: format!("worker-{unit}-{attempt}"),
            },
        )
    }

    pub fn failed(
        &self,
        unit: usize,
        attempt: u32,
        class: FailureClass,
        retryable: bool,
    ) -> Decoded<Event> {
        self.event(
            unit,
            attempt,
            EventBody::Failed {
                class,
                retryable,
                detail: format!("{} on attempt {attempt}", class.as_str()),
            },
        )
    }

    pub fn retry(&self, unit: usize, attempt: u32, failure: &Decoded<Event>) -> Decoded<Event> {
        self.event(
            unit,
            attempt,
            EventBody::Retry {
                failed_event: failure.digest().into(),
            },
        )
    }

    pub fn result(unit: usize, attempt: u32) -> Vec<u8> {
        format!("result of unit {unit} attempt {attempt}").into_bytes()
    }

    pub fn completed(&self, unit: usize, attempt: u32) -> Decoded<Event> {
        self.event(
            unit,
            attempt,
            EventBody::Completed {
                result: BlobRef::of(&Run::result(unit, attempt)),
            },
        )
    }

    pub fn excluded(&self, unit: usize, reason: &str) -> Decoded<Event> {
        self.event(
            unit,
            0,
            EventBody::Excluded {
                reason: reason.into(),
            },
        )
    }

    /// An HWM decision link for `completion` (not bound to a real ledger).
    pub fn decision(
        &self,
        completion: &Decoded<Event>,
        transition: Transition,
        batch: &str,
    ) -> Decoded<HwmDecision> {
        let e = completion.value();
        Decoded::new(HwmDecision {
            schema: Kind::HwmDecision.schema().into(),
            run_id: self.id().into(),
            work_id: e.work_id.clone(),
            attempt: e.attempt,
            completion: completion.digest().into(),
            record: BlobRef::of(format!("record {} {}", e.work_id, e.attempt).as_bytes()),
            identity_sha256: sha256_hex(e.work_id.as_bytes()),
            ledger_batch: batch.into(),
            ledger_path: format!("2026/09/{batch}.jsonl"),
            ledger_batch_sha256: sha256_hex(batch.as_bytes()),
            seq: 0,
            transition,
            qualification_sha256: None,
        })
        .unwrap()
    }
}
