//! Shared builders for the W8 progress tests.

use crate::canon::sha256_hex;
use crate::hwm::Transition;
use crate::metric::Metric;
use crate::progress::inventory::*;
use crate::record::*;

pub const SOURCE_COMMIT: &str = "c4f60b4c7a2e27a31fc41f875650c89900dc32ed";
pub const PUBLISHER: &str = "xbabe2";

pub fn spec(i: usize) -> WorkSpec {
    WorkSpec {
        stage: "gp-fit".into(),
        subject: format!("dataset-{i:03}"),
        inputs: vec![sha256_hex(format!("input-{i}").as_bytes())],
    }
}

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

/// A labelled event list, so a test can drop exactly one delivery by name.
pub struct Scenario {
    pub events: Vec<(&'static str, Decoded<Event>)>,
    pub decisions: Vec<(&'static str, Decoded<HwmDecision>)>,
}

impl Scenario {
    pub fn events(&self) -> Vec<Decoded<Event>> {
        self.events.iter().map(|(_, e)| e.clone()).collect()
    }

    pub fn decisions(&self) -> Vec<Decoded<HwmDecision>> {
        self.decisions.iter().map(|(_, d)| d.clone()).collect()
    }

    pub fn event(&self, label: &str) -> &Decoded<Event> {
        &self
            .events
            .iter()
            .find(|(l, _)| *l == label)
            .unwrap_or_else(|| panic!("no event {label}"))
            .1
    }

    pub fn without(&self, labels: &[&str]) -> Vec<Decoded<Event>> {
        for l in labels {
            self.event(l);
        }
        self.events
            .iter()
            .filter(|(l, _)| !labels.contains(l))
            .map(|(_, e)| e.clone())
            .collect()
    }
}

/// Five planned units, a budget of three attempts, and one of each terminal shape:
///
/// * unit 0 completes on attempt 1 and initializes its HWM;
/// * unit 1 times out, is retried, completes on attempt 2 and improves its HWM;
/// * unit 2 is refused by preflight (not retryable): terminal failure;
/// * unit 3 is excluded before scheduling (missing paired arm);
/// * unit 4 fails three retryable attempts and exhausts its budget.
pub fn standard() -> (Run, Scenario) {
    let r = run("run-w8-t01", 5, 3);
    let u1_f1 = r.failed(1, 1, FailureClass::Timeout, true);
    let u4_f1 = r.failed(4, 1, FailureClass::Timeout, true);
    let u4_f2 = r.failed(4, 2, FailureClass::WorkerLost, true);
    let u0_c1 = r.completed(0, 1);
    let u1_c2 = r.completed(1, 2);
    let events = vec![
        ("u0-scheduled", r.scheduled(0)),
        ("u0-started-1", r.started(0, 1)),
        ("u0-completed-1", u0_c1.clone()),
        ("u1-scheduled", r.scheduled(1)),
        ("u1-failed-1", u1_f1.clone()),
        ("u1-retry-2", r.retry(1, 2, &u1_f1)),
        ("u1-completed-2", u1_c2.clone()),
        ("u2-scheduled", r.scheduled(2)),
        (
            "u2-failed-1",
            r.failed(2, 1, FailureClass::PreflightRefusal, false),
        ),
        ("u3-excluded", r.excluded(3, "missing paired arm")),
        ("u4-scheduled", r.scheduled(4)),
        ("u4-failed-1", u4_f1.clone()),
        ("u4-retry-2", r.retry(4, 2, &u4_f1)),
        ("u4-failed-2", u4_f2.clone()),
        ("u4-retry-3", r.retry(4, 3, &u4_f2)),
        (
            "u4-failed-3",
            r.failed(4, 3, FailureClass::ParseRejection, true),
        ),
    ];
    let decisions = vec![
        (
            "u0-initialized",
            r.decision(
                &u0_c1,
                Transition::Initialized,
                "01K62AAAAAAAAAAAAAAAAAAAA0",
            ),
        ),
        (
            "u1-improved",
            r.decision(&u1_c2, Transition::Improved, "01K62AAAAAAAAAAAAAAAAAAAA1"),
        ),
    ];
    (r, Scenario { events, decisions })
}

/// An `hwm::ingest`-able record (legacy diagnostic pointer space: no comparison context).
pub fn hwm_record(cut: &str, value: f64) -> (HwmRecord, Vec<u8>) {
    let identity = HwmIdentity {
        comparison: None,
        task: "regression".into(),
        dataset_hash: "dataset".into(),
        cut_id: sha256_hex(cut.as_bytes()),
        evaluator_sha256: UNBOUND.into(),
        vocabulary_sha256: UNBOUND.into(),
        protocol: "test".into(),
    };
    let dsl = format!("(pipe X (ridge alpha={value}))");
    let blob = format!("{cut}/{value}").into_bytes();
    let record = HwmRecord {
        schema: HWM_SCHEMA.into(),
        identity_sha256: identity.sha256().unwrap(),
        identity,
        legacy_identity_sha256: None,
        genome: Genome {
            program_sha256: sha256_hex(dsl.as_bytes()),
            dsl,
            semantic_sha256: None,
        },
        score: Score {
            metric: Metric::Rmse,
            value,
            direction: Metric::Rmse.direction(),
            units: "original_target_units".into(),
        },
        cost: Cost::default(),
        evaluator: Evaluator {
            sha256: UNBOUND.into(),
            version: "test".into(),
        },
        benchmark_valid: false,
        sealed: false,
        replayable: false,
        provenance: Provenance {
            source: "test".into(),
            run_id: "run".into(),
            git_sha: UNBOUND.into(),
            bundle_id: UNBOUND.into(),
            record_sha256: sha256_hex(&blob),
        },
        prev_event_sha256: None,
        improvement_number: 0,
        ts_unix: 1,
    };
    (record, blob)
}
