use super::*;
use std::sync::{Arc, Mutex};
use tempfile::tempdir;

#[derive(Default)]
struct MockSchedulerState {
    jobs: Vec<SchedulerJob>,
    jobs_calls: usize,
    held: Vec<String>,
    cancelled: Vec<String>,
    race_on_jobs_call: Option<usize>,
    race_state: Option<String>,
}

#[derive(Clone)]
struct MockScheduler {
    state: Arc<Mutex<MockSchedulerState>>,
}

impl Scheduler for MockScheduler {
    fn jobs(&self) -> Result<Vec<SchedulerJob>> {
        let mut state = self.state.lock().unwrap();
        state.jobs_calls += 1;
        if state.race_on_jobs_call == Some(state.jobs_calls) {
            let raced = state
                .race_state
                .clone()
                .unwrap_or_else(|| "running".to_owned());
            for job in &mut state.jobs {
                job.state = raced.clone();
            }
        }
        Ok(state.jobs.clone())
    }

    fn hold(&self, job_id: &str) -> Result<()> {
        let mut state = self.state.lock().unwrap();
        let job = state
            .jobs
            .iter_mut()
            .find(|job| job.id == job_id)
            .context("mock hold job is absent")?;
        if matches!(job.state.as_str(), "queued" | "starting") {
            job.state = "held".to_owned();
        }
        state.held.push(job_id.to_owned());
        Ok(())
    }

    fn cancel(&self, job_id: &str) -> Result<()> {
        let mut state = self.state.lock().unwrap();
        let job = state
            .jobs
            .iter_mut()
            .find(|job| job.id == job_id)
            .context("mock cancel job is absent")?;
        if job.state != "held" {
            bail!("mock refuses to cancel a non-held job");
        }
        job.state = "cancelled".to_owned();
        state.cancelled.push(job_id.to_owned());
        Ok(())
    }
}

struct RetirementFixture {
    _temporary: tempfile::TempDir,
    supervisor: Supervisor,
    scheduler: Arc<Mutex<MockSchedulerState>>,
    registry: PathBuf,
    backlog: PathBuf,
}

fn retirement_fixture(scheduler_state: &str, project: &str) -> RetirementFixture {
    let temporary = tempdir().unwrap();
    let root = temporary.path().join("root");
    fs::create_dir_all(root.join("ops")).unwrap();
    fs::create_dir_all(root.join("manifests")).unwrap();
    let old_checkpoint = "a".repeat(64);
    let manifest = root.join("manifests/stale.json");
    atomic_json(
        &manifest,
        &json!({
            "schema": "JopeFlywheelCorpusEvalShard.v1",
            "checkpoint_epoch": format!("checkpoint-{old_checkpoint}"),
            "candidates": []
        }),
    )
    .unwrap();
    let manifest_sha256 = sha256_file(&manifest).unwrap();
    let backlog = root.join("ops/backlog.sqlite");
    let connection = Connection::open(&backlog).unwrap();
    connection
        .execute_batch(
            "CREATE TABLE claims(
                   manifest_sha256 TEXT NOT NULL,
                   manifest TEXT NOT NULL,
                   job_id TEXT PRIMARY KEY,
                   state TEXT NOT NULL,
                   reservation_owner TEXT,
                   updated_at_ms INTEGER NOT NULL
                 );",
        )
        .unwrap();
    connection
        .execute(
            "INSERT INTO claims VALUES(?1,?2,'job-stale','submitted','owner',1)",
            params![manifest_sha256, manifest.to_string_lossy()],
        )
        .unwrap();
    drop(connection);

    let registry = root.join("ops/pair-registry.sqlite");
    let connection = Connection::open(&registry).unwrap();
    connection
        .execute_batch(
            "CREATE TABLE pairs(
                   task TEXT,dataset_hash TEXT,semantic_sha256 TEXT,protocol_sha256 TEXT,
                   canonical_genome TEXT,state TEXT,reservation_owner TEXT,manifest_path TEXT,
                   manifest_sha256 TEXT,job_id TEXT,receipt_sha256 TEXT,failure_class TEXT,
                   infrastructure_retries INTEGER,created_at_ms INTEGER,updated_at_ms INTEGER,
                   PRIMARY KEY(task,dataset_hash,semantic_sha256,protocol_sha256)
                 );
                 CREATE TABLE attempts(
                   id INTEGER PRIMARY KEY,task TEXT,dataset_hash TEXT,semantic_sha256 TEXT,
                   protocol_sha256 TEXT,attempt_no INTEGER,state TEXT,reservation_owner TEXT,
                   manifest_path TEXT,manifest_sha256 TEXT,job_id TEXT,receipt_sha256 TEXT,
                   failure_class TEXT,detail_json TEXT,created_at_ms INTEGER,finalized_at_ms INTEGER
                 );
                 CREATE TABLE events(
                   id INTEGER PRIMARY KEY,kind TEXT,task TEXT,dataset_hash TEXT,
                   semantic_sha256 TEXT,protocol_sha256 TEXT,detail_json TEXT,created_at_ms INTEGER
                 );",
        )
        .unwrap();
    connection
        .execute(
            "INSERT INTO pairs VALUES(
                   'regression','dataset','semantic','protocol','genome','submitted','owner',
                   ?1,?2,'job-stale',NULL,NULL,0,1,1
                 )",
            params![manifest.to_string_lossy(), manifest_sha256],
        )
        .unwrap();
    connection
        .execute(
            "INSERT INTO attempts VALUES(
                   NULL,'regression','dataset','semantic','protocol',1,'submitted','owner',
                   ?1,?2,'job-stale',NULL,NULL,'{}',1,NULL
                 )",
            params![manifest.to_string_lossy(), manifest_sha256],
        )
        .unwrap();
    drop(connection);

    let scheduler = Arc::new(Mutex::new(MockSchedulerState {
        jobs: vec![SchedulerJob {
            id: "job-stale".to_owned(),
            state: scheduler_state.to_owned(),
            project: Some(project.to_owned()),
            ..SchedulerJob::default()
        }],
        ..MockSchedulerState::default()
    }));
    let mut supervisor = Supervisor::open(root, PathBuf::from("/not-used-in-this-test")).unwrap();
    supervisor.scheduler = Box::new(MockScheduler {
        state: scheduler.clone(),
    });
    let active = "b".repeat(64);
    supervisor.state.active_checkpoint = active.clone();
    supervisor.state.active_epoch = format!("checkpoint-{active}");
    RetirementFixture {
        _temporary: temporary,
        supervisor,
        scheduler,
        registry,
        backlog,
    }
}

#[path = "tests/identity.rs"]
mod identity;
#[path = "tests/retirement.rs"]
mod retirement;
