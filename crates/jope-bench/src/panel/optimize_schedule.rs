//! Controller-only rendezvous. One supervisor owns the pool; a bounded wave of
//! dataset controllers submits metadata batches. Existing pool round-robin then
//! interleaves actual fits across every dataset waiting in that wave.
use anyhow::{anyhow, ensure, Result};
use jope_pipeline::{
    execution::{ArmBudget, WorkerJob, WorkerOutcome},
    neural_search::PoolAccess,
};
use std::{
    collections::BTreeMap,
    sync::{
        atomic::AtomicBool,
        mpsc::{self, Sender},
    },
};

pub const POLICY: &str = "asm.shared-pool/dataset-wave-rendezvous-round-robin/quantum20/v1";
enum Message<T> {
    Arm(ArmBudget, Sender<std::result::Result<u64, String>>),
    Batch(
        usize,
        Vec<WorkerJob>,
        Sender<std::result::Result<Vec<WorkerOutcome>, String>>,
    ),
    Done(usize, std::result::Result<T, String>),
}
pub struct Proxy<T> {
    id: usize,
    send: Sender<Message<T>>,
}
impl<T: Send> PoolAccess for Proxy<T> {
    fn begin_arm(&mut self, arm: &ArmBudget) -> Result<u64> {
        let (send, recv) = mpsc::channel();
        self.send
            .send(Message::Arm(arm.clone(), send))
            .map_err(|_| anyhow!("pool supervisor ended"))?;
        recv.recv()
            .map_err(|_| anyhow!("pool supervisor ended"))?
            .map_err(anyhow::Error::msg)
    }
    fn run(&mut self, jobs: Vec<WorkerJob>, _cancelled: &AtomicBool) -> Result<Vec<WorkerOutcome>> {
        ensure!(
            jobs.len() <= 5000,
            "dataset exceeded bounded queue metadata"
        );
        let (send, recv) = mpsc::channel();
        self.send
            .send(Message::Batch(self.id, jobs, send))
            .map_err(|_| anyhow!("pool supervisor ended"))?;
        recv.recv()
            .map_err(|_| anyhow!("pool supervisor ended"))?
            .map_err(anyhow::Error::msg)
    }
}
struct Pending {
    jobs: std::collections::VecDeque<WorkerJob>,
    outcomes: Vec<WorkerOutcome>,
    reply: Sender<std::result::Result<Vec<WorkerOutcome>, String>>,
}
// A completed encoding batch must be returned before unrelated populations
// drain. At most twenty ready jobs enter a quantum, not the whole wave backlog.
fn quantum(
    pending: &mut BTreeMap<usize, Pending>,
    cursor: &mut usize,
) -> Result<(Vec<WorkerJob>, BTreeMap<String, usize>)> {
    let mut jobs = Vec::new();
    let mut owner = BTreeMap::new();
    let keys = pending.keys().copied().collect::<Vec<_>>();
    while jobs.len() < 20 && pending.values().any(|p| !p.jobs.is_empty()) {
        let id = keys[*cursor % keys.len()];
        *cursor = cursor.wrapping_add(1);
        if let Some(job) = pending
            .get_mut(&id)
            .expect("pending dataset")
            .jobs
            .pop_front()
        {
            ensure!(
                owner.insert(job.request.identity()?, id).is_none(),
                "two dataset controllers requested same fit"
            );
            jobs.push(job);
        }
    }
    Ok((jobs, owner))
}
/// All wave members reach their next numerical boundary before admission. Thus
/// no worker process waits on a controller and no dataset monopolizes the batch.
/// Wave membership/order/width is frozen in the campaign configuration.
pub fn wave<T: Send, F, P: PoolAccess>(
    pool: &mut P,
    cancel: &AtomicBool,
    tasks: Vec<F>,
) -> Result<Vec<Result<T>>>
where
    F: FnOnce(&mut Proxy<T>) -> Result<T> + Send,
{
    ensure!(
        !tasks.is_empty() && tasks.len() <= 20,
        "controller wave must contain1..20 datasets"
    );
    let n = tasks.len();
    std::thread::scope(|scope| {
        let (send, recv) = mpsc::channel();
        for (id, task) in tasks.into_iter().enumerate() {
            let send = send.clone();
            scope.spawn(move || {
                let mut proxy = Proxy {
                    id,
                    send: send.clone(),
                };
                let result =
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| task(&mut proxy)))
                        .map_err(|_| {
                            "dataset controller panicked; no replacement fit claimed".to_owned()
                        })
                        .and_then(|r| r.map_err(|e| format!("{e:#}")));
                let _ = send.send(Message::Done(id, result));
            });
        }
        drop(send);
        let mut results = (0..n).map(|_| None).collect::<Vec<_>>();
        let mut pending: BTreeMap<usize, Pending> = BTreeMap::new();
        let mut cursor = 0usize;
        while results.iter().any(Option::is_none) {
            match recv
                .recv()
                .map_err(|_| anyhow!("dataset controllers ended without inventory"))?
            {
                Message::Arm(arm, reply) => {
                    let _ = reply.send(pool.begin_arm(&arm).map_err(|e| format!("{e:#}")));
                }
                Message::Batch(id, jobs, reply) => {
                    ensure!(
                        results[id].is_none() && !pending.contains_key(&id),
                        "duplicate queued dataset batch"
                    );
                    if jobs.is_empty() {
                        let _ = reply.send(Ok(vec![]));
                    } else {
                        pending.insert(
                            id,
                            Pending {
                                jobs: jobs.into(),
                                outcomes: vec![],
                                reply,
                            },
                        );
                    }
                }
                Message::Done(id, result) => {
                    ensure!(results[id].is_none(), "duplicate controller result");
                    results[id] = Some(result);
                }
            }
            let active = results.iter().filter(|r| r.is_none()).count();
            while active > 0 && pending.len() == active {
                let (jobs, mut owner) = quantum(&mut pending, &mut cursor)?;
                match pool.run(jobs, cancel) {
                    Ok(outcomes) => {
                        for outcome in outcomes {
                            let id = owner
                                .remove(&outcome.request_sha256)
                                .ok_or_else(|| anyhow!("pool returned unrequested fit"))?;
                            pending
                                .get_mut(&id)
                                .expect("pending owner")
                                .outcomes
                                .push(outcome);
                        }
                        ensure!(owner.is_empty(), "pool omitted queued fit outcomes");
                        let done = pending
                            .iter()
                            .filter_map(|(id, p)| p.jobs.is_empty().then_some(*id))
                            .collect::<Vec<_>>();
                        for id in done {
                            let p = pending.remove(&id).expect("finished pending batch");
                            let _ = p.reply.send(Ok(p.outcomes));
                        }
                    }
                    Err(error) => {
                        let reason = format!("{error:#}");
                        for (_, p) in std::mem::take(&mut pending) {
                            let _ = p.reply.send(Err(reason.clone()));
                        }
                    }
                }
            }
        }
        Ok(results
            .into_iter()
            .map(|r| {
                r.expect("all controller inventory")
                    .map_err(anyhow::Error::msg)
            })
            .collect())
    })
}

/// Original manifest order is retained inside each task category, interleaved
/// across categories so a late global deadline cannot omit a whole category first.
pub fn dataset_order(tasks: &[jope_datasets::catalog::Task]) -> Vec<usize> {
    let mut queues: BTreeMap<&str, std::collections::VecDeque<usize>> = BTreeMap::new();
    for (i, task) in tasks.iter().enumerate() {
        queues.entry(task.as_str()).or_default().push_back(i);
    }
    let mut order = Vec::new();
    while order.len() < tasks.len() {
        for queue in queues.values_mut() {
            if let Some(i) = queue.pop_front() {
                order.push(i);
            }
        }
    }
    order
}
#[cfg(test)]
mod tests {
    #[test]
    fn frozen_category_round_robin_includes_every_dataset_once() {
        use jope_datasets::catalog::Task::*;
        let order = super::dataset_order(&[
            Regression, Regression, Binary, Binary, Multiclass, Multiclass,
        ]);
        assert_eq!(order, vec![2, 4, 0, 3, 5, 1]);
    }
    fn job(dataset: usize, index: usize) -> super::WorkerJob {
        use jope_pipeline::{
            execution::{Artifact, WorkerCommand},
            experiment::{EvaluationRequest, EvaluationStage, Metric, REQUEST_SCHEMA},
        };
        let request = EvaluationRequest {
            schema: REQUEST_SCHEMA.into(),
            experiment_sha256: "a".repeat(64),
            dataset_sha256: jope_pipeline::sha256_hex(format!("dataset{dataset}").as_bytes()),
            split_sha256: "b".repeat(64),
            fitting_rows_sha256: "c".repeat(64),
            scoring_rows_sha256: "d".repeat(64),
            fitting_rows: 10,
            scoring_rows: 2,
            recipe_sha256: jope_pipeline::sha256_hex(format!("recipe{index}").as_bytes()),
            evaluator_sha256: "e".repeat(64),
            checkpoint_sha256: None,
            task: jope_datasets::catalog::Task::Regression,
            metric: Metric::Rmse,
            classes: vec![],
            fold: 0,
            stage: EvaluationStage::Diagnostic,
            seed: 1,
            numerical_policy: "synthetic-queue-fixture".into(),
        };
        super::WorkerJob {
            directory: std::path::PathBuf::from("/synthetic/jobs")
                .join(request.identity().unwrap()),
            request,
            command: WorkerCommand {
                program: Artifact {
                    path: "/synthetic/not-executed".into(),
                    sha256: "f".repeat(64),
                },
                args: vec![],
                inputs: vec![],
                environment: Default::default(),
            },
            outputs: vec!["output.json".into()],
            arm: None,
        }
    }
    #[test]
    fn small_encoding_batch_returns_before_large_population_drains() -> anyhow::Result<()> {
        use super::{ArmBudget, PoolAccess, WorkerJob, WorkerOutcome};
        use jope_pipeline::{execution::ObservedConsumption, experiment::EvaluationStatus};
        struct FixturePool {
            batches: Vec<Vec<(String, String)>>,
        }
        impl PoolAccess for FixturePool {
            fn begin_arm(&mut self, _: &ArmBudget) -> anyhow::Result<u64> {
                Ok(120_000)
            }
            fn run(
                &mut self,
                jobs: Vec<WorkerJob>,
                _: &std::sync::atomic::AtomicBool,
            ) -> anyhow::Result<Vec<WorkerOutcome>> {
                self.batches.push(
                    jobs.iter()
                        .map(|j| {
                            (
                                j.request.dataset_sha256.clone(),
                                j.request.recipe_sha256.clone(),
                            )
                        })
                        .collect(),
                );
                jobs.into_iter()
                    .map(|j| {
                        Ok(WorkerOutcome {
                            schema: "jope.worker-execution/v2".into(),
                            job_sha256: j.identity()?,
                            request_sha256: j.request.identity()?,
                            command_status: EvaluationStatus::BudgetExhausted,
                            reason: Some("synthetic queue test; no process executed".into()),
                            exit_code: None,
                            signal: None,
                            outputs: vec![],
                            stdout: None,
                            stderr: None,
                            consumption: ObservedConsumption::default(),
                        })
                    })
                    .collect()
            }
        }
        let mut pool = FixturePool { batches: vec![] };
        let tasks = (0..2)
            .map(|dataset| {
                move |proxy: &mut super::Proxy<usize>| {
                    proxy.begin_arm(&ArmBudget {
                        id_sha256: jope_pipeline::sha256_hex(format!("arm{dataset}").as_bytes()),
                        seconds: 120,
                    })?;
                    if dataset == 0 {
                        assert_eq!(
                            proxy
                                .run(vec![job(0, 0)], &std::sync::atomic::AtomicBool::new(false))?
                                .len(),
                            1
                        );
                        assert_eq!(
                            proxy
                                .run(vec![job(0, 1)], &std::sync::atomic::AtomicBool::new(false))?
                                .len(),
                            1
                        );
                        Ok(2)
                    } else {
                        Ok(proxy
                            .run(
                                (0..60).map(|i| job(1, i)).collect(),
                                &std::sync::atomic::AtomicBool::new(false),
                            )?
                            .len())
                    }
                }
            })
            .collect();
        let results = super::wave(&mut pool, &std::sync::atomic::AtomicBool::new(false), tasks)?;
        assert_eq!(
            results.into_iter().collect::<anyhow::Result<Vec<_>>>()?,
            vec![2, 60]
        );
        assert!(pool.batches.iter().all(|b| b.len() <= 20));
        assert_eq!(pool.batches.iter().map(Vec::len).sum::<usize>(), 62);
        let proposal = job(0, 1);
        let target = (
            &proposal.request.dataset_sha256,
            &proposal.request.recipe_sha256,
        );
        let position = pool
            .batches
            .iter()
            .position(|b| b.iter().any(|(d, r)| (d, r) == target))
            .unwrap();
        assert_eq!(
            position, 1,
            "completed encoder was held until unrelated population drained"
        );
        Ok(())
    }
}
