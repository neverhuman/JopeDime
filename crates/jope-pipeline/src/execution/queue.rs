use super::*;

#[derive(Default)]
pub(super) struct FairQueue {
    datasets: VecDeque<String>,
    jobs: BTreeMap<String, VecDeque<WorkerJob>>,
}
impl FairQueue {
    pub(super) fn push(&mut self, dataset: String, job: WorkerJob) {
        if !self.jobs.contains_key(&dataset) {
            self.datasets.push_back(dataset.clone());
        }
        self.jobs.entry(dataset).or_default().push_back(job);
    }
    pub(super) fn pop(&mut self) -> Option<WorkerJob> {
        let dataset = self.datasets.pop_front()?;
        let jobs = self.jobs.get_mut(&dataset)?;
        let job = jobs.pop_front();
        if jobs.is_empty() {
            self.jobs.remove(&dataset);
        } else {
            self.datasets.push_back(dataset);
        }
        job
    }
    pub(super) fn is_empty(&self) -> bool {
        self.datasets.is_empty()
    }
}

pub(super) struct ActiveFits(pub(super) Vec<Running>);
impl std::ops::Deref for ActiveFits {
    type Target = Vec<Running>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl std::ops::DerefMut for ActiveFits {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
impl Drop for ActiveFits {
    fn drop(&mut self) {
        // Signal every watcher before joining any one of them on error paths.
        for running in &self.0 {
            running.stop.store(true, Ordering::Release);
        }
    }
}
