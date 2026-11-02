use super::*;

pub type RetainedWork = (SupervisorHandle, Liveness, Option<(Vec<u8>, Vec<u8>)>);

/// Deterministic in-memory supervisor for state-machine tests. The shared state lets the
/// test change liveness and retained results after the coordinator owns the supervisor.
#[derive(Default)]
pub struct FakeState {
    pub work: BTreeMap<String, RetainedWork>,
    pub launches: u32,
    pub terminations: u32,
}
#[derive(Clone, Default)]
pub struct FakeSupervisor(pub Arc<Mutex<FakeState>>);
impl FakeSupervisor {
    pub fn set(&self, dispatch_id: &str, l: Liveness, retained: Option<(Vec<u8>, Vec<u8>)>) {
        let mut s = self.0.lock().unwrap();
        let w = s.work.get_mut(dispatch_id).expect("launched work");
        w.1 = l;
        w.2 = retained;
    }
    /// Work started out of band (a crash between spawn and the launch record).
    pub fn inject(&self, dispatch_id: &str) {
        let mut s = self.0.lock().unwrap();
        s.work.insert(
            dispatch_id.into(),
            (
                SupervisorHandle {
                    supervisor: "fake".into(),
                    identity: dispatch_id.into(),
                },
                Liveness::Running,
                None,
            ),
        );
    }
    pub fn running(&self) -> usize {
        self.0
            .lock()
            .unwrap()
            .work
            .values()
            .filter(|w| w.1 == Liveness::Running)
            .count()
    }
}
impl Supervisor for FakeSupervisor {
    fn launch(&mut self, l: &RankLaunch<'_>) -> anyhow::Result<SupervisorHandle> {
        let mut s = self.0.lock().unwrap();
        anyhow::ensure!(!s.work.contains_key(l.dispatch_id), "duplicate launch");
        s.launches += 1;
        let h = SupervisorHandle {
            supervisor: "fake".into(),
            identity: l.dispatch_id.into(),
        };
        s.work
            .insert(l.dispatch_id.into(), (h.clone(), Liveness::Running, None));
        Ok(h)
    }
    fn find(&mut self, dispatch_id: &str) -> anyhow::Result<Option<SupervisorHandle>> {
        Ok(self
            .0
            .lock()
            .unwrap()
            .work
            .get(dispatch_id)
            .map(|w| w.0.clone()))
    }
    fn probe(&mut self, h: &SupervisorHandle) -> anyhow::Result<Liveness> {
        Ok(self
            .0
            .lock()
            .unwrap()
            .work
            .get(&h.identity)
            .map(|w| w.1.clone())
            .unwrap_or(Liveness::Gone))
    }
    fn terminate(&mut self, h: &SupervisorHandle) -> anyhow::Result<Liveness> {
        let mut s = self.0.lock().unwrap();
        s.terminations += 1;
        let w = s.work.get_mut(&h.identity).expect("work");
        if w.1 == Liveness::Running {
            w.1 = Liveness::Exited {
                code: None,
                signal: Some(15),
            };
        }
        Ok(w.1.clone())
    }
    fn retained_result(
        &mut self,
        h: &SupervisorHandle,
    ) -> anyhow::Result<Option<(Vec<u8>, Vec<u8>)>> {
        Ok(self
            .0
            .lock()
            .unwrap()
            .work
            .get(&h.identity)
            .and_then(|w| w.2.clone()))
    }
}

/// Snapshot of every path, type, length, mode, mtime and content digest under `root`.
pub fn tree(root: &Path) -> BTreeMap<PathBuf, String> {
    let mut out = BTreeMap::new();
    for e in walkdir::WalkDir::new(root).sort_by_file_name() {
        let e = e.unwrap();
        let m = fs::symlink_metadata(e.path()).unwrap();
        use std::os::unix::fs::MetadataExt;
        let content = if m.is_file() {
            sha(&fs::read(e.path()).unwrap())
        } else {
            String::new()
        };
        out.insert(
            e.path().strip_prefix(root).unwrap().to_path_buf(),
            format!(
                "{:?}:{}:{:o}:{}.{}:{content}",
                m.file_type(),
                m.len(),
                m.mode(),
                m.mtime(),
                m.mtime_nsec()
            ),
        );
    }
    out
}
