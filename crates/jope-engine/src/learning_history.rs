//! Read-only, restartable loss history. The trainer's saved logs are authoritative;
//! the observer cache only avoids reparsing unchanged files.
use serde::Deserialize;
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::{BufRead, BufReader, Seek, SeekFrom},
    os::unix::fs::MetadataExt,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc,
    },
    thread,
    time::{Duration, SystemTime},
};

#[derive(Clone, Debug, Deserialize)]
pub struct Validation {
    pub sequence_nll: f64,
    pub ranking_bt: f64,
    pub evaluated_step: Option<u64>,
    pub panel_sha256: Option<String>,
}
#[derive(Deserialize)]
struct Metric {
    schema: String,
    rank: u64,
    run: PathBuf,
    step: u64,
    timestamp: u64,
    corpus_sha256: String,
    sequence_nll: f64,
    validation: Option<Validation>,
    initialization: Option<Value>,
    #[serde(default)]
    invalidated: bool,
    #[serde(default)]
    batch_qualification: bool,
}
#[derive(Clone, Debug)]
pub struct Point {
    pub step: u64,
    pub timestamp: u64,
    pub train: f64,
    pub validation: Option<Validation>,
    pub attempt: usize,
}
#[derive(Clone, Debug, Default)]
pub struct Run {
    pub path: PathBuf,
    pub points: Vec<Point>,
    pub checkpoint: String,
    pub parent: String,
    pub corpora: BTreeSet<String>,
    pub status: String,
    pub malformed: usize,
    pub unbound_validation: usize,
    pub pending_line: bool,
    pub read_error: Option<String>,
    pub attempts: usize,
}
#[derive(Clone, Debug, Default)]
pub struct Snapshot {
    pub runs: Vec<Run>,
    pub ready: bool,
    pub errors: Vec<String>,
}
impl Snapshot {
    pub fn report(&self) -> Value {
        serde_json::json!({"schema":"parallel.learning_history.v1","read_only":true,"ready":self.ready,
            "records":self.runs.iter().map(|r|r.points.len()).sum::<usize>(),"errors":self.errors,
            "scope":"All persisted campaign, auxiliary and standalone training records; branches and restarts retained. Diagnostic batch runs excluded.",
            "runs":self.runs.iter().map(|r|serde_json::json!({"path":r.path,"status":r.status,
                "records":r.points.len(),"attempts":r.attempts,"parent":r.parent,"checkpoint":r.checkpoint,
                "first_timestamp":r.points.first().map(|p|p.timestamp),"last_timestamp":r.points.last().map(|p|p.timestamp),
                "first_step":r.points.first().map(|p|p.step),"last_step":r.points.last().map(|p|p.step),
                "first_training_nll":r.points.first().map(|p|p.train),"last_training_nll":r.points.last().map(|p|p.train),
                "bound_validation_points":r.points.iter().filter(|p|p.validation.is_some()).count(),
                "corpora":r.corpora,"malformed_records":r.malformed,"unbound_validation_rows":r.unbound_validation,
                "incomplete_final_line":r.pending_line,"read_error":r.read_error})).collect::<Vec<_>>()})
    }
}
#[derive(Default)]
struct FileCache {
    run: Run,
    offset: u64,
    inode: Option<(u64, u64)>,
    modified: Option<SystemTime>,
    length: u64,
}
#[derive(Default)]
pub struct Cache {
    files: BTreeMap<PathBuf, FileCache>,
}

impl FileCache {
    fn refresh(&mut self, root: &Path, path: &Path, live: &Value) {
        self.run.path = path.to_owned();
        self.run.read_error = None;
        let result = self.read(root, path);
        if let Err(error) = result {
            self.run.read_error = Some(error.to_string());
        }
        self.run.checkpoint = json(&path.join("latest.json"))["checkpoint"]
            .as_str()
            .unwrap_or("")
            .to_owned();
        self.run.status = if path.join("invalidated.json").exists() {
            "INVALID"
        }
        // Resumed auxiliary blocks retain the preceding block's completion receipt.
        else if live["phase"] == "training on real GP teaching corpus"
            && live["run"].as_str().is_some_and(|p| root.join(p) == path)
            && live["timestamp"]
                .as_u64()
                .is_some_and(|t| t <= crate::telemetry::now() && crate::telemetry::now() - t <= 5)
        {
            "LIVE"
        } else if path.join("complete.json").exists() {
            "SAVED"
        } else {
            "PAUSED"
        }
        .into();
    }
    fn read(&mut self, root: &Path, path: &Path) -> std::io::Result<()> {
        let file_path = path.join("metrics.jsonl");
        if file_path.is_symlink() {
            return Err(std::io::Error::other("Indirect metric log rejected"));
        }
        let mut file = File::open(file_path)?;
        let meta = file.metadata()?;
        let inode = (meta.dev(), meta.ino());
        let modified = meta.modified().ok();
        if self.inode != Some(inode)
            || meta.len() < self.length
            || (meta.len() == self.length && modified != self.modified)
        {
            *self = Self::default();
            self.run.path = path.to_owned();
        }
        self.inode = Some(inode);
        if meta.len() == self.length && modified == self.modified {
            return Ok(());
        }
        file.seek(SeekFrom::Start(self.offset))?;
        let mut reader = BufReader::new(file);
        let mut bytes = Vec::new();
        self.run.pending_line = false;
        loop {
            bytes.clear();
            let count = reader.read_until(b'\n', &mut bytes)?;
            if count == 0 {
                break;
            }
            if bytes.last() != Some(&b'\n') {
                self.run.pending_line = true;
                break;
            }
            self.offset += count as u64;
            let metric = match serde_json::from_slice::<Metric>(&bytes) {
                Ok(metric) => metric,
                Err(_) => {
                    self.run.malformed += 1;
                    continue;
                }
            };
            if metric.schema != "parallel.neural_training.v1"
                || metric.rank != 0
                || metric.invalidated
                || metric.batch_qualification
            {
                continue;
            }
            if root.join(&metric.run) != path
                || metric.corpus_sha256.is_empty()
                || metric.timestamp == 0
                || metric.step == 0
                || !metric.sequence_nll.is_finite()
                || metric.sequence_nll < 0.
            {
                self.run.malformed += 1;
                continue;
            }
            if self.run.points.last().is_none_or(|p| metric.step <= p.step) {
                self.run.attempts += 1;
            }
            self.run.corpora.insert(metric.corpus_sha256);
            if self.run.parent.is_empty() {
                if let Some(initialization) = metric.initialization {
                    self.run.parent = initialization["checkpoint"]
                        .as_str()
                        .or_else(|| initialization["kind"].as_str())
                        .unwrap_or("unknown")
                        .to_owned();
                }
            }
            let validation = metric.validation.filter(|v| {
                if v.evaluated_step.is_none()
                    || v.panel_sha256.as_ref().is_none_or(String::is_empty)
                {
                    self.run.unbound_validation += 1;
                    return false;
                }
                if !v.sequence_nll.is_finite()
                    || v.sequence_nll < 0.
                    || !v.ranking_bt.is_finite()
                    || v.ranking_bt < 0.
                {
                    self.run.malformed += 1;
                    return false;
                }
                // A checkpoint value repeated on later optimizer rows is one observation.
                v.evaluated_step == Some(metric.step)
            });
            self.run.points.push(Point {
                step: metric.step,
                timestamp: metric.timestamp,
                train: metric.sequence_nll,
                validation,
                attempt: self.run.attempts,
            });
        }
        self.length = meta.len();
        self.modified = modified;
        Ok(())
    }
}
impl Cache {
    pub fn refresh(&mut self, root: &Path) -> Snapshot {
        let root = match root.canonicalize() {
            Ok(root) => root,
            Err(error) => {
                return Snapshot {
                    ready: true,
                    errors: vec![error.to_string()],
                    ..Default::default()
                }
            }
        };
        let (paths, mut errors) = discover(&root);
        let live = json(&root.join("runs/neural-service.json"));
        for path in &paths {
            self.files
                .entry(path.clone())
                .or_default()
                .refresh(&root, path, &live);
        }
        // Disappearance does not erase already observed history from a live TUI.
        for (path, cached) in &mut self.files {
            if !paths.contains(path) {
                cached.run.read_error =
                    Some("Run disappeared; displaying last saved observation".into());
            }
        }
        let mut runs = self
            .files
            .values()
            .map(|c| c.run.clone())
            .collect::<Vec<_>>();
        runs.retain(|r| !r.points.is_empty() || !r.checkpoint.is_empty() || r.read_error.is_some());
        runs.sort_by(|a, b| {
            (a.points.first().map(|p| p.timestamp).unwrap_or(0), &a.path)
                .cmp(&(b.points.first().map(|p| p.timestamp).unwrap_or(0), &b.path))
        });
        for run in &runs {
            if let Some(error) = &run.read_error {
                errors.push(format!(
                    "{}: {error}",
                    run.path.join("metrics.jsonl").display()
                ));
            }
        }
        Snapshot {
            runs,
            ready: true,
            errors,
        }
    }
}

pub struct Watcher {
    receiver: mpsc::Receiver<Arc<Snapshot>>,
    stop: Arc<AtomicBool>,
}
impl Watcher {
    pub fn start(root: PathBuf) -> Self {
        let (sender, receiver) = mpsc::sync_channel(1);
        let stop = Arc::new(AtomicBool::new(false));
        let ending = stop.clone();
        thread::spawn(move || {
            let mut cache = Cache::default();
            while !ending.load(Ordering::Relaxed) {
                let snapshot = Arc::new(cache.refresh(&root));
                if let Err(mpsc::TrySendError::Disconnected(_)) = sender.try_send(snapshot) {
                    break;
                }
                for _ in 0..50 {
                    if ending.load(Ordering::Relaxed) {
                        return;
                    }
                    thread::sleep(Duration::from_millis(100));
                }
            }
        });
        Self { receiver, stop }
    }
    pub fn latest(&self) -> Option<Arc<Snapshot>> {
        self.receiver.try_iter().last()
    }
}
impl Drop for Watcher {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

#[derive(Default)]
pub struct Navigation {
    pub selected: Option<PathBuf>,
}
impl Navigation {
    pub fn move_by(&mut self, snapshot: &Snapshot, delta: isize) {
        if snapshot.runs.is_empty() {
            return;
        }
        let current = self
            .selected
            .as_ref()
            .and_then(|path| snapshot.runs.iter().position(|r| &r.path == path));
        let index = match current {
            Some(i) => i.saturating_add_signed(delta).min(snapshot.runs.len() - 1),
            None if delta < 0 => snapshot.runs.len() - 1,
            None => 0,
        };
        self.selected = Some(snapshot.runs[index].path.clone());
    }
    pub fn edge(&mut self, snapshot: &Snapshot, last: bool) {
        self.selected = if last {
            snapshot.runs.last()
        } else {
            snapshot.runs.first()
        }
        .map(|r| r.path.clone());
    }
}
mod chart;
pub use chart::{column_line, envelope, latest_summary, moving_average, render, trend_row};

mod discovery;
pub(super) use discovery::*;
