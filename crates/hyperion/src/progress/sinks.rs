/// One per-cell confidence update from the encoding sweep, serialized compactly as
/// `[row, col, confidence]` (tuple struct → JSON array, ~60% smaller than an object).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SweepCell(pub u32, pub u32, pub f32);

/// One live frame from the phase-0 HYPERION composite sweep — a SIDE-CHANNEL type, deliberately NOT
/// part of the governed [`Event`] contract (no schema-version bump; the web nests it under a
/// `"sweep"` payload key, mirroring the `reasoning.note` precedent).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SweepFrame {
    /// Current sweep index k (0-based).
    pub sweep: u32,
    /// Total sweeps K.
    pub sweeps: u32,
    /// Fraction of THIS sweep completed.
    pub frac: f64,
    /// Overall encode progress in \[0,1\] (rescaled across the train/hold share) — drives the UI %.
    pub done_frac: f64,
    /// Original matrix dims (target column dropped).
    pub rows: u64,
    pub cols: u64,
    /// Coordinate space of `cells` (== rows/cols when small; UI-grid-binned for large matrices).
    pub grid_rows: u32,
    pub grid_cols: u32,
    /// This chunk's running-mean confidence updates at ORIGINAL (or grid-binned) positions, in \[0,1\].
    pub cells: Vec<SweepCell>,
    /// Current running 1×128 dataset vector = mean over rows of the running row embeddings.
    pub vector: Vec<f32>,
    /// Seconds since pipeline start.
    pub t: f64,
}

/// A progress sink. Must be Send+Sync (held by the driver across phases).
pub trait Sink: Send + Sync {
    fn emit(&self, e: &Event);
    /// Cooperative cancellation hook. The pipeline checks this at phase/generation boundaries and
    /// halts promptly. Defaults to `false` so existing sinks are unaffected.
    fn is_cancelled(&self) -> bool {
        false
    }
    /// Cooperative one-shot phase advance hook. A sink returns true when the current long-running
    /// phase should stop at its next safe boundary and let the pipeline continue with partial/best
    /// results. This is intentionally separate from hard cancellation.
    fn take_advance_request(&self, _phase: u8) -> bool {
        false
    }
    /// Phase-0 encode side-channel: per-chunk sweep frames from the HYPERION composite sweep.
    /// Default no-op so existing sinks are unaffected. NOTE: wrapper sinks (e.g. the report tee)
    /// MUST forward this explicitly or frames are silently dropped.
    fn emit_sweep(&self, _f: &SweepFrame) {}
}

/// Discards events.
pub struct NullSink;
impl Sink for NullSink {
    fn emit(&self, _e: &Event) {}
}

pub struct RecordingSink<'a> {
    inner: &'a dyn Sink,
    events: Mutex<Vec<Event>>,
}

impl<'a> RecordingSink<'a> {
    pub fn new(inner: &'a dyn Sink) -> Self {
        Self {
            inner,
            events: Mutex::new(Vec::new()),
        }
    }

    pub fn events(&self) -> Vec<Event> {
        self.events
            .lock()
            .expect("progress recorder poisoned")
            .clone()
    }
}

impl Sink for RecordingSink<'_> {
    fn emit(&self, e: &Event) {
        self.events
            .lock()
            .expect("progress recorder poisoned")
            .push(e.clone());
        self.inner.emit(e);
    }

    fn is_cancelled(&self) -> bool {
        self.inner.is_cancelled()
    }

    fn take_advance_request(&self, phase: u8) -> bool {
        self.inner.take_advance_request(phase)
    }

    fn emit_sweep(&self, f: &SweepFrame) {
        self.inner.emit_sweep(f);
    }
}

/// Machine-readable sink: serializes each `Event` as one JSON line (JSONL). Used by the SageMaker
/// container path (`JAIN_PROGRESS=jsonl`). The `json diagnostics` stream lets
/// log consumers parse typed stage, ETA, metric, and model-diagnostic events.
/// The `Event` struct already derives `Serialize`; optional fields are skipped when `None`, so each
/// line is a compact sparse object. Writes to its sink (stdout by default) and flushes per line so a
/// streaming reader sees progress live.
pub struct JsonlSink {
    writer: Mutex<Box<dyn Write + Send>>,
}

impl JsonlSink {
    /// JSONL to stdout. Human progress stays on stderr, so both streams can be merged without
    /// interleaving the two formats.
    pub fn stdout() -> Self {
        Self::new(Box::new(std::io::stdout()))
    }

    pub fn new(writer: Box<dyn Write + Send>) -> Self {
        Self {
            writer: Mutex::new(writer),
        }
    }
}

impl Sink for JsonlSink {
    fn emit(&self, e: &Event) {
        if let Ok(line) = serde_json::to_string(e) {
            if let Ok(mut w) = self.writer.lock() {
                let _ = writeln!(w, "{line}");
                let _ = w.flush();
            }
        }
    }

    fn emit_sweep(&self, f: &SweepFrame) {
        // Nested under "sweep" so JSONL consumers can distinguish frames from contract Events.
        if let Ok(line) = serde_json::to_string(&serde_json::json!({ "sweep": f })) {
            if let Ok(mut w) = self.writer.lock() {
                let _ = writeln!(w, "{line}");
                let _ = w.flush();
            }
        }
    }
}

/// Stable run metadata used only by human progress sinks.
#[derive(Clone, Debug, Default)]
pub struct RunContext {
    pub command: Option<String>,
    pub dataset: Option<String>,
    pub task: Option<String>,
    pub rows: Option<u64>,
    pub hold_rows: Option<u64>,
    pub features: Option<u64>,
    pub selection_mode: Option<String>,
}

impl RunContext {
    pub fn new(command: impl Into<String>) -> Self {
        Self {
            command: Some(command.into()),
            ..Self::default()
        }
    }

    pub fn dataset(mut self, dataset: impl Into<String>) -> Self {
        self.dataset = Some(dataset.into());
        self
    }

    pub fn task(mut self, task: impl Into<String>) -> Self {
        self.task = Some(task.into());
        self
    }

    pub fn rows(mut self, train: usize, hold: usize) -> Self {
        self.rows = Some(train as u64);
        self.hold_rows = Some(hold as u64);
        self
    }

    pub fn features(mut self, features: usize) -> Self {
        self.features = Some(features as u64);
        self
    }

    pub fn selection_mode(mut self, mode: impl Into<String>) -> Self {
        self.selection_mode = Some(mode.into());
        self
    }
}

#[derive(Clone, Debug, Default)]
pub struct HumanProgressState {
    phase_started_at: [Option<f64>; 8],
    last_progress_at: Option<f64>,
    last_line: Option<String>,
}

impl HumanProgressState {
    pub fn render(
        &mut self,
        e: &Event,
        context: &RunContext,
        phase_label: &str,
        throttle_secs: f64,
    ) -> Option<String> {
        if e.event == "start" {
            if let Some(slot) = self.phase_started_at.get_mut(e.phase as usize) {
                *slot = Some(e.t);
            }
        }

        let immediate = is_immediate(e);
        if !immediate {
            if let Some(last) = self.last_progress_at {
                if e.t - last < throttle_secs {
                    return None;
                }
            }
        }

        let phase_elapsed = if e.event == "done" {
            self.phase_started_at
                .get(e.phase as usize)
                .and_then(|t| *t)
                .map(|start| (e.t - start).max(0.0))
        } else {
            None
        };
        let line = progress_line(e, context, phase_label, phase_elapsed);
        if !immediate && self.last_line.as_ref() == Some(&line) {
            return None;
        }
        if !immediate {
            self.last_progress_at = Some(e.t);
        }
        self.last_line = Some(line.clone());
        Some(line)
    }
}

/// Lightweight stderr sink for tests and diagnostics that should not depend on the CLI crate.
pub struct StderrProgressSink {
    context: RunContext,
    state: Mutex<HumanProgressState>,
}

impl StderrProgressSink {
    pub fn new(context: RunContext) -> Self {
        Self {
            context,
            state: Mutex::new(HumanProgressState::default()),
        }
    }
}

impl Sink for StderrProgressSink {
    fn emit(&self, e: &Event) {
        let phase = phase_label(e.phase);
        let mut state = self.state.lock().expect("progress lock poisoned");
        if let Some(line) = state.render(e, &self.context, phase, DEFAULT_THROTTLE_SECS) {
            eprintln!("{line}");
        }
    }
}
