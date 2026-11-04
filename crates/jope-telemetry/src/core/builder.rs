use super::*;

impl RecorderBuilder {
    pub fn new(run_dir: impl Into<PathBuf>, metadata: RunMetadata) -> Self {
        Self {
            run_dir: run_dir.into(),
            run_id: Ulid::new(),
            metadata,
            source_names: Vec::new(),
            queue_capacity: EVENT_QUEUE_CAPACITY,
        }
    }

    pub fn with_run_id(mut self, run_id: Ulid) -> Self {
        self.run_id = run_id;
        self
    }

    pub fn with_queue_capacity(mut self, capacity: usize) -> Result<Self> {
        if capacity == 0 || capacity > EVENT_QUEUE_CAPACITY {
            bail!("event queue capacity must be in 1..={EVENT_QUEUE_CAPACITY}");
        }
        self.queue_capacity = capacity;
        Ok(self)
    }

    /// Register a bounded producer at run start. Duplicate names and producer 17 are refused.
    pub fn add_producer(&mut self, name: impl Into<String>) -> Result<u16> {
        if self.source_names.len() >= MAX_PRODUCERS {
            bail!("a telemetry run supports at most {MAX_PRODUCERS} producers");
        }
        let name = name.into();
        if name.is_empty() || name.len() > 128 {
            bail!("producer name must contain 1..=128 UTF-8 bytes");
        }
        if self.source_names.iter().any(|s| s == &name) {
            bail!("producer {name:?} is already registered");
        }
        self.source_names.push(name);
        Ok(self.source_names.len() as u16)
    }

    pub fn start(mut self) -> Result<(Recorder, Vec<ProducerHandle>)> {
        let group_count = self
            .metadata
            .metric_groups
            .values()
            .map(BTreeMap::len)
            .sum::<usize>();
        let group_label_bytes = self
            .metadata
            .metric_groups
            .values()
            .flat_map(BTreeMap::values)
            .map(String::len)
            .sum::<usize>();
        if self.metadata.metric_groups.len() > 4096
            || group_count > 4096
            || group_label_bytes > 1 << 20
            || self
                .metadata
                .metric_groups
                .values()
                .flat_map(BTreeMap::values)
                .any(|label| label.is_empty() || label.len() > 128)
        {
            bail!("metric group registry exceeds the frozen 1 MiB / 4,096-label bounds");
        }
        if self.source_names.is_empty() {
            self.source_names.push("jope-telemetry".to_owned());
        }
        let parent = self
            .run_dir
            .parent()
            .ok_or_else(|| anyhow!("run directory must have a parent"))?;
        fs::create_dir_all(parent).context("create telemetry output parent")?;
        fs::create_dir(&self.run_dir).with_context(|| {
            format!(
                "create new telemetry run directory {}",
                self.run_dir.display()
            )
        })?;

        let sources = self
            .source_names
            .iter()
            .enumerate()
            .map(|(i, name)| SourceDef {
                id: (i + 1) as u16,
                name: name.clone(),
            })
            .collect::<Vec<_>>();
        self.metadata.sources = sources.clone();
        let mut manifest = RunManifest::new(self.run_id, self.metadata);
        manifest.event_queue_capacity = self.queue_capacity;
        manifest.event_queue_producers = sources.len();
        manifest.record_queue_capacity = RECORD_QUEUE_CAPACITY;
        manifest.max_record_bytes = MAX_RECORD_BYTES;
        let manifest_path = self.run_dir.join("manifest.json");
        manifest
            .write_atomic(&manifest_path)
            .context("write initial telemetry manifest")?;
        let mut writer = JsonlWriter::create(&self.run_dir).context("create telemetry JSONL")?;

        let run_id_bytes = self.run_id.to_bytes();
        let started = Instant::now();
        let stopping = Arc::new(AtomicBool::new(false));
        let close_requested = Arc::new(AtomicBool::new(false));
        let record_drops = Arc::new(AtomicU64::new(0));
        let record_in_flight = Arc::new(AtomicU64::new(0));
        let oversized_record_drops = Arc::new(AtomicU64::new(0));
        let sink_errors = Arc::new(AtomicU64::new(0));
        let (record_tx, record_rx) = mpsc::sync_channel(RECORD_QUEUE_CAPACITY);

        let mut producers = Vec::with_capacity(sources.len());
        let mut drain_producers = Vec::with_capacity(sources.len());
        for source in &sources {
            let (producer, consumer) = RingBuffer::new(self.queue_capacity);
            let stats = Arc::new(ProducerStats::default());
            producers.push(ProducerHandle {
                producer,
                stats: stats.clone(),
                stopping: stopping.clone(),
                started,
                run_id: run_id_bytes,
                source_id: source.id,
                sequence: 0,
                metric_layout: Arc::new(Vec::new()),
            });
            drain_producers.push(DrainProducer {
                consumer,
                stats,
                capacity: self.queue_capacity,
                dequeued: 0,
                high_water: 0,
            });
        }

        let worker_stop = stopping.clone();
        let worker_close = close_requested.clone();
        let worker_record_drops = record_drops.clone();
        let worker_record_in_flight = record_in_flight.clone();
        let worker_oversized_drops = oversized_record_drops.clone();
        let worker_sink_errors = sink_errors.clone();
        let run_dir = self.run_dir.clone();
        let run_id = self.run_id;
        let join = thread::Builder::new()
            .name("jope-telemetry-drain".to_owned())
            .stack_size(512 * 1024)
            .spawn(move || {
                drain_loop(
                    &run_dir,
                    manifest,
                    &mut writer,
                    drain_producers,
                    record_rx,
                    worker_record_in_flight,
                    worker_stop,
                    worker_close,
                    worker_record_drops,
                    worker_oversized_drops,
                    worker_sink_errors,
                )
            })
            .context("spawn telemetry drain thread")?;

        Ok((
            Recorder {
                run_dir: self.run_dir,
                run_id,
                stopping,
                close_requested,
                join: Some(join),
                record_tx,
                record_drops,
                record_in_flight,
                oversized_record_drops,
                sink_errors,
            },
            producers,
        ))
    }
}
