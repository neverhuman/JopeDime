impl PyRecorder {
    fn start(
        run_dir: PathBuf,
        producer_names: &[String],
        mut metadata: BTreeMap<String, Value>,
        mut metric_groups: MetricGroups,
    ) -> PyResult<Self> {
        if let Some(groups) = metadata.remove("metric_groups") {
            metric_groups.extend(parse_metric_groups(groups)?);
        }
        if producer_names.iter().any(|name| name == "host-probe") {
            if let Ok(nvml) = nvml_wrapper::Nvml::init() {
                if let Ok(count) = nvml.device_count() {
                    let labels: BTreeMap<u16, String> = (0..count.min(4096))
                        .map(|index| (index as u16, format!("nvml_device_{index}")))
                        .collect();
                    metric_groups.insert(crate::train::GPU_UTIL, labels.clone());
                    metric_groups.insert(crate::train::GPU_MEMORY_BYTES, labels);
                }
            }
        }
        let mut builder = RecorderBuilder::new(
            run_dir,
            RunMetadata {
                metadata,
                metric_groups: metric_groups.clone(),
                ..RunMetadata::default()
            },
        );
        for name in producer_names {
            builder.add_producer(name.clone()).map_err(value_error)?;
        }
        let (recorder, producers) = builder.start().map_err(runtime_error)?;
        let mut producers: Vec<Option<ProducerHandle>> = producers.into_iter().map(Some).collect();
        let host_collector =
            if let Some(index) = producer_names.iter().position(|name| name == "host-probe") {
                let producer = producers
                    .get_mut(index)
                    .and_then(Option::take)
                    .ok_or_else(|| PyRuntimeError::new_err("host-probe producer is unavailable"))?;
                Some(
                    SystemCollector::start(producer, std::time::Duration::from_secs(1)).map_err(
                        |error| PyRuntimeError::new_err(format!("start host collector: {error}")),
                    )?,
                )
            } else {
                None
            };
        let n = producers.len();
        Ok(Self {
            recorder: Some(recorder),
            submit_buffers: (0..n).map(|_| SubmitBuffers::new()).collect(),
            producers,
            host_collector,
            metric_groups,
            steps: (0..n).map(|_| None).collect(),
            h2d: (0..n).map(|_| None).collect(),
            io: None,
            round: None,
        })
    }

    fn producer_mut(&mut self, source: usize) -> PyResult<&mut ProducerHandle> {
        self.producers
            .get_mut(source)
            .ok_or_else(|| PyValueError::new_err("unknown producer index"))?
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("producer is unavailable"))
    }

    fn recorder_ref(&self) -> PyResult<&Recorder> {
        self.recorder
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("recorder is closed"))
    }

    fn set_step_layout(
        &mut self,
        source: usize,
        columns: Vec<String>,
        rows: usize,
    ) -> PyResult<()> {
        let layout =
            trainers::resolve_layout(&columns, &self.metric_groups).map_err(value_error)?;
        let batch = StepBatch::new(columns, rows).map_err(value_error)?;
        self.producer_mut(source)?
            .set_metric_layout(layout)
            .map_err(value_error)?;
        *self
            .steps
            .get_mut(source)
            .ok_or_else(|| PyValueError::new_err("unknown producer index"))? = Some(StepState {
            batch,
            timer: StepTimer::default(),
        });
        Ok(())
    }

    fn step_state(&mut self) -> PyResult<&mut StepState> {
        self.steps
            .get_mut(STEP_SOURCE)
            .and_then(Option::as_mut)
            .ok_or_else(|| PyRuntimeError::new_err("no step layout: call step_layout first"))
    }

    fn step_col(&mut self, column: &str) -> PyResult<usize> {
        let state = self.step_state()?;
        state
            .batch
            .col(column)
            .ok_or_else(|| PyValueError::new_err(format!("step layout has no {column} column")))
    }

    fn flush_all(&mut self) {
        for (state, producer) in self.steps.iter_mut().zip(self.producers.iter_mut()) {
            if let (Some(state), Some(producer)) = (state.as_mut(), producer.as_mut()) {
                state.batch.flush(producer);
            }
        }
    }

    fn h2d_mut(&mut self, source: usize) -> PyResult<&mut H2dSampler> {
        self.h2d
            .get_mut(source)
            .and_then(Option::as_mut)
            .ok_or_else(|| PyValueError::new_err(format!("source {source} has no H2D sampler")))
    }

    fn h2d_dropped_total(&self) -> u64 {
        self.h2d.iter().flatten().map(|s| s.dropped_samples).sum()
    }

    fn stop_collector(&mut self) {
        if let Some(collector) = &mut self.host_collector {
            collector.stop();
        }
        self.host_collector.take();
    }

    fn finalize(&mut self, complete: bool) -> PyResult<String> {
        self.flush_all();
        self.stop_collector();
        self.producers.clear();
        let recorder = self
            .recorder
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("recorder is already closed"))?;
        let manifest = if complete {
            recorder.close()
        } else {
            recorder.interrupt()
        }
        .map_err(runtime_error)?;
        serde_json::to_string(&manifest).map_err(runtime_error)
    }

    fn round_record(
        &self,
        queue_stats: &Bound<'_, PyDict>,
        extra: &[(&str, J)],
    ) -> PyResult<String> {
        let mut body = String::from("{\"round\":");
        body.push_str(&self.round.unwrap_or(0).to_string());
        for (k, v) in queue_stats.iter() {
            let key = k
                .cast::<PyString>()
                .map_err(|_| PyTypeError::new_err("queue_stats keys must be str"))?
                .to_cow()?
                .into_owned();
            if key == "round" || extra.iter().any(|(e, _)| *e == key) {
                continue;
            }
            body.push(',');
            body.push_str(&serde_json::to_string(&key).map_err(value_error)?);
            body.push(':');
            write_json(&v, &mut body, 1)?;
        }
        for (k, v) in extra {
            body.push(',');
            body.push_str(&serde_json::to_string(k).map_err(value_error)?);
            body.push(':');
            body.push_str(&serde_json::to_string(v).map_err(value_error)?);
        }
        body.push('}');
        Ok(body)
    }
}
