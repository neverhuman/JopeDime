fn parse_metric_groups(value: Value) -> PyResult<MetricGroups> {
    let outer = value
        .as_object()
        .ok_or_else(|| PyValueError::new_err("metric_groups must be an object"))?;
    let mut result = MetricGroups::new();
    let mut total_labels = 0_usize;
    for (metric, groups) in outer {
        let metric_id = metric
            .parse::<u16>()
            .map_err(|e| PyValueError::new_err(format!("metric_groups key {metric:?}: {e}")))?;
        let groups = groups
            .as_object()
            .ok_or_else(|| PyValueError::new_err("metric group labels must be objects"))?;
        let mut labels = BTreeMap::new();
        for (group, label) in groups {
            let group_id = group
                .parse::<u16>()
                .map_err(|e| PyValueError::new_err(format!("group id {group:?}: {e}")))?;
            let label = label
                .as_str()
                .ok_or_else(|| PyValueError::new_err("metric group label must be a string"))?;
            if label.is_empty() || label.len() > 128 {
                return Err(PyValueError::new_err(
                    "metric group label must contain 1..=128 bytes",
                ));
            }
            total_labels += label.len();
            if total_labels > 1 << 20
                || result.values().map(BTreeMap::len).sum::<usize>() + labels.len() >= 4096
            {
                return Err(PyValueError::new_err(
                    "metric group registry exceeds frozen bounds",
                ));
            }
            labels.insert(group_id, label.to_owned());
        }
        result.insert(metric_id, labels);
    }
    Ok(result)
}

/// P (`jope.train`, active-v2): one `python-trainer` producer with the loss components, LR and
/// host spans as its step row. The directory is the run directory itself.
#[pyfunction]
#[pyo3(signature = (run_dir, components, metadata=None))]
fn open_python_trainer(
    run_dir: PathBuf,
    components: Vec<String>,
    metadata: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyRecorder> {
    let (columns, groups) = trainers::python_trainer_layout(&components);
    let mut rec = PyRecorder::start(
        run_dir,
        &["python-trainer".to_owned()],
        metadata_map(metadata)?,
        groups,
    )?;
    rec.set_step_layout(STEP_SOURCE, columns, 32)?;
    Ok(rec)
}

/// W (`research/flywheel_selfplay`, one recorder per training round) under
/// `<run_root>/w-step-r<round>-s<seed>-<32 hex>/`: the step row with IO columns, two H2D
/// sources (main and evolution batch) and the 1 s host probe. `startup_reads` is the
/// `LocalReads` that observed the startup loads; its totals are taken (and reset) here.
#[pyfunction]
#[pyo3(signature = (run_root, round_idx, seed, batch_size, device, cuda_available, startup_reads=None))]
fn open_flywheel_step(
    run_root: PathBuf,
    round_idx: u32,
    seed: u64,
    batch_size: u64,
    device: &str,
    cuda_available: bool,
    startup_reads: Option<&Bound<'_, PyLocalReads>>,
) -> PyResult<PyRecorder> {
    let startup = startup_reads.map(|r| lock(&r.get().inner).take_snapshot());
    let cuda = device.starts_with("cuda") && cuda_available;
    let run_dir = run_root.join(format!(
        "w-step-r{round_idx}-s{seed}-{:032x}",
        ulid::Ulid::new().0
    ));
    let startup_json = match &startup {
        Some(s) => serde_json::to_value(s).map_err(runtime_error)?,
        None => serde_json::json!({"enabled": false}),
    };
    let metadata: BTreeMap<String, Value> = [
        ("trainer", Value::from("research.flywheel_selfplay")),
        ("round", round_idx.into()),
        ("seed", seed.into()),
        ("batch_size", batch_size.into()),
        ("device", device.into()),
        ("evo_queue", "enabled_or_disabled_per_run".into()),
        (
            "data_identity",
            "unavailable_without_extra_input_read".into(),
        ),
        ("not_fused_dope_training", true.into()),
        ("host", trainers::hostname().into()),
        ("io_startup_reads", startup_json),
        ("s3_get_capability", false.into()),
        ("object_cache_capability", false.into()),
        ("pin_memory_capability", false.into()),
        ("h2d_event_capability", cuda.into()),
        ("h2d_sample_stride", io::H2D_SAMPLE_STRIDE.into()),
        ("startup_io_capability", startup.is_some().into()),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_owned(), v))
    .collect();
    let producers: Vec<String> = trainers::W_PRODUCERS
        .iter()
        .map(|p| (*p).to_owned())
        .collect();
    let mut rec = PyRecorder::start(
        run_dir,
        &producers,
        metadata,
        trainers::flywheel_metric_groups(trainers::cpu_count()),
    )?;
    rec.round = Some(round_idx);
    rec.set_step_layout(STEP_SOURCE, trainers::flywheel_step_columns(), 32)?;
    let io = BatchIo::new(&rec.step_state()?.batch).map_err(runtime_error)?;
    rec.io = Some(io);
    for (source, evolution) in [
        (trainers::W_SOURCE_MAIN_H2D, false),
        (trainers::W_SOURCE_EVO_H2D, true),
    ] {
        rec.set_step_layout(source, trainers::flywheel_h2d_columns(evolution), 32)?;
        rec.h2d[source] = Some(H2dSampler::new(
            cuda,
            !evolution,
            io::H2D_SAMPLE_STRIDE,
            io::H2D_SLOTS,
        ));
    }
    trainers::emit_flywheel_startup(rec.producer_mut(STEP_SOURCE)?, startup.as_ref());
    if !cuda {
        for source in [trainers::W_SOURCE_MAIN_H2D, trainers::W_SOURCE_EVO_H2D] {
            let evolution = source == trainers::W_SOURCE_EVO_H2D;
            trainers::emit_h2d_unavailable(rec.producer_mut(source)?, evolution, true);
        }
    }
    Ok(rec)
}
