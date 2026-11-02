#[pymodule]
fn _telemetry(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    module.add_class::<PyRecorder>()?;
    module.add_class::<PyFullFlywheel>()?;
    module.add_class::<PyRecordSink>()?;
    module.add_class::<PyGpSession>()?;
    module.add_class::<PyGenerationRecorder>()?;
    module.add_class::<PyEvalBatch>()?;
    module.add_class::<PyValidationRecorder>()?;
    module.add_class::<PyDopeGuard>()?;
    module.add_class::<PyLocalReads>()?;
    module.add_class::<PyReadSpan>()?;
    module.add_class::<PyHardwareMonitor>()?;
    module.add_class::<PyLiveMetricsWriter>()?;
    module.add_class::<PyRatchetLive>()?;
    module.add_function(wrap_pyfunction!(metric_registry, module)?)?;
    module.add_function(wrap_pyfunction!(metric_id, module)?)?;
    module.add_function(wrap_pyfunction!(freeze_source_family_split_py, module)?)?;
    module.add_function(wrap_pyfunction!(derive_source_family, module)?)?;
    module.add_function(wrap_pyfunction!(build_binary_dataset_contracts, module)?)?;
    module.add_function(wrap_pyfunction!(open_python_trainer, module)?)?;
    module.add_function(wrap_pyfunction!(open_flywheel_step, module)?)?;
    module.add_function(wrap_pyfunction!(check_partition_label, module)?)?;
    module.add_function(wrap_pyfunction!(gpu_snapshot, module)?)?;
    module.add(
        "DopeFreezeViolationError",
        py.get_type::<DopeFreezeViolationError>(),
    )?;
    module.add(
        "SealedPartitionError",
        py.get_type::<SealedPartitionError>(),
    )?;
    module.add("DOPE_SUBMODULE", dope::DOPE_SUBMODULE)?;
    module.add(
        "DOPE_NORMALIZER_BUFFERS",
        dope::DOPE_NORMALIZER_BUFFERS.to_vec(),
    )?;
    Ok(())
}
