use super::*;
#[test]
fn admitted_window_is_preserved_when_timing_receipt_is_written() {
    let root = tempfile::tempdir().unwrap();
    let window = jope_pipeline::experiment::ExecutionWindow {
        experiment_sha256: "a".repeat(64),
        started_unix_ms: 1000,
        deadline_unix_ms: 3_601_000,
    };
    let path = root.path().join("window.json");
    write(&path, &window).unwrap();
    let original = fs::read(&path).unwrap();
    assert_eq!(
        record_timing_window(root.path(), "identity", &window).unwrap(),
        3_301_000
    );
    assert_eq!(fs::read(&path).unwrap(), original);
    assert_eq!(
        record_timing_window(root.path(), "identity", &window).unwrap(),
        3_301_000
    );
    let changed = jope_pipeline::experiment::ExecutionWindow {
        deadline_unix_ms: 7_201_000,
        ..window
    };
    assert!(record_timing_window(root.path(), "identity", &changed).is_err());
}
#[test]
fn phase_resume_preserves_cutoff_and_refuses_new_write() {
    let root = tempfile::tempdir().unwrap();
    let original = phase(root.path(), "direct", 1, now() + 10_000).unwrap();
    assert_eq!(
        phase(root.path(), "direct", 1200, now() + 3_600_000).unwrap(),
        original
    );
    assert!(write(
        &root.path().join("phase-direct.json"),
        &json!({"deadline_unix_ms":original+1})
    )
    .is_err());
}
#[test]
fn qualification_rejects_nonfinite_and_mismatched_vectors() {
    assert!(error(&[], &[]).is_err());
    assert!(error(&[1.], &[1., 2.]).is_err());
    assert!(error(&[f32::NAN], &[1.]).is_err());
    assert!(error(&[f32::INFINITY], &[1.]).is_err());
    assert_eq!(error(&[1., 2.], &[1., 2.]).unwrap(), 0.);
}
#[test]
fn explicit_cpu_device_is_refused_before_asset_loading() {
    use jope_policy::context_producer::AssetFile;
    let pin = AssetFile {
        path: PathBuf::from("/does-not-exist"),
        sha256: "0".repeat(64),
    };
    let files = EncoderFiles {
        hyperion_weights: pin.clone(),
        hyperion_manifest: pin.clone(),
        frozen_transforms: pin,
        component_bundle: PathBuf::from("/does-not-exist"),
    };
    let result = CudaProducer::load(&files, Device::Cpu);
    assert!(result.err().unwrap().to_string().contains("requires CUDA"));
}
