use super::*;

use super::*;
use stage1b::{SvdSlot, DOPE_FEATURES, TOKEN_WIDTH};

#[test]
fn gpu_entry_point_refuses_busy_or_missing_lease_before_callback() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("qualification.lock");
    let invoked = std::cell::Cell::new(false);
    assert!(with_gpu_qualification(&path, || {
        invoked.set(true);
        Ok(())
    })
    .is_err());
    assert!(!invoked.get());
    assert!(!path.exists());
    let lease = std::fs::File::create(&path).unwrap();
    lease.lock().unwrap();
    assert!(with_gpu_qualification(&path, || {
        invoked.set(true);
        Ok(())
    })
    .is_err());
    assert!(!invoked.get());
}

#[test]
fn gpu_lease_is_held_for_the_complete_callback_and_released_after_failure() {
    let lock = tempfile::NamedTempFile::new().unwrap();
    let result: Result<()> = with_gpu_qualification(lock.path(), || {
        let contender = OpenOptions::new()
            .read(true)
            .write(true)
            .open(lock.path())
            .unwrap();
        assert!(contender.try_lock().is_err());
        anyhow::bail!("GPU operation control failure")
    });
    assert!(result.unwrap_err().to_string().contains("control failure"));
    assert_eq!(with_gpu_qualification(lock.path(), || Ok(17)).unwrap(), 17);
}

fn slot(width: usize) -> SvdSlot {
    SvdSlot {
        mean: vec![0.0; width],
        std: vec![1.0; width],
        active: vec![1.0; width],
        projection: vec![0.0; width * TOKEN_WIDTH],
    }
}

// Arithmetic controls only: these are not model weights or M2 goldens.
fn projection_fixture() -> (Lm48Projection, Stage1bParameters) {
    let mut projection = Lm48Projection {
        weight: vec![0.0; 96],
        bias: vec![0.0; 48],
        in_dim: 2,
        out_dim: 48,
    };
    projection.weight[0] = 2.0;
    projection.bias[0] = 1.0;
    projection.weight[3] = 1.0;
    let mut parameters = Stage1bParameters {
        lm80_mean: [0.0; 80],
        lm80_std: [1.0; 80],
        lm48_mean: [0.0; 48],
        lm48_std: [1.0; 48],
        slot0: slot(184),
        slot1: slot(48),
        slot2: slot(DOPE_FEATURES),
        slot4: slot(DOPE_FEATURES),
        slot5: slot(DOPE_FEATURES),
        slot7: slot(184),
        slot9: slot(64),
    };
    parameters.slot0.projection[104 * TOKEN_WIDTH] = 1.0;
    parameters.slot2.projection[0] = 1.0;
    (projection, parameters)
}

#[test]
fn placement_serializes_actual_cpu_stages_and_worker_count() {
    for request in [DeviceRequest::Auto, DeviceRequest::Cpu] {
        let placement = cpu_placement(request, 7).unwrap();
        assert_eq!(
            serde_json::to_value(placement).unwrap(),
            serde_json::json!({
                "dope":"cpu", "ranker":"cpu", "evaluation":"cpu", "workers":7,
            })
        );
    }
    assert_eq!(
        cpu_placement(DeviceRequest::Cpu, 0),
        Err(ErrorCode::InvalidInput)
    );
}

#[test]
fn explicit_cuda_refuses_with_catalog_backend_error_without_probing_gpu() {
    for index in [0, 1, u32::MAX] {
        let error = cpu_placement(DeviceRequest::Cuda(index), 1).unwrap_err();
        assert_eq!(error, ErrorCode::BackendUnavailable);
        assert_eq!(error.as_str(), "BACKEND_UNAVAILABLE");
        assert_eq!(error.http_status(), Some(503));
        assert_eq!(error.exit_code(), Some(4));
    }
}

#[test]
fn cpu_reference_executes_both_existing_projection_primitives() {
    let (projection, parameters) = projection_fixture();
    let mut dope = [0.0; 629];
    dope[0] = 1.5;
    let result = cpu_projection_reference(&projection, &[1.0, 2.0], &dope, &parameters).unwrap();
    assert_eq!(&result.lm48[..2], &[3.0, 2.0]);
    assert_eq!(result.stage1b.tokens[0][0], 3.0);
    assert_eq!(result.stage1b.tokens[2][0], 1.5);
    assert_eq!(
        result.stage1b.mask,
        [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    );
    let changed = cpu_projection_reference(&projection, &[2.0, 2.0], &dope, &parameters).unwrap();
    assert_eq!(changed.lm48[0], 5.0);
    assert_eq!(changed.stage1b.tokens[0][0], 5.0);
    dope[0] = -2.0;
    let changed = cpu_projection_reference(&projection, &[1.0, 2.0], &dope, &parameters).unwrap();
    assert_eq!(changed.stage1b.tokens[2][0], -2.0);
}

#[test]
fn cpu_reference_rejects_invalid_shapes_and_nonfinite_arithmetic() {
    let (mut projection, mut parameters) = projection_fixture();
    assert!(cpu_projection_reference(&projection, &[1.0], &[0.0; 629], &parameters).is_err());
    assert!(cpu_projection_reference(&projection, &[1.0, 2.0], &[0.0; 628], &parameters).is_err());
    projection.weight.pop();
    assert!(cpu_projection_reference(&projection, &[1.0, 2.0], &[0.0; 629], &parameters).is_err());
    projection.weight.push(0.0);
    projection.weight[0] = f32::NAN;
    assert!(cpu_projection_reference(&projection, &[1.0, 2.0], &[0.0; 629], &parameters).is_err());
    projection.weight[0] = f32::MAX;
    assert!(cpu_projection_reference(&projection, &[2.0, 2.0], &[0.0; 629], &parameters).is_err());
    projection.weight[0] = 2.0;
    parameters.slot0.std[0] = 0.0;
    assert!(cpu_projection_reference(&projection, &[1.0, 2.0], &[0.0; 629], &parameters).is_err());
}

fn view<'a>(shape: &'a [usize], values: &'a [f32]) -> TensorView<'a> {
    TensorView { shape, values }
}

#[test]
fn compares_f32_absolute_and_reference_relative_tolerances() {
    let mut expected = [0.0; 48];
    expected[1] = -1000.0;
    let mut actual = expected;
    actual[0] = 1e-6;
    actual[1] += 0.05;
    let report = compare_stage(
        TensorStage::Lm48,
        view(&[48], &expected),
        view(&[48], &actual),
    )
    .unwrap();
    assert_eq!(report.values, 48);
    assert!(report.max_absolute_error > 0.04);
    actual[0] = 0.001;
    assert!(compare_stage(
        TensorStage::Lm48,
        view(&[48], &expected),
        view(&[48], &actual)
    )
    .is_err());
}

#[test]
fn rejects_empty_overflowing_and_axis_mismatched_shapes() {
    for shape in [&[][..], &[0, 48][..], &[usize::MAX, 48][..], &[47][..]] {
        assert!(compare_stage(TensorStage::Lm48, view(shape, &[]), view(shape, &[])).is_err());
    }
    assert!(compare_stage(
        TensorStage::Lm48,
        view(&[1, 48], &[0.0; 48]),
        view(&[48], &[0.0; 48])
    )
    .is_err());
}

#[test]
fn rejects_nonfinite_reference_or_actual_and_finite_extreme_drift() {
    let good = [0.0; 48];
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut bad = good;
        bad[0] = invalid;
        assert!(compare_stage(TensorStage::Lm48, view(&[48], &good), view(&[48], &bad)).is_err());
        assert!(compare_stage(TensorStage::Lm48, view(&[48], &bad), view(&[48], &good)).is_err());
    }
    assert!(compare_stage(
        TensorStage::Lm48,
        view(&[48], &[f32::MAX; 48]),
        view(&[48], &[-f32::MAX; 48])
    )
    .is_err());
}

#[test]
fn binary_mask_and_serving_slate_dimensions_are_enforced() {
    assert!(compare_stage(
        TensorStage::Mask,
        view(&[10], &[0.0; 10]),
        view(&[10], &[1e-6; 10])
    )
    .is_err());
    for n in [12, 20] {
        let costs = vec![0.0; n];
        assert!(compare_stage(TensorStage::Costs, view(&[n], &costs), view(&[n], &costs)).is_ok());
    }
    assert!(compare_stage(
        TensorStage::Costs,
        view(&[8], &[0.0; 8]),
        view(&[8], &[0.0; 8])
    )
    .is_err());
}

struct TraceFixture {
    context_id: &'static str,
    stages: Vec<Vec<f32>>,
}

impl TraceFixture {
    fn new() -> Self {
        Self {
            context_id: "context-a",
            stages: [256, 1088, 629, 48, 320, 10, 12]
                .map(|n| vec![0.0; n])
                .to_vec(),
        }
    }

    fn trace(&self) -> Trace<'_> {
        Trace {
            context_id: self.context_id,
            v256: view(&[256], &self.stages[0]),
            col: view(&[64, 17], &self.stages[1]),
            dope629: view(&[629], &self.stages[2]),
            lm48: view(&[48], &self.stages[3]),
            tokens: view(&[10, 32], &self.stages[4]),
            mask: view(&[10], &self.stages[5]),
            costs: view(&[12], &self.stages[6]),
        }
    }
}

#[test]
fn rejects_reference_and_actual_traces_from_different_contexts() {
    let reference = TraceFixture::new();
    let mut actual = TraceFixture::new();
    actual.context_id = "context-b";
    let error = compare_trace(reference.trace(), actual.trace()).unwrap_err();
    assert!(error.to_string().contains("context identities differ"));

    actual.context_id = "";
    assert!(compare_trace(actual.trace(), actual.trace()).is_err());
}

#[test]
fn every_stage_perturbation_is_detected_even_when_costs_are_unchanged() {
    let reference = TraceFixture::new();
    assert_eq!(
        compare_trace(reference.trace(), reference.trace())
            .unwrap()
            .len(),
        7
    );
    for stage in 0..7 {
        let mut actual = TraceFixture::new();
        actual.stages[stage][0] = 1.0;
        assert!(
            compare_trace(reference.trace(), actual.trace()).is_err(),
            "stage {stage}"
        );
    }
    let mut incomplete = TraceFixture::new();
    incomplete.stages[6].clear();
    assert!(compare_trace(reference.trace(), incomplete.trace()).is_err());
}

fn costs(proposals: usize) -> Vec<f32> {
    [-1000.0; 4]
        .into_iter()
        .chain((0..proposals).map(|n| n as f32))
        .collect()
}

#[test]
fn compares_admitted_sets_outside_the_boundary_band() {
    let reference = costs(8);
    let same = compare_admission(&reference, &reference, false, [7; 32]).unwrap();
    assert!(!same.ambiguous_boundary);
    assert_eq!(same.reference, [0, 1, 2, 3]);
    assert_eq!(same.boundary_gap, 1.0);
    let mut disconnected = reference.clone();
    disconnected.swap(7, 8);
    assert!(compare_admission(&reference, &disconnected, false, [7; 32]).is_err());
}

#[test]
fn counts_ambiguous_boundary_without_asserting_set_equality() {
    let mut reference = costs(8);
    reference[8] = reference[7];
    let mut actual = reference.clone();
    actual[7] += 1e-5;
    let report = compare_admission(&reference, &actual, false, [7; 32]).unwrap();
    assert!(report.ambiguous_boundary);
    assert_eq!(report.boundary_gap, 0.0);
    assert_ne!(report.reference, report.actual);
}

#[test]
fn exploration_replays_rng_and_uses_second_third_boundary() {
    let reference = costs(16);
    let report = compare_admission(&reference, &reference, true, [19; 32]).unwrap();
    assert_eq!(report.reference, report.actual);
    assert_eq!(&report.reference[..2], &[0, 1]);
    assert!(!report.ambiguous_boundary);
    assert_eq!(report.tie_band, 2.0 * (ATOL + RTOL));
    let mut tied = reference;
    tied[6] = tied[5];
    assert!(
        compare_admission(&tied, &tied, true, [19; 32])
            .unwrap()
            .ambiguous_boundary
    );
}

#[test]
fn signed_zero_ties_preserve_existing_admission_order() {
    let reference = [
        0.0, 0.0, 0.0, 0.0, -3.0, -2.0, -1.0, -0.0, 0.0, 1.0, 2.0, 3.0,
    ];
    let report = compare_admission(&reference, &reference, false, [0; 32]).unwrap();
    assert_eq!(report.reference, [0, 1, 2, 3]);
    assert!(report.ambiguous_boundary);
}

#[test]
fn admission_rejects_bad_slate_and_nonfinite_costs_on_either_side() {
    let good = costs(8);
    assert!(compare_admission(&good, &good[..11], false, [0; 32]).is_err());
    assert!(compare_admission(&good, &good, true, [0; 32]).is_err());
    for invalid in [f32::NAN, f32::INFINITY] {
        let mut bad = good.clone();
        bad[4] = invalid;
        assert!(compare_admission(&good, &bad, false, [0; 32]).is_err());
        assert!(compare_admission(&bad, &good, false, [0; 32]).is_err());
    }
}
