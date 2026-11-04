use feat_math::{
    transpile::{emit, EmitOptions},
    GenomeIr,
};
use std::{fs, process::Command};

#[test]
fn exp_next_04_tie_plateau_and_representative_pipelines_match_native_partition_point() {
    for tool in ["python3", "rustc", "g++", "node"] {
        let out = Command::new(tool)
            .arg("--version")
            .output()
            .unwrap_or_else(|e| panic!("{tool} must be available on PATH: {e}"));
        assert!(out.status.success(), "{tool} --version failed");
    }

    let check_finite_and_pointwise_equal = |expected: &[f64],
                                            actual: &[f64],
                                            allow_signed_zero_equivalence: bool|
     -> Result<(), String> {
        if expected.len() != actual.len() {
            return Err(format!(
                "length mismatch: expected {}, got {}",
                expected.len(),
                actual.len()
            ));
        }
        for (idx, (&e, &a)) in expected.iter().zip(actual).enumerate() {
            if !e.is_finite() {
                return Err(format!("row {idx}: expected prediction {e} is not finite"));
            }
            if !(e == a || (e.is_nan() && a.is_nan())) {
                return Err(format!(
                    "row {idx}: value mismatch expected={e:?} actual={a:?}"
                ));
            }
            if e.to_bits() != a.to_bits()
                && !(allow_signed_zero_equivalence && e == 0.0 && a == 0.0)
            {
                return Err(format!(
                    "row {idx}: bit mismatch expected={:#018x} ({e:?}) actual={:#018x} ({a:?})",
                    e.to_bits(),
                    a.to_bits()
                ));
            }
        }
        Ok(())
    };

    // Counterexample proving the old `fold(0.0_f64, f64::max)` reduction is NaN-blind
    // whereas `check_finite_and_pointwise_equal` rejects NaN-vs-finite and length mismatches.
    let nan_blind_max_abs = |a: &[f64], b: &[f64]| -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f64, f64::max)
    };
    assert_eq!(nan_blind_max_abs(&[1.0, 2.0], &[1.0, f64::NAN]), 0.0);
    assert!(check_finite_and_pointwise_equal(&[1.0, 2.0], &[1.0, f64::NAN], false).is_err());
    assert!(check_finite_and_pointwise_equal(&[1.0], &[1.0, 2.0], false).is_err());
    assert!(check_finite_and_pointwise_equal(&[1.0, -0.0], &[1.0, 0.0], false).is_err());
    assert!(check_finite_and_pointwise_equal(&[1.0, -0.0], &[1.0, 0.0], true).is_ok());

    // Verify the versioned recipe transition fixture for `partition_point_quantile_bounds`.
    let transition_genome =
        GenomeIr::parse("(pipe X (quantile_bins append=false bins=27) (ridge))").unwrap();
    let legacy_transition_emit = emit(&transition_genome, &EmitOptions::default()).unwrap();
    let fixed_transition_emit = emit(
        &transition_genome,
        &EmitOptions {
            partition_point_quantile_bounds: true,
            ..EmitOptions::default()
        },
    )
    .unwrap();
    let legacy_manifest: serde_json::Value =
        serde_json::from_str(&legacy_transition_emit.manifest).unwrap();
    let fixed_manifest: serde_json::Value =
        serde_json::from_str(&fixed_transition_emit.manifest).unwrap();
    let source_transition: serde_json::Value = serde_json::from_slice(
        &fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/compat_freeze/transitions/exp_next_04_emitted_quantile_partition_point.v1.json"
        ))
        .unwrap(),
    )
    .unwrap();
    assert_eq!(
        source_transition["schema"],
        "feat-math.emitted-quantile-partition-point-transition.v1"
    );
    assert_eq!(source_transition["program"], fixed_manifest["program"]);
    assert_eq!(source_transition["before"], legacy_manifest);
    assert_eq!(source_transition["after"], fixed_manifest);
    assert_eq!(legacy_manifest["emitter_version"], "2.2.0");
    assert_eq!(
        fixed_manifest["emitter_version"],
        "feat-math-quantile-partition-point-v1"
    );
    assert_ne!(legacy_transition_emit.rust, fixed_transition_emit.rust);
    assert_ne!(legacy_transition_emit.python, fixed_transition_emit.python);

    // Verify unambiguous combined `Manifest.emitter_version` labels when
    // `partition_point_quantile_bounds` is enabled alongside `guard_zero_width_roll`
    // and/or `MathClip`, while `partition_point_quantile_bounds: false` retains
    // the exact legacy labels and bytes.
    for (dsl, guard_roll, expected_legacy, expected_opt_in) in [
        (
            "(pipe X (roll k=3) (quantile_bins bins=8) (ridge))",
            true,
            "feat-math-roll-width-guard-v1",
            "feat-math-roll-width-guard-v1+quantile-partition-point-v1",
        ),
        (
            "(pipe X (quantile_bins bins=8) (map expr=\"clip(x,-1,1)\") (ridge))",
            false,
            "feat-math-clip-safety-v2",
            "feat-math-quantile-partition-point-v1+clip-safety-v2",
        ),
        (
            "(pipe X (roll k=3) (quantile_bins bins=8) (map expr=\"clip(x,-1,1)\") (ridge))",
            true,
            "feat-math-roll-width-guard-v1",
            "feat-math-roll-width-guard-v1+quantile-partition-point-v1+clip-safety-v2",
        ),
    ] {
        let g = GenomeIr::parse(dsl).unwrap();
        let legacy_combo = emit(
            &g,
            &EmitOptions {
                guard_zero_width_roll: guard_roll,
                partition_point_quantile_bounds: false,
                ..EmitOptions::default()
            },
        )
        .unwrap();
        let opt_in_combo = emit(
            &g,
            &EmitOptions {
                guard_zero_width_roll: guard_roll,
                partition_point_quantile_bounds: true,
                ..EmitOptions::default()
            },
        )
        .unwrap();
        let legacy_combo_manifest: serde_json::Value =
            serde_json::from_str(&legacy_combo.manifest).unwrap();
        let opt_in_combo_manifest: serde_json::Value =
            serde_json::from_str(&opt_in_combo.manifest).unwrap();
        assert_eq!(legacy_combo_manifest["emitter_version"], expected_legacy);
        assert_eq!(opt_in_combo_manifest["emitter_version"], expected_opt_in);
        assert_ne!(legacy_combo.python, opt_in_combo.python);
        assert_ne!(legacy_combo.rust, opt_in_combo.rust);
    }
}
