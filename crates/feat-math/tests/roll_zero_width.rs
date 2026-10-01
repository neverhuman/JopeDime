use feat_math::dsl::Stage;
use feat_math::genome::GenomeIr;
use feat_math::kernel::width::{MismatchKind, WidthMismatch, WidthReq};
use feat_math::model::{fit, fit_predict};
use feat_math::ops::{apply_op, apply_op_train, fit_op};
use feat_math::transpile::{emit, emit_cpp17, EmitOptions};
use feat_math::Rows;
use std::process::Command;

fn training() -> (Rows, Vec<f64>) {
    let rows: Rows = (0..24)
        .map(|i| vec![i as f64 / 24.0, (i % 5) as f64, (i % 3) as f64])
        .collect();
    let targets = rows
        .iter()
        .map(|row| row[0] + 2.0 * row[1] - row[2])
        .collect();
    (rows, targets)
}

fn zero_width() -> WidthMismatch {
    WidthMismatch {
        stage: 0,
        expected: WidthReq::AtLeast(1),
        got: 0,
        kind: MismatchKind::StageInput,
    }
}

#[test]
fn direct_roll_refuses_zero_width_and_preserves_valid_rotation() {
    let (rows, targets) = training();
    let op = fit_op(&Stage::new("roll"), &rows, &targets).unwrap();
    assert_eq!(apply_op(&op, &vec![vec![]]).unwrap_err(), zero_width());
    assert_eq!(
        apply_op_train(&op, &vec![vec![]]).unwrap_err(),
        zero_width()
    );
    assert_eq!(
        apply_op(&op, &vec![vec![1.0, 2.0, 3.0]]).unwrap(),
        vec![vec![3.0, 1.0, 2.0]]
    );
}

#[test]
fn public_prediction_and_fit_predict_return_the_width_error() {
    let genome = GenomeIr::parse("(pipe X (roll k=2) (ridge))").unwrap();
    let (rows, targets) = training();
    let model = fit(&genome, &rows, &targets).unwrap();
    let empty_rows = vec![vec![]; 3];
    assert_eq!(model.predict(&empty_rows), Err(zero_width()));
    assert_eq!(model.try_predict(&empty_rows), Err(zero_width()));
    let error = fit_predict(&genome, &rows, &targets, &empty_rows).unwrap_err();
    assert_eq!(error.downcast_ref::<WidthMismatch>(), Some(&zero_width()));
    assert!(model.predict(&Vec::new()).unwrap().is_empty());
}

#[test]
fn training_refuses_zero_width_before_running_roll() {
    let genome = GenomeIr::parse("(pipe X (roll k=2) (ridge))").unwrap();
    let empty_rows = vec![vec![]; 8];
    let targets = vec![1.0; 8];
    let error = fit(&genome, &empty_rows, &targets).unwrap_err();
    assert_eq!(error.downcast_ref::<WidthMismatch>(), Some(&zero_width()));
}

#[test]
fn versioned_emitted_recipes_refuse_zero_width_without_changing_legacy_bytes() {
    let genome = GenomeIr::parse("(pipe X (roll k=2) (ridge))").unwrap();
    let legacy = emit(&genome, &EmitOptions::default()).unwrap();
    let guarded = emit(
        &genome,
        &EmitOptions {
            guard_zero_width_roll: true,
            ..EmitOptions::default()
        },
    )
    .unwrap();
    let manifest: serde_json::Value = serde_json::from_str(&guarded.manifest).unwrap();
    let source_transition: serde_json::Value = serde_json::from_slice(
        &std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/compat_freeze/transitions/b02_emitted_roll_guard.v1.json"
        ))
        .unwrap(),
    )
    .unwrap();
    let legacy_manifest: serde_json::Value = serde_json::from_str(&legacy.manifest).unwrap();
    assert_eq!(
        source_transition["schema"],
        "feat-math.emitted-roll-width-transition.v1"
    );
    assert_eq!(source_transition["program"], manifest["program"]);
    assert_eq!(source_transition["before"], legacy_manifest);
    assert_eq!(source_transition["after"], manifest);
    assert_eq!(manifest["emitter_version"], "feat-math-roll-width-guard-v1");
    assert_ne!(legacy.rust, guarded.rust);
    assert_ne!(legacy.python, guarded.python);
    let cpp_refusal = emit_cpp17(&genome, &EmitOptions::default())
        .unwrap_err()
        .to_string();
    assert!(cpp_refusal.contains("Roll"), "{cpp_refusal}");

    let target = std::env::var("CARGO_TARGET_DIR")
        .unwrap_or_else(|_| format!("{}/../../target", env!("CARGO_MANIFEST_DIR")));
    let directory = tempfile::Builder::new()
        .prefix("b02-roll-guard-")
        .tempdir_in(target)
        .unwrap();

    let rust_source = guarded
        .rust
        .replacen("fn main() {", "fn emitted_main() {", 1);
    assert_ne!(
        rust_source, guarded.rust,
        "generated Rust driver entry point"
    );
    let rust_source = format!(
        "{rust_source}\nfn main() {{ let x = vec![vec![1.0, 2.0, 3.0]; 24]; \
         let y = vec![1.0; 24]; let model = Model::fit(&x, &y); \
         assert_eq!(model.predict(&x).len(), x.len()); \
         assert_eq!(model.try_predict(&vec![Vec::<f64>::new()]).unwrap_err(), \
                    \"stage 0 needs at least 1 input columns, got 0\"); \
         let _ = model.predict(&vec![Vec::<f64>::new()]); }}\n"
    );
    let rust_path = directory.path().join("model.rs");
    std::fs::write(&rust_path, rust_source).unwrap();
    let executable = directory.path().join("model");
    let compile = Command::new("rustc")
        .args(["--edition=2021", "-O"])
        .arg(&rust_path)
        .arg("-o")
        .arg(&executable)
        .output()
        .unwrap();
    assert!(
        compile.status.success(),
        "{}",
        String::from_utf8_lossy(&compile.stderr)
    );
    let run = Command::new(&executable).output().unwrap();
    assert_eq!(run.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(stderr.contains("stage 0 needs at least 1 input columns, got 0"));
    assert!(!stderr.contains("panicked"));

    let python_path = directory.path().join("model.py");
    std::fs::write(&python_path, guarded.python).unwrap();
    let python = Command::new("python3")
        .args(["-c", "import runpy, sys\nmodule = runpy.run_path(sys.argv[1], run_name='b02_test')\nmodel = module['Model']().fit([[1.0, 2.0, 3.0]] * 24, [1.0] * 24)\nassert len(model.predict([[1.0, 2.0, 3.0]])) == 1\ntry:\n    model.predict([[]])\nexcept ValueError as error:\n    assert str(error) == 'stage 0 needs at least 1 input columns, got 0'\nelse:\n    raise AssertionError('zero-width row was accepted')", python_path.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        python.status.success(),
        "{}",
        String::from_utf8_lossy(&python.stderr)
    );
}
