use super::*;
use crate::experiment::{EvaluationStage, Metric, Resources, EXPERIMENT_SCHEMA, REQUEST_SCHEMA};
use jope_datasets::catalog::Task;

fn fixture_experiment() -> Experiment {
    Experiment {
        schema: EXPERIMENT_SCHEMA.into(),
        panel_sha256: "a".repeat(64),
        methods_sha256: "b".repeat(64),
        source_sha: "c".repeat(40),
        checkpoints: BTreeMap::new(),
        seed: 1,
        resources: Resources {
            workers: 2,
            cpus: 2,
            memory_bytes: 256 * 1024 * 1024,
            swap_bytes: 0,
            gpu_tasks: 0,
            threads_per_worker: 1,
            total_seconds: 120,
            reporting_seconds: 10,
            per_fit_seconds: 2,
        },
        expected_artifacts: vec!["predictions".into()],
        protocol: "synthetic-test-only".into(),
        numerical_policy: "finite-v1".into(),
    }
}
fn fixture_job(dataset: char, recipe: char) -> WorkerJob {
    let request = EvaluationRequest {
        schema: REQUEST_SCHEMA.into(),
        experiment_sha256: fixture_experiment().identity().unwrap(),
        dataset_sha256: dataset.to_string().repeat(64),
        split_sha256: "b".repeat(64),
        fitting_rows_sha256: "c".repeat(64),
        scoring_rows_sha256: "d".repeat(64),
        fitting_rows: 3,
        scoring_rows: 2,
        recipe_sha256: recipe.to_string().repeat(64),
        evaluator_sha256: "e".repeat(64),
        checkpoint_sha256: None,
        task: Task::Regression,
        metric: Metric::Rmse,
        classes: vec![],
        fold: 0,
        stage: EvaluationStage::Screen,
        seed: 1,
        numerical_policy: "finite-v1".into(),
    };
    WorkerJob {
        directory: PathBuf::from("/private/fixture").join(request.identity().unwrap()),
        request,
        command: WorkerCommand {
            program: Artifact {
                path: PathBuf::from("/usr/bin/false"),
                sha256: "a".repeat(64),
            },
            args: vec![],
            inputs: vec![],
            environment: BTreeMap::new(),
        },
        outputs: vec![PathBuf::from("predictions.json")],
        arm: None,
    }
}
fn scratch() -> tempfile::TempDir {
    // Reuse the lane target tree; tests never create /tmp evidence.
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/tmp");
    fs::create_dir_all(&root).unwrap();
    tempfile::Builder::new()
        .prefix("execution-fixture-")
        .tempdir_in(fs::canonicalize(root).unwrap())
        .unwrap()
}
fn preflight_job(root: &Path) -> WorkerJob {
    let mut job = fixture_job('a', 'b');
    let program = root.join("program");
    let input = root.join("input");
    fs::write(&program, b"trusted program").unwrap();
    fs::write(&input, b"trusted input").unwrap();
    job.command.program = Artifact {
        sha256: hash_file(&program).unwrap(),
        path: program,
    };
    job.command.inputs = vec![Artifact {
        sha256: hash_file(&input).unwrap(),
        path: input,
    }];
    job
}

mod accounting;
mod preflight;
mod recovery;
mod setup_faults;
mod setup_lifecycle;
mod supervisor;

use setup_faults::{qualify_concurrency, qualify_setup_faults};
pub(super) use setup_lifecycle::{host_setup_probe, HostSetupStage};
use setup_lifecycle::{
    qualify_cancellation_and_resume, setup_group_drained, setup_host_job, HostSetupGuard,
    OwnedSetupOrphan,
};
