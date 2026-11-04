use super::*;

pub(crate) fn run(
    manifest_path: &Path,
    arm: Arm,
    selection_mode: SelectionMode,
    resume: bool,
    workers: usize,
    selected_dataset: Option<&str>,
) -> Result<PathBuf> {
    run_arm(
        manifest_path,
        arm,
        selection_mode,
        resume,
        workers,
        selected_dataset,
    )
}

pub(crate) fn validate(
    study_root: &Path,
    run_id: &str,
    sotarena_root: &Path,
    workers: usize,
    selected_dataset: Option<&str>,
) -> Result<PathBuf> {
    validate_run(study_root, run_id, sotarena_root, workers, selected_dataset)
}

pub(crate) fn score(
    study_root: &Path,
    run_id: &str,
    sotarena_root: &Path,
    workers: usize,
) -> Result<PathBuf> {
    score_run(study_root, run_id, sotarena_root, workers)
}

pub(crate) fn score_smart_v3(
    study_root: &Path,
    run_id: &str,
    sotarena_root: &Path,
    workers: usize,
) -> Result<PathBuf> {
    results::score_smart_v3_run(study_root, run_id, sotarena_root, workers)
}

pub(crate) fn report(
    study_root: &Path,
    run_id: &str,
    sotarena_root: &Path,
    workers: usize,
) -> Result<PathBuf> {
    report_run(study_root, run_id, sotarena_root, workers)
}

pub(crate) fn run_arm(
    manifest_path: &Path,
    arm: Arm,
    selection_mode: SelectionMode,
    resume: bool,
    workers: usize,
    selected_dataset: Option<&str>,
) -> Result<PathBuf> {
    execution::run_arm(
        manifest_path,
        arm,
        selection_mode,
        resume,
        workers,
        selected_dataset,
    )
}

pub(crate) fn validate_run(
    study_root: &Path,
    run_id: &str,
    sotarena_root: &Path,
    workers: usize,
    selected_dataset: Option<&str>,
) -> Result<PathBuf> {
    results::validate_run(study_root, run_id, sotarena_root, workers, selected_dataset)
}

pub(crate) fn score_run(
    study_root: &Path,
    run_id: &str,
    sotarena_root: &Path,
    workers: usize,
) -> Result<PathBuf> {
    results::score_run(study_root, run_id, sotarena_root, workers)
}

pub(crate) fn report_run(
    study_root: &Path,
    run_id: &str,
    sotarena_root: &Path,
    workers: usize,
) -> Result<PathBuf> {
    results::report_run(study_root, run_id, sotarena_root, workers)
}
