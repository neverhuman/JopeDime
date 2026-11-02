#[derive(Debug, Serialize, Deserialize)]
pub struct Predictor {
    pub schema: String,
    pub task: String,
    pub canonical: String,
    pub training_sha256: String,
    pub feature_count: usize,
    pub preprocessing: Preprocessing,
    pub model: Model,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fit_completion: Option<serde_json::Value>,
}

fn cpu_evaluator_binding(value: Option<std::ffi::OsString>) -> Result<std::path::PathBuf> {
    use std::os::unix::fs::PermissionsExt;
    let path = value
        .map(std::path::PathBuf::from)
        .context("explicit JOPE_CPU_EVALUATOR binding required; evaluator unavailable")?;
    ensure!(path.is_absolute(), "CPU evaluator path must be absolute");
    let metadata = fs::metadata(&path).context("CPU evaluator executable unavailable")?;
    ensure!(
        metadata.is_file() && metadata.permissions().mode() & 0o111 != 0,
        "CPU evaluator must be an executable regular file"
    );
    Ok(path.canonicalize()?)
}

/// Shipped builds start the explicitly bound evaluator executable directly.
#[cfg(not(test))]
fn evaluator_command(executable: &Path) -> Command {
    Command::new(executable)
}
/// A unit-test build has no product CLI: the same subcommand argv is handed to
/// this test executable's ignored fixture entry. Launcher only; it grants no
/// authority and changes no fit, witness or activation rule.
#[cfg(test)]
fn evaluator_command(executable: &Path) -> Command {
    let mut command = Command::new(executable);
    command.args(fit_contract_tests::CLI_ENTRY_ARGS);
    command
}
