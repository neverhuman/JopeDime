//! Native training admission and supervised Python orchestration. No engine/Torch dependency.
use anyhow::{bail, ensure, Context, Result};
use clap::{Args, Parser, Subcommand};
use jope_learning::{canon::canonical_json, eligibility::freeze_teaching, state::State};
use jope_pipeline::{
    execution::{Artifact, ExecutionPool, Supervisor},
    experiment::{ExecutionWindow, Experiment, Resources, EXPERIMENT_SCHEMA},
    supervisor,
};
use nix::sys::signal::{kill, Signal};
use nix::unistd::Pid;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    os::unix::{
        fs::{OpenOptionsExt, PermissionsExt},
        process::CommandExt,
    },
    path::{Path, PathBuf},
    process::{Command, ExitCode, Stdio},
    sync::atomic::Ordering,
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

const SCHEMA: &str = "jope.native-training-plan/v1";
const NUMERICS: &str =
    "torch-policy-v3:float32-storage:cuda-tf32=true:amp=false:threads=1:experimental/v1";
const LINEAGE: &str = "asm.experimental-python-policy-v3/v1";

#[cfg(feature = "qualified-training-context")]
mod corpus;

#[derive(Parser)]
pub struct TrainCli {
    #[command(subcommand)]
    command: TrainCommand,
}
#[derive(Subcommand)]
enum TrainCommand {
    /// Materialize admitted real regression corpus with verified contexts and family partition.
    #[cfg(feature = "qualified-training-context")]
    PrepareInputs(corpus::PrepareArgs),
    /// Construct an authenticated corpus request from admitted comparisons and retained context files.
    #[cfg(feature = "qualified-training-context")]
    PrepareRequest(corpus::RequestArgs),
    /// Freeze an experimental training plan; no training clock starts.
    Plan(PlanArgs),
    /// Start one supervised training window; real eligibility is recomputed.
    Run(RunArgs),
    /// Continue the original window and compatible optimizer state.
    Resume(RunArgs),
    /// Prepare immutable fresh/replay siblings using an already admitted window.
    ComparePlan(CompareArgs),
    /// Run/resume both siblings under the original supervised category cutoff.
    Compare(RunArgs),
    /// Reload and verify changed parameters, optimizer continuation and receipts.
    Verify(StateArgs),
    /// Export verified private experimental training artifacts.
    Export(ExportArgs),
    /// Internal gate: enter the controller-owned subgroup before Python exec.
    #[command(hide = true)]
    PythonWorker {
        #[arg(long)]
        request: PathBuf,
        #[arg(long)]
        request_sha256: String,
    },
}
#[derive(Args)]
struct PlanArgs {
    #[arg(long)]
    state: PathBuf,
    /// Installed Python environment with the Jope training package.
    #[arg(long)]
    python: PathBuf,
    #[arg(long, conflicts_with = "synthetic_fixture")]
    inputs: Option<PathBuf>,
    #[arg(long)]
    synthetic_fixture: bool,
    #[arg(long, default_value = "all-admitted", value_parser = ["regression", "binary", "multiclass", "all-admitted"])]
    task: String,
    #[arg(long, default_value = "cuda:0", value_parser = ["cuda:0", "cpu"])]
    device: String,
    #[arg(long)]
    source_sha: String,
    /// Learning store, distinct from this training campaign's state.
    #[arg(long)]
    learning_state: Option<PathBuf>,
    /// Controller-only receipt from prepare-inputs; never part of the Python input closure.
    #[arg(long)]
    preparation_receipt: Option<PathBuf>,
    #[arg(long, default_value_t = 7)]
    seed: u32,
    #[arg(long, default_value_t = 2)]
    passes: u16,
}
#[derive(Args)]
struct StateArgs {
    #[arg(long)]
    state: PathBuf,
}
#[derive(Args)]
struct ExportArgs {
    #[arg(long)]
    state: PathBuf,
    #[arg(long)]
    output: PathBuf,
}
#[derive(Args)]
struct CompareArgs {
    #[arg(long)]
    state: PathBuf,
    #[arg(long, value_parser = ["regression", "binary", "multiclass"])]
    task: String,
    #[arg(long)]
    fresh_inputs: PathBuf,
    #[arg(long)]
    fresh_preparation: Option<PathBuf>,
    #[arg(long)]
    updates: u64,
}
#[derive(Args)]
struct RunArgs {
    #[arg(long)]
    state: PathBuf,
    #[arg(long, hide = true)]
    supervisor_unit: Option<String>,
    #[arg(long, hide = true)]
    native_plan_sha256: Option<String>,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct NativePlan {
    schema: String,
    plan: Artifact,
    experiment: Artifact,
    native: Artifact,
    python: Artifact,
    python_invocation: PathBuf,
    environment: BTreeMap<String, String>,
    learning_state: Option<PathBuf>,
    #[serde(default)]
    input_preparation: Option<Artifact>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct LaunchRequest {
    schema: String,
    state: PathBuf,
    native_plan_sha256: String,
    operation: String,
    unit: String,
    nonce: String,
    deadline_unix_ms: u64,
    admission: Artifact,
}

fn now_ms() -> Result<u64> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)?
        .as_millis()
        .try_into()?)
}
fn nonce() -> Result<String> {
    let value = fs::read_to_string("/proc/sys/kernel/random/uuid")?
        .trim()
        .to_owned();
    ensure!(uuid(&value), "kernel nonce invalid");
    Ok(value)
}
fn uuid(value: &str) -> bool {
    value.len() == 36
        && value.bytes().enumerate().all(|(i, b)| {
            if [8, 13, 18, 23].contains(&i) {
                b == b'-'
            } else {
                b.is_ascii_digit() || (b'a'..=b'f').contains(&b)
            }
        })
}
fn valid_unit(unit: &str, experiment: &Experiment) -> Result<()> {
    let prefix = format!("jope-asm-{}-", experiment.identity()?);
    let invocation = unit
        .strip_prefix(&prefix)
        .and_then(|s| s.strip_suffix(".service"));
    ensure!(
        invocation.is_some_and(uuid),
        "supervisor unit differs from original training experiment"
    );
    Ok(())
}
fn digest(value: &impl Serialize) -> Result<String> {
    Ok(jope_pipeline::sha256_hex(&canonical_json(value)?))
}
fn read<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    ensure!(
        fs::metadata(path)?.len() <= 32 << 20,
        "training metadata exceeds32MiB"
    );
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}
fn artifact(path: &Path) -> Result<Artifact> {
    use sha2::{Digest, Sha256};
    let path = fs::canonicalize(path)?;
    let mut file = File::open(&path)?;
    ensure!(file.metadata()?.is_file(), "artifact is not a regular file");
    let mut hash = Sha256::new();
    let mut buffer = [0; 65536];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    Ok(Artifact {
        path,
        sha256: format!("{:x}", hash.finalize()),
    })
}
fn immutable(path: &Path, value: &impl Serialize) -> Result<()> {
    let bytes = canonical_json(value)?;
    if path.try_exists()? {
        ensure!(
            fs::read(path)? == bytes,
            "immutable training receipt changed: {}",
            path.display()
        );
        return Ok(());
    }
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    File::open(path.parent().context("receipt parent missing")?)?.sync_all()?;
    Ok(())
}
fn directory(path: &Path, create: bool) -> Result<PathBuf> {
    ensure!(
        path.is_absolute() && !path.starts_with("/tmp"),
        "training state requires durable absolute path"
    );
    if create {
        fs::create_dir_all(path)?;
    }
    let path = fs::canonicalize(path)?;
    ensure!(
        !path.starts_with("/tmp") && !path.ancestors().any(|p| p.join(".git").exists()),
        "private training state must stay outside Git working trees and /tmp"
    );
    fs::set_permissions(&path, fs::Permissions::from_mode(0o700))?;
    Ok(path)
}
fn invocation_path(path: &Path) -> Result<PathBuf> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()?.join(path)
    };
    Ok(fs::canonicalize(
        absolute
            .parent()
            .context("Python invocation parent missing")?,
    )?
    .join(
        absolute
            .file_name()
            .context("Python invocation file missing")?,
    ))
}
fn child_environment(state: &Path, device: &str, seed: u64) -> BTreeMap<String, String> {
    let mut env = BTreeMap::from([
        ("PATH".into(), "/usr/bin:/bin".into()),
        (
            "TMPDIR".into(),
            state.join("python-tmp").display().to_string(),
        ),
        ("PYTHONNOUSERSITE".into(), "1".into()),
        ("PYTHONSAFEPATH".into(), "1".into()),
        ("PYTHONDONTWRITEBYTECODE".into(), "1".into()),
        ("PYTHONHASHSEED".into(), seed.to_string()),
        (
            "CUDA_VISIBLE_DEVICES".into(),
            if device == "cpu" { "" } else { "0" }.into(),
        ),
        ("WORLD_SIZE".into(), "1".into()),
    ]);
    for key in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "RAYON_NUM_THREADS",
    ] {
        env.insert(key.into(), "1".into());
    }
    env
}
fn text<'a>(value: &'a Value, key: &str) -> Result<&'a str> {
    value[key]
        .as_str()
        .with_context(|| format!("missing training field {key}"))
}
fn tasks(plan: &Value) -> Result<Vec<String>> {
    let values = plan["tasks"].as_array().context("training tasks absent")?;
    let result = values
        .iter()
        .map(|v| v.as_str().map(str::to_owned).context("task must be text"))
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        !result.is_empty()
            && result.len() <= 3
            && result
                .iter()
                .all(|task| ["regression", "binary", "multiclass"].contains(&task.as_str()))
            && result.iter().collect::<BTreeSet<_>>().len() == result.len(),
        "invalid training tasks"
    );
    Ok(result)
}
fn experiment_for_plan(plan: &Value) -> Result<Experiment> {
    let categories = plan["category_inputs"]
        .as_object()
        .context("category inputs absent")?;
    let mut checkpoints = BTreeMap::new();
    for (task, category) in categories {
        for (name, reference) in category["inputs"]
            .as_object()
            .context("category artifacts absent")?
        {
            let reference: Artifact = serde_json::from_value(reference.clone())?;
            checkpoints.insert(format!("{task}:{name}"), reference.sha256);
        }
    }
    let result = Experiment {
        schema: EXPERIMENT_SCHEMA.into(),
        panel_sha256: text(plan, "plan_sha256")?.into(),
        methods_sha256: digest(&plan["runtime"])?,
        source_sha: text(plan, "source_sha")?.into(),
        checkpoints,
        seed: plan["seed"].as_u64().context("seed absent")?,
        resources: Resources {
            workers: 1,
            cpus: 20,
            memory_bytes: 64 << 30,
            swap_bytes: 0,
            gpu_tasks: u16::from(text(plan, "device")? == "cuda:0"),
            threads_per_worker: 1,
            total_seconds: 3600,
            reporting_seconds: 5,
            per_fit_seconds: 1200,
        },
        expected_artifacts: tasks(plan)?
            .into_iter()
            .map(|task| format!("{task}/head.json"))
            .collect(),
        protocol: "asm-training-v1:category=1200:total=3600:continuation-required".into(),
        numerical_policy: NUMERICS.into(),
    };
    result.validate()?;
    Ok(result)
}
fn load_native(
    state: &Path,
    expected: Option<&str>,
    verify_programs: bool,
) -> Result<(NativePlan, Value, Experiment)> {
    let path = state.join("native-plan.json");
    let identity = artifact(&path)?;
    if let Some(expected) = expected {
        ensure!(
            identity.sha256 == expected,
            "native plan changed since launch"
        );
    }
    let native: NativePlan = read(&path)?;
    ensure!(
        native.schema == SCHEMA
            && native.plan.path == state.join("plan.json")
            && native.experiment.path == state.join("experiment.json"),
        "native training plan paths/schema differ"
    );
    native.plan.verify()?;
    native.experiment.verify()?;
    let plan: Value = read(&native.plan.path)?;
    let mut body = plan.clone();
    body.as_object_mut()
        .context("training plan must be object")?
        .remove("plan_sha256");
    ensure!(
        text(&plan, "schema")? == "jope.training-plan/v1"
            && text(&plan, "lineage")? == LINEAGE
            && digest(&body)? == text(&plan, "plan_sha256")?,
        "training plan checksum/lineage changed"
    );
    ensure!(
        ["cpu", "cuda:0"].contains(&text(&plan, "device")?),
        "training device unsupported"
    );
    if text(&plan, "device")? == "cpu" {
        ensure!(
            plan["qualification_only"] == true && plan["data_kind"] == "synthetic_fixture",
            "CPU must be explicit synthetic qualification"
        );
    }
    ensure!(
        native.environment
            == child_environment(
                state,
                text(&plan, "device")?,
                plan["seed"].as_u64().context("seed missing")?
            ),
        "native Python environment changed"
    );
    let experiment = experiment_for_plan(&plan)?;
    let stored: Experiment = read(&native.experiment.path)?;
    ensure!(
        stored.identity()? == experiment.identity()?,
        "Python/native experiment identity differs"
    );
    let pinned_python: Artifact = serde_json::from_value(plan["runtime"]["python"].clone())?;
    ensure!(
        pinned_python.path == native.python.path && pinned_python.sha256 == native.python.sha256,
        "Python differs from runtime plan"
    );
    ensure!(
        plan["runtime"]["python_invocation"].as_str() == native.python_invocation.to_str()
            && fs::canonicalize(&native.python_invocation)? == native.python.path,
        "Python virtualenv invocation changed"
    );
    if verify_programs {
        native.native.verify()?;
        native.python.verify()?;
        let running = artifact(&std::env::current_exe()?)?;
        ensure!(
            running.path == native.native.path && running.sha256 == native.native.sha256,
            "native CLI changed since planning"
        );
    }
    Ok((native, plan, experiment))
}
fn python_command(native: &NativePlan, state: &Path, operation: &str) -> Command {
    let mut command = Command::new(&native.python_invocation);
    command
        .env_clear()
        .envs(&native.environment)
        .current_dir(state)
        .args([
            "-I",
            "-B",
            "-m",
            "jope.policy.training_campaign",
            operation,
            "--state",
        ])
        .arg(state)
        .stdin(Stdio::null());
    command
}
fn passive_command(mut command: Command, state: &Path, operation: &str) -> Result<Value> {
    let root = state.join("native-invocations").join(nonce()?);
    fs::create_dir_all(&root)?;
    let stdout = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(root.join("stdout.log"))?;
    let stderr = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(root.join("stderr.log"))?;
    let started = now_ms()?;
    let status = command.stdout(stdout).stderr(stderr).status();
    let receipt = match status {
        Ok(status) => json!({"schema":"jope.native-training-invocation/v1","operation":operation,
            "started_unix_ms":started,"ended_unix_ms":now_ms()?,"exit_code":status.code(),"ok":status.success(),
            "stdout":artifact(&root.join("stdout.log"))?,"stderr":artifact(&root.join("stderr.log"))?,
            "execution_budget_started":false}),
        Err(error) => json!({"schema":"jope.native-training-invocation/v1","operation":operation,
            "started_unix_ms":started,"ended_unix_ms":now_ms()?,"exit_code":null,"ok":false,"reason":error.to_string(),
            "execution_budget_started":false}),
    };
    immutable(&root.join("outcome.json"), &receipt)?;
    ensure!(
        receipt["ok"] == true,
        "Python {operation} failed; see {}",
        root.display()
    );
    Ok(json!({"ok":true,"operation":operation,"receipt":artifact(&root.join("outcome.json"))?}))
}
fn plan(args: PlanArgs) -> Result<Value> {
    ensure!(
        args.synthetic_fixture != args.inputs.is_some(),
        "select inputs or explicit synthetic fixture"
    );
    ensure!(
        args.device != "cpu" || args.synthetic_fixture,
        "CPU is synthetic qualification only"
    );
    ensure!(
        (2..=1000).contains(&args.passes),
        "continuation requires2..1000 passes"
    );
    let state = directory(&args.state, true)?;
    fs::create_dir_all(state.join("python-tmp"))?;
    let python_invocation = invocation_path(&args.python)?;
    let python = artifact(&python_invocation)?;
    let native_program = artifact(&std::env::current_exe()?)?;
    let environment = child_environment(&state, &args.device, args.seed.into());
    let mut command = Command::new(&python_invocation);
    command
        .env_clear()
        .envs(&environment)
        .current_dir(&state)
        .stdin(Stdio::null())
        .args([
            "-I",
            "-B",
            "-m",
            "jope.policy.training_campaign",
            "plan",
            "--state",
        ])
        .arg(&state)
        .args([
            "--task",
            &args.task,
            "--device",
            &args.device,
            "--source-sha",
            &args.source_sha,
            "--seed",
            &args.seed.to_string(),
            "--passes",
            &args.passes.to_string(),
        ]);
    if let Some(inputs) = args.inputs {
        command.arg("--inputs").arg(fs::canonicalize(inputs)?);
    } else {
        command.arg("--synthetic-fixture");
    }
    let receipt = passive_command(command, &state, "plan")?;
    let python_plan: Value = read(&state.join("plan.json"))?;
    let has_admitted = tasks(&python_plan)?
        .iter()
        .any(|task| python_plan["category_inputs"][task]["data_kind"] == "admitted");
    let native = NativePlan {
        schema: SCHEMA.into(),
        plan: artifact(&state.join("plan.json"))?,
        experiment: artifact(&state.join("experiment.json"))?,
        native: native_program,
        python,
        python_invocation,
        environment,
        learning_state: if !has_admitted {
            None
        } else {
            Some(fs::canonicalize(
                State::resolve(args.learning_state)?.root(),
            )?)
        },
        input_preparation: args
            .preparation_receipt
            .as_ref()
            .map(|path| artifact(&fs::canonicalize(path)?))
            .transpose()?,
    };
    immutable(&state.join("native-plan.json"), &native)?;
    let (_, plan, experiment) = load_native(&state, None, true)?;
    Ok(
        json!({"ok":true,"plan":native.plan,"native_plan":artifact(&state.join("native-plan.json"))?,
        "experiment_sha256":experiment.identity()?,"tasks":plan["tasks"],"fallbacks":plan["fallbacks"],
        "qualification_only":plan["qualification_only"],"execution_budget_started":false,"invocation":receipt}),
    )
}

/// Only native producer verifiers may construct these; there is no JSON/deserialization route.
#[derive(Serialize)]
struct VerifiedTrainingContext {
    task: String,
    context_id: String,
    artifact_sha256: String,
    producer_proof_sha256: String,
}
trait ContextVerifier {
    fn verify(
        &self,
        native: &NativePlan,
        plan: &Value,
        teaching: Option<&jope_learning::eligibility::TeachingManifest>,
    ) -> Result<Vec<VerifiedTrainingContext>>;
}
struct ProducerVerifier;
impl ContextVerifier for ProducerVerifier {
    fn verify(
        &self,
        native: &NativePlan,
        plan: &Value,
        teaching: Option<&jope_learning::eligibility::TeachingManifest>,
    ) -> Result<Vec<VerifiedTrainingContext>> {
        if tasks(plan)?
            .iter()
            .all(|task| plan["category_inputs"][task]["data_kind"] == "synthetic_fixture")
        {
            return Ok(vec![]);
        }
        #[cfg(feature = "qualified-training-context")]
        {
            qualified_cpu::verify(
                native,
                plan,
                teaching.context("measured teaching manifest required")?,
            )
        }
        #[cfg(not(feature = "qualified-training-context"))]
        {
            let _ = (native, teaching);
            bail!("real context admission requires the qualified-training-context feature");
        }
    }
}

fn admission(
    native: &NativePlan,
    plan: &Value,
    verifier: &dyn ContextVerifier,
    replay: bool,
) -> Result<Value> {
    let mut required = Vec::new();
    for task in tasks(plan)? {
        let category = &plan["category_inputs"][&task];
        match text(category, "data_kind")? {
            "synthetic_fixture" => (),
            "admitted" => required.push(serde_json::from_value::<Artifact>(
                category["inputs"]["admission"].clone(),
            )?),
            other => bail!("unsupported category data kind {other}"),
        }
    }
    let qualified = if required.is_empty() {
        None
    } else {
        let state = State::at(
            native
                .learning_state
                .as_deref()
                .context("real training needs pinned learning State")?,
        );
        let mut current = freeze_teaching(&state)?;
        let canonical = canonical_json(&current)?;
        controller_request(native, plan)?;
        let mut replay_entries = Vec::new();
        for requested in required {
            requested.verify()?;
            let requested: Value = read(&requested.path)?;
            ensure!(
                requested["schema"] == "jope.qualified-teaching-freeze/v1"
                    && (replay
                        || requested["projection_of_sha256"]
                            == jope_pipeline::sha256_hex(&canonical)),
                "teaching eligibility changed since plan; original campaign remains frozen"
            );
            let entries = requested["entries"]
                .as_array()
                .context("projected teaching entries absent")?;
            ensure!(!entries.is_empty(), "empty projected teaching admission");
            for entry in entries {
                if replay {
                    replay_entries.push(serde_json::from_value(entry.clone())?);
                } else {
                    ensure!(
                        current
                            .entries
                            .iter()
                            .any(|q| serde_json::to_value(q).ok().as_ref() == Some(entry)),
                        "projected teaching entry differs from current admission"
                    );
                }
            }
        }
        if replay {
            current = jope_learning::eligibility::revalidate_replay(&state, &replay_entries)?;
        }
        Some(current)
    };
    let contexts = verifier.verify(native, plan, qualified.as_ref())?;
    let qualified_sha256 = qualified.as_ref().map(digest).transpose()?;
    Ok(
        json!({"schema":"jope.native-training-admission/v1","qualified_teaching_sha256":qualified_sha256,
        "contexts":contexts,"lineage":LINEAGE,"python_started":false}),
    )
}

fn comparison_admission(
    state: &Path,
    native: &NativePlan,
    plan: &Value,
    verifier: &dyn ContextVerifier,
) -> Result<()> {
    let comparison: Value = read(&state.join("comparison.json"))?;
    ensure!(
        comparison["schema"] == "jope.training-comparison/v1"
            && comparison["plan_sha256"] == plan["plan_sha256"],
        "comparison source plan differs"
    );
    let task = text(&comparison, "task")?;
    ensure!(
        tasks(plan)?.iter().any(|t| t == task),
        "comparison task unplanned"
    );
    let fresh_ref: Artifact = serde_json::from_value(comparison["fresh_inputs"].clone())?;
    fresh_ref.verify()?;
    let mut inputs: Value = read(&fresh_ref.path)?;
    ensure!(
        inputs["schema"] == "jope.experimental-training-inputs/v1",
        "fresh input schema differs"
    );
    inputs
        .as_object_mut()
        .context("fresh inputs object required")?
        .remove("schema");
    let mut fresh_plan = plan.clone();
    fresh_plan["tasks"] = json!([task]);
    fresh_plan["requested_manifest"] = serde_json::to_value(&fresh_ref)?;
    fresh_plan["category_inputs"] =
        json!({task:{"data_kind":plan["category_inputs"][task]["data_kind"],"inputs":inputs}});
    let mut fresh_native = native.clone();
    fresh_native.input_preparation =
        serde_json::from_value(comparison["fresh_preparation"].clone())?;
    admission(&fresh_native, &fresh_plan, verifier, false)?;
    Ok(())
}

fn controller_request(native: &NativePlan, plan: &Value) -> Result<Value> {
    let receipt = native
        .input_preparation
        .as_ref()
        .context("real training needs controller preparation receipt")?;
    receipt.verify()?;
    let preparation: Value = read(&receipt.path)?;
    ensure!(
        preparation["schema"] == "jope.real-training-corpus-result/v1"
            && preparation["inputs"] == plan["requested_manifest"],
        "preparation receipt does not bind projected inputs"
    );
    let request: Artifact = serde_json::from_value(preparation["request"].clone())?;
    request.verify()?;
    let value: Value = read(&request.path)?;
    ensure!(
        value["schema"] == "jope.real-training-corpus-request/v1",
        "controller request schema differs"
    );
    Ok(value)
}
fn subgroup(supervisor: &Supervisor, invocation: &str) -> Result<PathBuf> {
    ensure!(uuid(invocation), "invalid controller-owned subgroup nonce");
    let path = supervisor
        .cgroup_root
        .join(format!("training-{invocation}"));
    ensure!(
        path.parent() == Some(supervisor.cgroup_root.as_path()),
        "subgroup escape"
    );
    Ok(path)
}
fn populated(path: &Path) -> Result<bool> {
    let text = fs::read_to_string(path.join("cgroup.events"))?;
    let value = text
        .lines()
        .find_map(|line| line.strip_prefix("populated "))
        .context("cgroup population unavailable")?;
    ensure!(matches!(value, "0" | "1"), "invalid cgroup population");
    Ok(value == "1")
}
fn drain(path: &Path, receipt: &Path) -> Result<()> {
    let started = Instant::now();
    let mut reported = false;
    loop {
        let kill_error = fs::write(path.join("cgroup.kill"), "1").err();
        if matches!(populated(path), Ok(false)) {
            return Ok(());
        }
        if started.elapsed() >= Duration::from_secs(1) && !reported {
            // Keep the host reservation while cleanup is unresolved. The enclosing
            // service retains its independently enforced original RuntimeMax/kill bound.
            let reason = kill_error
                .map(|e| e.to_string())
                .unwrap_or_else(|| "descendants remain populated".into());
            let _ = immutable(
                receipt,
                &json!({"schema":"jope.training-drain-pending/v1","reason":reason,
                "subgroup":path,"host_reservation_retained":true,"at_unix_ms":now_ms().ok()}),
            );
            reported = true;
        }
        thread::sleep(Duration::from_millis(10));
    }
}

fn supervised(args: RunArgs, operation: &str, verifier: &dyn ContextVerifier) -> Result<Value> {
    let state = directory(&args.state, false)?;
    let (native, plan, experiment) =
        load_native(&state, args.native_plan_sha256.as_deref(), false)?;
    let native_sha = artifact(&state.join("native-plan.json"))?.sha256;
    let Some(unit) = args.supervisor_unit else {
        let running = artifact(&std::env::current_exe()?)?;
        ensure!(
            running.path == native.native.path && running.sha256 == native.native.sha256,
            "native CLI differs from plan"
        );
        let launched_at = now_ms()?;
        let launch = supervisor::launch(
            &experiment,
            &state,
            &running,
            &[
                "train".into(),
                operation.into(),
                "--state".into(),
                state.display().to_string(),
                "--native-plan-sha256".into(),
                native_sha,
            ],
        )?;
        return Ok(json!({"ok":launch.exit_code==Some(0),"launch":launch,
            "training_report":current_training_report(&state,&plan,launched_at,operation)?}));
    };
    ensure!(
        args.native_plan_sha256.is_some(),
        "supervised training needs native plan pin"
    );
    valid_unit(&unit, &experiment)?;
    let supervisor = supervisor::for_unit(&unit)?;
    let mut pool = ExecutionPool::open(&experiment, &state, supervisor.clone())?;
    pool.reserve()?;
    let window = pool.admit()?.clone();
    let root = state.join("native-invocations").join(nonce()?);
    fs::create_dir_all(&root)?;
    let started = now_ms()?;
    immutable(
        &root.join("admitted-window.json"),
        &json!({"schema":"jope.native-training-window/v1",
        "window":window,"experiment_sha256":experiment.identity()?,"native_plan_sha256":native_sha,
        "admitted_before_python":true,"admitted_unix_ms":started}),
    )?;
    let result = (|| -> Result<Value> {
        load_native(&state, Some(&native_sha), true)?;
        let admitted = admission(&native, &plan, verifier, operation == "compare")?;
        if operation == "compare" {
            comparison_admission(&state, &native, &plan, verifier)?;
        }
        immutable(
            &root.join("admission.json"),
            &json!({"preflight":admitted,"window":window,
            "native_plan_sha256":native_sha,"completed_unix_ms":now_ms()?,
            "comparison":if operation == "compare" {Some(artifact(&state.join("comparison.json"))?)} else {None}}),
        )?;
        let request = LaunchRequest {
            schema: "jope.training-child/v1".into(),
            state: state.clone(),
            native_plan_sha256: native_sha.clone(),
            operation: operation.into(),
            unit: unit.clone(),
            nonce: root
                .file_name()
                .context("invocation missing")?
                .to_str()
                .context("invocation not UTF8")?
                .into(),
            deadline_unix_ms: window.deadline_unix_ms
                - experiment.resources.reporting_seconds * 1000,
            admission: artifact(&root.join("admission.json"))?,
        };
        launch_python(&request, &native, &root, &supervisor)
    })();
    let mut outcome = match result {
        Ok(value) => value,
        Err(error) => json!({"ok":false,"status":"refused","reason":format!("{error:#}")}),
    };
    if let Some(report) = current_training_report(&state, &plan, started, operation)? {
        outcome["training_report"] = report;
    }
    immutable(
        &root.join("native-outcome.json"),
        &json!({"schema":"jope.native-training-outcome/v1",
        "operation":operation,"native_plan_sha256":native_sha,"window":window,"started_unix_ms":started,
        "ended_unix_ms":now_ms()?,"cpu_seconds":null,"gpu_seconds":null,
        "accounting_reason":"active CPU/GPU time unmeasured; original wall reservation retained","outcome":outcome}),
    )?;
    // launch_python drains the owned subgroup before this reservation is released.
    drop(pool);
    Ok(
        json!({"ok":outcome["ok"]==true,"outcome":outcome,"receipt":artifact(&root.join("native-outcome.json"))?}),
    )
}
fn launch_python(
    request: &LaunchRequest,
    native: &NativePlan,
    root: &Path,
    supervisor: &Supervisor,
) -> Result<Value> {
    let group = subgroup(supervisor, &request.nonce)?;
    fs::create_dir(&group)?;
    let mut child = None;
    let operation = (|| -> Result<Value> {
        immutable(&root.join("request.json"), request)?;
        let pin = artifact(&root.join("request.json"))?;
        let stdout = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(root.join("stdout.log"))?;
        let stderr = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(root.join("stderr.log"))?;
        let cancelled = supervisor::cancellation()?;
        child = Some(
            Command::new(&native.native.path)
                .args(["train", "python-worker", "--request"])
                .arg(&pin.path)
                .args(["--request-sha256", &pin.sha256])
                .env_clear()
                .envs(&native.environment)
                .stdin(Stdio::null())
                .stdout(stdout)
                .stderr(stderr)
                .spawn()
                .context("spawn gated Python controller")?,
        );
        let process = child.as_mut().context("child absent")?;
        let mut released = false;
        let mut stop_reason = None;
        let status = loop {
            if let Some(status) = process.try_wait()? {
                break status;
            }
            if cancelled.load(Ordering::Acquire) || now_ms()? >= request.deadline_unix_ms {
                stop_reason = Some(if cancelled.load(Ordering::Acquire) {
                    "cancelled"
                } else {
                    "budget_exhausted"
                });
                let _ = kill(Pid::from_raw(process.id().try_into()?), Signal::SIGTERM);
                thread::sleep(Duration::from_millis(50));
                let _ = process.kill();
                break process.wait()?;
            }
            if !released && root.join("joined.json").try_exists()? {
                let joined: Value = read(&root.join("joined.json"))?;
                ensure!(
                    joined["pid"] == process.id() && joined["request_sha256"] == pin.sha256,
                    "child gate receipt differs"
                );
                let actual = fs::read_to_string(format!("/proc/{}/cgroup", process.id()))?;
                ensure!(
                    actual.lines().any(|line| line
                        == format!(
                            "0::/system.slice/{}/training-{}",
                            request.unit, request.nonce
                        )),
                    "Python gate is outside owned subgroup"
                );
                immutable(
                    &root.join("release.json"),
                    &json!({"request_sha256":pin.sha256}),
                )?;
                released = true;
            }
            thread::sleep(Duration::from_millis(10));
        };
        Ok(
            json!({"ok":status.success()&&stop_reason.is_none()&&released,
            "status":stop_reason.unwrap_or(if status.success(){"completed"}else{"failed"}),
            "exit_code":status.code(),"request":pin,"python_gate_released":released,
            "stdout":artifact(&root.join("stdout.log"))?,"stderr":artifact(&root.join("stderr.log"))?}),
        )
    })();
    // Cleanup executes on success, cancellation, failed gate and every fallible
    // operation after spawn. No Python session can escape the owned cgroup.
    if let Some(process) = child.as_mut() {
        let _ = process.kill();
        let _ = process.wait();
    }
    drain(&group, &root.join("drain-pending.json"))?;
    fs::remove_dir(&group)?;
    immutable(
        &root.join("drained.json"),
        &json!({"schema":"jope.training-drained/v1","subgroup":group,
        "ended_unix_ms":now_ms()?,"populated":false}),
    )?;
    let mut outcome = operation?;
    outcome["subgroup_drained"] = json!(true);
    Ok(outcome)
}

fn current_training_report(
    state: &Path,
    plan: &Value,
    started: u64,
    operation: &str,
) -> Result<Option<Value>> {
    let (filename, schema) = if operation == "compare" {
        (
            "comparison-report.json",
            "jope.training-comparison-result/v1",
        )
    } else {
        ("report.json", "jope.training-run/v1")
    };
    let path = state.join(filename);
    if !path.is_file() {
        return Ok(None);
    }
    let report: Value = read(&path)?;
    if report["schema"] == schema
        && report["plan_sha256"] == plan["plan_sha256"]
        && report["reported_unix_ms"]
            .as_u64()
            .is_some_and(|t| t >= started)
    {
        Ok(Some(json!({"artifact":artifact(&path)?,"value":report})))
    } else {
        Ok(None)
    }
}

fn python_worker(path: &Path, expected: &str) -> Result<Value> {
    let pin = Artifact {
        path: fs::canonicalize(path)?,
        sha256: expected.into(),
    };
    pin.verify()?;
    let request: LaunchRequest = read(&pin.path)?;
    ensure!(
        request.schema == "jope.training-child/v1"
            && ["run", "resume", "compare"].contains(&request.operation.as_str()),
        "invalid training child request"
    );
    let state = directory(&request.state, false)?;
    let root = state.join("native-invocations").join(&request.nonce);
    ensure!(
        pin.path == root.join("request.json"),
        "child request is not controller-owned"
    );
    let (native, _, experiment) = load_native(&state, Some(&request.native_plan_sha256), true)?;
    valid_unit(&request.unit, &experiment)?;
    request.admission.verify()?;
    ensure!(
        request.admission.path == root.join("admission.json"),
        "unbound native admission"
    );
    let window: ExecutionWindow = read(&state.join("window.json"))?;
    window.validate(&experiment)?;
    let admitted: Value = read(&request.admission.path)?;
    if request.operation == "compare" {
        let comparison: Artifact = serde_json::from_value(admitted["comparison"].clone())?;
        comparison.verify()?;
        ensure!(
            comparison.path == state.join("comparison.json"),
            "comparison path differs"
        );
    }
    ensure!(
        admitted["window"] == serde_json::to_value(&window)?
            && admitted["native_plan_sha256"] == request.native_plan_sha256,
        "Python gate admission differs from original window/native plan"
    );
    ensure!(
        request.deadline_unix_ms
            == window.deadline_unix_ms - experiment.resources.reporting_seconds * 1000,
        "child changed original deadline"
    );
    let supervisor = supervisor::for_unit(&request.unit)?;
    let group = subgroup(&supervisor, &request.nonce)?;
    ensure!(
        fs::canonicalize(&group)? == group,
        "subgroup is not physical"
    );
    let own = fs::read_to_string("/proc/self/cgroup")?;
    ensure!(
        own.lines()
            .any(|line| line == format!("0::/system.slice/{}/controller", request.unit)),
        "gate did not inherit verified controller subgroup"
    );
    fs::write(group.join("cgroup.procs"), std::process::id().to_string())?;
    ensure!(
        fs::read_to_string("/proc/self/cgroup")?
            .lines()
            .any(|line| line
                == format!(
                    "0::/system.slice/{}/training-{}",
                    request.unit, request.nonce
                )),
        "subgroup join failed"
    );
    immutable(
        &root.join("joined.json"),
        &json!({"pid":std::process::id(),"request_sha256":pin.sha256}),
    )?;
    loop {
        ensure!(
            now_ms()? < request.deadline_unix_ms,
            "original training deadline exhausted at gate"
        );
        if root.join("release.json").try_exists()? {
            let release: Value = read(&root.join("release.json"))?;
            ensure!(
                release["request_sha256"] == pin.sha256,
                "release from another request"
            );
            break;
        }
        thread::sleep(Duration::from_millis(10));
    }
    let error = python_command(&native, &state, &request.operation)
        .args(["--supervisor-unit", &request.unit])
        .exec();
    Err(error).context("exec pinned Python training controller")
}
fn run(cli: TrainCli) -> Result<Value> {
    match cli.command {
        #[cfg(feature = "qualified-training-context")]
        TrainCommand::PrepareInputs(args) => corpus::prepare(args),
        #[cfg(feature = "qualified-training-context")]
        TrainCommand::PrepareRequest(args) => corpus::prepare_request(args),
        TrainCommand::Plan(args) => plan(args),
        TrainCommand::Run(args) => supervised(args, "run", &ProducerVerifier),
        TrainCommand::Resume(args) => supervised(args, "resume", &ProducerVerifier),
        TrainCommand::Compare(args) => supervised(args, "compare", &ProducerVerifier),
        TrainCommand::ComparePlan(args) => {
            let state = directory(&args.state, false)?;
            let (native, _, _) = load_native(&state, None, true)?;
            let mut command = python_command(&native, &state, "compare-plan");
            command
                .args(["--task", &args.task, "--updates", &args.updates.to_string()])
                .arg("--fresh-inputs")
                .arg(args.fresh_inputs);
            if let Some(receipt) = args.fresh_preparation {
                command.arg("--fresh-preparation").arg(receipt);
            }
            passive_command(command, &state, "compare-plan")
        }
        TrainCommand::Verify(args) => {
            let state = directory(&args.state, false)?;
            let (native, _, _) = load_native(&state, None, true)?;
            passive_command(python_command(&native, &state, "verify"), &state, "verify")
        }
        TrainCommand::Export(args) => {
            let state = directory(&args.state, false)?;
            ensure!(
                args.output.is_absolute() && !args.output.starts_with("/tmp"),
                "export requires durable absolute output"
            );
            let (native, _, _) = load_native(&state, None, true)?;
            let mut command = python_command(&native, &state, "export");
            command.arg("--output").arg(args.output);
            passive_command(command, &state, "export")
        }
        TrainCommand::PythonWorker {
            request,
            request_sha256,
        } => python_worker(&request, &request_sha256),
    }
}
pub fn execute(cli: TrainCli) -> ExitCode {
    match run(cli) {
        Ok(value) => {
            println!("{value}");
            if value["ok"] == true {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        Err(error) => {
            eprintln!(
                "{}",
                json!({"schema":"jope.native-training-error/v1","ok":false,"error":format!("{error:#}")})
            );
            ExitCode::FAILURE
        }
    }
}

#[cfg(feature = "qualified-training-context")]
mod qualified_cpu {
    use super::*;
    use jope_learning::eligibility::TeachingManifest;
    use jope_pipeline::{
        cpu_context_bridge::{
            load_qualified, AdmissionExpectation, CpuContextFiles, ExecutionProof,
        },
        experiment::PanelDataset,
        prepared_data::parse_development,
        search_context::TrainingIdentity,
    };
    use jope_policy::context_producer::{
        self, CpuContextArtifact, EncoderIdentity, TrainingInput, TrainingRow,
    };

    /// Independently frozen caller pins. Attestations embedded in bundle are never authority.
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct CpuProof {
        producer: Artifact,
        producer_source_sha40: String,
        input: Artifact,
        qualification: Artifact,
        reference_manifest: Artifact,
        vector: Artifact,
        artifact: Artifact,
        frozen_transforms: Artifact,
        bundle: Artifact,
        context_execution: ExecutionProof,
        case_executions: Vec<ExecutionProof>,
    }

    pub(super) fn verify(
        native: &NativePlan,
        plan: &Value,
        teaching: &TeachingManifest,
    ) -> Result<Vec<VerifiedTrainingContext>> {
        let state = State::at(
            native
                .learning_state
                .as_deref()
                .context("learning State missing")?,
        );
        let mut verified = Vec::new();
        let request = controller_request(native, plan)?;
        for task in tasks(plan)? {
            let category = &plan["category_inputs"][&task];
            if category["data_kind"] == "synthetic_fixture" {
                continue;
            }
            let index_ref: Artifact =
                serde_json::from_value(category["inputs"]["contexts"].clone())?;
            index_ref.verify()?;
            let index: Value = read(&index_ref.path)?;
            ensure!(
                index["schema"] == "jope.training-context-index/v1",
                "context index schema differs"
            );
            let encoder_ref: Artifact =
                serde_json::from_value(category["inputs"]["encoder"].clone())?;
            encoder_ref.verify()?;
            let encoder: EncoderIdentity = read(&encoder_ref.path)?;
            let records = index["contexts"]
                .as_object()
                .context("context index is absent")?;
            let before = verified.len();
            for (context_id, record) in records {
                if record["task"] != task {
                    continue;
                }
                let comparison = text(record, "comparison_sha256")?;
                let qualification = teaching
                    .entries
                    .iter()
                    .find(|q| {
                        q.comparison.task == task
                            && q.comparison.dataset_hash
                                == record["dataset_id"].as_str().unwrap_or_default()
                            && q.freeze.comparison_sha256 == comparison
                    })
                    .context("context lacks exact qualified comparison")?;
                // The measured-teaching verifier already authenticated this freeze. Re-read its
                // content-addressed dataset and development bytes rather than trusting index rows.
                let frozen = Artifact {
                    path: state.blob_path(&qualification.freeze.freeze_sha256),
                    sha256: qualification.freeze.freeze_sha256.clone(),
                };
                frozen.verify()?;
                let policy: Value = read(&frozen.path)?;
                let dataset: PanelDataset = serde_json::from_value(policy["dataset"].clone())?;
                ensure!(
                    dataset.source.sha256 == qualification.comparison.dataset_hash
                        && dataset.task.as_str() == task,
                    "frozen context dataset/cut differs"
                );
                let development = Artifact {
                    path: state.blob_path(&dataset.split.development_sha256),
                    sha256: dataset.split.development_sha256.clone(),
                };
                development.verify()?;
                let data = parse_development(&fs::read(&development.path)?, &dataset)?;
                if task != "regression" {
                    verified.push(verify_classifier(
                        &state,
                        qualification,
                        &policy,
                        &data,
                        record,
                        (&encoder_ref, &encoder),
                        (&task, context_id),
                    )?);
                    continue;
                }
                let proof_value = &request["contexts"][comparison];
                ensure!(
                    jope_learning::canon::canonical_sha256(proof_value)?
                        == record["cpu_proof_sha256"],
                    "controller proof differs from projected context identity"
                );
                let proof: CpuProof = serde_json::from_value(proof_value.clone())
                    .context("qualified CPU context proof absent")?;
                ensure!(
                    proof.producer_source_sha40 == text(plan, "source_sha")?,
                    "context worker source differs from training source"
                );
                for reference in [
                    &proof.producer,
                    &proof.input,
                    &proof.qualification,
                    &proof.reference_manifest,
                ] {
                    reference.verify()?;
                }
                ensure!(
                    fs::metadata(&proof.input.path)?.len() <= 128 << 20,
                    "CPU training input exceeds authenticated panel limit"
                );
                let input: TrainingInput = serde_json::from_slice(&fs::read(&proof.input.path)?)?;
                verify_selection(&state, qualification, &policy, &data, input.identity.cut)?;
                let cut = data.cut(input.identity.cut)?;
                let rows = cut
                    .train
                    .rows
                    .iter()
                    .map(|row| TrainingRow {
                        id: row.id.clone(),
                        values: row.values.clone(),
                        label: row.target.clone(),
                    })
                    .collect::<Vec<_>>();
                let features = serde_json::to_value(&dataset.features)?;
                ensure!(
                    serde_json::to_value(&input.rows)? == serde_json::to_value(&rows)?
                        && serde_json::to_value(&input.features)? == features,
                    "encoder input differs from authenticated original train rows/features"
                );
                let training_sha = context_producer::training_rows_sha256(&rows)?;
                ensure!(
                    input.identity.dataset_sha256 == dataset.source.sha256
                        && input.identity.prepared_sha256 == dataset.split.development_sha256
                        && input.identity.training_sha256 == training_sha
                        && input.identity.fitting_rows_sha256 == cut.train.rows_sha256,
                    "CPU training identity differs from authenticated row serialization"
                );
                let prepared = input.prepare()?;
                let preprocessing_sha =
                    jope_pipeline::sha256_hex(&serde_json::to_vec(&prepared.preprocessing)?);
                let training = TrainingIdentity {
                    prepared_sha256: dataset.split.development_sha256.clone(),
                    training_sha256: training_sha,
                    selected_training_rows_sha256: cut.train.rows_sha256.clone(),
                    preprocessing_sha256: preprocessing_sha,
                    cut: input.identity.cut.into(),
                };
                ensure!(
                    record["fitting_rows_sha256"] == training.selected_training_rows_sha256
                        && record["encoder_sha256"] == encoder_ref.sha256,
                    "context index rows/encoder differs"
                );
                let request = &proof.context_execution.job.request;
                ensure!(
                    request.dataset_sha256 == dataset.source.sha256
                        && request.split_sha256 == dataset.split.sha256
                        && request.fitting_rows_sha256 == cut.train.rows_sha256
                        && request.fitting_rows == cut.train.rows.len()
                        && request.fold == input.identity.cut
                        && request.task == dataset.task,
                    "actual encoder request differs from authenticated fitting cut"
                );
                let expected = AdmissionExpectation {
                    source_sha40: proof.producer_source_sha40,
                    producer_sha256: proof.producer.sha256.clone(),
                    input_sha256: proof.input.sha256.clone(),
                    qualification_sha256: proof.qualification.sha256.clone(),
                    reference_manifest_sha256: proof.reference_manifest.sha256.clone(),
                    dataset_sha256: dataset.source.sha256.clone(),
                    training: training.clone(),
                };
                ensure!(
                    proof.context_execution.job.command.program.path == proof.producer.path
                        && proof.context_execution.job.command.program.sha256
                            == proof.producer.sha256,
                    "actual context executable differs from frozen producer"
                );
                let files = CpuContextFiles {
                    vector: proof.vector,
                    artifact: proof.artifact,
                    frozen_transforms: proof.frozen_transforms,
                    bundle: proof.bundle,
                    context_execution: proof.context_execution,
                    case_executions: proof.case_executions,
                    training: training.clone(),
                };
                let token = load_qualified(&files, &expected)?;
                let produced: CpuContextArtifact = read(&files.artifact.path)?;
                ensure!(
                    produced.encoder == encoder,
                    "qualified CPU encoder differs from training plan"
                );
                let wrapper_ref: Artifact =
                    serde_json::from_value(record["context_artifact"].clone())?;
                let producer_ref: Artifact =
                    serde_json::from_value(record["producer_receipt"].clone())?;
                wrapper_ref.verify()?;
                producer_ref.verify()?;
                let wrapper: Value = read(&wrapper_ref.path)?;
                let producer: Value = read(&producer_ref.path)?;
                let values: Vec<f32> = serde_json::from_value(wrapper["values"].clone())?;
                ensure!(
                    wrapper["lineage"] == "asm.experimental-regression-context/v1"
                        && wrapper["policy_compatible"] == false
                        && wrapper["training"] == serde_json::to_value(&training)?
                        && values == token.context()
                        && wrapper["context_sha256"] == files.vector.sha256,
                    "experimental wrapper does not contain verified CPU context"
                );
                ensure!(
                    producer["backend_lineage"] == context_producer::BACKEND
                        && producer["qualified_bundle_sha256"] == files.bundle.sha256
                        && producer["output_sha256"] == wrapper_ref.sha256
                        && producer["encoder_sha256"] == encoder_ref.sha256
                        && producer["training"] == serde_json::to_value(&training)?,
                    "training producer receipt loses qualified CPU lineage"
                );
                verified.push(VerifiedTrainingContext {
                    task: task.clone(),
                    context_id: context_id.clone(),
                    artifact_sha256: wrapper_ref.sha256,
                    producer_proof_sha256: files.bundle.sha256,
                });
            }
            ensure!(
                verified.len() > before,
                "real category has no verified contexts"
            );
        }
        Ok(verified)
    }

    fn verify_partition_identity(
        partition: &jope_learning::eligibility::InnerPartition,
        comparison_cut: &str,
        original: (&str, &str, &str),
        outer_fold: u8,
        ordered_rows: &[String],
    ) -> Result<()> {
        ensure!(
            partition.schema == "jope.teaching-inner-partition/v1"
                && partition.identity()? == comparison_cut
                && partition.dataset_sha256 == original.0
                && partition.original_split_sha256 == original.1
                && partition.development_sha256 == original.2
                && partition.outer_fold == outer_fold
                && partition.fitting_rows == ordered_rows,
            "context differs from authenticated inner partition and original panel cut"
        );
        Ok(())
    }

    fn verify_selection(
        state: &State,
        qualification: &jope_learning::eligibility::Qualification,
        policy: &Value,
        data: &jope_pipeline::prepared_data::DevelopmentData,
        outer_fold: u8,
    ) -> Result<()> {
        let dataset = &data.dataset;
        let cut = data.cut(outer_fold)?;
        if policy["inner_partition"].is_null() {
            ensure!(
                qualification.context_outer_fold.is_none()
                    && qualification.context_fitting_rows_sha256.is_none()
                    && qualification.comparison.cut_id == dataset.split.sha256,
                "legacy context comparison differs from original split"
            );
        } else {
            let partition: jope_learning::eligibility::InnerPartition =
                serde_json::from_value(policy["inner_partition"].clone())?;
            let ordered = cut
                .train
                .rows
                .iter()
                .map(|r| r.id.clone())
                .collect::<Vec<_>>();
            verify_partition_identity(
                &partition,
                &qualification.comparison.cut_id,
                (
                    &dataset.source.sha256,
                    &dataset.split.sha256,
                    &dataset.split.development_sha256,
                ),
                outer_fold,
                &ordered,
            )?;
            ensure!(
                qualification.context_outer_fold == Some(outer_fold)
                    && qualification.context_fitting_rows_sha256.as_deref()
                        == Some(cut.train.rows_sha256.as_str())
                    && partition.fitting_rows_sha256()? == cut.train.rows_sha256,
                "teaching context row selection differs from original outer training cut"
            );
            let retained = Artifact {
                path: state.blob_path(&cut.train.rows_sha256),
                sha256: cut.train.rows_sha256.clone(),
            };
            retained.verify()?;
            let saved: Vec<String> = read(&retained.path)?;
            ensure!(saved == ordered, "retained context row order differs");
            // Re-derive all groups, inner fold assignments and sampling/runtime
            // identities through the owning teaching implementation, not index claims.
            let expected = jope_learning::eligibility::derive_inner_partition(
                state,
                text(policy, "panel_sha256")?,
                &dataset.id,
                outer_fold,
                partition.evaluation_seed,
                &partition.executor_program_sha256,
                &partition.runtime_inventory_sha256,
            )?;
            ensure!(
                expected.identity()? == partition.identity()?,
                "inner group/fold assignment changed"
            );
        }
        Ok(())
    }

    fn verify_classifier(
        state: &State,
        qualification: &jope_learning::eligibility::Qualification,
        policy: &Value,
        data: &jope_pipeline::prepared_data::DevelopmentData,
        record: &Value,
        encoder: (&Artifact, &EncoderIdentity),
        identity: (&str, &str),
    ) -> Result<VerifiedTrainingContext> {
        use jope_pipeline::classification_bridge::{verify_classification, ClassificationFiles};
        use jope_policy::context_producer::classification::{
            ClassificationArtifact, ClassificationInput, ClassificationTask,
            CLASSIFICATION_FEATURE_POLICY,
        };
        let (encoder_ref, encoder) = encoder;
        let (task, context_id) = identity;
        let files: ClassificationFiles =
            serde_json::from_value(record["classification_proof"].clone())
                .context("real classifier training requires actual supervised component proof")?;
        files.input.verify()?;
        ensure!(
            fs::metadata(&files.input.path)?.len() <= 128 << 20,
            "classification input exceeds panel bound"
        );
        let observed: ClassificationInput = read(&files.input.path)?;
        let outer_fold = observed.identity.cut;
        verify_selection(state, qualification, policy, data, outer_fold)?;
        let cut = data.cut(outer_fold)?;
        let dataset = &data.dataset;
        let rows = cut
            .train
            .rows
            .iter()
            .map(|r| TrainingRow {
                id: r.id.clone(),
                values: r.values.clone(),
                label: r.target.clone(),
            })
            .collect::<Vec<_>>();
        let expected = ClassificationInput {
            schema: "jope.cpu-classification-training/v1".into(),
            split_sha256: dataset.split.sha256.clone(),
            task: match task {
                "binary" => ClassificationTask::Binary,
                "multiclass" => ClassificationTask::Multiclass,
                _ => anyhow::bail!("classifier task required"),
            },
            classes: dataset.classes.clone(),
            identity: context_producer::TrainingIdentity {
                dataset_sha256: dataset.source.sha256.clone(),
                prepared_sha256: dataset.split.development_sha256.clone(),
                training_sha256: context_producer::training_rows_sha256(&rows)?,
                fitting_rows_sha256: cut.train.rows_sha256.clone(),
                cut: outer_fold,
            },
            features: serde_json::from_value(serde_json::to_value(&dataset.features)?)?,
            rows,
        };
        ensure!(
            serde_json::to_value(&observed)? == serde_json::to_value(&expected)?,
            "classifier input changes original labels/nulls/row order/class order/schema"
        );
        let token = verify_classification(&files, &expected)?;
        let produced: ClassificationArtifact = read(&files.artifact.path)?;
        ensure!(
            &produced.reference_encoder == encoder,
            "classifier encoder differs from frozen training encoder"
        );
        let wrapper_ref: Artifact = serde_json::from_value(record["context_artifact"].clone())?;
        let producer_ref: Artifact = serde_json::from_value(record["producer_receipt"].clone())?;
        wrapper_ref.verify()?;
        producer_ref.verify()?;
        let wrapper: Value = read(&wrapper_ref.path)?;
        let producer: Value = read(&producer_ref.path)?;
        ensure!(
            wrapper == serde_json::to_value(token.context())?
                && record["lineage"] == jope_pipeline::search_context::CLASSIFICATION_LINEAGE
                && record["fitting_rows_sha256"] == cut.train.rows_sha256
                && record["encoder_sha256"] == encoder_ref.sha256,
            "classifier training wrapper differs from private verified context"
        );
        ensure!(
            producer["backend_lineage"] == CLASSIFICATION_FEATURE_POLICY
                && producer["component_qualification_sha256"] == token.qualification_sha256()
                && producer["output_sha256"] == wrapper_ref.sha256
                && producer["fitting_rows_sha256"] == cut.train.rows_sha256
                && producer["encoder_sha256"] == encoder_ref.sha256
                && producer["training"] == serde_json::to_value(token.training())?,
            "classifier producer receipt loses its experimental component lineage"
        );
        Ok(VerifiedTrainingContext {
            task: task.into(),
            context_id: context_id.into(),
            artifact_sha256: wrapper_ref.sha256,
            producer_proof_sha256: files.artifact.sha256,
        })
    }

    #[cfg(test)]
    mod selection_tests {
        use super::*;
        #[test]
        fn inner_comparison_and_original_split_are_distinct_and_both_bound() -> Result<()> {
            use jope_learning::eligibility::InnerPartition;
            let rows = vec!["original-row-a".into(), "original-row-b".into()];
            let partition = InnerPartition {
                schema: "jope.teaching-inner-partition/v1".into(),
                dataset_sha256: "a".repeat(64),
                original_split_sha256: "b".repeat(64),
                development_sha256: "c".repeat(64),
                outer_fold: 0,
                fitting_rows: rows.clone(),
                hold_rows: vec!["held-row".into()],
                fitting_groups: vec!["g1".into(), "g2".into()],
                hold_groups: vec!["g3".into()],
                inner_validation_rows: vec![vec!["original-row-a".into()]],
                assignment_policy: "synthetic-fixture".into(),
                evaluation_seed: 7,
                request_seed_policy: "fixture".into(),
                executor_program_sha256: "d".repeat(64),
                runtime_inventory_sha256: "e".repeat(64),
                fit_randomness_policy: "fixture".into(),
                raw_input_policy: "fixture".into(),
            };
            let comparison = partition.identity()?;
            assert_ne!(comparison, partition.original_split_sha256);
            let original = (
                partition.dataset_sha256.as_str(),
                partition.original_split_sha256.as_str(),
                partition.development_sha256.as_str(),
            );
            verify_partition_identity(&partition, &comparison, original, 0, &rows)?;
            assert!(verify_partition_identity(
                &partition,
                &partition.original_split_sha256,
                original,
                0,
                &rows
            )
            .is_err());
            assert!(
                verify_partition_identity(&partition, &comparison, original, 1, &rows).is_err()
            );
            let mut reversed = rows.clone();
            reversed.reverse();
            assert!(
                verify_partition_identity(&partition, &comparison, original, 0, &reversed).is_err()
            );
            assert!(verify_partition_identity(
                &partition,
                &comparison,
                (original.0, "different-split", original.2),
                0,
                &rows
            )
            .is_err());
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Produced by the Python experiment_for_plan/experiment_identity helpers.
    // UTF8, ordered tasks, sorted checkpoint map and field order are deliberate.
    const PYTHON_PLAN: &str = r#"{"plan_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","source_sha":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","seed":17,"device":"cuda:0","tasks":["regression","binary"],"runtime":{"python":{"path":"/opt/python3","sha256":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"},"python_invocation":"/opt/venv/bin/python","sources":{"policy/λ.py":"dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"}},"category_inputs":{"regression":{"data_kind":"synthetic_fixture","inputs":{"corpus":{"path":"/custody/regression.jsonl","sha256":"eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"}}},"binary":{"data_kind":"synthetic_fixture","inputs":{"corpus":{"path":"/custody/binary.jsonl","sha256":"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"}}}}}"#;
    const PYTHON_EXPERIMENT_ID: &str =
        "76226b54b655efe651de60bd0cf29ce35f2039c02360968e2dadf682b047cc00";

    #[test]
    fn experiment_identity_matches_python_fixture() {
        let plan: Value = serde_json::from_str(PYTHON_PLAN).unwrap();
        let experiment = experiment_for_plan(&plan).unwrap();
        assert_eq!(experiment.identity().unwrap(), PYTHON_EXPERIMENT_ID);
        assert_eq!(experiment.resources.per_fit_seconds, 1200);
        assert_eq!(experiment.resources.total_seconds, 3600);
        assert_eq!(experiment.resources.reporting_seconds, 5);
        assert_eq!(
            experiment.expected_artifacts,
            vec!["regression/head.json", "binary/head.json"]
        );
        // A sorted-key file digest is deliberately not the Rust struct identity.
        assert_ne!(digest(&experiment).unwrap(), PYTHON_EXPERIMENT_ID);
        let mut changed = plan;
        changed["seed"] = json!(18);
        assert_ne!(
            experiment_for_plan(&changed).unwrap().identity().unwrap(),
            PYTHON_EXPERIMENT_ID
        );
    }

    #[test]
    fn supervisor_uuid_and_subgroup_are_exact() {
        let experiment = experiment_for_plan(&serde_json::from_str(PYTHON_PLAN).unwrap()).unwrap();
        let id = "12345678-1234-1234-1234-123456789abc";
        let unit = format!("jope-asm-{}-{id}.service", experiment.identity().unwrap());
        valid_unit(&unit, &experiment).unwrap();
        assert!(valid_unit(
            &format!("jope-asm-{}.service", experiment.identity().unwrap()),
            &experiment
        )
        .is_err());
        assert!(valid_unit(
            &unit.replace(id, "12345678/1234-1234-1234-123456789abc"),
            &experiment
        )
        .is_err());
        let supervisor = supervisor::for_unit(&unit).unwrap();
        assert_eq!(
            subgroup(&supervisor, id).unwrap().parent(),
            Some(supervisor.cgroup_root.as_path())
        );
        assert!(subgroup(&supervisor, "../controller").is_err());
        assert!(subgroup(&supervisor, "1234567811234-1234-1234-123456789abc").is_err());
    }

    #[test]
    fn virtualenv_invocation_retains_link_but_pins_physical_binary() {
        use std::os::unix::fs::symlink;
        let directory = tempfile::tempdir().unwrap();
        let physical = directory.path().join("physical-python");
        fs::write(&physical, b"synthetic executable bytes").unwrap();
        let invocation = directory.path().join("venv-python");
        symlink(&physical, &invocation).unwrap();
        assert_eq!(invocation_path(&invocation).unwrap(), invocation);
        let actual = artifact(&invocation).unwrap();
        assert_eq!(actual.path, physical);
        assert_ne!(actual.path, invocation);
        actual.verify().unwrap();
        fs::write(&physical, b"changed executable bytes").unwrap();
        assert!(actual.verify().is_err());
    }

    #[test]
    fn explicit_environment_omits_credentials_and_caps_threads() {
        let environment = child_environment(Path::new("/custody/training"), "cpu", 7);
        for key in [
            "GH_TOKEN",
            "GITHUB_TOKEN",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "PYTHONPATH",
            "LD_PRELOAD",
        ] {
            assert!(!environment.contains_key(key));
        }
        assert_eq!(environment["CUDA_VISIBLE_DEVICES"], "");
        assert_eq!(environment["OMP_NUM_THREADS"], "1");
        assert_eq!(environment["WORLD_SIZE"], "1");
        assert_eq!(environment["TMPDIR"], "/custody/training/python-tmp");
        assert_eq!(
            child_environment(Path::new("/custody/training"), "cuda:0", 7)["CUDA_VISIBLE_DEVICES"],
            "0"
        );
    }

    #[test]
    fn immutable_receipts_reject_changed_bytes() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("receipt.json");
        immutable(&path, &json!({"value":1})).unwrap();
        immutable(&path, &json!({"value":1})).unwrap();
        assert!(immutable(&path, &json!({"value":2})).is_err());
    }

    #[test]
    fn controller_preparation_binds_projected_inputs_and_request_bytes() {
        let root =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/tmp/training-preparation");
        fs::create_dir_all(&root).unwrap();
        let dir = tempfile::tempdir_in(fs::canonicalize(root).unwrap()).unwrap();
        let request_path = dir.path().join("request.json");
        immutable(
            &request_path,
            &json!({"schema":"jope.real-training-corpus-request/v1","contexts":{}}),
        )
        .unwrap();
        let input_path = dir.path().join("inputs.json");
        immutable(
            &input_path,
            &json!({"schema":"jope.experimental-training-inputs/v1"}),
        )
        .unwrap();
        let input = artifact(&input_path).unwrap();
        let receipt_path = dir.path().join("receipt.json");
        immutable(
            &receipt_path,
            &json!({"schema":"jope.real-training-corpus-result/v1",
            "request":artifact(&request_path).unwrap(),"inputs":input}),
        )
        .unwrap();
        let native = NativePlan {
            schema: SCHEMA.into(),
            plan: input.clone(),
            experiment: input.clone(),
            native: input.clone(),
            python: input.clone(),
            python_invocation: input_path,
            environment: BTreeMap::new(),
            learning_state: None,
            input_preparation: Some(artifact(&receipt_path).unwrap()),
        };
        let plan = json!({"requested_manifest":input});
        assert!(controller_request(&native, &plan).is_ok());
        assert!(controller_request(&native, &json!({"requested_manifest":null})).is_err());
        let comparison_plan = json!({"plan_sha256":"fixture", "tasks":["regression"],
            "category_inputs":{"regression":{"data_kind":"synthetic_fixture"}}});
        immutable(
            &dir.path().join("comparison.json"),
            &json!({
            "schema":"jope.training-comparison/v1", "plan_sha256":"fixture", "task":"regression",
            "fresh_inputs":input, "fresh_preparation":null }),
        )
        .unwrap();
        assert!(
            comparison_admission(dir.path(), &native, &comparison_plan, &ProducerVerifier).is_ok()
        );
        fs::write(request_path, b"changed controller proof").unwrap();
        assert!(controller_request(&native, &plan).is_err());
    }

    #[test]
    fn context_booleans_cannot_replace_native_admission() {
        let plan = json!({"tasks":["regression"],"category_inputs":{"regression":{"data_kind":"admitted","verified":true}}});
        let placeholder = Artifact {
            path: "/custody/absent".into(),
            sha256: "a".repeat(64),
        };
        let native = NativePlan {
            schema: SCHEMA.into(),
            plan: placeholder.clone(),
            experiment: placeholder.clone(),
            native: placeholder.clone(),
            python: placeholder,
            python_invocation: "/custody/venv/bin/python".into(),
            environment: BTreeMap::new(),
            learning_state: None,
            input_preparation: None,
        };
        assert!(ProducerVerifier.verify(&native, &plan, None).is_err());
    }
}
