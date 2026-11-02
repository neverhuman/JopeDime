//! Mandatory numerical boundary shared by supported GP fitting commands.
//!
//! Only the long-lived authority may start and supervise a numerical executor.
//! Clients carry sealed inputs and receive references to its original receipt.
use crate::{
    eval_registry::{service, *},
    manifest::*,
};
use anyhow::{bail, ensure, Context, Result};
use feat_math::Model;
use serde_json::{json, Value};
use std::{
    fs::{File, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};

pub struct RegisteredModel {
    pub model: Model,
    pub completion: Value,
    pub model_reference: Value,
    pub reused: bool,
}

/// Same-host absolute execution deadline; request provenance, never fit identity.
pub fn deadline_after(duration: Duration) -> Result<u64> {
    let now = crate::host_os::monotonic_ns().context("monotonic execution clock unavailable")?;
    now.checked_add(duration.as_nanos().try_into()?)
        .context("execution deadline overflow")
}

fn reference(path: &Path, digest: &str) -> Result<Value> {
    ensure!(
        file_hash(path)? == digest,
        "original authority artifact changed"
    );
    Ok(json!({"path":path.canonicalize()?, "sha256":digest}))
}

/// No direct-child or empty-registry fallback is available when IPC is absent.
pub fn fit(
    program: &str,
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
    preprocessing: Vec<u8>,
    feature_view: Vec<u8>,
    destination: &Path,
) -> Result<RegisteredModel> {
    let socket = std::env::var_os("JOPE_FIT_SERVICE_SOCKET")
        .map(PathBuf::from)
        .context("mandatory fit authority service unavailable: fitting is fenced")?;
    let deadline = match std::env::var("JOPE_FIT_DEADLINE_MONOTONIC_NS") {
        Ok(value) => value
            .parse::<u64>()
            .context("invalid inherited execution deadline")?,
        Err(std::env::VarError::NotPresent) => deadline_after(Duration::from_secs(30))?,
        Err(error) => return Err(error.into()),
    };
    ensure!(
        deadline > deadline_after(Duration::ZERO)?,
        "original request deadline expired before fitting"
    );
    let request = RequestProvenance {
        scope: std::env::var("JOPE_FIT_SCOPE").context("fit access scope unavailable")?,
        task_id: destination.to_string_lossy().into_owned(),
        details: json!({"entry_point":"registered_fit", "destination":destination,
            "deadline_monotonic_ns":deadline}),
    };
    let inputs = service::ServiceInputs {
        program: program.into(),
        x,
        y,
        preprocessing,
        feature_view,
    };
    let result = if let Ok(job) = std::env::var("SCQ_JOB_ID") {
        let filename = destination
            .file_name()
            .and_then(|v| v.to_str())
            .context("native candidate destination")?;
        let candidate_index = filename
            .strip_prefix("candidate-")
            .and_then(|v| v.strip_suffix(".json"))
            .context("native candidate identity absent")?
            .parse()?;
        let native = crate::eval_registry::native_scq::NativeRequest {
            job_id: job.parse()?,
            candidate_index,
            descriptor_sha256: crate::eval_registry::native_scq::policy::hash(&serde_json::to_vec(
                &inputs,
            )?),
            deadline_monotonic_ns: deadline,
            destination: destination.to_path_buf(),
        };
        service::submit_native(
            &socket,
            &PathBuf::from(
                std::env::var_os("JOPE_FIT_AUTHORITY").context("native authority root absent")?,
            ),
            &PathBuf::from(std::env::var_os("JOPE_FIT_WITNESS").context("native witness absent")?),
            &inputs,
            native,
        )?
    } else {
        service::submit(&socket, &inputs, request)?
    };
    let completion = reference(&result.completion_path, &result.completion_sha256)?;
    if result.receipt.outcome != ExecutionOutcome::Success {
        let path = destination.with_extension("fit-outcome.json");
        let value = json!({"schema":"parallel.fit_terminal_reference.v1",
            "fit_completion":completion,"outcome":result.receipt.outcome,
            "numerical_runtime_seconds":result.receipt.elapsed_seconds,
            "evaluation_origin":if result.reused { "reuse" } else { "execution" }});
        if path.exists() {
            ensure!(
                read_json::<Value>(&path)? == value,
                "original terminal fit reference changed"
            );
        } else {
            write_json(&path, &value)?;
        }
        bail!("original fit has terminal non-success outcome");
    }
    Ok(RegisteredModel {
        model: serde_json::from_slice(&result.model_bytes()?)?,
        completion,
        model_reference: reference(
            result
                .model_path
                .as_deref()
                .context("original model unavailable")?,
            result
                .model_sha256
                .as_deref()
                .context("original model identity unavailable")?,
        )?,
        reused: result.reused,
    })
}

/// The retired public token route never opens storage or starts numerical work.
pub fn execute(_fd: i32, _output: &Path) -> Result<()> {
    bail!("direct registry-fit is retired; original service-owned executor custody is required")
}

/// Hidden worker entry point: the authority owns its original child, consumes
/// the kernel-authenticated start token and retains custody through observed exit.
pub fn execute_service() -> Result<()> {
    let (fit, permit, output) = service::await_start_from_environment()?;
    ensure!(!output.exists(), "original fit output already exists");
    permit.verify()?;
    let genome = feat_math::GenomeIr::parse(&fit.identity().canonical_program)?;
    let (x, y) = fit.training();
    let result = match feat_math::fit(&genome, x, y) {
        Ok(model) => json!({"status":"success", "model":model}),
        Err(error) => json!({"status":"deterministic_failure", "error":format!("{error:#}")}),
    };
    permit.verify()?;
    let mut artifact = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&output)?;
    artifact.write_all(&serde_json::to_vec(&result)?)?;
    artifact.sync_all()?;
    File::open(output.parent().context("executor artifact parent absent")?)?.sync_all()?;
    Ok(())
}

static STOP_REQUESTED: AtomicBool = AtomicBool::new(false);
extern "C" fn request_stop(_signal: i32) {
    STOP_REQUESTED.store(true, Ordering::SeqCst);
}

pub fn serve(
    authority: PathBuf,
    witness: PathBuf,
    socket: PathBuf,
    workers: usize,
    event_witness_config: &Path,
    event_witness_config_sha256: &str,
    funding_config: Option<&Path>,
) -> Result<()> {
    serve_with_native_policy(
        authority,
        witness,
        socket,
        workers,
        event_witness_config,
        event_witness_config_sha256,
        crate::fit_boundary::FitPolicies {
            funding_config,
            native_fit_policy: None,
        },
    )
}
/// Optional funding and native-fit policy bindings for the evaluator service.
#[derive(Clone, Copy)]
pub struct FitPolicies<'a> {
    pub funding_config: Option<&'a Path>,
    pub native_fit_policy: Option<&'a Path>,
}

pub fn serve_with_native_policy(
    authority: PathBuf,
    witness: PathBuf,
    socket: PathBuf,
    workers: usize,
    event_witness_config: &Path,
    event_witness_config_sha256: &str,
    options: FitPolicies<'_>,
) -> Result<()> {
    let FitPolicies {
        funding_config,
        native_fit_policy,
    } = options;

    if let Ok(value) = std::env::var("SCQ_CONTROL_FD") {
        let fd: i32 = value.parse()?;
        // SAFETY: fcntl inspects this caller-supplied integer descriptor and
        // returns a scalar flag value; the negative result is rejected below.
        let flags = unsafe { libc::fcntl(fd, libc::F_GETFD) };
        ensure!(
            // SAFETY: this runs only when F_GETFD succeeded; it updates flags on
            // the same open descriptor and passes no pointers.
            flags >= 0 && unsafe { libc::fcntl(fd, libc::F_SETFD, flags | libc::FD_CLOEXEC) } == 0,
            "native controller control descriptor unavailable"
        );
    }

    let evaluator = std::env::var_os("JOPE_CPU_EVALUATOR")
        .map(PathBuf::from)
        .context("explicit JOPE_CPU_EVALUATOR binding required; CPU evaluator unavailable")?;
    let witness_config = crate::eval_registry::witness::WitnessClientConfig::from_file(
        event_witness_config,
        event_witness_config_sha256,
    )?;
    let mut config =
        service::ServiceConfig::regression(authority, witness, socket, evaluator, workers);
    config.event_witness = Some(witness_config);
    config.funding_config = funding_config.map(Path::to_path_buf);
    config.native_fit_policy = native_fit_policy.map(Path::to_path_buf);
    let instance = service::Service::bind(config)?;
    let stop = Arc::new(AtomicBool::new(false));
    let finished = AtomicBool::new(false);
    STOP_REQUESTED.store(false, Ordering::SeqCst);
    // SAFETY: `request_stop` has the C signal-handler ABI and uses only
    // async-signal-safe atomic stores; signal takes scalar arguments.
    let previous_term = unsafe {
        libc::signal(
            libc::SIGTERM,
            request_stop as *const () as libc::sighandler_t,
        )
    };
    // SAFETY: same handler and scalar arguments as the SIGTERM registration.
    let previous_int = unsafe {
        libc::signal(
            libc::SIGINT,
            request_stop as *const () as libc::sighandler_t,
        )
    };
    let result = std::thread::scope(|threads| {
        let signal_stop = stop.clone();
        let finished_ref = &finished;
        threads.spawn(move || {
            while !finished_ref.load(Ordering::SeqCst) {
                if STOP_REQUESTED.load(Ordering::SeqCst) {
                    signal_stop.store(true, Ordering::SeqCst);
                }
                std::thread::sleep(Duration::from_millis(20));
            }
        });
        let result = instance.run(stop);
        finished.store(true, Ordering::SeqCst);
        result
    });
    // SAFETY: restore the exact handler values returned by the two successful
    // signal registrations; signal takes scalar arguments and returns a value.
    unsafe {
        libc::signal(libc::SIGTERM, previous_term);
        libc::signal(libc::SIGINT, previous_int);
    }
    println!("{}", serde_json::to_string(&result?)?);
    Ok(())
}
