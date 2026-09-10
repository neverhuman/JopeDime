//! Mandatory numerical boundary shared by all supported GP fitting commands.
//!
//! Each public operation either retrieves an original model or supervises one
//! executor that consumes a durable authority token against sealed input bytes.
//! Deployment remains gated on historical coverage and target qualification.
use anyhow::{bail, ensure, Context, Result};
use crate::{eval_registry::*, manifest::*};
use feat_math::Model;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{fs::{self, File, OpenOptions}, io::{BufRead, Read, Seek, SeekFrom, Write},
    os::{fd::{AsRawFd, FromRawFd}, unix::process::CommandExt}, path::{Path, PathBuf},
    process::{Child, Command, Stdio}, time::{Duration, Instant}};

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Inputs {
    program: String,
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
    preprocessing: Vec<u8>,
    feature_view: Vec<u8>,
}

fn contract() -> NumericalContract {
    NumericalContract { protocol: "featmath_fixed_seed0_f64_single_thread_v1".into(),
        precision: "f64".into(), omp_threads: 1, rayon_threads: 1, blas_threads: 1, mkl_threads: 1 }
}

impl Inputs {
    fn verify(&self) -> Result<VerifiedFit> {
        VerifiedFit::for_current_executor(&self.program, self.x.clone(), self.y.clone(),
            self.preprocessing.clone(), self.feature_view.clone(), 0, &contract())
    }
}

fn authority_paths() -> Result<(PathBuf, PathBuf)> {
    Ok((std::env::var_os("JOPE_FIT_AUTHORITY").map(PathBuf::from)
            .context("mandatory fit authority unavailable: fitting is fenced")?,
        std::env::var_os("JOPE_FIT_WITNESS").map(PathBuf::from)
            .context("independent fit high-water witness unavailable: fitting is fenced")?))
}

pub struct RegisteredModel {
    pub model: Model,
    pub completion: Value,
    pub model_reference: Value,
    pub reused: bool,
}

fn artifact_reference(path: &Path) -> Result<Value> {
    Ok(json!({"path":path.canonicalize()?, "sha256":file_hash(path)?}))
}

fn retained(authority: &mut Authority, root: &Path, completion: &str, reused: bool, destination: &Path) -> Result<RegisteredModel> {
    let receipt = authority.verify_committed_completion(completion)?;
    if receipt.outcome != ExecutionOutcome::Success {
        let path = destination.with_extension("fit-outcome.json");
        let value = json!({"schema":"parallel.fit_terminal_reference.v1",
            "fit_completion":artifact_reference(&root.join("objects").join(completion))?,
            "outcome":receipt.outcome,"numerical_runtime_seconds":receipt.elapsed_seconds,
            "evaluation_origin":if reused { "reuse" } else { "execution" }});
        if path.exists() { ensure!(read_json::<Value>(&path)? == value, "original terminal fit reference changed"); }
        else { write_json(&path, &value)?; }
    }
    ensure!(receipt.outcome == ExecutionOutcome::Success, "original fit has terminal non-success outcome");
    let model = authority.model_for_completion(completion)?;
    Ok(RegisteredModel { model: serde_json::from_slice(&model)?,
        completion: artifact_reference(&root.join("objects").join(completion))?,
        model_reference: artifact_reference(&root.join("objects").join(receipt.model_sha256.context("original model unavailable")?))?,
        reused })
}

struct OwnedChild(Child);
impl Drop for OwnedChild {
    fn drop(&mut self) {
        if self.0.try_wait().ok().flatten().is_none() {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
}

fn sealed(bytes: &[u8]) -> Result<File> {
    let name = std::ffi::CString::new("jope-fit-input")?;
    let fd = unsafe { libc::memfd_create(name.as_ptr(), libc::MFD_ALLOW_SEALING) };
    ensure!(fd >= 3, "sealed fit descriptor unavailable");
    let mut file = unsafe { File::from_raw_fd(fd) };
    file.write_all(bytes)?;
    file.seek(SeekFrom::Start(0))?;
    let seals = libc::F_SEAL_WRITE | libc::F_SEAL_GROW | libc::F_SEAL_SHRINK | libc::F_SEAL_SEAL;
    ensure!(unsafe { libc::fcntl(fd, libc::F_ADD_SEALS, seals) } == 0, "fit inputs could not be sealed");
    Ok(file)
}

fn readiness(stream: &mut impl Read, fd: i32, timeout: Duration) -> Result<Value> {
    let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
    ensure!(flags >= 0 && unsafe { libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) } == 0,
        "executor readiness could not be bounded");
    let deadline = Instant::now() + timeout;
    let mut bytes = Vec::new();
    loop {
        ensure!(Instant::now() < deadline, "executor readiness deadline exceeded; claimed work stays unavailable");
        let mut poll = libc::pollfd { fd, events: libc::POLLIN, revents: 0 };
        let milliseconds = deadline.saturating_duration_since(Instant::now()).as_millis().clamp(1, 100) as i32;
        let n = unsafe { libc::poll(&mut poll, 1, milliseconds) };
        if n < 0 {
            let error = std::io::Error::last_os_error();
            if error.kind() == std::io::ErrorKind::Interrupted { continue; }
            return Err(error.into());
        }
        if n == 0 { continue; }
        let mut block = [0u8; 4096];
        match stream.read(&mut block) {
            Ok(0) => bail!("executor exited before complete readiness"),
            Ok(count) => {
                bytes.extend_from_slice(&block[..count]);
                ensure!(bytes.len() <= 64 * 1024, "executor readiness exceeds protocol limit");
                if let Some(end) = bytes.iter().position(|b| *b == b'\n') {
                    return Ok(serde_json::from_slice(&bytes[..end])?);
                }
            }
            Err(error) if matches!(error.kind(), std::io::ErrorKind::WouldBlock | std::io::ErrorKind::Interrupted) => {},
            Err(error) => return Err(error.into()),
        }
    }
}

/// The only supported route to a new physical feat_math fit. An existing entry
/// without an original successful model is unavailable, never a fresh claim.
pub fn fit(program: &str, x: Vec<Vec<f64>>, y: Vec<f64>, preprocessing: Vec<u8>,
    feature_view: Vec<u8>, destination: &Path) -> Result<RegisteredModel> {
    let (root, witness) = authority_paths()?;
    let mut authority = Authority::open(&root, &witness)?;
    let inputs = Inputs { program: program.into(), x, y, preprocessing, feature_view };
    let fit = inputs.verify()?;
    let request = RequestProvenance {
        scope: std::env::var("JOPE_FIT_SCOPE").context("fit access scope unavailable")?,
        task_id: destination.to_string_lossy().to_string(),
        details: json!({"entry_point":"registered_fit", "destination":destination}),
    };
    let claim = match authority.claim(&fit, &request)? {
        ClaimOutcome::Claimed(claim) => claim,
        ClaimOutcome::Existing { state, completion_sha256: Some(completion), .. } if state == "completed" || state == "terminal_failure" =>
            return retained(&mut authority, &root, &completion, true, destination),
        other => bail!("fit unavailable; no automatic retry: {other:?}"),
    };
    let descriptor = sealed(&serde_json::to_vec(&inputs)?)?;
    let output = destination.with_extension(format!("fit-{}.json", claim.claim_token));
    ensure!(!output.exists(), "original fit output already exists");
    fs::create_dir_all(output.parent().context("fit destination has no parent")?)?;
    let mut command = Command::new(std::env::current_exe()?);
    command.args(["registry-fit", "--fd"]).arg(descriptor.as_raw_fd().to_string())
        .arg("--output").arg(&output).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::inherit());
    let parent = std::process::id();
    unsafe {
        command.pre_exec(move || {
            if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) != 0 || libc::getppid() as u32 != parent {
                return Err(std::io::Error::other("fit executor lost original supervisor"));
            }
            let limit = libc::rlimit { rlim_cur: 8 * 1024 * 1024 * 1024, rlim_max: 8 * 1024 * 1024 * 1024 };
            if libc::setrlimit(libc::RLIMIT_AS, &limit) != 0 { return Err(std::io::Error::last_os_error()); }
            Ok(())
        });
    }
    let mut child = OwnedChild(command.spawn()?);
    let mut ready = child.0.stdout.take().context("executor readiness pipe missing")?;
    let fd = ready.as_raw_fd();
    let ready = readiness(&mut ready, fd, Duration::from_secs(30))?;
    ensure!(ready["fit_key"] == fit.key(), "actual executor input/runtime identity differs");
    let process = ProcessIdentity::read(child.0.id())?;
    ensure!(ready["process"] == serde_json::to_value(&process)?, "executor readiness custody differs");
    let start = authority.authorize(&fit, &claim, &process)?;
    let mut input = child.0.stdin.take().context("executor authorization pipe missing")?;
    input.write_all(&serde_json::to_vec(&start)?)?;
    input.write_all(b"\n")?;
    drop(input);
    let observed = ObservedCompletion::wait(&mut child.0, &start, Duration::from_secs(30))?;
    let bytes = fs::read(&output).unwrap_or_default();
    let result: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    let (outcome, model) = match result["status"].as_str() {
        Some("success") => (ExecutionOutcome::Success, Some(serde_json::to_vec(&result["model"])?)),
        Some("deterministic_failure") => (ExecutionOutcome::DeterministicFailure, None),
        _ => (ExecutionOutcome::InfrastructureFailure, None),
    };
    let completion = authority.complete_observed(&fit, observed, outcome, &bytes, model.as_deref(), None, None)?;
    retained(&mut authority, &root, &completion, false, destination)
}

/// Private CLI executor. No caller-supplied JSON identity grants permission:
/// arrays, program, executable and loaded runtime are verified here again.
pub fn execute(fd: i32, output: &Path) -> Result<()> {
    let (root, witness) = authority_paths()?;
    ensure!(fd >= 3 && !output.exists(), "invalid executor input/output custody");
    let seals = libc::F_SEAL_WRITE | libc::F_SEAL_GROW | libc::F_SEAL_SHRINK | libc::F_SEAL_SEAL;
    ensure!(unsafe { libc::fcntl(fd, libc::F_GET_SEALS) } & seals == seals, "fit input is not fully sealed");
    let copy = unsafe { libc::dup(fd) };
    ensure!(copy >= 3, "fit input descriptor missing");
    let mut file = unsafe { File::from_raw_fd(copy) };
    ensure!(file.metadata()?.len() <= 2 * 1024 * 1024 * 1024, "fit input exceeds resource contract");
    file.seek(SeekFrom::Start(0))?;
    let inputs: Inputs = serde_json::from_reader(file)?;
    let fit = inputs.verify()?;
    let mut authority = Authority::open(&root, &witness)?;
    println!("{}", json!({"fit_key":fit.key(),"process":ProcessIdentity::current()?}));
    std::io::stdout().flush()?;
    let mut line = String::new();
    std::io::stdin().lock().take(64 * 1024).read_line(&mut line)?;
    let start: StartToken = serde_json::from_str(&line)?;
    let custody = authority.consume_start(&fit, &start)?;
    custody.verify()?;
    let genome = feat_math::GenomeIr::parse(&fit.identity().canonical_program)?;
    let (x, y) = fit.training();
    let result = match feat_math::fit(&genome, x, y) {
        Ok(model) => json!({"status":"success","model":model}),
        Err(error) => json!({"status":"deterministic_failure","error":format!("{error:#}")}),
    };
    custody.verify()?;
    let mut artifact = OpenOptions::new().write(true).create_new(true).open(output)?;
    artifact.write_all(&serde_json::to_vec(&result)?)?;
    artifact.sync_all()?;
    File::open(output.parent().context("executor artifact parent absent")?)?.sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sealed_inputs_reject_modification_and_truncation() {
        let file = sealed(b"owned numeric bytes").unwrap();
        assert_eq!(unsafe { libc::ftruncate(file.as_raw_fd(), 0) }, -1);
        assert_eq!(unsafe { libc::write(file.as_raw_fd(), b"x".as_ptr().cast(), 1) }, -1);
    }
    #[test]
    fn partial_readiness_does_not_reset_or_escape_deadline() {
        let (mut reader, mut writer) = std::os::unix::net::UnixStream::pair().unwrap();
        writer.write_all(b"{\"partial\":").unwrap();
        let began = Instant::now();
        let fd = reader.as_raw_fd();
        assert!(readiness(&mut reader, fd, Duration::from_millis(20)).unwrap_err()
            .to_string().contains("deadline"));
        assert!(began.elapsed() < Duration::from_secs(1));
    }
}
