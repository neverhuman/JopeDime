//! Native, fail-closed supervisor for one Foundation Encoder V2 dataset.
//!
//! The Python/PyTorch child is admitted only inside a Landlock sandbox. It
//! receives one inherited receipt FD, while this process independently hashes
//! inputs, runtime artifacts, the sandbox policy, and the finite 8,192-byte
//! output before producing the binding-capable execution receipt.

use anyhow::{ensure, Context, Result};
use clap::Parser;
use jope_dime_regressor::encoder_v2_execution::{
    runtime_artifacts_digest, validate_output_bytes, EncoderAttemptReceiptV2,
    EncoderExecutionReceiptV2, EncoderStageTiming, ObservedEncoderRuntime, ATTEMPT_SCHEMA,
    EXECUTION_SCHEMA,
};
use jope_dime_regressor::encoder_v2_identity::{
    cache_key, decode_dataset_input, decode_encoder_identity, ArtifactIdentity,
    FoundationDatasetInputV2, FoundationEncoderIdentityV2, OUTPUT_BYTES, OUTPUT_FEATURES,
};
use jope_dime_regressor::selfplay_data::canonical_json_sha256;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::ffi::{CString, OsString};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};
use std::os::unix::process::CommandExt;
use std::path::{Component, Path, PathBuf};
use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

#[path = "jope_encoder_supervisor/artifact_helpers.rs"]
mod artifact_helpers;
#[path = "jope_encoder_supervisor/sandbox.rs"]
mod sandbox;

use artifact_helpers::{
    collect_files, encoder_artifacts, sha256_bytes, verify_declared_inputs,
    verify_runtime_artifacts,
};
use sandbox::{
    LandlockRuleset, SandboxPolicy, ACCESS_EXECUTE, ACCESS_MAKE_DIR, ACCESS_MAKE_REG,
    ACCESS_MAKE_SYM, ACCESS_READ_DIR, ACCESS_READ_FILE, ACCESS_REFER, ACCESS_REMOVE_DIR,
    ACCESS_REMOVE_FILE, ACCESS_TRUNCATE, ACCESS_WRITE_FILE,
};

const CHILD_SCHEMA: &str = "JopeEncoderChildReceipt.v2";
const SANDBOX_SCHEMA: &str = "JopeEncoderLandlockPolicy.v1";
const CHILD_RECEIPT_FD: RawFd = 198;
#[derive(Parser, Debug)]
#[command(name = "jope-encoder-supervisor")]
struct Args {
    #[arg(long)]
    encoder_identity: PathBuf,
    #[arg(long)]
    dataset_input: PathBuf,
    #[arg(long)]
    runtime_root: PathBuf,
    #[arg(long)]
    input_root: PathBuf,
    #[arg(long)]
    output_root: PathBuf,
    #[arg(long)]
    scq_job_identity_sha256: String,
    #[arg(long, default_value = "output.f32le")]
    output_name: String,
    #[arg(long, default_value = "child-receipt.v2.json")]
    child_receipt_name: String,
    #[arg(long, default_value = "execution-receipt.v2.json")]
    execution_receipt_name: String,
    #[arg(long, default_value = "attempt.v2.json")]
    attempt_name: String,
    #[arg(long)]
    system_read_root: Vec<PathBuf>,
    #[arg(last = true, required = true)]
    command: Vec<OsString>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct ChildReceiptV2 {
    schema: String,
    observed: ObservedEncoderRuntime,
    output_name: String,
    output_vector_count: u32,
    output_feature_count: u32,
    output_size_bytes: u64,
    output_finite: bool,
    extra_output_count: u32,
    timing: EncoderStageTiming,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if let Err(error) = run(&args) {
        let _ = write_failed_attempt(&args, &error);
        return Err(error);
    }
    Ok(())
}

fn run(args: &Args) -> Result<()> {
    let started = Instant::now();
    validate_simple_name(&args.output_name)?;
    validate_simple_name(&args.child_receipt_name)?;
    validate_simple_name(&args.execution_receipt_name)?;
    validate_simple_name(&args.attempt_name)?;
    ensure!(
        BTreeSet::from([
            args.output_name.as_str(),
            args.child_receipt_name.as_str(),
            args.execution_receipt_name.as_str(),
            args.attempt_name.as_str(),
        ])
        .len()
            == 4,
        "encoder output names collide"
    );
    ensure!(
        is_sha256(&args.scq_job_identity_sha256),
        "invalid SCQ job identity"
    );
    let encoder = decode_encoder_identity(&fs::read(&args.encoder_identity)?)?;
    let input = decode_dataset_input(&fs::read(&args.dataset_input)?)?;
    ensure_private_directory(&args.input_root, false)?;
    ensure_private_directory(&args.output_root, true)?;

    let input_hash_started = Instant::now();
    verify_declared_inputs(&args.input_root, &input)?;
    let input_hash_ns = elapsed_ns(input_hash_started);
    let runtime_artifacts = encoder_artifacts(&encoder);
    verify_runtime_artifacts(&args.runtime_root, &runtime_artifacts)?;
    let runtime_artifacts_sha256 = runtime_artifacts_digest(&runtime_artifacts)?;

    let command_path = Path::new(&args.command[0]);
    ensure!(
        command_path.is_absolute(),
        "encoder child executable is not absolute"
    );
    ensure!(
        command_path.starts_with(&args.runtime_root),
        "encoder child executable is outside the immutable runtime"
    );
    let mut system_roots = vec![
        PathBuf::from("/usr"),
        PathBuf::from("/lib"),
        PathBuf::from("/lib64"),
        PathBuf::from("/etc/ld.so.cache"),
        PathBuf::from("/dev"),
        PathBuf::from("/proc"),
    ];
    system_roots.extend(args.system_read_root.iter().cloned());
    system_roots.retain(|path| path.exists());
    system_roots = system_roots
        .into_iter()
        .map(fs::canonicalize)
        .collect::<std::io::Result<Vec<_>>>()?;
    system_roots.sort();
    system_roots.dedup();
    for root in &system_roots {
        ensure!(
            root.is_absolute()
                && !root.starts_with("/home")
                && !root.to_string_lossy().contains("jope-dime"),
            "system read root is mutable or noncanonical: {}",
            root.display()
        );
    }
    let mut device_paths = fs::read_dir("/dev")?
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.file_name().to_string_lossy().starts_with("nvidia"))
        .map(|entry| entry.path())
        .collect::<Vec<_>>();
    device_paths.sort();

    let ruleset = LandlockRuleset::new()?;
    let read_execute = ACCESS_READ_FILE | ACCESS_READ_DIR | ACCESS_EXECUTE;
    let read_only = ACCESS_READ_FILE | ACCESS_READ_DIR;
    let write_output = ACCESS_READ_FILE
        | ACCESS_READ_DIR
        | ACCESS_WRITE_FILE
        | ACCESS_REMOVE_DIR
        | ACCESS_REMOVE_FILE
        | ACCESS_MAKE_DIR
        | ACCESS_MAKE_REG
        | ACCESS_MAKE_SYM
        | ACCESS_REFER
        | ACCESS_TRUNCATE;
    ruleset.allow_path(&args.runtime_root, read_execute)?;
    ruleset.allow_path(&args.input_root, read_only)?;
    ruleset.allow_path(&args.output_root, write_output)?;
    for root in &system_roots {
        ruleset.allow_path(root, read_execute)?;
    }
    for device in &device_paths {
        ruleset.allow_path(device, ACCESS_READ_FILE | ACCESS_WRITE_FILE)?;
    }
    let sandbox = SandboxPolicy {
        schema: SANDBOX_SCHEMA,
        landlock_abi: ruleset.abi,
        runtime_read_execute_root: &args.runtime_root,
        input_read_root: &args.input_root,
        output_write_root: &args.output_root,
        system_read_execute_roots: &system_roots,
        device_read_write_paths: &device_paths,
        denied_classes: [
            "live_checkouts",
            "source_registries",
            "test_files",
            "arbitrary_home_paths",
            "v1_cache_roots",
        ],
    };
    let sandbox_policy_sha256 = canonical_json_sha256(&serde_json::to_value(&sandbox)?)?;

    let child_receipt_path = args.output_root.join(&args.child_receipt_name);
    let mut child_receipt_file = OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(&child_receipt_path)?;
    let receipt_raw_fd = child_receipt_file.as_raw_fd();
    let ruleset_raw_fd = ruleset.fd.as_raw_fd();
    let mut command = Command::new(&args.command[0]);
    command
        .args(args.command.iter().skip(1))
        .current_dir(&args.input_root)
        .env_clear()
        .env("PATH", args.runtime_root.join("bin"))
        .env("JOPE_ENCODER_RECEIPT_FD", CHILD_RECEIPT_FD.to_string())
        .env("JOPE_ENCODER_INPUT_ROOT", &args.input_root)
        .env("JOPE_ENCODER_OUTPUT_ROOT", &args.output_root)
        .env("JOPE_ENCODER_OUTPUT_NAME", &args.output_name);
    // SAFETY: the closure makes only async-signal-safe syscalls (dup2, fcntl, prctl, Landlock) and
    // reads errno, never allocating or locking; child_receipt_file and ruleset outlive status().
    unsafe {
        command.pre_exec(move || {
            if libc::dup2(receipt_raw_fd, CHILD_RECEIPT_FD) < 0 {
                return Err(std::io::Error::last_os_error());
            }
            let flags = libc::fcntl(CHILD_RECEIPT_FD, libc::F_GETFD);
            if flags < 0
                || libc::fcntl(CHILD_RECEIPT_FD, libc::F_SETFD, flags & !libc::FD_CLOEXEC) < 0
            {
                return Err(std::io::Error::last_os_error());
            }
            if libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            if libc::syscall(libc::SYS_landlock_restrict_self, ruleset_raw_fd, 0u32) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    let status = command
        .status()
        .context("execute sandboxed encoder child")?;
    child_receipt_file.sync_all()?;
    ensure!(status.success(), "encoder child failed with {status}");

    let child_outputs = collect_files(&args.output_root)?;
    let expected_child_outputs =
        BTreeSet::from([args.output_name.clone(), args.child_receipt_name.clone()]);
    ensure!(
        child_outputs == expected_child_outputs,
        "encoder child created missing or extra output"
    );
    child_receipt_file.seek(SeekFrom::Start(0))?;
    let mut child_receipt_bytes = Vec::new();
    child_receipt_file.read_to_end(&mut child_receipt_bytes)?;
    ensure!(
        !child_receipt_bytes.is_empty(),
        "encoder child receipt is empty"
    );
    let child: ChildReceiptV2 =
        serde_json::from_slice(&child_receipt_bytes).context("decode encoder child receipt")?;
    ensure!(
        serde_json::to_vec(&child)? == child_receipt_bytes,
        "encoder child receipt is not one canonical document"
    );
    ensure!(
        child.schema == CHILD_SCHEMA,
        "invalid encoder child receipt schema"
    );
    ensure!(
        child.output_name == args.output_name,
        "child receipt output-name mismatch"
    );
    let child_receipt_sha256 = sha256_bytes(&child_receipt_bytes);
    let output_path = args.output_root.join(&args.output_name);
    let output = fs::read(&output_path)?;
    let output_sha256 = validate_output_bytes(&output)?;
    ensure!(
        child.output_vector_count == 1
            && child.output_feature_count == OUTPUT_FEATURES
            && child.output_size_bytes == OUTPUT_BYTES
            && child.output_finite
            && child.extra_output_count == 0,
        "child receipt output contract mismatch"
    );
    fs::set_permissions(&output_path, fs::Permissions::from_mode(0o444))?;

    let now = now_unix_ms()?;
    let encoder_identity_sha256 = encoder.digest()?;
    let input_identity_sha256 = input.digest()?;
    let cache_key_sha256 = cache_key(&encoder, &input)?;
    let mut timing = child.timing;
    timing.input_hash_ns = input_hash_ns;
    timing.total_ns = elapsed_ns(started);
    let execution = EncoderExecutionReceiptV2 {
        schema: EXECUTION_SCHEMA.to_owned(),
        generated_at_unix_ms: now,
        freshness_budget_seconds: 300,
        scq_job_identity_sha256: args.scq_job_identity_sha256.clone(),
        encoder_identity_sha256,
        input_identity_sha256,
        cache_key_sha256: cache_key_sha256.clone(),
        declared_inputs: input
            .files_read
            .iter()
            .map(|read| read.artifact.clone())
            .collect(),
        sandbox_policy_sha256,
        child_receipt_sha256,
        runtime_artifacts_sha256,
        observed: child.observed,
        output_sha256: output_sha256.clone(),
        output_vector_count: child.output_vector_count,
        output_feature_count: child.output_feature_count,
        output_size_bytes: child.output_size_bytes,
        output_finite: child.output_finite,
        extra_output_count: child.extra_output_count,
        timing,
    };
    execution.validate_for(&encoder, &input, now)?;
    let execution_sha256 = execution.digest()?;
    let attempt = EncoderAttemptReceiptV2 {
        schema: ATTEMPT_SCHEMA.to_owned(),
        generated_at_unix_ms: now,
        freshness_budget_seconds: 300,
        node: hostname()?,
        scq_job_identity_sha256: args.scq_job_identity_sha256.clone(),
        cache_key_sha256,
        status: "succeeded".to_owned(),
        failure_class: None,
        execution_receipt_sha256: Some(execution_sha256),
        output_sha256: Some(output_sha256),
        queue_wait_ns: 0,
        total_ns: elapsed_ns(started),
    };
    attempt.validate()?;
    write_json_new(
        &args.output_root.join(&args.execution_receipt_name),
        &execution,
    )?;
    write_json_new(&args.output_root.join(&args.attempt_name), &attempt)?;
    File::open(&args.output_root)?.sync_all()?;
    Ok(())
}

fn write_failed_attempt(args: &Args, error: &anyhow::Error) -> Result<()> {
    if !args.output_root.is_dir() || !is_sha256(&args.scq_job_identity_sha256) {
        return Ok(());
    }
    let encoder = decode_encoder_identity(&fs::read(&args.encoder_identity)?)?;
    let input = decode_dataset_input(&fs::read(&args.dataset_input)?)?;
    let text = format!("{error:#}").to_ascii_lowercase();
    let failure_class = if text.contains("landlock") {
        "capability"
    } else if text.contains("backend") || text.contains("device") {
        "backend_mismatch"
    } else if text.contains("package") || text.contains("runtime artifact") {
        "package_drift"
    } else if text.contains("input") || text.contains("test") {
        "input_drift"
    } else if text.contains("nonfinite") {
        "nonfinite_output"
    } else if text.contains("output") {
        "output_shape"
    } else if text.contains("receipt") {
        "malformed_receipt"
    } else {
        "child_exit"
    };
    let attempt = EncoderAttemptReceiptV2 {
        schema: ATTEMPT_SCHEMA.to_owned(),
        generated_at_unix_ms: now_unix_ms()?,
        freshness_budget_seconds: 300,
        node: hostname()?,
        scq_job_identity_sha256: args.scq_job_identity_sha256.clone(),
        cache_key_sha256: cache_key(&encoder, &input)?,
        status: "failed".to_owned(),
        failure_class: Some(failure_class.to_owned()),
        execution_receipt_sha256: None,
        output_sha256: None,
        queue_wait_ns: 0,
        total_ns: 1,
    };
    attempt.validate()?;
    let path = args.output_root.join(&args.attempt_name);
    if !path.exists() {
        write_json_new(&path, &attempt)?;
    }
    Ok(())
}

fn validate_simple_name(value: &str) -> Result<()> {
    ensure!(
        Path::new(value).components().count() == 1
            && matches!(
                Path::new(value).components().next(),
                Some(Component::Normal(_))
            ),
        "output name is not a simple relative filename"
    );
    Ok(())
}

fn ensure_private_directory(path: &Path, must_be_empty: bool) -> Result<()> {
    let metadata = fs::symlink_metadata(path)?;
    ensure!(
        metadata.is_dir() && !metadata.file_type().is_symlink(),
        "job path is not a directory"
    );
    ensure!(
        metadata.mode() & 0o002 == 0,
        "job-private directory is world-writable"
    );
    if must_be_empty {
        ensure!(
            fs::read_dir(path)?.next().is_none(),
            "output directory is not empty"
        );
    }
    Ok(())
}

fn write_json_new<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let bytes = serde_json::to_vec(value)?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o444)
        .open(path)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    Ok(())
}

fn now_unix_ms() -> Result<u64> {
    Ok(SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis() as u64)
}

fn elapsed_ns(started: Instant) -> u64 {
    started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64
}

fn hostname() -> Result<String> {
    let bytes = fs::read("/etc/hostname")?;
    let hostname = String::from_utf8(bytes)?.trim().to_owned();
    ensure!(!hostname.is_empty(), "hostname is empty");
    Ok(hostname)
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(test)]
#[path = "jope_encoder_supervisor/tests.rs"]
mod tests;
