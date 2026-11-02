use super::*;

#[derive(Debug)]
struct EncoderRun {
    status: ExitStatus,
    proof: EncoderExecutionProofV2,
}

fn encoder_process_group_exists(pid: libc::pid_t) -> Result<bool, String> {
    #[cfg(unix)]
    // SAFETY: kill with signal 0 sends nothing and reads no memory; every caller passes a
    // spawned child's PID (never 0 or 1), so -pid names one process group.
    unsafe {
        if libc::kill(-pid, 0) == 0 {
            return Ok(true);
        }
        let error = std::io::Error::last_os_error();
        if error.raw_os_error() == Some(libc::ESRCH) {
            return Ok(false);
        }
        Err(format!("probe Hyperion process group: {error}"))
    }
    #[cfg(not(unix))]
    {
        let _ = pid;
        Ok(false)
    }
}

fn wait_for_encoder_group_exit(pid: libc::pid_t, grace: Duration) -> Result<bool, String> {
    let deadline = Instant::now() + grace;
    loop {
        if !encoder_process_group_exists(pid)? {
            return Ok(true);
        }
        if Instant::now() >= deadline {
            return Ok(false);
        }
        thread::sleep(Duration::from_millis(10));
    }
}

fn terminate_encoder_group(child: &mut std::process::Child, grace: Duration) -> Result<(), String> {
    let pid = child.id() as libc::pid_t;
    #[cfg(unix)]
    // SAFETY: kill reads no memory; pid is the spawned child's PID (never 0 or 1), so -pid names
    // one process group and never the 0/-1 broadcast forms.
    unsafe {
        if libc::kill(-pid, libc::SIGTERM) != 0
            && std::io::Error::last_os_error().raw_os_error() != Some(libc::ESRCH)
        {
            return Err(format!(
                "terminate Hyperion process group: {}",
                std::io::Error::last_os_error()
            ));
        }
    }
    let deadline = Instant::now() + grace;
    let mut leader_reaped = false;
    while Instant::now() < deadline {
        if !leader_reaped
            && child
                .try_wait()
                .map_err(|error| format!("poll Hyperion process group: {error}"))?
                .is_some()
        {
            leader_reaped = true;
        }
        if leader_reaped && !encoder_process_group_exists(pid)? {
            return Ok(());
        }
        thread::sleep(Duration::from_millis(10));
    }
    #[cfg(unix)]
    // SAFETY: kill reads no memory; pid is the spawned child's PID (never 0 or 1), so -pid names
    // one process group and never the 0/-1 broadcast forms.
    unsafe {
        if libc::kill(-pid, libc::SIGKILL) != 0
            && std::io::Error::last_os_error().raw_os_error() != Some(libc::ESRCH)
        {
            return Err(format!(
                "kill Hyperion process group: {}",
                std::io::Error::last_os_error()
            ));
        }
    }
    if !leader_reaped {
        child
            .wait()
            .map_err(|error| format!("reap Hyperion process group: {error}"))?;
    }
    if !wait_for_encoder_group_exit(pid, grace)? {
        return Err("Hyperion process group survived SIGKILL grace period".into());
    }
    Ok(())
}

pub(crate) fn run_encoder_with_timeout(
    encoder: &HyperionEncoderIdentityV1,
    fit: &Path,
    hold: &Path,
    outputs: &[(&str, PathBuf, usize, usize); 4],
    timeout: Duration,
) -> Result<EncoderRun, String> {
    let mut header = [0u8; 4];
    fs::File::open(&encoder.binary.path)
        .and_then(|mut file| file.read_exact(&mut header))
        .map_err(|error| format!("read encoder binary header: {error}"))?;
    if header != [0x7f, b'E', b'L', b'F'] {
        return Err("Hyperion encoder must be a pinned native ELF binary".into());
    }
    let scales = encoder
        .scales
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let visible_device = encoder
        .runtime_versions
        .get("gpu_uuid")
        .filter(|value| !value.trim().is_empty())
        .cloned()
        .unwrap_or_else(|| encoder.device_ordinal.to_string());
    let encoder_identity_sha256 = encoder.identity()?;
    let argv_sha256 = canonical_sha256(&hyperion_argv_schema_v2())?;
    let environment_sha256 = canonical_sha256(&hyperion_environment_v2())?;
    let mut command = Command::new(&encoder.binary.path);
    command
        .arg("--train")
        .arg(fit)
        .arg("--test")
        .arg(hold)
        .arg("--out-cell-train")
        .arg(&outputs[0].1)
        .arg("--out-cell-test")
        .arg(&outputs[1].1)
        .arg("--out-row-train")
        .arg(&outputs[2].1)
        .arg("--out-row-test")
        .arg(&outputs[3].1)
        .arg("--weights")
        .arg(&encoder.weights.path)
        .arg("--manifest")
        .arg(&encoder.manifest.path)
        .arg("--sweeps")
        .arg(encoder.sweeps.to_string())
        .arg("--scales")
        .arg(scales)
        .arg("--seed")
        .arg(encoder.seed.to_string())
        .arg("--batch")
        .arg(encoder.batch_size.to_string())
        .env("CUDA_VISIBLE_DEVICES", &visible_device)
        .env("JAIN_HYPERION_CUDA_MAX_LOAD_RATIO", "1000000")
        .env("OMP_NUM_THREADS", "2")
        .env("OPENBLAS_NUM_THREADS", "2")
        .env("MKL_NUM_THREADS", "2")
        .env("BLIS_NUM_THREADS", "2")
        .env("VECLIB_MAXIMUM_THREADS", "2")
        .env("NUMEXPR_NUM_THREADS", "2")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(unix)]
    // SAFETY: the hook runs between fork and exec and makes only async-signal-safe calls (setsid,
    // prctl, getppid); its errors are errno codes, so nothing allocates in the child.
    unsafe {
        let parent_pid = std::process::id() as libc::pid_t;
        command.pre_exec(move || {
            if libc::setsid() < 0 {
                return Err(std::io::Error::last_os_error());
            }
            if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            if libc::getppid() != parent_pid {
                // The parent exited: refuse with an errno, which does not allocate
                // (an io::Error carrying a message would box it in the forked child).
                return Err(std::io::Error::from_raw_os_error(libc::ESRCH));
            }
            Ok(())
        });
    }
    let mut child = command
        .spawn()
        .map_err(|error| format!("start fold-local Hyperion encoder: {error}"))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "capture Hyperion encoder stdout failed".to_string())?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| "capture Hyperion encoder stderr failed".to_string())?;
    let stdout_reader = thread::spawn(move || {
        let mut bytes = Vec::new();
        let mut stdout = stdout;
        stdout.read_to_end(&mut bytes).map(|_| bytes)
    });
    let stderr_reader = thread::spawn(move || {
        let mut bytes = Vec::new();
        let mut stderr = stderr;
        stderr.read_to_end(&mut bytes).map(|_| bytes)
    });
    let started = Instant::now();
    let status = loop {
        let polled = match child.try_wait() {
            Ok(status) => status,
            Err(error) => {
                let cleanup = terminate_encoder_group(&mut child, Duration::from_millis(250));
                let _ = stdout_reader.join();
                let _ = stderr_reader.join();
                if let Err(cleanup) = cleanup {
                    return Err(format!(
                        "poll Hyperion encoder: {error}; cleanup failed: {cleanup}"
                    ));
                }
                return Err(format!("poll Hyperion encoder: {error}"));
            }
        };
        if let Some(status) = polled {
            if !encoder_process_group_exists(child.id() as libc::pid_t)? {
                break status;
            }
            terminate_encoder_group(&mut child, Duration::from_millis(250))?;
            return Err("Hyperion encoder exited with surviving process group".into());
        }
        if started.elapsed() >= timeout {
            let cleanup = terminate_encoder_group(&mut child, Duration::from_millis(250));
            let _ = stdout_reader.join();
            let _ = stderr_reader.join();
            cleanup?;
            return Err("fold-local Hyperion encoder timed out".into());
        }
        thread::sleep(Duration::from_millis(20));
    };
    let stdout = stdout_reader
        .join()
        .map_err(|_| "Hyperion stdout capture thread panicked".to_string())?
        .map_err(|error| format!("capture Hyperion encoder stdout: {error}"))?;
    let stderr = stderr_reader
        .join()
        .map_err(|_| "Hyperion stderr capture thread panicked".to_string())?
        .map_err(|error| format!("capture Hyperion encoder stderr: {error}"))?;
    let stdout = String::from_utf8(stdout)
        .map_err(|error| format!("Hyperion encoder stdout is not UTF-8: {error}"))?;
    let stderr = String::from_utf8(stderr)
        .map_err(|error| format!("Hyperion encoder stderr is not UTF-8: {error}"))?;
    let expected_device = "cuda:0";
    let proof = EncoderExecutionProofV2 {
        encoder_identity_sha256,
        gpu_uuid: visible_device.clone(),
        cuda_visible_devices: visible_device,
        argv_sha256,
        environment_sha256,
        observed_device: expected_device.into(),
        stdout_sha256: raw_sha256(stdout.as_bytes()),
        stderr_sha256: raw_sha256(stderr.as_bytes()),
        stdout,
        stderr,
        elapsed_milliseconds: elapsed_milliseconds(started),
    };
    Ok(EncoderRun { status, proof })
}

pub(crate) fn read_output_matrices(
    outputs: &[(&str, PathBuf, usize, usize)],
) -> Result<Vec<Vec<Vec<f64>>>, String> {
    outputs
        .iter()
        .map(|(role, path, rows, columns)| read_vs_matrix(path, *rows, *columns, role))
        .collect()
}

pub(crate) fn output_artifacts(
    outputs: &[(&str, PathBuf, usize, usize)],
) -> Result<Vec<MatrixArtifactV1>, String> {
    outputs
        .iter()
        .map(|(role, path, rows, columns)| {
            let bytes = fs::read(path)
                .map_err(|error| format!("read {role} sidecar {}: {error}", path.display()))?;
            Ok(MatrixArtifactV1 {
                role: (*role).into(),
                rows: *rows as u32,
                columns: *columns as u32,
                sha256: raw_sha256(&bytes),
            })
        })
        .collect()
}

pub(crate) fn read_vs_matrix(
    path: &Path,
    expected_rows: usize,
    expected_columns: usize,
    role: &str,
) -> Result<Vec<Vec<f64>>, String> {
    let bytes = fs::read(path)
        .map_err(|error| format!("read {role} sidecar {}: {error}", path.display()))?;
    let text = std::str::from_utf8(&bytes)
        .map_err(|error| format!("{role} sidecar is not UTF-8: {error}"))?;
    let matrix = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            line.split(|character: char| character == ',' || character.is_ascii_whitespace())
                .filter(|field| !field.is_empty())
                .map(|field| {
                    field
                        .parse::<f64>()
                        .map_err(|error| format!("parse {role} sidecar value: {error}"))
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    if matrix.len() != expected_rows
        || matrix
            .iter()
            .any(|row| row.len() != expected_columns || row.iter().any(|value| !value.is_finite()))
    {
        return Err(format!(
            "{role} sidecar shape/content is invalid; expected {expected_rows}x{expected_columns}"
        ));
    }
    Ok(matrix)
}

pub(crate) fn require_cuda_device(ordinal: u32) -> Result<(), String> {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=index", "--format=csv,noheader,nounits"])
        .output()
        .map_err(|error| format!("CUDA GPU unavailable: {error}"))?;
    if !output.status.success()
        || !String::from_utf8_lossy(&output.stdout)
            .lines()
            .any(|line| line.trim() == ordinal.to_string())
    {
        return Err(format!("pinned CUDA GPU ordinal {ordinal} is unavailable"));
    }
    Ok(())
}

pub(crate) fn verify_runtime_identity(encoder: &HyperionEncoderIdentityV1) -> Result<(), String> {
    let rustc = Command::new("rustc")
        .arg("--version")
        .output()
        .map_err(|error| format!("read pinned Rust runtime identity: {error}"))?;
    let observed_rustc = String::from_utf8_lossy(&rustc.stdout).trim().to_owned();
    if !rustc.status.success() || encoder.runtime_versions["rustc"] != observed_rustc {
        return Err("Rust runtime differs from the encoder identity".into());
    }
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,name,driver_version,uuid",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .map_err(|error| format!("read pinned CUDA runtime identity: {error}"))?;
    if !output.status.success() {
        return Err("CUDA runtime identity query failed".into());
    }
    let expected_ordinal = encoder.device_ordinal.to_string();
    let output_text = String::from_utf8_lossy(&output.stdout);
    let observed = output_text
        .lines()
        .find_map(|line| {
            let fields = line.split(',').map(str::trim).collect::<Vec<_>>();
            (fields.len() == 4 && fields[0] == expected_ordinal).then_some(fields)
        })
        .ok_or_else(|| "pinned CUDA device is absent from the runtime query".to_string())?;
    if encoder.runtime_versions["gpu_name"] != observed[1]
        || encoder.runtime_versions["cuda_driver"] != observed[2]
        || encoder.runtime_versions["gpu_uuid"] != observed[3]
    {
        return Err("CUDA device or driver differs from the encoder identity".into());
    }
    Ok(())
}

pub(crate) fn parse_regression_csv(
    bytes: &[u8],
    expected_features: usize,
) -> Result<RegressionTable, String> {
    let text = std::str::from_utf8(bytes)
        .map_err(|error| format!("authorized train CSV is not UTF-8: {error}"))?;
    let mut features = Vec::new();
    let mut targets = Vec::new();
    for (line_index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let mut values = line
            .split(',')
            .map(|field| {
                field.trim().parse::<f64>().map_err(|error| {
                    format!(
                        "authorized train row {} is nonnumeric: {error}",
                        line_index + 1
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        if values.len() != expected_features + 1 {
            return Err(format!(
                "authorized train row {} has the wrong feature width",
                line_index + 1
            ));
        }
        let target = values.pop().unwrap();
        if !target.is_finite() {
            return Err(format!(
                "authorized train row {} has a nonfinite target",
                line_index + 1
            ));
        }
        values.iter_mut().for_each(|value| {
            if !value.is_finite() {
                *value = 0.0;
            }
        });
        features.push(values);
        targets.push(target);
    }
    if features.is_empty() {
        return Err("authorized train CSV is empty".into());
    }
    Ok(RegressionTable { features, targets })
}

pub(crate) fn csv_for_rows(
    source: &RegressionTable,
    rows: &[usize],
    dummy_target: bool,
) -> Vec<u8> {
    let mut bytes = Vec::new();
    for row in rows {
        let target = if dummy_target {
            0.0
        } else {
            source.targets[*row]
        };
        let mut fields = source.features[*row]
            .iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>();
        fields.push(target.to_string());
        bytes.extend_from_slice(fields.join(",").as_bytes());
        bytes.push(b'\n');
    }
    bytes
}

pub(crate) fn select_rows(values: &[Vec<f64>], rows: &[usize]) -> Vec<Vec<f64>> {
    rows.iter().map(|row| values[*row].clone()).collect()
}

pub(crate) fn hash_selected_strings(values: &[String], rows: &[usize]) -> String {
    let mut bytes = Vec::new();
    for row in rows {
        bytes.extend_from_slice(values[*row].as_bytes());
        bytes.push(0);
    }
    raw_sha256(&bytes)
}

pub(crate) fn hash_strings(values: &[String]) -> String {
    let mut bytes = Vec::new();
    for value in values {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }
    raw_sha256(&bytes)
}

pub(crate) fn hash_matrix(values: &[Vec<f64>]) -> String {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&(values.first().map(Vec::len).unwrap_or(0) as u64).to_le_bytes());
    for row in values {
        for value in row {
            bytes.extend_from_slice(&value.to_bits().to_le_bytes());
        }
    }
    raw_sha256(&bytes)
}

pub(crate) fn hash_f64(values: &[f64]) -> String {
    let mut bytes = Vec::with_capacity(values.len() * 8);
    for value in values {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    raw_sha256(&bytes)
}

pub(crate) fn elapsed_milliseconds(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX)
}
