/// Largest preparation state a projected export input may carry.
const FIT_MEMORY_MAX_PREPROCESSING: usize = 16 * 1024 * 1024;

/// A sealed anonymous memory file holding exactly one cut's projected training
/// rows and the verified bytes of its learned preparation state: the export
/// worker's whole input (W1 follow-up 2c). Layout: three little-endian u64
/// (rows, width, preparation length), the preparation bytes, then the rows as
/// little-endian f64. Inheritable, so a spawned worker receives it by number.
pub fn fit_memory_input(rows: &[Vec<f64>], preprocessing: &[u8]) -> Result<fs::File> {
    use std::io::Write;
    use std::os::fd::{AsRawFd, FromRawFd};
    let width = rows.first().context("empty projected training rows")?.len();
    ensure!(
        width >= 2
            && rows.len() <= ROW_LIMIT
            && rows.iter().all(|r| r.len() == width)
            && preprocessing.len() <= FIT_MEMORY_MAX_PREPROCESSING,
        "invalid projected export input shape"
    );
    let name = std::ffi::CString::new("jope-fit-input").unwrap();
    // SAFETY: `name` is a CString that outlives the call, so the pointer is NUL-terminated and
    // valid; MFD_ALLOW_SEALING is a valid flag.
    let fd = unsafe { libc::memfd_create(name.as_ptr(), libc::MFD_ALLOW_SEALING) };
    ensure!(fd >= 0, "memfd_create failed");
    // SAFETY: memfd_create returned this fresh, nonnegative descriptor and nothing else owns
    // it; File takes sole ownership and closes it once.
    let file = unsafe { fs::File::from_raw_fd(fd) };
    {
        let mut out = std::io::BufWriter::new(&file);
        for n in [rows.len(), width, preprocessing.len()] {
            out.write_all(&(n as u64).to_le_bytes())?;
        }
        out.write_all(preprocessing)?;
        for row in rows {
            for value in row {
                out.write_all(&value.to_le_bytes())?;
            }
        }
        out.flush()?;
    }
    ensure!(
        // SAFETY: `file` owns the descriptor and outlives the call; F_ADD_SEALS takes an
        // integer seal mask and reads or writes no caller memory.
        unsafe {
            libc::fcntl(
                file.as_raw_fd(),
                libc::F_ADD_SEALS,
                libc::F_SEAL_WRITE | libc::F_SEAL_GROW | libc::F_SEAL_SHRINK | libc::F_SEAL_SEAL,
            )
        } == 0,
        "sealing projected export input failed"
    );
    Ok(file)
}

/// The export worker (hidden `fit-memory` command): fits exactly the sealed
/// projected input inherited as `fd`. It is handed no prepared path, root or
/// backing table and opens none; the rows and preparation state must match the
/// identities its parent read from the cut.
pub fn fit_memory(
    fd: i32,
    training_sha256: &str,
    preprocessing_sha256: &str,
    genome: &str,
    output: &Path,
) -> Result<()> {
    use std::io::{Read, Seek, SeekFrom};
    use std::os::fd::FromRawFd;
    ensure!(!output.exists(), "fitted predictor already exists");
    ensure!(fd >= 3, "invalid inherited projected input descriptor");
    // SAFETY: F_GET_SEALS takes no pointer and only reads the seals of `fd`; it does not take
    // ownership, and a failure returns -1, which is checked below.
    let seals = unsafe { libc::fcntl(fd, libc::F_GET_SEALS) };
    ensure!(
        seals >= 0 && seals & libc::F_SEAL_WRITE != 0 && seals & libc::F_SEAL_SHRINK != 0,
        "projected export input must be sealed memory"
    );
    // SAFETY: dup takes a plain integer and never affects the inherited descriptor; a number
    // that is not open makes it return -1, which is checked below.
    let copy = unsafe { libc::dup(fd) };
    ensure!(copy >= 0, "projected input descriptor missing");
    // SAFETY: dup just returned `copy` as a fresh nonnegative descriptor no one else owns; File
    // takes sole ownership and closes it once, leaving the inherited `fd` open.
    let mut file = unsafe { fs::File::from_raw_fd(copy) };
    file.seek(SeekFrom::Start(0))?;
    let mut header = [0u8; 24];
    file.read_exact(&mut header)?;
    let dims = header
        .chunks_exact(8)
        .map(|b| u64::from_le_bytes(b.try_into().unwrap()) as usize)
        .collect::<Vec<_>>();
    ensure!(
        dims[0] > 0
            && dims[0] <= ROW_LIMIT
            && dims[1] >= 2
            && dims[2] <= FIT_MEMORY_MAX_PREPROCESSING,
        "invalid projected export input header"
    );
    let mut preprocessing = vec![0u8; dims[2]];
    file.read_exact(&mut preprocessing)?;
    ensure!(
        hash(&preprocessing) == preprocessing_sha256,
        "projected preparation state differs from the cut's identity"
    );
    let cells = dims[0]
        .checked_mul(dims[1])
        .context("projected input dimensions overflow")?;
    let mut bytes = vec![0u8; cells * 8];
    file.read_exact(&mut bytes)?;
    let mut rest = [0u8; 1];
    ensure!(
        file.read(&mut rest)? == 0,
        "projected export input has trailing bytes"
    );
    let rows = bytes
        .chunks_exact(dims[1] * 8)
        .map(|row| {
            row.chunks_exact(8)
                .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    fit_projected(&rows, training_sha256, preprocessing, genome, output)
}

/// Export a fitted predictor under the same CPU, memory and time limits as a
/// teacher candidate. A failed export must not strand the neural cycle.
/// Legacy callers without a declared root: the parent projects only a
/// contained view (the cut's own tables); a pointer view is a receipted
/// refusal before any worker starts, never resolved from the view's
/// ancestors (W1 route R2). The selected-export handoff declares its root
/// ([`fit_view_isolated_in_root`]). Either way the worker receives only the
/// sealed projection.
pub fn fit_view_isolated(
    executable: &Path,
    prepared: &Path,
    cut: usize,
    genome: &str,
    output: &Path,
) -> Result<()> {
    isolated_export(executable, None, prepared, cut, genome, output)
}

/// The isolated fitted-view export (W1 follow-up 2c): the parent projects the
/// cut's fitting-only input under the declared `root` - digest-verified
/// training rows and preparation state - and the worker (`fit-memory`)
/// receives only that sealed projection; it is given no prepared path and
/// resolves no view. The receipt records the root, the projection and the
/// worker's exact argv. A projection that cannot be made under the root is a
/// receipted failure before any worker starts.
pub fn fit_view_isolated_in_root(
    executable: &Path,
    root: &Path,
    prepared: &Path,
    cut: usize,
    genome: &str,
    output: &Path,
) -> Result<()> {
    isolated_export(executable, Some(root), prepared, cut, genome, output)
}

fn isolated_export(
    executable: &Path,
    root: Option<&Path>,
    prepared: &Path,
    cut: usize,
    genome: &str,
    output: &Path,
) -> Result<()> {
    use std::os::fd::AsRawFd;
    use std::os::unix::process::CommandExt;
    ensure!(!output.exists(), "fitted predictor already exists");
    let executable_sha256 = file_hash(executable)?;
    let prepared_bytes = fs::read(prepared.join("prepared.json"))?;
    let prepared_sha256 = hash(&prepared_bytes);
    let program_sha256 = hash(genome.as_bytes());
    fs::create_dir_all(output.parent().context("predictor output parent absent")?)?;
    let log_path = output.with_extension("fit.log");
    let receipt_path = output.with_extension("fit.json");
    let log = fs::File::create(&log_path)?;
    let started = Instant::now();
    let mut projection = serde_json::Value::Null;
    let mut worker_argv: Vec<String> = Vec::new();
    let result = (|| -> Result<()> {
        let p: Prepared = serde_json::from_slice(&prepared_bytes)?;
        ensure!(
            p.schema == "parallel.prepared.v1" && p.task == "regression",
            "regression preparation required"
        );
        let c = p
            .cuts
            .iter()
            .find(|c| c.index == cut)
            .context("unknown cut")?;
        // The fitting-only projection, made here from the preparation read
        // above: under the declared root, or for a legacy caller only from a
        // contained view's own tables (W1 route R2).
        let rows = project_fitting_view(root, prepared, &p, cut)?;
        let preprocessing = read_verified(
            &prepared.join(format!("cut-{cut}/preprocessing.json")),
            &c.preprocessing_sha256,
            "preprocessing",
        )?;
        let memory = fit_memory_input(&rows, &preprocessing)?;
        projection = json!({"kind":"sealed_memory_projection","rows":rows.len(),
            "training_sha256":c.training_sha256,"preprocessing_sha256":c.preprocessing_sha256});
        worker_argv = [
            "fit-memory",
            "--fd",
            &memory.as_raw_fd().to_string(),
            "--training-sha256",
            &c.training_sha256,
            "--preprocessing-sha256",
            &c.preprocessing_sha256,
            "--genome",
            genome,
            "--output",
        ]
        .iter()
        .map(|a| a.to_string())
        .chain([output.display().to_string()])
        .collect();
        let mut command = evaluator_command(executable);
        command
            .args(&worker_argv)
            .env("RAYON_NUM_THREADS", "1")
            .env("OMP_NUM_THREADS", "1")
            .env("OPENBLAS_NUM_THREADS", "1")
            .env("MKL_NUM_THREADS", "1")
            .env(
                "JOPE_FIT_DEADLINE_MONOTONIC_NS",
                crate::fit_boundary::deadline_after(Duration::from_secs(30))?.to_string(),
            )
            .stdout(Stdio::from(log.try_clone()?))
            .stderr(Stdio::from(log.try_clone()?));
        // SAFETY: the hook only calls async-signal-safe setrlimit (given its stack-local rlimit)
        // and setpgid, and builds its error from errno without allocating.
        unsafe {
            command.pre_exec(|| {
                let limit = libc::rlimit {
                    rlim_cur: 8 * 1024 * 1024 * 1024,
                    rlim_max: 8 * 1024 * 1024 * 1024,
                };
                if libc::setrlimit(libc::RLIMIT_AS, &limit) != 0 || libc::setpgid(0, 0) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
        crate::campaign::check_deadline()?;
        let mut child = command.spawn()?;
        let waited = (|| -> Result<()> {
            loop {
                crate::campaign::check_deadline()?;
                if let Some(status) = child.try_wait()? {
                    ensure!(status.success(), "fitted predictor worker exited: {status}");
                    break;
                }
                ensure!(
                    started.elapsed() < Duration::from_secs(30),
                    "fitted predictor worker timeout after 30 seconds"
                );
                std::thread::sleep(Duration::from_millis(10));
            }
            ensure!(
                output.is_file(),
                "fitted predictor worker produced no predictor"
            );
            let fitted: Predictor = read_json(output)?;
            ensure!(
                fitted.canonical == GenomeIr::parse(genome)?.canonical(),
                "fitted predictor program changed"
            );
            ensure!(
                fitted.training_sha256 == c.training_sha256,
                "fitted predictor cut changed"
            );
            Ok(())
        })();
        if child.try_wait()?.is_none() {
            // SAFETY: kill takes only integers and has no memory-safety preconditions; the
            // worker is unreaped, so the group id it took with setpgid(0, 0) is not reused.
            unsafe {
                libc::kill(-(child.id() as i32), libc::SIGKILL);
            }
            let _ = child.wait();
        }
        drop(memory);
        waited
    })();
    write_json(
        &receipt_path,
        &json!({
            "schema":"parallel.fitted_view_export.v1", "passed":result.is_ok(),
            "executable_sha256":executable_sha256, "prepared_sha256":prepared_sha256,
            "cut":cut, "program_sha256":program_sha256,
            "predictor_sha256":if result.is_ok(){Some(file_hash(output)?)}else{None},
            "wall_seconds":started.elapsed().as_secs_f64(), "timeout_seconds":30,
            "memory_limit_gib":8, "threads":1, "cut_csv_copies":0,
            "artifact_root":root,
            "view_resolution":if root.is_some(){"declared_root"}else{"contained_view"},
            "worker_input":projection, "worker_argv":worker_argv,
            "worker_opens_prepared":false,
            "failure":result.as_ref().err().map(|e|format!("{e:#}")),
            "log_sha256":file_hash(&log_path)?, "final_tests_opened":false
        }),
    )?;
    result
}
