use crate::{
    manifest::*,
    prepare::{csv_bytes, Prepared, Preprocessing, Table},
};
use anyhow::{ensure, Context, Result};
use feat_math::{GenomeIr, Model};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{
    collections::BTreeSet,
    fs,
    path::Path,
    process::{Command, Stdio},
    time::{Duration, Instant},
};

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

#[cfg(test)]
mod fit_contract_tests;
pub mod handoff;
pub mod lineage;
pub mod outer_release;
pub mod outer_release_cli;
pub mod paired;
pub mod selection_replay;

/// The SHA-256 of `csv_bytes(rows)`, the rows' identity in cuts, feature views and score
/// identities, computed while the CSV is written: the table is never rendered into memory
/// (about 20 bytes per cell, several times the table) and each row's cell strings reuse one set
/// of buffers instead of allocating a `String` per cell. Same writer, same field text
/// (`f64` `Display`), so the same bytes reach the hasher.
fn csv_sha256(rows: &[Vec<f64>]) -> Result<String> {
    use sha2::Digest as _;
    use std::fmt::Write as _;
    let mut writer = csv::WriterBuilder::new()
        .has_headers(false)
        .from_writer(sha2::Sha256::new());
    let mut cells: Vec<String> = Vec::new();
    for row in rows {
        cells.resize_with(row.len(), String::new);
        for (cell, value) in cells.iter_mut().zip(row) {
            cell.clear();
            write!(cell, "{value}")?;
        }
        writer.write_record(&cells)?;
    }
    let hasher = writer
        .into_inner()
        .map_err(|error| anyhow::anyhow!("csv digest: {}", error.error()))?;
    Ok(hex::encode(hasher.finalize()))
}

#[cfg(test)]
mod csv_digest_tests {
    use super::*;

    #[test]
    fn streamed_csv_digest_equals_the_digest_of_the_rendered_csv() {
        let specials = [
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            0.0,
            -0.0,
            1e300,
            -1e-300,
            5e-324,
            f64::MAX,
            0.1 + 0.2,
            -123456.789,
            42.0,
        ];
        let tables: Vec<Vec<Vec<f64>>> = vec![
            vec![],
            vec![vec![]],
            vec![specials.to_vec()],
            specials.chunks(4).map(<[f64]>::to_vec).collect(),
            (0..200)
                .map(|i| (0..7).map(|j| (i * 7 + j) as f64 / 3.0 - 50.0).collect())
                .collect(),
        ];
        for rows in tables {
            assert_eq!(csv_sha256(&rows).unwrap(), hash(&csv_bytes(&rows).unwrap()));
        }
        // The csv writer refuses a record whose width differs from the first one; both paths
        // use it, so a ragged table is refused by both rather than given a digest.
        let ragged: Vec<Vec<f64>> = specials.chunks(5).map(<[f64]>::to_vec).collect();
        assert!(csv_bytes(&ragged).is_err());
        assert!(csv_sha256(&ragged).is_err());
    }
}

#[cfg(test)]
mod cpu_evaluator_tests {
    use super::*;

    #[test]
    fn explicit_evaluator_has_no_current_executable_fallback() {
        assert!(cpu_evaluator_binding(None).is_err());
        assert!(cpu_evaluator_binding(Some("relative-executable".into())).is_err());
        let executable = std::env::current_exe().unwrap();
        assert_eq!(
            cpu_evaluator_binding(Some(executable.clone().into_os_string())).unwrap(),
            executable.canonicalize().unwrap()
        );
        let directory = tempfile::tempdir().unwrap();
        assert!(cpu_evaluator_binding(Some(directory.path().as_os_str().into())).is_err());
        let plain = directory.path().join("not-executable");
        fs::write(&plain, b"not an executable").unwrap();
        assert!(cpu_evaluator_binding(Some(plain.into_os_string())).is_err());
    }
}
pub(crate) fn raw_feature_view(training: &str, preprocessing: &[u8]) -> Result<Vec<u8>> {
    Ok(serde_json::to_vec(
        &json!({"protocol":"raw_training_fold_v1",
        "training_sha256":training,"preprocessing_sha256":hash(preprocessing)}),
    )?)
}
fn relative_artifact(root: &Path, path: &Path) -> Result<serde_json::Value> {
    let absolute = path.canonicalize()?;
    let root = root.canonicalize()?;
    let relative = absolute
        .strip_prefix(&root)
        .context("original fit artifact outside workspace")?;
    Ok(json!({"path":relative,"sha256":file_hash(&absolute)?}))
}
fn relative_existing_reference(
    root: &Path,
    reference: &serde_json::Value,
) -> Result<serde_json::Value> {
    let path = Path::new(
        reference["path"]
            .as_str()
            .context("original fit reference missing")?,
    );
    let result = relative_artifact(
        root,
        &if path.is_absolute() {
            path.to_path_buf()
        } else {
            root.join(path)
        },
    )?;
    ensure!(
        result["sha256"] == reference["sha256"],
        "original fit reference changed"
    );
    Ok(result)
}

fn search_artifact_path(root: &Path, path: &Path) -> Result<std::path::PathBuf> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()?.join(path)
    };
    let relative = absolute
        .strip_prefix(root)
        .context("search artifact outside declared root")?;
    resolve(root, relative)
}

pub fn fit(train: &Path, genome: &str, output: &Path) -> Result<()> {
    ensure!(!output.exists(), "fitted predictor already exists");
    let table = Table::read(train, &serde_json::Value::Null)?;
    ensure!(table.rows.len() <= ROW_LIMIT, "fit row limit exceeded");
    let pp = Preprocessing::fit(&table.rows, "regression")?;
    let transformed = pp.transform(&table.rows)?;
    let (x, y) = xy(&transformed);
    let g = GenomeIr::parse(genome)?;
    let preprocessing = serde_json::to_vec_pretty(&pp)?;
    let feature_view = raw_feature_view(&csv_sha256(&transformed)?, &preprocessing)?;
    let fitted =
        crate::fit_boundary::fit(&g.canonical(), x, y, preprocessing, feature_view, output)?;
    let p = Predictor {
        schema: "parallel.predictor.v1".into(),
        task: "regression".into(),
        canonical: g.canonical(),
        training_sha256: file_hash(train)?,
        feature_count: table.width - 1,
        preprocessing: pp,
        model: fitted.model,
        fit_completion: Some(fitted.completion),
    };
    write_json(output, &p)
}
/// Fit a selected program directly from the immutable cut pointers. The
/// predictor stores the cut's training-only preprocessing for later raw inputs.
/// Without a declared root only a contained view (the cut's own tables) is
/// projected; a pointer view is refused, never resolved from the view's
/// ancestors (W1 route R2). The public `fit-view` verb is [`fit_view_in_root`].
pub fn fit_view(prepared: &Path, cut: usize, genome: &str, output: &Path) -> Result<()> {
    fit_view_with_root(None, prepared, cut, genome, output)
}

/// The public `fit-view` verb (W1 route R2): the cut's fitting rows are
/// projected once, from the preparation it read, under the declared `root`;
/// only that projection and the verified preparation state reach the fit
/// authority. No ancestor of the view is searched for its backing table.
pub fn fit_view_in_root(
    root: &Path,
    prepared: &Path,
    cut: usize,
    genome: &str,
    output: &Path,
) -> Result<()> {
    fit_view_with_root(Some(root), prepared, cut, genome, output)
}

/// The fitting-only projection of one cut (W1 route R2): under the declared
/// `root`, or with none only from a contained view's own tables. `p` is the
/// preparation the caller read; `prepared.json` is not reopened.
fn project_fitting_view(
    root: Option<&Path>,
    prepared: &Path,
    p: &Prepared,
    cut: usize,
) -> Result<Vec<Vec<f64>>> {
    match root {
        Some(root) => crate::prepare::load_bound_view_in_root(root, prepared, p, cut, true),
        None => crate::prepare::load_bound_contained_view(prepared, p, cut, true),
    }
}

fn fit_view_with_root(
    root: Option<&Path>,
    prepared: &Path,
    cut: usize,
    genome: &str,
    output: &Path,
) -> Result<()> {
    ensure!(!output.exists(), "fitted predictor already exists");
    let p: Prepared = read_json(&prepared.join("prepared.json"))?;
    ensure!(
        p.schema == "parallel.prepared.v1" && p.task == "regression",
        "regression preparation required"
    );
    let c = p
        .cuts
        .iter()
        .find(|c| c.index == cut)
        .context("unknown cut")?;
    let preprocessing_path = prepared.join(format!("cut-{cut}/preprocessing.json"));
    // Route R8: the stored state and the fitted feature view bind the same
    // verified bytes; the path is read once.
    let preprocessing_bytes = read_verified(
        &preprocessing_path,
        &c.preprocessing_sha256,
        "preprocessing",
    )
    .context("preprocessing changed")?;
    let table = project_fitting_view(root, prepared, &p, cut)?;
    fit_projected(
        &table,
        &c.training_sha256,
        preprocessing_bytes,
        genome,
        output,
    )
}

/// Fit exactly the given projected training rows (W1 follow-up 2c). Shared by
/// the in-process fitted-view fit and the sealed-memory export worker.
fn fit_projected(
    rows: &[Vec<f64>],
    training_sha256: &str,
    preprocessing_bytes: Vec<u8>,
    genome: &str,
    output: &Path,
) -> Result<()> {
    ensure!(
        csv_sha256(rows)? == training_sha256,
        "projected training rows differ from the cut's training identity"
    );
    let preprocessing: Preprocessing = serde_json::from_slice(&preprocessing_bytes)?;
    let (x, y) = xy(rows);
    let g = GenomeIr::parse(genome)?;
    let feature_view = raw_feature_view(training_sha256, &preprocessing_bytes)?;
    let fitted = crate::fit_boundary::fit(
        &g.canonical(),
        x,
        y,
        preprocessing_bytes,
        feature_view,
        output,
    )?;
    write_json(
        output,
        &Predictor {
            schema: "parallel.predictor.v1".into(),
            task: "regression".into(),
            canonical: g.canonical(),
            training_sha256: training_sha256.into(),
            feature_count: rows.first().context("empty projected training rows")?.len() - 1,
            preprocessing,
            model: fitted.model,
            fit_completion: Some(fitted.completion),
        },
    )
}

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
fn xy(table: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<f64>) {
    (
        table.iter().map(|r| r[..r.len() - 1].to_vec()).collect(),
        table.iter().map(|r| *r.last().unwrap()).collect(),
    )
}
pub fn predict(model: &Path, features: &Path, output: &Path) -> Result<()> {
    let p: Predictor = read_json(model)?;
    ensure!(
        p.schema == "parallel.predictor.v1" && p.task == "regression",
        "unsupported predictor schema/task"
    );
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path(features)?;
    let rows = reader
        .records()
        .map(|r| r.map(|r| r.iter().map(String::from).collect::<Vec<_>>()))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    ensure!(
        !rows.is_empty() && rows.iter().all(|r| r.len() == p.feature_count),
        "predict expects nonempty target-free feature rows"
    );
    let x = rows
        .iter()
        .map(|r| p.preprocessing.features(r))
        .collect::<Result<Vec<_>>>()?;
    let predictions = p.model.predict(&x)?;
    ensure!(
        predictions.len() == x.len() && predictions.iter().all(|p| p.is_finite()),
        "invalid fitted predictions"
    );
    let rows = predictions
        .into_iter()
        .map(|y| vec![y * p.preprocessing.target_scale + p.preprocessing.target_mean])
        .collect::<Vec<_>>();
    atomic_write(output, &csv_bytes(&rows)?)
}
pub fn metrics(actual: &[f64], predicted: &[f64]) -> Result<serde_json::Value> {
    let mut row = crate::task_contract::score_predictions("regression", actual, predicted, None)?;
    row["units"] = json!("original_target");
    Ok(row)
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelectionReceipt {
    pub schema: String,
    pub predictor_sha256: String,
    pub final_test_sha256: String,
    pub selection_evidence_path: std::path::PathBuf,
    pub selection_evidence_sha256: String,
    pub selected_on: String,
    pub frozen: bool,
}
/// Outer (sealed final-test) scoring (W1-T04). `receipt` must be a grant from
/// the separate outer release authority named by `JOPE_OUTER_RELEASE_AUTHORITY`;
/// the grant binds the authenticated fitted subject and the preparation's own
/// sealed-test identity. `output` receives the scorer's receipt, never the
/// metrics: they stay withheld until the authority releases them. A frozen
/// selection receipt alone is still validated, then refused before the test
/// path is opened, even for hashing.
pub fn score(model: &Path, test: &Path, receipt: &Path, output: &Path) -> Result<()> {
    let document: serde_json::Value = read_json(receipt)?;
    if document["schema"] == outer_release::GRANT_SCHEMA {
        let authority = std::env::var_os(outer_release::AUTHORITY_ENV)
            .context("an outer release grant needs its release authority")?;
        outer_release::Authority::open(Path::new(&authority))?
            .score(receipt, model, test, output)?;
        return Ok(());
    }
    let r: SelectionReceipt = serde_json::from_value(document)?;
    ensure!(
        r.schema == "parallel.selection.v1" && r.frozen && r.selected_on == "meta_validation",
        "a frozen meta-validation selection receipt is required"
    );
    ensure!(
        file_hash(model)? == r.predictor_sha256
            && file_hash(&r.selection_evidence_path)? == r.selection_evidence_sha256,
        "selection evidence changed"
    );
    let evidence: serde_json::Value = read_json(&r.selection_evidence_path)?;
    ensure!(
        evidence["predictor_sha256"] == r.predictor_sha256
            && evidence["partition"] == "meta_validation",
        "selection evidence does not bind this predictor/meta-validation"
    );
    anyhow::bail!(
        "a selection receipt does not release sealed labels: an outer score needs a grant from \
         the separate outer release authority ({}); the test path was not opened",
        outer_release::AUTHORITY_ENV
    )
}

/// Isolated worker: the parent enforces wall time and kills/reaps the process.
pub fn evaluate(train: &Path, validation: &Path, genome: &Path, output: &Path) -> Result<()> {
    let tr = Table::numeric(train)?;
    let va = Table::numeric(validation)?;
    evaluate_tables(
        &tr,
        &va,
        genome,
        output,
        "regression",
        None,
        br#"{"schema":"parallel.identity_preprocessing.v1"}"#,
    )
}
pub fn evaluate_view(prepared: &Path, cut: usize, genome: &Path, output: &Path) -> Result<()> {
    let p: crate::prepare::Prepared = crate::manifest::read_json(&prepared.join("prepared.json"))?;
    crate::task_contract::require_task(&p.task)?;

    let tr = crate::prepare::load_view(prepared, cut, true)?;
    let va = crate::prepare::load_view(prepared, cut, false)?;
    let mase = if p.task == "timeseries" {
        Some(naive_mase_denominator(&tr)?)
    } else {
        None
    };
    // Route R8: the preprocessing bound into the evaluation is the verified state.
    let c = p
        .cuts
        .iter()
        .find(|c| c.index == cut)
        .context("unknown cut")?;
    let preprocessing = read_verified(
        &prepared.join(format!("cut-{cut}/preprocessing.json")),
        &c.preprocessing_sha256,
        "preprocessing",
    )?;
    evaluate_tables(&tr, &va, genome, output, &p.task, mase, &preprocessing)
}
/// Sealed anonymous RAM file, shared with isolated candidate processes. No cut
/// CSVs or numeric cut copies are written to persistent storage.
pub fn memory_view(tr: &[Vec<f64>], va: &[Vec<f64>]) -> Result<fs::File> {
    use std::io::Write;
    use std::os::fd::{AsRawFd, FromRawFd};
    let width = tr.first().context("empty training view")?.len();
    ensure!(
        !va.is_empty() && width >= 2 && tr.iter().chain(va).all(|r| r.len() == width),
        "invalid in-memory view shape"
    );
    let name = std::ffi::CString::new("jope-cut").unwrap();
    // SAFETY: `name` is a CString that outlives the call, so the pointer is NUL-terminated and
    // valid; MFD_ALLOW_SEALING is a valid flag.
    let fd = unsafe { libc::memfd_create(name.as_ptr(), libc::MFD_ALLOW_SEALING) };
    ensure!(fd >= 0, "memfd_create failed");
    // SAFETY: memfd_create returned this fresh, nonnegative descriptor and nothing else owns
    // it; File takes sole ownership and closes it once.
    let file = unsafe { fs::File::from_raw_fd(fd) };
    {
        let mut out = std::io::BufWriter::new(&file);
        for n in [tr.len(), va.len(), width] {
            out.write_all(&(n as u64).to_le_bytes())?;
        }
        for row in tr.iter().chain(va) {
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
        "sealing cut memory failed"
    );
    Ok(file)
}
pub fn evaluate_memory(fd: i32, genome: &Path, output: &Path) -> Result<()> {
    evaluate_prepared_memory(fd, genome, output, None)
}
pub fn evaluate_prepared_memory(
    fd: i32,
    genome: &Path,
    output: &Path,
    prepared: Option<(&Path, usize)>,
) -> Result<()> {
    use std::io::{Read, Seek, SeekFrom};
    use std::os::fd::FromRawFd;
    ensure!(fd >= 3, "invalid inherited memory descriptor");
    // SAFETY: dup takes a plain integer and never affects the inherited descriptor; a number
    // that is not open makes it return -1, which is checked below.
    let copy = unsafe { libc::dup(fd) };
    ensure!(copy >= 0, "cut memory descriptor missing");
    // SAFETY: dup just returned `copy` as a fresh nonnegative descriptor no one else owns; File
    // takes sole ownership and closes it once, leaving the inherited `fd` open.
    let mut file = unsafe { fs::File::from_raw_fd(copy) };
    // SAFETY: F_GET_SEALS takes no pointer and only reads the seals of `fd`; it does not take
    // ownership, and a failure returns -1, which is checked below.
    let seals = unsafe { libc::fcntl(fd, libc::F_GET_SEALS) };
    ensure!(
        seals >= 0 && seals & libc::F_SEAL_WRITE != 0,
        "cut memory must be immutable"
    );
    file.seek(SeekFrom::Start(0))?;
    let mut header = [0u8; 24];
    file.read_exact(&mut header)?;
    let dims = header
        .chunks_exact(8)
        .map(|b| u64::from_le_bytes(b.try_into().unwrap()) as usize)
        .collect::<Vec<_>>();
    ensure!(
        dims[0] > 0 && dims[1] > 0 && dims[0] + dims[1] <= ROW_LIMIT && dims[2] >= 2,
        "invalid memory view header"
    );
    let cells = (dims[0] + dims[1])
        .checked_mul(dims[2])
        .context("view dimensions overflow")?;
    ensure!(
        cells <= 256 * 1024 * 1024,
        "view exceeds 2 GiB worker input budget"
    );
    let mut bytes = vec![0u8; cells * 8];
    file.read_exact(&mut bytes)?;
    let rows = bytes
        .chunks_exact(dims[2] * 8)
        .map(|row| {
            row.chunks_exact(8)
                .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    drop(bytes);
    let preprocessing = if let Some((prepared, cut)) = prepared {
        let p: Prepared = read_json(&prepared.join("prepared.json"))?;
        crate::task_contract::require_task(&p.task)?;
        let c = p
            .cuts
            .iter()
            .find(|c| c.index == cut)
            .context("unknown in-memory cut")?;
        ensure!(
            csv_sha256(&rows[..dims[0]])? == c.training_sha256
                && csv_sha256(&rows[dims[0]..])? == c.validation_sha256,
            "sealed memory differs from prepared cut"
        );
        let bytes = fs::read(prepared.join(format!("cut-{cut}/preprocessing.json")))?;
        ensure!(
            hash(&bytes) == c.preprocessing_sha256,
            "sealed fit preprocessing changed"
        );
        bytes
    } else {
        br#"{"schema":"parallel.identity_preprocessing.v1"}"#.to_vec()
    };
    evaluate_tables(
        &rows[..dims[0]],
        &rows[dims[0]..],
        genome,
        output,
        "regression",
        None,
        &preprocessing,
    )
}

#[derive(Clone, Serialize, Deserialize)]
pub struct SeedCandidate {
    pub genome: String,
    pub origin: String,
    // Historical files called this policy_logprob, but it is a normalized
    // critic rank weight, never a behavior/selection propensity for replay.
    #[serde(
        default,
        rename = "rank_log_weight",
        alias = "policy_logprob",
        skip_serializing_if = "Option::is_none"
    )]
    pub policy_logprob: Option<f64>,
    /// W4-T07: identity of this slot's proposal-ledger record (the draw, original tokens,
    /// scores and parser result it came from). Absent in older seed files.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proposal_record_sha256: Option<String>,
}
/// W5-T05: environment naming the search controller that started this evaluator as the
/// leader of its own process group under a SIGTERM parent-death signal.
pub const EVALUATOR_GROUP_GUARD: &str = "JOPE_EVALUATOR_GROUP_GUARD";

/// W5-T05: in an evaluator a search controller started, kill the evaluator's whole process
/// group (the evaluator and every process it started) when that controller dies, even by
/// SIGKILL. The kernel delivers SIGTERM on the controller's death (`PR_SET_PDEATHSIG`); the
/// handler turns it into one async-signal-safe SIGKILL of the evaluator's own group. A no-op
/// outside a guarded evaluator; refused in one that does not lead its own process group, so it
/// never signals a group it does not own.
pub fn guard_evaluator_group() -> Result<()> {
    let Some(controller) = std::env::var_os(EVALUATOR_GROUP_GUARD) else {
        return Ok(());
    };
    let controller: libc::pid_t = controller
        .to_str()
        .and_then(|pid| pid.parse().ok())
        .context("evaluator group guard names no controller")?;
    extern "C" fn kill_group(_: libc::c_int) {
        // SAFETY: kill is async-signal-safe and takes only integers; pid 0 signals this
        // process's own group, which the guard verified it leads before installing this.
        unsafe {
            libc::kill(0, libc::SIGKILL);
        }
    }
    // SAFETY: all-zero is a valid sigaction (integers, a sigset, a nullable restorer); calls get
    // live locals or a null old action, and the handler only calls async-signal-safe kill.
    unsafe {
        ensure!(
            libc::getpgrp() == libc::getpid(),
            "guarded evaluator does not lead its own process group"
        );
        let mut action: libc::sigaction = std::mem::zeroed();
        action.sa_sigaction = kill_group as *const () as libc::sighandler_t;
        libc::sigemptyset(&mut action.sa_mask);
        ensure!(
            libc::sigaction(libc::SIGTERM, &action, std::ptr::null_mut()) == 0,
            "evaluator group guard not installed"
        );
        // The controller may have died before the handler existed.
        if libc::getppid() != controller {
            libc::kill(0, libc::SIGKILL);
        }
    }
    Ok(())
}

/// W5-T01: directory of a search output holding the exact seed population and vocabulary
/// bytes its request pins, read by [`lineage::verify_search_recorded`].
pub const SEARCH_INPUTS: &str = "search-inputs";
/// The frozen seed population inside [`SEARCH_INPUTS`].
pub const SEARCH_SEEDS: &str = "seeds.json";
/// The frozen vocabulary inside [`SEARCH_INPUTS`].
pub const SEARCH_VOCABULARY: &str = "vocabulary.json";

/// Freeze `bytes` at `path` once; a resumed search must find exactly the same bytes.
fn freeze_search_input(path: &Path, bytes: &[u8]) -> Result<()> {
    match fs::read(path) {
        Ok(frozen) => ensure!(
            frozen == bytes,
            "frozen search input {} differs from the request",
            path.display()
        ),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => atomic_write(path, bytes)?,
        Err(error) => return Err(error.into()),
    }
    Ok(())
}

#[derive(Clone, Serialize, Deserialize)]
pub struct SeedPopulation {
    pub schema: String,
    pub prepared_sha256: String,
    pub cut: usize,
    pub training_sha256: String,
    pub context_sha256: String,
    pub checkpoint_sha256: String,
    pub candidates: Vec<SeedCandidate>,
}

/// Softmax log-weights on ranked genomes, never seed-list order.
pub fn attach_policy_logprobs(candidates: &mut [SeedCandidate], scores: &[f64]) -> Result<()> {
    ensure!(candidates.len() == scores.len(), "policy logprob alignment");
    let logprobs = crate::eval_registry::logprob_from_scores(scores)?;
    for (candidate, logprob) in candidates.iter_mut().zip(logprobs) {
        candidate.policy_logprob = Some(logprob);
    }
    Ok(())
}

/// Clean-completion-v2: a stray result.json must not clear a failed/timeout worker.
pub fn take_clean_result(failure: &mut Option<String>, result_exists: bool) -> bool {
    if failure.is_some() {
        return false;
    }
    if !result_exists {
        *failure = Some("missing_result".into());
        return false;
    }
    true
}

fn initial_execution_order(
    population: Option<&SeedPopulation>,
    size: usize,
    teacher_first: bool,
) -> Vec<usize> {
    let mut order = (0..size).collect::<Vec<_>>();
    if teacher_first {
        if let Some(population) = population {
            order.sort_by_key(|&i| match population.candidates[i].origin.as_str() {
                "incumbent_cut_teacher" => 0,
                "retained_cut_teacher" => 1,
                _ => 2,
            });
        }
    }
    order
}

// Version every new search and attempt. Historical requests without this
// protocol remain retained and cannot be resumed under different random draws.
// v3 (W5-T02) keys every stream by evaluation slot and the draw number within
// that slot, so a refused proposal in one slot never shifts another slot's
// draws; v2 keyed them by the search-wide proposal counter.
const RANDOM_PROTOCOL: &str = "parallel.gp_random_streams.v3";

fn random_word(seed: u64, slot: u64, draw: u64, stream: &str) -> u64 {
    use sha2::{Digest, Sha256};
    let mut digest = Sha256::new();
    digest.update(RANDOM_PROTOCOL.as_bytes());
    digest.update(seed.to_le_bytes());
    digest.update(slot.to_le_bytes());
    digest.update(draw.to_le_bytes());
    digest.update(stream.as_bytes());
    u64::from_le_bytes(digest.finalize()[..8].try_into().unwrap())
}

fn unit_quantile(word: u64) -> f64 {
    // Exactly representable, half-open [0, 1); a seed itself is never a draw.
    (word >> 11) as f64 * (1.0 / 9007199254740992.0)
}

fn execution_accounting(observations: &[serde_json::Value]) -> serde_json::Value {
    let mut recorded_starts = 0;
    let mut reuse = 0;
    let mut unknown = 0;
    for row in observations {
        let metrics = &row["metrics"];
        match (
            metrics["physical_fit_starts"].as_u64(),
            metrics["evaluation_origin"].as_str(),
        ) {
            (Some(1), Some("execution")) => recorded_starts += 1,
            (Some(0), Some("reuse")) => reuse += 1,
            _ => unknown += 1,
        }
    }
    json!({"logical_slots":observations.len(),"recorded_physical_fit_starts":recorded_starts,
        "physical_fit_starts":if unknown == 0 {Some(recorded_starts)} else {None},
        "reused_completions":reuse,"unknown_start_outcomes":unknown,
        "failed_or_unavailable":observations.iter().filter(|row|
            !row["failure"].is_null() || !row["metrics"].is_object()).count()})
}

/// Prefer stronger survivors. Policy weights apply to *ranked* genomes, never
/// to seed-population order: those are different programs after the first evals.
fn policy_weighted_parent(
    keep: usize,
    random_draw: u64,
    supplied: Option<&SeedPopulation>,
    ranked_genomes: Option<&[String]>,
) -> usize {
    if keep == 0 {
        return 0;
    }
    if let (Some(population), Some(genomes)) = (supplied, ranked_genomes) {
        let mut by_genome = std::collections::BTreeMap::<&str, f64>::new();
        for candidate in &population.candidates {
            if let Some(logprob) = candidate.policy_logprob {
                by_genome.insert(candidate.genome.as_str(), logprob);
            }
        }
        // Offspring often have no policy score. A missing score is not an opinion
        // of zero: zero is the TOP of the log-probability range and would outweigh
        // every scored survivor. Impute the minimum observed score instead, so an
        // unscored offspring stays eligible at the weight of the least likely
        // scored survivor and steering survives past the first evolved child that
        // enters the elite pool. A non-finite score is corruption rather than
        // absence and still collapses the whole pool to the uniform draw.
        let mut observed = Vec::with_capacity(keep);
        let mut corrupt = false;
        for genome in genomes.iter().take(keep) {
            match by_genome.get(genome.as_str()).copied() {
                Some(value) if value.is_finite() => observed.push(Some(value)),
                Some(_) => {
                    corrupt = true;
                    break;
                }
                None => observed.push(None),
            }
        }
        let floor = observed
            .iter()
            .flatten()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let scores = if corrupt || observed.len() != keep || !floor.is_finite() {
            None
        } else {
            Some(
                observed
                    .iter()
                    .map(|value| value.unwrap_or(floor))
                    .collect::<Vec<_>>(),
            )
        };
        if let Some(scores) = scores.filter(|scores| scores.len() == keep) {
            let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mut acc = 0.0;
            let mut weights = Vec::with_capacity(keep);
            for (i, score) in scores.iter().enumerate() {
                let w = (score - maximum).exp().max(1e-6);
                acc += w;
                weights.push((i, acc));
            }
            if acc > 0.0 {
                let pick = unit_quantile(random_draw) * acc;
                return weights
                    .iter()
                    .find(|(_, c)| pick < *c)
                    .map(|(i, _)| *i)
                    .unwrap_or(keep - 1);
            }
        }
    }
    // Missing policy scores, or scores that cannot be aligned to ranked
    // survivors, use the independently seeded uniform parent draw.
    (random_draw % keep as u64) as usize
}

#[cfg(test)]
mod parent_selection_tests {
    use super::{policy_weighted_parent, SeedCandidate, SeedPopulation};

    fn population(logprobs: &[Option<f64>]) -> SeedPopulation {
        SeedPopulation {
            schema: "t".into(),
            prepared_sha256: String::new(),
            cut: 0,
            training_sha256: String::new(),
            context_sha256: String::new(),
            checkpoint_sha256: String::new(),
            candidates: logprobs
                .iter()
                .map(|p| SeedCandidate {
                    genome: "g".into(),
                    origin: "neural_policy".into(),
                    policy_logprob: *p,
                    proposal_record_sha256: None,
                })
                .collect(),
        }
    }

    #[test]
    fn unmatched_seeds_keep_uniform_parent() {
        assert_eq!(policy_weighted_parent(10, 7, None, None), 7);
        let pop = population(&[None; 10]);
        assert_eq!(policy_weighted_parent(10, 7, Some(&pop), None), 7);
    }

    #[test]
    fn historical_and_failed_slots_never_invent_physical_starts() {
        use serde_json::json;
        let original = json!({"metrics":{"physical_fit_starts":1,"evaluation_origin":"execution"}});
        let reused = json!({"metrics":{"physical_fit_starts":0,"evaluation_origin":"reuse"}});
        let known = super::execution_accounting(&[original.clone(), reused.clone()]);
        assert_eq!(known["logical_slots"], 2);
        assert_eq!(known["physical_fit_starts"], 1);
        assert_eq!(known["reused_completions"], 1);
        let mixed = super::execution_accounting(&[
            original,
            reused,
            json!({"failure":"lost_worker_reply"}),
            json!({"metrics":{"rmse":0.}}),
        ]);
        assert_eq!(mixed["logical_slots"], 4);
        assert_eq!(mixed["recorded_physical_fit_starts"], 1);
        assert!(mixed["physical_fit_starts"].is_null());
        assert_eq!(mixed["unknown_start_outcomes"], 2);
    }

    #[test]
    fn rank_weights_do_not_emit_behavior_probability_claims() {
        let old = serde_json::json!({"genome":"(pipe X (ridge))","origin":"fixture","policy_logprob":-2.5});
        let candidate: SeedCandidate = serde_json::from_value(old).unwrap();
        let saved = serde_json::to_value(&candidate).unwrap();
        assert_eq!(saved["rank_log_weight"], -2.5);
        assert!(saved.get("policy_logprob").is_none());
        let restored: SeedCandidate = serde_json::from_value(saved).unwrap();
        assert_eq!(restored.policy_logprob, candidate.policy_logprob);
        let duplicate = serde_json::json!({"genome":"g","origin":"fixture","rank_log_weight":-1.,"policy_logprob":-2.});
        assert!(serde_json::from_value::<SeedCandidate>(duplicate).is_err());
    }

    #[test]
    fn policy_scores_weight_ranked_genomes_not_seed_order() {
        let mut pop = population(&[Some(10.0), Some(0.0), Some(0.0)]);
        pop.candidates[0].genome = "a".into();
        pop.candidates[1].genome = "b".into();
        pop.candidates[2].genome = "c".into();
        let aligned = vec!["a".into(), "b".into(), "c".into()];
        assert_eq!(
            policy_weighted_parent(3, u64::MAX / 2, Some(&pop), Some(&aligned)),
            0
        );
        // Ranked order is not seed order. Weight the ranked genomes.
        let ranked = vec!["b".into(), "c".into(), "a".into()];
        assert_eq!(
            policy_weighted_parent(3, u64::MAX / 2, Some(&pop), Some(&ranked)),
            2
        );
        // Scores without a ranked alignment must not index the wrong array.
        assert_eq!(policy_weighted_parent(3, 7, Some(&pop), None), 7 % 3);
    }

    #[test]
    fn nonfinite_scores_fall_back_for_the_whole_pool() {
        let genomes: Vec<String> = vec!["a".into(), "b".into(), "offspring".into()];
        let mut pop = population(&[Some(-50.), Some(-49.), None]);
        for (candidate, genome) in pop.candidates.iter_mut().zip(&genomes) {
            candidate.genome = genome.clone();
        }
        // Corruption, not absence: refuse to steer on it.
        for bad in [Some(f64::NAN), Some(f64::INFINITY), Some(f64::NEG_INFINITY)] {
            pop.candidates[2].policy_logprob = bad;
            for attempt in 0..1000 {
                let draw = super::random_word(8129661450841087504, 0, attempt, "left_parent");
                assert_eq!(
                    policy_weighted_parent(3, draw, Some(&pop), Some(&genomes)),
                    policy_weighted_parent(3, draw, None, None)
                );
            }
        }
    }

    #[test]
    fn unscored_offspring_keep_steering_alive_at_the_floor() {
        // Regression test for the defect that silently disabled policy steering
        // after the first evolved child entered the elite pool: one unscored
        // genome used to collapse the entire survivor pool to the uniform draw.
        let genomes: Vec<String> = vec!["a".into(), "b".into(), "offspring".into()];
        let mut pop = population(&[Some(-1.), Some(-9.), None]);
        for (candidate, genome) in pop.candidates.iter_mut().zip(&genomes) {
            candidate.genome = genome.clone();
        }
        pop.candidates[2].policy_logprob = None;
        let mut counts = [0usize; 3];
        let mut differs_from_uniform = 0;
        for attempt in 0..5000 {
            let draw = super::random_word(8129661450841087504, 0, attempt, "left_parent");
            let parent = policy_weighted_parent(3, draw, Some(&pop), Some(&genomes));
            counts[parent] += 1;
            if parent != policy_weighted_parent(3, draw, None, None) {
                differs_from_uniform += 1;
            }
        }
        // Steering is live: the pool is no longer the uniform draw.
        assert!(differs_from_uniform > 0, "steering collapsed to uniform");
        // The strongly scored survivor dominates the weakly scored one.
        assert!(counts[0] > counts[1], "policy score ignored: {counts:?}");
        // The unscored offspring is imputed at the floor, so it is reachable but
        // never preferred over the best scored survivor.
        assert!(
            counts[0] > counts[2],
            "offspring outweighed the best parent"
        );
    }

    #[test]
    fn seeded_draws_cover_parents_and_operators_independently() {
        let mut counts = [[0usize; 3]; 10];
        let mut quantiles = std::collections::BTreeSet::new();
        let mut pop = population(&[Some(1000.); 10]);
        let genomes: Vec<String> = (0..10).map(|i| format!("g{i}")).collect();
        for (candidate, genome) in pop.candidates.iter_mut().zip(&genomes) {
            candidate.genome = genome.clone();
        }
        for attempt in 0..10000 {
            let draw = super::random_word(8129661450841087504, 0, attempt, "left_parent");
            assert_eq!(
                draw,
                super::random_word(8129661450841087504, 0, attempt, "left_parent")
            );
            assert_ne!(
                draw,
                super::random_word(8129661450841087504, 0, attempt, "right_parent")
            );
            let quantile = super::unit_quantile(draw);
            assert!((0.0..1.0).contains(&quantile));
            quantiles.insert(quantile.to_bits());
            let parent = policy_weighted_parent(10, draw, Some(&pop), Some(&genomes));
            let operator = super::random_word(8129661450841087504, 0, attempt, "operator") % 16;
            counts[parent][if operator < 2 {
                0
            } else if operator < 9 {
                1
            } else {
                2
            }] += 1;
        }
        assert_eq!(quantiles.len(), 10000);
        assert_eq!(super::unit_quantile(0), 0.);
        assert!(super::unit_quantile(u64::MAX) < 1.);
        // Wide deterministic bounds detect the former constant quantile and
        // parent/operator parity coupling without a flaky stochastic test.
        for row in counts {
            assert!((850..1150).contains(&row.iter().sum::<usize>()));
            assert!((80..175).contains(&row[0]));
            assert!((330..550).contains(&row[1]));
            assert!((330..550).contains(&row[2]));
        }
    }

    #[test]
    fn stray_result_does_not_clear_worker_failure() {
        let mut failure = Some("worker_exit:1".into());
        assert!(!super::take_clean_result(&mut failure, true));
        assert_eq!(failure.as_deref(), Some("worker_exit:1"));
        let mut ok = None;
        assert!(super::take_clean_result(&mut ok, true));
        assert!(ok.is_none());
        let mut missing = None;
        assert!(!super::take_clean_result(&mut missing, false));
        assert_eq!(missing.as_deref(), Some("missing_result"));
    }

    #[test]
    fn logprobs_follow_ranking_scores_not_seed_order() {
        let mut candidates = vec![
            SeedCandidate {
                genome: "worse".into(),
                origin: "neural_policy".into(),
                policy_logprob: None,
                proposal_record_sha256: None,
            },
            SeedCandidate {
                genome: "best".into(),
                origin: "neural_policy".into(),
                policy_logprob: None,
                proposal_record_sha256: None,
            },
        ];
        super::attach_policy_logprobs(&mut candidates, &[0.1, 2.0]).unwrap();
        assert!(candidates[1].policy_logprob.unwrap() > candidates[0].policy_logprob.unwrap());
    }
}
fn naive_mase_denominator(train: &[Vec<f64>]) -> Result<f64> {
    ensure!(
        train.len() >= 2,
        "timeseries MASE needs at least two training rows"
    );
    let mut abs = 0.0;
    for pair in train.windows(2) {
        abs += (pair[1].last().context("empty row")? - pair[0].last().context("empty row")?).abs();
    }
    let denom = abs / (train.len() - 1) as f64;
    ensure!(
        denom.is_finite() && denom > 0.0,
        "invalid naive MASE denominator"
    );
    Ok(denom)
}

fn evaluate_tables(
    tr: &[Vec<f64>],
    va: &[Vec<f64>],
    genome: &Path,
    output: &Path,
    task: &str,
    mase_denominator: Option<f64>,
    preprocessing: &[u8],
) -> Result<()> {
    crate::task_contract::require_task(task)?;
    tchless_threads();
    let (x, y) = xy(tr);
    let (vx, vy) = xy(va);
    let g = GenomeIr::parse(&fs::read_to_string(genome)?)?;
    let start = Instant::now();
    // The ordered training and scoring row digests are bound into the feature view, the
    // baseline evidence and the score identity. Each one formats the whole table as CSV and
    // hashes it, so compute each once (training here, where it was first needed and timed;
    // scoring after the timed fit/predict, as before) and reuse the strings.
    let training_rows_sha256 = csv_sha256(tr)?;
    let feature_view = raw_feature_view(&training_rows_sha256, preprocessing)?;
    let fitted = crate::fit_boundary::fit(
        &g.canonical(),
        x,
        y,
        preprocessing.to_vec(),
        feature_view,
        output,
    )?;
    let pred = fitted.model.predict(&vx)?;
    let mut m = if task == "regression" {
        let mut row = metrics(&vy, &pred)?;
        row["units"] = json!("cut_normalized_target");
        row
    } else {
        crate::task_contract::score_predictions(task, &vy, &pred, mase_denominator)?
    };
    m["task"] = json!(task);
    m["primary_metric"] = json!(crate::task_contract::primary_metric(task)?);
    m["seconds"] = json!(start.elapsed().as_secs_f64());
    m["semantic_sha256"] = json!(hash(&serde_json::to_vec(&pred)?));
    let scoring_rows_sha256 = csv_sha256(va)?;
    // Produce the baseline from the very same authorized cut arrays. This is
    // a new diagnostic; historical results without it stay unavailable.
    let training_mean = tr.iter().map(|row| row[row.len() - 1]).sum::<f64>() / tr.len() as f64;
    let baseline = metrics(&vy, &vec![training_mean; vy.len()]);
    if let Some(baseline) = baseline
        .ok()
        .filter(|row| row["rmse"].as_f64().is_some_and(f64::is_finite))
    {
        m["baseline_evidence"] = json!({"schema":"parallel.cut_baseline.v1",
            "rule":"training_target_mean_on_scoring_rows","units":"cut_normalized_target",
            "rmse":baseline["rmse"],"training_target_mean":training_mean,
            "training_rows":tr.len(),"scoring_rows":va.len(),
            "training_sha256":training_rows_sha256,"scoring_sha256":scoring_rows_sha256,
            "preprocessing_sha256":hash(preprocessing)});
    } else {
        m["baseline_status"] = json!("nonfinite baseline; ratios unavailable");
    }
    // W5-T03: fit and score identities are distinct. The fit key is the
    // authority's fitted identity (program, ordered training rows,
    // preprocessing, feature view, numerical contract, seed) and may be reused.
    // The score identity additionally binds the exact fitted subject and the
    // scored rows and protocol, so a reused fit is always rescored on its own
    // rows and a score never transfers to another subject or membership.
    let receipt = crate::eval_registry::verify_completion_receipt(&fs::read(
        fitted.completion["path"]
            .as_str()
            .context("fit completion path absent")?,
    )?)?;
    let score_identity = json!({"schema":"parallel.gp_score_identity.v1",
        "fit_key":receipt.fit_key,"model_sha256":receipt.model_sha256,
        "training_rows_sha256":training_rows_sha256,"scoring_rows_sha256":scoring_rows_sha256,
        "preprocessing_sha256":hash(preprocessing),"task":task,
        "primary_metric":m["primary_metric"],"metric_units":m["units"],
        "evaluator_protocol":crate::label_admission::PROTOCOL,
        "predictions_sha256":m["semantic_sha256"]});
    m["score_identity_sha256"] = json!(hash(&serde_json::to_vec(&score_identity)?));
    m["score_identity"] = score_identity;
    m["fit_completion"] = fitted.completion;
    m["fitted_model"] = fitted.model_reference;
    m["evaluation_origin"] = json!(if fitted.reused { "reuse" } else { "execution" });
    m["physical_fit_starts"] = json!(usize::from(!fitted.reused));
    write_json(output, &m)
}
fn tchless_threads() {
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(
            std::env::var("RAYON_NUM_THREADS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(4),
        )
        .build_global();
}

/// Read complete records without changing original evidence. A torn tail is an
/// unavailable outcome requiring reconciliation, never permission to retry.
pub fn read_journal(path: &Path) -> Result<Vec<serde_json::Value>> {
    let bytes = fs::read(path)?;
    let end = bytes.iter().rposition(|&b| b == b'\n').map_or(0, |i| i + 1);
    ensure!(
        end == bytes.len(),
        "incomplete GP journal; original bytes retained for reconciliation"
    );
    let records = bytes[..end]
        .split(|&b| b == b'\n')
        .filter(|s| !s.is_empty())
        .map(serde_json::from_slice)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(records)
}

/// A journaled success is resumable measured evidence only while its original
/// completion attests exactly that output. Model eviction is not examined here;
/// `selection_replay` authenticates the complete fitted evidence independently.
fn journaled_measurement_attested(root: &Path, row: &serde_json::Value) -> Result<()> {
    let completion = crate::label_admission::artifact(root, &row["clean_completion"])
        .context("journaled measurement lacks its original completion")?;
    ensure!(
        completion["schema"] == crate::label_admission::COMPLETION_SCHEMA
            && completion["outcome"] == "success"
            && completion["failure"].is_null()
            && completion["evaluation_identity_sha256"] == row["evaluation_identity_sha256"]
            && completion["program_sha256"] == row["program_sha256"]
            && crate::label_admission::artifact(root, &completion["output"])? == row["metrics"],
        "journaled measurement differs from its original completion"
    );
    Ok(())
}

/// W5-T05: one live searcher per output directory. The exclusive advisory
/// lease is held until the searcher returns and is released by the kernel if
/// it dies, so crash recovery needs no manual cleanup. The descriptor is
/// close-on-exec: evaluator children never hold it.
fn search_lease(output: &Path) -> Result<fs::File> {
    let lease = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(output.join("search.lease"))?;
    match lease.try_lock() {
        Ok(()) => Ok(lease),
        Err(fs::TryLockError::WouldBlock) => {
            anyhow::bail!(
                "search output is held by another live searcher; concurrent dispatch refused"
            )
        }
        Err(fs::TryLockError::Error(error)) => Err(error.into()),
    }
}

/// Whether the evaluator has exited, without reaping it: an unreaped leader
/// keeps its process-group id reserved while the group is drained.
fn evaluator_exited(pid: i32) -> Result<bool> {
    // SAFETY: on Linux every siginfo_t union member is composed of integer and
    // raw-pointer fields, so all-zero is valid; it also supplies si_pid=0 when
    // WNOHANG finds no exited child and waitid leaves the rest untouched.
    let mut info: libc::siginfo_t = unsafe { std::mem::zeroed() };
    // SAFETY: `info` is a live writable siginfo_t for waitid; WNOWAIT leaves
    // the child unreaped so its pid stays reserved.
    let status = unsafe {
        libc::waitid(
            libc::P_PID,
            pid as libc::id_t,
            &mut info,
            libc::WEXITED | libc::WNOHANG | libc::WNOWAIT,
        )
    };
    ensure!(
        status == 0,
        "evaluator wait failed: {}",
        std::io::Error::last_os_error()
    );
    // SAFETY: `info` is initialized by the valid zeroed value above and waitid;
    // si_pid reads the integer pid field for the SIGCHLD layout.
    Ok(unsafe { info.si_pid() } != 0)
}

/// Live (non-zombie) members of a process group other than its leader.
fn process_group_members(group: i32) -> usize {
    let Ok(entries) = fs::read_dir("/proc") else {
        return 0;
    };
    entries
        .flatten()
        .filter_map(|entry| entry.file_name().to_str()?.parse::<i32>().ok())
        .filter(|&pid| pid != group)
        .filter(|pid| {
            let Ok(stat) = fs::read_to_string(format!("/proc/{pid}/stat")) else {
                return false;
            };
            // Fields after the parenthesised command: state, ppid, pgrp.
            let Some((_, rest)) = stat.rsplit_once(')') else {
                return false;
            };
            let mut fields = rest.split_whitespace();
            let state = fields.next();
            let pgrp = fields.nth(1).and_then(|g| g.parse::<i32>().ok());
            pgrp == Some(group) && state != Some("Z")
        })
        .count()
}

/// Kill the evaluator's whole process group and census what remains. Called
/// while the leader is unreaped (running on expiry, a zombie after exit).
fn drain_process_group(group: i32) -> serde_json::Value {
    let members_at_exit = process_group_members(group);
    // SAFETY: kill takes only integers and has no memory-safety preconditions; the caller holds
    // the unreaped leader, so `group` still names the evaluator's own process group.
    unsafe {
        libc::kill(-group, libc::SIGKILL);
    }
    let settle = Instant::now();
    let mut remaining = process_group_members(group);
    while remaining > 0 && settle.elapsed() < Duration::from_secs(2) {
        std::thread::sleep(Duration::from_millis(5));
        remaining = process_group_members(group);
    }
    json!({"isolated":true,"members_at_exit":members_at_exit,"members_after_drain":remaining})
}

fn experiment_failpoint(name: &str, isolated: bool) {
    // Only explicitly isolated experiments can exercise crash recovery gates.
    if isolated && std::env::var("JOPE_EXPERIMENT_CRASH_AFTER").ok().as_deref() == Some(name) {
        // SAFETY: raise has no memory-safety preconditions; SIGKILL ends this process at once,
        // which is the failpoint's purpose.
        unsafe {
            libc::raise(libc::SIGKILL);
        }
    }
}

pub fn search(
    prepared: &Path,
    cut: usize,
    output: &Path,
    evaluations: usize,
    seconds: u64,
    qualification: bool,
) -> Result<()> {
    search_seeded(
        prepared,
        cut,
        output,
        crate::gp::SearchOptions {
            evaluations,
            seconds,
            qualification,
            seeds: None,
            vocabulary: None,
        },
    )
}
/// A comparison search may write only below its explicitly bound experiment.
pub fn verify_experiment_output(output: &Path, experiment: &Path) -> Result<()> {
    let experiment = experiment.canonicalize()?;
    let manifest: serde_json::Value = read_json(&experiment.join("manifest.json"))?;
    ensure!(
        manifest["schema"] == "parallel.seed_experiment.v1"
            || manifest["schema"] == "parallel.seed_experiment.v2",
        "experiment manifest absent or unsupported"
    );
    ensure!(
        output.is_absolute()
            && output.starts_with(&experiment)
            && !output
                .components()
                .any(|c| matches!(c, std::path::Component::ParentDir)),
        "experiment output must remain inside its directory"
    );
    let ancestor = output
        .ancestors()
        .find(|p| p.exists())
        .context("output ancestor absent")?;
    for path in output
        .ancestors()
        .take_while(|p| p.starts_with(&experiment))
    {
        if let Ok(metadata) = fs::symlink_metadata(path) {
            ensure!(
                !metadata.file_type().is_symlink(),
                "experiment output symlink rejected"
            );
        }
    }
    ensure!(
        ancestor.canonicalize()?.starts_with(&experiment),
        "experiment output symlink escapes its directory"
    );
    Ok(())
}

/// Work limits, qualification mode and optional proposal inputs for a search.
#[derive(Clone, Copy)]
pub struct SearchOptions<'a> {
    pub evaluations: usize,
    pub seconds: u64,
    pub qualification: bool,
    pub seeds: Option<&'a Path>,
    pub vocabulary: Option<&'a Path>,
}

pub fn search_seeded(
    prepared: &Path,
    cut: usize,
    output: &Path,
    options: SearchOptions<'_>,
) -> Result<()> {
    search_seeded_with_rng(prepared, cut, output, options, SEED)
}

pub fn search_seeded_with_rng(
    prepared: &Path,
    cut: usize,
    output: &Path,
    options: SearchOptions<'_>,
    rng_seed: u64,
) -> Result<()> {
    let artifact_root = crate::prepare::workspace_root(prepared)?;
    search_seeded_with_rng_in_root(&artifact_root, prepared, cut, output, options, rng_seed)
}

/// Resolve the declared custody root before accepting any search or creating
/// candidate intents. CLI callers must pass their --root explicitly; a copied
/// preparation need not contain a Cargo workspace just to retain its receipts.
pub fn search_seeded_with_rng_in_root(
    artifact_root: &Path,
    prepared: &Path,
    cut: usize,
    output: &Path,
    options: SearchOptions<'_>,
    rng_seed: u64,
) -> Result<()> {
    search_inner(
        artifact_root,
        prepared,
        cut,
        output,
        options,
        rng_seed,
        None,
    )
}

/// Called only after native worker and fit authority validate the immutable genome task.
pub(crate) fn search_native_genome(
    root: &Path,
    reference: &crate::native_graph::Artifact,
    task: &crate::native_genome::GenomeTask,
    candidate: &crate::native_genome::Candidate,
    parent_row: Option<&serde_json::Value>,
) -> Result<()> {
    task.validate()?;
    search_inner(
        root,
        &resolve(root, &candidate.context.prepared)?,
        candidate.context.cut,
        &resolve(root, &task.output)?,
        crate::gp::SearchOptions {
            evaluations: task.evaluations,
            seconds: task.seconds,
            qualification: task.qualification,
            seeds: None,
            vocabulary: Some(&resolve(root, &task.vocabulary.path)?),
        },
        task.rng_seed,
        Some((reference, task, candidate, parent_row)),
    )
}

/// Metadata only: exact legacy GP initial population before fit admission or numerical work.
pub fn metadata_initial_population(rng_seed: u64) -> Result<Vec<String>> {
    let mut population = vec!["(pipe X (ridge))".to_owned()];
    let mut seen = population.iter().cloned().collect();
    fill_initial_population(&mut population, &mut seen, rng_seed, None)?;
    Ok(population)
}
/// Appends adapted bootstrap programs and returns, for each appended slot, its
/// original generator draw and generator seed. The search executes the single
/// adaptation of that original; it never adapts an already adapted program.
fn fill_initial_population(
    population: &mut Vec<String>,
    seen: &mut BTreeSet<String>,
    rng_seed: u64,
    vocab: Option<&crate::grammar::Vocab>,
) -> Result<Vec<(String, u64)>> {
    let mut seed = rng_seed;
    let mut originals = Vec::new();
    while population.len() < 80 {
        let generator_seed = seed;
        let raw = feat_math::evolve::random_canonical_pipeline(seed, 6);
        seed = seed.wrapping_add(1);
        let g = if let Some(v) = vocab {
            match v.project_proposal(&raw) {
                Ok(g) => g,
                Err(_) => continue,
            }
        } else {
            raw.clone()
        };
        ensure!(
            seed.wrapping_sub(rng_seed) < 100000,
            "unable to construct diverse grammar population"
        );
        if seen.insert(g.clone()) {
            population.push(g);
            originals.push((raw, generator_seed));
        }
    }
    Ok(originals)
}

fn search_inner(
    artifact_root: &Path,
    prepared: &Path,
    cut: usize,
    output: &Path,
    options: SearchOptions<'_>,
    rng_seed: u64,
    native: Option<(
        &crate::native_graph::Artifact,
        &crate::native_genome::GenomeTask,
        &crate::native_genome::Candidate,
        Option<&serde_json::Value>,
    )>,
) -> Result<()> {
    let SearchOptions {
        evaluations,
        seconds,
        qualification,
        seeds,
        vocabulary,
    } = options;

    let lifecycle_start = Instant::now();
    let completion_root = artifact_root
        .canonicalize()
        .context("search artifact root unavailable")?;
    ensure!(
        completion_root.is_dir(),
        "search artifact root must be a directory"
    );
    let prepared_path = search_artifact_path(&completion_root, prepared)?;
    let output_path = search_artifact_path(&completion_root, output)?;
    let authority_path = std::env::var_os("JOPE_FIT_AUTHORITY")
        .map(|path| search_artifact_path(&completion_root, Path::new(&path)))
        .transpose()
        .context("fit authority artifacts must share the declared search root")?;
    // A neural search process may rank candidates, but CPU fits use the
    // explicitly bound authority evaluator. Authority still verifies its loaded
    // executable and mappings before releasing a start token.
    let executable = cpu_evaluator_binding(std::env::var_os("JOPE_CPU_EVALUATOR"))?;
    let prepared = prepared_path.as_path();
    let output = output_path.as_path();
    let experiment = std::env::var_os("JOPE_EXPERIMENT_DIR").map(std::path::PathBuf::from);
    if let Some(directory) = &experiment {
        verify_experiment_output(output, directory)?;
        ensure!(
            qualification && evaluations <= 200 && seconds <= 180,
            "isolated comparison requires qualification and 200-evaluation / 180-second limits"
        );
    }
    let p: Prepared = read_json(&prepared.join("prepared.json"))?;
    crate::task_contract::require_task(&p.task)?;
    let native_overlay = if let Some((_, task, candidate, _)) = native {
        if let Some(overlay) = &task.preparation_overlay {
            ensure!(!qualification, "overlay cannot admit qualification search");
            crate::preparation_overlay::verify(&completion_root, overlay, candidate, &p)?;
            true
        } else {
            false
        }
    } else {
        false
    };
    ensure!(
        p.schema == "parallel.prepared.v1"
            && (p.clean_evaluation || (qualification && p.qualification_only && p.clean_status == "raw_training_reconstructed_qualification_only"))
            && (p.family_partition_frozen || qualification || native_overlay),
        "reconstructed cut preparation is required; limited evaluation needs explicit qualification admission"
    );
    // W1-T04: a search is task-level development search. Its observations are
    // teacher/learning inputs, so a sealed meta_test family never reaches it,
    // including under qualification admission; nothing has been loaded yet.
    ensure!(
        p.meta_partition != "meta_test",
        "sealed meta_test families are not released to search, including qualification"
    );
    ensure!(
        p.meta_partition == "meta_train" || qualification,
        "bootstrap/search training observations require meta-training families"
    );
    // Signed input verification is lifecycle overhead, outside the numerical budget.
    let start = if native.is_some() {
        Instant::now()
    } else {
        lifecycle_start
    };
    let c = p
        .cuts
        .iter()
        .find(|c| c.index == cut)
        .context("unknown cut")?;
    let dir = prepared.join(format!("cut-{cut}"));
    let training_view = crate::prepare::load_view_in_root(&completion_root, prepared, cut, true)?;
    let validation_view =
        crate::prepare::load_view_in_root(&completion_root, prepared, cut, false)?;
    // W5-T04: the cut's declared shape for the pre-dispatch feasibility
    // diagnostic (rows fitted or applied, raw feature width).
    let feasibility_shape = feat_math::feasibility::Shape {
        rows: training_view.len().max(validation_view.len()) as u64,
        cols: training_view
            .first()
            .map_or(0, |row| row.len().saturating_sub(1)) as u64,
    };
    let memory = memory_view(&training_view, &validation_view)?;
    drop((training_view, validation_view));
    ensure!(
        evaluations > 0 && evaluations <= 1000 && seconds > 0 && seconds <= 600,
        "pilot budget exceeds 1000 evaluations or 600 seconds"
    );
    fs::create_dir_all(output)?;
    // W5-T05: reserve the output before reading any resume state, so two
    // searchers can never dispatch the same reservation or both publish.
    let _lease = search_lease(output)?;
    ensure!(
        !output.join("search.json").exists(),
        "search is already complete"
    );
    let resuming = output.join("progress.json").exists();
    // Route R8: parse exactly the verified preprocessing bytes.
    let pp: Preprocessing = serde_json::from_slice(
        &read_verified(
            &dir.join("preprocessing.json"),
            &c.preprocessing_sha256,
            "preprocessing",
        )
        .context("preprocessing changed")?,
    )?;
    // W5-T01: the seed population and vocabulary are read once each; the search parses and
    // pins exactly those bytes, and freezes them beside its request (below).
    let read_input =
        |path: &Path| fs::read(path).with_context(|| format!("open {}", path.display()));
    let vocabulary_bytes = vocabulary.map(read_input).transpose()?;
    let seeds_bytes = seeds.map(read_input).transpose()?;
    let parse = |path: Option<&Path>| format!("parse {}", path.unwrap_or(Path::new("")).display());
    let vocab: Option<crate::grammar::Vocab> = vocabulary_bytes
        .as_deref()
        .map(serde_json::from_slice)
        .transpose()
        .with_context(|| parse(vocabulary))?;
    let vocabulary_sha256 = vocabulary_bytes.as_deref().map(hash);
    let supplied: Option<SeedPopulation> = seeds_bytes
        .as_deref()
        .map(serde_json::from_slice)
        .transpose()
        .with_context(|| parse(seeds))?;
    let seeds_sha256 = seeds_bytes.as_deref().map(hash);
    let neural_seeded = supplied
        .as_ref()
        .is_some_and(|s| s.candidates.iter().any(|c| c.origin == "neural_policy"));
    if let Some(s) = &supplied {
        ensure!(
            s.schema == "parallel.seed_population.v1"
                && s.prepared_sha256 == file_hash(&prepared.join("prepared.json"))?
                && s.cut == cut
                && s.training_sha256 == c.training_sha256
                && s.candidates.len() == 80,
            "neural population identity mismatch"
        );
        for candidate in &s.candidates {
            GenomeIr::parse(&candidate.genome)?;
        }
    }
    let mut population = supplied
        .as_ref()
        .map(|s| {
            s.candidates
                .iter()
                .map(|c| c.genome.clone())
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| vec!["(pipe X (ridge))".to_string()]);
    if let Some((_, task, candidate, _)) = native {
        // Direct means precisely one declared genotype. Continuation starts from
        // its original measured parent and spends its budget on new variation.
        population = match task.stage {
            crate::native_genome::Stage::Direct => vec![candidate.genome.clone()],
            crate::native_genome::Stage::Continuation => Vec::new(),
        };
    }
    let mut seen = BTreeSet::new();
    seen.extend(population.iter().cloned());
    ensure!(
        seen.len() == population.len(),
        "duplicate initial proposals"
    );
    // Original proposal and lineage source of every initial slot (W5-T01).
    // A critic weight estimates the original proposal, never its adaptation.
    let mut initial_sources: Vec<(String, serde_json::Value)> =
        if let Some((_, task, _, _)) = native {
            population
                .iter()
                .map(|genome| {
                    Ok((
                        genome.clone(),
                        json!({"kind":"native_candidate","slot":0,
                            "candidate":serde_json::to_value(&task.candidate)?}),
                    ))
                })
                .collect::<Result<_>>()?
        } else if let Some(s) = &supplied {
            s.candidates
                .iter()
                .enumerate()
                .map(|(slot, c)| {
                    let mut source = json!({"kind":"seed_population_slot","slot":slot,
                        "seeds_sha256":seeds_sha256,"origin":c.origin,
                        "rank_log_weight":c.policy_logprob,
                        "score_subject":"original_proposal"});
                    // W4-T07 join: the slot's proposal-ledger record, when the file names it.
                    if let Some(record) = &c.proposal_record_sha256 {
                        source["proposal_record_sha256"] = json!(record);
                    }
                    (c.genome.clone(), source)
                })
                .collect()
        } else {
            vec![(
                population[0].clone(),
                json!({"kind":"fixed_reference","slot":0}),
            )]
        };
    if native.is_none() {
        let first = initial_sources.len();
        let originals =
            fill_initial_population(&mut population, &mut seen, rng_seed, vocab.as_ref())?;
        initial_sources.extend(originals.into_iter().enumerate().map(
            |(offset, (raw, generator_seed))| {
                (
                    raw,
                    json!({"kind":"generic_bootstrap","slot":first + offset,
                        "generator":"feat_math.random_canonical_pipeline.depth6",
                        "generator_seed":generator_seed}),
                )
            },
        ));
    }
    ensure!(
        initial_sources.len() == population.len(),
        "initial proposal sources are incomplete"
    );
    let prepared_sha = file_hash(&prepared.join("prepared.json"))?;
    let evaluator_sha = file_hash(&executable)?;
    let champion_root = if p.storage.is_some() && experiment.is_none() {
        Some(completion_root.clone())
    } else {
        None
    };
    let request_path = output.join("search-request.json");
    let mut request = json!({"schema":"parallel.search_request.v1", "artifact_root":completion_root,
        "fit_authority":authority_path,
        "prepared":prepared.strip_prefix(&completion_root)?, "output":output.strip_prefix(&completion_root)?,
        "prepared_sha256":prepared_sha,"evaluator_sha256":evaluator_sha,"cut":cut,
        "training_sha256":c.training_sha256,"validation_sha256":c.validation_sha256,
        "preprocessing_sha256":c.preprocessing_sha256,"seeds_sha256":seeds_sha256,
        "vocabulary_sha256":vocabulary_sha256,"evaluations":evaluations,"seconds":seconds,
        "qualification":qualification,"rng_seed":rng_seed,"random_protocol":RANDOM_PROTOCOL,
        // W5-T02: the objective, evidence, tie and stop rules are frozen with
        // the request, so a resume under other rules is refused, not restarted.
        "selection_contract":selection_replay::CONTRACT,
        "proposal_cap_per_slot":selection_replay::PROPOSAL_CAP_PER_SLOT});
    if let Some((reference, task, _, _)) = native {
        request["native_genome_task"] = serde_json::to_value(reference)?;
        request["native_genome_stage"] = serde_json::to_value(task.stage)?;
    }
    // Check the recorded resume identity before comparing or recording the request.
    // Legacy outputs predate search-request.json: recording a refused request first
    // would make the refusal permanent and block the original identity's resume.
    let saved_progress: Option<serde_json::Value> = if resuming {
        let progress: serde_json::Value = read_json(&output.join("progress.json"))?;
        ensure!(
            progress["prepared_sha256"] == prepared_sha && progress["cut"] == cut,
            "resume identity mismatch"
        );
        ensure!(
            progress["seeds_sha256"] == json!(seeds_sha256),
            "resume neural seed identity mismatch"
        );
        ensure!(
            progress["rng_seed"].as_u64().unwrap_or(SEED) == rng_seed,
            "resume RNG seed mismatch"
        );
        Some(progress)
    } else {
        None
    };
    if request_path.exists() {
        ensure!(
            read_json::<serde_json::Value>(&request_path)? == request,
            "accepted search artifact root or request changed"
        );
    } else {
        write_json(&request_path, &request)?;
    }
    // W5-T01: the exact bytes the request pins, frozen beside it, so the lineage verifies
    // from the output alone (`lineage::verify_search_recorded`).
    for (name, bytes) in [
        (SEARCH_SEEDS, &seeds_bytes),
        (SEARCH_VOCABULARY, &vocabulary_bytes),
    ] {
        if let Some(bytes) = bytes {
            freeze_search_input(&output.join(SEARCH_INPUTS).join(name), bytes)?;
        }
    }
    let journal_path = output.join("observations.jsonl");
    let mut observations: Vec<serde_json::Value> = if journal_path.exists() {
        read_journal(&journal_path)?
    } else if resuming {
        read_json(&output.join("observations.json"))?
    } else {
        Vec::new()
    };
    ensure!(
        resuming || observations.is_empty(),
        "observation ledger has no resume identity"
    );
    // The journal is fsynced before progress.json, so it may contain the most
    // recent completed work after a crash. Never restore a smaller budget from
    // the older progress receipt. Old rows remain untouched: their per-row
    // runtimes cannot reconstruct exact cumulative search wall time.
    const SEARCH_CLOCK: &str = "parallel.gp_search_clock.v1";
    let mut last_completed_seconds = 0.0_f64;
    let mut legacy_clock_rows = 0usize;
    for observation in &observations {
        if observation["completed_search_seconds"].is_null() {
            legacy_clock_rows += 1;
            continue;
        }
        let completed = observation["completed_search_seconds"]
            .as_f64()
            .context("invalid completed search clock")?;
        ensure!(
            completed.is_finite() && completed >= last_completed_seconds,
            "completed search clock is negative or not monotonic"
        );
        ensure!(
            observation["search_clock"] == SEARCH_CLOCK,
            "unknown search clock definition"
        );
        last_completed_seconds = completed;
    }
    if !journal_path.exists() {
        let mut bytes = Vec::new();
        for observation in &observations {
            bytes.extend(serde_json::to_vec(observation)?);
            bytes.push(b'\n');
        }
        atomic_write(&journal_path, &bytes)?;
    }
    let mut journal = fs::OpenOptions::new().append(true).open(&journal_path)?;
    // (measured objective, executable program, ledger position; -1 = inherited).
    let mut ranked: Vec<(f64, String, i64)> = Vec::new();
    if let Some((_, task, candidate, Some(parent))) = native {
        ensure!(
            task.stage == crate::native_genome::Stage::Continuation,
            "direct native search cannot carry a parent score"
        );
        if parent["failure"].is_null() {
            let rmse = parent["metrics"]["rmse"]
                .as_f64()
                .context("original parent RMSE absent")?;
            ensure!(
                rmse.is_finite() && rmse >= 0.,
                "original parent score invalid"
            );
            ranked.push((rmse * pp.target_scale, candidate.genome.clone(), -1));
        }
    }
    let mut semantic = BTreeSet::new();
    let mut proposal_attempts = 0u64;
    let mut rejected_proposals = 0u64;
    // Existing journals retain their original slot order on recovery. New
    // searches protect known teachers from being starved by 64 slow neural fits.
    let teacher_first = saved_progress
        .as_ref()
        .is_none_or(|progress| progress["execution_order_protocol"] == "teacher_first_v1");
    let execution_order =
        initial_execution_order(supplied.as_ref(), population.len(), teacher_first);
    let execution_protocol = if teacher_first {
        "teacher_first_v1"
    } else {
        "legacy_seed_order_v1"
    };
    let previous_seconds = if let Some(progress) = &saved_progress {
        proposal_attempts = progress["proposal_attempts"]
            .as_u64()
            .unwrap_or(observations.len() as u64);
        rejected_proposals = progress["rejected_proposals"].as_u64().unwrap_or(0);
        let saved_seconds = progress["wall_seconds"]
            .as_f64()
            .context("resume elapsed time absent")?;
        ensure!(
            saved_seconds.is_finite() && saved_seconds >= 0.0,
            "invalid resume elapsed time"
        );
        saved_seconds.max(last_completed_seconds) + 30.0
    } else {
        0.0
    };
    if let Some(last) = observations.last() {
        proposal_attempts = last["proposal_attempts"]
            .as_u64()
            .unwrap_or(proposal_attempts);
        rejected_proposals = last["rejected_proposals"]
            .as_u64()
            .unwrap_or(rejected_proposals);
    }
    let clock_accuracy = if legacy_clock_rows > 0 {
        "conservative_resume_with_legacy_prefix"
    } else if resuming {
        "conservative_resume_offset"
    } else {
        "exact_active_elapsed"
    };
    let clock = json!({"schema":SEARCH_CLOCK,"definition":"monotonic active process elapsed from search entry through completed observation; excludes process downtime; resumed searches add a conservative 30-second allowance for unobserved interrupted work","accuracy":clock_accuracy,"resume_offset_seconds":previous_seconds,"resume_unobserved_allowance_seconds":if resuming {30.0} else {0.0},"legacy_observations_without_clock":legacy_clock_rows,"exact_wall_prefix_available":!resuming && legacy_clock_rows==0});
    let mut evaluated_programs = BTreeSet::new();
    if let Some((_, task, candidate, _)) = native {
        if task.stage == crate::native_genome::Stage::Continuation {
            evaluated_programs.insert(candidate.genome.clone());
        }
    }
    for (index, o) in observations.iter().enumerate() {
        ensure!(
            o["prepared_sha256"] == prepared_sha
                && o["cut"] == cut
                && o["evaluation_index"] == index
                && o["evaluator_sha256"] == evaluator_sha
                && o["training_sha256"] == c.training_sha256
                && o["validation_sha256"] == c.validation_sha256
                && o["seeds_sha256"] == json!(seeds_sha256)
                && o["vocabulary_sha256"] == json!(vocabulary_sha256)
                && o["rng_seed"].as_u64().unwrap_or(SEED) == rng_seed,
            "observation ledger identity mismatch"
        );
        evaluated_programs
            .insert(GenomeIr::parse(o["genome"].as_str().context("missing genome")?)?.canonical());
        if o["failure"].is_null() {
            let r = &o["metrics"];
            // Resumed rows obey the live admission rule and must still be the
            // exact output their original completion attests (W5-T06).
            let rmse = r["rmse"].as_f64().context("missing RMSE")? * pp.target_scale;
            ensure!(
                rmse.is_finite() && rmse >= 0.0,
                "journaled evaluator score invalid"
            );
            if !o["search_clock"].is_null() {
                journaled_measurement_attested(&completion_root, o)?;
            }
            if semantic.insert(
                r["semantic_sha256"]
                    .as_str()
                    .context("missing semantic hash")?
                    .to_string(),
            ) {
                ranked.push((
                    rmse,
                    o["genome"].as_str().context("missing genome")?.to_string(),
                    index as i64,
                ));
            }
        }
    }
    ranked.sort_by(|a, b| a.0.total_cmp(&b.0));
    ranked.truncate(80);
    if !resuming {
        write_json(&output.join("observations.json"), &observations)?;
    }
    // Persist the reserved resume allowance before doing more work, including
    // resumes that have already consumed their entire search budget.
    write_json(
        &output.join("progress.json"),
        &json!({"prepared_sha256":prepared_sha,"cut":cut,"seeds_sha256":seeds_sha256,"vocabulary_sha256":vocabulary_sha256,"execution_order_protocol":execution_protocol,"execution_order":execution_order,"wall_seconds":start.elapsed().as_secs_f64()+previous_seconds,"lifecycle_seconds":lifecycle_start.elapsed().as_secs_f64(),"search_clock":clock,"evaluations":observations.len(),"rng_seed":rng_seed,"proposal_attempts":proposal_attempts,"rejected_proposals":rejected_proposals}),
    )?;
    // W5-T05 terminal disposition, cross-checked by the independent replay.
    let mut stop_reason = "evaluation_budget";
    for i in observations.len()..evaluations {
        if start.elapsed().as_secs_f64() + previous_seconds >= seconds as f64 {
            stop_reason = "search_deadline";
            break;
        }
        // Projection can collapse distinct native mutations onto one legal
        // program. Reject that duplication before paying for a GP fit.
        let attempt_path = output.join(format!("candidate-{i}.attempt.json"));
        let saved_attempt: Option<serde_json::Value> = if attempt_path.exists() {
            let a: serde_json::Value = read_json(&attempt_path)?;
            ensure!(
                a["prepared_sha256"] == prepared_sha
                    && a["evaluator_sha256"] == evaluator_sha
                    && a["cut"] == cut
                    && a["evaluation_index"] == i
                    && a["seeds_sha256"] == json!(seeds_sha256)
                    && a["vocabulary_sha256"] == json!(vocabulary_sha256)
                    && a["evaluations"] == evaluations
                    && a["seconds"] == seconds
                    && a["rng_seed"] == rng_seed
                    && a["random_protocol"] == RANDOM_PROTOCOL,
                "incomplete candidate identity mismatch"
            );
            Some(a)
        } else {
            None
        };
        let slot = i as u64;
        let (raw_genome, genome, initial_proposal, lineage, slot_draw) = if let Some(a) =
            &saved_attempt
        {
            proposal_attempts = a["proposal_attempts"]
                .as_u64()
                .context("missing proposal counter")?;
            rejected_proposals = a["rejected_proposals"]
                .as_u64()
                .context("missing rejection counter")?;
            let slot_draw = a["slot_draw"].as_u64().context("missing slot draw")?;
            ensure!(
                a["random_streams"]
                    == json!({
                "operator":random_word(rng_seed,slot,slot_draw,"operator"),
                "left_parent":random_word(rng_seed,slot,slot_draw,"left_parent"),
                "right_parent":random_word(rng_seed,slot,slot_draw,"right_parent"),
                "variation":random_word(rng_seed,slot,slot_draw,"variation")}),
                "original candidate random streams changed"
            );
            let genome = a["genome"]
                .as_str()
                .context("missing candidate genome")?
                .to_string();
            ensure!(
                evaluated_programs.insert(GenomeIr::parse(&genome)?.canonical()),
                "pending candidate already committed"
            );
            (
                a["raw_genome"]
                    .as_str()
                    .context("missing raw genome")?
                    .to_string(),
                genome,
                a["initial_proposal"]
                    .as_bool()
                    .context("missing candidate origin")?,
                // Attempts recorded before the lineage envelope keep null.
                a.get("lineage").cloned().unwrap_or(serde_json::Value::Null),
                slot_draw,
            )
        } else {
            let mut selected = None;
            let mut out_of_time = false;
            for attempt in 0..selection_replay::PROPOSAL_CAP_PER_SLOT {
                if start.elapsed().as_secs_f64() + previous_seconds >= seconds as f64 {
                    out_of_time = true;
                    break;
                }
                proposal_attempts += 1;
                let initial = attempt == 0 && i < population.len();
                // Streams are keyed by (slot, draw within the slot), never by
                // the search-wide counter (W5-T02).
                let proposal_seed = random_word(rng_seed, slot, attempt, "variation");
                let operator = random_word(rng_seed, slot, attempt, "operator") % 16;
                // An explicit variation event: operator, parents and stream words.
                let variation = |kind: &str, parents: Vec<serde_json::Value>| {
                    json!({"kind":"variation","operator":kind,"parents":parents,
                        "slot":slot,"slot_draw":attempt,
                        "proposal_attempt":proposal_attempts,"variation_seed":proposal_seed,
                        "operator_word":operator})
                };
                let parent = |entry: &(f64, String, i64)| json!({"position":entry.2,"program_sha256":hash(entry.1.as_bytes())});
                let (raw, source) = if initial {
                    // The original proposal, never its already adapted form.
                    initial_sources[execution_order[i]].clone()
                } else {
                    let keep = ranked.len().min(10);
                    if keep == 0 && native.is_some() {
                        let (_, _, candidate, _) =
                            native.context("native continuation parent absent")?;
                        (
                            feat_math::evolve::mutate_canonical_pipeline(
                                &candidate.genome,
                                proposal_seed,
                                6,
                            )
                            .map_err(anyhow::Error::msg)?,
                            variation(
                                "continuation_mutate",
                                vec![json!({"position":null,"source":"native_candidate",
                                    "program_sha256":hash(candidate.genome.as_bytes())})],
                            ),
                        )
                    } else if keep == 0 || operator < 2 {
                        (
                            feat_math::evolve::random_canonical_pipeline(proposal_seed, 6),
                            variation("fresh_random", Vec::new()),
                        )
                    } else {
                        let ranked_genomes: Vec<String> = ranked
                            .iter()
                            .take(keep)
                            .map(|(_, genome, _)| genome.clone())
                            .collect();
                        let left = &ranked[policy_weighted_parent(
                            keep,
                            random_word(rng_seed, slot, attempt, "left_parent"),
                            supplied.as_ref(),
                            Some(ranked_genomes.as_slice()),
                        )];
                        if operator < 9 {
                            (
                                feat_math::evolve::mutate_canonical_pipeline(
                                    &left.1,
                                    proposal_seed,
                                    6,
                                )
                                .map_err(anyhow::Error::msg)?,
                                variation("mutate", vec![parent(left)]),
                            )
                        } else {
                            let right =
                                &ranked[(random_word(rng_seed, slot, attempt, "right_parent")
                                    % keep as u64)
                                    as usize];
                            (
                                feat_math::evolve::crossover_canonical_pipelines(
                                    &left.1,
                                    &right.1,
                                    proposal_seed,
                                    6,
                                )
                                .map_err(anyhow::Error::msg)?,
                                variation("crossover", vec![parent(left), parent(right)]),
                            )
                        }
                    }
                };
                let projected = if let Some(v) = &vocab {
                    v.project_proposal(&raw)
                } else {
                    Ok(raw.clone())
                };
                if let Ok(genome) = projected {
                    let canonical = GenomeIr::parse(&genome)?.canonical();
                    if evaluated_programs.insert(canonical) {
                        // Original → one explicit adaptation → executable recipe.
                        let lineage = json!({"schema":"parallel.gp_candidate_lineage.v1",
                            "original":{"program":raw,"program_sha256":hash(raw.as_bytes())},
                            "source":source,
                            "adaptation":if vocab.is_some() {
                                json!({"kind":"vocabulary_projection","applications":1,
                                    "vocabulary_sha256":vocabulary_sha256,
                                    "input_sha256":hash(raw.as_bytes()),
                                    "output_sha256":hash(genome.as_bytes()),
                                    "changed":raw != genome})
                            } else {
                                json!({"kind":"none","applications":0})
                            },
                            "executable":{"program":genome,"program_sha256":hash(genome.as_bytes())}});
                        selected = Some((raw, genome, initial, lineage, attempt));
                        break;
                    }
                }
                rejected_proposals += 1;
            }
            let Some(selected) = selected else {
                stop_reason = if out_of_time {
                    "search_deadline"
                } else {
                    "proposal_space_exhausted"
                };
                break;
            };
            selected
        };
        let genome_path = output.join(format!("candidate-{i}.dsl"));
        let mut candidate_identity = json!({"program_sha256":hash(genome.as_bytes()),
            "family":p.family,"partition":p.meta_partition,"prepared_sha256":prepared_sha,"cut":cut,
            "training_sha256":c.training_sha256,"validation_sha256":c.validation_sha256,
            "evaluator_sha256":evaluator_sha,"evaluator_protocol":crate::label_admission::PROTOCOL,
            "preprocessing_sha256":c.preprocessing_sha256,"target_scale":pp.target_scale,
            "metric_units":"cut_normalized_target","vocabulary_sha256":vocabulary_sha256,
            "feature_view_sha256":crate::label_admission::raw_view(&c.training_sha256,&c.preprocessing_sha256),
            "evaluation_seed":0,"resource_protocol":crate::label_admission::RESOURCE});
        candidate_identity["evaluation_identity_sha256"] = json!(hash(&serde_json::to_vec(
            &crate::label_admission::identity(&candidate_identity)
        )?));
        // W5-T04 pre-dispatch diagnostic under identical rules for every arm.
        // It is an estimate recorded beside the observed outcome; it does not
        // change dispatch (a refusal policy needs its own protocol identity).
        let feasibility = if let Some(a) = &saved_attempt {
            a.get("feasibility")
                .cloned()
                .unwrap_or(serde_json::Value::Null)
        } else {
            serde_json::to_value(feat_math::feasibility::classify(
                &GenomeIr::parse(&genome)?,
                feasibility_shape,
                &feat_math::feasibility::Limits::default(),
            ))?
        };
        if saved_attempt.is_some() {
            ensure!(
                fs::read(&genome_path)? == genome.as_bytes(),
                "pending candidate program changed"
            );
        } else {
            atomic_write(&genome_path, genome.as_bytes())?;
            write_json(
                &attempt_path,
                &json!({"prepared_sha256":prepared_sha,"evaluator_sha256":evaluator_sha,
                "cut":cut,"evaluation_index":i,"seeds_sha256":seeds_sha256,"vocabulary_sha256":vocabulary_sha256,
                "evaluations":evaluations,"seconds":seconds,"raw_genome":raw_genome,"genome":genome,
                "program_sha256":candidate_identity["program_sha256"],"evaluation_identity_sha256":candidate_identity["evaluation_identity_sha256"],
                "initial_proposal":initial_proposal,"rng_seed":rng_seed,"random_protocol":RANDOM_PROTOCOL,
                "slot_draw":slot_draw,
                "random_streams":{"operator":random_word(rng_seed,slot,slot_draw,"operator"),
                    "left_parent":random_word(rng_seed,slot,slot_draw,"left_parent"),
                    "right_parent":random_word(rng_seed,slot,slot_draw,"right_parent"),
                    "variation":random_word(rng_seed,slot,slot_draw,"variation")},
                "proposal_attempts":proposal_attempts,"rejected_proposals":rejected_proposals,
                "lineage":lineage,"feasibility":feasibility}),
            )?;
        }
        let result_path = output.join(format!("candidate-{i}.json"));
        experiment_failpoint("candidate_intent", experiment.is_some());
        let completion_path = output.join(format!("candidate-{i}.completion.json"));
        let saved_completion: Option<serde_json::Value> =
            if saved_attempt.is_some() && completion_path.exists() {
                Some(read_json(&completion_path)?)
            } else {
                None
            };
        let recovered_metric = saved_completion.is_some();
        let begun = Instant::now();
        let deadline_seconds =
            (seconds as f64 - start.elapsed().as_secs_f64() - previous_seconds).clamp(0.001, 30.);
        let deadline_monotonic_ns =
            crate::fit_boundary::deadline_after(Duration::from_secs_f64(deadline_seconds))?;
        let mut failure = None;
        let mut exit_code = None;
        let mut exit_signal = None;
        let mut exit_observed = false;
        let mut deadline_exceeded = false;
        let mut process_group = serde_json::Value::Null;
        if let Some(completion) = &saved_completion {
            failure = completion["failure"].as_str().map(String::from);
        } else if saved_attempt.is_some() {
            // An intent/result is not an observed successful exit. It might
            // already have fitted; never dispatch this candidate again.
            failure = Some("unknown_original_execution_without_completion".into());
        } else {
            let log = fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(output.join(format!("candidate-{i}.log")))?;
            let mut command = evaluator_command(&executable);
            use std::os::fd::AsRawFd;
            command
                .args(["evaluate-memory", "--fd"])
                .arg(memory.as_raw_fd().to_string())
                .arg("--prepared")
                .arg(prepared)
                .arg("--cut")
                .arg(cut.to_string())
                .arg("--genome")
                .arg(&genome_path)
                .arg("--output")
                .arg(&result_path)
                .env(
                    "JOPE_FIT_DEADLINE_MONOTONIC_NS",
                    deadline_monotonic_ns.to_string(),
                )
                .env(
                    "OMP_NUM_THREADS",
                    std::env::var("OMP_NUM_THREADS").unwrap_or_else(|_| "1".into()),
                )
                .env("OPENBLAS_NUM_THREADS", "1")
                .env("MKL_NUM_THREADS", "1")
                .env(
                    "RAYON_NUM_THREADS",
                    std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "1".into()),
                )
                .stdout(Stdio::from(log.try_clone()?))
                .stderr(Stdio::from(log));
            use std::os::unix::process::CommandExt;
            let controller_pid = std::process::id();
            // W5-T05: the evaluator's guard kills its whole process group when this
            // controller dies, even by SIGKILL (`guard_evaluator_group`).
            command.env(EVALUATOR_GROUP_GUARD, controller_pid.to_string());
            // SAFETY: the hook runs between fork and exec and makes only async-signal-safe calls
            // (setpgid, prctl, getppid, setrlimit); its errors are errno codes, so nothing
            // allocates in the child.
            unsafe {
                command.pre_exec(move || {
                    // SIGTERM, not SIGKILL: the guard turns it into SIGKILL for the group;
                    // an evaluator without the guard still ends by its default action.
                    if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM) != 0 {
                        return Err(std::io::Error::last_os_error());
                    }
                    // The controller exited: refuse with an errno, which does not allocate
                    // (an io::Error carrying a message would box it in the forked child).
                    if libc::getppid() as u32 != controller_pid {
                        return Err(std::io::Error::from_raw_os_error(libc::ESRCH));
                    }
                    let limit = libc::rlimit {
                        rlim_cur: 8 * 1024 * 1024 * 1024,
                        rlim_max: 8 * 1024 * 1024 * 1024,
                    };
                    // W5-T05: the evaluator leads its own process group so that
                    // every process it starts is drained with it.
                    if libc::setrlimit(libc::RLIMIT_AS, &limit) != 0 || libc::setpgid(0, 0) != 0 {
                        return Err(std::io::Error::last_os_error());
                    }
                    Ok(())
                });
            }
            let mut child = command.spawn()?;
            let group = i32::try_from(child.id()).context("evaluator pid out of range")?;
            loop {
                let exited = evaluator_exited(group)?;
                let expired = !exited && begun.elapsed().as_secs_f64() >= deadline_seconds;
                if exited || expired {
                    // The leader is still unreaped (a zombie when it exited), so
                    // its group id cannot be reused while the group is drained.
                    process_group = drain_process_group(group);
                    let status = child.wait()?;
                    use std::os::unix::process::ExitStatusExt;
                    exit_observed = true;
                    exit_code = status.code();
                    exit_signal = status.signal();
                    deadline_exceeded = expired || begun.elapsed().as_secs_f64() > deadline_seconds;
                    if deadline_exceeded {
                        failure = Some("candidate_or_view_timeout".into());
                    } else if !status.success() {
                        failure = Some(format!("worker_exit:{status}"));
                    }
                    break;
                }
                std::thread::sleep(Duration::from_millis(2));
            }
        }
        let result: Option<serde_json::Value> =
            if take_clean_result(&mut failure, result_path.exists()) {
                Some(read_json(&result_path)?)
            } else {
                None
            };
        let measured_runtime = if let Some(completion) = &saved_completion {
            completion["runtime_seconds"].as_f64()
        } else if saved_attempt.is_some() {
            None
        } else {
            Some(begun.elapsed().as_secs_f64())
        };
        let completion_reference = if completion_path.exists() {
            Some(relative_artifact(&completion_root, &completion_path)?)
        } else if saved_attempt.is_none() {
            let mut completion = candidate_identity.clone();
            completion["schema"] = json!(crate::label_admission::COMPLETION_SCHEMA);
            completion["attempt"] = relative_artifact(&completion_root, &attempt_path)?;
            completion["exit_observed"] = json!(exit_observed);
            completion["exit_code"] = json!(exit_code);
            completion["signal"] = json!(exit_signal);
            completion["deadline_seconds"] = json!(deadline_seconds);
            completion["deadline_exceeded"] = json!(deadline_exceeded);
            completion["runtime_seconds"] = json!(measured_runtime);
            completion["failure"] = json!(failure);
            completion["process_group"] = process_group;
            let log_path = output.join(format!("candidate-{i}.log"));
            if log_path.exists() {
                fs::File::open(&log_path)?.sync_all()?;
                completion["worker_log"] = relative_artifact(&completion_root, &log_path)?;
            }
            completion["outcome"] = json!(if failure.is_none() {
                "success"
            } else if deadline_exceeded {
                "timeout"
            } else {
                "infrastructure_failure"
            });
            completion["output"] = if result.is_some() {
                relative_artifact(&completion_root, &result_path)?
            } else {
                serde_json::Value::Null
            };
            if let Some(r) = &result {
                completion["fit_completion"] =
                    relative_existing_reference(&completion_root, &r["fit_completion"])?;
                completion["model"] =
                    relative_existing_reference(&completion_root, &r["fitted_model"])?;
            } else {
                let terminal_path = result_path.with_extension("fit-outcome.json");
                if terminal_path.exists() {
                    let terminal: serde_json::Value = read_json(&terminal_path)?;
                    completion["fit_completion"] =
                        relative_existing_reference(&completion_root, &terminal["fit_completion"])?;
                    if !deadline_exceeded && terminal["outcome"] == "deterministic_failure" {
                        completion["outcome"] = json!("fit_failure");
                    }
                    completion["original_fit_outcome"] = terminal["outcome"].clone();
                }
            }
            write_json(&completion_path, &completion)?;
            experiment_failpoint("candidate_completion", experiment.is_some());
            Some(relative_artifact(&completion_root, &completion_path)?)
        } else {
            None
        };
        if recovered_metric {
            let mut recovered = candidate_identity.clone();
            recovered["clean_completion"] = json!(completion_reference);
            recovered["failure"] = json!(failure);
            recovered["metrics"] = json!(result);
            recovered["runtime_seconds"] = json!(measured_runtime);
            let completion = saved_completion
                .as_ref()
                .context("original completion absent")?;
            ensure!(
                completion["schema"] == crate::label_admission::COMPLETION_SCHEMA
                    && completion["attempt"] == relative_artifact(&completion_root, &attempt_path)?,
                "recovered completion attempt changed"
            );
            for key in [
                "evaluation_identity_sha256",
                "program_sha256",
                "evaluator_sha256",
            ] {
                ensure!(
                    completion[key] == candidate_identity[key],
                    "recovered completion identity changed: {key}"
                );
            }
            if completion["fit_completion"].is_object() {
                let native = crate::label_admission::artifact(
                    &completion_root,
                    &completion["fit_completion"],
                )?;
                recovered["numerical_runtime_seconds"] = native["elapsed_seconds"].clone();
                crate::label_admission::validate_completion(&completion_root, &recovered)?;
            } else {
                // A denied dispatch has no numerical receipt and cannot pass
                // quality admission. It can still recover its original failure.
                ensure!(
                    result.is_none()
                        && failure.as_ref().is_some_and(|s| !s.is_empty())
                        && (completion["outcome"] == "infrastructure_failure"
                            || completion["outcome"] == "timeout")
                        && completion["exit_observed"] == true
                        && (completion["exit_code"]
                            .as_i64()
                            .is_some_and(|code| code != 0)
                            || completion["signal"]
                                .as_i64()
                                .is_some_and(|signal| signal > 0)
                            || completion["deadline_exceeded"] == true),
                    "recovered failure lacks original non-success process evidence"
                );
                crate::label_admission::artifact_bytes(
                    &completion_root,
                    &completion["worker_log"],
                )?;
            }
        }
        if let Some(r) = &result {
            experiment_failpoint("candidate_metric", experiment.is_some());
            let rmse = r["rmse"].as_f64().context("invalid evaluator RMSE")? * pp.target_scale;
            ensure!(rmse.is_finite() && rmse >= 0.0, "invalid evaluator score");
            let signature = r["semantic_sha256"]
                .as_str()
                .context("missing prediction signature")?;
            if semantic.insert(signature.to_string()) {
                ranked.push((rmse, genome.clone(), i as i64));
                ranked.sort_by(|a, b| a.0.total_cmp(&b.0));
                ranked.truncate(80);
                if i + 1 == 200 {
                    let _ = write_json(
                        &output.join("rerank-200.json"),
                        &json!({"schema":"parallel.gp_rerank.v1","evaluation_index":200,
                            "survivors":ranked.len(),"order":"validation_rmse_then_policy",
                            "parent_checkpoint_used":supplied.as_ref().map(|s|&s.checkpoint_sha256)}),
                    );
                }
            }
        }
        use std::io::Read;
        let mut diagnostic = Vec::new();
        if let Ok(log) = fs::File::open(output.join(format!("candidate-{i}.log"))) {
            log.take(8192).read_to_end(&mut diagnostic)?;
        }
        let diagnostic = String::from_utf8_lossy(&diagnostic).to_string();
        let origin = if let Some((_, task, _, _)) = native {
            match task.stage {
                crate::native_genome::Stage::Direct => "native_genome_direct_v1",
                crate::native_genome::Stage::Continuation => {
                    "native_budgeted_seeded_continuation_v1"
                }
            }
        } else if initial_proposal {
            supplied
                .as_ref()
                .map(|s| s.candidates[execution_order[i]].origin.as_str())
                .unwrap_or("generic_bootstrap")
        } else {
            "gp_evolution"
        };
        let observation = json!({"schema":"parallel.gp_observation.v1","dataset":p.dataset,"family":p.family,"partition":p.meta_partition,"cut":cut,"prepared_sha256":prepared_sha,"training_sha256":c.training_sha256,"validation_sha256":c.validation_sha256,"genome":genome,"program_sha256":file_hash(&genome_path)?,"evaluator_sha256":evaluator_sha,"evaluation_bias":"adaptive_cut_heldout_teacher","family_partition_frozen":p.family_partition_frozen,"qualification":qualification,"independent_evaluation":false,"target_access":"cut heldout optimized repeatedly; original final test sealed","origin":origin,"initial_seed_slot":if initial_proposal{Some(execution_order[i])}else{None},"execution_order_protocol":execution_protocol,"seeds_sha256":seeds_sha256,"vocabulary_sha256":vocabulary_sha256,"proposal_before_projection":if raw_genome!=genome{Some(&raw_genome)}else{None},"evaluation_index":i,"rng_seed":rng_seed,"proposal_attempts":proposal_attempts,"rejected_proposals":rejected_proposals,"recovered_completed_artifact":recovered_metric,"runtime_seconds":measured_runtime,"completed_search_seconds":start.elapsed().as_secs_f64()+previous_seconds,"search_clock":SEARCH_CLOCK,"search_clock_accuracy":clock_accuracy,"failure":failure,"worker_diagnostic":diagnostic,"metrics":result,"target_scale":pp.target_scale});
        let mut observation = observation;
        if let Some((reference, task, candidate, _)) = native {
            observation["native_genome_task"] = serde_json::to_value(reference)?;
            observation["native_candidate"] = serde_json::to_value(&task.candidate)?;
            observation["comparison_arm"] = serde_json::to_value(candidate.arm)?;
            observation["continuation_parent"] = serde_json::to_value(&task.direct_parent)?;
        }
        observation["task"] = json!("regression");
        // The pre-dispatch lineage envelope, repeated so the ledger is self-contained.
        observation["lineage"] = lineage;
        // The estimate beside its observed outcome (full stage bounds stay in
        // the attempt): verdict and reasons only.
        observation["feasibility"] = json!({"protocol":feasibility["protocol"],
            "verdict":feasibility["verdict"],"reasons":feasibility["reasons"]});
        observation["clean_completion"] = json!(completion_reference);
        if let Some(reference) = &completion_reference {
            let completion = crate::label_admission::artifact(&completion_root, reference)?;
            if completion["fit_completion"].is_object() {
                let native = crate::label_admission::artifact(
                    &completion_root,
                    &completion["fit_completion"],
                )?;
                observation["numerical_runtime_seconds"] = native["elapsed_seconds"].clone();
                observation["fit_outcome"] = native["outcome"].clone();
            }
        }
        observation["evaluation_origin"] = result
            .as_ref()
            .map(|r| r["evaluation_origin"].clone())
            .unwrap_or(serde_json::Value::Null);
        observation["evaluator_protocol"] = json!(crate::label_admission::PROTOCOL);
        observation["resource_protocol"] = json!(crate::label_admission::RESOURCE);
        observation["preprocessing_sha256"] = json!(c.preprocessing_sha256);
        observation["metric_units"] = json!("cut_normalized_target");
        observation["evaluation_seed"] = json!(0);
        observation["feature_view_sha256"] = json!(crate::label_admission::raw_view(
            &c.training_sha256,
            &c.preprocessing_sha256
        ));
        observation["evaluation_identity_sha256"] = json!(hash(&serde_json::to_vec(
            &crate::label_admission::identity(&observation)
        )?));
        use std::io::Write;
        let mut line = serde_json::to_vec(&observation)?;
        line.push(b'\n');
        // A journaled observation is acknowledged work. Commit a strict winner
        // first so recovery cannot skip a journal row whose winner was lost.
        if let Some(root) = &champion_root {
            crate::champions::record(root, &observation, &output.join("observations.jsonl"))?;
        }
        journal.write_all(&line)?;
        journal.sync_data()?;
        experiment_failpoint("observation_journal", experiment.is_some());
        observations.push(observation);
        write_json(
            &output.join("progress.json"),
            &json!({"prepared_sha256":prepared_sha,"cut":cut,"seeds_sha256":seeds_sha256,"vocabulary_sha256":vocabulary_sha256,"execution_order_protocol":execution_protocol,"execution_order":execution_order,"wall_seconds":start.elapsed().as_secs_f64()+previous_seconds,"lifecycle_seconds":lifecycle_start.elapsed().as_secs_f64(),"search_clock":clock,"evaluations":observations.len(),"rng_seed":rng_seed,"proposal_attempts":proposal_attempts,"rejected_proposals":rejected_proposals,"best_validation_rmse":ranked.first().map(|x|x.0),"timestamp":crate::telemetry::now()}),
        )?;
        experiment_failpoint("observation_progress", experiment.is_some());
        // Original attempts, programs, outputs and completions are durable
        // admission/recovery evidence. Artifact eviction never permits refits.
    }
    write_json(&output.join("observations.json"), &observations)?;
    experiment_failpoint("observations_snapshot", experiment.is_some());
    drop(journal);
    // Retain the durable journal until search publication; restart must never lose committed work.
    let observations_sha256 = file_hash(&output.join("observations.json"))?;
    let successful_measurements = observations
        .iter()
        .filter(|row| row["failure"].is_null() && row["metrics"].is_object())
        .count();
    let search_request = relative_artifact(&completion_root, &request_path)?;
    write_json(
        &output.join("search.json"),
        &json!({"schema":"parallel.search.v1","search_request":search_request,"search_id":file_hash(&request_path)?,"random_protocol":RANDOM_PROTOCOL,"execution_accounting":execution_accounting(&observations),"terminal":true,"stop_reason":stop_reason,"successful_measurements":successful_measurements,"failed_or_unavailable_measurements":observations.len()-successful_measurements,"observations_sha256":observations_sha256,"completed_at":crate::telemetry::now(),"evaluator_sha256":evaluator_sha,"evaluator_protocol":crate::label_admission::PROTOCOL,"vocabulary_sha256":vocabulary_sha256,"prepared_sha256":prepared_sha,"cut":cut,"completed_evaluations":observations.len(),"wall_seconds":start.elapsed().as_secs_f64()+previous_seconds,"lifecycle_seconds":lifecycle_start.elapsed().as_secs_f64(),"search_clock":clock,"execution_order_protocol":execution_protocol,"execution_order":execution_order,"resumed":resuming,"rng_seed":rng_seed,"proposal_attempts":proposal_attempts,"rejected_proposals":rejected_proposals,"unique_programs":evaluated_programs.len().saturating_sub(usize::from(native.is_some_and(|(_, task, _, _)| task.stage == crate::native_genome::Stage::Continuation))),"inherited_parent_excluded":native.is_some_and(|(_, task, _, _)| task.stage == crate::native_genome::Stage::Continuation),"best_validation_rmse":ranked.first().map(|x|x.0),"best_program":ranked.first().map(|x|&x.1),"seed_population_supplied":supplied.is_some(),"neural_seeded":neural_seeded,"seeds_sha256":seeds_sha256,"vocabulary_sha256":vocabulary_sha256,"final_tests_opened":false}),
    )?;
    experiment_failpoint("search_summary", experiment.is_some());
    // Keep the original journal after publication for historical reconciliation.
    Ok(())
}
