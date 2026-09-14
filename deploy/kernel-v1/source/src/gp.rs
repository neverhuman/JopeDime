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
}
pub fn fit(train: &Path, genome: &str, output: &Path) -> Result<()> {
    ensure!(!output.exists(), "fitted predictor already exists");
    let table = Table::read(train, &serde_json::Value::Null)?;
    ensure!(table.rows.len() <= ROW_LIMIT, "fit row limit exceeded");
    let pp = Preprocessing::fit(&table.rows, "regression")?;
    let transformed = pp.transform(&table.rows)?;
    let (x, y) = xy(&transformed);
    let g = GenomeIr::parse(genome)?;
    let model = feat_math::fit(&g, &x, &y)?;
    let p = Predictor {
        schema: "parallel.predictor.v1".into(),
        task: "regression".into(),
        canonical: g.canonical(),
        training_sha256: file_hash(train)?,
        feature_count: table.width - 1,
        preprocessing: pp,
        model,
    };
    write_json(output, &p)
}
/// Fit a selected program directly from the immutable cut pointers. The
/// predictor stores the cut's training-only preprocessing for later raw inputs.
pub fn fit_view(prepared: &Path, cut: usize, genome: &str, output: &Path) -> Result<()> {
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
    ensure!(
        file_hash(&preprocessing_path)? == c.preprocessing_sha256,
        "preprocessing changed"
    );
    let preprocessing: Preprocessing = read_json(&preprocessing_path)?;
    let table = crate::prepare::load_view(prepared, cut, true)?;
    let (x, y) = xy(&table);
    let g = GenomeIr::parse(genome)?;
    let model = feat_math::fit(&g, &x, &y)?;
    write_json(
        output,
        &Predictor {
            schema: "parallel.predictor.v1".into(),
            task: "regression".into(),
            canonical: g.canonical(),
            training_sha256: c.training_sha256.clone(),
            feature_count: table[0].len() - 1,
            preprocessing,
            model,
        },
    )
}
/// Export a fitted predictor under the same CPU, memory and time limits as a
/// teacher candidate. A failed export must not strand the neural cycle.
pub fn fit_view_isolated(
    executable: &Path,
    prepared: &Path,
    cut: usize,
    genome: &str,
    output: &Path,
) -> Result<()> {
    use std::os::unix::process::CommandExt;
    ensure!(!output.exists(), "fitted predictor already exists");
    let executable_sha256 = file_hash(executable)?;
    let prepared_sha256 = file_hash(&prepared.join("prepared.json"))?;
    let program_sha256 = hash(genome.as_bytes());
    fs::create_dir_all(output.parent().context("predictor output parent absent")?)?;
    let log_path = output.with_extension("fit.log");
    let receipt_path = output.with_extension("fit.json");
    let log = fs::File::create(&log_path)?;
    let mut command = Command::new(executable);
    command
        .args(["fit-view", "--prepared"])
        .arg(prepared)
        .arg("--cut")
        .arg(cut.to_string())
        .arg("--genome")
        .arg(genome)
        .arg("--output")
        .arg(output)
        .env("RAYON_NUM_THREADS", "1")
        .env("OMP_NUM_THREADS", "1")
        .env("OPENBLAS_NUM_THREADS", "1")
        .env("MKL_NUM_THREADS", "1")
        .stdout(Stdio::from(log.try_clone()?))
        .stderr(Stdio::from(log));
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
    let started = Instant::now();
    let mut child = command.spawn()?;
    let result = (|| -> Result<()> {
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
        let p: Prepared = read_json(&prepared.join("prepared.json"))?;
        let c = p
            .cuts
            .iter()
            .find(|c| c.index == cut)
            .context("fitted cut absent")?;
        ensure!(
            fitted.training_sha256 == c.training_sha256,
            "fitted predictor cut changed"
        );
        Ok(())
    })();
    if child.try_wait()?.is_none() {
        unsafe {
            libc::kill(-(child.id() as i32), libc::SIGKILL);
        }
        let _ = child.wait();
    }
    write_json(
        &receipt_path,
        &json!({
            "schema":"parallel.fitted_view_export.v1", "passed":result.is_ok(),
            "executable_sha256":executable_sha256, "prepared_sha256":prepared_sha256,
            "cut":cut, "program_sha256":program_sha256,
            "predictor_sha256":if result.is_ok(){Some(file_hash(output)?)}else{None},
            "wall_seconds":started.elapsed().as_secs_f64(), "timeout_seconds":30,
            "memory_limit_gib":8, "threads":1, "cut_csv_copies":0,
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
    let predictions = p.model.predict(&x);
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
    ensure!(
        !actual.is_empty()
            && actual.len() == predicted.len()
            && actual.iter().chain(predicted).all(|x| x.is_finite()),
        "invalid metric inputs"
    );
    let n = actual.len() as f64;
    let mean = actual.iter().sum::<f64>() / n;
    let sse = actual
        .iter()
        .zip(predicted)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>();
    let sst = actual.iter().map(|a| (a - mean).powi(2)).sum::<f64>();
    Ok(
        json!({"rmse":(sse/n).sqrt(),"mae":actual.iter().zip(predicted).map(|(a,b)|(a-b).abs()).sum::<f64>()/n,"r2":if sst>0.0{Some(1.0-sse/sst)}else{None},"r2_status":if sst>0.0{"defined"}else{"undefined_constant_target"},"rows":actual.len(),"units":"original_target"}),
    )
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
pub fn score(model: &Path, test: &Path, receipt: &Path, output: &Path) -> Result<()> {
    // Validate selection before opening the final test path, including for hashes.
    let r: SelectionReceipt = read_json(receipt)?;
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
    ensure!(
        file_hash(test)? == r.final_test_sha256,
        "sealed test identity mismatch"
    );
    let p: Predictor = read_json(model)?;
    let t = Table::read(test, &serde_json::Value::Null)?;
    ensure!(
        p.schema == "parallel.predictor.v1"
            && p.task == "regression"
            && t.width == p.feature_count + 1,
        "predictor/test schema mismatch"
    );
    let x = t
        .rows
        .iter()
        .map(|r| p.preprocessing.features(&r[..r.len() - 1]))
        .collect::<Result<Vec<_>>>()?;
    let y = t
        .rows
        .iter()
        .map(|r| {
            r.last()
                .unwrap()
                .parse::<f64>()
                .context("invalid test target")
        })
        .collect::<Result<Vec<_>>>()?;
    let predictions = p
        .model
        .predict(&x)
        .iter()
        .map(|v| v * p.preprocessing.target_scale + p.preprocessing.target_mean)
        .collect::<Vec<_>>();
    let mut result = metrics(&y, &predictions)?;
    result["selection_sha256"] = json!(file_hash(receipt)?);
    write_json(output, &result)
}

/// Isolated worker: the parent enforces wall time and kills/reaps the process.
pub fn evaluate(train: &Path, validation: &Path, genome: &Path, output: &Path) -> Result<()> {
    let tr = Table::numeric(train)?;
    let va = Table::numeric(validation)?;
    evaluate_tables(&tr, &va, genome, output)
}
pub fn evaluate_view(prepared: &Path, cut: usize, genome: &Path, output: &Path) -> Result<()> {
    let tr = crate::prepare::load_view(prepared, cut, true)?;
    let va = crate::prepare::load_view(prepared, cut, false)?;
    evaluate_tables(&tr, &va, genome, output)
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
    let fd = unsafe { libc::memfd_create(name.as_ptr(), libc::MFD_ALLOW_SEALING) };
    ensure!(fd >= 0, "memfd_create failed");
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
    use std::io::{Read, Seek, SeekFrom};
    use std::os::fd::FromRawFd;
    ensure!(fd >= 3, "invalid inherited memory descriptor");
    let copy = unsafe { libc::dup(fd) };
    ensure!(copy >= 0, "cut memory descriptor missing");
    let mut file = unsafe { fs::File::from_raw_fd(copy) };
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
    evaluate_tables(&rows[..dims[0]], &rows[dims[0]..], genome, output)
}

#[derive(Clone, Serialize, Deserialize)]
pub struct SeedCandidate {
    pub genome: String,
    pub origin: String,
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

fn initial_execution_order(population: Option<&SeedPopulation>, size: usize, teacher_first: bool) -> Vec<usize> {
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
fn evaluate_tables(tr: &[Vec<f64>], va: &[Vec<f64>], genome: &Path, output: &Path) -> Result<()> {
    tchless_threads();
    let (x, y) = xy(&tr);
    let (vx, vy) = xy(&va);
    let g = GenomeIr::parse(&fs::read_to_string(genome)?)?;
    let start = Instant::now();
    let model = feat_math::fit(&g, &x, &y)?;
    let pred = model.predict(&vx);
    let mut m = metrics(&vy, &pred)?;
    m["units"] = json!("cut_normalized_target");
    m["seconds"] = json!(start.elapsed().as_secs_f64());
    m["semantic_sha256"] = json!(hash(&serde_json::to_vec(&pred)?));
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

/// Read only complete, fsynced records. A torn final write can be retried;
/// malformed complete lines are corruption and stop the job.
pub fn read_journal(path: &Path) -> Result<Vec<serde_json::Value>> {
    let bytes = fs::read(path)?;
    let end = bytes.iter().rposition(|&b| b == b'\n').map_or(0, |i| i + 1);
    let records = bytes[..end]
        .split(|&b| b == b'\n')
        .filter(|s| !s.is_empty())
        .map(serde_json::from_slice)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if end != bytes.len() {
        fs::OpenOptions::new()
            .write(true)
            .open(path)?
            .set_len(end as u64)?;
    }
    Ok(records)
}

fn experiment_failpoint(name: &str, isolated: bool) {
    // Only explicitly isolated experiments can exercise crash recovery gates.
    if isolated && std::env::var("JOPE_EXPERIMENT_CRASH_AFTER").ok().as_deref() == Some(name) {
        unsafe { libc::raise(libc::SIGKILL); }
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
        evaluations,
        seconds,
        qualification,
        None,
        None,
    )
}
/// A comparison search may write only below its explicitly bound experiment.
pub fn verify_experiment_output(output: &Path, experiment: &Path) -> Result<()> {
    let experiment = experiment.canonicalize()?;
    let manifest: serde_json::Value = read_json(&experiment.join("manifest.json"))?;
    ensure!(manifest["schema"] == "parallel.seed_experiment.v1" || manifest["schema"] == "parallel.seed_experiment.v2", "experiment manifest absent or unsupported");
    ensure!(output.is_absolute() && output.starts_with(&experiment)
        && !output.components().any(|c| matches!(c, std::path::Component::ParentDir)),
        "experiment output must remain inside its directory");
    let ancestor = output.ancestors().find(|p| p.exists()).context("output ancestor absent")?;
    for path in output.ancestors().take_while(|p| p.starts_with(&experiment)) {
        if let Ok(metadata) = fs::symlink_metadata(path) {
            ensure!(!metadata.file_type().is_symlink(), "experiment output symlink rejected");
        }
    }
    ensure!(ancestor.canonicalize()?.starts_with(&experiment), "experiment output symlink escapes its directory");
    Ok(())
}

pub fn search_seeded(
    prepared: &Path,
    cut: usize,
    output: &Path,
    evaluations: usize,
    seconds: u64,
    qualification: bool,
    seeds: Option<&Path>,
    vocabulary: Option<&Path>,
) -> Result<()> {
    search_seeded_with_rng(prepared, cut, output, evaluations, seconds, qualification, seeds, vocabulary, SEED)
}

pub fn search_seeded_with_rng(
    prepared: &Path,
    cut: usize,
    output: &Path,
    evaluations: usize,
    seconds: u64,
    qualification: bool,
    seeds: Option<&Path>,
    vocabulary: Option<&Path>,
    rng_seed: u64,
) -> Result<()> {
    let start = Instant::now();
    let experiment = std::env::var_os("JOPE_EXPERIMENT_DIR").map(std::path::PathBuf::from);
    if let Some(directory) = &experiment {
        verify_experiment_output(output, directory)?;
        ensure!(qualification && evaluations <= 200 && seconds <= 180,
            "isolated comparison requires qualification and 200-evaluation / 180-second limits");
    }
    let p: Prepared = read_json(&prepared.join("prepared.json"))?;
    ensure!(
        p.schema == "parallel.prepared.v1"
            && (p.clean_evaluation || (qualification && p.qualification_only && p.clean_status == "raw_training_reconstructed_qualification_only"))
            && (p.family_partition_frozen || qualification),
        "reconstructed cut preparation is required; limited evaluation needs explicit qualification admission"
    );
    ensure!(
        p.task == "regression" && (p.meta_partition == "meta_train" || qualification),
        "bootstrap/search training observations require meta-training regression families"
    );
    let c = p
        .cuts
        .iter()
        .find(|c| c.index == cut)
        .context("unknown cut")?;
    let dir = prepared.join(format!("cut-{cut}"));
    let memory = memory_view(
        &crate::prepare::load_view(prepared, cut, true)?,
        &crate::prepare::load_view(prepared, cut, false)?,
    )?;
    ensure!(
        evaluations > 0 && evaluations <= 1000 && seconds > 0 && seconds <= 600,
        "pilot budget exceeds 1000 evaluations or 600 seconds"
    );
    ensure!(
        !output.join("search.json").exists(),
        "search is already complete"
    );
    let resuming = output.join("progress.json").exists();
    fs::create_dir_all(output)?;
    let pp: Preprocessing = read_json(&dir.join("preprocessing.json"))?;
    ensure!(
        file_hash(&dir.join("preprocessing.json"))? == c.preprocessing_sha256,
        "preprocessing changed"
    );
    let vocab: Option<crate::grammar::Vocab> = vocabulary.map(read_json).transpose()?;
    let vocabulary_sha256 = vocabulary.map(file_hash).transpose()?;
    let supplied: Option<SeedPopulation> = seeds.map(read_json).transpose()?;
    let seeds_sha256 = seeds.map(file_hash).transpose()?;
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
    let mut seen = BTreeSet::new();
    seen.extend(population.iter().cloned());
    ensure!(
        seen.len() == population.len(),
        "duplicate initial proposals"
    );
    let mut seed = rng_seed;
    while population.len() < 80 {
        let raw = feat_math::evolve::random_canonical_pipeline(seed, 6);
        seed = seed.wrapping_add(1);
        let g = if let Some(v) = &vocab {
            match v.project_proposal(&raw) {
                Ok(g) => g,
                Err(_) => continue,
            }
        } else {
            raw
        };
        ensure!(
            seed.wrapping_sub(rng_seed) < 100000,
            "unable to construct diverse grammar population"
        );
        if seen.insert(g.clone()) {
            population.push(g);
        }
    }
    let executable = std::env::current_exe()?;
    let prepared_sha = file_hash(&prepared.join("prepared.json"))?;
    let evaluator_sha = file_hash(&executable)?;
    let champion_root = if p.storage.is_some() && experiment.is_none() { Some(crate::prepare::workspace_root(prepared)?) } else { None };
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
    let mut ranked: Vec<(f64, String)> = Vec::new();
    let mut semantic = BTreeSet::new();
    let mut proposal_attempts = 0u64;
    let mut rejected_proposals = 0u64;
    // Existing journals retain their original slot order on recovery. New
    // searches protect known teachers from being starved by 64 slow neural fits.
    let teacher_first = !resuming || read_json::<serde_json::Value>(&output.join("progress.json"))?["execution_order_protocol"] == "teacher_first_v1";
    let execution_order = initial_execution_order(supplied.as_ref(), population.len(), teacher_first);
    let execution_protocol = if teacher_first { "teacher_first_v1" } else { "legacy_seed_order_v1" };
    let previous_seconds = if resuming {
        let progress: serde_json::Value = read_json(&output.join("progress.json"))?;
        ensure!(
            progress["prepared_sha256"] == prepared_sha && progress["cut"] == cut,
            "resume identity mismatch"
        );
        ensure!(
            progress["seeds_sha256"] == json!(seeds_sha256),
            "resume neural seed identity mismatch"
        );
        ensure!(progress["rng_seed"].as_u64().unwrap_or(SEED) == rng_seed,
            "resume RNG seed mismatch");
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
        proposal_attempts = last["proposal_attempts"].as_u64().unwrap_or(proposal_attempts);
        rejected_proposals = last["rejected_proposals"].as_u64().unwrap_or(rejected_proposals);
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
            if semantic.insert(
                r["semantic_sha256"]
                    .as_str()
                    .context("missing semantic hash")?
                    .to_string(),
            ) {
                ranked.push((
                    r["rmse"].as_f64().context("missing RMSE")? * pp.target_scale,
                    o["genome"].as_str().context("missing genome")?.to_string(),
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
        &json!({"prepared_sha256":prepared_sha,"cut":cut,"seeds_sha256":seeds_sha256,"vocabulary_sha256":vocabulary_sha256,"execution_order_protocol":execution_protocol,"execution_order":execution_order,"wall_seconds":start.elapsed().as_secs_f64()+previous_seconds,"search_clock":clock,"evaluations":observations.len(),"rng_seed":rng_seed,"proposal_attempts":proposal_attempts,"rejected_proposals":rejected_proposals}),
    )?;
    for i in observations.len()..evaluations {
        if start.elapsed().as_secs_f64() + previous_seconds >= seconds as f64 {
            break;
        }
        // Projection can collapse distinct native mutations onto one legal
        // program. Reject that duplication before paying for a GP fit.
        let attempt_path = output.join(format!("candidate-{i}.attempt.json"));
        let saved_attempt: Option<serde_json::Value> = if attempt_path.exists() {
            let a: serde_json::Value = read_json(&attempt_path)?;
            ensure!(a["prepared_sha256"] == prepared_sha && a["evaluator_sha256"] == evaluator_sha
                && a["cut"] == cut && a["evaluation_index"] == i
                && a["seeds_sha256"] == json!(seeds_sha256) && a["vocabulary_sha256"] == json!(vocabulary_sha256)
                && a["evaluations"] == evaluations && a["seconds"] == seconds && a["rng_seed"] == rng_seed,
                "incomplete candidate identity mismatch");
            Some(a)
        } else { None };
        let (raw_genome, genome, initial_proposal) = if let Some(a) = &saved_attempt {
            proposal_attempts = a["proposal_attempts"].as_u64().context("missing proposal counter")?;
            rejected_proposals = a["rejected_proposals"].as_u64().context("missing rejection counter")?;
            let genome = a["genome"].as_str().context("missing candidate genome")?.to_string();
            ensure!(evaluated_programs.insert(GenomeIr::parse(&genome)?.canonical()), "pending candidate already committed");
            (a["raw_genome"].as_str().context("missing raw genome")?.to_string(), genome,
             a["initial_proposal"].as_bool().context("missing candidate origin")?)
        } else {
            let mut selected = None;
            for attempt in 0..10000 {
                if start.elapsed().as_secs_f64() + previous_seconds >= seconds as f64 {
                    break;
                }
                proposal_attempts += 1;
                let initial = attempt == 0 && i < population.len();
                let proposal_seed = rng_seed.wrapping_add(proposal_attempts);
                let raw = if initial {
                    population[execution_order[i]].clone()
                } else {
                    let keep = ranked.len().min(10);
                    if keep == 0 || proposal_attempts % 8 == 0 {
                        feat_math::evolve::random_canonical_pipeline(proposal_seed, 6)
                    } else {
                        let left = &ranked[proposal_attempts as usize % keep].1;
                        let child = if proposal_attempts % 2 == 0 {
                            feat_math::evolve::mutate_canonical_pipeline(left, proposal_seed, 6)
                        } else {
                            feat_math::evolve::crossover_canonical_pipelines(
                                left,
                                &ranked[(proposal_attempts as usize + 1) % keep].1,
                                proposal_seed,
                                6,
                            )
                        };
                        child.map_err(anyhow::Error::msg)?
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
                        selected = Some((raw, genome, initial));
                        break;
                    }
                }
                rejected_proposals += 1;
            }
            let Some(selected) = selected else {
                break;
            };
            selected
        };
        let genome_path = output.join(format!("candidate-{i}.dsl"));
        if saved_attempt.is_some() {
            ensure!(fs::read(&genome_path)? == genome.as_bytes(), "pending candidate program changed");
        } else {
            atomic_write(&genome_path, genome.as_bytes())?;
            write_json(&attempt_path, &json!({"prepared_sha256":prepared_sha,"evaluator_sha256":evaluator_sha,
                "cut":cut,"evaluation_index":i,"seeds_sha256":seeds_sha256,"vocabulary_sha256":vocabulary_sha256,
                "evaluations":evaluations,"seconds":seconds,"raw_genome":raw_genome,"genome":genome,
                "initial_proposal":initial_proposal,"rng_seed":rng_seed,"proposal_attempts":proposal_attempts,"rejected_proposals":rejected_proposals}))?;
        }
        let result_path = output.join(format!("candidate-{i}.json"));
        experiment_failpoint("candidate_intent", experiment.is_some());
        let recovered_metric = saved_attempt.is_some() && result_path.exists();
        let begun = Instant::now();
        let mut failure = None;
        if !recovered_metric {
        let log = fs::OpenOptions::new().create(true).append(true).open(output.join(format!("candidate-{i}.log")))?;
        let mut command = Command::new(&executable);
        use std::os::fd::AsRawFd;
        command
            .args(["evaluate-memory", "--fd"])
            .arg(memory.as_raw_fd().to_string())
            .arg("--genome")
            .arg(&genome_path)
            .arg("--output")
            .arg(&result_path)
            .env(
                "OMP_NUM_THREADS",
                std::env::var("OMP_NUM_THREADS").unwrap_or_else(|_| "4".into()),
            )
            .env("OPENBLAS_NUM_THREADS", "1")
            .env("MKL_NUM_THREADS", "1")
            .env(
                "RAYON_NUM_THREADS",
                std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "4".into()),
            )
            .stdout(Stdio::from(log.try_clone()?))
            .stderr(Stdio::from(log));
        use std::os::unix::process::CommandExt;
        let controller_pid = std::process::id();
        unsafe {
            command.pre_exec(move || {
                if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) != 0 || libc::getppid() as u32 != controller_pid {
                    return Err(std::io::Error::other("evaluator lost controller"));
                }
                let limit = libc::rlimit {
                    rlim_cur: 8 * 1024 * 1024 * 1024,
                    rlim_max: 8 * 1024 * 1024 * 1024,
                };
                if libc::setrlimit(libc::RLIMIT_AS, &limit) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
        let mut child = command.spawn()?;
        loop {
            if let Some(status) = child.try_wait()? {
                if !status.success() {
                    failure = Some(format!("worker_exit:{status}"));
                }
                break;
            }
            if begun.elapsed() >= Duration::from_secs(30)
                || start.elapsed().as_secs_f64() + previous_seconds >= seconds as f64
            {
                let _ = child.kill();
                let _ = child.wait();
                failure = Some("candidate_or_view_timeout".into());
                break;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        }
        let result: Option<serde_json::Value> = if failure.is_none() || result_path.exists() {
            failure = None;
            Some(read_json(&result_path)?)
        } else {
            None
        };
        if let Some(r) = &result {
            experiment_failpoint("candidate_metric", experiment.is_some());
            let rmse = r["rmse"].as_f64().context("invalid evaluator RMSE")? * pp.target_scale;
            ensure!(rmse.is_finite() && rmse >= 0.0, "invalid evaluator score");
            let signature = r["semantic_sha256"]
                .as_str()
                .context("missing prediction signature")?;
            if semantic.insert(signature.to_string()) {
                ranked.push((rmse, genome.clone()));
                ranked.sort_by(|a, b| a.0.total_cmp(&b.0));
                ranked.truncate(80);
            }
        }
        use std::io::Read;
        let mut diagnostic = Vec::new();
        fs::File::open(output.join(format!("candidate-{i}.log")))?
            .take(8192)
            .read_to_end(&mut diagnostic)?;
        let diagnostic = String::from_utf8_lossy(&diagnostic).to_string();
        let origin = if initial_proposal {
            supplied
                .as_ref()
                .map(|s| s.candidates[execution_order[i]].origin.as_str())
                .unwrap_or("generic_bootstrap")
        } else {
            "gp_evolution"
        };
        let observation = json!({"schema":"parallel.gp_observation.v1","dataset":p.dataset,"family":p.family,"partition":p.meta_partition,"cut":cut,"prepared_sha256":prepared_sha,"training_sha256":c.training_sha256,"validation_sha256":c.validation_sha256,"genome":genome,"program_sha256":file_hash(&genome_path)?,"evaluator_sha256":evaluator_sha,"evaluation_bias":"adaptive_cut_heldout_teacher","family_partition_frozen":p.family_partition_frozen,"qualification":qualification,"independent_evaluation":false,"target_access":"cut heldout optimized repeatedly; original final test sealed","origin":origin,"initial_seed_slot":if initial_proposal{Some(execution_order[i])}else{None},"execution_order_protocol":execution_protocol,"seeds_sha256":seeds_sha256,"vocabulary_sha256":vocabulary_sha256,"proposal_before_projection":if raw_genome!=genome{Some(&raw_genome)}else{None},"evaluation_index":i,"rng_seed":rng_seed,"proposal_attempts":proposal_attempts,"rejected_proposals":rejected_proposals,"recovered_completed_artifact":recovered_metric,"runtime_seconds":if recovered_metric {result.as_ref().and_then(|r| r["seconds"].as_f64()).unwrap_or(0.0)} else {begun.elapsed().as_secs_f64()},"completed_search_seconds":start.elapsed().as_secs_f64()+previous_seconds,"search_clock":SEARCH_CLOCK,"search_clock_accuracy":clock_accuracy,"failure":failure,"worker_diagnostic":diagnostic,"metrics":result,"target_scale":pp.target_scale});
        use std::io::Write;
        let mut line = serde_json::to_vec(&observation)?;
        line.push(b'\n');
        journal.write_all(&line)?;
        journal.sync_data()?;
        experiment_failpoint("observation_journal", experiment.is_some());
        if let Some(root) = &champion_root {
            crate::champions::record(root, &observation, &output.join("observations.json"))?;
        }
        observations.push(observation);
        write_json(
            &output.join("progress.json"),
            &json!({"prepared_sha256":prepared_sha,"cut":cut,"seeds_sha256":seeds_sha256,"vocabulary_sha256":vocabulary_sha256,"execution_order_protocol":execution_protocol,"execution_order":execution_order,"wall_seconds":start.elapsed().as_secs_f64()+previous_seconds,"search_clock":clock,"evaluations":observations.len(),"rng_seed":rng_seed,"proposal_attempts":proposal_attempts,"rejected_proposals":rejected_proposals,"best_validation_rmse":ranked.first().map(|x|x.0),"timestamp":crate::telemetry::now()}),
        )?;
        experiment_failpoint("observation_progress", experiment.is_some());
        for path in [
            &genome_path,
            &result_path,
            &attempt_path,
            &output.join(format!("candidate-{i}.log")),
        ] {
            if path.exists() {
                fs::remove_file(path)?;
            }
        }
    }
    write_json(&output.join("observations.json"), &observations)?;
    experiment_failpoint("observations_snapshot", experiment.is_some());
    drop(journal);
    // Retain the durable journal until search publication; restart must never lose committed work.
    write_json(
        &output.join("search.json"),
        &json!({"schema":"parallel.search.v1","completed_evaluations":observations.len(),"wall_seconds":start.elapsed().as_secs_f64()+previous_seconds,"search_clock":clock,"execution_order_protocol":execution_protocol,"execution_order":execution_order,"resumed":resuming,"rng_seed":rng_seed,"proposal_attempts":proposal_attempts,"rejected_proposals":rejected_proposals,"unique_programs":evaluated_programs.len(),"best_validation_rmse":ranked.first().map(|x|x.0),"best_program":ranked.first().map(|x|&x.1),"seed_population_supplied":supplied.is_some(),"neural_seeded":neural_seeded,"seeds_sha256":seeds_sha256,"vocabulary_sha256":vocabulary_sha256,"final_tests_opened":false}),
    )?;
    experiment_failpoint("search_summary", experiment.is_some());
    fs::remove_file(&journal_path)?;
    Ok(())
}
