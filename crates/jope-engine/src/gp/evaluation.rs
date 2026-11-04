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
