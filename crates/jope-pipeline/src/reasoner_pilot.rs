//! Contract rules for the reasoner pilot (RP-02). These are the executable forms of the contract
//! fixtures FX-01 to FX-05 in `cards_and_fixtures/TYPED_EDIT_REPLAY_AND_ABLATION_CONTRACT_FIXTURES.json`.
//!
//! This module is a contract model over synthetic fixtures. It runs no model, reads no dataset, and
//! performs no evaluation. It records the rules that a later pilot must satisfy, so that they are
//! tested before any run exists.
use anyhow::{anyhow, bail, ensure, Context, Result};

pub mod adapters;

/// Runs one arm of the named reasoner study under the no-time-limit policy
/// (`NO_TIME_LIMIT_POLICY_20261019`). The teacher arm and every matched baseline arm must go through
/// this function, so the policy is identical for each by construction.
pub fn run_study_arm(options: &crate::run::RunOptions) -> Result<crate::receipt::Receipt> {
    crate::run::run_no_time_limit(options)
}

/// The one source kind the named native study admits: the TRAIN-only native teacher table.
pub const NATIVE_STUDY_SOURCE_KIND: &str = "native_teacher_train_only";
/// Written create-only once the winner is chosen on B, before O is opened.
pub const FROZEN_WINNER_FILE: &str = "frozen-winner.json";
/// Created exclusively when the one O assessment starts. A retry finds it and is refused.
pub const ASSESSMENT_CLAIM_FILE: &str = "assessment-claim.json";
/// The schema of the frozen-winner record.
pub const FROZEN_SCHEMA: &str = "jope.native-study-frozen-winner/v2";
/// The schema of the owner's admission manifest.
pub const ADMISSION_SCHEMA: &str = "jope.native-study-admission/v1";

/// SHA-256 of the owner-admitted admission manifest. It is not pinned in this build. The owner has to
/// supply the root hash and the manifest's locator. While this is `None`, every load refuses, so the
/// named study cannot start. A caller cannot supply its own root.
pub const OWNER_ADMISSION_ROOT_SHA256: Option<&str> = None;

/// One admitted study: the family, the source, the role, and the exact row bindings the owner admitted.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct NativeStudyAdmission {
    pub family_id: String,
    pub source_kind: String,
    pub role: String,
    pub ntest: u64,
    pub cut_input_digest: String,
    pub fit_a_rows_sha256: String,
    pub hold_b_rows_sha256: String,
}

#[derive(serde::Deserialize)]
struct AdmissionManifest {
    schema: String,
    admissions: Vec<NativeStudyAdmission>,
}

/// The admissions verified against a root hash. The only constructors check the manifest bytes against
/// a root that the caller does not choose, so a caller cannot make its own admission.
#[derive(Debug)]
pub struct TrustedAdmissions {
    admissions: Vec<NativeStudyAdmission>,
}

impl TrustedAdmissions {
    /// Loads the owner's manifest against the pinned root. Fails closed while no root is pinned.
    pub fn load_pinned(path: &std::path::Path) -> Result<Self> {
        let root = OWNER_ADMISSION_ROOT_SHA256.ok_or_else(|| {
            anyhow!(
                "BLOCKER: no owner admission root is pinned in this build; the owner must supply the root SHA-256 and the manifest locator"
            )
        })?;
        Self::verify(&std::fs::read(path)?, root)
    }

    /// Checks the manifest bytes against `root_sha256`, then parses them.
    fn verify(bytes: &[u8], root_sha256: &str) -> Result<Self> {
        ensure!(
            sha256_hex(bytes) == root_sha256,
            "the admission manifest does not match the pinned root"
        );
        let manifest: AdmissionManifest = serde_json::from_slice(bytes)?;
        ensure!(
            manifest.schema == ADMISSION_SCHEMA,
            "unknown admission schema {:?}",
            manifest.schema
        );
        Ok(Self {
            admissions: manifest.admissions,
        })
    }

    fn admits(&self, admission: &NativeStudyAdmission) -> bool {
        self.admissions.iter().any(|entry| entry == admission)
    }
}

/// A caller's role proof for the named native study. It holds row bindings only. Validation
/// recomputes each hash from the rows it is given and refuses any mismatch. The proof is not
/// authentication: authentication is the trusted admission in [`TrustedAdmissions`].
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct NativeStudyRoleProof {
    pub family_id: String,
    pub expected_cut_input_digest: String,
    pub fit_a_rows_sha256: String,
    pub hold_b_rows_sha256: String,
    pub role_attestation_sha256: String,
}

impl NativeStudyRoleProof {
    /// Computes the row-binding proof for `a_rows` (fit) and `b_rows` (hold). It is computed from the
    /// rows given, so it matches those rows and nothing else.
    pub fn attest_train_only(
        family_id: &str,
        features: &[Vec<f64>],
        targets: &[f64],
        a_rows: &[u32],
        b_rows: &[u32],
    ) -> Self {
        let fit = rows_sha256(features, targets, a_rows);
        let hold = rows_sha256(features, targets, b_rows);
        let digest = compute_cut_input_digest(features, targets, a_rows, b_rows);
        Self {
            family_id: family_id.into(),
            role_attestation_sha256: role_attestation_sha256(family_id, &fit, &hold, &digest),
            expected_cut_input_digest: digest,
            fit_a_rows_sha256: fit,
            hold_b_rows_sha256: hold,
        }
    }
}

/// The inputs of the named native study. `a_rows` are fitted, `b_rows` select the winner, and
/// `o_rows` are read once, at assessment, after the winner is frozen. The search never receives
/// `o_rows`: its only input is the materialised A/B cut.
#[derive(Clone, Copy, Debug)]
pub struct NativeStudySource<'a> {
    pub source_kind: &'a str,
    pub role: &'a str,
    pub ntest: u64,
    pub role_proof: &'a NativeStudyRoleProof,
    pub admissions: &'a TrustedAdmissions,
    pub features: &'a [Vec<f64>],
    pub targets: &'a [f64],
    pub a_rows: &'a [u32],
    pub b_rows: &'a [u32],
    pub o_rows: &'a [u32],
}

/// The bindings recomputed from the rows. A proof must match them; they are never copied from it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeStudyBindings {
    pub cut_input_digest: String,
    pub fit_a_rows_sha256: String,
    pub hold_b_rows_sha256: String,
    pub role_attestation_sha256: String,
}

/// The winner and the bindings at the freeze. Assessment reads nothing but this record.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FrozenNativeStudyWinner {
    pub schema: String,
    pub source_kind: String,
    pub family_id: String,
    pub cut_input_digest: String,
    pub fit_a_rows_sha256: String,
    pub hold_b_rows_sha256: String,
    pub role_attestation_sha256: String,
    pub winner: String,
    pub validation_rmse_b: Option<f64>,
    pub physical_fits_at_freeze: u64,
    pub model_calls_at_freeze: u64,
}

/// The A/B search of the named study. `search` is the actual `search::run_cuts_with` result.
pub struct NativeStudyRun {
    pub search: crate::search::CutSearch,
    pub frozen: FrozenNativeStudyWinner,
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    format!("{:x}", Sha256::digest(bytes))
}

/// Feeds one length-prefixed field into a SHA-256, so two different field lists cannot collide by
/// shifting bytes between fields.
fn absorb(hasher: &mut sha2::Sha256, field: &[u8]) {
    use sha2::Digest;
    hasher.update((field.len() as u64).to_le_bytes());
    hasher.update(field);
}

/// Absorbs each row: its original ID, then its features and its target, as exact bits.
fn absorb_rows(hasher: &mut sha2::Sha256, features: &[Vec<f64>], targets: &[f64], rows: &[u32]) {
    for &row in rows {
        absorb(hasher, &row.to_le_bytes());
        for value in &features[row as usize] {
            absorb(hasher, &value.to_bits().to_le_bytes());
        }
        absorb(hasher, &targets[row as usize].to_bits().to_le_bytes());
    }
}

/// SHA-256 over one partition's rows: original IDs, features and targets.
fn rows_sha256(features: &[Vec<f64>], targets: &[f64], rows: &[u32]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    absorb_rows(&mut hasher, features, targets, rows);
    format!("{:x}", hasher.finalize())
}

/// SHA-256 over the A and B partitions, tagged, in the order the search sees them.
fn compute_cut_input_digest(
    features: &[Vec<f64>],
    targets: &[f64],
    a_rows: &[u32],
    b_rows: &[u32],
) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    absorb(&mut hasher, b"A");
    absorb_rows(&mut hasher, features, targets, a_rows);
    absorb(&mut hasher, b"B");
    absorb_rows(&mut hasher, features, targets, b_rows);
    format!("{:x}", hasher.finalize())
}

/// SHA-256 over the role proof fields: the tag, the family, the TRAIN role, the source kind, ntest 0,
/// and the three hashes the proof attests.
fn role_attestation_sha256(family_id: &str, fit: &str, hold: &str, digest: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    absorb(&mut hasher, b"jope.native_study_role_proof.v1");
    absorb(&mut hasher, family_id.as_bytes());
    absorb(&mut hasher, b"TRAIN");
    absorb(&mut hasher, NATIVE_STUDY_SOURCE_KIND.as_bytes());
    absorb(&mut hasher, &0u32.to_le_bytes());
    absorb(&mut hasher, fit.as_bytes());
    absorb(&mut hasher, hold.as_bytes());
    absorb(&mut hasher, digest.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Refuses every role but TRAIN, every source but the TRAIN-only native table, `ntest > 0`, repeated
/// original row IDs, and any partition set the native evaluator rejects. It then recomputes the row
/// bindings and requires the proof to match them. Last, it requires a trusted admission for exactly
/// this family, role and row set. All of this runs before any fit.
fn validate_native_study_source(source: &NativeStudySource<'_>) -> Result<NativeStudyBindings> {
    match source.role {
        "TRAIN" => {}
        "E" | "HOLDOUT" | "TEST" | "SEALED" => {
            bail!("role {} is not admitted to the native search study", source.role)
        }
        other => bail!("unknown role {other:?} is not admitted to the native search study"),
    }
    ensure!(
        source.source_kind == NATIVE_STUDY_SOURCE_KIND,
        "source kind {:?} is not admitted (expected {NATIVE_STUDY_SOURCE_KIND})",
        source.source_kind
    );
    ensure!(source.ntest == 0, "source ntest must be 0, found {}", source.ntest);
    for (name, rows) in [("A", source.a_rows), ("B", source.b_rows), ("O", source.o_rows)] {
        let unique: std::collections::BTreeSet<u32> = rows.iter().copied().collect();
        ensure!(unique.len() == rows.len(), "partition {name} repeats an original row ID");
    }
    adapters::NativeTeacherFoldEvaluator::new(
        source.features,
        source.targets,
        source.a_rows,
        source.b_rows,
        source.o_rows,
    )
    .map(|_| ())
    .map_err(|error| anyhow!("invalid native study partitions: {error:?}"))?;

    let fit = rows_sha256(source.features, source.targets, source.a_rows);
    let hold = rows_sha256(source.features, source.targets, source.b_rows);
    let digest =
        compute_cut_input_digest(source.features, source.targets, source.a_rows, source.b_rows);
    let attestation = role_attestation_sha256(&source.role_proof.family_id, &fit, &hold, &digest);
    let proof = source.role_proof;
    ensure!(
        proof.expected_cut_input_digest == digest,
        "the role proof is not bound to the actual cut input digest"
    );
    ensure!(
        proof.fit_a_rows_sha256 == fit,
        "the role proof's A-row hash does not match the fit rows"
    );
    ensure!(
        proof.hold_b_rows_sha256 == hold,
        "the role proof's B-row hash does not match the hold rows"
    );
    ensure!(
        proof.role_attestation_sha256 == attestation,
        "the role attestation does not verify over the proof's fields"
    );
    let admission = NativeStudyAdmission {
        family_id: proof.family_id.clone(),
        source_kind: source.source_kind.into(),
        role: source.role.into(),
        ntest: source.ntest,
        cut_input_digest: digest.clone(),
        fit_a_rows_sha256: fit.clone(),
        hold_b_rows_sha256: hold.clone(),
    };
    ensure!(
        source.admissions.admits(&admission),
        "no trusted admission covers this family, role and row set; a self-attested proof is refused"
    );
    Ok(NativeStudyBindings {
        cut_input_digest: digest,
        fit_a_rows_sha256: fit,
        hold_b_rows_sha256: hold,
        role_attestation_sha256: attestation,
    })
}

/// One cut: `train = A`, `hold = B`. Built directly from the admitted rows. `prepare::partition` and
/// `prepare::build` are never called, because they would make a different fit/selection split.
fn materialize_native_cut(source: &NativeStudySource<'_>) -> crate::prepare::Cut {
    let rows = |idx: &[u32]| -> Vec<Vec<f64>> {
        idx.iter().map(|&row| source.features[row as usize].clone()).collect()
    };
    let labels = |idx: &[u32]| -> Vec<f64> {
        idx.iter().map(|&row| source.targets[row as usize]).collect()
    };
    crate::prepare::Cut {
        index: 0,
        train_x: rows(source.a_rows),
        train_y: labels(source.a_rows),
        hold_x: rows(source.b_rows),
        hold_y: labels(source.b_rows),
        hold_indices: (source.a_rows.len()..source.a_rows.len() + source.b_rows.len()).collect(),
    }
}

/// Test-only fault and trace hooks for the create-only durability sequence. Thread-local, so parallel
/// tests do not share them. They inject a failed directory sync in the test's own fixture. They do
/// not simulate a power failure.
#[cfg(test)]
pub(crate) mod durability_hooks {
    use std::cell::{Cell, RefCell};
    thread_local! {
        pub static TRACE: RefCell<Vec<&'static str>> = const { RefCell::new(Vec::new()) };
        pub static FAIL_NEXT_DIR_SYNC: Cell<bool> = const { Cell::new(false) };
    }
    pub fn clear() {
        TRACE.with(|t| t.borrow_mut().clear());
    }
    pub fn events() -> Vec<&'static str> {
        TRACE.with(|t| t.borrow().clone())
    }
    pub fn fail_next_dir_sync() {
        FAIL_NEXT_DIR_SYNC.with(|f| f.set(true));
    }
}

/// Records one step of the create-only durability sequence. Compiled out of non-test builds.
fn trace(event: &'static str) {
    #[cfg(test)]
    durability_hooks::TRACE.with(|t| t.borrow_mut().push(event));
    #[cfg(not(test))]
    let _ = event;
}

#[cfg(test)]
fn injected_dir_sync_fault() -> bool {
    durability_hooks::FAIL_NEXT_DIR_SYNC.with(|f| f.replace(false))
}

#[cfg(not(test))]
fn injected_dir_sync_fault() -> bool {
    false
}

/// Makes a create-only file's directory entry durable. A file's own `sync_all` does not persist the
/// name it was created under: a crash can lose the file, and a new create could then take its place.
/// Each caller runs this before it acknowledges the file or reads O. Errors propagate fail-closed.
fn sync_containing_dir(dir: &std::path::Path) -> Result<()> {
    ensure!(
        !injected_dir_sync_fault(),
        "injected directory sync failure (test fixture only) for {}",
        dir.display()
    );
    std::fs::File::open(dir)
        .with_context(|| format!("cannot open {} to sync it", dir.display()))?
        .sync_all()
        .with_context(|| format!("cannot sync directory {}", dir.display()))
}

/// The named native study's search arm. It runs `search::run_cuts_with` on the materialised A/B cut
/// under `without_time_limit()`, with the caller's counters shared into the search. It then writes
/// the frozen winner create-only, and returns. O is not opened here.
pub fn run_native_search_study_arm(
    source: NativeStudySource<'_>,
    budget: crate::search::Budget,
    seed: u64,
    counters: std::sync::Arc<crate::search::ExecutionBoundaryCounters>,
    output: &std::path::Path,
) -> Result<NativeStudyRun> {
    let bindings = validate_native_study_source(&source)?;
    ensure!(
        counters.caps().max_reasoning_tokens_per_call.is_none(),
        "the named native study runs under the natural-EOS token policy; a per-call token cap is refused"
    );
    let cut = materialize_native_cut(&source);
    let mut options = crate::search::SearchOptions::new(budget, seed).without_time_limit();
    options.fit_counters = Some(std::sync::Arc::clone(&counters));
    ensure!(options.time_limit().is_none(), "the named native study must run without a time limit");
    std::fs::create_dir_all(output)?;
    let search = crate::search::run_cuts_with(std::slice::from_ref(&cut), &options, output)?
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("the search returned no cut"))?;
    let winner = search
        .best_program
        .clone()
        .ok_or_else(|| anyhow!("the A/B search measured no winner"))?;
    let frozen = FrozenNativeStudyWinner {
        schema: FROZEN_SCHEMA.into(),
        source_kind: source.source_kind.into(),
        family_id: source.role_proof.family_id.clone(),
        cut_input_digest: bindings.cut_input_digest,
        fit_a_rows_sha256: bindings.fit_a_rows_sha256,
        hold_b_rows_sha256: bindings.hold_b_rows_sha256,
        role_attestation_sha256: bindings.role_attestation_sha256,
        winner,
        validation_rmse_b: search.best_validation_rmse,
        physical_fits_at_freeze: counters.physical_fits(),
        model_calls_at_freeze: counters.model_calls(),
    };
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output.join(FROZEN_WINNER_FILE))
        .context("the winner is already frozen")?;
    trace("frozen_created");
    std::io::Write::write_all(&mut file, &serde_json::to_vec_pretty(&frozen)?)?;
    file.sync_all()?;
    trace("frozen_file_synced");
    // The freeze is not acknowledged until its name is durable. An error here returns to the caller
    // with no run, so the arm is never marked frozen.
    sync_containing_dir(output)?;
    trace("frozen_dir_synced");
    Ok(NativeStudyRun { search, frozen })
}

/// The single O assessment. It validates the proof and the admission against the inputs, reads the
/// frozen winner from disk, checks that every binding matches it, claims the assessment exclusively,
/// and then scores the frozen winner on O once. The assessment fits on A and draws on the same
/// counters as the search. A failed or refused assessment keeps its claim: there is no second attempt.
/// Returns NaN when the winner does not fit or predict.
pub fn assess_frozen_winner(
    source: NativeStudySource<'_>,
    counters: std::sync::Arc<crate::search::ExecutionBoundaryCounters>,
    output: &std::path::Path,
) -> Result<f64> {
    let bindings = validate_native_study_source(&source)?;
    let frozen: FrozenNativeStudyWinner = serde_json::from_slice(
        &std::fs::read(output.join(FROZEN_WINNER_FILE))
            .context("no frozen winner exists; O stays closed")?,
    )?;
    ensure!(
        frozen.schema == FROZEN_SCHEMA,
        "the frozen winner has schema {:?}, expected {FROZEN_SCHEMA}",
        frozen.schema
    );
    ensure!(
        frozen.source_kind == source.source_kind && frozen.family_id == source.role_proof.family_id,
        "the frozen winner's source or family does not match the assessment source"
    );
    ensure!(
        frozen.cut_input_digest == bindings.cut_input_digest
            && frozen.fit_a_rows_sha256 == bindings.fit_a_rows_sha256
            && frozen.hold_b_rows_sha256 == bindings.hold_b_rows_sha256
            && frozen.role_attestation_sha256 == bindings.role_attestation_sha256,
        "the assessment inputs do not match the frozen A/B cut"
    );
    let mut claim = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output.join(ASSESSMENT_CLAIM_FILE))
        .context("the assessment is already claimed; a retry is refused")?;
    trace("claim_created");
    std::io::Write::write_all(
        &mut claim,
        &serde_json::to_vec_pretty(&serde_json::json!({ "winner": frozen.winner }))?,
    )?;
    claim.sync_all()?;
    trace("claim_file_synced");
    // The claim must be durable before O is read. If its name is lost in a crash, a second assessment
    // could run, so an error here stops the assessment before the evaluator is built.
    sync_containing_dir(output)?;
    trace("claim_dir_synced");
    let evaluator = adapters::NativeTeacherFoldEvaluator::new(
        source.features,
        source.targets,
        source.a_rows,
        source.b_rows,
        source.o_rows,
    )
    .map_err(|error| anyhow!("{error:?}"))?
    .with_shared_counters(counters);
    evaluator
        .score_assessment_once(&frozen.winner)
        .map_err(|error| anyhow!("{error:?}"))
}

/// The append-only study ledger's file name, inside the study directory.
pub const STUDY_LEDGER_FILE: &str = "study-ledger.json";
/// The exclusive study lock. The OS releases it when its holder exits, including after a crash.
pub const STUDY_LOCK_FILE: &str = "study.lock";
/// One JSON line per refused or failed durable operation. It never resets a counter.
pub const FAILURE_RECEIPTS_FILE: &str = "failure-receipts.jsonl";
const LEDGER_SCHEMA: &str = "jope.native-study-ledger/v2";

/// One external model call, as the caller reports it. `charged` before generation and `completed`
/// after EOS. A record is never removed. The completion tokens are the caller's report: they are not
/// physical proof that the E9 worker ran.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ExternalModelCall {
    pub call_id: String,
    pub state: String,
    pub generated_tokens: Option<u64>,
}

/// The counter totals written with each durable record, so a reopened study resumes them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CounterSnapshot {
    pub physical_fits: u64,
    pub model_calls: u64,
    pub generated_tokens: u64,
    pub max_call_tokens: u64,
}

impl CounterSnapshot {
    fn of(counters: &crate::search::ExecutionBoundaryCounters) -> Self {
        Self {
            physical_fits: counters.physical_fits(),
            model_calls: counters.model_calls(),
            generated_tokens: counters.generated_tokens_total(),
            max_call_tokens: counters.max_call_tokens_observed(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ArmState {
    /// Admitted and started. Its fit consumption is unknown until it freezes.
    Admitted,
    /// Frozen: the winner is on disk, and its bindings are recorded.
    Frozen,
    /// The one O assessment has completed.
    Assessed,
}

/// One admitted arm: its state and the bindings it was admitted under.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ArmRecord {
    pub name: String,
    pub state: ArmState,
    pub admission: Option<NativeStudyAdmission>,
    pub attestation: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
struct StudyLedgerFile {
    schema: String,
    caps: crate::search::ExecutionBoundaryCaps,
    reserved_assessment_fits: u64,
    expected_arms: Vec<String>,
    arms: Vec<ArmRecord>,
    counters: CounterSnapshot,
    external_calls: Vec<ExternalModelCall>,
}

/// Checks an arm's frozen-winner file against the bindings the arm was admitted under. Existence alone
/// does not qualify an arm.
fn check_frozen_against_record(dir: &std::path::Path, record: &ArmRecord) -> Result<()> {
    let admission = record
        .admission
        .as_ref()
        .ok_or_else(|| anyhow!("arm {:?} has no admission record", record.name))?;
    let attestation = record
        .attestation
        .as_deref()
        .ok_or_else(|| anyhow!("arm {:?} has no attestation record", record.name))?;
    let frozen: FrozenNativeStudyWinner = serde_json::from_slice(
        &std::fs::read(dir.join(FROZEN_WINNER_FILE))
            .with_context(|| format!("arm {:?} has no frozen winner", record.name))?,
    )?;
    ensure!(
        frozen.schema == FROZEN_SCHEMA,
        "arm {:?}: frozen winner schema {:?} is not {FROZEN_SCHEMA}",
        record.name,
        frozen.schema
    );
    ensure!(
        frozen.source_kind == admission.source_kind
            && frozen.family_id == admission.family_id
            && frozen.cut_input_digest == admission.cut_input_digest
            && frozen.fit_a_rows_sha256 == admission.fit_a_rows_sha256
            && frozen.hold_b_rows_sha256 == admission.hold_b_rows_sha256
            && frozen.role_attestation_sha256 == attestation,
        "arm {:?}: its frozen winner does not match the bindings it was admitted under",
        record.name
    );
    Ok(())
}

/// Accepts names that are non-empty, at most 64 characters, made of ASCII letters, digits, '-' and
/// '_', and unique. A name becomes a directory, so it cannot contain a separator.
fn validate_arm_names(arms: &[&str]) -> Result<()> {
    ensure!(!arms.is_empty(), "a study needs at least one arm");
    let mut seen = std::collections::BTreeSet::new();
    for arm in arms {
        ensure!(
            !arm.is_empty()
                && arm.len() <= 64
                && arm.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'),
            "arm name {arm:?} is not a safe directory name"
        );
        ensure!(seen.insert(*arm), "arm name {arm:?} is defined twice");
    }
    Ok(())
}

fn acquire_study_lock(dir: &std::path::Path) -> Result<std::fs::File> {
    std::fs::create_dir_all(dir)?;
    let lock = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(dir.join(STUDY_LOCK_FILE))?;
    match lock.try_lock() {
        Ok(()) => Ok(lock),
        Err(std::fs::TryLockError::WouldBlock) => {
            bail!("the study directory is already owned by another open ledger")
        }
        Err(std::fs::TryLockError::Error(error)) => Err(error.into()),
    }
}

/// The study-wide controller. Every admitted arm and every final O assessment of one study shares one
/// counter set. Search arms stop short of the reserved assessment fits. An assessment waits until
/// every expected arm is frozen, and each frozen record is checked against its admission. External
/// model calls are charged to the same counters and persisted before they may generate.
///
/// The object holds an exclusive lock on the study directory for its whole lifetime. A second open
/// on the same directory is refused. Each durable write goes through a temporary file, `sync_all`,
/// a rename, and a `sync_all` of the directory.
pub struct NativeStudyLedger {
    dir: std::path::PathBuf,
    _lock: std::fs::File,
    counters: std::sync::Arc<crate::search::ExecutionBoundaryCounters>,
    file: StudyLedgerFile,
}

impl NativeStudyLedger {
    /// Creates a study directory and its ledger. It refuses, before any search, a reservation below
    /// the arm count, an unsafe or duplicate arm name, and an existing ledger.
    pub fn create(
        dir: &std::path::Path,
        caps: crate::search::ExecutionBoundaryCaps,
        reserved_assessment_fits: u64,
        expected_arms: &[&str],
    ) -> Result<Self> {
        validate_arm_names(expected_arms)?;
        ensure!(
            reserved_assessment_fits >= expected_arms.len() as u64,
            "the reservation of {reserved_assessment_fits} fits is below the {} arms; each arm needs its assessment fit",
            expected_arms.len()
        );
        ensure!(
            reserved_assessment_fits <= caps.max_physical_fits,
            "the reservation exceeds the physical-fit ceiling"
        );
        let lock = acquire_study_lock(dir)?;
        ensure!(
            !dir.join(STUDY_LEDGER_FILE).exists(),
            "a study ledger already exists in this directory; open it instead"
        );
        let file = StudyLedgerFile {
            schema: LEDGER_SCHEMA.into(),
            caps,
            reserved_assessment_fits,
            expected_arms: expected_arms.iter().map(|arm| (*arm).to_string()).collect(),
            arms: Vec::new(),
            counters: CounterSnapshot {
                physical_fits: 0,
                model_calls: 0,
                generated_tokens: 0,
                max_call_tokens: 0,
            },
            external_calls: Vec::new(),
        };
        let ledger = Self {
            dir: dir.to_path_buf(),
            _lock: lock,
            counters: std::sync::Arc::new(
                crate::search::ExecutionBoundaryCounters::new(caps)
                    .with_reserved_assessment_fits(reserved_assessment_fits),
            ),
            file,
        };
        ledger.write_new()?;
        Ok(ledger)
    }

    /// Reopens an existing study. It refuses a second owner. It refuses any study that cannot prove its
    /// fit and model-call totals: an arm admitted but never frozen, an assessment claimed but not
    /// completed, or an external call charged without a completion. Such a study is halted until the
    /// owner reconciles it.
    pub fn open(dir: &std::path::Path) -> Result<Self> {
        let lock = acquire_study_lock(dir)?;
        let file: StudyLedgerFile = serde_json::from_slice(
            &std::fs::read(dir.join(STUDY_LEDGER_FILE))
                .context("no study ledger in this directory")?,
        )?;
        ensure!(
            file.schema == LEDGER_SCHEMA,
            "study ledger schema {:?} is not {LEDGER_SCHEMA}",
            file.schema
        );
        for record in &file.arms {
            match record.state {
                ArmState::Admitted => bail!(
                    "arm {:?} was admitted but never froze; its fit consumption is unknown and the study is halted",
                    record.name
                ),
                ArmState::Frozen => ensure!(
                    !dir.join("arms").join(&record.name).join(ASSESSMENT_CLAIM_FILE).exists(),
                    "arm {:?} has an assessment claim with no completed record; its fits are unknown and the study is halted",
                    record.name
                ),
                ArmState::Assessed => {}
            }
        }
        for call in &file.external_calls {
            ensure!(
                call.state == "completed",
                "external call {:?} was charged with no completion; its outcome is unknown and the study is halted",
                call.call_id
            );
        }
        let snapshot = file.counters;
        let counters = std::sync::Arc::new(crate::search::ExecutionBoundaryCounters::restored(
            file.caps,
            file.reserved_assessment_fits,
            snapshot.physical_fits,
            snapshot.model_calls,
            snapshot.generated_tokens,
            snapshot.max_call_tokens,
        ));
        Ok(Self {
            dir: dir.to_path_buf(),
            _lock: lock,
            counters,
            file,
        })
    }

    /// The counters every arm and assessment of this study shares.
    pub fn counters(&self) -> std::sync::Arc<crate::search::ExecutionBoundaryCounters> {
        std::sync::Arc::clone(&self.counters)
    }

    /// Where an arm writes its search output and its frozen winner.
    pub fn arm_dir(&self, arm: &str) -> std::path::PathBuf {
        self.dir.join("arms").join(arm)
    }

    /// Runs one admitted arm on the shared counters. The source is validated, and the arm is recorded
    /// as admitted and persisted, before any fit. An arm runs once. If the run fails, the arm stays
    /// admitted and the study halts on reopen.
    pub fn run_arm(
        &mut self,
        arm: &str,
        source: NativeStudySource<'_>,
        budget: crate::search::Budget,
        seed: u64,
    ) -> Result<NativeStudyRun> {
        ensure!(
            self.file.expected_arms.iter().any(|expected| expected == arm),
            "arm {arm:?} is not part of this study"
        );
        ensure!(
            !self.file.arms.iter().any(|record| record.name == arm),
            "arm {arm:?} was already admitted"
        );
        let bindings = validate_native_study_source(&source)?;
        self.file.arms.push(ArmRecord {
            name: arm.into(),
            state: ArmState::Admitted,
            admission: Some(NativeStudyAdmission {
                family_id: source.role_proof.family_id.clone(),
                source_kind: source.source_kind.into(),
                role: source.role.into(),
                ntest: source.ntest,
                cut_input_digest: bindings.cut_input_digest,
                fit_a_rows_sha256: bindings.fit_a_rows_sha256,
                hold_b_rows_sha256: bindings.hold_b_rows_sha256,
            }),
            attestation: Some(bindings.role_attestation_sha256),
        });
        self.persist()?;
        let run = match run_native_search_study_arm(
            source,
            budget,
            seed,
            self.counters(),
            &self.arm_dir(arm),
        ) {
            Ok(run) => run,
            Err(error) => {
                self.record_failure("run_arm", &format!("{error:#}"));
                return Err(error);
            }
        };
        self.set_state(arm, ArmState::Frozen);
        self.persist()?;
        Ok(run)
    }

    /// The final O assessment of one arm. It is refused until every expected arm is frozen, and each
    /// frozen record matches the bindings its arm was admitted under.
    pub fn assess_arm(&mut self, arm: &str, source: NativeStudySource<'_>) -> Result<f64> {
        validate_native_study_source(&source)?;
        for expected in &self.file.expected_arms {
            let record = self
                .file
                .arms
                .iter()
                .find(|record| &record.name == expected)
                .ok_or_else(|| anyhow!("arm {expected:?} has not run; assessments wait for every arm"))?;
            ensure!(
                record.state != ArmState::Admitted,
                "arm {expected:?} never froze; assessments wait for every arm"
            );
            check_frozen_against_record(&self.arm_dir(expected), record)?;
        }
        let target = self
            .file
            .arms
            .iter()
            .find(|record| record.name == arm)
            .ok_or_else(|| anyhow!("arm {arm:?} has not run"))?;
        ensure!(
            target.state == ArmState::Frozen,
            "arm {arm:?} is not in the frozen state, so it cannot be assessed"
        );
        let value = match assess_frozen_winner(source, self.counters(), &self.arm_dir(arm)) {
            Ok(value) => value,
            Err(error) => {
                self.record_failure("assess_arm", &format!("{error:#}"));
                return Err(error);
            }
        };
        self.set_state(arm, ArmState::Assessed);
        self.persist()?;
        Ok(value)
    }

    /// Charges one external model call before it generates. The call counts against the call ceiling,
    /// and it is persisted as `charged` before this returns. A call id is charged once.
    pub fn charge_external_model_call_before_generate(&mut self, call_id: &str) -> Result<()> {
        ensure!(!call_id.is_empty(), "a model call needs an id");
        ensure!(
            !self.file.external_calls.iter().any(|call| call.call_id == call_id),
            "model call {call_id:?} was already charged"
        );
        self.counters
            .record_model_call(0)
            .map_err(|reason| anyhow!("external model call {call_id:?} refused: {reason}"))?;
        self.file.external_calls.push(ExternalModelCall {
            call_id: call_id.into(),
            state: "charged".into(),
            generated_tokens: None,
        });
        if let Err(error) = self.persist() {
            // The call stays counted. Generation must not start: the caller gets this error.
            self.record_failure("charge_external_model_call", &format!("{error:#}"));
            return Err(error);
        }
        Ok(())
    }

    /// Records the generated tokens of a charged call after its EOS. The tokens are the caller's report,
    /// not physical proof of execution. A call that was never charged, or is already completed, is
    /// refused.
    pub fn complete_external_model_call_after_eos(
        &mut self,
        call_id: &str,
        generated_tokens: u64,
    ) -> Result<()> {
        let call = self
            .file
            .external_calls
            .iter_mut()
            .find(|call| call.call_id == call_id)
            .ok_or_else(|| anyhow!("model call {call_id:?} was never charged; refused"))?;
        ensure!(call.state == "charged", "model call {call_id:?} is already completed");
        self.counters
            .add_generated_tokens(generated_tokens)
            .map_err(|reason| anyhow!("model call {call_id:?} refused: {reason}"))?;
        call.state = "completed".into();
        call.generated_tokens = Some(generated_tokens);
        if let Err(error) = self.persist() {
            self.record_failure("complete_external_model_call", &format!("{error:#}"));
            return Err(error);
        }
        Ok(())
    }

    fn set_state(&mut self, arm: &str, state: ArmState) {
        if let Some(record) = self.file.arms.iter_mut().find(|record| record.name == arm) {
            record.state = state;
        }
    }

    /// Creates the ledger file, create-only, and makes it durable with its directory.
    fn write_new(&self) -> Result<()> {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(self.dir.join(STUDY_LEDGER_FILE))
            .context("the study ledger already exists")?;
        std::io::Write::write_all(&mut file, &self.serialized()?)?;
        file.sync_all()?;
        std::fs::File::open(&self.dir)?.sync_all()?;
        Ok(())
    }

    /// The ledger with the current counter totals, as JSON.
    fn serialized(&self) -> Result<Vec<u8>> {
        let mut file = self.file.clone();
        file.counters = CounterSnapshot::of(&self.counters);
        Ok(serde_json::to_vec_pretty(&file)?)
    }

    /// Writes the whole ledger through a temporary file, `sync_all`, and a rename, then syncs the
    /// directory, so a reader sees either the old file or the new one.
    fn persist(&self) -> Result<()> {
        let temporary = self.dir.join(format!("{STUDY_LEDGER_FILE}.tmp"));
        {
            let mut file = std::fs::File::create(&temporary)?;
            std::io::Write::write_all(&mut file, &self.serialized()?)?;
            file.sync_all()?;
        }
        std::fs::rename(&temporary, self.dir.join(STUDY_LEDGER_FILE))?;
        std::fs::File::open(&self.dir)?.sync_all()?;
        Ok(())
    }

    /// Appends one failure receipt, durably, with the counter totals as they stand. It never resets a
    /// counter. The caller's error is returned whether or not this line lands.
    fn record_failure(&self, operation: &str, reason: &str) {
        let utc = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let line = serde_json::json!({
            "unix_seconds": utc,
            "operation": operation,
            "reason": reason,
            "counters": CounterSnapshot::of(&self.counters),
        });
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.dir.join(FAILURE_RECEIPTS_FILE))
        {
            let _ = std::io::Write::write_all(&mut file, format!("{line}\n").as_bytes());
            let _ = file.sync_all();
        }
    }
}

/// A measured value that is either recorded or explicitly unknown. An unknown value is never zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Measured<T> {
    Null,
    Value(T),
}

impl Measured<u64> {
    /// Adds recorded values. One unknown addend makes the total unknown.
    pub fn sum_known(parts: &[Measured<u64>]) -> Measured<u64> {
        let mut total = 0u64;
        for part in parts {
            match part {
                Measured::Null => return Measured::Null,
                Measured::Value(v) => total += v,
            }
        }
        Measured::Value(total)
    }
}

/// The outcome of one typed-edit attempt. Only a constructive edit verified on inner CV earns credit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EditOutcome {
    ConstructiveVerified,
    Rejected,
    Invalid,
    Timeout,
    JudgedOrReranked,
}

/// Number of credited edits. Every other outcome earns nothing.
pub fn credited_edits(outcomes: &[EditOutcome]) -> usize {
    outcomes
        .iter()
        .filter(|o| **o == EditOutcome::ConstructiveVerified)
        .count()
}

/// Stage 1 admits evidence only when a constructive verified edit is present. Judging or reranking
/// candidates alone is refused. The pass criterion and the equal-budget value are owner decisions
/// that are not set here, so this function admits evidence for comparison and does not declare a pass.
pub fn admit_stage1_evidence(outcomes: &[EditOutcome]) -> Result<()> {
    ensure!(
        !outcomes.is_empty(),
        "stage 1 has no edit attempts to admit"
    );
    if credited_edits(outcomes) == 0 {
        bail!("no constructive verified edit: judging or reranking alone does not qualify for stage 1");
    }
    Ok(())
}

/// The two numerical interfaces. Their widths and vocabularies are distinct, and a receipt for one
/// is never valid for the other.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Interface {
    Kumo2051,
    Hyperion2048,
}

impl Interface {
    pub fn embedding_width(self) -> u32 {
        match self {
            Interface::Kumo2051 => 2051,
            Interface::Hyperion2048 => 2048,
        }
    }
}

/// Refuses a receipt that was produced under a different interface than the target.
pub fn check_receipt_interface(receipt: Interface, target: Interface) -> Result<()> {
    ensure!(
        receipt == target,
        "receipt for {:?} presented to a {:?} interface; the two are never interchanged",
        receipt,
        target
    );
    Ok(())
}

/// Status of a pilot stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StageStatus {
    Pass,
    Fail,
    Locked,
}

/// Stage 3 may start only after stage 1 and stage 2 both pass.
pub fn stage3_may_start(stage1: StageStatus, stage2: StageStatus) -> Result<()> {
    ensure!(
        stage1 == StageStatus::Pass && stage2 == StageStatus::Pass,
        "stage 3 is locked until stage 1 and stage 2 both pass (got {stage1:?} and {stage2:?})"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FX-01: a rejected edit is recorded, its cost is counted, and it earns no credit.
    #[test]
    fn fx01_rejected_edit_costs_are_counted_and_earn_no_credit() {
        let attempts = [
            EditOutcome::Rejected,
            EditOutcome::ConstructiveVerified,
            EditOutcome::Timeout,
        ];
        assert_eq!(credited_edits(&attempts), 1);
        // Every attempt is a physical start and counts toward cost, whatever its outcome.
        let starts: Vec<Measured<u64>> = attempts.iter().map(|_| Measured::Value(1)).collect();
        assert_eq!(Measured::sum_known(&starts), Measured::Value(3));
        let only_rejected = [EditOutcome::Rejected, EditOutcome::Invalid];
        assert_eq!(credited_edits(&only_rejected), 0);
    }

    /// FX-02: candidate judging or reranking without a verified constructive edit is refused.
    #[test]
    fn fx02_judging_or_reranking_alone_is_refused_for_stage1() {
        let judged_only = [EditOutcome::JudgedOrReranked, EditOutcome::JudgedOrReranked];
        let err = admit_stage1_evidence(&judged_only).unwrap_err().to_string();
        assert!(err.contains("judging or reranking alone"), "{err}");
        assert!(admit_stage1_evidence(&[EditOutcome::ConstructiveVerified]).is_ok());
        assert!(admit_stage1_evidence(&[]).is_err());
    }

    /// FX-03: a Kumo2051 receipt is refused by a Hyperion2048 interface, and the reverse is refused too.
    #[test]
    fn fx03_kumo2051_and_hyperion2048_receipts_are_never_interchanged() {
        assert_eq!(Interface::Kumo2051.embedding_width(), 2051);
        assert_eq!(Interface::Hyperion2048.embedding_width(), 2048);
        assert!(check_receipt_interface(Interface::Kumo2051, Interface::Hyperion2048).is_err());
        assert!(check_receipt_interface(Interface::Hyperion2048, Interface::Kumo2051).is_err());
        assert!(check_receipt_interface(Interface::Kumo2051, Interface::Kumo2051).is_ok());
        assert!(check_receipt_interface(Interface::Hyperion2048, Interface::Hyperion2048).is_ok());
    }

    /// FX-04: stage 3 cannot start while stage 1 or stage 2 is not a pass.
    #[test]
    fn fx04_stage3_stays_locked_until_stages_1_and_2_pass() {
        assert!(stage3_may_start(StageStatus::Pass, StageStatus::Pass).is_ok());
        assert!(stage3_may_start(StageStatus::Pass, StageStatus::Fail).is_err());
        assert!(stage3_may_start(StageStatus::Fail, StageStatus::Pass).is_err());
        assert!(stage3_may_start(StageStatus::Pass, StageStatus::Locked).is_err());
        assert!(stage3_may_start(StageStatus::Locked, StageStatus::Locked).is_err());
    }

    /// FX-05: an unknown measured value never turns into zero in a total.
    #[test]
    fn fx05_a_null_measured_field_never_becomes_zero() {
        let parts = [Measured::Value(3), Measured::Null];
        let total = Measured::sum_known(&parts);
        assert_eq!(total, Measured::Null);
        assert_ne!(total, Measured::Value(0));
        assert_eq!(Measured::sum_known(&[]), Measured::Value(0));
    }
}

#[cfg(test)]
mod native_study_tests {
    use super::*;
    use crate::search::{Budget, ExecutionBoundaryCaps, ExecutionBoundaryCounters};
    use std::sync::Arc;

    /// Sixty rows, two features. A is 0..30, B is 30..45, O is 45..60.
    fn fixture() -> (Vec<Vec<f64>>, Vec<f64>) {
        let features: Vec<Vec<f64>> = (0..60)
            .map(|i| vec![i as f64 / 10.0, ((i * 7) % 13) as f64])
            .collect();
        let targets = features
            .iter()
            .map(|row| 2.0 * row[0] - row[1] + 0.5 * row[0] * row[1])
            .collect();
        (features, targets)
    }

    fn partitions() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
        ((0..30).collect(), (30..45).collect(), (45..60).collect())
    }

    fn budget() -> Budget {
        Budget {
            evaluations: 32,
            seconds: 60,
        }
    }

    fn honest_proof(x: &[Vec<f64>], y: &[f64], a: &[u32], b: &[u32]) -> NativeStudyRoleProof {
        NativeStudyRoleProof::attest_train_only("fixture-family", x, y, a, b)
    }

    /// A generated owner manifest admitting exactly the given rows under `role`. The root is the
    /// manifest's own SHA-256, computed here. In production the root is pinned outside the caller.
    fn admit_with_role(
        x: &[Vec<f64>],
        y: &[f64],
        a: &[u32],
        b: &[u32],
        role: &str,
    ) -> TrustedAdmissions {
        let entry = NativeStudyAdmission {
            family_id: "fixture-family".into(),
            source_kind: NATIVE_STUDY_SOURCE_KIND.into(),
            role: role.into(),
            ntest: 0,
            cut_input_digest: compute_cut_input_digest(x, y, a, b),
            fit_a_rows_sha256: rows_sha256(x, y, a),
            hold_b_rows_sha256: rows_sha256(x, y, b),
        };
        let bytes = serde_json::to_vec(&serde_json::json!({
            "schema": ADMISSION_SCHEMA,
            "admissions": [entry],
        }))
        .unwrap();
        TrustedAdmissions::verify(&bytes, &sha256_hex(&bytes)).unwrap()
    }

    fn admitted(x: &[Vec<f64>], y: &[f64], a: &[u32], b: &[u32]) -> TrustedAdmissions {
        admit_with_role(x, y, a, b, "TRAIN")
    }

    fn source<'a>(
        features: &'a [Vec<f64>],
        targets: &'a [f64],
        a: &'a [u32],
        b: &'a [u32],
        o: &'a [u32],
        role_proof: &'a NativeStudyRoleProof,
        admissions: &'a TrustedAdmissions,
    ) -> NativeStudySource<'a> {
        NativeStudySource {
            source_kind: NATIVE_STUDY_SOURCE_KIND,
            role: "TRAIN",
            ntest: 0,
            role_proof,
            admissions,
            features,
            targets,
            a_rows: a,
            b_rows: b,
            o_rows: o,
        }
    }

    fn natural_counters() -> Arc<ExecutionBoundaryCounters> {
        Arc::new(ExecutionBoundaryCounters::new(
            ExecutionBoundaryCaps::reasoner_natural_eos(),
        ))
    }

    fn run_once(
        x: &[Vec<f64>],
        y: &[f64],
        counters: Arc<ExecutionBoundaryCounters>,
        dir: &std::path::Path,
    ) -> Result<NativeStudyRun> {
        let (a, b, o) = partitions();
        let proof = honest_proof(x, y, &a, &b);
        let admissions = admitted(x, y, &a, &b);
        run_native_search_study_arm(
            source(x, y, &a, &b, &o, &proof, &admissions),
            budget(),
            5,
            counters,
            dir,
        )
    }

    /// Physical fits of one search of the fixture under the natural-EOS policy, with no reservation.
    fn search_fits() -> u64 {
        let (x, y) = fixture();
        let dir = tempfile::tempdir().unwrap();
        run_once(&x, &y, natural_counters(), dir.path())
            .unwrap()
            .frozen
            .physical_fits_at_freeze
    }

    #[test]
    fn native_study_refuses_held_out_roles_and_bad_sources_before_any_fit() {
        let (x, y) = fixture();
        let (a, b, o) = partitions();
        let proof = honest_proof(&x, &y, &a, &b);
        let admissions = admitted(&x, &y, &a, &b);
        let cases = [
            ("HOLDOUT", NATIVE_STUDY_SOURCE_KIND, 0),
            ("E", NATIVE_STUDY_SOURCE_KIND, 0),
            ("TEST", NATIVE_STUDY_SOURCE_KIND, 0),
            ("SEALED", NATIVE_STUDY_SOURCE_KIND, 0),
            ("TRAIN", "legacy_table", 0),
            ("TRAIN", NATIVE_STUDY_SOURCE_KIND, 3),
        ];
        for (role, kind, ntest) in cases {
            let counters = natural_counters();
            let mut s = source(&x, &y, &a, &b, &o, &proof, &admissions);
            s.role = role;
            s.source_kind = kind;
            s.ntest = ntest;
            let dir = tempfile::tempdir().unwrap();
            assert!(
                run_native_search_study_arm(s, budget(), 5, counters.clone(), dir.path()).is_err(),
                "{role} / {kind} / ntest {ntest} was admitted"
            );
            assert_eq!(counters.physical_fits(), 0, "{role} reached a fit");
        }
    }

    #[test]
    fn forged_or_misbound_role_proofs_are_refused_before_any_fit() {
        let (x, y) = fixture();
        let (a, b, o) = partitions();
        let honest = honest_proof(&x, &y, &a, &b);
        let admissions = admitted(&x, &y, &a, &b);
        let forged = "0".repeat(64);
        let mut cases: Vec<(&str, NativeStudyRoleProof)> = Vec::new();
        let mut p = honest.clone();
        p.expected_cut_input_digest = forged.clone();
        cases.push(("expected_cut_input_digest", p));
        let mut p = honest.clone();
        p.fit_a_rows_sha256 = forged.clone();
        cases.push(("fit_a_rows_sha256", p));
        let mut p = honest.clone();
        p.hold_b_rows_sha256 = forged.clone();
        cases.push(("hold_b_rows_sha256", p));
        let mut p = honest.clone();
        p.role_attestation_sha256 = forged.clone();
        cases.push(("role_attestation_sha256", p));
        let short_b: Vec<u32> = (30..40).collect();
        cases.push(("proof of a different B", honest_proof(&x, &y, &a, &short_b)));

        for (field, proof) in cases {
            let counters = natural_counters();
            let dir = tempfile::tempdir().unwrap();
            assert!(
                run_native_search_study_arm(
                    source(&x, &y, &a, &b, &o, &proof, &admissions),
                    budget(),
                    5,
                    counters.clone(),
                    dir.path()
                )
                .is_err(),
                "a forged {field} was admitted"
            );
            assert_eq!(counters.physical_fits(), 0, "a forged {field} reached a fit");
            assert!(!dir.path().join(FROZEN_WINNER_FILE).exists(), "{field} froze a winner");
        }
    }

    /// A self-consistent proof is not authentication. A proof whose rows the owner never admitted is
    /// refused before any fit, and so is a proof for a role the owner admitted under another name.
    #[test]
    fn self_attested_proof_outside_the_trusted_admission_is_refused_before_any_fit() {
        let (x, y) = fixture();
        let (a, b, o) = partitions();
        let admissions = admitted(&x, &y, &a, &b);

        let short_b: Vec<u32> = (30..40).collect();
        let self_attested = honest_proof(&x, &y, &a, &short_b);
        let counters = natural_counters();
        let dir = tempfile::tempdir().unwrap();
        assert!(
            run_native_search_study_arm(
                source(&x, &y, &a, &short_b, &o, &self_attested, &admissions),
                budget(),
                5,
                counters.clone(),
                dir.path()
            )
            .is_err(),
            "a self-attested proof over rows the owner never admitted was accepted"
        );
        assert_eq!(counters.physical_fits(), 0);

        let holdout_only = admit_with_role(&x, &y, &a, &b, "HOLDOUT");
        let proof = honest_proof(&x, &y, &a, &b);
        let counters = natural_counters();
        let dir = tempfile::tempdir().unwrap();
        assert!(
            run_native_search_study_arm(
                source(&x, &y, &a, &b, &o, &proof, &holdout_only),
                budget(),
                5,
                counters.clone(),
                dir.path()
            )
            .is_err(),
            "a TRAIN source was admitted under a HOLDOUT manifest entry"
        );
        assert_eq!(counters.physical_fits(), 0);
    }

    #[test]
    fn the_trusted_manifest_must_match_its_root_and_the_pinned_root_fails_closed() {
        let (x, y) = fixture();
        let (a, b, _) = partitions();
        let entry = NativeStudyAdmission {
            family_id: "fixture-family".into(),
            source_kind: NATIVE_STUDY_SOURCE_KIND.into(),
            role: "TRAIN".into(),
            ntest: 0,
            cut_input_digest: compute_cut_input_digest(&x, &y, &a, &b),
            fit_a_rows_sha256: rows_sha256(&x, &y, &a),
            hold_b_rows_sha256: rows_sha256(&x, &y, &b),
        };
        let bytes = serde_json::to_vec(&serde_json::json!({
            "schema": ADMISSION_SCHEMA,
            "admissions": [entry],
        }))
        .unwrap();
        let root = sha256_hex(&bytes);
        assert!(TrustedAdmissions::verify(&bytes, &root).is_ok());
        assert!(
            TrustedAdmissions::verify(&bytes, &"0".repeat(64)).is_err(),
            "a manifest was accepted against a wrong root"
        );
        let mut tampered = bytes.clone();
        tampered.push(b' ');
        assert!(
            TrustedAdmissions::verify(&tampered, &root).is_err(),
            "a tampered manifest was accepted"
        );
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("admission.json");
        std::fs::write(&path, &bytes).unwrap();
        let error = TrustedAdmissions::load_pinned(&path).unwrap_err().to_string();
        assert!(
            error.contains("BLOCKER"),
            "the unpinned path did not fail closed with the blocker: {error}"
        );
    }

    #[test]
    fn search_sees_only_a_and_b_and_o_is_assessed_once_after_the_freeze() {
        let (x, y) = fixture();
        let (a, b, o) = partitions();
        let dir = tempfile::tempdir().unwrap();
        let counters = natural_counters();
        let run = run_once(&x, &y, counters.clone(), dir.path()).unwrap();
        let fits = run.frozen.physical_fits_at_freeze;
        assert!(fits > 0, "the search made no physical fit");
        assert!(
            run.search.successful_measurements > 0,
            "the bootstrap fixture produced no successful native measurement"
        );
        assert_eq!(counters.physical_fits(), fits);

        let proof = honest_proof(&x, &y, &a, &b);
        let admissions = admitted(&x, &y, &a, &b);
        let cut = materialize_native_cut(&source(&x, &y, &a, &b, &o, &proof, &admissions));
        let a_x: Vec<Vec<f64>> = a.iter().map(|&r| x[r as usize].clone()).collect();
        let a_y: Vec<f64> = a.iter().map(|&r| y[r as usize]).collect();
        let b_x: Vec<Vec<f64>> = b.iter().map(|&r| x[r as usize].clone()).collect();
        let b_y: Vec<f64> = b.iter().map(|&r| y[r as usize]).collect();
        assert_eq!(cut.train_x, a_x, "train rows are not exactly A");
        assert_eq!(cut.train_y, a_y, "train labels are not exactly A");
        assert_eq!(cut.hold_x, b_x, "hold rows are not exactly B");
        assert_eq!(cut.hold_y, b_y, "hold labels are not exactly B");
        assert_eq!(cut.hold_indices, (30..45).collect::<Vec<usize>>());

        assert_eq!(run.frozen.fit_a_rows_sha256, proof.fit_a_rows_sha256);
        assert_eq!(run.frozen.hold_b_rows_sha256, proof.hold_b_rows_sha256);
        assert_eq!(run.frozen.role_attestation_sha256, proof.role_attestation_sha256);
        assert_eq!(run.frozen.schema, FROZEN_SCHEMA);
        let on_disk: FrozenNativeStudyWinner =
            serde_json::from_slice(&std::fs::read(dir.path().join(FROZEN_WINNER_FILE)).unwrap())
                .unwrap();
        assert_eq!(on_disk, run.frozen);

        let rmse = assess_frozen_winner(
            source(&x, &y, &a, &b, &o, &proof, &admissions),
            counters.clone(),
            dir.path(),
        )
        .unwrap();
        assert!(rmse.is_finite());
        assert_eq!(
            counters.physical_fits(),
            fits + 1,
            "the assessment is not exactly one more physical fit"
        );
        assert!(
            assess_frozen_winner(
                source(&x, &y, &a, &b, &o, &proof, &admissions),
                counters.clone(),
                dir.path()
            )
            .is_err(),
            "a second assessment was not refused"
        );
        assert_eq!(counters.physical_fits(), fits + 1, "a refused retry changed the quota");
    }

    #[test]
    fn poisoned_o_labels_leave_the_search_and_the_winner_unchanged() {
        let (x, mut y) = fixture();
        let (_, _, o) = partitions();
        let first_dir = tempfile::tempdir().unwrap();
        let first = run_once(&x, &y, natural_counters(), first_dir.path()).unwrap();
        for &row in &o {
            y[row as usize] = f64::NAN;
        }
        let second_dir = tempfile::tempdir().unwrap();
        let second = run_once(&x, &y, natural_counters(), second_dir.path()).unwrap();
        assert_eq!(first.frozen.cut_input_digest, second.frozen.cut_input_digest);
        assert_eq!(first.frozen.winner, second.frozen.winner);
        assert_eq!(first.frozen.validation_rmse_b, second.frozen.validation_rmse_b);
        assert_eq!(
            first.frozen.physical_fits_at_freeze,
            second.frozen.physical_fits_at_freeze
        );
    }

    #[test]
    fn assessment_refuses_a_missing_freeze_and_a_changed_cut_without_consuming_the_claim() {
        let (x, y) = fixture();
        let (a, b, o) = partitions();
        let proof = honest_proof(&x, &y, &a, &b);
        let admissions = admitted(&x, &y, &a, &b);
        let dir = tempfile::tempdir().unwrap();
        let counters = natural_counters();
        assert!(
            assess_frozen_winner(
                source(&x, &y, &a, &b, &o, &proof, &admissions),
                counters.clone(),
                dir.path()
            )
            .is_err(),
            "O opened without a frozen winner"
        );
        assert_eq!(counters.physical_fits(), 0);

        let run = run_once(&x, &y, counters.clone(), dir.path()).unwrap();
        let fits = counters.physical_fits();
        let mut changed = x.clone();
        changed[30][0] += 1.0;
        let changed_proof = honest_proof(&changed, &y, &a, &b);
        let changed_admissions = admitted(&changed, &y, &a, &b);
        assert!(
            assess_frozen_winner(
                source(&changed, &y, &a, &b, &o, &changed_proof, &changed_admissions),
                counters.clone(),
                dir.path()
            )
            .is_err(),
            "a changed B row was assessed"
        );
        assert_eq!(counters.physical_fits(), fits);
        assert!(!dir.path().join(ASSESSMENT_CLAIM_FILE).exists(), "a refused input took the claim");
        assert_eq!(run.frozen.physical_fits_at_freeze, fits);
        assert!(
            assess_frozen_winner(
                source(&x, &y, &a, &b, &o, &proof, &admissions),
                counters.clone(),
                dir.path()
            )
            .is_ok(),
            "the correct inputs were refused after the refusal"
        );
    }

    #[test]
    fn a_shared_ceiling_refuses_the_assessment_fit_and_keeps_the_claim() {
        let (x, y) = fixture();
        let (a, b, o) = partitions();
        let proof = honest_proof(&x, &y, &a, &b);
        let admissions = admitted(&x, &y, &a, &b);
        let search_fits = search_fits();

        let dir = tempfile::tempdir().unwrap();
        let counters = Arc::new(ExecutionBoundaryCounters::new(ExecutionBoundaryCaps {
            max_physical_fits: search_fits,
            ..ExecutionBoundaryCaps::reasoner_natural_eos()
        }));
        run_native_search_study_arm(
            source(&x, &y, &a, &b, &o, &proof, &admissions),
            budget(),
            5,
            counters.clone(),
            dir.path(),
        )
        .unwrap();
        assert!(
            assess_frozen_winner(
                source(&x, &y, &a, &b, &o, &proof, &admissions),
                counters.clone(),
                dir.path()
            )
            .is_err(),
            "the assessment fit ran past the shared ceiling"
        );
        assert_eq!(counters.physical_fits(), search_fits, "a refused fit was counted");
        assert!(dir.path().join(ASSESSMENT_CLAIM_FILE).exists(), "the failed claim was released");
    }

    #[test]
    fn the_named_study_refuses_the_legacy_per_call_token_cap() {
        let (x, y) = fixture();
        let dir = tempfile::tempdir().unwrap();
        let counters = Arc::new(ExecutionBoundaryCounters::new(ExecutionBoundaryCaps::default()));
        assert!(
            run_once(&x, &y, counters.clone(), dir.path()).is_err(),
            "the named study accepted the legacy 8192 cap"
        );
        assert_eq!(counters.physical_fits(), 0);
    }

    #[test]
    fn search_arms_cannot_spend_the_reserved_assessment_fits() {
        let (x, y) = fixture();
        let (a, b, o) = partitions();
        let proof = honest_proof(&x, &y, &a, &b);
        let admissions = admitted(&x, &y, &a, &b);
        let fits = search_fits();

        let dir = tempfile::tempdir().unwrap();
        let mut ledger = NativeStudyLedger::create(
            dir.path(),
            ExecutionBoundaryCaps {
                max_physical_fits: fits + 1,
                ..ExecutionBoundaryCaps::reasoner_natural_eos()
            },
            1,
            &["arm-a"],
        )
        .unwrap();
        let run = ledger
            .run_arm("arm-a", source(&x, &y, &a, &b, &o, &proof, &admissions), budget(), 5)
            .unwrap();
        assert_eq!(run.frozen.physical_fits_at_freeze, fits);
        ledger
            .assess_arm("arm-a", source(&x, &y, &a, &b, &o, &proof, &admissions))
            .unwrap();
        assert_eq!(ledger.counters().physical_fits(), fits + 1);

        let dir = tempfile::tempdir().unwrap();
        let mut tight = NativeStudyLedger::create(
            dir.path(),
            ExecutionBoundaryCaps {
                max_physical_fits: fits,
                ..ExecutionBoundaryCaps::reasoner_natural_eos()
            },
            1,
            &["arm-a"],
        )
        .unwrap();
        let outcome = tight.run_arm("arm-a", source(&x, &y, &a, &b, &o, &proof, &admissions), budget(), 5);
        assert_eq!(
            tight.counters().physical_fits(),
            fits - 1,
            "the search dipped into the reserved assessment fit"
        );
        if let Ok(run) = outcome {
            assert!(
                run.search.failed_or_unavailable_measurements > 0,
                "the search met the ceiling without recording a refusal"
            );
            assert!(run.frozen.physical_fits_at_freeze < fits);
        }
    }

    #[test]
    fn assessment_waits_for_every_expected_arm_to_freeze() {
        let (x, y) = fixture();
        let (a, b, o) = partitions();
        let proof = honest_proof(&x, &y, &a, &b);
        let admissions = admitted(&x, &y, &a, &b);
        let dir = tempfile::tempdir().unwrap();
        let mut ledger = NativeStudyLedger::create(
            dir.path(),
            ExecutionBoundaryCaps::reasoner_natural_eos(),
            2,
            &["arm-a", "arm-b"],
        )
        .unwrap();
        ledger
            .run_arm("arm-a", source(&x, &y, &a, &b, &o, &proof, &admissions), budget(), 5)
            .unwrap();
        let before = ledger.counters().physical_fits();
        assert!(
            ledger
                .assess_arm("arm-a", source(&x, &y, &a, &b, &o, &proof, &admissions))
                .is_err(),
            "an assessment ran before arm-b froze"
        );
        assert_eq!(ledger.counters().physical_fits(), before, "a refused assessment fitted");

        ledger
            .run_arm("arm-b", source(&x, &y, &a, &b, &o, &proof, &admissions), budget(), 6)
            .unwrap();
        ledger
            .assess_arm("arm-a", source(&x, &y, &a, &b, &o, &proof, &admissions))
            .unwrap();
        assert!(
            ledger
                .run_arm("arm-a", source(&x, &y, &a, &b, &o, &proof, &admissions), budget(), 5)
                .is_err(),
            "an admitted arm ran twice"
        );
        assert!(
            ledger
                .run_arm("arm-c", source(&x, &y, &a, &b, &o, &proof, &admissions), budget(), 7)
                .is_err(),
            "an arm outside the study was admitted"
        );
    }

    #[test]
    fn assessment_refuses_a_tampered_frozen_record_and_accepts_the_original() {
        let (x, y) = fixture();
        let (a, b, o) = partitions();
        let proof = honest_proof(&x, &y, &a, &b);
        let admissions = admitted(&x, &y, &a, &b);
        let dir = tempfile::tempdir().unwrap();
        let mut ledger = NativeStudyLedger::create(
            dir.path(),
            ExecutionBoundaryCaps::reasoner_natural_eos(),
            2,
            &["arm-a", "arm-b"],
        )
        .unwrap();
        ledger
            .run_arm("arm-a", source(&x, &y, &a, &b, &o, &proof, &admissions), budget(), 5)
            .unwrap();
        ledger
            .run_arm("arm-b", source(&x, &y, &a, &b, &o, &proof, &admissions), budget(), 6)
            .unwrap();
        let frozen_path = ledger.arm_dir("arm-a").join(FROZEN_WINNER_FILE);
        let original = std::fs::read(&frozen_path).unwrap();
        let mut record: FrozenNativeStudyWinner = serde_json::from_slice(&original).unwrap();
        record.cut_input_digest = "0".repeat(64);
        std::fs::write(&frozen_path, serde_json::to_vec_pretty(&record).unwrap()).unwrap();
        assert!(
            ledger
                .assess_arm("arm-a", source(&x, &y, &a, &b, &o, &proof, &admissions))
                .is_err(),
            "a tampered frozen record passed the barrier"
        );
        std::fs::write(&frozen_path, &original).unwrap();
        assert!(
            ledger
                .assess_arm("arm-a", source(&x, &y, &a, &b, &o, &proof, &admissions))
                .is_ok(),
            "the original frozen record was refused after the restore"
        );
    }

    #[test]
    fn create_rejects_a_reservation_below_the_arm_count_and_unsafe_arm_names() {
        let caps = ExecutionBoundaryCaps::reasoner_natural_eos();
        let dir = tempfile::tempdir().unwrap();
        assert!(
            NativeStudyLedger::create(&dir.path().join("s1"), caps, 1, &["a", "b"]).is_err(),
            "a reservation of 1 was accepted for 2 arms"
        );
        assert!(
            NativeStudyLedger::create(&dir.path().join("s2"), caps, 2, &["a", "a"]).is_err(),
            "a duplicate arm name was accepted"
        );
        assert!(
            NativeStudyLedger::create(&dir.path().join("s3"), caps, 1, &["a/b"]).is_err(),
            "a path separator in an arm name was accepted"
        );
        assert!(
            NativeStudyLedger::create(&dir.path().join("s4"), caps, 1, &[""]).is_err(),
            "an empty arm name was accepted"
        );
        assert!(
            NativeStudyLedger::create(&dir.path().join("s5"), caps, 2, &["a", "b"]).is_ok(),
            "a reservation equal to the arm count was refused"
        );
    }

    #[test]
    fn reopen_refuses_a_second_owner_and_resumes_the_counters() {
        let dir = tempfile::tempdir().unwrap();
        let mut ledger = NativeStudyLedger::create(
            dir.path(),
            ExecutionBoundaryCaps::reasoner_natural_eos(),
            1,
            &["arm-a"],
        )
        .unwrap();
        ledger.charge_external_model_call_before_generate("call-1").unwrap();
        ledger.complete_external_model_call_after_eos("call-1", 8_193).unwrap();
        let second = NativeStudyLedger::open(dir.path());
        assert!(second.is_err(), "a second owner opened the study");
        assert!(second.err().unwrap().to_string().contains("already owned"));
        drop(ledger);

        let reopened = NativeStudyLedger::open(dir.path()).unwrap();
        assert_eq!(reopened.counters().model_calls(), 1, "the call count was not resumed");
        assert_eq!(reopened.counters().generated_tokens_total(), 8_193, "the token total was not resumed");
    }

    #[test]
    fn reopen_halts_on_an_arm_that_was_admitted_but_never_froze() {
        let (x, y) = fixture();
        let (a, b, o) = partitions();
        let proof = honest_proof(&x, &y, &a, &b);
        let admissions = admitted(&x, &y, &a, &b);
        let dir = tempfile::tempdir().unwrap();
        let mut ledger = NativeStudyLedger::create(
            dir.path(),
            ExecutionBoundaryCaps {
                max_physical_fits: 1,
                ..ExecutionBoundaryCaps::reasoner_natural_eos()
            },
            1,
            &["arm-a"],
        )
        .unwrap();
        assert!(
            ledger
                .run_arm("arm-a", source(&x, &y, &a, &b, &o, &proof, &admissions), budget(), 5)
                .is_err(),
            "a search with no reservable fit reached a winner"
        );
        assert!(dir.path().join(FAILURE_RECEIPTS_FILE).exists(), "the failure was not receipted");
        drop(ledger);
        let error = NativeStudyLedger::open(dir.path()).err().unwrap().to_string();
        assert!(error.contains("never froze"), "{error}");
    }

    #[test]
    fn reopen_halts_on_an_external_call_charged_with_no_completion() {
        let dir = tempfile::tempdir().unwrap();
        let mut ledger = NativeStudyLedger::create(
            dir.path(),
            ExecutionBoundaryCaps::reasoner_natural_eos(),
            1,
            &["arm-a"],
        )
        .unwrap();
        ledger.charge_external_model_call_before_generate("call-1").unwrap();
        drop(ledger);
        let error = NativeStudyLedger::open(dir.path()).err().unwrap().to_string();
        assert!(error.contains("no completion"), "{error}");
    }

    #[test]
    fn external_model_calls_are_charged_before_generation_and_completed_once() {
        let dir = tempfile::tempdir().unwrap();
        let mut ledger = NativeStudyLedger::create(
            dir.path(),
            ExecutionBoundaryCaps::reasoner_natural_eos(),
            1,
            &["arm-a"],
        )
        .unwrap();
        assert!(
            ledger
                .complete_external_model_call_after_eos("call-1", 10)
                .is_err(),
            "a post-hoc completion was accepted"
        );
        assert_eq!(ledger.counters().model_calls(), 0);
        ledger.charge_external_model_call_before_generate("call-1").unwrap();
        assert!(
            ledger.charge_external_model_call_before_generate("call-1").is_err(),
            "a call was charged twice"
        );
        ledger.complete_external_model_call_after_eos("call-1", 8_193).unwrap();
        assert!(
            ledger.complete_external_model_call_after_eos("call-1", 1).is_err(),
            "a call completed twice"
        );
        assert_eq!(ledger.counters().model_calls(), 1);
        assert_eq!(ledger.counters().generated_tokens_total(), 8_193);

        let persisted: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.path().join(STUDY_LEDGER_FILE)).unwrap())
                .unwrap();
        let calls = persisted["external_calls"].as_array().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["state"], "completed");
        assert_eq!(calls[0]["generated_tokens"], 8_193);
    }

    #[test]
    fn external_calls_under_natural_eos_accept_beyond_8192_and_are_counted() {
        let dir = tempfile::tempdir().unwrap();
        let mut ledger = NativeStudyLedger::create(
            dir.path(),
            ExecutionBoundaryCaps::reasoner_natural_eos(),
            1,
            &["arm-a"],
        )
        .unwrap();
        ledger.charge_external_model_call_before_generate("call-1").unwrap();
        ledger.complete_external_model_call_after_eos("call-1", 8_193).unwrap();
        ledger.charge_external_model_call_before_generate("call-2").unwrap();
        ledger.complete_external_model_call_after_eos("call-2", 24_000).unwrap();
        let counters = ledger.counters();
        assert_eq!(counters.generated_tokens_total(), 32_193);
        assert_eq!(counters.model_calls(), 2);
        assert_eq!(counters.max_call_tokens_observed(), 24_000);
    }

    #[test]
    fn an_external_call_refused_at_a_ceiling_is_never_recorded() {
        let dir = tempfile::tempdir().unwrap();
        let mut ledger = NativeStudyLedger::create(
            dir.path(),
            ExecutionBoundaryCaps {
                max_model_calls: 1,
                ..ExecutionBoundaryCaps::reasoner_natural_eos()
            },
            1,
            &["arm-a"],
        )
        .unwrap();
        ledger.charge_external_model_call_before_generate("call-1").unwrap();
        assert!(
            ledger.charge_external_model_call_before_generate("call-2").is_err(),
            "a call past the call ceiling was charged"
        );
        let persisted: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.path().join(STUDY_LEDGER_FILE)).unwrap())
                .unwrap();
        assert_eq!(persisted["external_calls"].as_array().unwrap().len(), 1);

        let dir = tempfile::tempdir().unwrap();
        let mut capped = NativeStudyLedger::create(
            dir.path(),
            ExecutionBoundaryCaps {
                max_generated_tokens_total: Some(100),
                ..ExecutionBoundaryCaps::reasoner_natural_eos()
            },
            1,
            &["arm-a"],
        )
        .unwrap();
        capped.charge_external_model_call_before_generate("call-1").unwrap();
        capped.complete_external_model_call_after_eos("call-1", 60).unwrap();
        capped.charge_external_model_call_before_generate("call-2").unwrap();
        assert!(
            capped.complete_external_model_call_after_eos("call-2", 50).is_err(),
            "tokens past the aggregate ceiling were recorded"
        );
        assert_eq!(capped.counters().generated_tokens_total(), 60);
    }

    // S9. These fixtures inject a failed directory sync into this test's own temporary directory. They
    // do not simulate a power failure, and they do not write a crash image. The claim is what a
    // real crash would have to preserve, so the tests check the order of steps and the refusals that
    // follow an injected failure.

    #[test]
    fn the_freeze_and_the_claim_sync_their_directories_in_order_before_any_o_read() {
        let (x, y) = fixture();
        let (a, b, o) = partitions();
        let proof = honest_proof(&x, &y, &a, &b);
        let admissions = admitted(&x, &y, &a, &b);
        let dir = tempfile::tempdir().unwrap();

        durability_hooks::clear();
        run_once(&x, &y, natural_counters(), dir.path()).unwrap();
        assert_eq!(
            durability_hooks::events(),
            vec!["frozen_created", "frozen_file_synced", "frozen_dir_synced"],
            "the freeze was not made durable in order"
        );

        durability_hooks::clear();
        assess_frozen_winner(
            source(&x, &y, &a, &b, &o, &proof, &admissions),
            natural_counters(),
            dir.path(),
        )
        .unwrap();
        assert_eq!(
            durability_hooks::events(),
            vec!["claim_created", "claim_file_synced", "claim_dir_synced"],
            "the claim was not made durable in order before the O read"
        );
    }

    #[test]
    fn a_failed_claim_directory_sync_stops_before_any_o_read_and_keeps_the_claim() {
        let (x, y) = fixture();
        let (a, b, o) = partitions();
        let proof = honest_proof(&x, &y, &a, &b);
        let admissions = admitted(&x, &y, &a, &b);
        let dir = tempfile::tempdir().unwrap();
        run_once(&x, &y, natural_counters(), dir.path()).unwrap();

        let counters = natural_counters();
        let fits_before = counters.physical_fits();
        durability_hooks::clear();
        durability_hooks::fail_next_dir_sync();
        let error = assess_frozen_winner(
            source(&x, &y, &a, &b, &o, &proof, &admissions),
            counters.clone(),
            dir.path(),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("injected directory sync failure"), "{error}");
        assert_eq!(
            durability_hooks::events(),
            vec!["claim_created", "claim_file_synced"],
            "the evaluator ran after a failed claim sync"
        );
        assert_eq!(counters.physical_fits(), fits_before, "a fit ran past the failed sync");
        assert!(
            dir.path().join(ASSESSMENT_CLAIM_FILE).exists(),
            "the claim was not kept after a failed sync"
        );

        durability_hooks::clear();
        assert!(
            assess_frozen_winner(
                source(&x, &y, &a, &b, &o, &proof, &admissions),
                counters.clone(),
                dir.path()
            )
            .is_err(),
            "a second assessment ran after the claim was created"
        );
        assert!(durability_hooks::events().is_empty(), "a refused retry reached the claim path");
        assert_eq!(counters.physical_fits(), fits_before);
    }

    #[test]
    fn a_failed_freeze_directory_sync_is_not_acknowledged() {
        let (x, y) = fixture();
        let (a, b, o) = partitions();
        let proof = honest_proof(&x, &y, &a, &b);
        let admissions = admitted(&x, &y, &a, &b);
        let dir = tempfile::tempdir().unwrap();

        durability_hooks::clear();
        durability_hooks::fail_next_dir_sync();
        let outcome = run_native_search_study_arm(
            source(&x, &y, &a, &b, &o, &proof, &admissions),
            budget(),
            5,
            natural_counters(),
            dir.path(),
        );
        assert!(outcome.is_err(), "a freeze with an unsynced name was acknowledged");
        assert_eq!(
            durability_hooks::events(),
            vec!["frozen_created", "frozen_file_synced"],
            "the freeze continued past a failed directory sync"
        );
    }

    #[test]
    fn a_failed_freeze_sync_leaves_the_arm_admitted_so_the_study_halts_on_reopen() {
        let (x, y) = fixture();
        let (a, b, o) = partitions();
        let proof = honest_proof(&x, &y, &a, &b);
        let admissions = admitted(&x, &y, &a, &b);
        let dir = tempfile::tempdir().unwrap();
        let mut ledger = NativeStudyLedger::create(
            dir.path(),
            ExecutionBoundaryCaps::reasoner_natural_eos(),
            1,
            &["arm-a"],
        )
        .unwrap();
        durability_hooks::fail_next_dir_sync();
        assert!(
            ledger
                .run_arm("arm-a", source(&x, &y, &a, &b, &o, &proof, &admissions), budget(), 5)
                .is_err(),
            "the arm was marked frozen after a failed freeze sync"
        );
        drop(ledger);
        let error = NativeStudyLedger::open(dir.path()).err().unwrap().to_string();
        assert!(error.contains("never froze"), "{error}");
    }

    #[test]
    fn a_failed_claim_sync_leaves_a_claim_that_halts_the_study_on_reopen() {
        let (x, y) = fixture();
        let (a, b, o) = partitions();
        let proof = honest_proof(&x, &y, &a, &b);
        let admissions = admitted(&x, &y, &a, &b);
        let dir = tempfile::tempdir().unwrap();
        let mut ledger = NativeStudyLedger::create(
            dir.path(),
            ExecutionBoundaryCaps::reasoner_natural_eos(),
            1,
            &["arm-a"],
        )
        .unwrap();
        ledger
            .run_arm("arm-a", source(&x, &y, &a, &b, &o, &proof, &admissions), budget(), 5)
            .unwrap();
        durability_hooks::fail_next_dir_sync();
        assert!(
            ledger
                .assess_arm("arm-a", source(&x, &y, &a, &b, &o, &proof, &admissions))
                .is_err(),
            "the assessment completed after a failed claim sync"
        );
        drop(ledger);
        let error = NativeStudyLedger::open(dir.path()).err().unwrap().to_string();
        assert!(error.contains("assessment claim"), "{error}");
    }
}
