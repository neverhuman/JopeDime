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

/// A caller's role proof for the named native study. It attests the TRAIN-only role over the exact
/// rows admitted. Validation recomputes every hash from the rows it is given, and refuses any
/// mismatch before a fit. The attestation is a hash: it binds the proof to the input, but it does not
/// authenticate who made the proof.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct NativeStudyRoleProof {
    pub family_id: String,
    pub expected_cut_input_digest: String,
    pub fit_a_rows_sha256: String,
    pub hold_b_rows_sha256: String,
    pub role_attestation_sha256: String,
}

impl NativeStudyRoleProof {
    /// Attests TRAIN-only admission of `a_rows` (fit) and `b_rows` (hold) for `family_id`. Every field
    /// is computed from the given rows, so the proof verifies against exactly those rows.
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

/// SHA-256 over the role proof fields: the proof's tag, family, TRAIN role, source kind, ntest 0, and
/// the three hashes it attests.
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
/// original row IDs, and any partition set the native evaluator rejects. Then recomputes the bindings
/// from the rows and requires the caller's proof to match each one. All of this runs before any fit.
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
    let digest = compute_cut_input_digest(source.features, source.targets, source.a_rows, source.b_rows);
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
        schema: "jope.native-study-frozen-winner/v2".into(),
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
    std::io::Write::write_all(&mut file, &serde_json::to_vec_pretty(&frozen)?)?;
    Ok(NativeStudyRun { search, frozen })
}

/// The single O assessment. It validates the proof against the inputs, reads the frozen winner from
/// disk, checks that every binding matches it, claims the assessment exclusively, and then scores the
/// frozen winner on O once. The assessment fits on A and draws on the same counters as the search. A
/// failed or refused assessment keeps its claim: there is no second attempt. Returns NaN when the
/// winner does not fit or predict.
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
    std::io::Write::write_all(
        &mut claim,
        &serde_json::to_vec_pretty(&serde_json::json!({ "winner": frozen.winner }))?,
    )?;
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

/// One external model call. It is `charged` before generation and `completed` after EOS, with its
/// generated tokens. A record is never removed.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ExternalModelCall {
    pub call_id: String,
    pub state: String,
    pub generated_tokens: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
struct StudyLedgerFile {
    schema: String,
    reserved_assessment_fits: u64,
    expected_arms: Vec<String>,
    arms_admitted: Vec<String>,
    external_calls: Vec<ExternalModelCall>,
}

/// The study-wide controller. Every admitted search arm and every final O assessment of one study
/// shares one `Arc<ExecutionBoundaryCounters>`. Search arms stop short of the reserved assessment
/// fits. An assessment is refused until every expected arm has a frozen winner on disk. External
/// `E9` model calls are charged to the same counters and recorded in the ledger file before they are
/// allowed to generate.
pub struct NativeStudyLedger {
    dir: std::path::PathBuf,
    counters: std::sync::Arc<crate::search::ExecutionBoundaryCounters>,
    file: StudyLedgerFile,
}

impl NativeStudyLedger {
    /// Creates the study ledger create-only, reserving `reserved_assessment_fits` of the physical-fit
    /// ceiling for the O assessments. `expected_arms` names every arm the study will admit.
    pub fn create(
        dir: &std::path::Path,
        caps: crate::search::ExecutionBoundaryCaps,
        reserved_assessment_fits: u64,
        expected_arms: &[&str],
    ) -> Result<Self> {
        ensure!(!expected_arms.is_empty(), "a study needs at least one arm");
        ensure!(
            reserved_assessment_fits <= caps.max_physical_fits,
            "the reservation exceeds the physical-fit ceiling"
        );
        std::fs::create_dir_all(dir)?;
        let ledger = Self {
            dir: dir.to_path_buf(),
            counters: std::sync::Arc::new(
                crate::search::ExecutionBoundaryCounters::new(caps)
                    .with_reserved_assessment_fits(reserved_assessment_fits),
            ),
            file: StudyLedgerFile {
                schema: "jope.native-study-ledger/v1".into(),
                reserved_assessment_fits,
                expected_arms: expected_arms.iter().map(|arm| (*arm).to_string()).collect(),
                arms_admitted: Vec::new(),
                external_calls: Vec::new(),
            },
        };
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(dir.join(STUDY_LEDGER_FILE))
            .context("the study ledger already exists")?;
        std::io::Write::write_all(&mut file, &serde_json::to_vec_pretty(&ledger.file)?)?;
        Ok(ledger)
    }

    /// The counters every arm and assessment of this study shares.
    pub fn counters(&self) -> std::sync::Arc<crate::search::ExecutionBoundaryCounters> {
        std::sync::Arc::clone(&self.counters)
    }

    /// Where an arm writes its search output and its frozen winner.
    pub fn arm_dir(&self, arm: &str) -> std::path::PathBuf {
        self.dir.join("arms").join(arm)
    }

    /// Runs one admitted arm on the shared counters. An arm is admitted once: it is recorded as
    /// admitted before it runs, so a failed arm cannot be retried.
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
            !self.file.arms_admitted.iter().any(|admitted| admitted == arm),
            "arm {arm:?} was already admitted"
        );
        self.file.arms_admitted.push(arm.into());
        self.persist()?;
        run_native_search_study_arm(source, budget, seed, self.counters(), &self.arm_dir(arm))
    }

    /// The final O assessment of one arm. It is refused until every expected arm has a frozen winner
    /// on disk, so no search arm can still need the reserved fits when an assessment starts.
    pub fn assess_arm(&self, arm: &str, source: NativeStudySource<'_>) -> Result<f64> {
        for expected in &self.file.expected_arms {
            ensure!(
                self.arm_dir(expected).join(FROZEN_WINNER_FILE).exists(),
                "arm {expected:?} has no frozen winner yet; assessments wait for every arm"
            );
        }
        assess_frozen_winner(source, self.counters(), &self.arm_dir(arm))
    }

    /// Charges one external model call before it generates. The call counts against the call
    /// ceiling, and it is persisted as `charged` before this returns. A call id is charged once.
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
        self.persist()
    }

    /// Records the generated tokens of a charged call after its EOS. A call that was never charged, or
    /// is already completed, is refused: a post-hoc record cannot be created.
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
        self.persist()
    }

    /// Writes the whole ledger through a temporary file and a rename, so a reader sees either the old
    /// file or the new one.
    fn persist(&self) -> Result<()> {
        let temporary = self.dir.join(format!("{STUDY_LEDGER_FILE}.tmp"));
        std::fs::write(&temporary, serde_json::to_vec_pretty(&self.file)?)?;
        std::fs::rename(&temporary, self.dir.join(STUDY_LEDGER_FILE))?;
        Ok(())
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

    fn source<'a>(
        features: &'a [Vec<f64>],
        targets: &'a [f64],
        a: &'a [u32],
        b: &'a [u32],
        o: &'a [u32],
        role_proof: &'a NativeStudyRoleProof,
    ) -> NativeStudySource<'a> {
        NativeStudySource {
            source_kind: NATIVE_STUDY_SOURCE_KIND,
            role: "TRAIN",
            ntest: 0,
            role_proof,
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
        run_native_search_study_arm(source(x, y, &a, &b, &o, &proof), budget(), 5, counters, dir)
    }

    #[test]
    fn native_study_refuses_held_out_roles_and_bad_sources_before_any_fit() {
        let (x, y) = fixture();
        let (a, b, o) = partitions();
        let proof = honest_proof(&x, &y, &a, &b);
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
            let mut s = source(&x, &y, &a, &b, &o, &proof);
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
        // A proof attested over a different B set is misbound to this cut.
        let short_b: Vec<u32> = (30..40).collect();
        cases.push(("proof of a different B", honest_proof(&x, &y, &a, &short_b)));

        for (field, proof) in cases {
            let counters = natural_counters();
            let dir = tempfile::tempdir().unwrap();
            assert!(
                run_native_search_study_arm(
                    source(&x, &y, &a, &b, &o, &proof),
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

        // The materialised cut is exactly A (train) and B (hold), and holds no O row.
        let proof = honest_proof(&x, &y, &a, &b);
        let cut = materialize_native_cut(&source(&x, &y, &a, &b, &o, &proof));
        let a_x: Vec<Vec<f64>> = a.iter().map(|&r| x[r as usize].clone()).collect();
        let a_y: Vec<f64> = a.iter().map(|&r| y[r as usize]).collect();
        let b_x: Vec<Vec<f64>> = b.iter().map(|&r| x[r as usize].clone()).collect();
        let b_y: Vec<f64> = b.iter().map(|&r| y[r as usize]).collect();
        assert_eq!(cut.train_x, a_x, "train rows are not exactly A");
        assert_eq!(cut.train_y, a_y, "train labels are not exactly A");
        assert_eq!(cut.hold_x, b_x, "hold rows are not exactly B");
        assert_eq!(cut.hold_y, b_y, "hold labels are not exactly B");
        assert_eq!(cut.hold_indices, (30..45).collect::<Vec<usize>>());

        // The frozen record binds the exact A and B row hashes the proof attested.
        assert_eq!(run.frozen.fit_a_rows_sha256, proof.fit_a_rows_sha256);
        assert_eq!(run.frozen.hold_b_rows_sha256, proof.hold_b_rows_sha256);
        assert_eq!(run.frozen.role_attestation_sha256, proof.role_attestation_sha256);
        let on_disk: FrozenNativeStudyWinner =
            serde_json::from_slice(&std::fs::read(dir.path().join(FROZEN_WINNER_FILE)).unwrap())
                .unwrap();
        assert_eq!(on_disk, run.frozen);

        let rmse = assess_frozen_winner(source(&x, &y, &a, &b, &o, &proof), counters.clone(), dir.path())
            .unwrap();
        assert!(rmse.is_finite());
        assert_eq!(
            counters.physical_fits(),
            fits + 1,
            "the assessment is not exactly one more physical fit"
        );
        assert!(
            assess_frozen_winner(source(&x, &y, &a, &b, &o, &proof), counters.clone(), dir.path())
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
        let dir = tempfile::tempdir().unwrap();
        let counters = natural_counters();
        assert!(
            assess_frozen_winner(source(&x, &y, &a, &b, &o, &proof), counters.clone(), dir.path())
                .is_err(),
            "O opened without a frozen winner"
        );
        assert_eq!(counters.physical_fits(), 0);

        let run = run_once(&x, &y, counters.clone(), dir.path()).unwrap();
        let fits = counters.physical_fits();
        let mut changed = x.clone();
        changed[30][0] += 1.0;
        let changed_proof = honest_proof(&changed, &y, &a, &b);
        assert!(
            assess_frozen_winner(
                source(&changed, &y, &a, &b, &o, &changed_proof),
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
            assess_frozen_winner(source(&x, &y, &a, &b, &o, &proof), counters.clone(), dir.path())
                .is_ok(),
            "the correct inputs were refused after the refusal"
        );
    }

    #[test]
    fn a_shared_ceiling_refuses_the_assessment_fit_and_keeps_the_claim() {
        let (x, y) = fixture();
        let (a, b, o) = partitions();
        let proof = honest_proof(&x, &y, &a, &b);
        let probe_dir = tempfile::tempdir().unwrap();
        let probe = run_once(&x, &y, natural_counters(), probe_dir.path()).unwrap();
        let search_fits = probe.frozen.physical_fits_at_freeze;

        let dir = tempfile::tempdir().unwrap();
        let counters = Arc::new(ExecutionBoundaryCounters::new(ExecutionBoundaryCaps {
            max_physical_fits: search_fits,
            ..ExecutionBoundaryCaps::reasoner_natural_eos()
        }));
        run_native_search_study_arm(
            source(&x, &y, &a, &b, &o, &proof),
            budget(),
            5,
            counters.clone(),
            dir.path(),
        )
        .unwrap();
        assert!(
            assess_frozen_winner(source(&x, &y, &a, &b, &o, &proof), counters.clone(), dir.path())
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
    fn search_arms_cannot_spend_the_reserved_assessment_fits() {
        let (x, y) = fixture();
        let (a, b, o) = partitions();
        let proof = honest_proof(&x, &y, &a, &b);
        let fits = search_fits();

        // One fit reserved under a ceiling of fits + 1: the search takes `fits`, and the assessment
        // takes the last one.
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
            .run_arm("arm-a", source(&x, &y, &a, &b, &o, &proof), budget(), 5)
            .unwrap();
        assert_eq!(run.frozen.physical_fits_at_freeze, fits);
        ledger
            .assess_arm("arm-a", source(&x, &y, &a, &b, &o, &proof))
            .unwrap();
        assert_eq!(ledger.counters().physical_fits(), fits + 1);

        // Under a ceiling of exactly `fits` with the same reservation, the search stops one fit short
        // and never reaches the reserved fit.
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
        let outcome = tight.run_arm("arm-a", source(&x, &y, &a, &b, &o, &proof), budget(), 5);
        assert_eq!(
            tight.counters().physical_fits(),
            fits - 1,
            "the search dipped into the reserved assessment fit"
        );
        // The search wanted `fits` fits, so the refused candidates must appear as failed measurements.
        // Whether the run then fails or freezes a winner on the fits it had, it never passed the line.
        if let Ok(run) = outcome {
            assert!(
                run.search.failed_or_unavailable_measurements > 0,
                "the search met the ceiling without recording a refusal"
            );
            assert!(run.frozen.physical_fits_at_freeze <= fits - 1);
        }
    }

    #[test]
    fn assessment_waits_for_every_expected_arm_to_freeze() {
        let (x, y) = fixture();
        let (a, b, o) = partitions();
        let proof = honest_proof(&x, &y, &a, &b);
        let dir = tempfile::tempdir().unwrap();
        let mut ledger = NativeStudyLedger::create(
            dir.path(),
            ExecutionBoundaryCaps::reasoner_natural_eos(),
            2,
            &["arm-a", "arm-b"],
        )
        .unwrap();
        ledger
            .run_arm("arm-a", source(&x, &y, &a, &b, &o, &proof), budget(), 5)
            .unwrap();
        let before = ledger.counters().physical_fits();
        assert!(
            ledger
                .assess_arm("arm-a", source(&x, &y, &a, &b, &o, &proof))
                .is_err(),
            "an assessment ran before arm-b froze"
        );
        assert_eq!(ledger.counters().physical_fits(), before, "a refused assessment fitted");

        ledger
            .run_arm("arm-b", source(&x, &y, &a, &b, &o, &proof), budget(), 6)
            .unwrap();
        ledger
            .assess_arm("arm-a", source(&x, &y, &a, &b, &o, &proof))
            .unwrap();
        assert!(
            ledger
                .run_arm("arm-a", source(&x, &y, &a, &b, &o, &proof), budget(), 5)
                .is_err(),
            "an admitted arm ran twice"
        );
        assert!(
            ledger
                .run_arm("arm-c", source(&x, &y, &a, &b, &o, &proof), budget(), 7)
                .is_err(),
            "an arm outside the study was admitted"
        );
    }

    #[test]
    fn external_model_calls_are_charged_before_generation_and_completed_once() {
        let dir = tempfile::tempdir().unwrap();
        let mut ledger = NativeStudyLedger::create(
            dir.path(),
            ExecutionBoundaryCaps::reasoner_natural_eos(),
            0,
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
            0,
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
            0,
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
            0,
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
}
