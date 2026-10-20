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

/// The inputs of the named native study. `a_rows` are fitted, `b_rows` select the winner, and
/// `o_rows` are read once, at assessment, after the winner is frozen. The search never receives
/// `o_rows`: its only input is the materialised A/B cut.
#[derive(Clone, Copy, Debug)]
pub struct NativeStudySource<'a> {
    pub source_kind: &'a str,
    pub role: &'a str,
    pub ntest: u64,
    pub features: &'a [Vec<f64>],
    pub targets: &'a [f64],
    pub a_rows: &'a [u32],
    pub b_rows: &'a [u32],
    pub o_rows: &'a [u32],
}

/// The winner and the counters at the freeze. Assessment reads nothing but this record.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FrozenWinner {
    pub schema: String,
    pub source_kind: String,
    pub cut_digest: String,
    pub winner: String,
    pub validation_rmse_b: Option<f64>,
    pub physical_fits_at_freeze: u64,
    pub model_calls_at_freeze: u64,
}

/// The A/B search of the named study. `search` is the actual `search::run_cuts_with` result.
pub struct NativeStudyRun {
    pub search: crate::search::CutSearch,
    pub frozen: FrozenWinner,
}

/// Refuses every role but TRAIN, every source but the TRAIN-only native table, `ntest > 0`, repeated
/// original row IDs, and any partition set the native evaluator rejects. Runs before any fit.
fn validate_native_study_source(source: &NativeStudySource<'_>) -> Result<()> {
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
    .map_err(|error| anyhow!("invalid native study partitions: {error:?}"))
}

/// SHA-256 over the A then B original row IDs, features and labels: the cut the search sees.
fn native_cut_digest(source: &NativeStudySource<'_>) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    for (tag, rows) in [(b'A', source.a_rows), (b'B', source.b_rows)] {
        hasher.update([tag]);
        for &row in rows {
            hasher.update(row.to_le_bytes());
            for value in &source.features[row as usize] {
                hasher.update(value.to_bits().to_le_bytes());
            }
            hasher.update(source.targets[row as usize].to_bits().to_le_bytes());
        }
    }
    format!("{:x}", hasher.finalize())
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
    validate_native_study_source(&source)?;
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
    let frozen = FrozenWinner {
        schema: "jope.native-study-frozen-winner/v1".into(),
        source_kind: source.source_kind.into(),
        cut_digest: native_cut_digest(&source),
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

/// The single O assessment. It needs the frozen winner on disk, checks the inputs match the frozen
/// cut, claims the assessment exclusively, and then scores the frozen winner on O once. The
/// assessment fits on A and draws on the same counters as the search. A failed or refused assessment
/// keeps its claim: there is no second attempt. Returns NaN when the winner does not fit or predict.
pub fn assess_frozen_winner(
    source: NativeStudySource<'_>,
    counters: std::sync::Arc<crate::search::ExecutionBoundaryCounters>,
    output: &std::path::Path,
) -> Result<f64> {
    validate_native_study_source(&source)?;
    let frozen: FrozenWinner = serde_json::from_slice(
        &std::fs::read(output.join(FROZEN_WINNER_FILE))
            .context("no frozen winner exists; O stays closed")?,
    )?;
    ensure!(
        frozen.source_kind == source.source_kind,
        "the frozen winner's source kind does not match the assessment source"
    );
    ensure!(
        frozen.cut_digest == native_cut_digest(&source),
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

    fn source<'a>(
        features: &'a [Vec<f64>],
        targets: &'a [f64],
        a: &'a [u32],
        b: &'a [u32],
        o: &'a [u32],
    ) -> NativeStudySource<'a> {
        NativeStudySource {
            source_kind: NATIVE_STUDY_SOURCE_KIND,
            role: "TRAIN",
            ntest: 0,
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

    #[test]
    fn native_study_refuses_held_out_roles_and_bad_sources_before_any_fit() {
        let (x, y) = fixture();
        let (a, b, o) = partitions();
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
            let mut s = source(&x, &y, &a, &b, &o);
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
    fn search_sees_only_a_and_b_and_o_is_assessed_once_after_the_freeze() {
        let (x, y) = fixture();
        let (a, b, o) = partitions();
        let dir = tempfile::tempdir().unwrap();
        let counters = natural_counters();
        let run = run_native_search_study_arm(
            source(&x, &y, &a, &b, &o),
            budget(),
            5,
            counters.clone(),
            dir.path(),
        )
        .unwrap();
        let fits = run.frozen.physical_fits_at_freeze;
        assert!(fits > 0, "the search made no physical fit");
        assert_eq!(counters.physical_fits(), fits);
        let on_disk: FrozenWinner =
            serde_json::from_slice(&std::fs::read(dir.path().join(FROZEN_WINNER_FILE)).unwrap())
                .unwrap();
        assert_eq!(on_disk, run.frozen);

        let rmse = assess_frozen_winner(source(&x, &y, &a, &b, &o), counters.clone(), dir.path())
            .unwrap();
        assert!(rmse.is_finite());
        assert_eq!(
            counters.physical_fits(),
            fits + 1,
            "the assessment is not exactly one more physical fit"
        );
        assert!(
            assess_frozen_winner(source(&x, &y, &a, &b, &o), counters.clone(), dir.path()).is_err(),
            "a second assessment was not refused"
        );
        assert_eq!(counters.physical_fits(), fits + 1, "a refused retry changed the quota");
    }

    #[test]
    fn poisoned_o_labels_leave_the_search_and_the_winner_unchanged() {
        let (x, mut y) = fixture();
        let (a, b, o) = partitions();
        let first_dir = tempfile::tempdir().unwrap();
        let first = run_native_search_study_arm(
            source(&x, &y, &a, &b, &o),
            budget(),
            5,
            natural_counters(),
            first_dir.path(),
        )
        .unwrap();
        for &row in &o {
            y[row as usize] = f64::NAN;
        }
        let second_dir = tempfile::tempdir().unwrap();
        let second = run_native_search_study_arm(
            source(&x, &y, &a, &b, &o),
            budget(),
            5,
            natural_counters(),
            second_dir.path(),
        )
        .unwrap();
        assert_eq!(first.frozen.cut_digest, second.frozen.cut_digest);
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
        let dir = tempfile::tempdir().unwrap();
        let counters = natural_counters();
        assert!(
            assess_frozen_winner(source(&x, &y, &a, &b, &o), counters.clone(), dir.path()).is_err(),
            "O opened without a frozen winner"
        );
        assert_eq!(counters.physical_fits(), 0);

        let run = run_native_search_study_arm(
            source(&x, &y, &a, &b, &o),
            budget(),
            5,
            counters.clone(),
            dir.path(),
        )
        .unwrap();
        let fits = counters.physical_fits();
        let mut changed = x.clone();
        changed[30][0] += 1.0;
        assert!(
            assess_frozen_winner(source(&changed, &y, &a, &b, &o), counters.clone(), dir.path())
                .is_err(),
            "a changed B row was assessed"
        );
        assert_eq!(counters.physical_fits(), fits);
        assert!(!dir.path().join(ASSESSMENT_CLAIM_FILE).exists(), "a refused input took the claim");
        assert_eq!(run.frozen.physical_fits_at_freeze, fits);
        assert!(
            assess_frozen_winner(source(&x, &y, &a, &b, &o), counters.clone(), dir.path()).is_ok(),
            "the correct inputs were refused after the refusal"
        );
    }

    #[test]
    fn a_shared_ceiling_refuses_the_assessment_fit_and_keeps_the_claim() {
        let (x, y) = fixture();
        let (a, b, o) = partitions();
        let probe_dir = tempfile::tempdir().unwrap();
        let probe = run_native_search_study_arm(
            source(&x, &y, &a, &b, &o),
            budget(),
            5,
            natural_counters(),
            probe_dir.path(),
        )
        .unwrap();
        let search_fits = probe.frozen.physical_fits_at_freeze;

        let dir = tempfile::tempdir().unwrap();
        let counters = Arc::new(ExecutionBoundaryCounters::new(ExecutionBoundaryCaps {
            max_physical_fits: search_fits,
            ..ExecutionBoundaryCaps::reasoner_natural_eos()
        }));
        run_native_search_study_arm(
            source(&x, &y, &a, &b, &o),
            budget(),
            5,
            counters.clone(),
            dir.path(),
        )
        .unwrap();
        assert!(
            assess_frozen_winner(source(&x, &y, &a, &b, &o), counters.clone(), dir.path()).is_err(),
            "the assessment fit ran past the shared ceiling"
        );
        assert_eq!(counters.physical_fits(), search_fits, "a refused fit was counted");
        assert!(dir.path().join(ASSESSMENT_CLAIM_FILE).exists(), "the failed claim was released");
    }

    #[test]
    fn the_named_study_refuses_the_legacy_per_call_token_cap() {
        let (x, y) = fixture();
        let (a, b, o) = partitions();
        let dir = tempfile::tempdir().unwrap();
        let counters = Arc::new(ExecutionBoundaryCounters::new(ExecutionBoundaryCaps::default()));
        assert!(
            run_native_search_study_arm(
                source(&x, &y, &a, &b, &o),
                budget(),
                5,
                counters.clone(),
                dir.path()
            )
            .is_err(),
            "the named study accepted the legacy 8192 cap"
        );
        assert_eq!(counters.physical_fits(), 0);
    }
}
