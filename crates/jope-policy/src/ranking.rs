//! Port of jope-dime `proposal_ranking.rs` (07304d6f): score a program pool with the frozen
//! policy's reward branch and a saved training context only.
//!
//! Selection `mean_reward_heads_desc_program_hash_v1`: order by the mean of the four reward
//! heads, descending; exact ties (including +0/-0) by program sha256 ascending, then by program
//! text. Runtime is reported and never penalises quality. Reward outputs are within-context
//! ranking logits, not calibrated RMSE estimates.
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BTreeSet;

use super::grammar::{BoundVocabulary, Vocab, BOS, EOS, PAD};
use super::model::PolicyModel;
use super::{
    context_sha256, is_accepted_state, is_sha256_hex, refuse_rejected_checkpoint,
    require_accepted_policy, require_accepted_vocabulary, sha256_hex, CHECKPOINT_VOCABULARY_SHA256,
    PRODUCT_VOCABULARY_SHA256,
};
use std::sync::Arc;

pub const SELECTION_ID: &str = "mean_reward_heads_desc_program_hash_v1";
pub const RANKING_SCHEMA: &str = "parallel.proposal_quality_ranking.v1";
pub const BACKEND_CPU: &str = "candle_cpu_float32";
pub const ORDERING: &str = "mean of four reward heads descending; exact ties by program SHA256 then text; runtime reported without a quality penalty";

/// The identities a ranking is stamped with. Fields are private: [`Bindings::new`] cannot name
/// the accepted (or rejected) state, and [`Bindings::for_accepted_policy`] takes the state from a
/// policy proven to be the accepted checkpoint and its vocabulary from a [`BoundVocabulary`] that
/// is the step-60000 vocabulary or its documented product restriction; scoring under such
/// bindings requires that very vocabulary table. Deserialization exists only for test support.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[cfg_attr(any(test, feature = "testing"), derive(Deserialize))]
#[serde(deny_unknown_fields)]
pub struct Bindings {
    checkpoint_sha256: String,
    conditioning_context_sha256: String,
    vocabulary_sha256: String,
    /// The bound vocabulary these bindings were made for; scoring must use this table.
    #[serde(skip)]
    vocabulary: Option<Arc<Vocab>>,
}

impl Bindings {
    /// Bindings for a non-accepted checkpoint state (for example an experiment or test policy).
    /// Refuses the accepted and the rejected state and any malformed digest.
    pub fn new(
        checkpoint_sha256: &str,
        conditioning_context_sha256: &str,
        vocabulary_sha256: &str,
    ) -> Result<Self> {
        ensure!(
            !is_accepted_state(checkpoint_sha256),
            "refused: only a policy loaded through PolicyModel::load_bound may claim the accepted state (use Bindings::for_accepted_policy)"
        );
        let bindings = Self::unchecked(
            checkpoint_sha256,
            conditioning_context_sha256,
            vocabulary_sha256,
        );
        bindings.check()?;
        Ok(bindings)
    }

    /// Bindings stamped with the accepted state, taken from `policy`'s proven identity, and with
    /// the vocabulary hash [`bind_vocabulary`](super::grammar::bind_vocabulary) computed, which
    /// must be the step-60000 vocabulary or its documented product restriction.
    pub fn for_accepted_policy(
        policy: &PolicyModel,
        conditioning_context_sha256: &str,
        vocabulary: &BoundVocabulary,
    ) -> Result<Self> {
        let bound = require_accepted_policy(policy)?;
        require_accepted_vocabulary(vocabulary)?;
        let bindings = Self::unchecked(
            bound.state_sha256(),
            conditioning_context_sha256,
            vocabulary.sha256(),
        )
        .with_vocabulary(vocabulary);
        bindings.check()?;
        Ok(bindings)
    }

    /// Crate-internal: the caller has already established that `checkpoint_sha256` is the state
    /// of the policy it scores with.
    pub(crate) fn unchecked(
        checkpoint_sha256: &str,
        conditioning_context_sha256: &str,
        vocabulary_sha256: &str,
    ) -> Self {
        Self {
            checkpoint_sha256: checkpoint_sha256.to_string(),
            conditioning_context_sha256: conditioning_context_sha256.to_string(),
            vocabulary_sha256: vocabulary_sha256.to_string(),
            vocabulary: None,
        }
    }

    /// Crate-internal: tie these bindings to `vocabulary` (its hash and its table).
    pub(crate) fn with_vocabulary(mut self, vocabulary: &BoundVocabulary) -> Self {
        self.vocabulary_sha256 = vocabulary.sha256().to_string();
        self.vocabulary = Some(vocabulary.shared_vocab());
        self
    }

    fn check(&self) -> Result<()> {
        for sha in [
            &self.checkpoint_sha256,
            &self.conditioning_context_sha256,
            &self.vocabulary_sha256,
        ] {
            ensure!(is_sha256_hex(sha), "invalid scoring identity hash");
        }
        refuse_rejected_checkpoint(&self.checkpoint_sha256, "")
    }

    pub fn checkpoint_sha256(&self) -> &str {
        &self.checkpoint_sha256
    }

    pub fn conditioning_context_sha256(&self) -> &str {
        &self.conditioning_context_sha256
    }

    pub fn vocabulary_sha256(&self) -> &str {
        &self.vocabulary_sha256
    }

    /// `policy` may be scored under these bindings: a claim of the accepted state needs the bound
    /// accepted model and the bound accepted vocabulary table, and a bound model is scored only
    /// under its own state.
    fn check_policy(&self, policy: &PolicyModel, vocabulary: &Vocab) -> Result<()> {
        if let Some(bound) = &self.vocabulary {
            ensure!(
                **bound == *vocabulary,
                "refused: scoring vocabulary differs from the bound vocabulary {}",
                self.vocabulary_sha256
            );
        }
        if is_accepted_state(&self.checkpoint_sha256) {
            require_accepted_policy(policy)?;
            ensure!(
                self.vocabulary.is_some()
                    && (self.vocabulary_sha256 == CHECKPOINT_VOCABULARY_SHA256
                        || self.vocabulary_sha256 == PRODUCT_VOCABULARY_SHA256),
                "refused: the accepted state needs the bound step-60000 vocabulary or its product restriction"
            );
        } else if let Some(bound) = policy.bound_identity() {
            ensure!(
                bound.state_sha256() == self.checkpoint_sha256,
                "refused: policy bound to {} scored under checkpoint {}",
                bound.state_sha256(),
                self.checkpoint_sha256
            );
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct Options {
    pub batch_size: usize,
    pub keep: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            batch_size: 64,
            keep: 64,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Prediction {
    pub program: String,
    pub program_sha256: String,
    pub token_count: usize,
    pub predicted_quality: f64,
    pub predicted_runtime_seconds: f64,
    pub reward_heads: [f64; 4],
}

/// A scored pool. Fields are private and the only production constructor is [`score_pool`];
/// deserialization exists only for test support.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[cfg_attr(any(test, feature = "testing"), derive(Deserialize))]
pub struct Ranking {
    schema: String,
    selection_id: String,
    bindings: Bindings,
    pool_sha256: String,
    pool_size: usize,
    chosen_count: usize,
    scoring_batches: usize,
    batch_size: usize,
    backend: String,
    quality_units: String,
    ordering: String,
    training_context_only: bool,
    heldout_labels_read: bool,
    gp_scores_read: bool,
    /// Sorted best first. The first `chosen_count` entries are the selected pool.
    ranked: Vec<Prediction>,
}

impl Ranking {
    pub fn selected(&self) -> &[Prediction] {
        &self.ranked[..self.chosen_count]
    }

    pub fn schema(&self) -> &str {
        &self.schema
    }

    pub fn selection_id(&self) -> &str {
        &self.selection_id
    }

    pub fn bindings(&self) -> &Bindings {
        &self.bindings
    }

    pub fn pool_sha256(&self) -> &str {
        &self.pool_sha256
    }

    pub fn pool_size(&self) -> usize {
        self.pool_size
    }

    pub fn chosen_count(&self) -> usize {
        self.chosen_count
    }

    pub fn scoring_batches(&self) -> usize {
        self.scoring_batches
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn backend(&self) -> &str {
        &self.backend
    }

    pub fn quality_units(&self) -> &str {
        &self.quality_units
    }

    pub fn ordering(&self) -> &str {
        &self.ordering
    }

    pub fn training_context_only(&self) -> bool {
        self.training_context_only
    }

    pub fn heldout_labels_read(&self) -> bool {
        self.heldout_labels_read
    }

    pub fn gp_scores_read(&self) -> bool {
        self.gp_scores_read
    }

    /// Every scored program, best first.
    pub fn ranked(&self) -> &[Prediction] {
        &self.ranked
    }
}

/// Test support (`testing`): a ranking over `predictions` in the given order, with
/// bindings from [`Bindings::new`] (so never the accepted state unless deserialized in a test,
/// which the test assembler refuses).
#[cfg(any(test, feature = "testing"))]
pub fn ranking_for_testing(bindings: Bindings, predictions: Vec<Prediction>) -> Ranking {
    Ranking {
        schema: RANKING_SCHEMA.into(),
        selection_id: SELECTION_ID.into(),
        bindings,
        pool_sha256: sha256_hex(b"test pool"),
        pool_size: predictions.len(),
        chosen_count: predictions.len(),
        scoring_batches: 1,
        batch_size: predictions.len().max(1),
        backend: "test".into(),
        quality_units: "test".into(),
        ordering: "test".into(),
        training_context_only: true,
        heldout_labels_read: false,
        gp_scores_read: false,
        ranked: predictions,
    }
}

/// The documented total order (callers must have rejected non-finite qualities).
pub fn compare_predictions(a: &Prediction, b: &Prediction) -> Ordering {
    b.predicted_quality
        .partial_cmp(&a.predicted_quality)
        .unwrap_or(Ordering::Equal)
        .then_with(|| a.program_sha256.cmp(&b.program_sha256))
        .then_with(|| a.program.cmp(&b.program))
}

/// Validate and sort predictions best first.
pub fn sorted_predictions(mut rows: Vec<Prediction>) -> Result<Vec<Prediction>> {
    ensure!(
        rows.iter().all(|p| p.predicted_quality.is_finite()
            && p.predicted_runtime_seconds.is_finite()
            && p.predicted_runtime_seconds >= 0.0
            && p.reward_heads.iter().all(|v| v.is_finite())),
        "nonfinite quality/runtime prediction"
    );
    rows.sort_by(compare_predictions);
    Ok(rows)
}

/// `pool_sha256`: sha256 of the JSON array of program hashes in ascending order.
pub fn pool_sha256(programs: &[String]) -> Result<String> {
    let mut hashes = programs
        .iter()
        .map(|p| (sha256_hex(p.as_bytes()), p.clone()))
        .collect::<Vec<_>>();
    hashes.sort();
    Ok(sha256_hex(&serde_json::to_vec(
        &hashes.iter().map(|h| &h.0).collect::<Vec<_>>(),
    )?))
}

/// jope-dime `score_pool`. There is deliberately no input for GP scores or held-out labels.
///
/// Bindings that name the accepted state are refused unless `policy` came from
/// `PolicyModel::load_bound` with the accepted state and parameter hash, and a bound policy is
/// refused under any state other than its own.
pub fn score_pool(
    policy: &PolicyModel,
    vocabulary: &Vocab,
    training_context: &[f32],
    programs: &[String],
    bindings: Bindings,
    options: &Options,
) -> Result<Ranking> {
    ensure!(
        (1..=256).contains(&options.batch_size)
            && !programs.is_empty()
            && programs.len() <= 4096
            && options.keep > 0
            && options.keep <= programs.len(),
        "invalid proposal pool, retained count or scoring batch size"
    );
    let config = policy.config();
    ensure!(
        config.dataset_dim == 1536
            && config.target_dim == 512
            && training_context.len() == 2048
            && training_context.iter().all(|v| v.is_finite()),
        "invalid saved training context"
    );
    ensure!(
        config.vocab_size == vocabulary.tokens.len(),
        "scoring vocabulary differs from policy"
    );
    bindings.check()?;
    // A ranking stamped with the accepted state must come from the bound accepted weights.
    bindings.check_policy(policy, vocabulary)?;
    ensure!(
        context_sha256(training_context) == bindings.conditioning_context_sha256,
        "scoring context bytes differ from its receipt"
    );
    let mut seen = BTreeSet::new();
    let mut encoded = Vec::with_capacity(programs.len());
    for program in programs {
        ensure!(
            seen.insert(program),
            "duplicate generated program in ranking pool"
        );
        feat_math::GenomeIr::parse(program).context("invalid generated GP program")?;
        let tokens = vocabulary
            .encode(program)
            .context("generated program lacks exact frozen tokens")?;
        ensure!(
            tokens.len() > 1
                && tokens.len() <= config.max_seq
                && tokens.first() == Some(&BOS)
                && tokens.last() == Some(&EOS),
            "generated program exceeds scoring sequence limit or lacks BOS/EOS"
        );
        encoded.push((sha256_hex(program.as_bytes()), program.clone(), tokens));
    }
    // Stable batching makes pool permutations use exactly the same arithmetic.
    encoded.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    let pool_sha256 = pool_sha256(programs)?;
    let mut predictions = Vec::with_capacity(encoded.len());
    let mut batches = 0;
    for batch in encoded.chunks(options.batch_size) {
        let length = batch.iter().map(|v| v.2.len()).max().unwrap_or(0);
        let rows = batch
            .iter()
            .map(|(_, _, tokens)| {
                let mut row = tokens.clone();
                row.resize(length, PAD);
                row
            })
            .collect::<Vec<_>>();
        let (heads, runtime) = policy.reward_runtime(training_context, &rows)?;
        ensure!(
            heads.len() == batch.len() && runtime.len() == batch.len(),
            "policy scoring head shape changed"
        );
        for (i, (sha, program, sequence)) in batch.iter().enumerate() {
            // Same reduction as the reference: float32 mean of the four heads.
            let quality = heads[i].iter().sum::<f32>() / 4.0;
            predictions.push(Prediction {
                program: program.clone(),
                program_sha256: sha.clone(),
                token_count: sequence.len(),
                predicted_quality: quality as f64,
                predicted_runtime_seconds: runtime[i] as f64,
                reward_heads: heads[i].map(|x| x as f64),
            });
        }
        batches += 1;
    }
    let ranked = sorted_predictions(predictions)?;
    Ok(Ranking {
        schema: RANKING_SCHEMA.into(),
        selection_id: SELECTION_ID.into(),
        bindings,
        pool_sha256,
        pool_size: programs.len(),
        chosen_count: options.keep,
        scoring_batches: batches,
        batch_size: options.batch_size,
        backend: if policy.device().is_cuda() {
            "candle_cuda_float32"
        } else {
            BACKEND_CPU
        }
        .into(),
        quality_units: "within_context_ranking_logit_not_calibrated_rmse".into(),
        ordering: ORDERING.into(),
        training_context_only: true,
        heldout_labels_read: false,
        gp_scores_read: false,
        ranked,
    })
}
