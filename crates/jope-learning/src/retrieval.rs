//! Frozen, source-separated retrieval selection. Archive loading belongs to the caller; this
//! module accepts an immutable candidate snapshot and returns materialized seed data.
use anyhow::{ensure, Result};
use jope_pipeline::{
    seeds::{SeedAncestryV2, SeedDrawV2, SeedSetV2, SeedSourceV2, SEEDS_V2_SCHEMA_ID},
    sha256_hex,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `Candidate` is a record for the candidate contract in this module: Frozen, source-separated retrieval selection. Archive loading belongs to the caller; this module accepts an immutable candidate snapshot and returns materialized seed data.
pub struct Candidate {
    /// Identifier for the associated candidate.
    pub candidate_id: String,
    /// Canonical program or genome representation.
    pub genome: String,
    /// SHA-256 digest of the canonical program source.
    pub program_sha256: String,
    /// Collection of parent families values carried by this candidate.
    pub parent_families: Vec<String>,
    /// Collection of dataset families values carried by this candidate.
    pub dataset_families: Vec<String>,
    /// Collection of context values carried by this candidate.
    pub context: Vec<f64>,
    /// Data or authority role assigned to this value.
    pub role: String,
    /// SHA-256 digest identifying the associated role manifest.
    pub role_manifest_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `RetrievalRequest` is a record for the retrieval request contract in this module: Frozen, source-separated retrieval selection. Archive loading belongs to the caller; this module accepts an immutable candidate snapshot and returns materialized seed data.
pub struct RetrievalRequest {
    /// Identifier for the associated request.
    pub request_id: String,
    /// Collection of own parent families values carried by this retrieval request.
    pub own_parent_families: Vec<String>,
    /// Collection of own dataset families values carried by this retrieval request.
    pub own_dataset_families: Vec<String>,
    /// SHA-256 digest identifying the associated own program.
    pub own_program_sha256: Vec<String>,
    /// Digest of the caller-verified frozen role roster.
    pub role_manifest_sha256: String,
    /// Collection of context values carried by this retrieval request.
    pub context: Vec<f64>,
    /// Collection of eligible support values carried by this retrieval request.
    pub eligible_support: Vec<String>,
    /// Cap recorded for this retrieval request.
    pub cap: usize,
    /// Requested by source recorded for this retrieval request.
    pub requested_by_source: std::collections::BTreeMap<SeedSourceV2, usize>,
    /// SHA-256 digest identifying the associated snapshot.
    pub snapshot_sha256: String,
    /// Cold cost ms recorded for this retrieval request.
    pub cold_cost_ms: u64,
    /// Warm cost ms recorded for this retrieval request.
    pub warm_cost_ms: u64,
    /// Cache warmup ms recorded for this retrieval request.
    pub cache_warmup_ms: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
/// `RetrievalMode` is an enum for the retrieval mode contract in this module: Frozen, source-separated retrieval selection. Archive loading belongs to the caller; this module accepts an immutable candidate snapshot and returns materialized seed data.
pub enum RetrievalMode {
    /// The `Cold` case of `RetrievalMode`.
    Cold,
    /// The `Warm` case of `RetrievalMode`.
    Warm,
    /// The `ShuffledContext` case of `RetrievalMode`.
    ShuffledContext,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `RetrievalResult` is a record for the retrieval result contract in this module: Frozen, source-separated retrieval selection. Archive loading belongs to the caller; this module accepts an immutable candidate snapshot and returns materialized seed data.
pub struct RetrievalResult {
    /// Mode recorded for this retrieval result.
    pub mode: RetrievalMode,
    /// Seeds recorded for this retrieval result.
    pub seeds: SeedSetV2,
    /// SHA-256 digest identifying the associated eligible support.
    pub eligible_support_sha256: String,
    /// Resource cost ms recorded for this retrieval result.
    pub resource_cost_ms: u64,
    /// Cache warmup ms recorded for this retrieval result.
    pub cache_warmup_ms: u64,
    /// Collection of selected candidate IDs values carried by this retrieval result.
    pub selected_candidate_ids: Vec<String>,
    /// Candidate counts recorded for this retrieval result.
    pub candidate_counts: CandidateCounts,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `CandidateCounts` is a record for the candidate counts contract in this module: Frozen, source-separated retrieval selection. Archive loading belongs to the caller; this module accepts an immutable candidate snapshot and returns materialized seed data.
pub struct CandidateCounts {
    /// Offered recorded for this candidate counts.
    pub offered: usize,
    /// Outside frozen support recorded for this candidate counts.
    pub outside_frozen_support: usize,
    /// Non train role recorded for this candidate counts.
    pub non_train_role: usize,
    /// Missing ancestry recorded for this candidate counts.
    pub missing_ancestry: usize,
    /// Own lineage recorded for this candidate counts.
    pub own_lineage: usize,
    /// Eligible recorded for this candidate counts.
    pub eligible: usize,
}

fn validate_request(request: &RetrievalRequest) -> Result<()> {
    ensure!(!request.request_id.is_empty(), "request identity required");
    ensure!(request.cap > 0, "explicit positive seed cap required");
    ensure!(
        !request.context.is_empty() && request.context.iter().all(|x| x.is_finite()),
        "finite context required"
    );
    ensure!(
        is_hex64(&request.snapshot_sha256),
        "snapshot digest required"
    );
    ensure!(
        is_hex64(&request.role_manifest_sha256),
        "role manifest digest required"
    );
    ensure!(
        !request.eligible_support.is_empty(),
        "eligible support required"
    );
    let mut ids = BTreeSet::new();
    ensure!(
        request
            .eligible_support
            .iter()
            .all(|id| !id.is_empty() && ids.insert(id)),
        "eligible support ids must be unique"
    );
    Ok(())
}

/// Select only complete, TRAIN, ancestry-disjoint candidates. Candidate IDs must be in the
/// caller-frozen support roster, so cold/warm/shuffled arms share the same eligible support.
pub fn materialize(
    request: &RetrievalRequest,
    candidates: &[Candidate],
    mode: RetrievalMode,
) -> Result<RetrievalResult> {
    validate_request(request)?;
    let support: BTreeSet<_> = request.eligible_support.iter().collect();
    let own_parents: BTreeSet<_> = request.own_parent_families.iter().collect();
    let own_datasets: BTreeSet<_> = request.own_dataset_families.iter().collect();
    let own_programs: BTreeSet<_> = request.own_program_sha256.iter().collect();
    let mut eligible = Vec::new();
    let mut counts = CandidateCounts {
        offered: candidates.len(),
        ..CandidateCounts::default()
    };
    for candidate in candidates {
        if !support.contains(&candidate.candidate_id) {
            counts.outside_frozen_support += 1;
            continue;
        }
        if candidate.role != "train" {
            counts.non_train_role += 1;
            continue;
        }
        ensure!(
            candidate.role_manifest_sha256 == request.role_manifest_sha256,
            "candidate role proof is from a different manifest"
        );
        ensure!(
            candidate.program_sha256 == sha256_hex(candidate.genome.as_bytes()),
            "candidate {} digest mismatch",
            candidate.candidate_id
        );
        ensure!(
            !candidate.context.is_empty()
                && candidate.context.len() == request.context.len()
                && candidate.context.iter().all(|x| x.is_finite()),
            "candidate context incomplete"
        );
        // Missing ancestry cannot be inferred from names or bank keys: require both nonempty.
        if candidate.parent_families.is_empty() || candidate.dataset_families.is_empty() {
            counts.missing_ancestry += 1;
            continue;
        }
        if candidate
            .parent_families
            .iter()
            .any(|x| own_parents.contains(x))
            || candidate
                .dataset_families
                .iter()
                .any(|x| own_datasets.contains(x))
            || own_programs.contains(&candidate.program_sha256)
        {
            counts.own_lineage += 1;
            continue;
        }
        eligible.push(candidate);
    }
    counts.eligible = eligible.len();
    let eligible_ids: BTreeSet<_> = eligible.iter().map(|c| c.candidate_id.as_str()).collect();
    let support_digest = sha256_hex(serde_json::to_vec(&eligible_ids)?.as_slice());
    let query = match mode {
        RetrievalMode::ShuffledContext if request.context.len() > 1 => request
            .context
            .iter()
            .skip(1)
            .chain(request.context.iter().take(1))
            .copied()
            .collect::<Vec<_>>(),
        _ => request.context.clone(),
    };
    eligible.sort_by(|a, b| {
        distance(&a.context, &query)
            .total_cmp(&distance(&b.context, &query))
            .then_with(|| a.candidate_id.cmp(&b.candidate_id))
    });
    let take = request.cap.min(eligible.len());
    let selected = &eligible[..take];
    let draws = selected
        .iter()
        .enumerate()
        .map(|(i, c)| SeedDrawV2 {
            draw_id: format!("{}:{i}:{}", request.request_id, c.candidate_id),
            genome: c.genome.clone(),
            program_sha256: c.program_sha256.clone(),
            source: SeedSourceV2::OtherLineageRetrieval,
            ancestry: Some(SeedAncestryV2 {
                parent_families: c.parent_families.clone(),
                dataset_families: c.dataset_families.clone(),
                canonical_program_sha256: vec![c.program_sha256.clone()],
            }),
        })
        .collect();
    let seeds = SeedSetV2 {
        schema: SEEDS_V2_SCHEMA_ID.into(),
        source_snapshot_sha256: request.snapshot_sha256.clone(),
        cap: request.cap,
        requested_by_source: request.requested_by_source.clone(),
        draws,
    };
    seeds.check()?;
    let warm = mode == RetrievalMode::Warm;
    let cost = if warm {
        request.warm_cost_ms
    } else {
        request.cold_cost_ms
    };
    Ok(RetrievalResult {
        mode,
        seeds,
        eligible_support_sha256: support_digest,
        resource_cost_ms: cost + if warm { request.cache_warmup_ms } else { 0 },
        cache_warmup_ms: if warm { request.cache_warmup_ms } else { 0 },
        selected_candidate_ids: selected.iter().map(|c| c.candidate_id.clone()).collect(),
        candidate_counts: counts,
    })
}

fn distance(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
}

fn is_hex64(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit())
}
