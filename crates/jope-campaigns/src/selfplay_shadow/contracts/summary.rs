use super::*;

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CacheDispositionCounts {
    pub novel: usize,
    pub exact_cache_hit: usize,
    pub active_or_reserved: usize,
    pub cross_protocol_blocked: usize,
    pub requalification: usize,
    pub evaluated: usize,
}

impl CacheDispositionCounts {
    pub fn record(&mut self, disposition: CacheDisposition) {
        match disposition {
            CacheDisposition::Novel => self.novel += 1,
            CacheDisposition::ExactCacheHit => self.exact_cache_hit += 1,
            CacheDisposition::ActiveOrReserved => self.active_or_reserved += 1,
            CacheDisposition::CrossProtocolBlocked => self.cross_protocol_blocked += 1,
            CacheDisposition::Requalification => self.requalification += 1,
            CacheDisposition::Evaluated => self.evaluated += 1,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ContextCompositionSummary {
    pub dataset_content_sha256: Sha256Digest,
    pub nominal_d: usize,
    pub actual_d: usize,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceEligibility {
    pub reward_and_stability: bool,
    pub matched_dl_efficiency: bool,
    pub promotion: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WaveSummaryV1 {
    pub schema: String,
    pub wave_id: String,
    pub task: Task,
    pub nominal_d: usize,
    pub actual_d_total: usize,
    pub native_direct_count: usize,
    pub matched_gp_count: usize,
    pub context_composition: Vec<ContextCompositionSummary>,
    pub cache_dispositions: CacheDispositionCounts,
    pub zero_cross_context_backfill: bool,
    pub recovery_accounting_only: bool,
    pub evidence_eligibility: EvidenceEligibility,
}
