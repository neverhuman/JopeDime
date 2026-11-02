use super::*;

use super::super::index::IndexAgg;

pub(super) struct Loaded {
    pub(super) as_of_unix: u64,
    pub(super) policy: Policy,
    pub(super) policy_rec: InputRecord,
    pub(super) ledger: Ledger,
    pub(super) lookup: GroupLookup,
    pub(super) agg: IndexAgg,
    pub(super) inputs: Vec<InputRecord>,
    pub(super) refs: RefSet,
    pub(super) ref_counts: BTreeMap<String, u64>,
    pub(super) unmatched_in_scope: u64,
    pub(super) unmatched_sample: Vec<String>,
    pub(super) other_buckets: u64,
    pub(super) bucketless_resolved: u64,
    pub(super) bucketless_unresolved: u64,
    pub(super) refs_by_group: BTreeMap<usize, BTreeMap<Role, u64>>,
    pub(super) sources_by_group: BTreeMap<usize, BTreeSet<String>>,
    pub(super) pins: BTreeSet<(Role, String, String, usize)>,
    pub(super) pin_sources: BTreeMap<(Role, String, String), String>,
}

pub(super) struct Evaluated {
    pub(super) loaded: Loaded,
    pub(super) lifecycles: Lifecycles,
    pub(super) inv: Option<Inventory>,
    pub(super) rows: Vec<super::PlanRow>,
}

pub(super) struct Analysis {
    pub(super) hazards: BTreeMap<(String, String), (u64, f64)>,
    pub(super) notes: BTreeMap<(String, String), (u64, f64)>,
    pub(super) inventory_summary: Option<InventorySummary>,
    pub(super) findings: Vec<String>,
    pub(super) plan_jsonl: Vec<u8>,
    pub(super) plan_sha: String,
}
