use super::*;

use super::super::super::index::IndexAgg;

pub(crate) struct Loaded {
    pub(crate) as_of_unix: u64,
    pub(crate) policy: Policy,
    pub(crate) policy_rec: InputRecord,
    pub(crate) ledger: Ledger,
    pub(crate) lookup: GroupLookup,
    pub(crate) agg: IndexAgg,
    pub(crate) inputs: Vec<InputRecord>,
    pub(crate) refs: RefSet,
    pub(crate) ref_counts: BTreeMap<String, u64>,
    pub(crate) unmatched_in_scope: u64,
    pub(crate) unmatched_sample: Vec<String>,
    pub(crate) other_buckets: u64,
    pub(crate) bucketless_resolved: u64,
    pub(crate) bucketless_unresolved: u64,
    pub(crate) refs_by_group: BTreeMap<usize, BTreeMap<Role, u64>>,
    pub(crate) sources_by_group: BTreeMap<usize, BTreeSet<String>>,
    pub(crate) pins: BTreeSet<(Role, String, String, usize)>,
    pub(crate) pin_sources: BTreeMap<(Role, String, String), String>,
}

pub(crate) struct Evaluated {
    pub(crate) loaded: Loaded,
    pub(crate) lifecycles: Lifecycles,
    pub(crate) inv: Option<Inventory>,
    pub(crate) rows: Vec<super::super::PlanRow>,
}

pub(crate) struct Analysis {
    pub(crate) hazards: BTreeMap<(String, String), (u64, f64)>,
    pub(crate) notes: BTreeMap<(String, String), (u64, f64)>,
    pub(crate) inventory_summary: Option<InventorySummary>,
    pub(crate) findings: Vec<String>,
    pub(crate) plan_jsonl: Vec<u8>,
    pub(crate) plan_sha: String,
}
