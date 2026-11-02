#[path = "legacy_and_archive/legacy.rs"]
mod legacy;
#[path = "legacy_and_archive/smart_candidates.rs"]
mod smart_candidates;

use legacy::{legacy_refill, load_parent_archive};
use smart_candidates::{
    load_smart_candidates, merge_candidate, mixed_seed, select_with_head_cap,
    validate_initial_populations,
};

mod g20_dataset;
pub(super) use g20_dataset::run_g20_dataset;
