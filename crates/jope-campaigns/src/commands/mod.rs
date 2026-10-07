pub(crate) mod binary_cv;
// frozen, hash-pinned module; bytes immutable until unfrozen
#[allow(
    dead_code,
    clippy::too_many_arguments,
    clippy::needless_borrow,
    clippy::needless_range_loop,
    clippy::manual_is_multiple_of,
    clippy::useless_vec
)]
pub(crate) mod exact_cv;
pub(crate) mod exact_cv_v2;
pub(crate) mod jain_regressor;
pub(crate) mod train_only_redteam;
pub(crate) mod train_only_study;
