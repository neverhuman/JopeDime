use super::details::decimal_count;
use super::evidence::{
    age, completed, count_pair, duration, live_operations, memory_throttle_reason, metric_tail,
    metrics, number, phase, policy_receipt, state, status_age, status_age_with_limit,
};
use super::prelude::*;
use super::tasks::{campaign_task_state, label, training_hold};

mod gpu;
mod milestones;
mod phase;
mod view;

pub(crate) use gpu::{gpu_reason, stage_work_detail};
pub(crate) use milestones::milestone_lines;
pub(crate) use phase::goal_phase;
pub use view::campaign;
