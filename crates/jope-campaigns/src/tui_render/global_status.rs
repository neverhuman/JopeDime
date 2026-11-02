#[path = "global_status/fleet.rs"]
mod fleet;
#[path = "global_status/learning.rs"]
mod learning;

pub(crate) use fleet::{
    cohort_label, render_alerts, render_events, render_global, render_header, render_nodes,
    render_queue, render_wave,
};
pub(crate) use learning::{render_dl_global, render_dl_status};
