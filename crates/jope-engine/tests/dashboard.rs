use jope_parallel::{
    dashboard::{self, History},
    telemetry::{now, Gpu, Hardware},
};
use ratatui::{backend::TestBackend, Terminal};

#[path = "dashboard_tests/corpus.rs"]
mod corpus;
#[path = "dashboard_tests/history.rs"]
mod history;
#[path = "dashboard_tests/layout.rs"]
mod layout;
#[path = "dashboard_tests/search.rs"]
mod search;
#[path = "dashboard_tests/telemetry.rs"]
mod telemetry;
