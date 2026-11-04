#[path = "dashboard_gp.rs"]
mod library_gp;

mod prelude {
    pub(super) use super::library_gp;
    pub(super) use crate::{
        manifest::read_json,
        telemetry::{now, Hardware},
    };
    pub(super) use anyhow::{ensure, Context, Result};
    pub(super) use ratatui::{prelude::*, widgets::*};
    pub(super) use serde_json::Value;
    pub(super) use std::{
        collections::{BTreeSet, VecDeque},
        io::{Read, Seek, SeekFrom},
        path::Path,
        time::{Duration, Instant},
    };

    pub(super) const INK: Color = Color::Rgb(218, 229, 242);
    pub(super) const MUTED: Color = Color::Rgb(126, 149, 174);
    pub(super) const BLUE: Color = Color::Rgb(92, 205, 255);
    pub(super) const GREEN: Color = Color::Rgb(99, 224, 174);
    pub(super) const GOLD: Color = Color::Rgb(250, 197, 101);
    pub(super) const VIOLET: Color = Color::Rgb(181, 159, 255);
    pub(super) const RED: Color = Color::Rgb(255, 116, 135);
}

mod app;
mod campaign;
mod details;
mod evidence;
mod layout;
mod render;
mod tasks;

pub use app::{run, run_with_manifest};
pub use campaign::campaign;
pub use evidence::campaign_with_manifest;
pub use layout::{DashboardData, History};
pub use render::render;
