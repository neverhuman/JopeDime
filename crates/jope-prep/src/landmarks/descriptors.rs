#[path = "descriptors/building.rs"]
mod building;
#[path = "descriptors/interaction.rs"]
mod interaction;
#[path = "descriptors/report.rs"]
mod report;
#[path = "descriptors/table.rs"]
mod table;

pub(crate) use building::building_blocks;
pub(crate) use interaction::inter_descriptors;
pub(crate) use report::report_descriptors;
pub(crate) use table::table_descriptors;
