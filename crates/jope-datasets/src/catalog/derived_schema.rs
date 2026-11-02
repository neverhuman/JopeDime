use super::*;

pub fn catalog_schema() -> serde_json::Value {
    serde_json::from_str(CATALOG_SCHEMA_JSON).expect("committed catalog schema parses")
}

pub fn collection_schema() -> serde_json::Value {
    serde_json::from_str(COLLECTION_SCHEMA_JSON).expect("committed collection schema parses")
}

/// The schema schemars derives from [`CatalogRow`].
pub fn derived_catalog_schema() -> serde_json::Value {
    schemars::schema_for!(CatalogRow).to_value()
}

/// The schema schemars derives from [`Collection`].
pub fn derived_collection_schema() -> serde_json::Value {
    schemars::schema_for!(Collection).to_value()
}
