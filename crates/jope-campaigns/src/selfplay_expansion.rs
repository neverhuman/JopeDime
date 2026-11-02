//! Rust-only expanded dataset cache and append-only catalog implementation.
//!
//! The authoritative candidate tree is always compared with its local mirror.
//! Cache writes are locked, staged, content-verified, and atomically renamed.
//! Missing train encodings fail closed unless an explicit native Rust encoder is
//! supplied.  No command in this module invokes Python.

mod prelude {
    pub(super) use crate::paths;
    pub(super) use crate::recovery_contracts::V1EncodingQuarantineV1;
    pub(super) use crate::selfplay_data::{
        canonical_json_sha256, source_family, CacheIdentity, CatalogEntryIdentity, CatalogIdentity,
        DatasetIdentity,
    };
    pub(super) use crate::selfplay_runtime::{
        atomic_json, now_ms, read_json, sha256_bytes, sha256_file,
    };
    pub(super) use anyhow::{bail, ensure, Context, Result};
    pub(super) use serde::{Deserialize, Serialize};
    pub(super) use serde_json::{json, Value};
    pub(super) use sha2::{Digest, Sha256};
    pub(super) use std::collections::{BTreeMap, HashSet};
    pub(super) use std::fs::{self, File, OpenOptions};
    pub(super) use std::io::{BufRead, BufReader, Read, Write};
    pub(super) use std::os::fd::AsRawFd;
    pub(super) use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
    pub(super) use std::path::{Path, PathBuf};
    pub(super) use std::process::Command;
    pub(super) use zip::ZipArchive;
}

pub(super) mod activation;
pub(super) mod cache;
pub(super) mod context;
pub(super) mod inventory;
pub(super) mod live;
pub(super) mod model;
pub(super) mod sealed;
pub(super) mod sidecars;
#[cfg(test)]
mod tests;

pub use activation::{
    activate_tranche, load_inventory_file, select_unactivated_cached, status,
    validate_catalog_pointer, write_inventory_file,
};
pub use cache::{materialize_cache, validate_cache};
pub use inventory::{audit_discover, cache_path, compare_mirror, discover, write_conflict_summary};
pub use live::{merge_live, SIDECAR_NAMES};
pub use model::{
    corpus_dir_name, sealed_target, AuditInventory, CacheManifest, ExpansionPaths, InventoryEntry,
    MirrorConflict, CONTEXT_FLOATS, CONTEXT_PROTOCOL, CONTEXT_STRIDE, DATASET_DIM,
    ENCODER_PROTOCOL, OPTIONAL_ENCODINGS, REQUIRED_ENCODINGS, TARGET_DIM, TASKS, TRANCHE_MAX,
};
pub use sealed::{adaptive_inventory, freeze_sealed};
pub use sidecars::install_sidecars;
