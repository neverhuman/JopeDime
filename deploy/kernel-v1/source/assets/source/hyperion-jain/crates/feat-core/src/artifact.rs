//! Versioned train/predict artifact support.
mod loaded;
mod manifest;
mod publish;
mod scoring;

pub use loaded::{
    artifact_healthy_loaded, predict_base_matrix, predict_csv_text, predict_dir, LoadedArtifact,
};
pub use manifest::{
    read_manifest, saved_fuser_from_plan, write_manifest, ModelManifest, SavedFuser, SavedMember,
    SavedModelEntry, ARTIFACT_VERSION, MANIFEST_FILE,
};
pub use publish::{artifact_staging_dir, publish_artifact};
pub use scoring::{
    materialize_selected, materialize_selected_for_domain, materialize_selected_train,
    transform_scoring_csv, transform_scoring_csv_text,
};
