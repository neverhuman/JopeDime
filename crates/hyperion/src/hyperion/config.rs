mod types {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/hyperion/config/types.rs"
    ));
}

pub use types::{
    HyperionAugment, HyperionConfig, HyperionEngine, HyperionFeaturePlacement, HyperionMode,
    HyperionModelIdentity, HyperionQuantileBounds, HyperionState, HyperionSweepSettings,
};
use types::{ManifestFile, WeightProjection};

mod transform {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/hyperion/config/transform.rs"
    ));
}

pub use transform::{fit_transform_pair, hyperion_output_names};
