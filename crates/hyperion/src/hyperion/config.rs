mod types {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/hyperion/config/types.rs"
    ));
}

use types::{
    classic_engine_wire, default_pad_u8, default_quant_scale, projection_alias_wire,
    state_engine_projection, ManifestFile, WeightProjection,
};
pub use types::{
    HyperionAugment, HyperionConfig, HyperionEngine, HyperionFeaturePlacement, HyperionMode,
    HyperionModelIdentity, HyperionQuantileBounds, HyperionState, HyperionSweepSettings,
};

mod transform {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/hyperion/config/transform.rs"
    ));
}

pub use transform::{fit_transform_pair, hyperion_output_names};
