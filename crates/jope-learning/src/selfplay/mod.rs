//! Version-pinned self-play snapshot publication.

pub mod authority;
pub mod local_authority;
pub use local_authority::StateSnapshotAuthority;
pub mod pins;
pub mod snapshot;
