//! Write synchronized Hyperion cell scores and row embeddings for a headerless train/test CSV pair.
//!
//! Each input CSV is expected to have the target in the final column. The target is dropped before
//! Hyperion sees the data. Quantile/sweep state is fit on train and replayed for test through the
//! same API the main pipeline uses. Cell-score percentiles are also fit on train only.

#[cfg(feature = "hyperion-cpu")]
#[path = "hyperion-v5-reference/imp.rs"]
mod imp;

#[cfg(feature = "hyperion-cpu")]
fn main() -> anyhow::Result<()> {
    imp::main_impl()
}

#[cfg(not(feature = "hyperion-cpu"))]
fn main() {
    eprintln!(
        "hyperion-v5-reference requires the hyperion-cpu or hyperion-cuda feature:\n  \
         cargo build -p hyperion --features hyperion-cpu --bin hyperion-v5-reference"
    );
    std::process::exit(2);
}
