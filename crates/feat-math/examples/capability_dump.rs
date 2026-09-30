//! Deterministic views of the kernel capability declaration.

use feat_math::kernel::{capabilities, profile};

fn main() {
    let mut args = std::env::args().skip(1);
    let output = match (args.next().as_deref(), args.next()) {
        (None, None) => capabilities::manifest_json(),
        (Some("--profiles"), None) => profile::render_profiles(),
        (Some("--markdown"), None) => capabilities::render_markdown(),
        _ => std::process::exit(2),
    };
    print!("{output}");
}
