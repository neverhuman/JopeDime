//! Regenerate assets/grammar/featmath-jain-v1.json from featmath.json and the
//! checked-in jain-math executable op list. `--check` compares instead.
use anyhow::{ensure, Result};
use jope_parallel::jain_vocabulary::{
    derive_restricted_vocabulary, BASE_VOCABULARY, EXECUTABLE_OPS, RESTRICTED_VOCABULARY,
};
use std::path::Path;

fn main() -> Result<()> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let derived = derive_restricted_vocabulary(
        &std::fs::read(root.join(BASE_VOCABULARY))?,
        &std::fs::read(root.join(EXECUTABLE_OPS))?,
    )?;
    let target = root.join(RESTRICTED_VOCABULARY);
    if std::env::args().any(|arg| arg == "--check") {
        ensure!(
            std::fs::read(&target)? == derived,
            "{RESTRICTED_VOCABULARY} is stale"
        );
    } else {
        std::fs::write(&target, &derived)?;
    }
    println!("{}", target.display());
    Ok(())
}
