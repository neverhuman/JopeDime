//! Owner-admitted Q1 retry host with the original numerical worker.
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments: Vec<_> = std::env::args_os().collect();
    if arguments.len() != 3 {
        return Err("usage: q1_infra_retry ABSOLUTE_BUNDLE_DIR ABSOLUTE_PINNED_WORKER".into());
    }
    let forwards = jope_service::engine::serve_q1_infra_retry(
        Path::new(&arguments[1]),
        Path::new(&arguments[2]),
    )?;
    println!("{{\"learned_forward_count\":{forwards},\"qualification_granted\":false}}");
    Ok(())
}
