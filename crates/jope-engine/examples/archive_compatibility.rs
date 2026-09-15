//! Exact frozen-grammar reuse audit. Never projects, snaps or evaluates programs.
use anyhow::{ensure, Result};
use feat_math::dsl::parse_dsl;
use jope_parallel::{grammar::Vocab, manifest::read_json};
use serde_json::{json, Value};
use std::{env, path::Path};

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    ensure!(
        args.len() == 3,
        "usage: archive_compatibility VOCABULARY PROGRAMS_JSON"
    );
    let vocab: Vocab = read_json(Path::new(&args[1]))?;
    let programs: Vec<String> = read_json(Path::new(&args[2]))?;
    let rows: Vec<Value> = programs.iter().map(|program| {
        let canonical = parse_dsl(program).map(|p| p.to_string());
        let encoded = vocab.encode(program);
        match (canonical, encoded) {
            (Ok(canonical), Ok(tokens)) if tokens.len() <= jope_parallel::corpus::SEQUENCE =>
                json!({"input":program,"canonical_program":canonical,"exact_compatible":true,
                       "tokens":tokens,"rejection_reason":null}),
            (Ok(canonical), Ok(_)) => json!({"input":program,"canonical_program":canonical,
                "exact_compatible":false,"rejection_reason":"program exceeds frozen sequence length"}),
            (Ok(canonical), Err(error)) => json!({"input":program,"canonical_program":canonical,
                "exact_compatible":false,"rejection_reason":format!("{error:#}")}),
            (Err(error), _) => json!({"input":program,"canonical_program":null,
                "exact_compatible":false,"rejection_reason":format!("{error:#}")}),
        }
    }).collect();
    println!("{}", serde_json::to_string(&rows)?);
    Ok(())
}
