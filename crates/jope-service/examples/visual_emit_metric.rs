//! Scoped warm repeated-emission measurement; never neural decoding throughput.
use anyhow::{ensure, Context, Result};
use jope_runtime::model_object::ModelObject;
use jope_service::lex::{count_tokens, SourceLanguage, LEXER_VERSION};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{fs, io::Write, path::Path, time::Instant};

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn main() -> Result<()> {
    let args = std::env::args_os().skip(1).collect::<Vec<_>>();
    ensure!(
        args.len() == 3,
        "usage: visual_emit_metric MODEL_DIR VERIFIED_RUST_SOURCE RECEIPT_JSON"
    );
    let model = ModelObject::read(Path::new(&args[0]), 256 * 1024 * 1024)?;
    let expected = fs::read(&args[1])?;
    let output_path = Path::new(&args[2]);
    ensure!(
        !output_path.exists(),
        "refuse to replace existing metric receipt"
    );
    let source = std::str::from_utf8(&expected)?;
    let tokens = count_tokens(SourceLanguage::Rust, source)?;
    ensure!(tokens > 0 && !expected.is_empty(), "empty verified source");
    ensure!(
        model.export_source("rust")? == expected,
        "local emitter differs from verified signed-engine artifact"
    );
    // Observe resolution and clock-pair overhead. These samples never include
    // emission and never contribute tokens or time to the reported batch.
    let mut pair_ns = Vec::with_capacity(1000);
    for _ in 0..1000 {
        let begin = Instant::now();
        pair_ns.push(begin.elapsed().as_nanos());
    }
    pair_ns.sort_unstable();
    let resolution = *pair_ns
        .iter()
        .find(|n| **n > 0)
        .context("monotonic clock did not advance")?;
    let p99 = pair_ns[989];
    let minimum_ns = 100_000_000_u128.max(1000 * resolution).max(100 * p99);
    // Calibration repeats fresh invocations. Source bytes are retained so hash,
    // tokenization and byte comparison happen after the one outer timed span.
    let mut repetitions = 1usize;
    let (elapsed_ns, batch) = loop {
        ensure!(
            repetitions <= 65_536,
            "emitter below reporting interval within repetition budget"
        );
        ensure!(
            repetitions
                .checked_mul(expected.len())
                .is_some_and(|bytes| bytes <= 256 * 1024 * 1024),
            "emission batch exceeds memory budget"
        );
        let mut outputs = Vec::with_capacity(repetitions);
        let begin = Instant::now();
        for _ in 0..repetitions {
            outputs.push(std::hint::black_box(model.export_source("rust")?));
        }
        let elapsed = begin.elapsed().as_nanos();
        if elapsed >= minimum_ns {
            break (elapsed, outputs);
        }
        repetitions *= 2;
    };
    ensure!(
        batch.iter().all(|bytes| *bytes == expected),
        "timed emission bytes changed"
    );
    let counted = batch
        .iter()
        .try_fold(0usize, |sum, bytes| -> Result<usize> {
            let text = std::str::from_utf8(bytes)?;
            Ok(sum + count_tokens(SourceLanguage::Rust, text)?)
        })?;
    ensure!(counted == repetitions * tokens, "batch token count differs");
    ensure!(
        fs::read(&args[1])? == expected,
        "verified source changed during measurement"
    );
    let rate = counted as f64 * 1e9 / elapsed_ns as f64;
    let receipt = json!({
        "schema":"jope.visual-emit-batch-metric/v1","state":"measured",
        "metric_name":"viz.emit_batch.rust_tokens_per_s","value":rate,
        "label":"Warm batch source emission · Rust lexical tokens/s · repeated same artifact · loop overhead included",
        "scope":"fresh ModelObject::export_source(\"rust\") invocations; model load, warmup, calibration, tokenization, hashing and verification excluded; allocation and loop overhead included",
        "build_profile":if cfg!(debug_assertions) {
            "development build with debug assertions; diagnostic local host measurement"
        } else {
            "optimized release build without debug assertions; local host measurement"
        },
        "debug_assertions":cfg!(debug_assertions),
        "model_id":model.id(),"source_sha256":sha(&expected),"source_bytes":expected.len(),
        "lexer":LEXER_VERSION,"rust_tokens_per_artifact":tokens,"repetitions":repetitions,
        "total_emitted_tokens":counted.to_string(),"batch_elapsed_ns":elapsed_ns.to_string(),
        "minimum_reporting_interval_ns":minimum_ns.to_string(),"clock_resolution_observed_ns":resolution.to_string(),
        "clock_pair_p99_ns":p99.to_string(),"source_bytes_equal_verified_artifact":true,
        "repeated_artifact":true,"unique_source_artifacts":1,
        "neural_decoding_tokens_per_second":null,"newly_published_source_metric_0x012a":null
    });
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output_path)?;
    file.write_all(&serde_json::to_vec_pretty(&receipt)?)?;
    file.sync_all()?;
    println!("PASS warm repeated Rust emission: {rate:.1} lexical tokens/s; {repetitions} identical emissions; interval {} ms; receipt {}",
        elapsed_ns as f64 / 1e6, output_path.display());
    Ok(())
}
