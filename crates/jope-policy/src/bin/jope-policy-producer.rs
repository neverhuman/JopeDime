//! Private CPU policy worker. Invoke through the bounded execution pool.
use anyhow::{ensure, Context, Result};
use jope_policy::{
    config::CheckpointBinding, grammar::bind_vocabulary, model::PolicyModel, producer, sha256_hex,
};
use std::{
    collections::BTreeMap,
    fs,
    io::Write,
    path::{Path, PathBuf},
};

const USAGE: &str = "jope-policy-producer --request FILE --source-manifest FILE --context FILE --context-receipt FILE --transforms FILE --checkpoint-receipt FILE --policy FILE --vocabulary FILE --output FILE [--context-bundle FILE]";

fn physical(path: &Path) -> Result<()> {
    ensure!(
        path.is_absolute()
            && fs::canonicalize(path)? == path
            && fs::symlink_metadata(path)?.file_type().is_file(),
        "input must be an absolute physical file: {}",
        path.display()
    );
    Ok(())
}
fn checked_bytes(path: &Path, expected: Option<&str>) -> Result<Vec<u8>> {
    physical(path)?;
    let bytes = fs::read(path)?;
    if let Some(expected) = expected {
        ensure!(
            sha256_hex(&bytes) == expected,
            "input digest mismatch: {}",
            path.display()
        );
    }
    Ok(bytes)
}
fn run() -> Result<()> {
    let started = std::time::Instant::now();
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args == ["--help"] {
        println!("{USAGE}");
        return Ok(());
    }
    ensure!(args.len() == 18 || args.len() == 20, "{USAGE}");
    let required = [
        "--request",
        "--source-manifest",
        "--context",
        "--context-receipt",
        "--transforms",
        "--checkpoint-receipt",
        "--policy",
        "--vocabulary",
        "--output",
    ];
    let mut paths = BTreeMap::new();
    for pair in args.chunks_exact(2) {
        ensure!(
            (required.contains(&pair[0].as_str()) || pair[0] == "--context-bundle")
                && paths
                    .insert(pair[0].as_str(), PathBuf::from(&pair[1]))
                    .is_none(),
            "unknown/duplicate flag: {}",
            pair[0]
        );
    }
    for name in required {
        ensure!(paths.contains_key(name), "missing {name}");
    }
    let request: producer::Request =
        serde_json::from_slice(&checked_bytes(&paths["--request"], None)?)?;
    request.validate()?;
    checked_bytes(
        &paths["--source-manifest"],
        Some(&request.worker_source_sha256),
    )?;
    let context_bytes = checked_bytes(&paths["--context"], Some(&request.context_sha256))?;
    ensure!(
        context_bytes.len() == 2048 * 4,
        "context must be exactly2048 float32 values"
    );
    let context = context_bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().expect("four bytes")))
        .collect::<Vec<_>>();
    ensure!(
        context.iter().all(|x| x.is_finite() && x.abs() <= 32.),
        "context is nonfinite or not frozen-clipped"
    );
    if let Some(path) = paths.get("--context-bundle") {
        #[cfg(feature = "context-producer")]
        {
            let bundle: producer::QualifiedContextBundle =
                serde_json::from_slice(&checked_bytes(path, None)?)?;
            ensure!(
                bundle.artifact
                    == checked_bytes(
                        &paths["--context-receipt"],
                        Some(&bundle.pins.artifact_sha256)
                    )?,
                "CPU context receipt differs from admitted bundle"
            );
            let verified = bundle.verify()?;
            ensure!(
                verified.training().prepared_sha256 == request.prepared_sha256
                    && verified.training().training_sha256 == request.training_sha256
                    && verified.training().cut as usize == request.cut
                    && jope_policy::context_sha256(verified.context()) == request.context_sha256,
                "qualified CPU context differs from producer request"
            );
            ensure!(
                verified.context() == context.as_slice(),
                "qualified context vector differs from actual policy input"
            );
        }
        #[cfg(not(feature = "context-producer"))]
        {
            let _ = path;
            anyhow::bail!("qualified CPU context support was not built");
        }
    } else {
        let meta: serde_json::Value =
            serde_json::from_slice(&checked_bytes(&paths["--context-receipt"], None)?)?;
        ensure!(
            meta["schema"] == "parallel.encoding.v1"
                && meta["dataset_dimensions"] == 1536
                && meta["target_dimensions"] == 512
                && meta["prepared_sha256"] == request.prepared_sha256
                && meta["training_sha256"] == request.training_sha256
                && meta["cut"] == request.cut
                && meta["output_sha256"] == request.context_sha256,
            "context receipt identity mismatch"
        );
        ensure!(
            meta["encoder_spec_sha256"]
                == "6582df23f5c1ee048d11040a5f2bd1447032932fc23f6c5a3aa9cb4f2d6896cb"
                && meta["numerical_backend"]
                    == "Rust/LibTorch 2.7 CUDA FP32; original frozen transforms",
            "unsupported context lineage"
        );
        ensure!(meta["partition"].as_str().is_some_and(|p|["meta_train","meta_validation"].contains(&p)) && meta.get("task").is_none_or(|t|t == "regression") && meta.get("qualification").is_none_or(|q|q.as_bool() == Some(false)) && meta.get("final_test_targets_used").and_then(serde_json::Value::as_bool) == Some(false), "only accepted train-side regression context is supported; experimental classification remains separate");
    }
    eprintln!(
        "policy stage=context_verified elapsed_seconds={:.6}",
        started.elapsed().as_secs_f64()
    );
    checked_bytes(
        &paths["--transforms"],
        Some("1b779c6ff7a1a812e3001342d663353d3c7ff54eb93392a600f01813f4c66e22"),
    )?;
    let checkpoint = CheckpointBinding::from_receipt_bytes(&checked_bytes(
        &paths["--checkpoint-receipt"],
        None,
    )?)?;
    checkpoint.require_accepted_identity()?;
    let vocabulary = bind_vocabulary(
        &checked_bytes(&paths["--vocabulary"], Some(&request.vocabulary_sha256))?,
        checkpoint.vocabulary_sha256(),
    )?;
    physical(&paths["--policy"])?;
    let model = PolicyModel::load_bound(&paths["--policy"], &checkpoint, candle_core::Device::Cpu)?;
    eprintln!(
        "policy stage=weights_verified elapsed_seconds={:.6}",
        started.elapsed().as_secs_f64()
    );
    let output = producer::produce(&request, &model, &checkpoint, &vocabulary, &context)?;
    eprintln!(
        "policy stage=proposals_ranked elapsed_seconds={:.6}",
        started.elapsed().as_secs_f64()
    );
    let path = &paths["--output"];
    ensure!(
        path.is_absolute() && !path.starts_with("/tmp"),
        "output must be in the durable private worker directory"
    );
    let parent = path.parent().context("output has no parent")?;
    ensure!(
        fs::canonicalize(parent)? == parent,
        "output parent must be physical"
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        let mut file = fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .mode(0o600)
            .open(path)?;
        file.write_all(&output)?;
        file.sync_all()?;
        fs::File::open(parent)?.sync_all()?;
    }
    #[cfg(not(unix))]
    anyhow::bail!("private0600 output requires a Unix worker");
    println!(
        "{}",
        serde_json::json!({"schema":producer::SCHEMA,"output_sha256":sha256_hex(&output),"numerical_backend":"candle_cpu_float32","native_cuda_parity_qualified":false})
    );
    Ok(())
}
fn main() {
    if let Err(error) = run() {
        eprintln!("policy producer: {error:#}");
        std::process::exit(1);
    }
}
