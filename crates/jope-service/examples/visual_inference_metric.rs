//! Resident CPU helper for bounded local CSV -> descriptors -> selected DSL -> Rust timing.
//! Neural selection is supplied by the external resident orchestrator. This is not
//! a production H256 serving adapter. Only original input contracts are cached.
use anyhow::{ensure, Context, Result};
use jope_runtime::{
    limits::Limits,
    math_pipeline::{InputContract, NumericColumn, PipelineState, INPUT_PART, PIPELINE_PART},
    model_object::{Manifest, ModelObject, Part, Producer, MATH_PIPELINE, SCHEMA},
};
use jope_service::{
    lex::{count_tokens, SourceLanguage, LEXER_VERSION},
    ranker::landmarks::fit_transform_and_extract_bundle_high,
};
use jope_telemetry::hooks::io::{rates, RateCounters};
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::{BufRead, Read, Write},
    path::{Path, PathBuf},
    time::Instant,
};

const MAX_MODEL: u64 = 256 * 1024 * 1024;
const MAX_CSV: u64 = 8 * 1024 * 1024;
const MAX_REQUEST: usize = 128 * 1024;
const MAX_SOURCE: usize = 16 * 1024 * 1024;
const MAX_CONTRACTS: usize = 16;
const PROTOCOL: &str = "jain.local-inference-helper/v1";

#[derive(Deserialize)]
#[serde(tag = "mode", rename_all = "kebab-case", deny_unknown_fields)]
enum Request {
    Encode {
        request_id: String,
        case_id: String,
        csv_path: PathBuf,
        split_path: PathBuf,
        original_model_dir: PathBuf,
    },
    FitRust {
        request_id: String,
        case_id: String,
        csv_path: PathBuf,
        split_path: PathBuf,
        original_model_dir: PathBuf,
        selected_dsl: String,
    },
    Proof {
        request_id: String,
        case_id: String,
        expected_rust_sha256: String,
        round_trip_ns: u64,
    },
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Split {
    fit: Vec<usize>,
    holdout: Vec<usize>,
    seed: u64,
    label: String,
    #[serde(default)]
    input_sha256: Option<String>,
}

#[derive(Clone)]
struct Contract {
    input: InputContract,
    model_id: String,
    input_sha256: String,
}
struct Prepared {
    input: InputContract,
    csv_sha256: String,
    split_sha256: String,
    original_model_id: String,
    input_contract_sha256: String,
    fit_ids: Vec<usize>,
    holdout_ids: Vec<usize>,
    raw_fit_x: Vec<f32>,
    raw_fit_y: Vec<f32>,
    fit_x: Vec<Vec<f64>>,
    fit_y: Vec<f64>,
}
struct Pending {
    request_id: String,
    case_id: String,
    source: Vec<u8>,
    canonical_dsl: String,
    model_id: String,
}

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn read_bounded(path: &Path, maximum: u64) -> Result<Vec<u8>> {
    let meta = fs::symlink_metadata(path)?;
    ensure!(
        meta.is_file() && meta.len() <= maximum,
        "invalid/oversized file: {}",
        path.display()
    );
    let mut bytes = Vec::new();
    fs::File::open(path)?
        .take(maximum + 1)
        .read_to_end(&mut bytes)?;
    ensure!(bytes.len() as u64 <= maximum, "input grew past bound");
    Ok(bytes)
}
fn numeric(cell: Option<&str>) -> Result<f64> {
    let value = cell
        .context("missing numeric cell")?
        .trim()
        .parse::<f64>()?;
    ensure!(value.is_finite(), "nonfinite numeric cell");
    Ok(value)
}
fn encode(value: f64, column: &NumericColumn) -> Result<f64> {
    let result = if column.constant {
        0.0
    } else {
        ((value - column.min) / (column.max - column.min)).clamp(0.0, 1.0)
    };
    ensure!(result.is_finite(), "normalization overflow");
    Ok(result)
}
fn f32_sha(values: &[f32]) -> String {
    sha(&values
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>())
}
fn f64_sha(values: impl Iterator<Item = f64>) -> String {
    sha(&values.flat_map(f64::to_le_bytes).collect::<Vec<_>>())
}
fn check_split(split: &Split, rows: usize) -> Result<()> {
    ensure!(
        (4..=1024).contains(&rows) && !split.fit.is_empty() && !split.holdout.is_empty(),
        "require 4..1024 rows with FIT and HOLD"
    );
    let all: BTreeSet<_> = split.fit.iter().chain(&split.holdout).copied().collect();
    ensure!(
        split.fit.len() + split.holdout.len() == rows
            && all.len() == rows
            && all.iter().copied().eq(0..rows),
        "split is not a disjoint complete partition"
    );
    let expected = jope_service::discover::splits::random_split(
        rows,
        split.fit.len(),
        split.seed,
        &split.label,
    )?;
    ensure!(
        expected.0 == split.fit && expected.1 == split.holdout,
        "split differs from native seed/label algorithm"
    );
    Ok(())
}
fn contract(path: &Path, cache: &mut BTreeMap<PathBuf, Contract>) -> Result<Contract> {
    let path = fs::canonicalize(path)?;
    if let Some(value) = cache.get(&path) {
        return Ok(value.clone());
    }
    ensure!(
        cache.len() < MAX_CONTRACTS,
        "resident input-contract cache exceeds sixteen cases"
    );
    let original = ModelObject::read(&path, MAX_MODEL)?;
    original.verify_executable()?;
    let old = original.math_pipeline()?.input;
    ensure!(
        old.prediction_version == 2,
        "require original f64 prediction_version=2"
    );
    let input = InputContract::new(old.columns, old.selected, old.target)?;
    ensure!(
        input.columns.len() == 4 && input.selected.len() == 4,
        "bounded preview requires four numeric selected features"
    );
    let value = Contract {
        input_sha256: sha(&serde_json::to_vec(&input)?),
        input,
        model_id: original.id().to_owned(),
    };
    cache.insert(path, value.clone());
    Ok(value)
}
fn prepare(
    csv_path: &Path,
    split_path: &Path,
    model: &Path,
    cache: &mut BTreeMap<PathBuf, Contract>,
) -> Result<Prepared> {
    // CSV and split are freshly read on every request; neither data nor fitted GP is cached.
    let csv = read_bounded(csv_path, MAX_CSV)?;
    let split_bytes = read_bounded(split_path, 1024 * 1024)?;
    let split: Split = serde_json::from_slice(&split_bytes)?;
    let csv_sha256 = sha(&csv);
    let expected = if let Some(expected) = &split.input_sha256 {
        expected.clone()
    } else {
        let summary = split_path.with_file_name("summary.json");
        let receipt = split_path.with_file_name("receipt.json");
        let (path, field) = if summary.is_file() {
            (summary, "input_sha256")
        } else {
            (receipt, "input_csv")
        };
        let proof: Value = serde_json::from_slice(&read_bounded(&path, 1024 * 1024)?)?;
        if field == "input_sha256" {
            proof[field].as_str()
        } else {
            proof[field]["sha256"].as_str()
        }
        .context("split needs input_sha256 or adjacent executed CSV proof")?
        .to_owned()
    };
    ensure!(csv_sha256 == expected, "CSV differs from split proof");
    let table = jope_runtime::table::parse_csv(&csv, &Limits::default())?;
    check_split(&split, table.rows.len())?;
    let original = contract(model, cache)?;
    let input = original.input;
    let at = input
        .columns
        .iter()
        .map(|column| {
            table
                .headers
                .iter()
                .position(|name| name == &column.name)
                .context("missing feature")
        })
        .collect::<Result<Vec<_>>>()?;
    let target_at = table
        .headers
        .iter()
        .position(|name| name == &input.target.name)
        .context("missing target")?;
    ensure!(!at.contains(&target_at), "target leaked into features");
    let raw_x = table
        .rows
        .iter()
        .map(|row| {
            at.iter()
                .map(|&j| numeric(row[j].as_deref()))
                .collect::<Result<Vec<_>>>()
        })
        .collect::<Result<Vec<_>>>()?;
    let raw_y = table
        .rows
        .iter()
        .map(|row| numeric(row[target_at].as_deref()))
        .collect::<Result<Vec<_>>>()?;
    for (j, column) in input.columns.iter().enumerate() {
        let lo = split
            .fit
            .iter()
            .map(|&i| raw_x[i][j])
            .reduce(f64::min)
            .context("empty FIT")?;
        let hi = split
            .fit
            .iter()
            .map(|&i| raw_x[i][j])
            .reduce(f64::max)
            .context("empty FIT")?;
        ensure!(
            lo.to_bits() == column.min.to_bits()
                && hi.to_bits() == column.max.to_bits()
                && column.constant == (lo == hi),
            "FIT-only feature range mismatch"
        );
    }
    let lo = split
        .fit
        .iter()
        .map(|&i| raw_y[i])
        .reduce(f64::min)
        .context("empty FIT target")?;
    let hi = split
        .fit
        .iter()
        .map(|&i| raw_y[i])
        .reduce(f64::max)
        .context("empty FIT target")?;
    ensure!(
        lo.to_bits() == input.target.min.to_bits()
            && (hi - lo).max(1e-12).to_bits() == input.target.scale.to_bits(),
        "FIT-only target range mismatch"
    );
    let fit_x = split
        .fit
        .iter()
        .map(|&i| {
            input
                .selected
                .iter()
                .map(|&j| encode(raw_x[i][j], &input.columns[j]))
                .collect::<Result<Vec<_>>>()
        })
        .collect::<Result<Vec<_>>>()?;
    let fit_y = split
        .fit
        .iter()
        .map(|&i| (raw_y[i] - input.target.min) / input.target.scale)
        .collect::<Vec<_>>();
    ensure!(
        fit_y.iter().all(|v| v.is_finite()),
        "normalized target overflow"
    );
    let raw_fit_x = split
        .fit
        .iter()
        .flat_map(|&i| raw_x[i].iter().map(|v| *v as f32))
        .collect::<Vec<_>>();
    let raw_fit_y = split
        .fit
        .iter()
        .map(|&i| raw_y[i] as f32)
        .collect::<Vec<_>>();
    ensure!(
        raw_fit_x.iter().chain(&raw_fit_y).all(|v| v.is_finite()),
        "raw FIT overflows f32"
    );
    Ok(Prepared {
        input,
        csv_sha256,
        split_sha256: sha(&split_bytes),
        original_model_id: original.model_id,
        input_contract_sha256: original.input_sha256,
        fit_ids: split.fit,
        holdout_ids: split.holdout,
        raw_fit_x,
        raw_fit_y,
        fit_x,
        fit_y,
    })
}
fn identity(prepared: &Prepared) -> Value {
    json!({"dataset_sha256":prepared.csv_sha256,"split_sha256":prepared.split_sha256,
        "fit_row_ids":prepared.fit_ids,"holdout_row_ids":prepared.holdout_ids,"fit_count":prepared.fit_ids.len(),
        "original_model_id":prepared.original_model_id,"input_contract_sha256":prepared.input_contract_sha256,
        "raw_fit_x_f32_sha256":f32_sha(&prepared.raw_fit_x),"raw_fit_y_f32_sha256":f32_sha(&prepared.raw_fit_y),
        "normalized_fit_x_f64_sha256":f64_sha(prepared.fit_x.iter().flatten().copied()),
        "normalized_fit_y_f64_sha256":f64_sha(prepared.fit_y.iter().copied()),
        "descriptor_input_scope":"original numeric FIT cells cast f32; high recipe performs its own scaling",
        "gp_input_scope":"same-case version2 FIT-only contract; selected f64 0..1 features and scaled f64 targets"})
}
fn encode_case(case_id: &str, prepared: Prepared) -> Result<Value> {
    let begin = Instant::now();
    let descriptors = fit_transform_and_extract_bundle_high(
        &prepared.raw_fit_x,
        prepared.fit_ids.len(),
        4,
        &prepared.raw_fit_y,
    )?;
    let descriptor_ns = begin.elapsed().as_nanos().to_string();
    let v256 = descriptors.vector().to_vec();
    let col = descriptors
        .columns()
        .iter()
        .flatten()
        .copied()
        .collect::<Vec<_>>();
    ensure!(
        v256.len() == 256 && col.len() == 1088,
        "descriptor shape mismatch"
    );
    Ok(
        json!({"schema":PROTOCOL,"phase":"encoded","case_id":case_id,"identity":identity(&prepared),
        "v256":v256,"col64x17":col,"v256_f32_sha256":f32_sha(&v256),"col64x17_f32_sha256":f32_sha(&col),
        "descriptor_ns":descriptor_ns}),
    )
}
fn fit_rust(
    request_id: String,
    case_id: String,
    prepared: Prepared,
    dsl: &str,
) -> Result<(Value, Pending)> {
    ensure!(dsl.len() <= 64 * 1024, "selected DSL exceeds bound");
    let genome = feat_math::GenomeIr::parse(dsl.trim())?;
    let canonical = genome.canonical();
    let begin = Instant::now();
    let fitted = feat_math::model::fit(&genome, &prepared.fit_x, &prepared.fit_y)?;
    let fit_ns = begin.elapsed().as_nanos().to_string();
    let parts: BTreeMap<String, Vec<u8>> = BTreeMap::from([
        (
            PIPELINE_PART.into(),
            serde_json::to_vec(&PipelineState {
                dsl: canonical.clone(),
                model: fitted,
            })?,
        ),
        (INPUT_PART.into(), serde_json::to_vec(&prepared.input)?),
    ]);
    let object = ModelObject::new(
        Manifest {
            schema: SCHEMA.into(),
            minimum_reader_version: 1,
            format: MATH_PIPELINE.into(),
            producer: Producer::default(),
            primary: PIPELINE_PART.into(),
            required_capabilities: vec![],
            files: parts
                .iter()
                .map(|(path, bytes)| Part::new(path, "application/json", bytes))
                .collect(),
            metadata: json!({"fitting":"local_selected_dsl_fit_only","prediction_version":2,"training_rows":prepared.fit_ids.len(),"missing_input_policy":"refused by this numeric proof helper"}),
        },
        parts,
        MAX_MODEL,
    )?;
    object.verify_executable()?;
    let begin = Instant::now();
    let rust = object.export_source("rust")?;
    let export_ns = begin.elapsed().as_nanos().to_string();
    ensure!(
        !rust.is_empty() && rust.len() <= MAX_SOURCE,
        "Rust source exceeds bound"
    );
    let source = std::str::from_utf8(&rust)?;
    let response = json!({"schema":PROTOCOL,"phase":"code_ready","case_id":case_id,"rust_source":source,
        "rust_source_bytes":rust.len(),"canonical_dsl":canonical,"fitted_model_id":object.id(),
        "fit_ns":fit_ns,"export_ns":export_ns,"identity":identity(&prepared),
        "clock_boundary":"stop outer monotonic span after receiving this complete Rust record; hash and token count require separate proof request"});
    let pending = Pending {
        request_id,
        case_id,
        source: rust,
        canonical_dsl: canonical,
        model_id: object.id().to_owned(),
    };
    Ok((response, pending))
}
fn rust_metric(source: &[u8], elapsed_ns: u64) -> Result<Value> {
    ensure!(elapsed_ns > 0, "round_trip_ns must be positive");
    let tokens = count_tokens(SourceLanguage::Rust, std::str::from_utf8(source)?)?;
    ensure!(tokens > 0, "empty Rust token stream");
    let seconds = elapsed_ns as f64 / 1e9;
    let events = rates(
        seconds,
        &RateCounters {
            rust_tokens: Some(tokens as f64),
            all_tokens: Some(tokens as f64),
            ..Default::default()
        },
    )?;
    Ok(
        json!({"lexer":LEXER_VERSION,"rust_tokens":tokens,"output_languages":["rust"],"round_trip_ns":elapsed_ns,
        "total_ms":elapsed_ns as f64/1e6,"rust_tokens_per_second":tokens as f64/seconds,
        "metric_id":"0x012a","metric_name":"throughput.tokens_per_s.rust","metric_events":events,
        "rust_timed_alone":false,"all_languages_tokens":tokens,
        "compatibility_scope":"Rust-only output; all-languages count equals Rust count; both share the full round-trip span",
        "denominator_scope":"caller supplied one outer monotonic CSV-read through complete Rust JSONL record receipt span",
        "neural_decoding_tokens_per_second":null}),
    )
}
fn handle(
    request: Request,
    cache: &mut BTreeMap<PathBuf, Contract>,
    pending: &mut Option<Pending>,
) -> Result<Value> {
    match request {
        Request::Encode {
            request_id: _,
            case_id,
            csv_path,
            split_path,
            original_model_dir,
        } => {
            ensure!(pending.is_none(), "proof required before next request");
            encode_case(
                &case_id,
                prepare(&csv_path, &split_path, &original_model_dir, cache)?,
            )
        }
        Request::FitRust {
            request_id,
            case_id,
            csv_path,
            split_path,
            original_model_dir,
            selected_dsl,
        } => {
            ensure!(pending.is_none(), "proof required before next request");
            let (value, proof) = fit_rust(
                request_id,
                case_id,
                prepare(&csv_path, &split_path, &original_model_dir, cache)?,
                &selected_dsl,
            )?;
            *pending = Some(proof);
            Ok(value)
        }
        Request::Proof {
            request_id,
            case_id,
            expected_rust_sha256,
            round_trip_ns,
        } => {
            let actual = pending
                .as_ref()
                .context("no pending complete Rust artifact")?;
            ensure!(
                actual.case_id == case_id && actual.request_id == request_id,
                "proof case/request differs from pending Rust"
            );
            ensure!(
                sha(&actual.source) == expected_rust_sha256,
                "complete Rust differs from frozen selected-DSL artifact"
            );
            let metric = rust_metric(&actual.source, round_trip_ns)?;
            let result = json!({"schema":PROTOCOL,"phase":"proof","case_id":case_id,"source_sha256":expected_rust_sha256,
                "source_bytes":actual.source.len(),"canonical_dsl":actual.canonical_dsl,"fitted_model_id":actual.model_id,"metric":metric,
                "source_bytes_equal_verified_artifact":true,"proof_timing_scope":"hashing, canonical Rust tokenization and verification occur only after caller stops round-trip clock"});
            *pending = None;
            Ok(result)
        }
    }
}
fn main() -> Result<()> {
    ensure!(
        !cfg!(debug_assertions),
        "run --release: native fitting must match qualified Rust -O source"
    );
    ensure!(
        std::env::args_os().len() == 1,
        "visual_inference_metric takes JSONL requests on stdin, not CLI arguments"
    );
    let mut input = std::io::stdin().lock();
    let mut output = std::io::stdout().lock();
    let mut cache = BTreeMap::new();
    let mut pending = None;
    loop {
        let mut bytes = Vec::new();
        let read = (&mut input)
            .take((MAX_REQUEST + 1) as u64)
            .read_until(b'\n', &mut bytes)?;
        if read == 0 {
            break;
        }
        ensure!(
            bytes.len() <= MAX_REQUEST && bytes.last() == Some(&b'\n'),
            "JSONL request too large or incomplete"
        );
        let begin = Instant::now();
        let request: Request = serde_json::from_slice(&bytes)?;
        let request_id = match &request {
            Request::Encode { request_id, .. }
            | Request::FitRust { request_id, .. }
            | Request::Proof { request_id, .. } => request_id.clone(),
        };
        ensure!(
            !request_id.is_empty() && request_id.len() <= 128,
            "request_id must have 1..128 bytes"
        );
        let mut response = handle(request, &mut cache, &mut pending)?;
        response["request_id"] = json!(request_id);
        response["helper_request_ns"] = json!(begin.elapsed().as_nanos().to_string());
        response["helper_source_sha256"] = json!(sha(include_bytes!("visual_inference_metric.rs")));
        response["native_source_commit"] = json!(option_env!("JOPE_BUILD_SOURCE_SHA"));
        response["native_debug_assertions"] = json!(cfg!(debug_assertions));
        serde_json::to_writer(&mut output, &response)?;
        output.write_all(b"\n")?;
        output.flush()?;
    }
    ensure!(pending.is_none(), "EOF before pending Rust artifact proof");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn canonical_rust_only_count_and_nanosecond_conversion() {
        let source = b"fn main() {} // Python source is not emitted or counted\n";
        let metric = rust_metric(source, 250_000_000).unwrap();
        assert_eq!(metric["lexer"], LEXER_VERSION);
        assert_eq!(metric["rust_tokens"], 6);
        assert_eq!(metric["total_ms"], 250.0);
        assert_eq!(metric["rust_tokens_per_second"], 24.0);
        assert_eq!(metric["output_languages"], json!(["rust"]));
        assert_eq!(metric["all_languages_tokens"], 6);
        assert!(metric["metric_events"]
            .as_array()
            .unwrap()
            .iter()
            .any(|v| v[0] == 0x012au16 && v[1] == 24.0));
    }
    #[test]
    fn sixteen_contracts_allow_cached_hits_and_refuse_seventeenth_miss() -> Result<()> {
        let path = std::env::current_dir()?.canonicalize()?;
        let input = InputContract::new(
            (0..4)
                .map(|index| NumericColumn {
                    name: format!("x{index}"),
                    min: 0.0,
                    max: 1.0,
                    constant: false,
                })
                .collect(),
            vec![0, 1, 2, 3],
            jope_runtime::math_pipeline::TargetScale {
                name: "y".into(),
                min: 0.0,
                scale: 1.0,
            },
        )?;
        let value = Contract {
            input,
            model_id: "cached-test-contract".into(),
            input_sha256: "test".into(),
        };
        let mut cache = BTreeMap::from([(path.clone(), value.clone())]);
        for index in 1..MAX_CONTRACTS {
            cache.insert(
                path.join(format!("unused-cache-key-{index}")),
                value.clone(),
            );
        }
        assert_eq!(cache.len(), 16);
        assert_eq!(
            contract(&path, &mut cache)?.model_id,
            "cached-test-contract"
        );
        let error = contract(path.parent().context("test path needs parent")?, &mut cache)
            .err()
            .context("seventeenth uncached contract was accepted")?;
        assert!(error.to_string().contains("exceeds sixteen cases"));
        assert_eq!(cache.len(), 16);
        Ok(())
    }

    #[test]
    fn zero_time_and_malformed_rust_are_refused() {
        assert!(rust_metric(b"fn main() {}", 0).is_err());
        assert!(rust_metric(b"r#\"unterminated", 1).is_err());
    }
}
