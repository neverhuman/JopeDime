//! Operator-admitted numeric REAL S3 FIT only; the manifest is a declaration, not S3 attestation.
//! Set JOPE_SESSION_BOUND_CSV, JOPE_SESSION_BOUND_MANIFEST, JOPE_SESSION_BOUND_DEVICE=cpu|cuda:N.
//! Run the ignored test alone in an optimized build; load/model startup and input clones are excluded.
//! CUDA requires an external `flock -n ~/jope_plan/qualification.lock <test command>` reservation.
//! No fetching, training, sealed-data discovery, cold-start, retention or quality qualification.
use super::{collect, start_training_observed, SessionOptions, SessionOutcome, TrainingInput};
use crate::dope_encoder::{
    DopeConsumer, DopeConsumerProduct, DopeDevice, DopeEncoder, Head, LandmarkArgs, LandmarkBundle,
    LandmarkConsumer, LandmarkMode, DOPE512_RECIPE, DOPE_WEIGHTS_SHA256,
};
use anyhow::{anyhow, ensure, Result};
use sha2::{Digest, Sha256};
use std::{cell::Cell, collections::BTreeSet, path::Path, sync::Arc, time::Instant};

const D0: &str = "d0b6960328d1955386632ff78b512e3add53a9b0256db7bbf067e2e0be5f0c6c";
const WEIGHTS: &[u8] =
    include_bytes!("../../../../../deploy/appliance/assets/dope-mlc-v3/model.safetensors");
#[derive(Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct FitManifest {
    s3_uri: String,
    role: String,
    csv_sha256: String,
    target: String,
    columns: Vec<String>,
    row_ids: Vec<u64>,
    task: String,
}
fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn admit(m: &FitManifest) -> Result<()> {
    let uri = m
        .s3_uri
        .strip_prefix("s3://")
        .and_then(|s| s.split_once('/'));
    ensure!(
        uri.is_some_and(|(b, k)| !b.is_empty() && !k.is_empty()),
        "S3 FIT provenance required"
    );
    ensure!(
        m.role == "fit" && (1..=1024).contains(&m.row_ids.len()),
        "FIT role/row bound required"
    );
    ensure!(
        m.row_ids.iter().collect::<BTreeSet<_>>().len() == m.row_ids.len(),
        "duplicate physical ID"
    );
    ensure!(
        m.columns.len() > 1 && m.columns.iter().collect::<BTreeSet<_>>().len() == m.columns.len(),
        "invalid column schema"
    );
    ensure!(m.columns.contains(&m.target), "missing target");
    ensure!(
        ["regression", "classification", "multiclass"].contains(&m.task.as_str()),
        "invalid task"
    );
    Ok(())
}
fn parse_fit(bytes: &[u8], m: &FitManifest) -> Result<TrainingInput> {
    admit(m)?;
    ensure!(hash(bytes) == m.csv_sha256, "CSV hash mismatch");
    let mut reader = csv::ReaderBuilder::new().from_reader(bytes);
    let header = reader
        .headers()
        .map_err(|_| anyhow!("invalid CSV header"))?;
    ensure!(
        header.iter().eq(m.columns.iter().map(String::as_str)),
        "CSV/schema mismatch"
    );
    let target = m
        .columns
        .iter()
        .position(|c| c == &m.target)
        .ok_or_else(|| anyhow!("missing target"))?;
    let (mut x, mut y) = (Vec::new(), Vec::new());
    for record in reader.records() {
        let record = record.map_err(|_| anyhow!("invalid CSV record"))?;
        ensure!(x.len() < m.row_ids.len(), "extra CSV rows");
        let values = record
            .iter()
            .map(|s| {
                s.parse::<f64>()
                    .map_err(|_| anyhow!("numeric CSV required"))
            })
            .collect::<Result<Vec<_>>>()?;
        ensure!(
            values.iter().all(|v| v.is_finite()),
            "finite numeric CSV required"
        );
        y.push(values[target]);
        x.push(
            values
                .into_iter()
                .enumerate()
                .filter_map(|(i, v)| (i != target).then_some(v))
                .collect(),
        );
    }
    ensure!(y.len() == m.row_ids.len(), "CSV/physical ID row mismatch");
    let mut classes = y.clone();
    classes.sort_by(f64::total_cmp);
    classes.dedup_by(|a, b| *a == *b);
    let head = match m.task.as_str() {
        "regression" => Head::Regression,
        "classification" => {
            ensure!(classes.len() == 2, "binary classes required");
            Head::Binary
        }
        _ => {
            ensure!(classes.len() >= 3, "multiclass classes required");
            Head::Multiclass(classes.len())
        }
    };
    let mut input = TrainingInput::numeric(x, y, m.row_ids.clone(), head);
    input.source_digest = Some(m.csv_sha256.clone());
    input.fold = Some("fit".into());
    Ok(input)
}
fn device_index(s: &str) -> Result<Option<usize>> {
    if s == "cpu" {
        return Ok(None);
    }
    let digits = s
        .strip_prefix("cuda:")
        .filter(|v| !v.is_empty() && v.bytes().all(|b| b.is_ascii_digit()));
    Ok(Some(
        digits
            .ok_or_else(|| anyhow!("use cpu or cuda:N"))?
            .parse()
            .map_err(|_| anyhow!("invalid CUDA ordinal"))?,
    ))
}
fn read_regular(path: &Path, limit: u64) -> Result<Vec<u8>> {
    let meta =
        std::fs::symlink_metadata(path).map_err(|_| anyhow!("input metadata unavailable"))?;
    ensure!(
        meta.is_file() && !meta.file_type().is_symlink() && meta.len() <= limit,
        "regular bounded input required"
    );
    std::fs::read(path).map_err(|_| anyhow!("input read failed"))
}
#[derive(Debug, serde::Serialize)]
struct Summary {
    overhead_max_ms: f64,
    complete_p50_ms: f64,
    complete_p95_ms: f64,
}
fn reduce(read_ms: f64, samples: &[[f64; 2]]) -> Result<Summary> {
    ensure!(
        samples.len() == 6 && read_ms.is_finite() && read_ms >= 0.,
        "invalid measurement set"
    );
    let mut totals = Vec::new();
    for [prepare, overhead] in samples {
        ensure!(
            [*prepare, *overhead]
                .iter()
                .all(|x| x.is_finite() && *x >= 0.),
            "invalid timing"
        );
        let total = read_ms + prepare + overhead;
        ensure!(total.is_finite() && total > 0., "invalid complete timing");
        totals.push(total);
    }
    totals.remove(0);
    totals.sort_by(f64::total_cmp);
    Ok(Summary {
        overhead_max_ms: samples[1..].iter().map(|s| s[1]).fold(0., f64::max),
        complete_p50_ms: totals[2],
        complete_p95_ms: totals[3] + (totals[4] - totals[3]) * 0.8,
    })
}
fn require_bounds(s: &Summary) -> Result<()> {
    ensure!(
        s.overhead_max_ms <= 5. && s.complete_p50_ms <= 30.,
        "resident/session release bound failed"
    );
    Ok(())
}

#[test]
#[ignore = "real S3 FIT operator declaration, optimized build and explicit device/resource reservation required"]
fn real_s3_fit_none_session_release_bound() -> Result<()> {
    let env = |key| std::env::var(key).map_err(|_| anyhow!("required session-bound input missing"));
    let manifest_bytes = read_regular(Path::new(&env("JOPE_SESSION_BOUND_MANIFEST")?), 1 << 20)?;
    let manifest: FitManifest =
        serde_json::from_slice(&manifest_bytes).map_err(|_| anyhow!("invalid FIT manifest"))?;
    admit(&manifest)?; // Refuse non-FIT declarations before opening the CSV.
    let start = Instant::now();
    let bytes = read_regular(Path::new(&env("JOPE_SESSION_BOUND_CSV")?), 64 << 20)?;
    let input = parse_fit(&bytes, &manifest)?;
    let read_ms = start.elapsed().as_secs_f64() * 1000.;
    let ordinal = device_index(&env("JOPE_SESSION_BOUND_DEVICE")?)?;
    let device = match ordinal {
        None => DopeDevice::Cpu,
        Some(n) => DopeEncoder::cuda(n)?,
    };
    ensure!(
        device.is_cpu() == ordinal.is_none(),
        "requested device mismatch"
    );
    ensure!(
        DOPE_WEIGHTS_SHA256 == D0 && hash(WEIGHTS) == D0,
        "released D0 checkpoint required"
    );
    let encoder = DopeEncoder::from_bytes(WEIGHTS, &device)?;
    let pool = rayon::ThreadPoolBuilder::new().num_threads(8).build()?;
    let args = LandmarkArgs {
        landmark_mode: Some(LandmarkMode::None),
        landmark_consumer: Some(LandmarkConsumer::Direct),
        ..Default::default()
    };
    let mut samples = Vec::new();
    let mut binding = None;
    for _ in 0..6 {
        let handoff = input.clone(); // Resident input handoff; excluded from all recorded spans.
        let resident = encoder.clone();
        let (sample, digests) = pool.install(|| -> Result<_> {
            ensure!(
                rayon::current_num_threads() == 8,
                "host preparation pool differs"
            );
            let boundary = Cell::new(None);
            let start = Instant::now();
            let session = start_training_observed(
                handoff,
                resident,
                &args,
                SessionOptions {
                    consumer: Some(Arc::new(DopeConsumer::encoding_only())),
                    ..Default::default()
                },
                || boundary.set(Some(Instant::now())),
            )?;
            let values = session.initial().embedding.to_vec()?;
            let end = Instant::now(); // Completion/readback included; worker join is excluded.
            let split = boundary
                .get()
                .ok_or_else(|| anyhow!("preparation observer not called"))?;
            ensure!(
                split >= start && split <= end,
                "invalid preparation boundary"
            );
            let initial = &session.initial().receipt;
            ensure!(
                values.len() == 512 && values.iter().all(|v| v.is_finite()),
                "invalid Dope512 output"
            );
            ensure!(
                initial.version == 0
                    && initial.mask.iter().all(|v| !v)
                    && initial.status == LandmarkBundle::zero().status,
                "not a zero snapshot"
            );
            ensure!(
                initial.weights_sha256 == D0 && initial.embedding_identity == DOPE512_RECIPE,
                "checkpoint/recipe changed"
            );
            ensure!(
                matches!(
                    session.initial().consumer.as_ref().map(|c| c.product()),
                    Some(DopeConsumerProduct::Direct(_))
                ),
                "direct product absent"
            );
            let digests = (
                initial.source_digest.clone(),
                initial.conditioning_digest.clone(),
            );
            let mut events = 0;
            let run = collect(session, |_| {
                events += 1;
                Ok(())
            })?;
            ensure!(
                events == 1 && run.outcome == SessionOutcome::Complete,
                "none session did not complete once"
            );
            ensure!(
                run.final_snapshot.receipt.source_digest == digests.0
                    && run.final_snapshot.receipt.conditioning_digest == digests.1,
                "session scope changed"
            );
            Ok((
                [
                    (split - start).as_secs_f64() * 1000.,
                    (end - split).as_secs_f64() * 1000.,
                ],
                digests,
            ))
        })?;
        if let Some(previous) = &binding {
            ensure!(previous == &digests, "run conditioning changed");
        }
        binding = Some(digests);
        samples.push(sample);
    }
    ensure!(hash(WEIGHTS) == D0, "checkpoint bytes changed");
    let summary = reduce(read_ms, &samples)?;
    println!(
        "{}",
        serde_json::json!({"schema":"jope.none-session-release-bound/v1",
        "csv_sha256":manifest.csv_sha256,"manifest_sha256":hash(&manifest_bytes),"weights_sha256":D0,
        "embedding_identity":DOPE512_RECIPE,
        "rows":input.y.len(),"features":input.categories.len(),"task":manifest.task,
        "requested_device":ordinal.map_or_else(||"cpu".to_owned(),|n|format!("cuda:{n}")),"actual_device":format!("{:?}",device.location()),
        "host_rayon_threads":8,"cpu_forward_threads":if device.is_cpu(){1}else{0},"runs":6,"discard_first":1,
        "read_once_ms":read_ms,"samples_validation_prepare_and_overhead_ms":samples,"summary":summary,
        "source_and_conditioning_digests":binding,"provenance":"operator declaration; S3 not independently attested",
        "span":"single actual session; validation+prepare, then FitScope+consumer+conditioning+forward+spawn+host readback; join excluded",
        "model_load_and_input_handoff_clones_excluded":true,"cold_5x_or_quality_claimed":false})
    );
    require_bounds(&summary)
}

#[test]
fn reducer_discards_one_warmup_and_uses_interpolated_p95() -> Result<()> {
    let s = reduce(
        1.,
        &[
            [100., 100.],
            [8., 1.],
            [10., 2.],
            [12., 3.],
            [14., 4.],
            [16., 5.],
        ],
    )?;
    assert_eq!((s.overhead_max_ms, s.complete_p50_ms), (5., 16.));
    assert!((s.complete_p95_ms - 21.4).abs() < 1e-12);
    require_bounds(&reduce(1., &[[24., 5.]; 6])?)
}
#[test]
fn reducer_rejects_invalid_measurements_and_actual_limit_breaches() {
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.] {
        assert!(reduce(value, &[[1., 1.]; 6]).is_err());
        assert!(reduce(0., &[[value, 1.]; 6]).is_err());
        assert!(reduce(0., &[[1., value]; 6]).is_err());
    }
    assert!(reduce(1., &[[1., 1.]; 5]).is_err());
    assert!(reduce(0., &[[0., 0.]; 6]).is_err());
    assert!(reduce(0., &[[0., 1.]; 6]).is_ok());
    let mut zero_warmup = [[0., 1.]; 6];
    zero_warmup[0] = [0., 0.];
    assert!(reduce(0., &zero_warmup).is_err());
    assert!(require_bounds(&reduce(0., &[[1., 5.00001]; 6]).unwrap()).is_err());
    assert!(require_bounds(&reduce(0., &[[25.00001, 5.]; 6]).unwrap()).is_err());
}
#[test]
fn numeric_fit_admission_refuses_bad_scope_hash_rows_schema_and_values() -> Result<()> {
    let bytes = b"x,y\n1,0\n2,1\n3,2\n4,0\n";
    let m = FitManifest {
        s3_uri: "s3://unit-fixture/fit.csv".into(),
        role: "fit".into(),
        csv_sha256: hash(bytes),
        target: "y".into(),
        columns: vec!["x".into(), "y".into()],
        row_ids: vec![9, 12, 22, 38],
        task: "regression".into(),
    };
    assert!(parse_fit(bytes, &m).is_ok());
    let mut other = m.clone();
    other.task = "multiclass".into();
    assert!(parse_fit(bytes, &other).is_ok());
    other.task = "classification".into();
    assert!(parse_fit(bytes, &other).is_err());
    let binary = b"x,y\n1,0\n2,1\n3,0\n4,1\n";
    other.csv_sha256 = hash(binary);
    assert!(parse_fit(binary, &other).is_ok());
    for modify in 0..8 {
        let mut bad = m.clone();
        match modify {
            0 => bad.role = "test".into(),
            1 => bad.s3_uri = "file:///fixture".into(),
            2 => bad.csv_sha256 = "0".repeat(64),
            3 => bad.row_ids[1] = bad.row_ids[0],
            4 => bad.columns.reverse(),
            5 => {
                bad.row_ids.pop();
            }
            6 => bad.columns[0] = bad.target.clone(),
            _ => bad.row_ids = (0..1025).collect(),
        }
        assert!(parse_fit(bytes, &bad).is_err());
    }
    for value in ["NaN", "inf", "bad"] {
        for body in [
            format!("x,y\n{value},0\n2,1\n3,2\n4,0\n"),
            format!("x,y\n1,{value}\n2,1\n3,2\n4,0\n"),
        ] {
            let mut bad = m.clone();
            bad.csv_sha256 = hash(body.as_bytes());
            assert!(parse_fit(body.as_bytes(), &bad).is_err());
        }
    }
    for alias in [
        "auto",
        "cuda",
        "gpu",
        "cpu:0",
        "cuda:-1",
        "cuda:+1",
        "cuda:",
        "cuda:999999999999999999999999999999",
    ] {
        assert!(device_index(alias).is_err());
    }
    assert_eq!(device_index("cpu")?, None);
    assert_eq!(device_index("cuda:02")?, Some(2));
    Ok(())
}
