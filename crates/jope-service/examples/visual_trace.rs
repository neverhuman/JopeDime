//! Private, CPU-only diagnostic of the authenticated frozen fused graph.
//! Run with an admitted release directory or signed appliance file, and an
//! external output directory. This does not run product search or claim lineage.
use anyhow::{ensure, Context, Result};
use jope_runtime::backend::DeviceRequest;
use jope_service::{
    ranker::{
        adapter::{PredictCostsInputs, SCALE},
        cuda::{release, FusedForward, StageOutput},
        featurizer::Featurizer,
        landmarks::fit_transform_and_extract_bundle_high,
        proj16::proj16,
    },
    visual_trace::{capture, validate_affine},
};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{fs, path::Path, time::Instant};

const ROWS: usize = 96;
const FEATURES: usize = 4;

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn table(case: usize) -> (Vec<f32>, Vec<f32>) {
    let mut x = Vec::with_capacity(ROWS * FEATURES);
    let mut y = Vec::with_capacity(ROWS);
    for row in 0..ROWS {
        let t = (row + 1) as f32;
        let r = if case == 0 {
            [
                (t * 0.13).sin(),
                (t * 0.17).cos(),
                (t % 11.) / 5. - 1.,
                (t * 0.07).sin() * (t * 0.11).cos(),
            ]
        } else {
            [
                1.25 * (t * 0.13).sin() + 0.3,
                (t * 0.09).cos(),
                (t % 13.) / 6. - 1.,
                (t * 0.19).sin(),
            ]
        };
        let target = if case == 0 {
            2.5 * r[0].sin() + 0.8 * r[1] * r[2] + 0.35 * r[3] * r[3] - 1.2 * r[1]
        } else {
            -1.8 * r[0] + 2.2 * r[2] * r[2] + 0.4 * r[3]
        };
        x.extend(r);
        y.push(target);
    }
    (x, y)
}

fn read_table(path: &Path) -> Result<(Vec<f32>, Vec<f32>, String)> {
    let csv = fs::read_to_string(path)?;
    let mut lines = csv.lines();
    ensure!(
        matches!(lines.next(), Some("x0,x1,x2,x3,y" | "x0,x1,x2,x3,target")),
        "unexpected diagnostic CSV header"
    );
    let mut x = Vec::new();
    let mut y = Vec::new();
    for line in lines {
        let values = line
            .split(',')
            .map(str::parse::<f32>)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        ensure!(
            values.len() == 5 && values.iter().all(|v| v.is_finite()),
            "invalid diagnostic CSV row"
        );
        x.extend_from_slice(&values[..4]);
        y.push(values[4]);
    }
    ensure!(y.len() == ROWS, "diagnostic requires exactly 96 rows");
    Ok((x, y, csv))
}

fn same(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

fn stages_same(a: &StageOutput, b: &StageOutput) -> bool {
    same(&a.v256, &b.v256)
        && same(&a.col, &b.col)
        && same(&a.dope629, &b.dope629)
        && same(&a.lm48, &b.lm48)
        && same(&a.lm48_z, &b.lm48_z)
        && same(&a.tokens, &b.tokens)
        && same(&a.mask, &b.mask)
}

fn main() -> Result<()> {
    let args = std::env::args_os().skip(1).collect::<Vec<_>>();
    ensure!(
        (2..=3).contains(&args.len()),
        "usage: visual_trace RELEASE_DIR_OR_SIGNED_APPLIANCE OUT_DIR [ENGINE_CSV_DIR]"
    );
    let asset_path = Path::new(&args[0]);
    let out = Path::new(&args[1]);
    ensure!(
        !out.join("trace.json").exists(),
        "refuse to replace an existing capture"
    );
    let bundle = if asset_path.is_dir() {
        release::load_r882_release(asset_path)?
    } else {
        let package =
            jope_assets::Package::open(asset_path)?.context("signed appliance required")?;
        package.verify_full()?;
        release::load_r882_embedded(&package)?
    };
    let graph = FusedForward::from_bundle(&bundle, DeviceRequest::Cpu, 1)?;
    let mut assets = serde_json::Map::new();
    for name in [
        "dope.safetensors",
        "jope.safetensors",
        "manifest.json",
        "featurizer.json",
    ] {
        let bytes = bundle
            .asset(name)
            .with_context(|| format!("missing authenticated asset {name}"))?;
        assets.insert(
            name.into(),
            json!({"sha256":digest(bytes),"bytes":bytes.len()}),
        );
    }
    fs::create_dir_all(out)?;
    let mut cases = Vec::new();
    for index in 0..2 {
        let id = if index == 0 { "case-a" } else { "case-b" };
        let (full_x, full_y, csv) = if let Some(input) = args.get(2) {
            read_table(&Path::new(input).join(format!("{id}.csv")))?
        } else {
            let (x, y) = table(index);
            let mut csv = String::from("x0,x1,x2,x3,y\n");
            for (r, target) in x.chunks_exact(FEATURES).zip(&y) {
                csv.push_str(&format!(
                    "{:.9},{:.9},{:.9},{:.9},{:.9}\n",
                    r[0], r[1], r[2], r[3], target
                ));
            }
            (x, y, csv)
        };
        let (fit_ids, holdout_ids) = if args.len() == 3 {
            jope_service::discover::splits::random_split(ROWS, 76, 20261103, "public-run/fit-v1")?
        } else {
            ((0..ROWS).collect(), Vec::new())
        };
        let x = fit_ids
            .iter()
            .flat_map(|i| full_x[i * FEATURES..(i + 1) * FEATURES].iter().copied())
            .collect::<Vec<_>>();
        let y = fit_ids.iter().map(|i| full_y[*i]).collect::<Vec<_>>();
        let fit_rows = fit_ids.len();
        let rows = x
            .chunks_exact(FEATURES)
            .map(|r| r.iter().map(|v| f64::from(*v)).collect())
            .collect::<Vec<Vec<f64>>>();
        let targets = y.iter().map(|v| f64::from(*v)).collect::<Vec<_>>();
        fs::write(out.join(format!("{id}.csv")), &csv)?;
        let descriptors = fit_transform_and_extract_bundle_high(&x, fit_rows, FEATURES, &y)?;
        let col = descriptors
            .columns()
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        let featurizer = Featurizer::default();
        let programs = (0..12)
            .map(|i| {
                format!(
                    "(pipe X (standardize) (ridge lambda_frac={}))",
                    0.001 * (i + 1) as f64
                )
            })
            .collect::<Vec<_>>();
        let mut prog = Vec::new();
        let mut proj = Vec::new();
        let mut fitness = Vec::new();
        for (i, program) in programs.iter().enumerate() {
            let features = featurizer.featurize(program)?;
            proj.extend(proj16(&features.map(f64::from)).map(|v| v as f32));
            prog.extend(features);
            if i < 4 {
                let model =
                    feat_math::model::fit(&feat_math::GenomeIr::parse(program)?, &rows, &targets)?;
                let predictions = model.predict(&rows)?;
                let mse = predictions
                    .iter()
                    .zip(&targets)
                    .map(|(p, t)| (p - t).powi(2))
                    .sum::<f64>()
                    / fit_rows as f64;
                fitness.push(mse.sqrt() as f32);
            }
        }
        let infer = || -> Result<_> {
            let stages = graph.stages(descriptors.vector(), &col)?;
            let costs = graph.predict_costs(PredictCostsInputs {
                batch: 1,
                candidates: 12,
                prog_412: &prog,
                proj_16: &proj,
                tokens: &stages.tokens,
                token_mask: &stages.mask,
                revealed_indices: &[0, 1, 2, 3],
                revealed_fitness: &fitness,
                revealed_gates: &[1.; 4],
                ctx_feat: &[0.; 584],
                base_prior_cost: &[0.; 12],
                sos_prior: &[0.; 24],
                scale: SCALE,
            })?;
            Ok((stages, costs))
        };
        let before = Instant::now();
        let reference = infer()?;
        let reference_elapsed_ns = before.elapsed().as_nanos().to_string();
        let ((stages, costs), trace) = capture(infer)?;
        let parity = stages_same(&reference.0, &stages)
            && same(&reference.1.total_cost, &costs.total_cost)
            && same(&reference.1.anchor_cost, &costs.anchor_cost)
            && reference.1.proposal_kl_to_anchor == costs.proposal_kl_to_anchor
            && reference.1.exceeds_kl_limit == costs.exceeds_kl_limit;
        ensure!(parity, "tracing changed native inference");
        let mut corrupt = trace
            .affine
            .first()
            .context("no affine observations")?
            .clone();
        corrupt.output[0] += 1.;
        ensure!(
            validate_affine(&corrupt).is_err(),
            "corrupt output passed closure"
        );
        ensure!(
            same(&stages.dope629[..256], descriptors.vector()),
            "descriptor bypass differs"
        );
        let preview = x
            .chunks_exact(FEATURES)
            .zip(&y)
            .take(5)
            .map(|(r, target)| r.iter().copied().chain([*target]).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        println!("PASS {id}: {} affine ops; native traced/untraced outputs bitwise equal; corrupted tap rejected", trace.affine.len());
        cases.push(json!({
            "id":id,"dataset_sha256":digest(csv.as_bytes()),"rows":fit_rows,"input_rows":ROWS,"features":FEATURES,
            "fit_row_ids":fit_ids,"holdout_row_ids":holdout_ids,"split_seed":if args.len()==3 {Some(20261103_u64)} else {None},
            "input_header":csv.lines().next().unwrap().split(',').collect::<Vec<_>>(),"table_preview":preview,
            "diagnostic":true,"scope":"native frozen fused inference on recorded FIT rows of synthetic regression table; manually constructed candidate slate; not product search or output lineage",
            "source":null,"tokens_per_second":null,"reference_elapsed_ns":reference_elapsed_ns,
            "timing_scope":"one local CPU descriptor-conditioned fused graph call; excludes descriptors, asset load, tracing, source emission and verification; single diagnostic sample",
            "outputs":{"dope629":stages.dope629,"tokens":stages.tokens,"costs":costs.total_cost},
            "trace":trace,"validation":{"trace_on_off_bitwise_equal":parity,"closure_negative_control_rejected":true,"descriptor_passthrough_equal":true},
            "candidate_input_scope":"12 declared standardize/ridge programs; first four fitness inputs measured FIT RMSE; no holdout/search/quality claim"
        }));
    }
    let source_commit = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|r| r.status.success())
        .map(|r| String::from_utf8_lossy(&r.stdout).trim().to_owned());
    let payload = json!({"schema":"jope.visual-trace-diagnostic/v1","bundle_digest":bundle.digest().to_string(),
        "source_commit":source_commit,"assets":assets,"placement":graph.placement(),"cases":cases,
        "capture_authority":"native FusedForward::stages and CpuAdapter::predict_costs; weights enter only through authenticated release loader",
        "public_release":false,"timing":"recorded inference; cinematic motion is explanatory time expansion"});
    fs::write(
        out.join("trace.json.partial"),
        serde_json::to_vec(&payload)?,
    )?;
    fs::rename(out.join("trace.json.partial"), out.join("trace.json"))?;
    println!("PRIVATE trace: {}", out.join("trace.json").display());
    Ok(())
}
