//! Reproduce bounded TRAIN-only B05–B07 utility searches and artifact inventories.
#[path = "scalar_utility_v2/worker.rs"]
mod worker;
use feat_math::evolve::FaultPolicy;
use feat_math::kernel::scalar_program::{
    ScalarCarrier, ScalarLanguage, ScalarProducerKind, ScalarProducerProfile,
};
use feat_math::{metrics as regression_metrics, Rows};
#[path = "scalar_utility_v2/metrics.rs"]
mod utility_metrics;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{fs, path::Path, time::Instant};

fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn profile(card: usize, width: usize, enabled: bool) -> Result<ScalarProducerProfile, String> {
    let kind = [
        ScalarProducerKind::InverseLinks,
        ScalarProducerKind::StableExponentials,
        ScalarProducerKind::NumericalPolicy,
    ][card];
    let profile = ScalarProducerProfile::new(kind, ScalarCarrier::Row, width)?;
    // No preceding B05/B06 promotion is qualified or owner-authorized yet.
    // Both arms therefore retain the currently accepted Baseline body catalogue;
    // earlier Experimental additions keep their draw slots but stay disabled.
    let profile = match card {
        0 => profile,
        1 => profile.inverse_links_disabled(),
        _ => profile
            .inverse_links_disabled()
            .stable_exponentials_disabled(),
    };
    Ok(if enabled {
        profile
    } else {
        match card {
            0 => profile.inverse_links_disabled(),
            1 => profile.stable_exponentials_disabled(),
            _ => profile.numerical_policy_disabled(),
        }
    })
}

fn bits(values: &Value) -> Result<Vec<f64>, String> {
    values
        .as_array()
        .ok_or("expected bit array")?
        .iter()
        .map(|v| {
            v.as_u64()
                .map(f64::from_bits)
                .ok_or("invalid f64 bits".into())
        })
        .collect()
}

fn cohort_manifest(bytes: &[u8]) -> Result<Value, Box<dyn std::error::Error>> {
    let manifest: Value = serde_json::from_slice(bytes)?;
    if hash(bytes) != "74b05d8be0caa6a595815f9c89deb6be5ab716b1dedcd8a8669c7c7577296840"
        || manifest["schema"] != "feat-math.scalar-utility-common-cohort.v2"
    {
        return Err("unknown common utility cohort".into());
    }
    Ok(manifest)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 5 || args[3].len() != 40 || !args[3].bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(
            "usage: utility COHORT_MANIFEST OUTPUT_DIRECTORY SOURCE_SHA Penalize|Legacy".into(),
        );
    }
    let policy = match args[4].as_str() {
        "Penalize" => FaultPolicy::Penalize {
            lambda_domain: 0.05,
            lambda_saturation: 0.0,
            tie_break: true,
        },
        "Legacy" => FaultPolicy::Legacy,
        _ => return Err("unknown utility fault policy".into()),
    };
    let manifest_path = Path::new(&args[1]);
    let manifest_bytes = fs::read(manifest_path)?;
    let manifest = cohort_manifest(&manifest_bytes)?;
    let datasets = manifest["datasets"]
        .as_array()
        .ok_or("missing utility datasets")?;
    if datasets.len() < 6
        || manifest["search_budget_candidates"] != 72
        || manifest["initial_population"] != 16
        || manifest["inner_folds"] != 3
    {
        return Err("utility cohort or budget mismatch".into());
    }
    let seeds: Vec<_> = manifest["seeds"]
        .as_array()
        .ok_or("missing utility seeds")?
        .iter()
        .map(|seed| seed.as_u64().ok_or("invalid utility seed"))
        .collect::<Result<_, _>>()?;
    if seeds != [3, 17, 29, 101, 701] {
        return Err("utility seed set mismatch".into());
    }
    let out = Path::new(&args[2]);
    fs::create_dir_all(out)?;
    let mut index = vec![];
    let mut measurements = vec![];
    for input in datasets {
        let name = input["name"]
            .as_str()
            .ok_or("missing utility dataset name")?;
        if !name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-') {
            return Err("invalid utility dataset name".into());
        }
        let relative = input["path"].as_str().ok_or("missing input path")?;
        if Path::new(relative).components().count() != 1 {
            return Err("utility input path must be one filename".into());
        }
        let bytes = fs::read(
            manifest_path
                .parent()
                .ok_or("no cohort directory")?
                .join(relative),
        )?;
        let input_hash = hash(&bytes);
        if input["sha256"] != input_hash {
            return Err("utility input hash mismatch".into());
        }
        let data: Value = serde_json::from_slice(bytes)?;
        if data["schema"] != "feat-math.scalar-utility-common-input.v2"
            || data["labels_unchanged"] != true
            || data["worker_train_count"] != 64
            || data["report_train_internal_count"] != 32
        {
            return Err("utility input contract mismatch".into());
        }
        let rows: Rows = data["rows_bits"]
            .as_array()
            .ok_or("missing rows")?
            .iter()
            .map(bits)
            .collect::<Result<_, _>>()?;
        let labels = bits(&data["labels_bits"])?;
        if rows.len() != 96
            || labels.len() != 96
            || rows.iter().any(|row| {
                row.is_empty() || row.len() != rows[0].len() || row.iter().any(|v| !v.is_finite())
            })
            || labels.iter().any(|v| !v.is_finite() || v.abs() > 1e12)
        {
            return Err("utility numeric shape".into());
        }
        let training = rows[..64].to_vec();
        let report_rows = rows[64..].to_vec();
        // Fold membership depends only on the worker's training cut. Dataset
        // and reporting identities remain metadata outside its input closure.
        let fold_key = worker::training_fold_key(&training, &labels[..64]);
        for card in 0..3 {
            // Manual old-formula controls share training-only folds/objectives,
            // but remain outside the equal-budget GP population and RNG streams.
            let name_card = ["B05", "B06", "B07"][card];
            let baseline = profile(card, rows[0].len(), false)?;
            let (controls, comparisons) = worker::compare_existing_formulas(
                card,
                &baseline,
                &training,
                &labels[..64],
                policy,
            )?;
            let control_dir = out.join(format!("{name}-{name_card}-manual-controls"));
            fs::create_dir(&control_dir)?;
            fs::write(
                control_dir.join("comparison.json"),
                serde_json::to_vec_pretty(&comparisons)?,
            )?;
            let mut control_receipts = vec![];
            for (formula_name, model) in controls {
                let directory = control_dir.join(&formula_name);
                fs::create_dir(&directory)?;
                let state = model.to_json()?;
                let (predictions, faults) = model.predict(&report_rows)?;
                if model.to_json()? != state {
                    return Err("formula reporting changed fitted model".into());
                }
                let (features, feature_faults) = model.transform(&report_rows)?;
                if faults != feature_faults {
                    return Err("formula report fault disagreement".into());
                }
                let report = json!({"r2_bits":regression_metrics::r2(&labels[64..],&predictions).to_bits(),"predictions_bits":predictions.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"features_bits":features.iter().map(|r|r.iter().map(|v|v.to_bits()).collect::<Vec<_>>()).collect::<Vec<_>>(),"faults":faults,"report_rows":32,"training_internal_report":true});
                fs::write(directory.join("winner.json"), &state)?;
                fs::write(
                    directory.join("report.json"),
                    serde_json::to_vec_pretty(&report)?,
                )?;
                let mut artifacts = vec![];
                for (language, extension, token_language) in [
                    (ScalarLanguage::Rust, "rs", feat_math::tokens::Lang::Rust),
                    (
                        ScalarLanguage::Python,
                        "py",
                        feat_math::tokens::Lang::Python,
                    ),
                    (ScalarLanguage::Cpp17, "cpp", feat_math::tokens::Lang::Cpp),
                ] {
                    let source = model.emit_recipe(language)?;
                    let tokens = feat_math::tokens::count_tokens(&source, token_language);
                    fs::write(directory.join(format!("recipe.{extension}")), &source)?;
                    artifacts.push(json!({"language":format!("{language:?}"),"sha256":hash(source.as_bytes()),"bytes":source.len(),"code_tokens":tokens.code_tokens,"doc_tokens":tokens.doc_tokens}));
                }
                let fitted = model.emit_fitted_artifact()?;
                fs::write(directory.join("fitted.cpp"), &fitted.source)?;
                fs::write(directory.join("fitted-codec.json"), &fitted.codec)?;
                control_receipts.push(json!({"source_sha":args[3],"cohort_manifest_sha256":hash(&manifest_bytes),"dataset_sha256":input_hash,"training_only_fold_key":fold_key,"card":name_card,"formula":formula_name,"fault_policy":args[4],"execution_route":"manual-unbound-native-old-formula; no producer admission claim","model_sha256":hash(state.as_bytes()),"recipes":artifacts,"fitted_source_sha256":fitted.source_sha256,"fitted_codec_sha256":fitted.codec_sha256,"fitted_source_bytes":fitted.source.len(),"fitted_codec_bytes":fitted.codec.len(),"emitted_sources_executed":false,"status":"manual native controls; exact four-lane execution pending"}));
            }
            fs::write(
                control_dir.join("native-controls-index.json"),
                serde_json::to_vec_pretty(&control_receipts)?,
            )?;
            for &seed in &seeds {
                for enabled in [false, true] {
                    let name_card = ["B05", "B06", "B07"][card];
                    let arm = if enabled { "B" } else { "A" };
                    let directory = out.join(format!("{name}-{name_card}-{seed}-{arm}"));
                    fs::create_dir(&directory)?;
                    let declaration = profile(card, rows[0].len(), enabled)?;
                    let start = Instant::now();
                    // Only the training cut enters the worker. Reporting labels
                    // and rows are introduced after immutable selection/refit.
                    let result =
                        worker::search(&declaration, &training, &labels[..64], seed, 72, policy)?;
                    let wall = start.elapsed().as_secs_f64();
                    let before = result.model.to_json()?;
                    let (predictions, faults) = result.model.predict(&report_rows)?;
                    if result.model.to_json()? != before {
                        return Err("utility report changed fitted model".into());
                    }
                    let (features, feature_faults) = result.model.transform(&report_rows)?;
                    if faults != feature_faults {
                        return Err("utility report fault disagreement".into());
                    }
                    let report = json!({"r2_bits":regression_metrics::r2(&labels[64..],&predictions).to_bits(),"predictions_bits":predictions.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"features_bits":features.iter().map(|r|r.iter().map(|v|v.to_bits()).collect::<Vec<_>>()).collect::<Vec<_>>(),"faults":faults,"report_rows":32,"training_internal_report":true});
                    let deterministic = serde_json::to_vec_pretty(&result.deterministic)?;
                    fs::write(directory.join("search.json"), &deterministic)?;
                    fs::write(directory.join("winner.json"), &before)?;
                    fs::write(
                        directory.join("report.json"),
                        serde_json::to_vec_pretty(&report)?,
                    )?;
                    fs::write(
                        directory.join("wall.json"),
                        serde_json::to_vec_pretty(
                            &json!({"search_seconds":wall,"search_seconds_per_attempt":wall/72.0}),
                        )?,
                    )?;
                    let mut artifacts = vec![];
                    for (language, extension, token_language) in [
                        (ScalarLanguage::Rust, "rs", feat_math::tokens::Lang::Rust),
                        (
                            ScalarLanguage::Python,
                            "py",
                            feat_math::tokens::Lang::Python,
                        ),
                        (ScalarLanguage::Cpp17, "cpp", feat_math::tokens::Lang::Cpp),
                    ] {
                        let source = result.model.emit_recipe(language)?;
                        fs::write(directory.join(format!("recipe.{extension}")), &source)?;
                        let tokens = feat_math::tokens::count_tokens(&source, token_language);
                        artifacts.push(json!({"language":format!("{language:?}"),"sha256":hash(source.as_bytes()),"bytes":source.len(),"code_tokens":tokens.code_tokens,"doc_tokens":tokens.doc_tokens}));
                    }
                    let fitted = result.model.emit_fitted_artifact()?;
                    fs::write(directory.join("fitted.cpp"), &fitted.source)?;
                    fs::write(directory.join("fitted-codec.json"), &fitted.codec)?;
                    let receipt = json!({"source_sha":args[3],"cohort_manifest_sha256":hash(&manifest_bytes),"dataset_sha256":input_hash,"training_only_fold_key":fold_key,"cut_id":"first64-training-only/report-next32-training-internal","evaluator":"native-scalar-ridge-inner-cv-v2","training_rows_bits_sha256":hash(serde_json::to_string(&data["rows_bits"].as_array().unwrap()[..64])?.as_bytes()),"training_labels_bits_sha256":hash(serde_json::to_string(&data["labels_bits"].as_array().unwrap()[..64])?.as_bytes()),"card":name_card,"arm":arm,"seed":seed,"fault_policy":args[4],"profile_sha256":declaration.sha256(),"search_sha256":hash(&deterministic),"model_sha256":hash(before.as_bytes()),"recipes":artifacts,"fitted_source_sha256":fitted.source_sha256,"fitted_codec_sha256":fitted.codec_sha256,"fitted_source_bytes":fitted.source.len(),"fitted_codec_bytes":fitted.codec.len(),"emitted_sources_executed":false,"status":"native search and emitted sources only; four-lane execution/gates pending"});
                    fs::write(
                        directory.join("receipt.json"),
                        serde_json::to_vec_pretty(&receipt)?,
                    )?;
                    measurements.push(utility_metrics::Run {
                        dataset: name.to_owned(),
                        card,
                        enabled,
                        seed,
                        search: result.deterministic,
                        receipt: receipt.clone(),
                        wall,
                    });
                    index.push(receipt);
                }
            }
        }
    }
    fs::write(
        out.join("native-search-index.json"),
        serde_json::to_vec_pretty(&index)?,
    )?;
    let summary = utility_metrics::summarize(&measurements)?;
    fs::write(
        out.join("utility-metrics.json"),
        serde_json::to_vec_pretty(&summary)?,
    )?;
    println!(
        "{} native searches recorded; emitted execution and promotion gates remain pending",
        index.len()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn committed_cohort_is_admitted_by_entrypoint_guard() {
        let bytes = include_bytes!("../tests/fixtures/scalar_v2/utility/manifest.json");
        let admitted = cohort_manifest(bytes).unwrap();
        assert_eq!(
            admitted["schema"],
            "feat-math.scalar-utility-common-cohort.v2"
        );
        assert_eq!(admitted["datasets"].as_array().unwrap().len(), 8);
        let mut changed = bytes.to_vec();
        changed.push(b'\n');
        assert!(cohort_manifest(&changed).is_err());
    }
}
