//! Equal-budget native fixtures with explicit cost and fault accounting.
use feat_math::advanced_v3::RawTable;
use feat_math::kernel::ordered_axis::{FoldBoundary, GroupKey, OrderedInputIdentity, TaskMode};
use feat_math::kernel::timescale_nabla::OrderedRow;
use feat_math::kernel::timescale_nabla::{ExperimentalProfile, TimeGradientSearch};
use feat_math::kernel::wire_v2::VersionedProgram;
use feat_math::{GenomeIr, Pcg32};
use serde_json::{json, Value};
use std::{collections::BTreeSet, fs, time::Instant};

fn inputs(data: &Value) -> (Vec<OrderedRow>, Vec<f64>) {
    let raw = data["rows"].as_array().unwrap();
    let rows: Vec<OrderedRow> = raw
        .iter()
        .map(|row| OrderedRow {
            group: GroupKey::categorical(row["group"].as_str().unwrap()).unwrap(),
            timestamp: f64::from_bits(row["timestamp_bits"].as_u64().unwrap()),
            values: row["values_bits"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| f64::from_bits(v.as_u64().unwrap()))
                .collect(),
        })
        .collect();
    let labels: Vec<f64> = raw
        .iter()
        .map(|r| f64::from_bits(r["label_bits"].as_u64().unwrap()))
        .collect();
    (rows, labels)
}

fn search(data: &Value, seed: u64, enabled: bool) -> Value {
    let (rows, labels) = inputs(data);
    let train = data["inner_train_end"].as_u64().unwrap() as usize;
    let width = data["width"].as_u64().unwrap() as usize;
    let identity = OrderedInputIdentity::new("series", "sample_time");
    let profile = ExperimentalProfile {
        time_gradient: enabled,
        prior: 0.125,
    };
    let mut rng = Pcg32::new(seed, 0x7574696c696679);
    let mut evaluations = Vec::new();
    let mut unique = BTreeSet::new();
    let mut ranking = Vec::new();
    for index in 0..64 {
        let start = Instant::now();
        let lambda =
            [0.0, 0.000001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0][rng.next_u32() as usize % 8];
        let downstream =
            GenomeIr::parse(&format!("(pipe X (ridge lambda_frac={lambda}))")).unwrap();
        let search =
            TimeGradientSearch::new(width, identity.clone(), downstream, TaskMode::Forecasting)
                .unwrap();
        let candidate = search.generate(rng.next_u32() as u64, profile).unwrap();
        let bytes = candidate.to_bytes().unwrap();
        let decoded = VersionedProgram::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.to_bytes().unwrap(), bytes);
        unique.insert(bytes.clone());
        let mut fold_losses = Vec::new();
        let mut prediction_words = Vec::new();
        let mut fault_cells = 0;
        let order = match &candidate {
            VersionedProgram::TimeGradientV2(p) => Some(p.config().minimum_gap()),
            _ => None,
        };
        for (fold_index, (train, total)) in [(64, 96), (96, 128)].into_iter().enumerate() {
            let (predictions, faults) =
                evaluate(&candidate, &rows, &labels, train, total, fold_index);
            assert_eq!(predictions.len(), total - train);
            assert!(predictions.iter().all(|v| v.is_finite()));
            let loss = predictions
                .iter()
                .zip(&labels[train..total])
                .map(|(p, y)| (p - y).powi(2))
                .sum::<f64>()
                / (total - train) as f64;
            fold_losses.push(loss);
            fault_cells += faults;
            prediction_words.push(predictions.iter().map(|p| p.to_bits()).collect::<Vec<_>>());
        }
        let (domain_cells, saturation_cells, evaluated_cells) =
            stage_fault_metrics(&candidate, &rows, width);
        let nodes = if order.is_some() { 4 } else { 3 };
        let depth = if order.is_some() { 3 } else { 2 };
        assert!(nodes <= 15 && depth <= 4);
        let elapsed = start.elapsed().as_nanos();
        let mse = fold_losses.iter().sum::<f64>() / fold_losses.len() as f64;
        assert!(mse.is_finite());
        // Finite fixture cells have no domain/saturation faults. Boundary faults
        // arise in training; all groups already have enough continuation here.
        assert_eq!(fault_cells, 0);
        ranking.push((mse, index));
        evaluations.push(json!({"evaluation":index+1,"mse":mse,"wire75_minimum_gap":order,
            "program_bytes":bytes.len(),"program_wire":bytes,"stages":if order.is_some(){2}else{1},"wall_ns":elapsed,"validation_fault_cells":fault_cells,
            "fold_losses":fold_losses,"prediction_words":prediction_words,"domain_cells":domain_cells,
            "saturation_cells":saturation_cells,"evaluated_cells":evaluated_cells,"nodes":nodes,"depth":depth,
            "resource_admission_prototype_fits":1+usize::from(order.is_some())}));
    }
    ranking.sort_by(|a, b| {
        a.0.total_cmp(&b.0)
            .then(
                evaluations[a.1]["stages"]
                    .as_u64()
                    .cmp(&evaluations[b.1]["stages"].as_u64()),
            )
            .then(a.1.cmp(&b.1))
    });
    let elite = &ranking[..16];
    let best = ranking[0];
    let final_best = best.0;
    // Error distance from final best, rather than treating a loss as a reward.
    let first_loss = evaluations[0]["mse"].as_f64().unwrap();
    let threshold = final_best + 0.05 * (first_loss - final_best);
    let target_eval = evaluations
        .iter()
        .position(|r| r["mse"].as_f64().unwrap() <= threshold)
        .unwrap()
        + 1;
    let winner_wire = evaluations[best.1]["program_wire"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u8)
        .collect::<Vec<_>>();
    let winner = VersionedProgram::from_bytes(&winner_wire).unwrap();
    let cost = emitted_cost(&winner, &rows, &labels, train, width);
    let uses = elite
        .iter()
        .filter(|(_, i)| !evaluations[*i]["wire75_minimum_gap"].is_null())
        .count();
    json!({"dataset":data["id"],"seed":seed,"arm":if enabled {"B"} else {"A"},
        "evaluations":64,"inner_folds":[[64,96],[96,128]],"fitness_fits":128,"population_archive":16,"max_nodes":15,"max_depth":4,
        "best_two_fold_cv_mse":best.0,"best_inner_validation_mse":best.0,"top10_median_mse":(ranking[4].0+ranking[5].0)/2.0,
        "evaluations_to_95pct_final_loss_improvement":target_eval,
        "hard_rejects":0,"invalid_offspring":0,"unique_children":unique.len(),
        "new_token_elite_share":uses as f64/16.0,
        "elite_mean_nodes":elite.iter().map(|(_,i)|evaluations[*i]["nodes"].as_u64().unwrap() as f64).sum::<f64>()/16.,
        "elite_mean_depth":elite.iter().map(|(_,i)|evaluations[*i]["depth"].as_u64().unwrap() as f64).sum::<f64>()/16.,
        "domain_cells":evaluations.iter().map(|e|e["domain_cells"].as_u64().unwrap()).sum::<u64>(),
        "saturation_cells":evaluations.iter().map(|e|e["saturation_cells"].as_u64().unwrap()).sum::<u64>(),
        "evaluated_cells":evaluations.iter().map(|e|e["evaluated_cells"].as_u64().unwrap()).sum::<u64>(),
        "wall_ns_per_evaluation":evaluations.iter().map(|e|e["wall_ns"].as_u64().unwrap()).sum::<u64>()/64,
        "resource_admission_prototype_fits":evaluations.iter().map(|e|e["resource_admission_prototype_fits"].as_u64().unwrap()).sum::<u64>(),
        "emitted_cost":cost,"winner":evaluations[best.1],"trace":evaluations})
}

fn evaluate(
    candidate: &VersionedProgram,
    rows: &[OrderedRow],
    labels: &[f64],
    train: usize,
    total: usize,
    fold: usize,
) -> (Vec<f64>, usize) {
    match candidate {
        VersionedProgram::TimeGradientV2(program) => {
            let mut model = program
                .fit_training(
                    &rows[..train],
                    &labels[..train],
                    TaskMode::Forecasting,
                    FoldBoundary::new(format!("utility-inner-{fold}"), train, total).unwrap(),
                )
                .unwrap();
            let output = model.predict_with_faults(&rows[train..total]).unwrap();
            let faults = output.faults.iter().flatten().filter(|&&v| v != 0).count();
            model.finish().unwrap();
            (output.values, faults)
        }
        VersionedProgram::LegacyV1(feat_math::kernel::Program::LegacyV1(genome)) => {
            let x: Vec<_> = rows[..train].iter().map(|r| r.values.clone()).collect();
            let model = feat_math::fit_legacy_numeric_bound(genome, &x, &labels[..train]).unwrap();
            let table = RawTable {
                headers: model
                    .schema
                    .columns
                    .iter()
                    .map(|c| c.name.clone())
                    .collect(),
                rows: rows[train..total]
                    .iter()
                    .map(|r| r.values.iter().map(f64::to_string).collect())
                    .collect(),
            };
            (model.predict_table(&table).unwrap(), 0)
        }
        _ => unreachable!(),
    }
}
fn stage_fault_metrics(
    candidate: &VersionedProgram,
    rows: &[OrderedRow],
    _width: usize,
) -> (usize, usize, usize) {
    use feat_math::math_expr::{FaultSet, FaultSeverity};
    let mut domain = 0;
    let mut saturation = 0;
    let mut cells = 0;
    for (train, total) in [(64, 96), (96, 128)] {
        match candidate {
            VersionedProgram::TimeGradientV2(program) => {
                let mut state = feat_math::kernel::timescale_nabla::TimeGradientState::new(
                    program.config().clone(),
                    TaskMode::Forecasting,
                    program.input_identity(),
                    FoldBoundary::new("fault-audit", train, total).unwrap(),
                )
                .unwrap();
                for row in &rows[..total] {
                    let output = state
                        .push(row.group.clone(), row.timestamp, &row.values)
                        .unwrap();
                    for flag in output.faults {
                        let scalar = FaultSet::from_bits(flag & FaultSet::ALL.bits()).unwrap();
                        domain += usize::from(
                            scalar.iter().any(|c| c.severity() == FaultSeverity::Domain),
                        );
                        saturation += usize::from(
                            scalar
                                .iter()
                                .any(|c| c.severity() == FaultSeverity::Saturation),
                        );
                        cells += 1;
                    }
                }
                state.finish().unwrap();
            }
            VersionedProgram::LegacyV1(_) => {
                for row in &rows[..total] {
                    assert!(row.values.iter().all(|x| x.is_finite() && x.abs() < 1e12));
                    cells += row.values.len();
                }
            }
            _ => unreachable!(),
        }
    }
    (domain, saturation, cells)
}
fn source_cost(source: &str, lang: feat_math::tokens::Lang) -> Value {
    let tokens = feat_math::tokens::count_tokens(source, lang);
    json!({"bytes":tokens.bytes,"code_tokens":tokens.code_tokens,"doc_tokens":tokens.doc_tokens,"total_tokens":tokens.code_tokens+tokens.doc_tokens})
}
fn emitted_cost(
    candidate: &VersionedProgram,
    rows: &[OrderedRow],
    labels: &[f64],
    train: usize,
    width: usize,
) -> Value {
    use feat_math::kernel::timescale_nabla::TimeGradientLanguage as Language;
    use feat_math::tokens::Lang;
    let (rust, python, cpp, fitted, prototype_fits) = match candidate {
        VersionedProgram::TimeGradientV2(program) => {
            let model = program
                .fit_training(
                    &rows[..train],
                    &labels[..train],
                    TaskMode::Forecasting,
                    FoldBoundary::new("utility-artifact", train, rows.len()).unwrap(),
                )
                .unwrap();
            let fitted = model.emit_fitted_cpp17().unwrap();
            let replay =
                feat_math::kernel::timescale_nabla::TimeGradientFittedCpp::from_codec_json(
                    &fitted.codec,
                )
                .unwrap();
            assert_eq!(replay.source, fitted.source);
            (
                program.emit_recipe(Language::Rust).unwrap().source,
                program.emit_recipe(Language::Python).unwrap().source,
                program.emit_recipe(Language::Cpp17).unwrap().source,
                fitted.source,
                3,
            )
        }
        VersionedProgram::LegacyV1(feat_math::kernel::Program::LegacyV1(genome)) => {
            let x: Vec<_> = rows[..train].iter().map(|r| r.values.clone()).collect();
            let model = feat_math::fit_legacy_numeric_bound(genome, &x, &labels[..train]).unwrap();
            let opts = feat_math::transpile::EmitOptions::default();
            let recipe = feat_math::transpile::emit(genome, &opts).unwrap();
            (
                recipe.rust,
                recipe.python,
                feat_math::kernel::expression_recipe::emit_cpp17(genome, &opts)
                    .unwrap()
                    .source,
                feat_math::advanced_v3::emit_fitted_cpp17(&model)
                    .unwrap()
                    .source,
                0,
            )
        }
        _ => unreachable!(),
    };
    assert!(width > 0);
    json!({"rust":source_cost(&rust,Lang::Rust),"python":source_cost(&python,Lang::Python),"cpp17":source_cost(&cpp,Lang::Cpp),"fitted_cpp17":source_cost(&fitted,Lang::Cpp),"posthoc_winner_fit":1,"ordered_recipe_prototype_fits":prototype_fits,"within_fitness_budget":false})
}

fn fixture_utility(index: usize) {
    use sha2::{Digest, Sha256};
    let corpus = include_str!("../fixtures/math_boundary/v2/timescale_nabla_utility.json");
    let fixtures: Vec<Value> = serde_json::from_str(corpus).unwrap();
    assert_eq!(fixtures.len(), 6);
    let data = &fixtures[index];
    let mut runs = Vec::new();
    for seed in 0..5 {
        for enabled in [false, true] {
            runs.push(search(data, seed, enabled));
        }
    }
    assert_eq!(runs.len(), 10);
    assert_eq!(
        runs.iter()
            .map(|r| r["evaluations"].as_u64().unwrap())
            .sum::<u64>(),
        640
    );
    assert_eq!(
        runs.iter()
            .map(|r| r["fitness_fits"].as_u64().unwrap())
            .sum::<u64>(),
        1280
    );
    assert!(runs
        .iter()
        .all(|r| r["domain_cells"] == 0 && r["saturation_cells"] == 0));
    let (rows, labels) = inputs(data);
    let train = data["inner_train_end"].as_u64().unwrap() as usize;
    let dir = tempfile::tempdir().unwrap().keep();
    let mut winners = BTreeSet::new();
    for run in &runs {
        let bytes = run["winner"]["program_wire"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u8)
            .collect::<Vec<_>>();
        if winners.insert(bytes.clone()) {
            let node = VersionedProgram::from_bytes(&bytes).unwrap();
            super::utility_exports::check(
                &node,
                &rows,
                &labels,
                train,
                &dir,
                &format!("winner{}", winners.len()),
            );
        }
    }
    assert!(!winners.is_empty());
    let ordered_winners = winners
        .iter()
        .filter(|bytes| {
            matches!(
                VersionedProgram::from_bytes(bytes).unwrap(),
                VersionedProgram::TimeGradientV2(_)
            )
        })
        .count();
    let paired_deltas: Vec<Value> = runs.chunks_exact(2).map(|pair| {
        assert_eq!(pair[0]["seed"], pair[1]["seed"]);
        let costs: serde_json::Map<String, Value> = ["rust", "python", "cpp17", "fitted_cpp17"].into_iter().map(|lang| {
            let row: serde_json::Map<String, Value> = ["bytes", "code_tokens", "doc_tokens", "total_tokens"].into_iter().map(|metric| {
                let a = pair[0]["emitted_cost"][lang][metric].as_i64().unwrap();
                let b = pair[1]["emitted_cost"][lang][metric].as_i64().unwrap();
                (metric.to_string(), json!(b-a))
            }).collect();
            (lang.to_string(), json!(row))
        }).collect();
        json!({"seed":pair[0]["seed"],"mse_delta":pair[1]["best_two_fold_cv_mse"].as_f64().unwrap()-pair[0]["best_two_fold_cv_mse"].as_f64().unwrap(),
            "token_byte_delta":costs})
    }).collect();
    // Artifact fits are post-selection and reported outside both search budgets.
    let report = json!({"scope":"synthetic training fixtures; Experimental only",
        "dataset":data["id"],"dataset_sha256":format!("{:x}",Sha256::digest(serde_json::to_vec(data).unwrap())),
        "corpus_sha256":format!("{:x}",Sha256::digest(corpus.as_bytes())),
        "enabled_profile":ExperimentalProfile {time_gradient:true,prior:0.125}.identity().unwrap(),
        "disabled_profile":ExperimentalProfile {time_gradient:false,prior:0.125}.identity().unwrap(),
        "fold_identity":"inner-64-96-and-96-128; artifact-prefix96",
        "runs":runs,"distinct_winners":winners.len(),"compiled_executed_artifacts":winners.len()*4,
        "winner_artifact_training_fits":winners.len()*4,"winner_artifact_recipe_admission_prototype_fits":ordered_winners*3,
        "paired_wins_ties_losses":{"wins":paired_deltas.iter().filter(|r|r["mse_delta"].as_f64().unwrap()<0.).count(),"ties":paired_deltas.iter().filter(|r|r["mse_delta"].as_f64().unwrap()==0.).count(),"losses":paired_deltas.iter().filter(|r|r["mse_delta"].as_f64().unwrap()>0.).count()},
        "paired_deltas":paired_deltas,"baseline_promotion":false,
        "formal_promotion":"fixed-cohort G-A through G-E remains required",
        "resource_admission_prototype_fits":"reported separately from equal fitness budget"});
    fs::write(
        dir.join("equal-budget-utility.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
    eprintln!(
        "wire75 fixture{} equal-budget+winner receipt: {}",
        index,
        dir.display()
    );
}

#[test]
fn local_speed_five_seed_equal_budget_and_exported_winners() {
    fixture_utility(0);
}
#[test]
fn change_rate_five_seed_equal_budget_and_exported_winners() {
    fixture_utility(1);
}
#[test]
fn irregular_velocity_five_seed_equal_budget_and_exported_winners() {
    fixture_utility(2);
}
#[test]
fn group_slopes_five_seed_equal_budget_and_exported_winners() {
    fixture_utility(3);
}
#[test]
fn pulse_rate_five_seed_equal_budget_and_exported_winners() {
    fixture_utility(4);
}
#[test]
fn level_control_five_seed_equal_budget_and_exported_winners() {
    fixture_utility(5);
}
