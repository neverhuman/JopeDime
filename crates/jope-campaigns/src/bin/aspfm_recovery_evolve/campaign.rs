use super::*;

fn campaign_receipt_path(output: &Path, dataset: &str) -> PathBuf {
    output.join("receipts").join(format!("{dataset}.json"))
}

pub(super) fn run_campaign(campaign: Campaign, args: &Args) -> Result<Value> {
    let receipt_path = campaign_receipt_path(&args.output, &campaign.dataset);
    if receipt_path.is_file() {
        return Ok(serde_json::from_slice(&fs::read(receipt_path)?)?);
    }
    let root = args.output.join("work").join(&campaign.dataset);
    fs::create_dir_all(&root)?;
    let mut ranked = campaign.seeds.clone();
    ranked.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.semantic.cmp(&right.semantic))
    });
    ranked.truncate(POPULATION);
    let mut terminal = None;
    for generation in 1..=GENERATIONS {
        let mut population =
            next_population(&campaign.dataset, &ranked, generation, campaign.features)?;
        evaluate_generation(
            &campaign,
            &args.evaluator,
            &root,
            generation,
            &mut population,
            args.evaluator_threads,
        )?;
        ranked.extend(
            population
                .into_iter()
                .filter(|member| member.score.is_finite()),
        );
        ranked.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.semantic.cmp(&right.semantic))
        });
        ranked.dedup_by(|left, right| left.semantic == right.semantic);
        if let Some(winner) = ranked.iter().find(|member| {
            member.score > campaign.target
                && !score_matches(member.score, campaign.target)
                && member.receipt.is_some()
        }) {
            terminal = Some((generation, winner.clone()));
            break;
        }
        ranked.truncate(POPULATION);
    }
    let receipt = if let Some((generation, winner)) = terminal {
        json!({
            "schema":"sotarena.aspfm.recovery-evolution.v1",
            "status":"target-beaten",
            "task":campaign.task,
            "dataset_hash":campaign.dataset,
            "dataset_id":campaign.dataset_id,
            "historical_target":campaign.target_score,
            "generation":generation,
            "semantic_sha256":winner.semantic,
            "canonical_genome":winner.genome,
            "replayed_score":metric_score(&campaign.task, winner.score),
            "evaluator_receipt":winner.receipt,
        })
    } else {
        json!({
            "schema":"sotarena.aspfm.recovery-evolution.v1",
            "status":"exhausted",
            "task":campaign.task,
            "dataset_hash":campaign.dataset,
            "dataset_id":campaign.dataset_id,
            "historical_target":campaign.target_score,
            "generations":GENERATIONS,
            "best_semantic_sha256":ranked[0].semantic,
            "best_canonical_genome":ranked[0].genome,
            "best_replayed_score":metric_score(&campaign.task, ranked[0].score),
            "best_evaluator_receipt":ranked[0].receipt,
        })
    };
    atomic_json(&receipt_path, &receipt)?;
    Ok(receipt)
}

pub(super) fn build_campaigns(replay: &Value, args: &Args) -> Result<Vec<Campaign>> {
    let task = args.task.as_str();
    let mut targets = BTreeMap::<String, f64>::new();
    let mut accepted = BTreeMap::<String, Vec<Member>>::new();
    for result in replay["results"]
        .as_array()
        .context("replay results missing")?
    {
        let candidate = &result["candidate"];
        if candidate["task"].as_str() != Some(task) {
            continue;
        }
        let dataset = candidate["dataset_hash"]
            .as_str()
            .context("dataset hash missing")?
            .to_owned();
        if let Some(score) = candidate["recorded_score"].as_f64() {
            targets
                .entry(dataset.clone())
                .and_modify(|current| *current = current.max(score))
                .or_insert(score);
        }
        if result["status"].as_str() == Some("accepted") {
            if let (Some(score), Some(genome), Some(semantic)) = (
                result["replayed_score"].as_f64(),
                candidate["canonical_genome"].as_str(),
                candidate["semantic_sha256"].as_str(),
            ) {
                accepted.entry(dataset).or_default().push(Member {
                    genome: genome.to_owned(),
                    semantic: semantic.to_owned(),
                    score: fitness(task, score),
                    receipt: None,
                });
            }
        }
    }
    let recovery_winners = replay["winners"]
        .as_array()
        .context("replay winners missing")?
        .iter()
        .filter(|winner| {
            winner["task"].as_str() == Some(task)
                && winner["evidence_class"].as_str() == Some("adaptive")
                && winner["candidate"]["provenance_family"].as_str() == Some("recovery-replay")
        })
        .filter_map(|winner| {
            Some((
                winner["dataset_hash"].as_str()?.to_owned(),
                winner["replayed_score"].as_f64()?,
            ))
        })
        .collect::<BTreeMap<_, _>>();
    let datasets = replay["datasets"].as_array().context("datasets missing")?;
    let mut campaigns = Vec::new();
    for dataset in datasets
        .iter()
        .filter(|dataset| dataset["task"].as_str() == Some(task))
    {
        let hash = dataset["dataset_hash"].as_str().context("hash missing")?;
        if args.dataset.as_deref().is_some_and(|wanted| wanted != hash) {
            continue;
        }
        let target_score = if task == "regression" {
            args.historical_target
                .context("regression target missing")?
        } else {
            let Some(target) = targets.get(hash).copied() else {
                continue;
            };
            target
        };
        let target = fitness(task, target_score);
        let Some(winner) = recovery_winners.get(hash).copied() else {
            continue;
        };
        if fitness(task, winner) > target || score_matches(winner, target_score) {
            continue;
        }
        let dataset_id = dataset["dataset_id"]
            .as_str()
            .context("dataset id missing")?;
        let train = args.data_root.join(task).join(dataset_id).join("train.csv");
        let test = args.data_root.join(task).join(dataset_id).join("test.csv");
        ensure!(
            train.is_file() && test.is_file(),
            "missing split for {hash}"
        );
        let mut seeds = accepted
            .remove(hash)
            .context("no accepted recovery seeds")?;
        seeds.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.semantic.cmp(&right.semantic))
        });
        seeds.dedup_by(|left, right| left.semantic == right.semantic);
        ensure!(seeds.len() >= PARENT_POOL, "too few seeds for {hash}");
        campaigns.push(Campaign {
            task: task.to_owned(),
            dataset: hash.to_owned(),
            dataset_id: dataset_id.to_owned(),
            target_score,
            target,
            train_sha256: dataset["train"]["sha256"]
                .as_str()
                .context("train hash missing")?
                .to_owned(),
            test_sha256: dataset["test"]["sha256"]
                .as_str()
                .context("test hash missing")?
                .to_owned(),
            features: numeric_columns(&train)?,
            train,
            test,
            seeds,
        });
    }
    campaigns.sort_by(|left, right| left.dataset.cmp(&right.dataset));
    Ok(campaigns)
}

pub(super) fn integrate(replay: &mut Value, receipts: &[Value]) -> Result<()> {
    let mut exhaustions = replay["recovery_exhaustions"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    for campaign in receipts {
        let task = campaign["task"].as_str().context("receipt task missing")?;
        let dataset = campaign["dataset_hash"]
            .as_str()
            .context("receipt dataset missing")?;
        let exhausted = campaign["status"].as_str() == Some("exhausted");
        let semantic = campaign[if exhausted {
            "best_semantic_sha256"
        } else {
            "semantic_sha256"
        }]
        .as_str()
        .context("receipt semantic missing")?;
        let genome = campaign[if exhausted {
            "best_canonical_genome"
        } else {
            "canonical_genome"
        }]
        .as_str()
        .context("receipt genome missing")?;
        ensure!(
            sha256(genome.as_bytes()) == semantic,
            "evolution semantic mismatch"
        );
        let score = campaign[if exhausted {
            "best_replayed_score"
        } else {
            "replayed_score"
        }]
        .as_f64()
        .context("receipt score missing")?;
        let evaluator = &campaign[if exhausted {
            "best_evaluator_receipt"
        } else {
            "evaluator_receipt"
        }];
        let winner_index = replay["winners"]
            .as_array()
            .context("winners missing")?
            .iter()
            .position(|winner| {
                winner["task"].as_str() == Some(task)
                    && winner["dataset_hash"].as_str() == Some(dataset)
                    && winner["evidence_class"].as_str() == Some("adaptive")
            })
            .context("adaptive winner missing")?;
        let current_score = replay["winners"][winner_index]["replayed_score"]
            .as_f64()
            .context("winner score missing")?;
        let improves = if task == "regression" {
            score < current_score
        } else {
            score > current_score
        };
        if improves && evaluator.is_object() {
            let mut candidate = replay["winners"][winner_index]["candidate"].clone();
            let source_bytes = serde_json::to_vec(campaign)?;
            candidate["semantic_sha256"] = json!(semantic);
            candidate["canonical_genome"] = json!(genome);
            candidate["recorded_score"] = Value::Null;
            candidate["benchmark_valid"] = json!(false);
            candidate["evidence_class"] = json!("adaptive");
            candidate["provenance_family"] = json!("recovery-replay");
            candidate["source_path"] =
                json!(format!("recovery://evolution/{task}/{dataset}/{semantic}"));
            candidate["source_sha256"] = json!(sha256(&source_bytes));
            candidate["prediction_sha256"] = evaluator["predictions_sha256"].clone();
            candidate["runtime_seconds"] = evaluator["runtime_seconds"].clone();
            let result = json!({
                "candidate":candidate,
                "status":"accepted","reason":null,"replayed_score":score,
                "prediction_sha256":evaluator["predictions_sha256"],
                "prediction_count":evaluator["prediction_count"],
                "runtime_seconds":evaluator["runtime_seconds"],
                "train_sha256":evaluator["train_sha256"],"test_sha256":evaluator["test_sha256"],
            });
            replay["results"]
                .as_array_mut()
                .context("results missing")?
                .push(result.clone());
            replay["winners"][winner_index] = json!({
                "task":task,"dataset_hash":dataset,"evidence_class":"adaptive",
                "candidate":result["candidate"],"replayed_score":score,
                "prediction_sha256":result["prediction_sha256"],"prediction_count":result["prediction_count"],
                "runtime_seconds":result["runtime_seconds"],"train_sha256":result["train_sha256"],"test_sha256":result["test_sha256"],
            });
        }
        if exhausted {
            let winner = &replay["winners"][winner_index];
            let source_bytes = serde_json::to_vec(campaign)?;
            exhaustions.push(json!({
                "schema":"sotarena.aspfm.recovery-evolution.v1",
                "status":"exhausted",
                "task":task,
                "dataset_hash":dataset,
                "dataset_id":campaign["dataset_id"],
                "historical_target":campaign["historical_target"],
                "generations":campaign["generations"],
                "best_semantic_sha256":winner["candidate"]["semantic_sha256"],
                "best_canonical_genome":winner["candidate"]["canonical_genome"],
                "best_replayed_score":winner["replayed_score"],
                "source_sha256":sha256(&source_bytes),
            }));
        }
    }
    replay["results"]
        .as_array_mut()
        .context("results missing")?
        .sort_by(|left, right| {
            (
                left["candidate"]["task"].as_str(),
                left["candidate"]["dataset_hash"].as_str(),
                left["candidate"]["semantic_sha256"].as_str(),
            )
                .cmp(&(
                    right["candidate"]["task"].as_str(),
                    right["candidate"]["dataset_hash"].as_str(),
                    right["candidate"]["semantic_sha256"].as_str(),
                ))
        });
    replay["winners"]
        .as_array_mut()
        .context("winners missing")?
        .sort_by(|left, right| {
            (
                left["task"].as_str(),
                left["dataset_hash"].as_str(),
                left["evidence_class"].as_str(),
            )
                .cmp(&(
                    right["task"].as_str(),
                    right["dataset_hash"].as_str(),
                    right["evidence_class"].as_str(),
                ))
        });
    replay["recovery_exhaustions"] = Value::Array(exhaustions);
    Ok(())
}
