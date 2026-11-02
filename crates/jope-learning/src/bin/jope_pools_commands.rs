use super::*;

pub(super) fn run() -> Result<()> {
    match Cli::parse().command {
        Command::IngestCoverage {
            coverage,
            partition,
            exclude_contexts,
            exclude_pairs,
            protocol_sha256,
            out,
        } => {
            let mut excluded = BTreeSet::new();
            for p in &exclude_contexts {
                excluded.extend(id_lines(&read(p)?)?);
            }
            let mut pairs = BTreeSet::new();
            for p in &exclude_pairs {
                for line in id_lines(&read(p)?)? {
                    let (cid, program) = line
                        .split_once('\t')
                        .context("--exclude-pairs lines are context_id<TAB>program_sha256")?;
                    pairs.insert((cid.to_owned(), program.to_owned()));
                }
            }
            let doc: serde_json::Value = serde_json::from_slice(&read(&coverage)?)?;
            let (rows, report) =
                source::coverage_rows(&doc, &partition, &excluded, &pairs, &protocol_sha256)?;
            create_new(&out, &jsonl(&rows)?)?;
            println!("{}", serde_json::to_string(&report)?);
        }
        Command::IngestRosterFeatures {
            roster,
            pack,
            manifest,
            out,
        } => {
            let manifest: Manifest = serde_json::from_slice(&read(&manifest)?)?;
            let wanted: BTreeSet<String> = manifest
                .pools
                .iter()
                .map(|e| e.context_id.clone())
                .collect();
            let lines = source::roster_vectors(
                std::str::from_utf8(&read(&roster)?)?,
                &read(&pack)?,
                &wanted,
            )?;
            create_new(&out, &jsonl(&lines)?)?;
            println!("exported {} context vectors", lines.len());
        }
        Command::Views {
            pools,
            manifest,
            out,
        } => {
            let pools = read_pools(&read(&pools)?)?;
            let manifest: Manifest = serde_json::from_slice(&read(&manifest)?)?;
            let views = export_views(&pools, &manifest)?;
            create_new(&out, &jsonl(&views)?)?;
            println!("exported {} outcome-free views", views.len());
        }
        Command::Freeze {
            rows,
            out,
            min_pool,
            max_pool,
            salt,
            min_distinct_measured,
            source_grade,
        } => {
            let bytes = read(&rows)?;
            let parsed: Vec<MeasurementRow> = std::str::from_utf8(&bytes)?
                .lines()
                .enumerate()
                .filter(|(_, l)| !l.trim().is_empty())
                .map(|(i, l)| {
                    serde_json::from_str(l).with_context(|| format!("row line {}", i + 1))
                })
                .collect::<Result<_>>()?;
            let rule = FreezeRule {
                min_pool,
                max_pool,
                salt,
                min_distinct_measured,
                source_grade,
            };
            let (pools, manifest) = freeze(&parsed, &rule, &sha256_hex(&bytes))?;
            fs::create_dir(&out)
                .with_context(|| format!("create {} (must not exist)", out.display()))?;
            create_new(&out.join("pools.jsonl"), &jsonl(&pools)?)?;
            create_new(&out.join("manifest.json"), &pretty(&manifest)?)?;
            println!(
                "froze {} pools ({} excluded) from {} rows; manifest {}",
                manifest.pools.len(),
                manifest.excluded.len(),
                manifest.input_rows,
                manifest.sha256()?
            );
        }
        Command::Evaluate {
            pools,
            manifest,
            reference,
            context_features,
            program_features,
            s3_prior_parquet,
            external,
            unavailable,
            budget,
            folds,
            fold_salt,
            prior_weight,
            knn_k,
            ridge_lambda,
            bootstrap_draws,
            bootstrap_seed,
            alpha,
            source_commit,
            out,
        } => {
            anyhow::ensure!(
                source_commit.len() == 40
                    && source_commit
                        .bytes()
                        .all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')),
                "--source-commit must be a full lowercase commit"
            );
            let mut digests = Vec::new();
            let mut load = |p: &Path| -> Result<Vec<u8>> {
                let b = read(p)?;
                digests.push(InputDigest {
                    path: p.display().to_string(),
                    sha256: sha256_hex(&b),
                });
                Ok(b)
            };
            let pools = read_pools(&load(&pools)?)?;
            let manifest: Manifest = serde_json::from_slice(&load(&manifest)?)?;
            let cf = context_features
                .map(|p| Features::from_jsonl(std::str::from_utf8(&load(&p)?)?))
                .transpose()?;
            let pf = program_features
                .map(|p| Features::from_jsonl(std::str::from_utf8(&load(&p)?)?))
                .transpose()?;
            let s3_rows = s3_prior_parquet
                .map(|p| {
                    jope_learning::retention::distilled::read(
                        &load(&p)?,
                        &jope_learning::pools::flywheel::known_scores(),
                    )
                })
                .transpose()?;
            let docs: Vec<ExternalDocument> = external
                .iter()
                .map(|p| Ok(serde_json::from_slice(&load(p)?)?))
                .collect::<Result<_>>()?;
            let externals: Vec<External> = docs
                .iter()
                .map(External::from_document)
                .collect::<Result<_>>()?;
            let unavailable = unavailable
                .iter()
                .map(|s| {
                    let (name, reason) = s
                        .split_once('=')
                        .context("--unavailable takes name=reason")?;
                    Ok(Unavailable {
                        name: name.to_owned(),
                        reason: reason.to_owned(),
                    })
                })
                .collect::<Result<_>>()?;
            let inputs = Inputs {
                context_features: cf.as_ref(),
                program_features: pf.as_ref(),
                external: externals.iter().map(|e| e as &dyn Selector).collect(),
                unavailable,
            };
            let cfg = Config {
                reference,
                budget,
                folds,
                fold_salt,
                prior_weight,
                knn_k,
                ridge_lambda,
                bootstrap_draws,
                bootstrap_seed,
                alpha,
            };
            let fold_priors = match s3_rows.as_deref() {
                Some(rows) => {
                    let (p, _) =
                        jope_learning::pools::flywheel::build_fold_s3_priors(&pools, &cfg, rows)?;
                    Some(p)
                }
                None => None,
            };
            let receipt =
                jope_learning::pools::evaluate::evaluate_with_fold_train_pools_and_priors(
                    &pools,
                    None,
                    fold_priors.as_deref(),
                    &manifest,
                    &inputs,
                    &cfg,
                )?;
            let output = EvaluationOutput {
                receipt_sha256: receipt.sha256()?,
                source_commit: source_commit.clone(),
                binary_sha256: sha256_hex(&read(&std::env::current_exe()?)?),
                inputs: digests,
                external_provenance: docs.iter().map(|d| d.provenance.clone()).collect(),
                receipt: &receipt,
            };
            create_new(&out, &pretty(&output)?)?;
            println!(
                "{} contexts; strongest baseline {}; ranking_learnability {}; receipt {}",
                receipt.contexts.len(),
                receipt.strongest_baseline,
                receipt.ranking_learnability,
                output.receipt_sha256
            );
        }
        Command::FlywheelDistill {
            pools,
            manifest,
            reference,
            context_features,
            program_features,
            s3_prior_parquet,
            external,
            budget,
            eval_budgets,
            rounds,
            initial_budget,
            step_budget,
            proposal_pool_k,
            score_name,
            folds,
            fold_salt,
            prior_weight,
            knn_k,
            ridge_lambda,
            bootstrap_draws,
            bootstrap_seed,
            alpha,
            parquet_dir,
            source_commit,
            out,
        } => {
            anyhow::ensure!(
                source_commit.len() == 40
                    && source_commit
                        .bytes()
                        .all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')),
                "--source-commit must be a full lowercase commit"
            );
            let mut digests = Vec::new();
            let mut load = |p: &Path| -> Result<Vec<u8>> {
                let b = read(p)?;
                digests.push(InputDigest {
                    path: p.display().to_string(),
                    sha256: sha256_hex(&b),
                });
                Ok(b)
            };
            let pools = read_pools(&load(&pools)?)?;
            let manifest: Manifest = serde_json::from_slice(&load(&manifest)?)?;
            let cf = Features::from_jsonl(std::str::from_utf8(&load(&context_features)?)?)?;
            let pf = program_features
                .map(|p| Features::from_jsonl(std::str::from_utf8(&load(&p)?)?))
                .transpose()?;
            let s3_rows = s3_prior_parquet
                .map(|p| {
                    jope_learning::retention::distilled::read(
                        &load(&p)?,
                        &jope_learning::pools::flywheel::known_scores(),
                    )
                })
                .transpose()?;
            let docs: Vec<ExternalDocument> = external
                .iter()
                .map(|p| Ok(serde_json::from_slice(&load(p)?)?))
                .collect::<Result<_>>()?;
            let externals: Vec<External> = docs
                .iter()
                .map(External::from_document)
                .collect::<Result<_>>()?;
            let ext_refs: Vec<&dyn Selector> =
                externals.iter().map(|e| e as &dyn Selector).collect();
            let cfg = jope_learning::pools::flywheel::FlywheelConfig {
                eval: Config {
                    reference,
                    budget,
                    folds,
                    fold_salt,
                    prior_weight,
                    knn_k,
                    ridge_lambda,
                    bootstrap_draws,
                    bootstrap_seed,
                    alpha,
                },
                eval_budgets,
                rounds,
                initial_budget_per_context: initial_budget,
                step_budget_per_context: step_budget,
                proposal_pool_k,
                score_name,
                tilt: jope_learning::clover::expansion::MeasuredTiltPolicyConfig {
                    temperature_tau: 0.25,
                    advantage_clip: 2.0,
                    ref_new_support_mass: 0.05,
                    policy_new_support_cap: 0.12,
                    step_size_eta: 0.20,
                },
                max_conditional_base_kl: 0.20,
                max_new_support_mass: 0.15,
            };
            let receipt = jope_learning::pools::flywheel::run_distilled_flywheel_with_s3_prior(
                &pools,
                &manifest,
                &cf,
                pf.as_ref(),
                &ext_refs,
                s3_rows.as_deref(),
                &cfg,
                parquet_dir.as_deref(),
            )?;
            #[derive(Serialize)]
            struct FlywheelOutput<'a> {
                receipt_sha256: String,
                source_commit: String,
                binary_sha256: String,
                inputs: Vec<InputDigest>,
                external_provenance: Vec<serde_json::Value>,
                receipt: &'a jope_learning::pools::flywheel::FlywheelReceipt,
            }
            let output = FlywheelOutput {
                receipt_sha256: receipt.sha256()?,
                source_commit,
                binary_sha256: sha256_hex(&read(&std::env::current_exe()?)?),
                inputs: digests,
                external_provenance: docs.iter().map(|d| d.provenance.clone()).collect(),
                receipt: &receipt,
            };
            create_new(&out, &pretty(&output)?)?;
            println!(
                "{} rounds; active_discovery_established {}; data_accumulation_lift_established {}; flywheel_established {}; ranking_learnability {}; receipt {}",
                receipt.rounds.len(),
                receipt.active_discovery_established,
                receipt.data_accumulation_lift_established,
                receipt.flywheel_established,
                receipt.final_round_receipt.ranking_learnability,
                output.receipt_sha256
            );
        }
        Command::QualifyRounds {
            plan,
            identity,
            ratchets,
            test_scores,
            parity,
            parent_map,
            prereg,
            out,
        } => {
            use jope_learning::clover::round_qual::{
                parse_parent_map, parse_strict_json_value, qualify_rounds_with_prereg,
                ParityRecord, QualTestScore, RoundQualInputs, RoundQualPlan,
            };
            use jope_learning::clover::ExperimentIdentity;

            let plan_val = parse_strict_json_value(&read(&plan)?, "--plan")?;
            let plan: RoundQualPlan = serde_json::from_value(plan_val).context("parse --plan")?;
            let identity_val = parse_strict_json_value(&read(&identity)?, "--identity")?;
            let identity: ExperimentIdentity =
                serde_json::from_value(identity_val).context("parse --identity")?;
            let ratchets: Vec<serde_json::Value> =
                parse_strict_json_or_jsonl(&read(&ratchets)?, "--ratchets")?;
            let test_scores: Vec<QualTestScore> =
                parse_json_or_jsonl(&read(&test_scores)?, "--test-scores")?;
            let parity: Vec<ParityRecord> = parse_json_or_jsonl(&read(&parity)?, "--parity")?;
            let parent_map = parse_parent_map(&read(&parent_map)?)?;
            let prereg_bytes = match prereg.as_deref() {
                Some(p) => Some(read_bounded(p, 1024 * 1024)?),
                None => None,
            };

            let report = qualify_rounds_with_prereg(
                RoundQualInputs {
                    plan: &plan,
                    identity: &identity,
                    ratchet_outputs: &ratchets,
                    test_scores: &test_scores,
                    parity: &parity,
                    parent_of_context: &parent_map,
                },
                prereg_bytes.as_deref(),
            )?;
            let digest = report.canonical_sha256()?;
            create_new(&out, &pretty(&report)?)?;
            println!(
                "{} contexts ({} complete paired across {} parent clusters); verdict {:?}; margin_cleared {}; provenance_binding_status {}; qualified_claim {}; report_sha256 {}",
                report.n_ratchet_contexts,
                report.n_complete_paired_contexts,
                report.n_observed_parent_clusters,
                report.verdict,
                report.margin_cleared,
                report.provenance_binding_status,
                report.qualified_claim,
                digest
            );
        }
    }
    Ok(())
}
