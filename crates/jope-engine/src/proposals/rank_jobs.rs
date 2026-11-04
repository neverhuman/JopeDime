use super::*;

pub(super) fn generate_rank_jobs(
    root: &Path,
    output: &Path,
    rank: usize,
    world: usize,
    binding: &crate::training::topology::Binding,
    shuffled: bool,
    vocab: &Vocab,
    model: &Policy,
    device: Device,
    programs: &BTreeMap<usize, Vec<String>>,
    proposal_seed: u64,
    jobs: &[(usize, &ContextRow)],
    donors: &[&ContextRow],
    donor_order: &[usize],
    started: std::time::Instant,
    checkpoint_sha: &str,
    champion_sha: &str,
    champion_snapshot: &crate::champions::Snapshot,
    generated_jobs: &mut Vec<GpJob>,
) -> Result<()> {
    for (j, (context_id, c)) in jobs.iter().enumerate().filter(|(j, _)| binding.owns(*j)) {
        let destination = output.join(&c.dataset).join(format!("cut-{}", c.cut));
        fs::create_dir_all(&destination)?;
        ensure!(
            !destination.join("ranking.json").exists()
                && !destination.join("best-predicted.json").exists(),
            "legacy filtered-pool proposal artifacts require a distinct output directory"
        );
        let actual: &ContextRow = if shuffled { donors[donor_order[j]] } else { c };
        let bytes = fs::read(resolve(root, &actual.path)?)?;
        ensure!(
            hash(&bytes) == actual.sha256 && bytes.len() == 8192,
            "conditioning context changed"
        );
        let values = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>();
        let p: Prepared = read_json(&resolve(root, &c.prepared)?.join("prepared.json"))?;
        let cut = p
            .cuts
            .iter()
            .find(|v| v.index == c.cut)
            .context("cut missing")?;
        ensure!(
            file_hash(&resolve(root, &c.prepared)?.join("prepared.json"))? == c.prepared_sha256,
            "proposal cut identity changed"
        );
        let seeds = destination.join("seeds.json");
        let generation = destination.join("generation.json");
        if seeds.exists() && generation.exists() {
            let g: Value = read_json(&generation)?;
            let s: SeedPopulation = read_json(&seeds)?;
            ensure!(g["schema"] == "parallel.proposal_generation.v2"
                && g["pure_prediction_protocol"] == PURE_PROTOCOL
                && g["gp_assembly_protocol"] == proposal_population::ASSEMBLY_PROTOCOL,
                "saved proposal uses an incompatible prediction protocol; use a new output directory");
            let pure: Value = read_json(&destination.join("pure-ranking.json"))?;
            let gp: Value = read_json(&destination.join("gp-ranking.json"))?;
            let best: Value = read_json(&destination.join("pure-best-predicted.json"))?;
            let assembly: Value = read_json(&destination.join("gp-population-assembly.json"))?;
            let pure_ranking: crate::proposal_ranking::Ranking =
                serde_json::from_value(pure["ranking"].clone())?;
            let gp_ranking: crate::proposal_ranking::Ranking =
                serde_json::from_value(gp["ranking"].clone())?;
            for ranking in [&pure_ranking, &gp_ranking] {
                ensure!(
                    ranking.selection_id == crate::proposal_ranking::SELECTION_ID
                        && ranking.bindings.conditioning_context_sha256 == actual.sha256
                        && ranking.bindings.checkpoint_sha256 == checkpoint_sha
                        && ranking.bindings.vocabulary_sha256
                            == champion_snapshot.vocabulary_sha256
                        && ranking.pool_size == 64
                        && ranking.chosen_count == 64
                        && ranking.ranked.len() == 64
                        && ranking.training_context_only
                        && !ranking.heldout_labels_read
                        && !ranking.gp_scores_read,
                    "saved fixed-pool ranking identity changed"
                );
            }
            let pure_sha = file_hash(&destination.join("pure-ranking.json"))?;
            let assembly_sha = file_hash(&destination.join("gp-population-assembly.json"))?;
            ensure!(
                s.schema == "parallel.seed_population.v1"
                    && s.candidates.len() == 80
                    && g["seeds_sha256"] == file_hash(&seeds)?
                    && g["champion_snapshot_sha256"] == champion_sha
                    && g["pure_ranking_sha256"] == pure_sha
                    && g["gp_ranking_sha256"] == file_hash(&destination.join("gp-ranking.json"))?
                    && g["pure_best_predicted_sha256"]
                        == file_hash(&destination.join("pure-best-predicted.json"))?
                    && g["gp_population_assembly_sha256"] == assembly_sha
                    && pure["schema"] == "parallel.pure_neural_ranking.v1"
                    && pure["protocol"] == PURE_PROTOCOL
                    && pure["teacher_population_consulted"] == false
                    && gp["schema"] == "parallel.gp_population_ranking.v1"
                    && gp["protocol"] == proposal_population::ASSEMBLY_PROTOCOL
                    && gp["teacher_population_consulted"] == true
                    && gp["training_context_only_end_to_end"] == false
                    && gp["pure_ranking_sha256"] == pure_sha
                    && gp["population_assembly_sha256"] == assembly_sha
                    && gp["champion_snapshot_sha256"] == champion_sha
                    && assembly["schema"] == "parallel.gp_population_assembly.v1"
                    && assembly["protocol"] == proposal_population::ASSEMBLY_PROTOCOL
                    && assembly["pure_ranking_sha256"] == pure_sha
                    && assembly["champion_snapshot_sha256"] == champion_sha
                    && assembly["checkpoint_sha256"] == checkpoint_sha
                    && assembly["conditioning_context_sha256"] == actual.sha256
                    && assembly["pure_programs_sha256"]
                        == hash(&serde_json::to_vec(
                            &pure_ranking
                                .selected()
                                .iter()
                                .map(|p| &p.program)
                                .collect::<Vec<_>>()
                        )?)
                    && assembly["retained_tail_sha256"]
                        == hash(&serde_json::to_vec(&s.candidates[64..])?)
                    && s.candidates
                        .iter()
                        .take(64)
                        .map(|c| c.genome.as_str())
                        .eq(gp_ranking.selected().iter().map(|p| p.program.as_str()))
                    && s.candidates
                        .iter()
                        .map(|c| &c.genome)
                        .collect::<BTreeSet<_>>()
                        .len()
                        == 80
                    && g["checkpoint_sha256"] == checkpoint_sha
                    && g["conditioning_context_sha256"] == actual.sha256
                    && s.prepared_sha256 == c.prepared_sha256
                    && s.cut == c.cut
                    && s.training_sha256 == cut.training_sha256
                    && s.context_sha256 == c.sha256
                    && s.checkpoint_sha256 == checkpoint_sha,
                "completed proposal identity changed"
            );
            ensure!(
                best["schema"] == "parallel.pure_best_predicted_program.v1"
                    && best["protocol"] == PURE_PROTOCOL
                    && best["pure_ranking_sha256"] == pure_sha
                    && best["dataset"] == c.dataset
                    && best["cut"] == c.cut
                    && best["prepared_sha256"] == c.prepared_sha256
                    && best["training_context_sha256"] == c.sha256
                    && best["conditioning_context_sha256"] == actual.sha256
                    && best["conditioning_dataset"] == actual.dataset
                    && best["checkpoint_sha256"] == checkpoint_sha
                    && best["vocabulary_sha256"] == champion_snapshot.vocabulary_sha256
                    && best["prediction"] == serde_json::to_value(&pure_ranking.selected()[0])?
                    && best["training_context_only_end_to_end"] == true
                    && best["teacher_population_consulted"] == false
                    && best["gp_scores_used_for_prediction"] == false
                    && best["heldout_targets_used_for_prediction"] == false
                    && best["final_tests_opened"] == false,
                "saved pure prediction identity changed"
            );
            generated_jobs.push(GpJob {
                prepared: c.prepared.clone(),
                prepared_sha256: c.prepared_sha256.clone(),
                cut: c.cut,
                output: destination.join("search").strip_prefix(root)?.to_path_buf(),
                seeds: Some(seeds.strip_prefix(root)?.to_path_buf()),
                seeds_sha256: Some(file_hash(&seeds)?),
                ..Default::default()
            });
            write_json(
                &output.join(format!("rank-{rank}.json")),
                &json!({"rank":rank,"world":world,"completed":generated_jobs.len(),"phase":"verified saved proposals","timestamp":now()}),
            )?;
            continue;
        }
        let pure_fallback_seed =
            crate::training::mix(proposal_seed ^ j as u64 ^ 0x707572655f66696c);
        let (candidates, attempted, valid, duplicate, neural) = pure_pool_impl(
            &model,
            &vocab,
            &values,
            device,
            crate::training::mix(proposal_seed ^ j as u64),
            pure_fallback_seed,
            "regression",
            Some((&destination.join("all-valid-neural.json"), shuffled)),
        )?;
        let bindings = crate::proposal_ranking::Bindings {
            checkpoint_sha256: checkpoint_sha.clone(),
            conditioning_context_sha256: actual.sha256.clone(),
            vocabulary_sha256: champion_snapshot.vocabulary_sha256.clone(),
        };
        // This pool has never consulted an incumbent or a retained program.
        let pure_ranking = crate::proposal_ranking::score_pool(
            &model,
            &vocab,
            &values,
            &candidates
                .iter()
                .map(|c| c.genome.clone())
                .collect::<Vec<_>>(),
            bindings.clone(),
            crate::proposal_ranking::Options {
                device,
                batch_size: 64,
                keep: 64,
            },
        )?;
        let mut by_genome = candidates
            .into_iter()
            .map(|c| (c.genome.clone(), c))
            .collect::<BTreeMap<_, _>>();
        let pure_candidates = pure_ranking
            .selected()
            .iter()
            .map(|p| {
                by_genome
                    .remove(&p.program)
                    .context("pure ranked program absent")
            })
            .collect::<Result<Vec<_>>>()?;
        ensure!(
            by_genome.is_empty() && pure_candidates.len() == 64,
            "pure pool identity changed"
        );
        write_json(
            &destination.join("pure-ranking.json"),
            &json!({
                "schema":"parallel.pure_neural_ranking.v1","protocol":PURE_PROTOCOL,
                "ranking":pure_ranking,"sampling_seed":crate::training::mix(proposal_seed ^ j as u64),
                "fallback_seed":pure_fallback_seed,"attempted":attempted,"valid":valid,
                "within_neural_duplicates":duplicate,"unique_neural":neural,
                "teacher_population_consulted":false,
            }),
        )?;
        let pure_ranking_sha = file_hash(&destination.join("pure-ranking.json"))?;
        write_json(
            &destination.join("pure-best-predicted.json"),
            &json!({
                "schema":"parallel.pure_best_predicted_program.v1","protocol":PURE_PROTOCOL,
                "dataset":c.dataset,"cut":c.cut,"prepared_sha256":c.prepared_sha256,
                "training_context_sha256":c.sha256,"conditioning_context_sha256":actual.sha256,
                "conditioning_dataset":actual.dataset,"checkpoint_sha256":checkpoint_sha,
                "vocabulary_sha256":bindings.vocabulary_sha256,
                "context_mode":if shuffled{"shuffled_between_source_families"}else{"matching_training_cut"},
                "selection_id":pure_ranking.selection_id,"pure_ranking_sha256":pure_ranking_sha,
                "prediction":pure_ranking.selected()[0],"quality_units":pure_ranking.quality_units,
                "predicted_runtime_is_estimate":true,"training_context_only_end_to_end":true,
                "input_scope":"saved training context, frozen checkpoint/vocabulary and independent RNG; excludes current teacher identities and scores",
                "teacher_population_consulted":false,"gp_scores_used_for_prediction":false,
                "heldout_targets_used_for_prediction":false,"final_tests_opened":false,
            }),
        )?;

        // Adaptive teachers enter only after the pure prediction is frozen.
        let mut seen = BTreeSet::new();
        let mut random_seed = proposal_seed.wrapping_add(j as u64 * 100003);
        let mut insert_random = |candidates: &mut Vec<SeedCandidate>,
                                 seen: &mut BTreeSet<String>,
                                 origin: &str|
         -> Result<()> {
            for _ in 0..10000 {
                random_seed = random_seed.wrapping_add(1);
                let raw = feat_math::evolve::random_canonical_pipeline(random_seed, 6);
                let Ok(genome) = vocab.project_proposal(&raw) else {
                    continue;
                };
                if seen.insert(genome.clone()) {
                    candidates.push(SeedCandidate {
                        genome,
                        origin: origin.into(),
                        policy_logprob: None,
                        proposal_record_sha256: None,
                    });
                    return Ok(());
                }
            }
            anyhow::bail!("unable to fill diverse proposal population")
        };
        // The two policy arms share these exact retained/exploration slots.
        let mut retained = Vec::new();
        let mut incumbent_evidence = Vec::new();
        for champion in &champion_snapshot.entries {
            if retained.len() == 10 {
                break;
            }
            if champion.identity.dataset != c.dataset || champion.identity.cut != c.cut {
                continue;
            }
            champion.validate()?;
            ensure!(
                champion.identity.prepared_sha256 == c.prepared_sha256
                    && champion.identity.training_sha256 == cut.training_sha256
                    && champion.identity.validation_sha256 == cut.validation_sha256
                    && champion.identity.vocabulary_sha256 == champion_snapshot.vocabulary_sha256,
                "incumbent belongs to a different cut or evaluator"
            );
            // Exact grammar encoding, without parameter projection or score reuse.
            vocab.encode(champion.genome())?;
            if seen.insert(champion.genome().into()) {
                retained.push(SeedCandidate {
                    genome: champion.genome().into(),
                    origin: "incumbent_cut_teacher".into(),
                    policy_logprob: None,
                    proposal_record_sha256: None,
                });
                incumbent_evidence.push(json!({"identity":champion.identity,"observation_sha256":champion.observation_sha256,
                    "source":champion.source,"evaluator_matches_current":champion.identity.evaluator_sha256 == champion_snapshot.evaluator_sha256,
                    "reevaluation_required":true,"score_transferred":false}));
            }
        }
        if let Some(leaders) = programs.get(context_id) {
            for genome in leaders {
                if retained.len() == 10 {
                    break;
                }
                if seen.insert(genome.clone()) {
                    retained.push(SeedCandidate {
                        genome: genome.clone(),
                        origin: "retained_cut_teacher".into(),
                        policy_logprob: None,
                        proposal_record_sha256: None,
                    });
                }
            }
        }
        for champion in &champion_snapshot.entries {
            if retained.len() >= 16 {
                break;
            }
            if champion.identity.family != c.family || champion.identity.cut == c.cut {
                continue;
            }
            if seen.insert(champion.genome().into()) {
                retained.push(SeedCandidate {
                    genome: champion.genome().into(),
                    origin: "same_family_cross_cut_teacher".into(),
                    policy_logprob: None,
                    proposal_record_sha256: None,
                });
            }
        }
        while retained.len() < 10 {
            insert_random(&mut retained, &mut seen, "retained_slot_fallback")?;
        }
        while retained.len() < 16 {
            insert_random(&mut retained, &mut seen, "random_exploration")?;
        }

        let assembly = proposal_population::assemble(
            &pure_candidates,
            &retained,
            &vocab,
            crate::training::mix(proposal_seed ^ j as u64 ^ 0x67705f6f7665726c),
        )?;
        let mut assembly_receipt = serde_json::to_value(&assembly)?;
        assembly_receipt["pure_ranking_sha256"] = pure_ranking_sha.clone().into();
        assembly_receipt["champion_snapshot_sha256"] = champion_sha.clone().into();
        assembly_receipt["checkpoint_sha256"] = checkpoint_sha.clone().into();
        assembly_receipt["conditioning_context_sha256"] = actual.sha256.clone().into();
        write_json(
            &destination.join("gp-population-assembly.json"),
            &assembly_receipt,
        )?;
        // A filtered/refilled GP pool has a distinct ranking and input claim.
        let gp_ranking = if assembly.overlaps.is_empty() {
            pure_ranking.clone()
        } else {
            crate::proposal_ranking::score_pool(
                &model,
                &vocab,
                &values,
                &assembly
                    .neural_slots
                    .iter()
                    .map(|c| c.genome.clone())
                    .collect::<Vec<_>>(),
                bindings,
                crate::proposal_ranking::Options {
                    device,
                    batch_size: 64,
                    keep: 64,
                },
            )?
        };
        write_json(
            &destination.join("gp-ranking.json"),
            &json!({
                "schema":"parallel.gp_population_ranking.v1","protocol":proposal_population::ASSEMBLY_PROTOCOL,
                "ranking":gp_ranking,"pure_ranking_sha256":pure_ranking_sha,
                "population_assembly_sha256":file_hash(&destination.join("gp-population-assembly.json"))?,
                "champion_snapshot_sha256":champion_sha,"teacher_population_consulted":true,
                "training_context_only_end_to_end":false,
            }),
        )?;
        let mut by_genome = assembly
            .neural_slots
            .into_iter()
            .map(|c| (c.genome.clone(), c))
            .collect::<BTreeMap<_, _>>();
        let mut candidates = gp_ranking
            .selected()
            .iter()
            .map(|p| {
                by_genome
                    .remove(&p.program)
                    .context("GP ranked program absent")
            })
            .collect::<Result<Vec<_>>>()?;
        ensure!(
            by_genome.is_empty() && candidates.len() == 64,
            "GP ranked pool identity changed"
        );
        let scores: Vec<f64> = gp_ranking
            .selected()
            .iter()
            .map(|p| p.predicted_quality)
            .collect();
        crate::gp::attach_policy_logprobs(&mut candidates, &scores)?;
        let gp_neural = candidates
            .iter()
            .filter(|c| c.origin == "neural_policy")
            .count();
        candidates.extend(retained);
        let population = SeedPopulation {
            schema: "parallel.seed_population.v1".into(),
            prepared_sha256: c.prepared_sha256.clone(),
            cut: c.cut,
            training_sha256: cut.training_sha256.clone(),
            context_sha256: c.sha256.clone(),
            checkpoint_sha256: checkpoint_sha.clone(),
            candidates,
        };
        let seeds = destination.join("seeds.json");
        write_json(&seeds, &population)?;
        write_json(
            &destination.join("generation.json"),
            &json!({"schema":"parallel.proposal_generation.v2","pure_prediction_protocol":PURE_PROTOCOL,
                "gp_assembly_protocol":proposal_population::ASSEMBLY_PROTOCOL,"attempted":attempted,"valid":valid,
                "within_neural_duplicates":duplicate,"pure_unique_neural":neural,"unique_neural":gp_neural,
                "filled_population":80,"context_mode":if shuffled{"shuffled_between_source_families"}else{"matching_training_cut"},
                "conditioning_context_sha256":actual.sha256,"checkpoint_sha256":checkpoint_sha,
                "seeds_sha256":file_hash(&seeds)?,"champion_snapshot_sha256":champion_sha,"incumbents":incumbent_evidence,
                "pure_ranking_sha256":pure_ranking_sha,"pure_best_predicted_sha256":file_hash(&destination.join("pure-best-predicted.json"))?,
                "gp_ranking_sha256":file_hash(&destination.join("gp-ranking.json"))?,
                "gp_population_assembly_sha256":file_hash(&destination.join("gp-population-assembly.json"))?,
                "teacher_overlaps":assembly.overlaps,"overlap_refills":assembly.refills,
                "selection_id":gp_ranking.selection_id,"seed":crate::training::mix(proposal_seed^j as u64),"final_tests_opened":false}),
        )?;
        generated_jobs.push(GpJob {
            prepared: c.prepared.clone(),
            prepared_sha256: c.prepared_sha256.clone(),
            cut: c.cut,
            output: destination.join("search").strip_prefix(root)?.to_path_buf(),
            seeds: Some(seeds.strip_prefix(root)?.to_path_buf()),
            seeds_sha256: Some(file_hash(&seeds)?),
            ..Default::default()
        });
        write_json(
            &output.join(format!("rank-{rank}.json")),
            &json!({"rank":rank,"world":world,"completed":generated_jobs.len(),"phase":"grammar-masked neural proposals","timestamp":now(),"seconds":started.elapsed().as_secs_f64(),"unique_neural_last_cut":gp_neural}),
        )?;
    }
    Ok(())
}
