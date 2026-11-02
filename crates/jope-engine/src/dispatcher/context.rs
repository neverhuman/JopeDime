use super::*;

pub(super) fn handle(root: &Path, command: Cmd) -> Result<Option<Cmd>> {
    match command {
        #[cfg(feature = "neural")]
        Cmd::ApplyBinaryContexts {
            jobs,
            hyperion,
            foundation,
            target,
            assembly,
            output,
            aliases,
        } => jope_parallel::contexts::apply_binary_qualification(
            &root,
            &jobs,
            &hyperion,
            &foundation,
            &target,
            &assembly,
            &output,
            &aliases,
        )?,
        Cmd::Enqueue {
            jobs,
            contexts,
            campaign_contexts,
        } => {
            println!(
                "{} new GP jobs admitted with verified saved contexts",
                jope_parallel::queue::enqueue_bound(
                    &root,
                    &jobs,
                    &contexts,
                    campaign_contexts.as_deref()
                )?
            );
        }
        Cmd::CampaignCoverage {
            campaign,
            union,
            output,
            gp_root,
        } => {
            let roots: Vec<_> = gp_root.iter().map(|p| root.join(p)).collect();
            let refs: Vec<_> = roots.iter().map(|p| p.as_path()).collect();
            println!(
                "{}",
                serde_json::to_string_pretty(&jope_parallel::queue::write_coverage(
                    &root, &campaign, &union, &refs, &output
                )?)?
            );
        }
        #[cfg(feature = "neural")]
        Cmd::ApplyContexts {
            jobs,
            hyperion,
            foundation,
            target,
            assembly,
            output,
            qualification,
        } => jope_parallel::contexts::apply(
            &root,
            &jobs,
            &hyperion,
            &foundation,
            &target,
            &assembly,
            &output,
            qualification,
        )?,
        #[cfg(feature = "neural")]
        Cmd::UnionContexts { indexes, output } => {
            jope_parallel::contexts::union(&root, &indexes, &output)?
        }
        #[cfg(feature = "neural")]
        Cmd::AssembleContexts {
            jobs,
            hyperion,
            foundation,
            target,
            output,
            registry,
            qualification,
        } => {
            if let Some(registry) = registry {
                jope_parallel::contexts::assemble_registered(
                    &root,
                    &jobs,
                    &hyperion,
                    &foundation,
                    &target,
                    &registry,
                    &output,
                    qualification,
                )?;
            } else {
                jope_parallel::contexts::assemble(
                    &root,
                    &jobs,
                    &hyperion,
                    &foundation,
                    &target,
                    &output,
                    qualification,
                )?;
            }
        }

        #[cfg(feature = "neural")]
        Cmd::FoundationService {
            jobs,
            spec,
            output,
            batch,
        } => jope_parallel::components::service(&root, &jobs, &spec, &output, batch, "foundation")?,
        #[cfg(feature = "neural")]
        Cmd::FoundationRank {
            jobs,
            spec,
            output,
            rank,
            world,
            batch,
        } => jope_parallel::components::foundation_rank(
            &root, &jobs, &spec, &output, rank, world, batch,
        )?,

        #[cfg(feature = "neural")]
        Cmd::TargetService {
            jobs,
            spec,
            output,
            batch,
        } => jope_parallel::components::service(&root, &jobs, &spec, &output, batch, "target")?,
        #[cfg(feature = "neural")]
        Cmd::TargetRank {
            jobs,
            spec,
            output,
            rank,
            world,
            batch,
        } => jope_parallel::components::target_rank(
            &root, &jobs, &spec, &output, rank, world, batch,
        )?,

        Cmd::HyperionService { jobs, spec, output } => {
            jope_parallel::hyperion::service(&root, &jobs, &spec, &output)?
        }
        Cmd::HyperionRank {
            jobs,
            spec,
            output,
            rank,
            world,
        } => jope_parallel::hyperion::rank(&root, &jobs, &spec, &output, rank, world)?,
        Cmd::ExpandCuts {
            prepared,
            output,
            rounds,
        } => {
            prepare::expand_cuts(&prepared, &output, rounds)?;
        }
        Cmd::IndexCuts {
            prepared,
            backing_manifest,
            metadata,
            output,
        } => prepare::index_existing(&prepared, &backing_manifest, &metadata, &output)?,
        Cmd::EvaluateView {
            prepared,
            cut,
            genome,
            output,
        } => jope_parallel::gp::evaluate_view(&prepared, cut, &genome, &output)?,
        Cmd::EvaluateMemory {
            fd,
            genome,
            output,
            prepared,
            cut,
        } => {
            // W5-T05: a guarded evaluator's process group dies with its search controller.
            jope_parallel::gp::guard_evaluator_group()?;
            jope_parallel::gp::evaluate_prepared_memory(
                fd,
                &genome,
                &output,
                prepared.as_deref().zip(cut),
            )?
        }
        other => return Ok(Some(other)),
    }
    Ok(None)
}
