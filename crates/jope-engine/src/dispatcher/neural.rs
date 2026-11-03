use super::*;

pub(crate) fn handle(root: &Path, command: EngineCmd) -> Result<Option<EngineCmd>> {
    match command {
        #[cfg(feature = "neural")]
        EngineCmd::PolicyParity { directory } => jope_parallel::neural::parity(&directory)?,
        #[cfg(feature = "neural")]
        EngineCmd::ExportPolicy {
            checkpoint,
            output,
            receipt,
            expect_state_sha256,
            parent,
        } => println!(
            "{}",
            jope_parallel::policy_export::export(&jope_parallel::policy_export::ExportRequest {
                checkpoint,
                output,
                receipt,
                expect_state_sha256,
                parent,
            })?
        ),
        #[cfg(feature = "neural")]
        EngineCmd::LossOptimizerParity { directory } => {
            jope_parallel::neural::loss_optimizer_parity(&directory)?
        }
        #[cfg(feature = "neural")]
        EngineCmd::Encode {
            prepared,
            cut,
            spec,
            output,
            gpu,
        } => {
            let e = jope_parallel::encoding::encode(&root, &prepared, cut, &spec, &output, gpu)?;
            println!("{}", serde_json::to_string(&e)?);
        }
        #[cfg(feature = "neural")]
        EngineCmd::DistributedProbe {
            world,
            output,
            timeout,
            fail_rank,
        } => jope_parallel::controller::probe(world, &output, timeout, fail_rank)?,
        #[cfg(feature = "neural")]
        EngineCmd::TrainingProbe {
            output,
            vocabulary,
            microbatch,
            global_batch,
            steps,
            timeout,
        } => jope_parallel::controller::training_probe(
            &output,
            &vocabulary,
            microbatch,
            global_batch,
            steps,
            timeout,
        )?,
        #[cfg(feature = "neural")]
        EngineCmd::TrainingProbeRank {
            directory,
            vocabulary,
            rank,
            world,
            microbatch,
            global_batch,
            steps,
        } => jope_parallel::neural::training_probe(
            rank,
            world,
            &directory,
            &vocabulary,
            microbatch,
            global_batch,
            steps,
        )?,
        #[cfg(feature = "neural")]
        EngineCmd::NcclProbe {
            directory,
            rank,
            world,
        } => jope_parallel::neural::nccl_probe(rank, world, &directory)?,
        other => return Ok(Some(other)),
    }
    Ok(None)
}
