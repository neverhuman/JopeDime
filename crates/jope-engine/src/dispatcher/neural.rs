use super::*;

pub(super) fn handle(root: &Path, command: Cmd) -> Result<Option<Cmd>> {
    match command {
        Cmd::PolicyParity { directory } => jope_parallel::neural::parity(&directory)?,
        #[cfg(feature = "neural")]
        Cmd::ExportPolicy {
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
        Cmd::LossOptimizerParity { directory } => {
            jope_parallel::neural::loss_optimizer_parity(&directory)?
        }
        #[cfg(feature = "neural")]
        Cmd::Encode {
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
        Cmd::DistributedProbe {
            world,
            output,
            timeout,
            fail_rank,
        } => jope_parallel::controller::probe(world, &output, timeout, fail_rank)?,
        #[cfg(feature = "neural")]
        Cmd::TrainingProbe {
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
        Cmd::TrainingProbeRank {
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
        Cmd::NcclProbe {
            directory,
            rank,
            world,
        } => jope_parallel::neural::nccl_probe(rank, world, &directory)?,
        other => return Ok(Some(other)),
    }
    Ok(None)
}
