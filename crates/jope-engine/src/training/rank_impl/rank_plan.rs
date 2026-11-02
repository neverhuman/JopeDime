use super::*;

pub(super) fn validate_rank_plan(
    assignment: Option<&topology::Assignment>,
    admitted: Option<&crate::native_loop::bounded::launch::AdmittedBootstrap>,
    rank: usize,
    world: usize,
    micro: usize,
    steps: usize,
    checkpoint_steps: usize,
    launch: &Path,
    partition_version: partition::Version,
    global_batch: usize,
    batch_qualification: bool,
) -> Result<(CommunicatorSource, partition::Execution)> {
    // W7-T03: a frozen host/device assignment is validated before any input,
    // device or communicator work. Without one, the pre-W7 single-host
    // mapping applies and is recorded as such.
    if let Some(assignment) = assignment {
        topology::validate(assignment)?;
        ensure!(
            assignment.world == world,
            "requested world differs from the frozen assignment"
        );
    }
    // W7-T04: which rendezvous this rank may use is decided here, before any
    // input or device work.
    let communicator = communicator_source(assignment, admitted, rank, launch)?;
    ensure!(
        rank < world && steps > 0 && checkpoint_steps > 0,
        "invalid training topology"
    );
    // W7-T06: the v1 gate is unchanged; partition v2 runs only when the
    // request selects it, with any world from one to eight.
    let execution = partition::execution(
        partition_version,
        world,
        micro,
        global_batch,
        batch_qualification,
    )?;
    Ok((communicator, execution))
}
