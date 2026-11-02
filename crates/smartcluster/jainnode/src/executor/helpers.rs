use super::*;

pub(crate) async fn record_and_emit_chunk<E: EventSink>(
    events: &E,
    emitter: &mut Emitter,
    chunk: DockerLogChunk,
    stdout: &mut Vec<u8>,
    stderr: &mut Vec<u8>,
    stdout_budget: u64,
    stderr_budget: u64,
) -> Result<(), ExecutionError> {
    if chunk.bytes.len() > MAX_STREAM_CHUNK_BYTES {
        return Err(ExecutionError::StreamProtocolViolation);
    }
    let (recorded, budget) = if chunk.stderr {
        (stderr, stderr_budget)
    } else {
        (stdout, stdout_budget)
    };
    let new_length = recorded
        .len()
        .checked_add(chunk.bytes.len())
        .ok_or(ExecutionError::LogBudgetExceeded)?;
    if u64::try_from(new_length).map_err(|_| ExecutionError::LogBudgetExceeded)? > budget {
        return Err(ExecutionError::LogBudgetExceeded);
    }
    let identity = emitter.next(chunk.bytes.len() as u64);
    recorded.extend_from_slice(&chunk.bytes);
    events
        .emit(if chunk.stderr {
            NodeEvent::StderrChunk {
                identity,
                bytes: chunk.bytes,
            }
        } else {
            NodeEvent::StdoutChunk {
                identity,
                bytes: chunk.bytes,
            }
        })
        .await
}

pub(crate) fn ensure_success(status: ExitStatus) -> Result<(), ExecutionError> {
    if status.oom_killed {
        Err(ExecutionError::MemoryLimitExceeded)
    } else if status.code != 0 {
        Err(ExecutionError::NonZeroExit(status.code))
    } else {
        Ok(())
    }
}

pub(crate) fn platform_environment(
    work: &VerifiedWorkUnit,
    placement: &Placement,
    phase: ExecutionPhase,
) -> BTreeMap<String, String> {
    BTreeMap::from([
        ("JAIN_RUN_ID".to_owned(), work.run_id.to_string()),
        ("JAIN_ATTEMPT".to_owned(), work.attempt.to_string()),
        (
            "JAIN_ATTEMPT_FENCE".to_owned(),
            work.attempt_fence.to_string(),
        ),
        (
            "JAIN_CONTROL_POSITION".to_owned(),
            work.control_position.to_string(),
        ),
        ("JAIN_NODE_ID".to_owned(), placement.node_id.clone()),
        ("JAIN_PHASE".to_owned(), phase_name(phase).to_owned()),
        ("JAIN_OUTPUT_DIR".to_owned(), "/outputs".to_owned()),
    ])
}

pub(crate) fn phase_name(phase: ExecutionPhase) -> &'static str {
    match phase {
        ExecutionPhase::Smoke => "smoke",
        ExecutionPhase::Run => "run",
    }
}

pub(crate) struct Emitter {
    run_id: String,
    attempt: u32,
    fence: u64,
    cursor: u64,
    stream_offset: u64,
    control_position: u64,
    phase: ExecutionPhase,
}

impl Emitter {
    pub(crate) fn new(work: &VerifiedWorkUnit, phase: ExecutionPhase) -> Self {
        Self {
            run_id: work.run_id.to_string(),
            attempt: work.attempt,
            fence: work.attempt_fence,
            cursor: 0,
            stream_offset: 0,
            control_position: work.control_position,
            phase,
        }
    }

    pub(crate) fn next(&mut self, stream_bytes: u64) -> EventIdentity {
        self.cursor = self.cursor.saturating_add(1);
        let identity = EventIdentity {
            run_id: self.run_id.clone(),
            attempt: self.attempt,
            fence: self.fence,
            cursor: self.cursor,
            stream_offset: self.stream_offset,
            control_position: self.control_position,
            phase: self.phase,
        };
        self.stream_offset = self.stream_offset.saturating_add(stream_bytes);
        identity
    }
}
