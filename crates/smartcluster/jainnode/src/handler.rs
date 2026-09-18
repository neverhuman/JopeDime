use std::sync::Arc;

use async_trait::async_trait;

use crate::{
    ClockPort, ControlError, ControlFlow, ControlFrameHandler, ExecutionError, NodeControlFrame,
    NodeExecutionPort, NodeProtocolState, ProtocolError, SessionControlAction,
};

pub struct ProductionControlHandler<R, K>
where
    R: NodeExecutionPort,
    K: ClockPort,
{
    protocol: NodeProtocolState,
    runtime: Arc<R>,
    clock: Arc<K>,
}

impl<R, K> ProductionControlHandler<R, K>
where
    R: NodeExecutionPort,
    K: ClockPort,
{
    pub fn new(
        mut protocol: NodeProtocolState,
        runtime: Arc<R>,
        clock: Arc<K>,
    ) -> Result<Self, ControlError> {
        let active = runtime.active_count().map_err(ControlError::Execution)?;
        let recovering = runtime
            .committed_recovery_count(protocol.control_position())
            .map_err(ControlError::Execution)?;
        let total = active
            .checked_add(recovering)
            .ok_or(ControlError::Protocol(ProtocolError::AdmissionDenied))?;
        protocol
            .restore_inflight(total)
            .map_err(ControlError::Protocol)?;
        runtime
            .recover_committed(protocol.control_position())
            .map_err(ControlError::Execution)?;
        if runtime.active_count().map_err(ControlError::Execution)? != total {
            return Err(ControlError::Execution(ExecutionError::StaleAuthority));
        }
        Ok(Self {
            protocol,
            runtime,
            clock,
        })
    }

    fn account_completions(&mut self) -> Result<(), ControlError> {
        let completed = self
            .runtime
            .drain_completions()
            .map_err(ControlError::Execution)?;
        for _ in 0..completed {
            self.protocol
                .complete_infer()
                .map_err(ControlError::Protocol)?;
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) const fn protocol(&self) -> &NodeProtocolState {
        &self.protocol
    }
}

#[async_trait]
impl<R, K> ControlFrameHandler for ProductionControlHandler<R, K>
where
    R: NodeExecutionPort + 'static,
    K: ClockPort + 'static,
{
    async fn handle(&mut self, frame: NodeControlFrame) -> Result<ControlFlow, ControlError> {
        self.account_completions()?;
        match frame {
            frame @ NodeControlFrame::Infer { .. } => {
                let now_ms = self.clock.now_ms().map_err(ControlError::Execution)?;
                let admitted = self
                    .protocol
                    .verify_infer(frame, now_ms)
                    .map_err(ControlError::Protocol)?;
                let run_id = admitted.work.run_id.to_string();
                let attempt = admitted.work.attempt;
                let fence = admitted.work.attempt_fence;
                self.runtime
                    .prepare(admitted.clone(), now_ms)
                    .await
                    .map_err(ControlError::Execution)?;
                if let Err(error) = self.protocol.commit_verified_infer(&admitted) {
                    self.runtime
                        .discard_prepared(&run_id, attempt, fence)
                        .map_err(ControlError::Execution)?;
                    return Err(ControlError::Protocol(error));
                }
                self.runtime
                    .launch(&run_id, attempt, fence)
                    .map_err(ControlError::Execution)?;
                Ok(ControlFlow::Continue)
            }
            frame @ NodeControlFrame::Cancel { .. } => {
                let NodeControlFrame::Cancel {
                    run_id,
                    attempt,
                    attempt_fence,
                    ..
                } = &frame
                else {
                    unreachable!();
                };
                let work = self
                    .runtime
                    .active_work(run_id, *attempt, *attempt_fence)
                    .map_err(ControlError::Execution)?
                    .ok_or(ControlError::Protocol(ProtocolError::FenceRejected))?;
                let sequence = self
                    .protocol
                    .verify_cancel(&frame, &work)
                    .map_err(ControlError::Protocol)?;
                self.runtime
                    .cancel(run_id, *attempt, *attempt_fence)
                    .await
                    .map_err(ControlError::Execution)?;
                self.protocol
                    .commit_verified_sequence(sequence)
                    .map_err(ControlError::Protocol)?;
                Ok(ControlFlow::Continue)
            }
            frame @ NodeControlFrame::Rotate { .. } => {
                if !self
                    .runtime
                    .ready_for_rotation()
                    .map_err(ControlError::Execution)?
                {
                    return Err(ControlError::Protocol(ProtocolError::AdmissionDenied));
                }
                let now_ms = self.clock.now_ms().map_err(ControlError::Execution)?;
                let verified = self
                    .protocol
                    .verify_session_control(&frame, now_ms)
                    .map_err(ControlError::Protocol)?;
                if !matches!(verified.action, SessionControlAction::Rotate { .. }) {
                    return Err(ControlError::UnexpectedFrameDirection);
                }
                self.protocol
                    .apply_verified_session_control(&verified)
                    .map_err(ControlError::Protocol)?;
                Ok(ControlFlow::Stop)
            }
            frame @ NodeControlFrame::Revoke { .. } => {
                let now_ms = self.clock.now_ms().map_err(ControlError::Execution)?;
                let verified = self
                    .protocol
                    .verify_session_control(&frame, now_ms)
                    .map_err(ControlError::Protocol)?;
                if !matches!(verified.action, SessionControlAction::Revoke) {
                    return Err(ControlError::UnexpectedFrameDirection);
                }
                self.runtime
                    .revoke_all()
                    .await
                    .map_err(ControlError::Execution)?;
                self.protocol
                    .apply_verified_session_control(&verified)
                    .map_err(ControlError::Protocol)?;
                Ok(ControlFlow::Stop)
            }
            frame @ NodeControlFrame::StorageControl { .. } => {
                let now_ms = self.clock.now_ms().map_err(ControlError::Execution)?;
                self.protocol
                    .verify_storage_control(&frame, now_ms)
                    .map_err(ControlError::Protocol)?;
                Err(ControlError::StorageCompositionUnavailable)
            }
            NodeControlFrame::ConfirmOutbound { .. }
            | NodeControlFrame::Enroll { .. }
            | NodeControlFrame::SessionHello { .. }
            | NodeControlFrame::AssignmentAccepted { .. }
            | NodeControlFrame::Event { .. }
            | NodeControlFrame::Terminal { .. }
            | NodeControlFrame::StorageTelemetry { .. }
            | NodeControlFrame::StorageFragmentReceipt { .. } => {
                Err(ControlError::UnexpectedFrameDirection)
            }
        }
    }
}
