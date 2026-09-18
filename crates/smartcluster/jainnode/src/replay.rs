use crate::{
    control::{controller_to_node, validate_outbound_session_hello, wait_before_reconnect},
    ControlConnection, ControlConnector, ControlError, ControlFlow, ControlFrameHandler,
    DurableControlOutbox, MtlsClientConfig, NodeControlFrame, OutboundConfirmation,
    ReconnectPolicy,
};

const MAX_CONFIRMATION_CLOCK_SKEW_MS: u64 = 300_000;

#[allow(clippy::too_many_arguments)]
pub async fn supervise_durable_outbound_control<F, N>(
    connector: &dyn ControlConnector,
    config: &MtlsClientConfig,
    policy: ReconnectPolicy,
    handler: &mut dyn ControlFrameHandler,
    outbox: &DurableControlOutbox,
    maximum_confirmation_clock_skew_ms: u64,
    mut now_ms: N,
    mut hello_factory: F,
) -> Result<(), ControlError>
where
    F: FnMut() -> Result<NodeControlFrame, ControlError> + Send,
    N: FnMut() -> Result<u64, ControlError> + Send,
{
    config.validate_shape()?;
    policy.validate()?;
    if maximum_confirmation_clock_skew_ms == 0
        || maximum_confirmation_clock_skew_ms > MAX_CONFIRMATION_CLOCK_SKEW_MS
    {
        return Err(ControlError::InvalidConfiguration);
    }
    let mut failures = 0;
    let mut delay = policy.initial_delay;
    'connections: loop {
        let mut connection = match connector.connect(config).await {
            Ok(connection) => connection,
            Err(error) => {
                wait_before_reconnect(error, &mut failures, &mut delay, policy).await?;
                continue;
            }
        };
        let hello = hello_factory()?;
        validate_outbound_session_hello(&hello)?;
        outbox.verify_session_hello_identity(&hello)?;
        if let Err(error) = connection.send_frame(&hello).await {
            wait_before_reconnect(error, &mut failures, &mut delay, policy).await?;
            continue;
        }
        let mut sent_pending = match send_oldest_pending(connection.as_mut(), outbox).await {
            Ok(sequence) => sequence,
            Err(error) => {
                wait_before_reconnect(error, &mut failures, &mut delay, policy).await?;
                continue;
            }
        };
        loop {
            let frame = match connection.receive_frame().await {
                Ok(frame) => frame,
                Err(error) => {
                    wait_before_reconnect(error, &mut failures, &mut delay, policy).await?;
                    continue 'connections;
                }
            };
            if !controller_to_node(&frame) {
                return Err(ControlError::UnexpectedFrameDirection);
            }
            if let NodeControlFrame::ConfirmOutbound {
                outbound_sequence, ..
            } = &frame
            {
                let durable_oldest = outbox.oldest()?.map(|pending| pending.sequence);
                if durable_oldest == Some(*outbound_sequence)
                    && sent_pending != Some(*outbound_sequence)
                {
                    return Err(ControlError::UnexpectedFrameDirection);
                }
                match outbox.confirm_controller(
                    &frame,
                    now_ms()?,
                    maximum_confirmation_clock_skew_ms,
                )? {
                    OutboundConfirmation::Retired => {
                        sent_pending = match send_oldest_pending(connection.as_mut(), outbox).await
                        {
                            Ok(sequence) => sequence,
                            Err(error) => {
                                wait_before_reconnect(error, &mut failures, &mut delay, policy)
                                    .await?;
                                continue 'connections;
                            }
                        };
                    }
                    OutboundConfirmation::AlreadyConfirmed => {}
                }
                continue;
            }
            match handler.handle(frame).await? {
                ControlFlow::Continue => {}
                ControlFlow::Stop => return Ok(()),
            }
        }
    }
}

async fn send_oldest_pending(
    connection: &mut dyn ControlConnection,
    outbox: &DurableControlOutbox,
) -> Result<Option<u64>, ControlError> {
    let Some(pending) = outbox.oldest()? else {
        return Ok(None);
    };
    connection.send_frame(&pending.frame).await?;
    Ok(Some(pending.sequence))
}
