use super::*;

#[derive(Clone)]
pub(super) struct EventJournal {
    log: Arc<Mutex<EventLog<jain_smartcluster_core::EventRecord>>>,
    fanout: EventFanout<jain_smartcluster_core::EventRecord>,
}

impl EventJournal {
    pub(super) fn new(
        log: EventLog<jain_smartcluster_core::EventRecord>,
        subscriber_capacity: usize,
    ) -> Self {
        Self {
            log: Arc::new(Mutex::new(log)),
            fanout: EventFanout::new(subscriber_capacity),
        }
    }

    pub(super) fn append(
        &self,
        job_id: JobId,
        attempt: Option<FencedAttempt>,
        event: JobEvent,
        at: Timestamp,
    ) -> Result<jain_smartcluster_core::EventRecord, EventLogError> {
        let mut log = self
            .log
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let offset = log.next_offset();
        let record = jain_smartcluster_core::EventRecord {
            offset: EventOffset::new(offset),
            job_id,
            attempt,
            at,
            event,
        };
        let committed = log.append(&record)?;
        debug_assert_eq!(committed, offset);
        drop(log);
        self.fanout.publish(offset, record.clone());
        Ok(record)
    }

    pub(super) fn snapshot_and_subscribe(
        &self,
    ) -> (u64, FanoutSubscriber<jain_smartcluster_core::EventRecord>) {
        let log = self
            .log
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let end = log.next_offset();
        let subscriber = self.fanout.subscribe(end);
        (end, subscriber)
    }
}

pub(super) struct EventStream {
    pub(super) stream_id: StreamId,
    pub(super) requested: u64,
    pub(super) snapshot_end: u64,
    pub(super) filter: EventFilter,
    pub(super) peer: AuthenticatedPeer,
    pub(super) authorizer: Authorizer,
    pub(super) store: Arc<Store>,
    pub(super) journal: EventJournal,
    pub(super) subscriber: FanoutSubscriber<jain_smartcluster_core::EventRecord>,
    pub(super) sink: ConnectionSink,
    pub(super) cancelled: watch::Receiver<bool>,
    pub(super) activated: oneshot::Receiver<()>,
    pub(super) actor: mpsc::Sender<ActorCommand>,
}

pub(super) fn spawn_event_stream(mut stream: EventStream) {
    tokio::spawn(async move {
        let _ended = StreamEndGuard {
            stream_id: stream.stream_id,
            actor: stream.actor.clone(),
        };
        tokio::select! {
            result = &mut stream.activated => {
                if result.is_err() { return; }
            }
            () = stream.sink.disconnected() => return,
            changed = stream.cancelled.changed() => {
                if changed.is_err() || *stream.cancelled.borrow() { return; }
            }
        }
        let mut cursor = stream.requested;
        while cursor < stream.snapshot_end
            && !*stream.cancelled.borrow()
            && !stream.sink.is_disconnected()
        {
            let batch = {
                let log = stream
                    .journal
                    .log
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                log.read_from(cursor, EVENT_READ_BATCH)
            };
            let Ok(batch) = batch else {
                stream.sink.disconnect();
                return;
            };
            if batch.resume_offset == cursor && batch.items.is_empty() {
                break;
            }
            for item in batch.items {
                match item {
                    LogItem::Event(event) => {
                        if event.offset >= stream.snapshot_end {
                            break;
                        }
                        if event_visible(
                            &stream.store,
                            &stream.authorizer,
                            &stream.peer,
                            &stream.filter,
                            &event.event,
                        ) && stream
                            .sink
                            .try_send(ServerFrame::Event(EventEnvelope {
                                stream_id: stream.stream_id,
                                event: event.event,
                            }))
                            .is_err()
                        {
                            return;
                        }
                    }
                    LogItem::Gap(gap) => {
                        let marker = resumable_gap(
                            gap.first_missing,
                            gap.next_available,
                            stream.snapshot_end,
                            "event segments were compacted",
                        );
                        if stream
                            .sink
                            .try_send(ServerFrame::StreamGap {
                                stream_id: stream.stream_id,
                                gap: marker,
                            })
                            .is_err()
                        {
                            return;
                        }
                    }
                }
            }
            cursor = batch.resume_offset.min(stream.snapshot_end);
        }

        loop {
            tokio::select! {
                () = stream.sink.disconnected() => return,
                changed = stream.cancelled.changed() => {
                    if changed.is_err() || *stream.cancelled.borrow() { return; }
                }
                delivery = stream.subscriber.receive() => match delivery {
                    FanoutDelivery::Event(event) => {
                        if event_visible(
                            &stream.store,
                            &stream.authorizer,
                            &stream.peer,
                            &stream.filter,
                            &event.value,
                        ) && stream.sink.try_send(ServerFrame::Event(EventEnvelope {
                            stream_id: stream.stream_id,
                            event: event.value,
                        })).is_err() {
                            return;
                        }
                    }
                    FanoutDelivery::Disconnected(gap) => {
                        let marker = resumable_gap(
                            gap.requested_offset,
                            gap.resume_from_offset,
                            gap.resume_from_offset,
                            "subscriber fell behind the bounded event fanout",
                        );
                        if stream.sink.try_send(ServerFrame::StreamGap {
                            stream_id: stream.stream_id,
                            gap: marker,
                        }).is_ok() {
                            let _ = stream.sink.close_stream(stream.stream_id, None);
                        }
                        return;
                    }
                    FanoutDelivery::Closed => return,
                }
            }
        }
    });
}

pub(super) fn resumable_gap(
    requested: u64,
    available_from: u64,
    end_offset: u64,
    reason: &str,
) -> StreamGap {
    let available_from = available_from.max(requested.saturating_add(1));
    StreamGap {
        requested: EventOffset::new(requested),
        available_from: EventOffset::new(available_from),
        latest: EventOffset::new(end_offset.saturating_sub(1).max(available_from)),
        reason: reason.to_owned(),
    }
}

struct StreamEndGuard {
    stream_id: StreamId,
    actor: mpsc::Sender<ActorCommand>,
}

impl Drop for StreamEndGuard {
    fn drop(&mut self) {
        let _ = self
            .actor
            .try_send(ActorCommand::StreamEnded(self.stream_id));
    }
}

pub(super) fn event_visible(
    store: &Store,
    authorizer: &Authorizer,
    peer: &AuthenticatedPeer,
    filter: &EventFilter,
    event: &jain_smartcluster_core::EventRecord,
) -> bool {
    let Ok(Some(job)) = store.get_job(event.job_id) else {
        return false;
    };
    authorizer
        .authorize_job_access(peer, &job.submitted_by)
        .is_ok()
        && matches_event_filter(&job, filter)
}

pub(super) fn matches_event_filter(job: &JobRecord, filter: &EventFilter) -> bool {
    filter.job_id.is_none_or(|job_id| job.id == job_id)
        && filter
            .tenant
            .as_ref()
            .is_none_or(|tenant| &job.spec.tenant == tenant)
        && filter
            .project
            .as_ref()
            .is_none_or(|project| &job.spec.project == project)
        && filter.owner.as_ref().is_none_or(|owner| {
            matches!(&job.submitted_by, Principal::Web { owner: actual } if actual == owner)
        })
}

pub(super) fn matches_list_filter(job: &JobRecord, filter: &ListFilter) -> bool {
    filter
        .tenant
        .as_ref()
        .is_none_or(|tenant| &job.spec.tenant == tenant)
        && filter
            .project
            .as_ref()
            .is_none_or(|project| &job.spec.project == project)
        && filter.owner.as_ref().is_none_or(|owner| {
            matches!(&job.submitted_by, Principal::Web { owner: actual } if actual == owner)
        })
        && (filter.states.is_empty() || filter.states.contains(&job.state))
        && filter
            .labels
            .iter()
            .all(|(name, value)| job.spec.labels.get(name) == Some(value))
}
