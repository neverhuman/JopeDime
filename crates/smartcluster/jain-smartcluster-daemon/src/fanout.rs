use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use tokio::sync::broadcast;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Published<T> {
    pub offset: u64,
    pub value: T,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FanoutGap {
    pub requested_offset: u64,
    pub resume_from_offset: u64,
    pub missed_events: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FanoutDelivery<T> {
    Event(Published<T>),
    Disconnected(FanoutGap),
    Closed,
}

#[derive(Debug)]
pub struct EventFanout<T: Clone> {
    sender: broadcast::Sender<Published<T>>,
    latest_offset: Arc<AtomicU64>,
}

impl<T: Clone> Clone for EventFanout<T> {
    fn clone(&self) -> Self {
        Self {
            sender: self.sender.clone(),
            latest_offset: Arc::clone(&self.latest_offset),
        }
    }
}

impl<T: Clone> EventFanout<T> {
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "fanout capacity must be positive");
        let (sender, _) = broadcast::channel(capacity);
        Self {
            sender,
            latest_offset: Arc::new(AtomicU64::new(0)),
        }
    }

    pub fn publish(&self, offset: u64, value: T) {
        self.latest_offset.fetch_max(offset, Ordering::Release);
        let _ = self.sender.send(Published { offset, value });
    }

    pub fn subscribe(&self, next_offset: u64) -> FanoutSubscriber<T> {
        FanoutSubscriber {
            receiver: self.sender.subscribe(),
            latest_offset: Arc::clone(&self.latest_offset),
            next_offset,
            disconnected: false,
        }
    }
}

pub struct FanoutSubscriber<T: Clone> {
    receiver: broadcast::Receiver<Published<T>>,
    latest_offset: Arc<AtomicU64>,
    next_offset: u64,
    disconnected: bool,
}

impl<T: Clone> FanoutSubscriber<T> {
    pub async fn receive(&mut self) -> FanoutDelivery<T> {
        if self.disconnected {
            return FanoutDelivery::Closed;
        }
        loop {
            match self.receiver.recv().await {
                Ok(event) if event.offset < self.next_offset => continue,
                Ok(event) => {
                    self.next_offset = event.offset.saturating_add(1);
                    return FanoutDelivery::Event(event);
                }
                Err(broadcast::error::RecvError::Lagged(missed)) => {
                    self.disconnected = true;
                    let latest = self.latest_offset.load(Ordering::Acquire);
                    return FanoutDelivery::Disconnected(FanoutGap {
                        requested_offset: self.next_offset,
                        resume_from_offset: latest.saturating_add(1),
                        missed_events: missed,
                    });
                }
                Err(broadcast::error::RecvError::Closed) => {
                    self.disconnected = true;
                    return FanoutDelivery::Closed;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn slow_subscriber_is_disconnected_with_resume_offset() {
        let fanout = EventFanout::new(2);
        let mut subscriber = fanout.subscribe(1);
        for offset in 1..=4 {
            fanout.publish(offset, offset);
        }
        assert_eq!(
            subscriber.receive().await,
            FanoutDelivery::Disconnected(FanoutGap {
                requested_offset: 1,
                resume_from_offset: 5,
                missed_events: 2,
            })
        );
        assert_eq!(subscriber.receive().await, FanoutDelivery::Closed);
    }

    #[tokio::test]
    async fn old_offsets_are_filtered() {
        let fanout = EventFanout::new(4);
        let mut subscriber = fanout.subscribe(3);
        fanout.publish(2, "old");
        fanout.publish(3, "wanted");
        assert_eq!(
            subscriber.receive().await,
            FanoutDelivery::Event(Published {
                offset: 3,
                value: "wanted"
            })
        );
    }
}
