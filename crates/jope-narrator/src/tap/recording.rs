use super::*;

/// Explicit test support; absent from normal production builds.
#[cfg(feature = "testing")]
pub struct RecordingTap {
    tap: Tap,
    facts: RecordedFacts,
}
#[cfg(feature = "testing")]
impl Default for RecordingTap {
    fn default() -> Self {
        let (mut tap, receiver) = channel_with(Limits::default());
        drop(receiver);
        // The collector has its own explicit lifetime; no production receiver is used.
        tap.shared.closing.store(false, Ordering::Release);
        let facts = Arc::new(Mutex::new(Vec::new()));
        tap.recording = Some(facts.clone());
        Self { tap, facts }
    }
}
#[cfg(feature = "testing")]
impl RecordingTap {
    pub fn tap(&self) -> Tap {
        self.tap.clone()
    }
    pub fn facts(&self) -> Vec<LineageFact> {
        self.linked_facts().into_iter().map(|p| p.0).collect()
    }
    pub fn linked_facts(&self) -> Vec<(LineageFact, Links)> {
        self.facts.lock().expect("test collector poisoned").clone()
    }
}
