use super::*;

type Slot = Option<(usize, Child, Instant)>;
pub(super) struct OwnedSlots(Vec<Slot>);
impl std::ops::Deref for OwnedSlots {
    type Target = Vec<Slot>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl std::ops::DerefMut for OwnedSlots {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
impl Drop for OwnedSlots {
    fn drop(&mut self) {
        for (_, child, _) in self.0.iter_mut().flatten() {
            // SAFETY: kill takes only integers and has no memory-safety preconditions; every
            // path that reaps a worker frees its slot first, so a child still in a slot is
            // unreaped and its setpgid(0, 0) group id cannot have been reused.
            unsafe {
                libc::kill(-(child.id() as i32), libc::SIGKILL);
            }
            let _ = child.wait();
        }
    }
}
pub(super) static STOP: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
pub(super) extern "C" fn stop(_: libc::c_int) {
    STOP.store(true, std::sync::atomic::Ordering::Relaxed);
}
