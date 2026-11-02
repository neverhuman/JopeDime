use super::*;
#[test]
fn malformed_owned_frame_is_an_error_not_clean_closure() {
    let (tap, mut receiver) = fact_channel(2, 4096);
    tap.shared.bytes.store(1, Ordering::Release);
    tap.shared.queued.store(1, Ordering::Release);
    tap.sender.try_send(Pending { bytes: vec![b'x'] }).unwrap();
    tap.close();
    assert!(receiver.recv_linked().is_err());
}
