//! Focused GPU queue tests.

use super::*;
use std::sync::atomic::AtomicUsize;

fn temp_queue_dir(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "jain-gpu-queue-test-{name}-{}-{}",
        std::process::id(),
        TICKET_SEQ.fetch_add(1, Ordering::Relaxed)
    ))
}

#[test]
fn queue_acquires_in_fifo_filename_order() {
    let dir = temp_queue_dir("fifo");
    fs::create_dir_all(&dir).unwrap();
    let old = dir.join("00000000000000000001-0000000001-0000000000000000.ticket");
    let new = dir.join("00000000000000000002-0000000001-0000000000000000.ticket");
    fs::write(&new, "new").unwrap();
    fs::write(&old, "old").unwrap();

    let tickets = list_tickets(&dir).unwrap();
    assert_eq!(tickets, vec![old, new]);

    let _ = fs::remove_dir_all(dir);
}

#[test]
fn queue_cleans_expired_tickets() {
    let dir = temp_queue_dir("expired");
    fs::create_dir_all(&dir).unwrap();
    let expired = dir.join("00000000000000000001-0000000001-0000000000000000.ticket");
    fs::write(&expired, "expired").unwrap();
    std::thread::sleep(Duration::from_millis(5));

    let removed = cleanup_expired_tickets(&dir, Duration::from_millis(1)).unwrap();
    assert_eq!(removed, 1);
    assert!(!expired.exists());

    let _ = fs::remove_dir_all(dir);
}

#[test]
fn ticket_storage_handles_missing_directories_and_heartbeats() {
    let dir = temp_queue_dir("storage");
    assert!(list_tickets(&dir).unwrap().is_empty());

    fs::create_dir_all(&dir).unwrap();
    let ticket = create_ticket(&dir).unwrap();
    touch_ticket(&ticket).unwrap();

    let contents = fs::read_to_string(&ticket).unwrap();
    assert!(contents.contains(&format!("pid={}", std::process::id())));
    assert!(contents.contains("heartbeat_ns="));
    assert_eq!(list_tickets(&dir).unwrap(), vec![ticket]);

    let _ = fs::remove_dir_all(dir);
}

#[test]
fn thirty_x_bypass_reports_cpu_handoff() {
    let reason = gpu_30x_bypass_reason(
        Duration::from_secs(1),
        Duration::from_secs(31),
        Duration::from_millis(1),
    )
    .unwrap();
    assert!(reason.contains("CPU"));
    assert!(reason.contains("30x"));

    let badge = gpu_cpu_handoff_badge(reason);
    assert_eq!(badge.backend, "cpu");
    assert!(badge.cpu_handoff_reason.contains("30x"));
}

#[test]
fn cpu_handoff_reason_classifies_queue_timeout() {
    let reason =
        gpu_cpu_handoff_reason("ridge batch", "GPU queue wait timed out after 1s (max 1s)");
    assert!(reason.contains("queue contention"));
    assert!(reason.contains("CPU handoff"));
}

#[test]
fn interrupted_wait_preserves_type_and_removes_its_ticket() {
    let dir = temp_queue_dir("interrupted");
    fs::create_dir_all(&dir).unwrap();
    let blocker = dir.join("00000000000000000001-0000000001-0000000000000000.ticket");
    fs::write(&blocker, "blocker").unwrap();
    let config = GpuQueueConfig {
        dir: dir.clone(),
        expired_after: Duration::from_secs(60),
        poll_interval: Duration::from_millis(1),
        heartbeat_interval: Duration::from_millis(1),
        max_wait: Duration::from_secs(1),
    };
    let calls = AtomicUsize::new(0);
    let checkpoint = || {
        if calls.fetch_add(1, Ordering::Relaxed) >= 1 {
            Err(GpuInterruption::Cancelled.into())
        } else {
            Ok(())
        }
    };

    let error = acquire_gpu_lease_with_request_and_checkpoint(
        &config,
        GpuLeaseRequest::default(),
        &checkpoint,
    )
    .unwrap_err();
    assert_eq!(
        error.downcast_ref::<GpuInterruption>(),
        Some(&GpuInterruption::Cancelled)
    );
    let mapped = gpu_cpu_handoff_unless_interrupted("test", GpuInterruption::Deadline.into());
    assert_eq!(
        mapped.downcast_ref::<GpuInterruption>(),
        Some(&GpuInterruption::Deadline)
    );
    assert_eq!(list_tickets(&dir).unwrap(), vec![blocker]);

    let _ = fs::remove_dir_all(dir);
}

#[test]
fn default_lease_removes_ticket_on_drop() {
    let dir = temp_queue_dir("lease");
    let config = GpuQueueConfig {
        dir: dir.clone(),
        expired_after: Duration::from_secs(60),
        poll_interval: Duration::from_millis(1),
        heartbeat_interval: Duration::from_millis(1),
        max_wait: Duration::from_secs(1),
    };
    let ticket;
    {
        let lease = acquire_gpu_lease(&config).unwrap();
        ticket = lease.ticket_path().to_path_buf();
        assert!(lease.waited() < Duration::from_secs(1));
        assert!(ticket.exists());
    }
    assert!(!ticket.exists());

    let _ = fs::remove_dir_all(dir);
}
