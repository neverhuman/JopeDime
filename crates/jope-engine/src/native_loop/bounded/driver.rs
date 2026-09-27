//! The bounded-run coordinator process (`jope-native-loop coordinate`, W7-T05/T07).
//!
//! One invocation drives the journaled [`Coordinator`] state machine to a terminal outcome:
//! * a restarted coordinator (a non-terminal journal that already dispatched work) only
//!   reconciles: it commits a complete authenticated group once, or drains the whole group;
//! * otherwise it journals a preflight of a fresh census, opens the rendezvous, dispatches
//!   every rank through its [`Supervisor`], gathers the authenticated full group over the
//!   pinned-TLS transport, journals the admission before delivering it, then heartbeats the
//!   group and polls until the group commits or drains (rank failure, lost control, grant
//!   expiry, deadline or a `cancel` request).
//!
//! The coordinator presents its planned host certificate and signs with its planned key; both
//! are checked against the frozen plan before anything is dispatched.
use super::{
    controller::{Coordinator, StatusReport, Supervisor},
    participant::load_signing_key,
    rendezvous::{self, write_frame, Connected, ControlFrame},
    transport::{self, HostCredential, Timeouts, TlsAcceptor},
};
use anyhow::{ensure, Context, Result};
use std::{
    net::SocketAddr,
    path::PathBuf,
    sync::atomic::{AtomicBool, Ordering},
    time::Duration,
};

#[derive(Clone, Debug)]
pub struct CoordinateArgs {
    pub run_dir: PathBuf,
    pub plan_sha256: String,
    /// The merged census, preflighted and journaled before dispatch.
    pub census: PathBuf,
    pub listen: SocketAddr,
    pub certificate: PathBuf,
    pub certificate_key: PathBuf,
    pub coordinator_key: PathBuf,
    /// Heartbeat interval; must be below the plan's heartbeat timeout.
    pub heartbeat: Duration,
}

fn now_ms() -> u64 {
    crate::resident_protocol::now_ms().unwrap_or(0)
}
fn boot_id() -> Result<String> {
    Ok(std::fs::read_to_string("/proc/sys/kernel/random/boot_id")?
        .trim()
        .to_string())
}
fn heartbeat_all(connected: &mut [Connected], id: u64, limit: u32) {
    let frame =
        serde_json::to_vec(&ControlFrame::Heartbeat { delivery_id: id }).unwrap_or_default();
    for c in connected.iter_mut() {
        // A failed write is observed through the rank's own exit (lost control) at poll.
        let _ = write_frame(&mut c.stream, &frame, limit);
    }
}

/// Drive the planned run to a terminal outcome, or until `stop` is set (the process may then
/// be restarted; the journal makes that a reconciliation).
pub fn coordinate<S: Supervisor>(
    args: &CoordinateArgs,
    supervisor: S,
    stop: &AtomicBool,
) -> Result<StatusReport> {
    let mut c = Coordinator::open(&args.run_dir, &args.plan_sha256, supervisor, now_ms())?;
    if c.needs_reconcile() {
        return c.reconcile(now_ms());
    }
    let status = c.status()?;
    if status.outcome.is_some() {
        return Ok(status);
    }
    let contract = c.contract().clone();
    ensure!(
        args.heartbeat < Duration::from_millis(contract.rendezvous.heartbeat_timeout_ms),
        "the heartbeat interval must be below the plan's heartbeat timeout"
    );
    let key = load_signing_key(&args.coordinator_key)?;
    ensure!(
        hex::encode(key.verifying_key().to_bytes()) == contract.coordinator_public_key,
        "coordinator key is not the planned coordinator key"
    );
    let credential = HostCredential::load(&args.certificate, &args.certificate_key)?;
    let (pinned, host) = transport::coordinator_certificate(&contract)?;
    ensure!(
        credential.certificate_sha256() == pinned,
        "coordinator certificate is not the planned certificate of host `{host}`"
    );
    let census = std::fs::read(&args.census).context("census")?;
    c.preflight(&census, now_ms())?;
    let mut acceptor = TlsAcceptor::bind(
        args.listen,
        &credential,
        transport::planned_certificates(&contract)?,
        Timeouts::from_plan(&contract),
    )?;
    let mut rv = c.open_rendezvous(&key, &boot_id()?, &rendezvous::fresh_nonce()?, now_ms())?;
    c.dispatch_all(now_ms())?;
    let (admission, mut connected) = match rendezvous::gather(&mut acceptor, &mut rv, &key, &now_ms)
    {
        Ok(gathered) => gathered,
        Err(refusal) => {
            c.record_refusal(&refusal, now_ms())?;
            return c.status();
        }
    };
    let persisted = c.record_admission(admission, now_ms())?;
    c.deliver_admission(&persisted, &mut connected, now_ms())?;
    let limit = contract.rendezvous.max_message_bytes;
    let mut next = 0u64;
    loop {
        ensure!(
            !stop.load(Ordering::Acquire),
            "coordinator stopped; restart it to reconcile"
        );
        next += 1;
        heartbeat_all(&mut connected, next, limit);
        let status = c.poll(now_ms())?;
        if status.outcome.is_some() {
            return Ok(status);
        }
        std::thread::sleep(args.heartbeat);
    }
}
