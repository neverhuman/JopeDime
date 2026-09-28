//! W7-T07 cross-host supervision boundaries: a host agent serves only its planned ranks of its
//! own frozen plan and only to the plan's coordinator certificate; the coordinator's remote
//! supervisor reaches each rank only through the planned certificate of that rank's host.
//! In-memory supervisor behind the agent; loopback TLS with in-test certificates. The real
//! SmartCluster path is `bounded_smartcluster`.
#[path = "support/bounded_fixture.rs"]
mod fixture;
#[path = "support/tls_fixture.rs"]
mod tls;
use fixture::*;
use jope_parallel::native_loop::bounded::{
    agent::{agent_acceptor, Agent, AgentReply, AgentRequest, RemoteSupervisor, AGENT_FRAME_LIMIT},
    contract::BoundedRunContract,
    controller::{Liveness, RankLaunch, Supervisor, SupervisorHandle},
    rendezvous,
    transport::{self, HostCredential, Sequenced, Timeouts},
};
use std::{
    collections::{BTreeMap, BTreeSet},
    net::SocketAddr,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread,
    time::{Duration, Instant},
};
use tls::*;

fn credential(c: &TestCertificate) -> HostCredential {
    HostCredential::from_der(c.der.clone(), c.pkcs8.clone()).unwrap()
}
/// The fixture plan (coordinator on `fixture-host-1`) with in-test host certificates.
struct Planned {
    contract: BoundedRunContract,
    plan: Vec<u8>,
    sha: String,
    certs: Vec<TestCertificate>,
}
fn planned() -> Planned {
    planned_with(|_| {})
}
fn planned_with(adjust: impl FnOnce(&mut BoundedRunContract)) -> Planned {
    let k = keys(3);
    let certs: Vec<_> = (0..3)
        .map(|i| current(&format!("fixture-host-{i}"), 61 + i as u8))
        .collect();
    let mut c = contract(now_ms(), 3, &k);
    c.run = format!("w7-agent-{}", certs[0].sha256.get(..8).unwrap());
    for (p, cert) in c.participants.iter_mut().zip(&certs) {
        p.host.certificate_sha256 = cert.sha256.clone();
    }
    adjust(&mut c);
    resign(&mut c, &k);
    c.validate_structure().unwrap();
    let plan = c.canonical_bytes().unwrap();
    Planned {
        sha: sha(&plan),
        contract: c,
        plan,
        certs,
    }
}
fn dispatch(p: &Planned, rank: u32) -> String {
    rendezvous::dispatch_id(&p.sha, p.contract.attempt, rank).unwrap()
}
fn refused(reply: AgentReply) -> String {
    match reply {
        AgentReply::Refused { reason } => reason,
        other => panic!("expected a refusal, got {other:?}"),
    }
}

#[test]
fn an_agent_serves_only_its_own_plan_and_planned_ranks() {
    let p = planned();
    let fake = FakeSupervisor::default();
    // Its plan must be the reviewed canonical bytes on a host the plan names for every rank.
    let other = sha(b"another plan");
    let e = Agent::new(&p.plan, &other, "fixture-host-0", None, fake.clone()).err();
    assert!(e
        .unwrap()
        .to_string()
        .contains("differ from the reviewed plan digest"));
    let e = Agent::new(
        &p.plan,
        &p.sha,
        "fixture-host-0",
        Some(BTreeSet::from([1])),
        fake.clone(),
    )
    .err();
    assert!(e
        .unwrap()
        .to_string()
        .contains("not planned for every rank"));
    assert!(Agent::new(&p.plan, &p.sha, "fixture-host-9", None, fake.clone()).is_err());
    let mut agent = Agent::new(&p.plan, &p.sha, "fixture-host-0", None, fake.clone()).unwrap();
    // Another rank, or a dispatch id that is not the rank's, starts nothing.
    for (global_rank, id) in [
        (1, dispatch(&p, 1)),
        (0, dispatch(&p, 1)),
        (0, sha(b"forged")),
    ] {
        let reason = refused(agent.handle(AgentRequest::Launch {
            dispatch_intent_unix_ms: now_ms(),
            global_rank,
            dispatch_id: id,
        }));
        assert!(reason.contains("not a rank this agent serves"), "{reason}");
    }
    let reason = refused(agent.handle(AgentRequest::Find {
        dispatch_id: dispatch(&p, 2),
    }));
    assert!(reason.contains("not a rank this agent serves"), "{reason}");
    assert_eq!(fake.0.lock().unwrap().launches, 0);
    // Positive control: its own rank launches once, under the plan's dispatch id.
    let handle = match agent.handle(AgentRequest::Launch {
        dispatch_intent_unix_ms: now_ms(),
        global_rank: 0,
        dispatch_id: dispatch(&p, 0),
    }) {
        AgentReply::Handle { handle } => handle,
        other => panic!("{other:?}"),
    };
    assert_eq!(handle.identity, dispatch(&p, 0));
    assert_eq!(fake.0.lock().unwrap().launches, 1);
    assert_eq!(
        agent.handle(AgentRequest::Find {
            dispatch_id: dispatch(&p, 0)
        }),
        AgentReply::Found {
            handle: Some(handle)
        }
    );
    // The supervisor's own refusal is a reply, never a partial action.
    let reason = refused(agent.handle(AgentRequest::Launch {
        dispatch_intent_unix_ms: now_ms(),
        global_rank: 0,
        dispatch_id: dispatch(&p, 0),
    }));
    assert!(reason.contains("duplicate launch"), "{reason}");
    assert_eq!(fake.0.lock().unwrap().launches, 1);
}

/// An agent for `host` presenting `cert`, serving on loopback until the flag is set.
fn serve(
    p: &Planned,
    host: &str,
    cert: &TestCertificate,
    fake: FakeSupervisor,
) -> (SocketAddr, Arc<AtomicBool>, thread::JoinHandle<()>) {
    let mut agent = Agent::new(&p.plan, &p.sha, host, None, fake).unwrap();
    let acceptor = agent_acceptor(
        &p.contract,
        "127.0.0.1:0".parse().unwrap(),
        &credential(cert),
        host,
    )
    .unwrap();
    let address = acceptor.local_addr().unwrap();
    let stop = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&stop);
    let t = thread::spawn(move || agent.serve(&acceptor, &flag).unwrap());
    (address, stop, t)
}

#[test]
fn remote_supervision_is_mutually_pinned_by_the_plan() {
    let p = planned();
    // An agent presents its own host's planned certificate, nothing else.
    let e = agent_acceptor(
        &p.contract,
        "127.0.0.1:0".parse().unwrap(),
        &credential(&p.certs[2]),
        "fixture-host-0",
    )
    .err();
    assert!(e
        .unwrap()
        .to_string()
        .contains("not host `fixture-host-0`'s planned certificate"));
    let fake = FakeSupervisor::default();
    let (address, stop, t) = serve(&p, "fixture-host-0", &p.certs[0], fake.clone());
    // A planned host that is not the coordinator is refused by the agent's handshake and
    // delivers nothing.
    let t2 = Timeouts {
        connect: Duration::from_secs(5),
        handshake: Duration::from_secs(5),
        io: Duration::from_secs(5),
    };
    let delivered = transport::connect(
        address,
        &credential(&p.certs[2]),
        (&p.certs[0].sha256, "fixture-host-0"),
        t2,
    )
    .and_then(|s| {
        let mut ch = Sequenced::new(s, AGENT_FRAME_LIMIT);
        ch.send(
            &serde_json::to_vec(&AgentRequest::Launch {
                dispatch_intent_unix_ms: now_ms(),
                global_rank: 0,
                dispatch_id: dispatch(&p, 0),
            })
            .unwrap(),
        )?;
        ch.receive()
    });
    assert!(
        delivered.is_err(),
        "a non-coordinator certificate was served"
    );
    assert_eq!(fake.0.lock().unwrap().launches, 0);
    // The remote supervisor presents the coordinator's planned certificate only, and needs an
    // agent for every planned rank.
    let all = |a: SocketAddr| (0..3u32).map(|r| (r, a)).collect::<BTreeMap<_, _>>();
    let e = RemoteSupervisor::new(&p.contract, &p.sha, credential(&p.certs[0]), all(address)).err();
    assert!(e
        .unwrap()
        .to_string()
        .contains("presents its planned host certificate"));
    let mut partial = all(address);
    partial.remove(&2);
    let e = RemoteSupervisor::new(&p.contract, &p.sha, credential(&p.certs[1]), partial).err();
    assert!(e.unwrap().to_string().contains("exactly one agent address"));
    let e = RemoteSupervisor::new(
        &p.contract,
        &sha(b"x"),
        credential(&p.certs[1]),
        all(address),
    )
    .err();
    assert!(e.unwrap().to_string().contains("does not name this plan"));
    // Every rank is routed to host 0's agent here: rank 0 is served; rank 1's operations
    // reach an agent that presents host 0's certificate, not host 1's, and are refused.
    let mut remote =
        RemoteSupervisor::new(&p.contract, &p.sha, credential(&p.certs[1]), all(address)).unwrap();
    let id0 = dispatch(&p, 0);
    let handle = remote
        .launch(&RankLaunch {
            dispatch_intent_unix_ms: now_ms(),
            contract: &p.contract,
            plan_sha256: &p.sha,
            global_rank: 0,
            dispatch_id: &id0,
        })
        .unwrap();
    assert_eq!(handle.supervisor, "remote:0:fake");
    assert_eq!(remote.probe(&handle).unwrap(), Liveness::Running);
    assert_eq!(remote.find(&id0).unwrap(), Some(handle.clone()));
    let id1 = dispatch(&p, 1);
    assert!(remote
        .launch(&RankLaunch {
            dispatch_intent_unix_ms: now_ms(),
            contract: &p.contract,
            plan_sha256: &p.sha,
            global_rank: 1,
            dispatch_id: &id1,
        })
        .is_err());
    assert_eq!(fake.0.lock().unwrap().launches, 1);
    // A handle naming no remote rank, or a dispatch id naming no planned rank, is refused
    // before anything is sent.
    let mut forged = handle.clone();
    forged.supervisor = "fake".into();
    assert!(remote.probe(&forged).is_err());
    assert!(remote.find(&sha(b"no rank")).is_err());
    // Terminate and the retained result travel back through the agent.
    assert_eq!(
        remote.terminate(&handle).unwrap(),
        Liveness::Exited {
            code: None,
            signal: Some(15)
        }
    );
    assert_eq!(remote.retained_result(&handle).unwrap(), None);
    // A restarted coordinator's new connection replaces the old one.
    drop(remote);
    let mut again =
        RemoteSupervisor::new(&p.contract, &p.sha, credential(&p.certs[1]), all(address)).unwrap();
    assert_eq!(again.find(&id0).unwrap(), Some(handle));
    stop.store(true, Ordering::Release);
    t.join().unwrap();
}

#[test]
fn an_agent_refuses_malformed_and_replayed_requests() {
    let p = planned();
    let fake = FakeSupervisor::default();
    let (address, stop, t) = serve(&p, "fixture-host-0", &p.certs[0], fake.clone());
    let timeouts = Timeouts::from_plan(&p.contract);
    let pin = (p.certs[0].sha256.as_str(), "fixture-host-0");
    // A malformed request is answered with a refusal; the channel stays usable.
    let stream = transport::connect(address, &credential(&p.certs[1]), pin, timeouts).unwrap();
    let mut ch = Sequenced::new(stream, AGENT_FRAME_LIMIT);
    ch.send(br#"{"op":"launch","global_rank":0}"#).unwrap();
    let (_, reply) = ch.receive().unwrap();
    let reason = refused(serde_json::from_slice(&reply).unwrap());
    assert!(reason.starts_with("malformed request"), "{reason}");
    // A frame that repeats a delivery id ends the connection with nothing served.
    use rendezvous::{read_frame, write_frame};
    let frame = |id: u64, body: &[u8]| [&id.to_be_bytes()[..], body].concat();
    let launch = serde_json::to_vec(&AgentRequest::Launch {
        dispatch_intent_unix_ms: now_ms(),
        global_rank: 0,
        dispatch_id: dispatch(&p, 0),
    })
    .unwrap();
    let find = br#"{"op":"find","dispatch_id":"x"}"#;
    let mut stream = transport::connect(address, &credential(&p.certs[1]), pin, timeouts).unwrap();
    write_frame(&mut stream, &frame(1, find), AGENT_FRAME_LIMIT).unwrap();
    let reply = read_frame(&mut stream, AGENT_FRAME_LIMIT).unwrap();
    let reason = refused(serde_json::from_slice(&reply[8..]).unwrap());
    assert!(reason.contains("not a rank this agent serves"), "{reason}");
    write_frame(&mut stream, &frame(1, &launch), AGENT_FRAME_LIMIT).unwrap();
    assert!(read_frame(&mut stream, AGENT_FRAME_LIMIT).is_err());
    // An oversized frame is refused without being read.
    let mut stream = transport::connect(address, &credential(&p.certs[1]), pin, timeouts).unwrap();
    let huge = frame(1, &vec![b' '; AGENT_FRAME_LIMIT as usize]);
    let _ = write_frame(&mut stream, &huge, AGENT_FRAME_LIMIT + 8);
    assert!(read_frame(&mut stream, AGENT_FRAME_LIMIT).is_err());
    assert_eq!(fake.0.lock().unwrap().launches, 0);
    stop.store(true, Ordering::Release);
    t.join().unwrap();
}

/// The fixture plan with a 300 ms I/O (heartbeat) timeout.
fn quick_io() -> (Planned, Duration) {
    (
        planned_with(|c| c.rendezvous.heartbeat_timeout_ms = 300),
        Duration::from_millis(300),
    )
}

/// A remote call's request and reply end by its reply deadline: an agent that authenticates
/// with its planned certificate and then trickles its reply cannot hold the coordinator
/// (and so its supervision, deadline and stop checks) any longer.
#[test]
fn a_trickling_agent_cannot_hold_a_remote_call_past_its_reply_deadline() {
    let (p, io) = quick_io();
    let acceptor = agent_acceptor(
        &p.contract,
        "127.0.0.1:0".parse().unwrap(),
        &credential(&p.certs[0]),
        "fixture-host-0",
    )
    .unwrap();
    let address = acceptor.local_addr().unwrap();
    let agent = thread::spawn(move || {
        let start = Instant::now();
        let mut s = loop {
            if let Ok(Some(s)) = acceptor.poll_accept() {
                break s;
            }
            assert!(start.elapsed() < Duration::from_secs(10), "no coordinator");
            thread::sleep(Duration::from_millis(5));
        };
        rendezvous::read_frame(&mut s, AGENT_FRAME_LIMIT).unwrap();
        trickle(s, &1_000u32.to_be_bytes());
    });
    let all = (0..3u32).map(|r| (r, address)).collect();
    let mut remote =
        RemoteSupervisor::new(&p.contract, &p.sha, credential(&p.certs[1]), all).unwrap();
    let handle = SupervisorHandle {
        supervisor: "remote:0:fake".into(),
        identity: dispatch(&p, 0),
    };
    let start = Instant::now();
    assert!(
        remote.probe(&handle).is_err(),
        "a trickled reply was accepted"
    );
    ended_at_deadline(
        start.elapsed(),
        io,
        "a remote call held by a trickled reply",
    );
    drop(remote);
    agent.join().expect("the agent never reached its trickle");
}

/// An agent serves one request at a time; a request frame that has started must arrive whole
/// within the plan's I/O timeout, so a coordinator that trickles it cannot hold the agent
/// past its stop flag.
#[test]
fn a_trickling_coordinator_cannot_hold_an_agent_past_its_stop() {
    let (p, io) = quick_io();
    let (address, stop, served) =
        serve(&p, "fixture-host-0", &p.certs[0], FakeSupervisor::default());
    let s = transport::connect(
        address,
        &credential(&p.certs[1]),
        (&p.certs[0].sha256, "fixture-host-0"),
        Timeouts::from_plan(&p.contract),
    )
    .unwrap();
    let coordinator = thread::spawn(move || trickle(s, &1_000u32.to_be_bytes()));
    thread::sleep(Duration::from_millis(100));
    let start = Instant::now();
    stop.store(true, Ordering::Release);
    while !served.is_finished() {
        assert!(
            start.elapsed() < Duration::from_secs(10),
            "the agent never stopped"
        );
        thread::sleep(Duration::from_millis(5));
    }
    let elapsed = start.elapsed();
    served.join().unwrap();
    assert!(
        elapsed <= io + SLACK,
        "the agent stopped {elapsed:?} after its stop flag; a started frame has {io:?}"
    );
    coordinator.join().unwrap();
}
