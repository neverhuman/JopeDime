//! Cross-host supervision for the bounded run (W7-T07).
//!
//! The coordinator runs on one host; each rank is a SmartCluster job of its own host's daemon,
//! whose socket is local to that host. An [`Agent`] on every host serves the [`Supervisor`]
//! operations of that host's supervisor to the coordinator over the pinned-TLS transport, and a
//! [`RemoteSupervisor`] on the coordinator routes each rank's operations to that rank's agent.
//! SSH is never the scheduler.
//!
//! * An agent accepts only the coordinator host's planned certificate; the remote supervisor
//!   accepts only the planned certificate of each rank's host.
//! * Requests and replies travel on a [`Sequenced`] channel: bounded frames, deadlines and
//!   strictly consecutive delivery ids.
//! * An agent holds its own copy of the frozen plan (by digest) and serves only the ranks it is
//!   given, all planned on its own host, under the dispatch ids the plan derives for them. It
//!   never starts work from a request that its plan does not name, and its host supervisor
//!   probes, cancels or reads only jobs of that plan.
use super::{
    contract::BoundedRunContract,
    controller::{Liveness, RankLaunch, Supervisor, SupervisorHandle},
    rendezvous,
    transport::{self, HostCredential, Sequenced, Timeouts, TlsAcceptor, TlsStream},
};
use anyhow::{anyhow, bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    net::SocketAddr,
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};

/// Frame bound for agent requests and replies (a retained result travels as hex).
pub const AGENT_FRAME_LIMIT: u32 = 16 * 1024 * 1024;
/// How long an idle agent waits for a request before it rechecks its stop flag and listener.
const IDLE_POLL: Duration = Duration::from_millis(500);
/// Bound on writing one reply, however slowly the coordinator reads it.
const REPLY_WRITE: Duration = Duration::from_secs(30);

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum AgentRequest {
    Launch {
        global_rank: u32,
        dispatch_id: String,
        dispatch_intent_unix_ms: u64,
    },
    Find {
        dispatch_id: String,
    },
    Probe {
        handle: SupervisorHandle,
    },
    Terminate {
        handle: SupervisorHandle,
    },
    RetainedResult {
        handle: SupervisorHandle,
    },
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "reply", rename_all = "snake_case", deny_unknown_fields)]
pub enum AgentReply {
    Handle {
        handle: SupervisorHandle,
    },
    Found {
        handle: Option<SupervisorHandle>,
    },
    Liveness {
        liveness: Liveness,
    },
    /// `(signed envelope, result bytes)`, hex.
    Retained {
        retained: Option<(String, String)>,
    },
    Refused {
        reason: String,
    },
}

/// Serves one host's supervisor to the planned coordinator.
pub struct Agent<S: Supervisor> {
    contract: BoundedRunContract,
    plan_sha256: String,
    ranks: BTreeSet<u32>,
    supervisor: S,
}
impl<S: Supervisor> Agent<S> {
    /// `plan` must be the canonical frozen plan bytes of `plan_sha256`. The agent serves the
    /// planned ranks of `host`, or only `ranks` of them when given.
    pub fn new(
        plan: &[u8],
        plan_sha256: &str,
        host: &str,
        ranks: Option<BTreeSet<u32>>,
        supervisor: S,
    ) -> Result<Self> {
        ensure!(
            crate::manifest::hash(plan) == plan_sha256,
            "plan bytes differ from the reviewed plan digest"
        );
        let contract = BoundedRunContract::parse(plan).map_err(|e| anyhow!("{e}"))?;
        ensure!(
            contract.canonical_bytes()? == plan,
            "plan bytes are not the canonical contract"
        );
        let planned: BTreeSet<u32> = contract
            .participants
            .iter()
            .filter(|p| p.host.host == host)
            .map(|p| p.global_rank)
            .collect();
        let ranks = ranks.unwrap_or_else(|| planned.clone());
        ensure!(
            !ranks.is_empty() && ranks.is_subset(&planned),
            "host `{host}` is not planned for every rank this agent would serve"
        );
        Ok(Self {
            contract,
            plan_sha256: plan_sha256.into(),
            ranks,
            supervisor,
        })
    }
    pub fn contract(&self) -> &BoundedRunContract {
        &self.contract
    }
    /// The served rank a dispatch id names, derived from this agent's own plan.
    fn served_rank(&self, dispatch_id: &str) -> Result<u32> {
        for rank in &self.ranks {
            if rendezvous::dispatch_id(&self.plan_sha256, self.contract.attempt, *rank)?
                == dispatch_id
            {
                return Ok(*rank);
            }
        }
        bail!("dispatch {dispatch_id} is not a rank this agent serves")
    }
    fn serve_one(&mut self, request: AgentRequest) -> Result<AgentReply> {
        Ok(match request {
            AgentRequest::Launch {
                global_rank,
                dispatch_id,
                dispatch_intent_unix_ms,
            } => {
                ensure!(
                    self.served_rank(&dispatch_id)? == global_rank,
                    "dispatch {dispatch_id} is not rank {global_rank}'s"
                );
                let handle = self.supervisor.launch(&RankLaunch {
                    contract: &self.contract,
                    plan_sha256: &self.plan_sha256,
                    global_rank,
                    dispatch_id: &dispatch_id,
                    dispatch_intent_unix_ms,
                })?;
                AgentReply::Handle { handle }
            }
            AgentRequest::Find { dispatch_id } => {
                self.served_rank(&dispatch_id)?;
                AgentReply::Found {
                    handle: self.supervisor.find(&dispatch_id)?,
                }
            }
            AgentRequest::Probe { handle } => AgentReply::Liveness {
                liveness: self.supervisor.probe(&handle)?,
            },
            AgentRequest::Terminate { handle } => AgentReply::Liveness {
                liveness: self.supervisor.terminate(&handle)?,
            },
            AgentRequest::RetainedResult { handle } => AgentReply::Retained {
                retained: self
                    .supervisor
                    .retained_result(&handle)?
                    .map(|(e, b)| (hex::encode(e), hex::encode(b))),
            },
        })
    }
    /// Answer one request. A refusal is a reply, never a partial action.
    pub fn handle(&mut self, request: AgentRequest) -> AgentReply {
        self.serve_one(request)
            .unwrap_or_else(|e| AgentReply::Refused {
                reason: format!("{e:#}"),
            })
    }
    /// Serve the coordinator until `stop` is set, one request at a time on one connection.
    /// Between requests a newly authenticated coordinator connection replaces the current
    /// one: a restarted coordinator reconnects while its predecessor's connection may still
    /// look open, and the run's coordinator lock admits one coordinator at a time. A
    /// connection that breaks its delivery sequence is dropped; one that fails to authenticate
    /// never delivers a request.
    ///
    /// Every blocking step is bounded by an absolute deadline, so `stop` is seen within a
    /// bounded time however a peer paces its bytes: an accept by the handshake allowance, an idle
    /// wait by `IDLE_POLL`, a started request by the plan's I/O timeout from its first byte, a
    /// reply by `REPLY_WRITE`, and a dropped stream's close by [`transport::CLOSE_TIMEOUT`].
    pub fn serve(&mut self, acceptor: &TlsAcceptor, stop: &AtomicBool) -> Result<()> {
        use super::rendezvous::ControlStream;
        let request_within = Duration::from_millis(self.contract.rendezvous.heartbeat_timeout_ms);
        let mut channel: Option<Sequenced<TlsStream>> = None;
        while !stop.load(Ordering::Acquire) {
            if let Ok(Some(stream)) = acceptor.poll_accept() {
                // Short idle wait so an idle coordinator never pins the loop past `stop` or
                // past a newer connection; an idle wait consumes nothing.
                if stream
                    .set_timeouts(Some(IDLE_POLL), Some(REPLY_WRITE))
                    .is_ok()
                {
                    channel = Some(Sequenced::new(stream, AGENT_FRAME_LIMIT));
                }
            }
            let Some(current) = channel.as_mut() else {
                std::thread::sleep(Duration::from_millis(20));
                continue;
            };
            let frame = match current.poll_receive(IDLE_POLL, request_within) {
                Ok(Some((_, frame))) => frame,
                Ok(None) => continue,
                Err(_) => {
                    channel = None;
                    continue;
                }
            };
            let reply = match serde_json::from_slice::<AgentRequest>(&frame) {
                Ok(request) => self.handle(request),
                Err(e) => AgentReply::Refused {
                    reason: format!("malformed request: {e}"),
                },
            };
            let reply = serde_json::to_vec(&reply)?;
            if current
                .send_by(&reply, Instant::now() + REPLY_WRITE)
                .is_err()
            {
                channel = None;
            }
        }
        Ok(())
    }
}
/// The listener an agent serves on: it presents `host`'s planned certificate and accepts only
/// the plan's coordinator certificate.
pub fn agent_acceptor(
    contract: &BoundedRunContract,
    listen: SocketAddr,
    credential: &HostCredential,
    host: &str,
) -> Result<TlsAcceptor> {
    let planned = transport::planned_certificates(contract)?;
    ensure!(
        planned
            .get(credential.certificate_sha256())
            .map(String::as_str)
            == Some(host),
        "this agent's certificate is not host `{host}`'s planned certificate"
    );
    let (coordinator, coordinator_host) = transport::coordinator_certificate(contract)?;
    TlsAcceptor::bind(
        listen,
        credential,
        BTreeMap::from([(coordinator, coordinator_host)]),
        Timeouts::from_plan(contract),
    )
}

/// The coordinator's supervisor when ranks run on other hosts: each rank's operations go to
/// the agent named for that rank. Handles carry the rank (`remote:<rank>:<supervisor>`).
pub struct RemoteSupervisor {
    contract: BoundedRunContract,
    plan_sha256: String,
    credential: HostCredential,
    agents: BTreeMap<u32, SocketAddr>,
    timeouts: Timeouts,
    channels: BTreeMap<u32, Sequenced<TlsStream>>,
}
impl RemoteSupervisor {
    pub fn new(
        contract: &BoundedRunContract,
        plan_sha256: &str,
        credential: HostCredential,
        agents: BTreeMap<u32, SocketAddr>,
    ) -> Result<Self> {
        ensure!(
            crate::manifest::hash(&contract.canonical_bytes()?) == plan_sha256,
            "plan digest does not name this plan"
        );
        let (coordinator, _) = transport::coordinator_certificate(contract)?;
        ensure!(
            credential.certificate_sha256() == coordinator,
            "the coordinator presents its planned host certificate to agents"
        );
        ensure!(
            agents.keys().copied().eq(0..contract.world_size() as u32),
            "every planned rank needs exactly one agent address"
        );
        Ok(Self {
            contract: contract.clone(),
            plan_sha256: plan_sha256.into(),
            credential,
            agents,
            timeouts: Timeouts::from_plan(contract),
            channels: BTreeMap::new(),
        })
    }
    fn call(&mut self, rank: u32, request: &AgentRequest) -> Result<AgentReply> {
        if !self.channels.contains_key(&rank) {
            let address = *self.agents.get(&rank).context("no agent for rank")?;
            let host = &self.contract.participant(rank).context("rank")?.host;
            let stream = transport::connect(
                address,
                &self.credential,
                (&host.certificate_sha256, &host.host),
                self.timeouts,
            )?;
            self.channels
                .insert(rank, Sequenced::new(stream, AGENT_FRAME_LIMIT));
        }
        let channel = self.channels.get_mut(&rank).context("agent channel")?;
        // Replies are bounded: fast operations by the plan's I/O deadline, a launch or a
        // terminate (which waits for SmartCluster to stop the job) by longer fixed bounds.
        // The bound is absolute over the whole exchange, request and reply: an agent that
        // trickles bytes inside a per-read timeout cannot hold the coordinator any longer.
        let reply_deadline = match request {
            AgentRequest::Terminate { .. } => Duration::from_secs(90),
            AgentRequest::Launch { .. } => Duration::from_secs(30),
            _ => self.timeouts.io,
        };
        {
            use super::rendezvous::ControlStream;
            channel
                .get_ref()
                .set_timeouts(Some(reply_deadline), Some(self.timeouts.io))?;
        }
        let deadline = Instant::now() + reply_deadline;
        let request = serde_json::to_vec(request)?;
        let exchanged = channel
            .send_by(&request, deadline)
            .and_then(|_| channel.receive_by(deadline));
        match exchanged {
            Ok((_, frame)) => match serde_json::from_slice(&frame)? {
                AgentReply::Refused { reason } => {
                    bail!("agent for rank {rank} refused: {reason}")
                }
                reply => Ok(reply),
            },
            Err(e) => {
                // A broken channel is reopened on the next call; this call fails.
                self.channels.remove(&rank);
                Err(e.context(format!("agent for rank {rank}")))
            }
        }
    }
    fn wrap(rank: u32, inner: SupervisorHandle) -> SupervisorHandle {
        SupervisorHandle {
            supervisor: format!("remote:{rank}:{}", inner.supervisor),
            identity: inner.identity,
        }
    }
    fn unwrap(handle: &SupervisorHandle) -> Result<(u32, SupervisorHandle)> {
        let rest = handle
            .supervisor
            .strip_prefix("remote:")
            .context("handle is not a remote handle")?;
        let (rank, supervisor) = rest.split_once(':').context("remote handle rank")?;
        Ok((
            rank.parse().context("remote handle rank")?,
            SupervisorHandle {
                supervisor: supervisor.to_string(),
                identity: handle.identity.clone(),
            },
        ))
    }
}
impl Supervisor for RemoteSupervisor {
    fn launch(&mut self, l: &RankLaunch<'_>) -> Result<SupervisorHandle> {
        ensure!(
            l.plan_sha256 == self.plan_sha256,
            "launch names another plan"
        );
        match self.call(
            l.global_rank,
            &AgentRequest::Launch {
                global_rank: l.global_rank,
                dispatch_id: l.dispatch_id.to_string(),
                dispatch_intent_unix_ms: l.dispatch_intent_unix_ms,
            },
        )? {
            AgentReply::Handle { handle } => Ok(Self::wrap(l.global_rank, handle)),
            _ => bail!("agent launch reply mismatch"),
        }
    }
    fn find(&mut self, dispatch_id: &str) -> Result<Option<SupervisorHandle>> {
        let rank = (0..self.contract.world_size() as u32)
            .find(|r| {
                rendezvous::dispatch_id(&self.plan_sha256, self.contract.attempt, *r)
                    .ok()
                    .as_deref()
                    == Some(dispatch_id)
            })
            .context("dispatch id names no planned rank")?;
        match self.call(
            rank,
            &AgentRequest::Find {
                dispatch_id: dispatch_id.into(),
            },
        )? {
            AgentReply::Found { handle } => Ok(handle.map(|h| Self::wrap(rank, h))),
            _ => bail!("agent find reply mismatch"),
        }
    }
    fn probe(&mut self, handle: &SupervisorHandle) -> Result<Liveness> {
        let (rank, inner) = Self::unwrap(handle)?;
        match self.call(rank, &AgentRequest::Probe { handle: inner })? {
            AgentReply::Liveness { liveness } => Ok(liveness),
            _ => bail!("agent probe reply mismatch"),
        }
    }
    fn terminate(&mut self, handle: &SupervisorHandle) -> Result<Liveness> {
        let (rank, inner) = Self::unwrap(handle)?;
        match self.call(rank, &AgentRequest::Terminate { handle: inner })? {
            AgentReply::Liveness { liveness } => Ok(liveness),
            _ => bail!("agent terminate reply mismatch"),
        }
    }
    fn retained_result(&mut self, handle: &SupervisorHandle) -> Result<Option<(Vec<u8>, Vec<u8>)>> {
        let (rank, inner) = Self::unwrap(handle)?;
        match self.call(rank, &AgentRequest::RetainedResult { handle: inner })? {
            AgentReply::Retained { retained } => retained
                .map(|(e, b)| Ok((hex::decode(e)?, hex::decode(b)?)))
                .transpose(),
            _ => bail!("agent retained-result reply mismatch"),
        }
    }
}
