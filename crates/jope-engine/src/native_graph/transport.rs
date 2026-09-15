//! Root-registered, same-connection authenticated native graph RPCs.
use super::*;
use crate::eval_registry::native_scq::{self, policy};
use crate::native_loop::{RunApprovalBody, SignedRunApproval};
use jain_smartcluster::{
    AttemptInspection, GetRequest, InspectAttemptRequest, JobRecord, OperationAck, Request,
    Response, SendControlRequest, SubmitRequest,
};
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Registration {
    schema: String,
    approval: Artifact,
}
#[derive(Clone)]
pub(super) struct Transport {
    body: RunApprovalBody,
}
impl Transport {
    pub(super) fn load(graph: &Graph, client: &Client) -> Result<Self> {
        let key = hash(&serde_json::to_vec(&(&graph.root, &graph.directory))?);
        let path =
            PathBuf::from("/etc/jope-dime/native-graph-registrations").join(format!("{key}.json"));
        let registration: Registration = serde_json::from_slice(&policy::protected(&path)?)?;
        ensure!(
            registration.schema == "jope.native_graph_registration.v1"
                && registration.approval.path.is_absolute(),
            "graph requires an independently installed exact registration"
        );
        let bytes = policy::protected(&registration.approval.path)?;
        ensure!(
            hash(&bytes) == registration.approval.sha256,
            "registered graph approval changed"
        );
        let signed: SignedRunApproval = serde_json::from_slice(&bytes)?;
        crate::native_loop::verify_installed_run_signature(
            &signed,
            crate::resident_protocol::now_ms()?,
        )?;
        let b = signed.body;
        ensure!(
            b.root == graph.root
                && b.graph_state == graph.directory
                && b.worker_state == graph.worker_state
                && client.socket_path() == b.daemon_socket,
            "graph storage or actual client socket differs from approval"
        );
        let census: crate::native_loop::OriginalJobCensus =
            crate::native_genome::load(&graph.root, &b.original_job_census)?;
        ensure!(
            census.tenant == graph.tenant && census.project == graph.project,
            "graph tenant/project differs from approval"
        );
        let config: crate::native_loop::Config =
            crate::native_genome::load(&graph.root, &b.config)?;
        ensure!(
            config.run_approval == registration.approval.path,
            "graph registration is not the original config approval"
        );
        Ok(Self { body: b })
    }
    fn call(&self, request: Request) -> Result<Response> {
        ensure!(
            crate::resident_protocol::now_ms()? < self.body.expires_unix_ms,
            "original graph approval expired"
        );
        // authenticated_exchange authenticates this exact UnixStream and holds
        // the peer executable FD while checking PID birth; no precheck race.
        native_scq::authenticated_exchange(
            &self.body.daemon_socket,
            self.body.daemon_uid,
            &self.body.daemon_executable_sha256,
            None,
            request,
        )
    }
    pub(super) async fn get(&self, job_id: JobId) -> Result<Option<JobRecord>> {
        match self.call(Request::Get(GetRequest { job_id }))? {
            Response::Job(j) => Ok(j),
            _ => anyhow::bail!("authenticated native get response mismatch"),
        }
    }
    pub(super) async fn inspect_attempt(
        &self,
        attempt: FencedAttempt,
    ) -> Result<Option<AttemptInspection>> {
        match self.call(Request::InspectAttempt(InspectAttemptRequest { attempt }))? {
            Response::AttemptInspected(j) => Ok(j),
            _ => anyhow::bail!("authenticated native attempt response mismatch"),
        }
    }
    pub(super) async fn submit(&self, spec: JobSpec) -> Result<JobRecord> {
        match self.call(Request::Submit(Box::new(SubmitRequest {
            spec,
            effective_principal: None,
        })))? {
            Response::Submitted(j) => Ok(j),
            _ => anyhow::bail!("authenticated native submit response mismatch"),
        }
    }
    pub(super) async fn dispatch_subtask(
        &self,
        job_id: JobId,
        intent: &Path,
        command: &SubtaskCommand,
    ) -> Result<OperationAck> {
        let payload = serde_json::to_vec(command)?;
        ensure!(payload.len() <= 64 * 1024, "subtask command size");
        crate::smartcluster_worker::freeze_dispatch_intent(intent, command)?;
        match self.call(Request::SendControl(SendControlRequest {
            job_id,
            control: jain_smartcluster::ControlMessage::Application {
                name: crate::smartcluster_worker::CONTROL_NAME.to_owned(),
                payload,
            },
        }))? {
            Response::ControlAccepted(ack) => Ok(ack),
            _ => anyhow::bail!("authenticated native control response mismatch"),
        }
    }
    pub(super) fn approval(&self) -> &RunApprovalBody {
        &self.body
    }
    pub(super) fn verify_dispatch_image(&self) -> Result<()> {
        ensure!(
            crate::manifest::file_hash(Path::new("/proc/self/exe"))?
                == self.body.loop_executable_sha256,
            "only independently approved Loop image may dispatch managed graph work"
        );
        Ok(())
    }
}
