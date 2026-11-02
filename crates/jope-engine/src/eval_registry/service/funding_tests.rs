//! Actual native socket, SO_PEERCRED, sealed-descriptor and signed-witness
//! integration. Only loaded CPU runtime metadata is observed; no numerical
//! worker or CUDA context is started. Private installation exists only in tests.
use super::*;
use crate::eval_registry::{
    funding_config::InstalledFunding, witness::funding_tests::Fixture as WitnessFixture,
};
use rusqlite::Connection;

struct Fixture {
    original: WitnessFixture,
    installation: InstalledFunding,
    config: ServiceConfig,
    descriptor: Value,
}
struct Running {
    stop: Arc<AtomicBool>,
    state: Arc<State>,
    thread: Option<std::thread::JoinHandle<Result<ServiceStats>>>,
}
impl Running {
    fn finish(mut self) -> ServiceStats {
        self.stop.store(true, Ordering::SeqCst);
        self.thread.take().unwrap().join().unwrap().unwrap()
    }
}
impl Drop for Running {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}
impl Fixture {
    fn new(observed_runtime: bool) -> Self {
        Self::from_witness(WitnessFixture::new(), observed_runtime)
    }
    fn new_resident_training(observed_runtime: bool) -> Self {
        Self::from_witness(WitnessFixture::new_resident_training(), observed_runtime)
    }
    fn from_witness(mut original: WitnessFixture, observed_runtime: bool) -> Self {
        if observed_runtime {
            let actual = crate::runtime_custody::loaded_runtime_descriptor(
                &ProcessIdentity::current().unwrap(),
            )
            .unwrap();
            original.use_observed_runtime(actual["runtime_sha256"].as_str().unwrap());
        }
        let installation = InstalledFunding::for_test(&original.installation).unwrap();
        let installed = installation.revalidate().unwrap();
        let mut config = ServiceConfig::regression(
            original.authority.clone(),
            original.watermark.clone(),
            installed.socket.clone(),
            std::env::current_exe().unwrap(),
            2,
        );
        config.event_witness = Some(original.client.clone());
        config.funding_fixture = Some(installation.clone());
        let descriptor = serde_json::from_slice(&original.descriptor).unwrap();
        Self {
            original,
            installation,
            config,
            descriptor,
        }
    }
    fn launch(&self) -> Running {
        let service = Service::bind(self.config.clone()).unwrap();
        let state = service.state.clone();
        let stop = Arc::new(AtomicBool::new(false));
        let signal = stop.clone();
        Running {
            stop,
            state,
            thread: Some(std::thread::spawn(move || service.run(signal))),
        }
    }
    fn request(&self, request: &FundingRequest) -> Result<Value> {
        funding_ipc::funding_request_installed(&self.installation, request, request_monotonic_ns()?)
    }
    fn start(&self) -> FundingRequest {
        FundingRequest::Start {
            authorization: self.original.authorization.clone(),
            reservation: self.original.reservation.clone(),
            descriptor_json: self.descriptor.clone(),
        }
    }
    fn participant(&self, slot: u32) -> FundingRequest {
        FundingRequest::Participant {
            authorization: self.original.authorization.clone(),
            reservation: self.original.reservation.clone(),
            descriptor_json: self.descriptor.clone(),
            participant_slot: slot,
        }
    }
    fn reserve(&self) {
        self.request(&FundingRequest::Register {
            authorization: self.original.authorization.clone(),
        })
        .unwrap();
        self.request(&FundingRequest::Reserve {
            authorization: self.original.authorization.clone(),
            reservation: self.original.reservation.clone(),
            descriptor_json: self.descriptor.clone(),
        })
        .unwrap();
        let proof = self
            .request(&FundingRequest::Preflight {
                authorization: self.original.authorization.clone(),
                reservation: self.original.reservation.clone(),
                descriptor_json: self.descriptor.clone(),
            })
            .unwrap();
        assert_eq!(proof["numerical_start_released"], false);
    }
    fn raw_response(&self, request: &FundingRequest) -> Value {
        // Deliberately bypass only the client-side validator, never the actual
        // service's peer authentication, sealed FD transport or request parser.
        let deadline = request_monotonic_ns().unwrap() + 30_000_000_000;
        let bytes = serde_json::to_vec(request).unwrap();
        let input = sealed(&bytes).unwrap();
        let mut stream =
            connect_bounded(&self.config.socket, remaining_request(deadline).unwrap()).unwrap();
        send_json(
            &mut stream,
            &Request::Funding {
                descriptor_sha256: sha(&bytes),
                deadline_monotonic_ns: deadline,
            },
            remaining_request(deadline).unwrap(),
        )
        .unwrap();
        let sent = send_fd(
            &stream,
            input.as_raw_fd(),
            remaining_request(deadline).unwrap(),
        );
        let response: Value =
            receive_json(&mut stream, remaining_request(deadline).unwrap()).unwrap();
        // A rejection before descriptor admission may close its receiving end
        // before sendmsg runs. Preserve that original negative response.
        assert!(sent.is_ok() || response["ok"] == false);
        response
    }
    fn count(&self, table: &str) -> u64 {
        assert!([
            "funding_authorizations",
            "funding_reservations",
            "funding_participants",
            "funding_witness"
        ]
        .contains(&table));
        Connection::open(self.original.authority.join("registry.sqlite"))
            .unwrap()
            .query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |r| r.get(0))
            .unwrap()
    }
}

include!("funding_tests/socket.rs");
include!("funding_tests/controller.rs");
include!("funding_tests/grants.rs");
