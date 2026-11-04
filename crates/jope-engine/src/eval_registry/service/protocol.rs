#[derive(Serialize, Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case", deny_unknown_fields)]
enum Request {
    SubmitNative {
        request: super::native_scq::NativeRequest,
    },
    Funding {
        descriptor_sha256: String,
        deadline_monotonic_ns: u64,
    },
    Submit {
        provenance: RequestProvenance,
        descriptor_sha256: String,
    },
    Consume {
        ticket: String,
        fit_key: String,
    },
    Verify {
        completion_sha256: String,
        scope: String,
    },
    QueryEvaluate {
        descriptor_sha256: String,
    },
    QueryRetrieve {
        descriptor_sha256: String,
        deadline_monotonic_ns: u64,
    },
    Stats,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ServiceIdentity {
    pub(super) schema: String,
    pub(super) process: ProcessIdentity,
    pub(super) executable_sha256: String,
    pub(super) authority_instance: String,
    pub(super) authority_root: PathBuf,
    pub(super) witness: PathBuf,
    pub(super) manifest_sha256: String,
    pub(super) socket_device: u64,
    pub(super) socket_inode: u64,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WorkerLaunch {
    pub(super) service: ServiceIdentity,
    pub(super) contract: NumericalContract,
    pub(super) fit_seed: u64,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Permission {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) native_placement: Option<jain_smartcluster::TrustedFitChildPlacement>,
    pub(super) start: StartToken,
    pub(super) custody_device: u64,
    pub(super) custody_inode: u64,
    pub(super) custody_path: PathBuf,
    pub(super) authority_instance: String,
}
