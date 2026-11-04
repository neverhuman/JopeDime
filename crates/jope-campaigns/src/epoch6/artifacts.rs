use super::*;

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactKind {
    Evaluator,
    Checkpoint,
    Panel,
    FrozenCatalog,
    RuntimeHelper,
    ModelBundle,
}

#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactRefV1 {
    pub schema: String,
    pub kind: ArtifactKind,
    pub object_sha256: String,
    pub object_manifest_sha256: String,
    pub protocol_sha256: String,
    pub byte_length: u64,
}

impl ArtifactRefV1 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == ARTIFACT_REF_SCHEMA,
            "invalid artifact-ref schema"
        );
        sha256(&self.object_sha256, "artifact object")?;
        sha256(&self.object_manifest_sha256, "artifact object manifest")?;
        sha256(&self.protocol_sha256, "artifact protocol")?;
        ensure!(self.byte_length > 0, "artifact is empty");
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct ArtifactObjectManifestV1 {
    schema: String,
    kind: ArtifactKind,
    object_sha256: String,
    protocol_sha256: String,
    byte_length: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluatorRefV2 {
    pub schema: String,
    pub artifact: ArtifactRefV1,
    pub build_identity_sha256: String,
    pub evaluator_protocol_sha256: String,
}

impl EvaluatorRefV2 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == EVALUATOR_REF_SCHEMA,
            "invalid evaluator-ref schema"
        );
        self.artifact.validate()?;
        ensure!(
            self.artifact.kind == ArtifactKind::Evaluator,
            "evaluator ref names a non-evaluator artifact"
        );
        sha256(&self.build_identity_sha256, "evaluator build identity")?;
        sha256(&self.evaluator_protocol_sha256, "evaluator protocol")?;
        ensure!(
            self.artifact.protocol_sha256 == self.evaluator_protocol_sha256,
            "evaluator protocol/ref mismatch"
        );
        Ok(())
    }
}

pub struct ArtifactCas {
    root: PathBuf,
}

impl ArtifactCas {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn publish_bytes(
        &self,
        kind: ArtifactKind,
        protocol_sha256: &str,
        bytes: &[u8],
    ) -> Result<ArtifactRefV1> {
        sha256(protocol_sha256, "artifact protocol")?;
        ensure!(!bytes.is_empty(), "refusing to publish an empty artifact");
        let object_sha256 = hex::encode(Sha256::digest(bytes));
        let manifest = ArtifactObjectManifestV1 {
            schema: ARTIFACT_MANIFEST_SCHEMA.to_owned(),
            kind,
            object_sha256: object_sha256.clone(),
            protocol_sha256: protocol_sha256.to_owned(),
            byte_length: bytes.len() as u64,
        };
        let manifest_bytes = serde_json::to_vec(&manifest)?;
        let manifest_sha256 = hex::encode(Sha256::digest(&manifest_bytes));
        publish_cas_file(&self.object_path(&object_sha256), bytes)?;
        publish_cas_file(&self.manifest_path(&manifest_sha256), &manifest_bytes)?;
        let reference = ArtifactRefV1 {
            schema: ARTIFACT_REF_SCHEMA.to_owned(),
            kind,
            object_sha256,
            object_manifest_sha256: manifest_sha256,
            protocol_sha256: protocol_sha256.to_owned(),
            byte_length: bytes.len() as u64,
        };
        self.verify(&reference)?;
        Ok(reference)
    }

    pub fn publish_file(
        &self,
        kind: ArtifactKind,
        protocol_sha256: &str,
        path: &Path,
    ) -> Result<ArtifactRefV1> {
        ensure!(path.is_file(), "artifact source is not a regular file");
        self.publish_bytes(kind, protocol_sha256, &fs::read(path)?)
    }

    pub fn verify(&self, reference: &ArtifactRefV1) -> Result<()> {
        reference.validate()?;
        let object = fs::read(self.object_path(&reference.object_sha256))
            .context("read artifact CAS object")?;
        ensure!(
            object.len() as u64 == reference.byte_length,
            "artifact CAS object length drift"
        );
        ensure!(
            hex::encode(Sha256::digest(&object)) == reference.object_sha256,
            "artifact CAS object digest drift"
        );
        let manifest_bytes = fs::read(self.manifest_path(&reference.object_manifest_sha256))
            .context("read artifact CAS manifest")?;
        ensure!(
            hex::encode(Sha256::digest(&manifest_bytes)) == reference.object_manifest_sha256,
            "artifact CAS manifest digest drift"
        );
        let manifest: ArtifactObjectManifestV1 = serde_json::from_slice(&manifest_bytes)?;
        ensure!(
            manifest.schema == ARTIFACT_MANIFEST_SCHEMA,
            "artifact CAS manifest schema drift"
        );
        ensure!(
            manifest.kind == reference.kind
                && manifest.object_sha256 == reference.object_sha256
                && manifest.protocol_sha256 == reference.protocol_sha256
                && manifest.byte_length == reference.byte_length,
            "artifact CAS manifest/reference mismatch"
        );
        Ok(())
    }

    pub fn materialize(&self, reference: &ArtifactRefV1, destination: &Path) -> Result<()> {
        self.verify(reference)?;
        validate_materialization_path(destination)?;
        ensure!(
            !destination.exists(),
            "artifact materialization destination exists"
        );
        let parent = destination
            .parent()
            .context("artifact destination has no parent")?;
        fs::create_dir_all(parent)?;
        let mut input = File::open(self.object_path(&reference.object_sha256))?;
        let mut output = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o444)
            .open(destination)?;
        std::io::copy(&mut input, &mut output)?;
        output.sync_all()?;
        fs::set_permissions(destination, fs::Permissions::from_mode(0o444))?;
        sync_directory(parent)
    }

    fn object_path(&self, digest: &str) -> PathBuf {
        self.root
            .join("objects/sha256")
            .join(&digest[..2])
            .join(digest)
    }

    fn manifest_path(&self, digest: &str) -> PathBuf {
        self.root
            .join("manifests/sha256")
            .join(&digest[..2])
            .join(format!("{digest}.json"))
    }
}

fn validate_materialization_path(path: &Path) -> Result<()> {
    ensure!(
        path.is_absolute(),
        "artifact materialization destination is not absolute"
    );
    ensure!(
        path.components()
            .all(|part| !matches!(part, Component::ParentDir | Component::CurDir)),
        "artifact materialization destination is not normalized"
    );
    Ok(())
}

fn publish_cas_file(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path.parent().context("CAS path has no parent")?;
    fs::create_dir_all(parent)?;
    match OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o444)
        .open(path)
    {
        Ok(mut file) => {
            file.write_all(bytes)?;
            file.sync_all()?;
            fs::set_permissions(path, fs::Permissions::from_mode(0o444))?;
            sync_directory(parent)?;
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            let mut existing = Vec::new();
            File::open(path)?.read_to_end(&mut existing)?;
            ensure!(existing == bytes, "content-addressed artifact conflict");
        }
        Err(error) => return Err(error.into()),
    }
    Ok(())
}

fn sync_directory(path: &Path) -> Result<()> {
    File::open(path)?.sync_all()?;
    Ok(())
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactGcRootClass {
    Queues,
    Observations,
    Repairs,
    Canaries,
    Promotions,
    BenchmarkPlans,
    LiftManifests,
    Quarantine,
    RollbackEvidence,
    ModelBundles,
}

impl ArtifactGcRootClass {
    pub const ALL: [Self; 10] = [
        Self::Queues,
        Self::Observations,
        Self::Repairs,
        Self::Canaries,
        Self::Promotions,
        Self::BenchmarkPlans,
        Self::LiftManifests,
        Self::Quarantine,
        Self::RollbackEvidence,
        Self::ModelBundles,
    ];
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactGcRootV1 {
    pub class: ArtifactGcRootClass,
    pub readable: bool,
    pub artifacts: Vec<ArtifactRefV1>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactGcRootsV1 {
    pub schema: String,
    pub roots: Vec<ArtifactGcRootV1>,
}

impl ArtifactGcRootsV1 {
    pub fn validate_for_sweep(&self) -> Result<BTreeSet<String>> {
        ensure!(
            self.schema == ARTIFACT_GC_ROOTS_SCHEMA,
            "invalid artifact GC-root schema"
        );
        ensure!(
            self.roots.len() == ArtifactGcRootClass::ALL.len(),
            "artifact GC root classes are incomplete"
        );
        let classes = self.roots.iter().map(|root| root.class).collect::<Vec<_>>();
        ensure!(
            classes == ArtifactGcRootClass::ALL,
            "artifact GC roots are absent, duplicated, or noncanonical"
        );
        let mut reachable = BTreeSet::new();
        for root in &self.roots {
            ensure!(
                root.readable,
                "artifact GC root {:?} is unreadable",
                root.class
            );
            for artifact in &root.artifacts {
                artifact.validate()?;
                reachable.insert(artifact.object_sha256.clone());
            }
        }
        Ok(reachable)
    }
}
