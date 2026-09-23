use super::*;
use serde_json::Value;

pub const VERIFICATION_POLICY: &str = "jope.cpu-context-verification/v1:8-native-meta-train-cases:2-families:cuts0,1,2,3:all-component-and-context-maxabs<=0.0003:finite-f32:accepted-transforms:exact-source-program-assets:nine-controller-verified-execution-pins:threads1:each-job30s:original-proof-window:native-index6c81da15d75dff6abeaf1d40fb46ec77e98d6146f535392905bfaf1153574b9c:no-held-labels";
pub fn verification_policy_sha256() -> String {
    sha(VERIFICATION_POLICY.as_bytes())
}
pub const REFERENCE_REGISTRY_SHA256: &str =
    "22b5de5f2f847fb6e086d125d6fec945a1b4edb7c00b94b49e1ad95e7540cfc4";
pub const REFERENCE_INDEX_SHA256: &str =
    "6c81da15d75dff6abeaf1d40fb46ec77e98d6146f535392905bfaf1153574b9c";
const MAX_ABS: f32 = 3e-4;

/// Manifest must be selected and independently pinned before qualification runs.
/// Native bytes remain private evidence, never rewritten to imitate a new backend.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReferenceManifest {
    pub schema: String,
    pub verification_policy_sha256: String,
    pub registry_bytes: Vec<u8>,
    pub index_bytes: Vec<u8>,
    pub cases: Vec<ReferenceCase>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReferenceCase {
    pub id: String,
    /// Exact native, train-only preprocessed numeric rows supplied to the child.
    /// Features followed by the regression target. No source/corpus path.
    pub prepared_rows: Vec<Vec<f64>>,
    pub prepared_rows_sha256: String,
    pub context_receipt: Vec<u8>,
    pub context_vector: Vec<u8>,
    pub components: BTreeMap<String, NativeComponent>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeComponent {
    pub receipt: Vec<u8>,
    pub vector: Vec<u8>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProducedCase {
    pub id: String,
    pub prepared_rows_sha256: String,
    pub components: Components,
    pub vector: Vec<f32>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Qualification {
    pub schema: String,
    pub source_sha256: String,
    pub producer_sha256: String,
    pub verification_policy_sha256: String,
    pub reference_manifest_sha256: String,
    pub encoder: EncoderIdentity,
    /// Exact immutable bytes from eight independently bounded case workers.
    pub case_outputs: Vec<Vec<u8>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QualificationCaseResult {
    pub schema: String,
    pub source_sha256: String,
    pub producer_sha256: String,
    pub reference_case_sha256: String,
    pub encoder: EncoderIdentity,
    pub produced: ProducedCase,
}

/// A *normalized attestation* made only by the controller after validating the
/// actual ExecutionPool receipt (program, input/output bytes, cgroup evidence,
/// completion, limits and cancellation). This type cannot establish process
/// execution by itself. Its digest must come through a separate trusted pin;
/// workers must never choose their own attestation/pins.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionAttestation {
    pub schema: String,
    pub supervisor_receipt_sha256: String,
    pub program_sha256: String,
    pub input_sha256: String,
    pub output_sha256: String,
    pub completed: bool,
    pub cpu_limit: u32,
    pub memory_max: u64,
    pub swap_max: u64,
    pub threads: u32,
    pub elapsed_seconds: f64,
    pub deadline_seconds: f64,
}
/// These pins are controller inputs, not fields trusted from artifact JSON.
/// Before calling this verifier the controller/child must validate the original
/// bounded execution receipts and derive all nine normalized attestation hashes.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdmissionPins {
    pub source_sha256: String,
    pub producer_sha256: String,
    pub artifact_sha256: String,
    pub qualification_sha256: String,
    pub reference_manifest_sha256: String,
    pub context_execution_sha256: String,
    /// One controller-derived attestation pin for every case, in manifest order.
    pub qualification_execution_sha256: Vec<String>,
    pub verification_policy_sha256: String,
    pub training: TrainingIdentity,
    pub input_sha256: String,
    pub preprocessing_sha256: String,
}
pub struct VerifiedCpuContext {
    context: Vec<f32>,
    training: TrainingIdentity,
    input_sha256: String,
    preprocessing_sha256: String,
    original_labels_sha256: String,
    artifact_sha256: String,
    qualification_sha256: String,
}
impl VerifiedCpuContext {
    pub fn context(&self) -> &[f32] {
        &self.context
    }
    pub fn training(&self) -> &TrainingIdentity {
        &self.training
    }
    pub fn input_sha256(&self) -> &str {
        &self.input_sha256
    }
    pub fn preprocessing_sha256(&self) -> &str {
        &self.preprocessing_sha256
    }
    pub fn original_labels_sha256(&self) -> &str {
        &self.original_labels_sha256
    }
    pub fn artifact_sha256(&self) -> &str {
        &self.artifact_sha256
    }
    pub fn qualification_sha256(&self) -> &str {
        &self.qualification_sha256
    }
}

impl CpuProducer {
    /// One case per bounded job (30s, one numerical thread). The caller gives
    /// this worker only one canonical ReferenceCase payload from case_inputs.
    pub fn qualify_case(
        &self,
        case_bytes: &[u8],
        expected_case_sha256: &str,
        source_sha256: &str,
        producer_sha256: &str,
    ) -> Result<QualificationCaseResult> {
        ensure!(
            case_bytes.len() <= 128 << 20 && sha(case_bytes) == expected_case_sha256,
            "qualification case differs from controller input pin"
        );
        validate_hex(source_sha256, 40)?;
        validate_hex(producer_sha256, 64)?;
        let reference: ReferenceCase = serde_json::from_slice(case_bytes)?;
        validate_rows(&reference.prepared_rows)?;
        ensure!(
            reference.prepared_rows_sha256 == sha(&serde_json::to_vec(&reference.prepared_rows)?),
            "qualification case numerical rows changed"
        );
        let components = self.encode_prepared(&reference.prepared_rows)?;
        let vector = self.assemble(&components)?;
        let produced = ProducedCase {
            id: reference.id.clone(),
            prepared_rows_sha256: reference.prepared_rows_sha256.clone(),
            components,
            vector,
        };
        compare(&reference, &produced)?;
        Ok(QualificationCaseResult {
            schema: "jope.cpu-context-case/v1".into(),
            source_sha256: source_sha256.into(),
            producer_sha256: producer_sha256.into(),
            reference_case_sha256: expected_case_sha256.into(),
            encoder: self.identity.clone(),
            produced,
        })
    }
}

/// Validate all reference provenance before scheduling, then freeze these exact
/// case bytes and their SHA as the eight independent worker input artifacts.
/// Preserve the existing proof window across retries; this grants no new clock.
pub fn case_inputs(manifest_bytes: &[u8], expected_manifest_sha256: &str) -> Result<Vec<Vec<u8>>> {
    ensure!(
        sha(manifest_bytes) == expected_manifest_sha256,
        "reference manifest differs from pre-registered pin"
    );
    read_manifest(manifest_bytes)?
        .cases
        .iter()
        .map(|case| Ok(serde_json::to_vec(case)?))
        .collect()
}

/// Parent-only arithmetic validation/aggregation after all eight worker outputs
/// are immutable. Never executes an encoder or replaces timed-out/missing cases.
/// Execution attestations are separately checked by verify_qualified.
pub fn aggregate_qualification(
    manifest_bytes: &[u8],
    expected_manifest_sha256: &str,
    case_outputs: Vec<Vec<u8>>,
    source_sha256: &str,
    producer_sha256: &str,
) -> Result<Qualification> {
    ensure!(
        sha(manifest_bytes) == expected_manifest_sha256,
        "reference manifest differs from pre-registered pin"
    );
    validate_hex(source_sha256, 40)?;
    validate_hex(producer_sha256, 64)?;
    let manifest = read_manifest(manifest_bytes)?;
    ensure!(
        case_outputs.len() == manifest.cases.len(),
        "missing qualification case outcomes"
    );
    let first: QualificationCaseResult =
        serde_json::from_slice(case_outputs.first().context("no case outcomes")?)?;
    let result = Qualification {
        schema: "jope.cpu-context-qualification/v1".into(),
        source_sha256: source_sha256.into(),
        producer_sha256: producer_sha256.into(),
        verification_policy_sha256: verification_policy_sha256(),
        reference_manifest_sha256: expected_manifest_sha256.into(),
        encoder: first.encoder,
        case_outputs,
    };
    verify_case_outputs(&manifest, &result)?;
    Ok(result)
}

fn verify_case_outputs(manifest: &ReferenceManifest, qualification: &Qualification) -> Result<()> {
    ensure!(
        qualification.case_outputs.len() == manifest.cases.len(),
        "incomplete qualification cases"
    );
    for (reference, bytes) in manifest.cases.iter().zip(&qualification.case_outputs) {
        ensure!(
            bytes.len() <= 2 << 20,
            "qualification case output too large"
        );
        let result: QualificationCaseResult = serde_json::from_slice(bytes)?;
        ensure!(
            result.schema == "jope.cpu-context-case/v1"
                && result.source_sha256 == qualification.source_sha256
                && result.producer_sha256 == qualification.producer_sha256
                && result.encoder == qualification.encoder,
            "qualification case source/program/assets differ"
        );
        ensure!(
            result.reference_case_sha256 == sha(&serde_json::to_vec(reference)?),
            "qualification case input differs from frozen reference manifest"
        );
        compare(reference, &result.produced)?;
    }
    Ok(())
}

/// Recompute all hashes, native receipt chains and numerical comparisons. No
/// `passed:true` field is accepted. Execution attestations must have independent,
/// controller-derived pins; supplying self-attested pins is not qualification.
pub fn verify_qualified(
    artifact_bytes: &[u8],
    qualification_bytes: &[u8],
    manifest_bytes: &[u8],
    context_execution_bytes: &[u8],
    qualification_execution_bytes: &[Vec<u8>],
    pins: &AdmissionPins,
) -> Result<VerifiedCpuContext> {
    ensure!(
        pins.verification_policy_sha256 == verification_policy_sha256(),
        "verification policy differs"
    );
    for (bytes, pin, maximum) in [
        (artifact_bytes, &pins.artifact_sha256, 16 << 20),
        (qualification_bytes, &pins.qualification_sha256, 16 << 20),
        (manifest_bytes, &pins.reference_manifest_sha256, 128 << 20),
        (
            context_execution_bytes,
            &pins.context_execution_sha256,
            1 << 20,
        ),
    ] {
        ensure!(
            bytes.len() <= maximum && sha(bytes) == *pin,
            "context admission bytes differ from independent pin"
        );
    }
    let artifact: CpuContextArtifact = serde_json::from_slice(artifact_bytes)?;
    artifact.validate()?;
    let qualification: Qualification = serde_json::from_slice(qualification_bytes)?;
    ensure!(
        qualification.schema == "jope.cpu-context-qualification/v1"
            && qualification.verification_policy_sha256 == pins.verification_policy_sha256
            && qualification.reference_manifest_sha256 == pins.reference_manifest_sha256,
        "qualification protocol changed"
    );
    ensure!(
        qualification.source_sha256 == pins.source_sha256
            && artifact.source_sha256 == pins.source_sha256
            && qualification.producer_sha256 == pins.producer_sha256
            && artifact.producer_sha256 == pins.producer_sha256,
        "qualification source/program identity differs"
    );
    ensure!(
        artifact.training == pins.training
            && artifact.input_sha256 == pins.input_sha256
            && artifact.preprocessing_sha256 == pins.preprocessing_sha256,
        "context training input differs from controller cut"
    );
    ensure!(
        artifact.encoder == qualification.encoder,
        "qualified encoder assets/recipe differ from producer"
    );
    verify_execution(
        context_execution_bytes,
        &pins.producer_sha256,
        &pins.input_sha256,
        &pins.artifact_sha256,
        30.0,
    )?;
    let manifest = read_manifest(manifest_bytes)?;
    verify_case_outputs(&manifest, &qualification)?;
    ensure!(
        qualification_execution_bytes.len() == 8 && pins.qualification_execution_sha256.len() == 8,
        "missing bounded qualification execution evidence"
    );
    for (index, reference) in manifest.cases.iter().enumerate() {
        let evidence = &qualification_execution_bytes[index];
        ensure!(
            evidence.len() <= 1 << 20
                && sha(evidence) == pins.qualification_execution_sha256[index],
            "qualification execution differs from independent controller pin"
        );
        verify_execution(
            evidence,
            &pins.producer_sha256,
            &sha(&serde_json::to_vec(reference)?),
            &sha(&qualification.case_outputs[index]),
            30.0,
        )?;
    }
    Ok(VerifiedCpuContext {
        context: artifact.vector,
        training: artifact.training,
        input_sha256: artifact.input_sha256,
        preprocessing_sha256: artifact.preprocessing_sha256,
        original_labels_sha256: artifact.original_labels_sha256,
        artifact_sha256: pins.artifact_sha256.clone(),
        qualification_sha256: pins.qualification_sha256.clone(),
    })
}
fn verify_execution(
    bytes: &[u8],
    program: &str,
    input: &str,
    output: &str,
    maximum_seconds: f64,
) -> Result<()> {
    ensure!(
        maximum_seconds.is_finite() && maximum_seconds > 0.0 && maximum_seconds <= 30.0,
        "invalid predeclared encoding qualification allocation"
    );
    let e: ExecutionAttestation = serde_json::from_slice(bytes)?;
    validate_hex(&e.supervisor_receipt_sha256, 64)?;
    ensure!(
        e.schema == "jope.cpu-context-execution/controller-v1"
            && e.completed
            && e.program_sha256 == program
            && e.input_sha256 == input
            && e.output_sha256 == output,
        "producer execution binding differs"
    );
    ensure!(
        (1..=20).contains(&e.cpu_limit)
            && e.threads == 1
            && e.memory_max > 0
            && e.memory_max <= 64u64 << 30
            && e.swap_max == 0,
        "producer execution exceeded resource envelope"
    );
    ensure!(
        e.elapsed_seconds.is_finite()
            && e.elapsed_seconds >= 0.0
            && e.deadline_seconds.is_finite()
            && e.deadline_seconds > 0.0
            && e.deadline_seconds <= maximum_seconds
            && e.elapsed_seconds <= e.deadline_seconds,
        "producer execution failed deadline"
    );
    Ok(())
}
fn read_manifest(bytes: &[u8]) -> Result<ReferenceManifest> {
    ensure!(bytes.len() <= 128 << 20, "qualification manifest too large");
    let m: ReferenceManifest = serde_json::from_slice(bytes)?;
    ensure!(
        m.schema == "jope.cpu-context-reference/v1"
            && m.verification_policy_sha256 == verification_policy_sha256()
            && m.cases.len() == 8,
        "qualification manifest protocol changed"
    );
    ensure!(
        sha(&m.registry_bytes) == REFERENCE_REGISTRY_SHA256,
        "reference family registry identity changed"
    );
    let registry: Value = serde_json::from_slice(&m.registry_bytes)?;
    ensure!(
        sha(&m.index_bytes) == REFERENCE_INDEX_SHA256,
        "native reference index changed"
    );
    let index: Value = serde_json::from_slice(&m.index_bytes)?;
    ensure!(
        index["schema"] == "parallel.context_index.v1"
            && index["assembly_sha256"] == ASSEMBLY_SHA256
            && index["registry"]["sha256"] == REFERENCE_REGISTRY_SHA256,
        "native reference index lineage changed"
    );

    let mut families: BTreeMap<String, BTreeSet<u64>> = BTreeMap::new();
    let mut ids = BTreeSet::new();
    for case in &m.cases {
        ensure!(ids.insert(&case.id), "duplicate reference case");
        validate_rows(&case.prepared_rows)?;
        ensure!(
            sha(&serde_json::to_vec(&case.prepared_rows)?) == case.prepared_rows_sha256,
            "reference training rows changed"
        );
        let receipt: Value = serde_json::from_slice(&case.context_receipt)?;
        // Native training_sha256 is the headerless csv::Writer representation.
        // Finite f64 Display contains no CSV quoting characters.
        let native_csv = case
            .prepared_rows
            .iter()
            .map(|row| {
                let mut line = row
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(",");
                line.push('\n');
                line
            })
            .collect::<String>();
        ensure!(
            receipt["training_sha256"] == sha(native_csv.as_bytes()),
            "qualification numerical rows differ from retained native training cut"
        );
        let family = receipt["family"]
            .as_str()
            .context("reference family absent")?;
        let dataset = receipt["dataset"]
            .as_str()
            .context("reference dataset absent")?;
        let cut = receipt["cut"].as_u64().context("reference cut absent")?;
        let indexed = index["encodings"]
            .as_array()
            .context("native index entries absent")?
            .iter()
            .filter(|e| {
                e["dataset"] == dataset
                    && e["cut"] == cut
                    && e["prepared_sha256"] == receipt["prepared_sha256"]
            })
            .collect::<Vec<_>>();
        ensure!(
            indexed.len() == 1
                && indexed[0]["family"] == family
                && indexed[0]["partition"] == "meta_train"
                && indexed[0]["receipt_sha256"] == sha(&case.context_receipt)
                && indexed[0]["sha256"] == sha(&case.context_vector),
            "reference is not the retained native indexed receipt/vector"
        );

        ensure!(
            cut < 4
                && receipt["partition"] == "meta_train"
                && registry["families"][family]["partition"] == "meta_train",
            "reference is not a selected native training cut"
        );
        ensure!(
            case.id == format!("{dataset}/cut-{cut}"),
            "reference case identity differs"
        );
        let datasets = registry["families"][family]["datasets"]
            .as_array()
            .context("registry family datasets absent")?;
        ensure!(
            datasets.iter().any(|d| d["dataset"] == dataset
                && d["preparation_eligible"] == true
                && d["prepared"].as_array().is_some_and(|ps| ps
                    .iter()
                    .any(|p| p["sha256"] == receipt["prepared_sha256"]))),
            "reference prepared input not in training registry"
        );
        ensure!(
            registry["held_review_datasets"].get(dataset).is_none(),
            "reference dataset held by review"
        );
        ensure!(
            families.entry(family.into()).or_default().insert(cut),
            "duplicate family/cut reference"
        );
        ensure!(
            receipt["schema"] == "parallel.encoding.v1"
                && receipt["encoder_spec_sha256"] == ASSEMBLY_SHA256
                && receipt["dataset_dimensions"] == 1536
                && receipt["target_dimensions"] == 512,
            "native context lineage differs"
        );
        ensure!(
            receipt["output_sha256"] == sha(&case.context_vector),
            "native context vector bytes changed"
        );
        decode_f32(&case.context_vector, 2048)?;
        ensure!(
            case.components.len() == 3,
            "reference component inventory differs"
        );
        for (name, length) in [("hyperion", 512), ("foundation", 2048), ("target", 512)] {
            let component = case
                .components
                .get(name)
                .context("reference component absent")?;
            let r: Value = serde_json::from_slice(&component.receipt)?;
            let bound = &receipt["components"][name];
            ensure!(
                bound["receipt_sha256"] == sha(&component.receipt)
                    && bound["signature_sha256"] == sha(&component.vector),
                "native component receipt/vector chain differs"
            );
            ensure!(
                r["signature_sha256"] == sha(&component.vector)
                    && r["spec_sha256"] == bound["spec_sha256"]
                    && r["final_test_targets_used"] == false,
                "native component input closure differs"
            );
            for field in [
                "dataset",
                "family",
                "cut",
                "prepared_sha256",
                "training_sha256",
                "preprocessing_sha256",
            ] {
                ensure!(
                    r[field] == receipt[field] && !r[field].is_null(),
                    "native component training identity differs: {field}"
                );
            }
            decode_f32(&component.vector, length)?;
        }
    }
    ensure!(
        families.len() == 2 && families.values().all(|cuts| cuts.iter().copied().eq(0..4)),
        "qualification requires 2 training families and exactly cuts0..3"
    );
    Ok(m)
}
fn compare(reference: &ReferenceCase, produced: &ProducedCase) -> Result<()> {
    ensure!(
        reference.id == produced.id
            && reference.prepared_rows_sha256 == produced.prepared_rows_sha256,
        "qualification case/input identity differs"
    );
    produced.components.validate()?;
    for (name, actual, length) in [
        ("hyperion", &produced.components.hyperion, 512),
        ("foundation", &produced.components.foundation, 2048),
        ("target", &produced.components.target, 512),
    ] {
        let expected = decode_f32(
            &reference
                .components
                .get(name)
                .context("missing native component")?
                .vector,
            length,
        )?;
        max_abs(actual, &expected)?;
    }
    max_abs(
        &produced.vector,
        &decode_f32(&reference.context_vector, 2048)?,
    )?;
    Ok(())
}
fn decode_f32(bytes: &[u8], length: usize) -> Result<Vec<f32>> {
    ensure!(bytes.len() == length * 4, "native FP32 vector size differs");
    let values = bytes
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().expect("four-byte chunk")))
        .collect::<Vec<_>>();
    vector(&values, length)?;
    Ok(values)
}
fn max_abs(actual: &[f32], expected: &[f32]) -> Result<f32> {
    vector(actual, expected.len())?;
    vector(expected, actual.len())?;
    let error = actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    ensure!(
        error.is_finite() && error <= MAX_ABS,
        "CPU/native parity failed: max_abs={error}, tolerance={MAX_ABS}"
    );
    Ok(error)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn finite_component_gate_refuses_wrong_values_shapes_and_nan() {
        assert!(max_abs(&[0.0, 1.0], &[0.0, 1.0]).is_ok());
        assert!(max_abs(&[0.0], &[0.0, 1.0]).is_err());
        assert!(max_abs(&[f32::NAN], &[0.0]).is_err());
        assert!(max_abs(&[0.001], &[0.0]).is_err());
    }
    #[test]
    fn cpu_manifest_never_accepts_unpinned_registry_or_fewer_cases() {
        let m = ReferenceManifest {
            schema: "jope.cpu-context-reference/v1".into(),
            verification_policy_sha256: verification_policy_sha256(),
            registry_bytes: b"{}".to_vec(),
            index_bytes: b"{}".to_vec(),
            cases: vec![],
        };
        assert!(read_manifest(&serde_json::to_vec(&m).unwrap()).is_err());
    }
    #[test]
    fn execution_requires_completion_limits_and_matching_outputs() {
        let mut e = ExecutionAttestation {
            schema: "jope.cpu-context-execution/controller-v1".into(),
            supervisor_receipt_sha256: "a".repeat(64),
            program_sha256: "b".repeat(64),
            input_sha256: "c".repeat(64),
            output_sha256: "d".repeat(64),
            completed: true,
            cpu_limit: 1,
            memory_max: 1 << 30,
            swap_max: 0,
            threads: 1,
            elapsed_seconds: 1.0,
            deadline_seconds: 2.0,
        };
        let check = |e: &ExecutionAttestation| {
            verify_execution(
                &serde_json::to_vec(e).unwrap(),
                &"b".repeat(64),
                &"c".repeat(64),
                &"d".repeat(64),
                30.0,
            )
        };
        assert!(check(&e).is_ok());
        e.swap_max = 1;
        assert!(check(&e).is_err());
        e.swap_max = 0;
        e.completed = false;
        assert!(check(&e).is_err());
        e.completed = true;
        e.elapsed_seconds = 3.0;
        assert!(check(&e).is_err());
    }
}
