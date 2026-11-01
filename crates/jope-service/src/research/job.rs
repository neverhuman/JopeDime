//! Source-approved, bounded TRAIN recording. Authentication is not quality qualification.
use super::{
    capture::digest,
    qualified::{self, OperatorOrigin, OperatorProvenance, QualifiedCaptureProvider},
    AcceptedJobAdmission, CaptureConfig, NativeCaptureInputs, RecordEvidence, RecordIdentity,
    RecordInventory, RecordTask, RecordingBackend, TrustedJobResolver, DOPE_SHA256,
};
use crate::{
    dataset::jmat,
    discover::r#loop::{NativeSpace, Plan, ProcessExecutor, RoundContext, SearchSpace},
    ranker::{
        admission::{
            CpuRoundAdmission, RevealedRule, RoundInputProvider, RoundInputs, RoundProgram,
        },
        cuda::release::{load_r882_release, R882},
        stage1b::TokenProjection,
    },
    run::{learned::AdmittedRanker, service::BackendSpace},
    store::Store,
};
use anyhow::{ensure, Context, Result};
use jope_api::{ids::RunId, Effort};
use jope_runtime::backend::DeviceRequest;
use rand_chacha::ChaCha12Rng;
use serde::Serialize;
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

const JOB: &str = "qlt03-slump85-v1";
const CATALOG: &str = "e5f732fd784bf67bfc16f4173c585b0861e6e8a6026808df857097a721bbf0ba";
const EXCLUSIONS: &str = "8e786b2dcb153a5c1c4f43dbcb773b53df6f0dd7d325c94f8d7ec2f73352d3e8";
const MANIFEST: &str = "acacfba8c5c083cd16108c25b2a53d82050376fa945bb4fd19f7f7a84a861924";
const TRAIN: &str = "b777779435c9e4685735e95ce41a9e28e14e01eb4e2de5cb685219f65414c4d8";
const DATASET: &str = "b9be38a2d975279c";
const FAMILY: &str = "e55692cd1b1b484e2c084b8bc0cde5dfa0db33c7";
const TARGET: &str = "Compressive_Strength_Mpa";

#[derive(Debug, Serialize)]
pub struct RecordingReceipt {
    pub job_id: String,
    pub work_dir: PathBuf,
    pub admission_receipt: PathBuf,
    pub admission_receipt_sha256: String,
    pub inventory_path: PathBuf,
    pub inventory: RecordInventory,
    pub quality_qualified: bool,
}

fn pinned(root: &Path, name: &str, size: u64, sha: &str) -> Result<Vec<u8>> {
    let path = root.join(name);
    ensure!(
        fs::symlink_metadata(&path)?.file_type().is_file(),
        "input must be a regular file: {name}"
    );
    let file = File::open(path)?;
    ensure!(file.metadata()?.len() == size, "input size differs: {name}");
    let mut bytes = Vec::new();
    file.take(size + 1).read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() as u64 == size && digest(&bytes) == sha,
        "input digest differs: {name}"
    );
    Ok(bytes)
}

// All policy metadata is pinned before any table payload is opened. This is an
// explicit allow-list, not a job document capable of minting an Engine context.
fn metadata(root: &Path) -> Result<()> {
    let catalog: Value =
        serde_json::from_slice(&pinned(root, "catalog.json", 7_804_577, CATALOG)?)?;
    let exclusions: Value =
        serde_json::from_slice(&pinned(root, "exclusions.json", 193_315, EXCLUSIONS)?)?;
    let manifest: Value =
        serde_json::from_slice(&pinned(root, "dataset-manifest.json", 1_440, MANIFEST)?)?;
    let row = catalog["datasets"]
        .as_array()
        .context("catalog dataset list missing")?
        .iter()
        .find(|row| row["id"] == DATASET)
        .context("dataset absent from catalog")?;
    ensure!(
        row["task"] == "regression"
            && row["family_key"] == FAMILY
            && row["family_key_source"] == "provenance.dedupe_fingerprint"
            && row["train_sha256"] == TRAIN
            && row["train_rows"] == 85,
        "catalog role/family differs"
    );
    ensure!(
        exclusions["sealed_exclusion_already_applied"] == true
            && exclusions["authoritative_bound_family_keys"]
                .as_array()
                .context("family closure missing")?
                .iter()
                .any(|key| key == FAMILY)
            && !exclusions["excluded_family_groups"]
                .as_array()
                .context("exclusions missing")?
                .iter()
                .any(|row| row["family_group_id"] == FAMILY),
        "family is not admitted by the exclusion-aware closure"
    );
    ensure!(
        manifest["id"] == DATASET
            && manifest["origin"] == "real"
            && manifest["task"] == "regression"
            && manifest["provenance"]["dedupe_fingerprint"] == FAMILY
            && manifest["files"]["train"]["sha256"] == TRAIN
            && manifest["files"]["train"]["rows"] == 85
            && manifest["files"]["train"]["columns"] == 10
            && manifest["files"]["train"]["has_header"] == false
            && manifest["target"]["index"] == -1
            && manifest["target"]["name"] == TARGET,
        "dataset manifest differs from approved TRAIN job"
    );
    Ok(())
}

fn create(path: &Path, bytes: &[u8]) -> Result<()> {
    #[cfg(unix)]
    use std::os::unix::fs::OpenOptionsExt;
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    options.mode(0o600);
    let mut file = options.open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    File::open(path.parent().context("missing output parent")?)?.sync_all()?;
    Ok(())
}

/// Run the single source-approved job using this binary's isolated native worker.
/// The two-round budget is operational capture evidence, never a quality gate.
pub fn record_catalog_job(
    job_id: &str,
    input_root: &Path,
    output: &Path,
    evaluate_all: bool,
) -> Result<RecordingReceipt> {
    ensure!(
        job_id == JOB,
        "no source-approved research job matches this ID"
    );
    super::outbox::validate_destination(output)?;
    let inventory_path = output.with_extension("inventory.json");
    ensure!(
        !inventory_path.try_exists()? && inventory_path != output,
        "inventory destination exists or aliases output"
    );
    let source = option_env!("JOPE_BUILD_SOURCE_SHA").unwrap_or("");
    ensure!(
        super::hex(source, 40),
        "recording requires a source-bound build"
    );
    ensure!(
        fs::symlink_metadata(input_root)?.file_type().is_dir(),
        "input root must be a real directory"
    );
    let root = input_root.canonicalize()?;
    metadata(&root)?;
    let bundle = load_r882_release(&root.join("bundle"))?;
    let train = pinned(&root, "train.csv", 9_276, TRAIN)?;
    let features: Vec<_> = (0..9).map(|i| format!("x{i}")).collect();
    let mut csv = format!("{},{}\n", features.join(","), TARGET).into_bytes();
    csv.extend_from_slice(&train);
    let table = jope_runtime::table::parse_csv(&csv, &Default::default())?;
    ensure!(
        table.rows.len() == 85 && table.headers.len() == 10,
        "TRAIN shape changed"
    );
    ensure!(
        table.rows.iter().all(|row| row.len() == 10
            && row.iter().all(|v| v
                .as_deref()
                .is_some_and(|v| v.parse::<f64>().is_ok_and(f64::is_finite)))),
        "TRAIN must be finite numeric data"
    );
    let matrix = jmat::encode(&table)?;
    let seeds: BTreeSet<_> = NativeSpace
        .seeds()?
        .iter()
        .map(|dsl| NativeSpace.canonical(dsl))
        .collect::<Result<_>>()?;
    ensure!(
        seeds.len() >= 4 && seeds.len() <= 16,
        "invalid native seed roster"
    );
    let run = RunId::new();
    let work_dir = output
        .parent()
        .context("missing output parent")?
        .join(format!("{JOB}-{run}"));
    let mut builder = fs::DirBuilder::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder.create(&work_dir)?;
    let work_dir = work_dir.canonicalize()?;
    let matrix_path = work_dir.join("train.jmat");
    create(&matrix_path, &matrix)?;
    let plan = Plan {
        run_id: run,
        dataset_id: format!("ds_{}", &TRAIN[..24]).parse()?,
        problem_digest: digest(&serde_json::to_vec(
            &json!({"protocol":"qlt03.slump85/v1", "matrix":digest(&matrix), "target":TARGET, "features":features, "objective":"rmse"}),
        )?),
        bundle_digest: R882.trim_start_matches("sha256:").into(),
        binary_version: format!("qlt03/{source}"),
        matrix_path,
        matrix_sha256: digest(&matrix),
        target: TARGET.into(),
        features,
        objective: "rmse".into(),
        seed: 250119,
        planned_workers: 1,
        effort: Effort::Fast,
        evaluations: seeds.len() as u64 + if evaluate_all { 8 } else { 4 },
        wall_seconds: 300,
        evaluation_wall_ms: 10_000,
        evaluation_rss_bytes: 1024 * 1024 * 1024,
        groups: None,
        times: None,
        gap_rows: 0,
    };
    let admitted = AdmittedRanker::r882(load_r882_release(&root.join("bundle"))?)?;
    let mut learned = admitted.instantiate(DeviceRequest::Cpu, 1)?;
    learned.bind_fit(&matrix, &plan)?;
    let conditioning = learned.research_conditioning(&plan)?.clone();
    let conditioning_sha256 = qualified::conditioning_sha256(&plan, &conditioning)?;
    let provenance = Arc::new(LiveProvenance::new(&plan, &seeds)?);
    let provider = Arc::new(QualifiedCaptureProvider::from_admitted(
        &bundle,
        &plan,
        &learned,
        provenance.clone(),
    )?);
    let identity = RecordIdentity {
        run_id: plan.run_id.to_string(),
        task: RecordTask::Regression,
        evidence: RecordEvidence::Engine,
        partition: "train".into(),
        source_commit: source.into(),
        corpus_manifest_sha256: CATALOG.into(),
        dataset_sha256: TRAIN.into(),
        family_sha256: digest(FAMILY.as_bytes()),
        problem_sha256: plan.problem_digest.clone(),
        bundle_sha256: plan.bundle_digest.clone(),
        conditioning_sha256: conditioning_sha256.clone(),
        evaluator_sha256: qualified::evaluator_sha256(),
        dope_weights_sha256: DOPE_SHA256.into(),
    };
    let config = CaptureConfig {
        max_attempts: plan.evaluations,
        max_rounds: 2,
        max_bytes: 16 * 1024 * 1024,
        evaluate_all_proposals: evaluate_all,
        seed_protocol_sha256: qualified::seed_protocol_sha256(),
    };
    let context = provider.engine_context(identity.clone(), config.clone())?;
    let receipt_bytes = serde_json::to_vec_pretty(&json!({
        "schema":"jope.research.input-admission/v1", "job_id": JOB, "identity":identity,
        "plan":plan, "config":config, "dataset_manifest_sha256":MANIFEST, "exclusion_closure_sha256":EXCLUSIONS,
        "family_key":FAMILY, "family_digest_recipe":"sha256(UTF-8 authoritative family_key)",
        "header_recipe":"x0..x8,Compressive_Strength_Mpa; prepend only; preserve all 85 TRAIN rows and labels",
        "quality_qualified":false, "release_qualified":false
    }))?;
    let admission_receipt = work_dir.join("input-admission.json");
    create(&admission_receipt, &receipt_bytes)?;
    let receipt_sha = digest(&receipt_bytes);
    let mut resolver = TrustedJobResolver::new();
    resolver.register(AcceptedJobAdmission::from_verified_receipts(
        JOB,
        &plan,
        &receipt_sha,
        context,
    )?)?;
    let context = resolver.resolve(JOB, &plan)?;
    let inputs =
        NativeCaptureInputs::from_bundle(&bundle, &plan, &conditioning_sha256, &conditioning)?;
    let admission = CpuRoundAdmission::from_bundle(
        &bundle,
        &plan,
        NativeRoundInputs {
            inputs,
            conditioning,
        },
        RevealedRule::SurrogateFitness,
    )?;
    let executor = ProcessExecutor::new(&std::env::current_exe()?, 1, &work_dir.join("workers"))?;
    let space = BackendSpace {
        native: RecordingSpace { provenance },
        learned: true,
    };
    let store = Arc::new(Store::open(work_dir.join("store"))?);
    let inventory = super::record_run(
        store,
        plan,
        context,
        RecordingBackend {
            executor: &executor,
            space: &space,
            admission: &admission,
            live_workers: 1,
        },
        output,
    )?;
    create(&inventory_path, &serde_json::to_vec_pretty(&inventory)?)?;
    ensure!(
        inventory.counts.rounds == 2 && inventory.counts.incomplete_rounds == 0,
        "bounded job did not produce two complete Engine rounds; preserve evidence"
    );
    Ok(RecordingReceipt {
        job_id: JOB.into(),
        work_dir,
        admission_receipt,
        admission_receipt_sha256: receipt_sha,
        inventory_path,
        inventory,
        quality_qualified: false,
    })
}

struct NativeRoundInputs {
    inputs: NativeCaptureInputs,
    conditioning: TokenProjection,
}
impl NativeRoundInputs {
    fn program(&self, dsl: &str) -> Result<RoundProgram> {
        let p = self.inputs.program(dsl, "native_input")?;
        Ok(RoundProgram {
            dsl: dsl.into(),
            features412: p
                .feat412
                .try_into()
                .map_err(|_| anyhow::anyhow!("feature width"))?,
            projection16: p
                .proj16
                .into_iter()
                .map(|v| v as f32)
                .collect::<Vec<_>>()
                .try_into()
                .map_err(|_| anyhow::anyhow!("projection width"))?,
        })
    }
}
impl RoundInputProvider for NativeRoundInputs {
    fn observe(&self, context: &RoundContext<'_>, proposals: &[String]) -> Result<RoundInputs> {
        Ok(RoundInputs {
            pool: context
                .measured
                .iter()
                .map(|m| self.program(m.dsl))
                .collect::<Result<_>>()?,
            pool_gates: context
                .measured
                .iter()
                .map(|m| m.surrogate_fitness < 100.0)
                .collect(),
            proposals: proposals
                .iter()
                .map(|dsl| self.program(dsl))
                .collect::<Result<_>>()?,
            conditioning: self.conditioning.clone(),
        })
    }
}

struct LiveProvenance {
    plan_sha256: String,
    origins: Mutex<BTreeMap<(u64, String), OperatorOrigin>>,
    pending: Mutex<BTreeMap<String, OperatorOrigin>>,
}
impl LiveProvenance {
    fn new(plan: &Plan, seeds: &BTreeSet<String>) -> Result<Self> {
        Ok(Self {
            plan_sha256: digest(&serde_json::to_vec(plan)?),
            origins: Mutex::new(
                seeds
                    .iter()
                    .map(|dsl| ((0, digest(dsl.as_bytes())), OperatorOrigin::Seed))
                    .collect(),
            ),
            pending: Mutex::new(BTreeMap::new()),
        })
    }
    fn generated(&self, dsl: &str, arm: &'static str) {
        self.pending
            .lock()
            .expect("provenance mutex poisoned")
            .entry(digest(dsl.as_bytes()))
            .or_insert(OperatorOrigin::Native(arm));
    }
}
impl OperatorProvenance for LiveProvenance {
    fn validate_plan(&self, plan: &Plan) -> Result<()> {
        ensure!(
            self.plan_sha256 == digest(&serde_json::to_vec(plan)?),
            "provenance Plan changed"
        );
        Ok(())
    }
    fn observe_round(&self, round: u64, dsls: &[String]) -> Result<()> {
        if round == 0 {
            return Ok(());
        }
        let mut pending = self.pending.lock().expect("provenance mutex poisoned");
        let selected = dsls
            .iter()
            .map(|dsl| {
                let sha = digest(dsl.as_bytes());
                Ok((
                    (round, sha.clone()),
                    *pending
                        .get(&sha)
                        .context("program was not observed at the native generator")?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        self.origins
            .lock()
            .expect("provenance mutex poisoned")
            .extend(selected);
        pending.clear();
        Ok(())
    }
    fn origin(&self, round: u64, sha: &str) -> Result<OperatorOrigin> {
        self.origins
            .lock()
            .expect("provenance mutex poisoned")
            .get(&(round, sha.into()))
            .copied()
            .context("unobserved native program origin")
    }
}
struct RecordingSpace {
    provenance: Arc<LiveProvenance>,
}
impl SearchSpace for RecordingSpace {
    fn canonical(&self, dsl: &str) -> Result<String> {
        NativeSpace.canonical(dsl)
    }
    fn seeds(&self) -> Result<Vec<String>> {
        NativeSpace.seeds()
    }
    fn propose(
        &self,
        parents: &[String],
        count: usize,
        rng: &mut ChaCha12Rng,
    ) -> Result<Vec<String>> {
        NativeSpace.propose_observed(parents, count, rng, |dsl, arm| {
            self.provenance.generated(dsl, arm)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_chacha::rand_core::{RngCore, SeedableRng};

    #[test]
    fn unknown_job_refuses_before_inputs_or_output() {
        let dir = tempfile::tempdir().unwrap();
        let output = dir.path().join("rounds.jsonl");
        for job in ["", "../qlt03-slump85-v1", "arbitrary.json"] {
            let error =
                record_catalog_job(job, &dir.path().join("missing"), &output, true).unwrap_err();
            assert!(error.to_string().contains("no source-approved"));
            assert!(!output.exists());
        }
    }

    #[test]
    fn pinned_inputs_reject_modified_bytes_size_and_symlinks() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("input"), b"TRAIN").unwrap();
        assert_eq!(
            pinned(dir.path(), "input", 5, &digest(b"TRAIN")).unwrap(),
            b"TRAIN"
        );
        assert!(pinned(dir.path(), "input", 4, &digest(b"TRAIN")).is_err());
        assert!(pinned(dir.path(), "input", 5, &digest(b"train")).is_err());
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(dir.path().join("input"), dir.path().join("link")).unwrap();
            assert!(pinned(dir.path(), "link", 5, &digest(b"TRAIN")).is_err());
        }
    }

    #[test]
    fn generator_observation_preserves_candidates_and_rng() {
        let parents = NativeSpace.seeds().unwrap();
        let mut plain_rng = ChaCha12Rng::seed_from_u64(250119);
        let mut observed_rng = plain_rng.clone();
        let expected = NativeSpace.propose(&parents, 32, &mut plain_rng).unwrap();
        let mut seen = Vec::new();
        let actual = NativeSpace
            .propose_observed(&parents, 32, &mut observed_rng, |dsl, arm| {
                assert!(arm.starts_with("native_"));
                seen.push(dsl.to_owned());
            })
            .unwrap();
        assert!(!actual.is_empty());
        assert_eq!(actual, expected);
        assert_eq!(seen, actual);
        assert_eq!(plain_rng.next_u64(), observed_rng.next_u64());
    }
}
