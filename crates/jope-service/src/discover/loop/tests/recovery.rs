use super::*;

fn checkpoint_champion(champion: &Option<Champion>) -> Value {
    champion
        .as_ref()
        .map(|champion| {
            json!({
                "candidate": &champion.candidate,
                "loss": champion.loss,
                "value": champion.value,
                "predictions_digest": &champion.predictions_digest,
                "model": &champion.model,
            })
        })
        .unwrap_or(Value::Null)
}

#[test]
#[ignore = "requires the hash-pinned RES-07 CPU packet and frozen RES-06 bundle"]
fn learned_cpu_admission_recovery_matches_uninterrupted_execution() {
    use crate::ranker::{
        admission::{
            CpuRoundAdmission, RevealedRule, RoundInputProvider, RoundInputs, RoundProgram,
        },
        stage1b::TokenProjection,
    };
    use serde_json::Value;
    use sha2::{Digest, Sha256};
    use std::{
        collections::HashMap,
        env, fs,
        sync::atomic::{AtomicUsize, Ordering},
    };

    const WEIGHTS_SHA256: &str = "ce39c75f37624e64dfbf89e6493adbf74811a5248f9355997161491e93a132ef";
    const INDEX_SHA256_ENV: &str = "JOPE_GOLDEN_INDEX_SHA256";

    fn read_pinned(path: &Path, expected: &str) -> Vec<u8> {
        assert!(!fs::symlink_metadata(path).unwrap().file_type().is_symlink());
        let bytes = fs::read(path).unwrap();
        assert_eq!(
            format!("{:x}", Sha256::digest(&bytes)),
            expected,
            "{}",
            path.display()
        );
        bytes
    }

    fn tensor_f32(bytes: &[u8], name: &str) -> Vec<f32> {
        let tensors = safetensors::SafeTensors::deserialize(bytes).unwrap();
        let tensor = tensors.tensor(name).unwrap();
        assert_eq!(tensor.dtype(), safetensors::Dtype::F32);
        tensor
            .data()
            .chunks_exact(4)
            .map(|raw| f32::from_le_bytes(raw.try_into().unwrap()))
            .collect()
    }

    struct Slate {
        prog: Vec<f32>,
        proj: Vec<f32>,
        tokens: Vec<f32>,
        mask: Vec<f32>,
        gates: Vec<f32>,
        revealed_fitness: Vec<f32>,
    }

    fn load_slate(root: &Path, index: &Value, cid: &str, width: usize) -> Slate {
        let name = format!("slates/{cid}-{width}.safetensors");
        let entry = &index["files"][name.as_str()];
        let path = root.join(&name);
        assert!(path.canonicalize().unwrap().starts_with(root));
        let bytes = read_pinned(
            &path,
            entry["sha256"].as_str().expect("slate must be indexed"),
        );
        assert_eq!(bytes.len() as u64, entry["bytes"].as_u64().unwrap());
        Slate {
            prog: tensor_f32(&bytes, "input.prog_412"),
            proj: tensor_f32(&bytes, "input.proj_16"),
            tokens: tensor_f32(&bytes, "input.mm_tokens_10x32"),
            mask: tensor_f32(&bytes, "input.mm_mask_10"),
            gates: tensor_f32(&bytes, "input.revealed_gate"),
            revealed_fitness: tensor_f32(&bytes, "input.revealed_raw_f"),
        }
    }

    struct ReplayProvider {
        by_dsl: HashMap<String, usize>,
        slates: HashMap<u64, Slate>,
        calls: Arc<AtomicUsize>,
    }
    impl RoundInputProvider for ReplayProvider {
        fn observe(&self, context: &RoundContext<'_>, proposals: &[String]) -> Result<RoundInputs> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let slate = self
                .slates
                .get(&context.round)
                .context("unexpected learned round")?;
            let program = |dsl: &str| -> Result<RoundProgram> {
                let index = *self
                    .by_dsl
                    .get(dsl)
                    .context("program missing from pinned packet")?;
                ensure!(index < 20, "replay program outside pinned packet subset");
                Ok(RoundProgram {
                    dsl: dsl.to_owned(),
                    features412: slate.prog[index * 412..(index + 1) * 412]
                        .try_into()
                        .unwrap(),
                    projection16: slate.proj[index * 16..(index + 1) * 16].try_into().unwrap(),
                })
            };
            let pool = context
                .measured
                .iter()
                .map(|m| program(m.dsl))
                .collect::<Result<Vec<_>>>()?;
            let pool_gates = context
                .measured
                .iter()
                .map(|m| {
                    let index = *self.by_dsl.get(m.dsl).context("measured program missing")?;
                    ensure!(
                        index < 4,
                        "learned replay changed the measured four-program pool"
                    );
                    ensure!(
                        slate.gates[index] == 0.0 || slate.gates[index] == 1.0,
                        "nonbinary revealed gate"
                    );
                    Ok(slate.gates[index] == 1.0)
                })
                .collect::<Result<Vec<_>>>()?;
            let proposals = proposals
                .iter()
                .map(|dsl| program(dsl))
                .collect::<Result<Vec<_>>>()?;
            let conditioning = TokenProjection {
                tokens: std::array::from_fn(|row| {
                    std::array::from_fn(|col| slate.tokens[row * 32 + col])
                }),
                mask: slate.mask.as_slice().try_into().unwrap(),
            };
            Ok(RoundInputs {
                pool,
                pool_gates,
                proposals,
                conditioning,
            })
        }
    }

    struct ReplaySpace {
        dsls: Vec<String>,
        proposals: AtomicUsize,
    }
    impl SearchSpace for ReplaySpace {
        fn canonical(&self, dsl: &str) -> Result<String> {
            Ok(dsl.to_owned())
        }
        fn seeds(&self) -> Result<Vec<String>> {
            Ok(self.dsls[..4].to_vec())
        }
        fn propose(&self, _: &[String], count: usize, _: &mut ChaCha12Rng) -> Result<Vec<String>> {
            ensure!(count == 8, "expected non-exploring eight-proposal wave");
            let wave = self.proposals.fetch_add(1, Ordering::SeqCst);
            let start = 4 + wave * 8;
            ensure!(
                start + 8 <= self.dsls.len(),
                "test fixture proposal waves exhausted"
            );
            Ok(self.dsls[start..start + 8].to_vec())
        }
    }

    struct ReplayExecutor {
        indices: HashMap<String, usize>,
        seed_fitness: Vec<f32>,
    }
    impl Executor for ReplayExecutor {
        fn evaluate(&self, request: &Request) -> Response {
            let index = self.indices[&request.dsl];
            let (surrogate, objective) = if index < 4 {
                (f64::from(self.seed_fitness[index]), index as f64)
            } else {
                (index as f64, 100.0 + index as f64)
            };
            Response {
                attempt: request.attempt.clone(),
                scores: vec![surrogate, objective, objective],
                fold_scores: vec![],
                predictions_digest: digest(request.dsl.as_bytes()),
                wall_ms: 0,
                failure: None,
            }
        }
        fn fitted_model(&self, request: &Request) -> Result<Vec<u8>> {
            Ok(request.dsl.as_bytes().to_vec())
        }
    }

    fn run_case(
        root_dir: &Path,
        index: &Value,
        root: &Path,
        cid: &str,
        dsls: &[String],
        weights: &[u8],
        interrupted: bool,
    ) -> (Value, usize) {
        let case_dir = root_dir.join(if interrupted {
            "recovered"
        } else {
            "uninterrupted"
        });
        fs::create_dir_all(&case_dir).unwrap();
        let mut p = plan(&case_dir, 30, Effort::Fast, 20);
        p.planned_workers = 8;
        let store = Arc::new(Store::open(case_dir.join("store")).unwrap());
        let mut discovery = Discovery::open(store.clone(), p.clone()).unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let slate12 = load_slate(root, index, cid, 12);
        let seed_fitness = slate12.revealed_fitness.clone();
        let slate20 = load_slate(root, index, cid, 20);
        let provider = ReplayProvider {
            by_dsl: dsls
                .iter()
                .enumerate()
                .map(|(i, dsl)| (dsl.clone(), i))
                .collect(),
            slates: [(1, slate12), (2, slate20)].into_iter().collect(),
            calls: calls.clone(),
        };
        let admission = CpuRoundAdmission::for_cpu_replay_weights(
            weights,
            p.clone(),
            provider,
            RevealedRule::StorageOrder,
        )
        .unwrap();
        let executor = ReplayExecutor {
            indices: dsls
                .iter()
                .enumerate()
                .map(|(i, dsl)| (dsl.clone(), i))
                .collect(),
            seed_fitness,
        };
        let space = ReplaySpace {
            dsls: dsls.to_vec(),
            proposals: AtomicUsize::new(0),
        };
        discovery
            .advance(&space, &admission, &executor, 8, 1)
            .unwrap();
        assert_eq!(
            calls.load(Ordering::SeqCst),
            0,
            "seed round bypasses learned admission"
        );
        discovery
            .advance(
                &space,
                &admission,
                &executor,
                8,
                if interrupted { 1 } else { 2 },
            )
            .unwrap();
        assert_eq!(
            calls.load(Ordering::SeqCst),
            2,
            "both learned proposals were evaluated before recovery"
        );
        if interrupted {
            assert_eq!(
                discovery.checkpoint.pending.len(),
                1,
                "one admitted sibling wave is durably pending"
            );
            let retained = serde_json::to_value(&discovery.checkpoint.pending).unwrap();
            discovery.pause().unwrap();
            drop(discovery);
            let mut recovered = Discovery::open(store, p).unwrap();
            assert_eq!(
                serde_json::to_value(&recovered.checkpoint.pending).unwrap(),
                retained
            );
            recovered
                .advance(&space, &admission, &executor, 1, 1)
                .unwrap();
            assert_eq!(
                calls.load(Ordering::SeqCst),
                2,
                "recovery must consume persisted admission without reranking"
            );
            (
                serde_json::json!({
                    "programs": recovered.checkpoint.programs,
                    "memo": recovered.checkpoint.memo,
                    "elites": recovered.checkpoint.elites,
                    "champion": checkpoint_champion(&recovered.checkpoint.champion),
                    "pending": recovered.checkpoint.pending,
                }),
                calls.load(Ordering::SeqCst),
            )
        } else {
            (
                serde_json::json!({
                    "programs": discovery.checkpoint.programs,
                    "memo": discovery.checkpoint.memo,
                    "elites": discovery.checkpoint.elites,
                    "champion": checkpoint_champion(&discovery.checkpoint.champion),
                    "pending": discovery.checkpoint.pending,
                }),
                calls.load(Ordering::SeqCst),
            )
        }
    }

    let golden =
        PathBuf::from(env::var_os("JOPE_GOLDEN_DIR").expect("hash-pinned CPU packet required"));
    let golden = golden.canonicalize().unwrap();
    let index_sha = env::var(INDEX_SHA256_ENV).expect("trusted index SHA required");
    let index_bytes = read_pinned(&golden.join("index.json"), &index_sha);
    let index: Value = serde_json::from_slice(&index_bytes).unwrap();
    assert_eq!(index["schema"], "jope.golden-fixtures/v1");
    assert_eq!(index["contexts"], 50);
    let cid = index["files"]
        .as_object()
        .unwrap()
        .keys()
        .find_map(|name| name.strip_prefix("contexts/")?.strip_suffix(".safetensors"))
        .expect("pinned context inventory empty")
        .to_owned();
    let programs_path = golden.join("programs.jsonl");
    let programs_bytes = read_pinned(
        &programs_path,
        index["files"]["programs.jsonl"]["sha256"].as_str().unwrap(),
    );
    let dsls: Vec<String> = std::str::from_utf8(&programs_bytes)
        .unwrap()
        .lines()
        .take(20)
        .map(|line| {
            serde_json::from_str::<Value>(line).unwrap()["dsl"]
                .as_str()
                .unwrap()
                .to_owned()
        })
        .collect();
    assert_eq!(dsls.len(), 20);
    let bundle = PathBuf::from(
        env::var_os("JOPE_PARITY_BUNDLE_DIR").expect("frozen RES-06 bundle required"),
    );
    let weights = read_pinned(&bundle.join("jope.safetensors"), WEIGHTS_SHA256);
    let dir = tempfile::tempdir().unwrap();
    let uninterrupted = run_case(dir.path(), &index, &golden, &cid, &dsls, &weights, false);
    let recovered = run_case(dir.path(), &index, &golden, &cid, &dsls, &weights, true);
    assert_eq!(
        uninterrupted.0, recovered.0,
        "recovered Discovery differs from uninterrupted execution"
    );
    assert_eq!(uninterrupted.0["programs"].as_object().unwrap().len(), 12);
    assert_eq!(uninterrupted.1, 2);
    assert_eq!(recovered.1, 2);
    println!("RNK10_S2_REAL_CPU_REPLAY contexts=1 proposal_rounds=2 interruptions=1 capacity=8->1 learned_invocations=2->2 state_equal=true");
}
