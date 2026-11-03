use super::*;

// M5 CPU prep only: this drives the real frozen adapter on a serving-shaped
// 12-candidate slate, but uses synthetic program channels. It is not the
// required RES-07 serving-fixture qualification or disconnected-ranker gate.
#[test]
#[ignore = "requires externally staged, hash-pinned RES-06 CPU bundle"]
fn real_cpu_adapter_admission_is_sensitive_to_program_channels_and_revealed_fitness() {
    use sha2::{Digest, Sha256};
    use std::{env, fs, path::PathBuf};

    const JOPE_SHA256: &str = "ce39c75f37624e64dfbf89e6493adbf74811a5248f9355997161491e93a132ef";

    fn execute(
        adapter: &CpuAdapter,
        fixture: &AdmissionSlateFixture,
        fitness_override: Option<f64>,
    ) -> [usize; ADMITTED] {
        let mut measured = fixture.measured();
        if let Some(fitness) = fitness_override {
            measured[0].surrogate_fitness = fitness;
        }
        let programs = fixture.programs();
        let slate = ServingAdmissionSlate::prepare(
            &measured,
            &programs[..6],
            &programs[6..14],
            &fixture.conditioning,
            RevealedRule::StorageOrder,
            false,
        )
        .unwrap();
        let predicted = adapter.predict_costs(slate.inputs().unwrap()).unwrap();
        let costs: Vec<_> = predicted
            .total_cost
            .iter()
            .map(|&cost| f64::from(cost))
            .collect();
        let mut selection_rng = ChaCha12Rng::seed_from_u64(0x4d35);
        slate.select(&costs, &mut selection_rng).unwrap()
    }

    let root = PathBuf::from(
        env::var_os("JOPE_PARITY_BUNDLE_DIR")
            .expect("JOPE_PARITY_BUNDLE_DIR required for the real CPU adapter prep test"),
    );
    let bytes = fs::read(root.join("jope.safetensors")).expect("read frozen CPU weights");
    assert_eq!(
        format!("{:x}", Sha256::digest(&bytes)),
        JOPE_SHA256,
        "frozen CPU adapter identity"
    );
    let adapter =
        CpuAdapter::from_bytes_on_device(&bytes, &candle_core_0_11_0::Device::Cpu).unwrap();

    let baseline_fixture = AdmissionSlateFixture::new();
    let baseline = execute(&adapter, &baseline_fixture, None);

    let mut program_channels_changed = false;
    'program_channels: for candidate in 6..14 {
        for value in [-10_000.0, -1_000.0, -100.0, 100.0, 1_000.0, 10_000.0] {
            let mut fixture = AdmissionSlateFixture::new();
            fixture.features[candidate].fill(value);
            fixture.projections[candidate].fill(value);
            if execute(&adapter, &fixture, None) != baseline {
                program_channels_changed = true;
                break 'program_channels;
            }
        }
    }
    assert!(
        program_channels_changed,
        "real CPU ranker admitted set did not change under deterministic program-channel perturbations"
    );

    let mut fitness_changed = false;
    for fitness in [-10_000.0, -1_000.0, -100.0, 100.0, 1_000.0, 10_000.0] {
        if execute(&adapter, &baseline_fixture, Some(fitness)) != baseline {
            fitness_changed = true;
            break;
        }
    }
    assert!(
        fitness_changed,
        "real CPU ranker admission did not change under deterministic revealed-fitness perturbations"
    );
}
