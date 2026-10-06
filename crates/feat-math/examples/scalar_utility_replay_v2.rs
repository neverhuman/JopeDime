//! Replay actual utility winners, manual controls, and GP EEC phases.
#[path = "scalar_utility_v2/replay_harness.rs"]
mod harness;
use feat_math::kernel::scalar_program::{
    ScalarCandidate, ScalarCarrier, ScalarExpressionProgram, ScalarLanguage, ScalarModel,
    ScalarNode, ScalarProducerKind, ScalarProducerProfile,
};
use feat_math::math_expr::Var;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Path, PathBuf},
};
fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn read(path: &Path) -> Value {
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}
fn bits(value: &Value) -> Vec<f64> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| f64::from_bits(v.as_u64().unwrap()))
        .collect()
}
fn bytes(value: &Value) -> Vec<u8> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| u8::try_from(v.as_u64().unwrap()).unwrap())
        .collect()
}
fn record(
    model: ScalarModel,
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
    probe: Vec<Vec<f64>>,
    identity: Value,
    cases: &mut Vec<harness::RecipeCase>,
    ids: &mut Vec<Value>,
) {
    let frozen = model.to_json().unwrap();
    let restored = ScalarModel::from_json(&frozen).unwrap();
    assert_eq!(restored.to_json().unwrap(), frozen);
    let direct = model.predict(&probe).unwrap();
    let replay = restored.predict(&probe).unwrap();
    assert_eq!(
        direct.0.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        replay.0.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    assert_eq!(direct.1, replay.1);
    assert_eq!(model.to_json().unwrap(), frozen);
    assert_eq!(restored.to_json().unwrap(), frozen);
    let program = model.program().clone();
    let carrier = model.carrier();
    let lambda = model.lambda();
    let program_bytes = program.to_bytes().unwrap();
    assert_eq!(
        ScalarExpressionProgram::from_bytes(&program_bytes)
            .unwrap()
            .to_bytes()
            .unwrap(),
        program_bytes
    );
    ids.push(json!({"input":identity,"model_sha256":model.identity().unwrap(),"program_sha256":hash(&program_bytes),"model":serde_json::from_str::<Value>(&frozen).unwrap()}));
    cases.push((program, carrier, x, y, probe, lambda, model));
}
fn stored(
    dir: &Path,
    training: (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>),
    identity: Value,
    source: &str,
    cases: &mut Vec<harness::RecipeCase>,
    ids: &mut Vec<Value>,
) {
    let (x, y, probe) = training;
    let frozen = fs::read_to_string(dir.join("winner.json")).unwrap();
    let model = ScalarModel::from_json(&frozen).unwrap();
    assert_eq!(model.to_json().unwrap(), frozen);
    let receipt = read(&dir.join("receipt.json"));
    assert_eq!(receipt["source_sha"], source);
    assert_eq!(receipt["model_sha256"], hash(frozen.as_bytes()));
    for (language, extension) in [
        (ScalarLanguage::Rust, "rs"),
        (ScalarLanguage::Python, "py"),
        (ScalarLanguage::Cpp17, "cpp"),
    ] {
        let actual = model.emit_recipe(language).unwrap();
        assert_eq!(
            actual.as_bytes(),
            fs::read(dir.join(format!("recipe.{extension}"))).unwrap()
        );
        let entry = receipt["recipes"]
            .as_array()
            .unwrap()
            .iter()
            .find(|a| a["language"] == format!("{language:?}"))
            .unwrap();
        assert_eq!(entry["sha256"], hash(actual.as_bytes()));
        assert_eq!(entry["bytes"], actual.len());
    }
    let fitted = model.emit_fitted_artifact().unwrap();
    assert_eq!(
        fitted.source.as_bytes(),
        fs::read(dir.join("fitted.cpp")).unwrap()
    );
    assert_eq!(
        fitted.codec.as_bytes(),
        fs::read(dir.join("fitted-codec.json")).unwrap()
    );
    assert_eq!(receipt["fitted_source_sha256"], fitted.source_sha256);
    assert_eq!(receipt["fitted_codec_sha256"], fitted.codec_sha256);
    record(model, x, y, probe, identity, cases, ids);
}
fn replay_eec_event(
    event: &Value,
) -> (ScalarProducerProfile, [(&'static str, ScalarCandidate); 3]) {
    let carrier = match event["carrier"].as_str().unwrap() {
        "row" => ScalarCarrier::Row,
        "elementwise" => ScalarCarrier::Elementwise,
        _ => panic!("unknown EEC carrier"),
    };
    let profile = ScalarProducerProfile::new(ScalarProducerKind::All, carrier, 2).unwrap();
    assert_eq!(profile.sha256(), event["profile_sha256"]);
    let initial = profile
        .admit_candidate(
            ScalarExpressionProgram::new(ScalarNode::Var(Var::Col(0))).unwrap(),
            0,
        )
        .unwrap();
    let created = initial
        .mutate(event["create_seed"].as_u64().unwrap())
        .unwrap();
    assert!(created.changed && created.opt_in_admitted);
    assert_eq!(
        created.candidate.program().to_bytes().unwrap(),
        bytes(&event["created_program"])
    );
    let changed = created
        .candidate
        .mutate(event["change_seed"].as_u64().unwrap())
        .unwrap();
    let removed = created
        .candidate
        .mutate(event["remove_seed"].as_u64().unwrap())
        .unwrap();
    assert!(changed.changed && removed.changed);
    let phases = [
        ("created_program", created.candidate.clone()),
        ("changed_program", changed.candidate),
        ("removed_program", removed.candidate),
    ];
    for (phase, candidate) in &phases {
        assert_eq!(
            candidate.program().to_bytes().unwrap(),
            bytes(&event[*phase])
        );
    }
    (profile, phases)
}

fn cohort_manifest(raw: &[u8]) -> Value {
    assert_eq!(
        hash(raw),
        "74b05d8be0caa6a595815f9c89deb6be5ab716b1dedcd8a8669c7c7577296840"
    );
    serde_json::from_slice(raw).unwrap()
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(
        args.len(),
        6,
        "usage: proof COHORT_MANIFEST NATIVE_OUTPUT EEC_EVENTS OUTPUT NATIVE_SOURCE_SHA"
    );
    let manifest = PathBuf::from(&args[1]);
    let native = PathBuf::from(&args[2]);
    let out = PathBuf::from(&args[4]);
    let source = args[5].as_str();
    assert_eq!(source.len(), 40);
    let raw = fs::read(&manifest).unwrap();
    let manifest = cohort_manifest(&raw);
    let mut cases = vec![];
    let mut identities = vec![];
    for data in manifest["datasets"].as_array().unwrap() {
        let raw = fs::read(
            Path::new(&args[1])
                .parent()
                .unwrap()
                .join(data["path"].as_str().unwrap()),
        )
        .unwrap();
        assert_eq!(hash(&raw), data["sha256"]);
        let input: Value = serde_json::from_slice(&raw).unwrap();
        let rows: Vec<Vec<f64>> = input["rows_bits"]
            .as_array()
            .unwrap()
            .iter()
            .map(bits)
            .collect();
        let y = bits(&input["labels_bits"]);
        let name = data["name"].as_str().unwrap();
        for card in ["B05", "B06", "B07"] {
            for seed in [3, 17, 29, 101, 701] {
                for arm in ["A", "B"] {
                    let dir = native.join(format!("{name}-{card}-{seed}-{arm}"));
                    stored(
                        &dir,
                        (rows[..64].to_vec(), y[..64].to_vec(), rows[64..].to_vec()),
                        json!({"kind":"gp-winner","dataset":name,"card":card,"seed":seed,"arm":arm}),
                        source,
                        &mut cases,
                        &mut identities,
                    );
                }
            }
            let directory = native.join(format!("{name}-{card}-manual-controls"));
            let index = read(&directory.join("native-controls-index.json"));
            for item in index.as_array().unwrap() {
                let formula = item["formula"].as_str().unwrap();
                let dir = directory.join(formula);
                assert_eq!(item["source_sha"], source);
                // Controls have their original full artifact receipt in the retained index.
                let model =
                    ScalarModel::from_json(&fs::read_to_string(dir.join("winner.json")).unwrap())
                        .unwrap();
                assert_eq!(item["model_sha256"], model.identity().unwrap());
                for (language, extension) in [
                    (ScalarLanguage::Rust, "rs"),
                    (ScalarLanguage::Python, "py"),
                    (ScalarLanguage::Cpp17, "cpp"),
                ] {
                    let emitted = model.emit_recipe(language).unwrap();
                    assert_eq!(
                        emitted.as_bytes(),
                        fs::read(dir.join(format!("recipe.{extension}"))).unwrap()
                    );
                    let entry = item["recipes"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .find(|a| a["language"] == format!("{language:?}"))
                        .unwrap();
                    assert_eq!(entry["sha256"], hash(emitted.as_bytes()));
                }
                let fitted = model.emit_fitted_artifact().unwrap();
                assert_eq!(
                    fitted.source.as_bytes(),
                    fs::read(dir.join("fitted.cpp")).unwrap()
                );
                assert_eq!(
                    fitted.codec.as_bytes(),
                    fs::read(dir.join("fitted-codec.json")).unwrap()
                );
                assert_eq!(item["fitted_source_sha256"], fitted.source_sha256);
                assert_eq!(item["fitted_codec_sha256"], fitted.codec_sha256);
                record(
                    model,
                    rows[..64].to_vec(),
                    y[..64].to_vec(),
                    rows[64..].to_vec(),
                    json!({"kind":"manual-control","dataset":name,"card":card,"formula":formula}),
                    &mut cases,
                    &mut identities,
                );
            }
        }
    }
    assert_eq!(
        identities
            .iter()
            .filter(|v| v["input"]["kind"] == "gp-winner")
            .count(),
        240
    );
    assert_eq!(
        identities
            .iter()
            .filter(|v| v["input"]["kind"] == "manual-control")
            .count(),
        302
    );
    let events = read(Path::new(&args[3]));
    assert_eq!(events["disabled_mutation_checks"], 8192);
    assert_eq!(events["events"].as_array().unwrap().len(), 20);
    let x = vec![
        vec![0.25, -0.25],
        vec![0.5, -0.5],
        vec![0.75, -0.75],
        vec![1.0, -1.0],
    ];
    let y = vec![0.5, -0.5, 0.25, -0.25];
    let mut probe = x.clone();
    probe.extend([
        vec![f64::NAN, f64::INFINITY],
        vec![f64::NEG_INFINITY, -0.0],
        vec![1e200, -1e200],
        vec![3e200, 4e200],
        vec![1e-300, -1e-300],
        vec![0.0, -0.0],
    ]);
    let mut event_kinds = std::collections::BTreeSet::new();
    for event in events["events"].as_array().unwrap() {
        assert!(event_kinds.insert((
            event["carrier"].as_str().unwrap().to_owned(),
            event["kind"].as_str().unwrap().to_owned(),
        )));
        let (profile, phases) = replay_eec_event(event);
        for (phase, candidate) in phases {
            let encoded = bytes(&event[phase]);
            assert_eq!(candidate.program().to_bytes().unwrap(), encoded);
            let program = ScalarExpressionProgram::from_bytes(&encoded).unwrap();
            assert_eq!(program.to_bytes().unwrap(), encoded);
            let model = profile.fit_training(&program, &x, &y, 0.02).unwrap();
            record(
                model,
                x.clone(),
                y.clone(),
                probe.clone(),
                json!({"kind":"actual-gp-eec-phase","event":event,"phase":phase}),
                &mut cases,
                &mut identities,
            );
        }
    }
    assert_eq!(
        identities
            .iter()
            .filter(|v| v["input"]["kind"] == "actual-gp-eec-phase")
            .count(),
        60
    );
    let expected_kinds: std::collections::BTreeSet<_> = ["row", "elementwise"]
        .into_iter()
        .flat_map(|carrier| {
            [
                "numerical_policy",
                "func13",
                "func19",
                "func20",
                "func21",
                "func22",
                "RowSum",
                "RowMean",
                "RowL1",
                "RowL2",
            ]
            .into_iter()
            .map(move |kind| (carrier.to_owned(), kind.to_owned()))
        })
        .collect();
    assert_eq!(event_kinds, expected_kinds);
    assert_eq!(cases.len(), 602);
    let unique_inputs: std::collections::BTreeSet<_> = identities
        .iter()
        .map(|v| serde_json::to_string(&v["input"]).unwrap())
        .collect();
    assert_eq!(unique_inputs.len(), cases.len());
    fs::create_dir(&out).unwrap();
    fs::write(
        out.join("case-identities.json"),
        serde_json::to_vec_pretty(&identities).unwrap(),
    )
    .unwrap();
    for (batch, chunk) in cases.chunks(32).enumerate() {
        let directory = out.join(format!("batch-{batch:03}"));
        harness::check_complete(chunk, &directory.join("complete"));
        let fitted: Vec<_> = chunk
            .iter()
            .map(|(_, _, _, _, probe, _, model)| (model.clone(), probe.clone()))
            .collect();
        harness::check_fitted(&fitted, &directory.join("fitted"));
        println!(
            "completed batch {batch}: {} exact four-lane and fitted cases",
            chunk.len()
        );
    }
    let summary = json!({"native_source_sha":source,"cases":cases.len(),"gp_winners":240,"manual_controls":identities.iter().filter(|v|v["input"]["kind"]=="manual-control").count(),"actual_gp_eec_phases":60,"comparison":"exact F/Q/S/P native/Rust/Python/g++17/clang++20; exact fitted P/Q g++17/clang++20","command_deadline_seconds":240,"status":"actual emitted source replay passed; lifecycle/promotion decision separate"});
    fs::write(
        out.join("summary.json"),
        serde_json::to_vec_pretty(&summary).unwrap(),
    )
    .unwrap();
    println!("{summary}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn committed_cohort_is_admitted_by_entrypoint_guard() {
        let bytes = include_bytes!("../tests/fixtures/scalar_v2/utility/manifest.json");
        let admitted = cohort_manifest(bytes);
        assert_eq!(admitted["datasets"].as_array().unwrap().len(), 8);
        let mut changed = bytes.to_vec();
        changed.push(b'\n');
        assert!(std::panic::catch_unwind(|| cohort_manifest(&changed)).is_err());
    }

    fn events() -> Value {
        serde_json::from_str(include_str!(
            "../tests/fixtures/scalar_v2/producer-eec-events.json"
        ))
        .unwrap()
    }

    #[test]
    fn captured_phases_replay_the_actual_gp_mutations() {
        let fixture = events();
        let events = fixture["events"].as_array().unwrap();
        assert_eq!(events.len(), 20);
        let mut phases = 0;
        for event in events {
            let (_, replayed) = replay_eec_event(event);
            for (phase, candidate) in replayed {
                let encoded = candidate.program().to_bytes().unwrap();
                assert_eq!(encoded, bytes(&event[phase]));
                assert_eq!(
                    ScalarExpressionProgram::from_bytes(&encoded)
                        .unwrap()
                        .to_bytes()
                        .unwrap(),
                    encoded
                );
                phases += 1;
            }
        }
        assert_eq!(phases, 60);
    }

    #[test]
    fn forged_phase_bytes_and_seed_refuse_before_emission() {
        let event = events()["events"][0].clone();
        replay_eec_event(&event);
        for phase in ["created_program", "changed_program", "removed_program"] {
            let mut forged = event.clone();
            forged[phase][0] = json!(0);
            assert!(
                std::panic::catch_unwind(|| replay_eec_event(&forged)).is_err(),
                "forged {phase}"
            );
        }
        let mut forged = event;
        forged["create_seed"] = json!(0);
        assert!(std::panic::catch_unwind(|| replay_eec_event(&forged)).is_err());
    }
}
