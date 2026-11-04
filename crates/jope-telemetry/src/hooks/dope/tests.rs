use super::*;

fn digest(name: &str, val: f32) -> TensorDigest {
    TensorDigest::new(name.into(), "<f4", &[1], &val.to_le_bytes())
}

fn obs(snapshot: Snapshot) -> Observation {
    Observation {
        snapshot,
        counts: param_counts(
            &[
                NamedParam {
                    name: "dope_model.w".into(),
                    numel: 1,
                    requires_grad: false,
                },
                NamedParam {
                    name: "head.w".into(),
                    numel: 4,
                    requires_grad: true,
                },
                NamedParam {
                    name: "scale".into(),
                    numel: 1,
                    requires_grad: true,
                },
            ],
            5,
            1,
        ),
        dope_params_in_optimizer: 0,
        dope_grads_present: 0,
        training_submodules: Vec::new(),
    }
}

#[test]
fn digest_matches_ns3_layout() {
    // sha256(b"<f4:[2, 3]:" + bytes) as hashlib computes it.
    let bytes = [0u8; 24];
    let mut h = Sha256::new();
    h.update(b"<f4:[2, 3]:");
    h.update(bytes);
    assert_eq!(tensor_le_sha256("<f4", &[2, 3], &bytes), hex(h.finalize()));
    assert_eq!(shape_repr(&[]), "[]");
    assert_eq!(shape_repr(&[7]), "[7]");
}

#[test]
fn snapshot_orders_and_prefixes_like_ns3() {
    let a = snapshot(
        vec![digest("b", 1.0), digest("a", 2.0)],
        vec![digest("running", 0.0)],
        vec![
            digest("sd_dm24", 1.0),
            digest("lm80_mu", 1.0),
            digest("other", 1.0),
        ],
    )
    .unwrap();
    let keys: Vec<&String> = a.tensor_hashes.keys().collect();
    assert_eq!(
        keys,
        [
            "dope_model.a",
            "dope_model.b",
            "dope_model.buffer.running",
            "normalizer.lm80_mu",
            "normalizer.sd_dm24"
        ]
    );
    let mut hp = Sha256::new();
    hp.update(format!("dope_model.a={}\n", digest("a", 2.0).sha256).as_bytes());
    hp.update(format!("dope_model.b={}\n", digest("b", 1.0).sha256).as_bytes());
    assert_eq!(a.params_sha256, hex(hp.finalize()));
    assert_eq!((a.num_param_tensors, a.num_buffer_tensors), (2, 3));
    assert!(snapshot(vec![], vec![], vec![]).is_err());
}

#[test]
fn guard_fails_closed_on_every_broken_invariant() {
    let mut g = DopeGuard::new(100);
    let s0 = snapshot(vec![digest("w", 1.0)], vec![], vec![]).unwrap();
    let mut rec = g.observe("run_start", 0, false, obs(s0), 1.0);
    assert!(enforce(&mut rec).is_empty());
    assert_eq!(rec.get("hashes_equal"), Some(&J::Null));
    let s1 = snapshot(vec![digest("w", 1.5)], vec![], vec![]).unwrap();
    let mut o = obs(s1);
    o.dope_grads_present = 2;
    o.training_submodules = vec!["".into(), "drop".into()];
    let mut rec = g.observe("epoch_end", 1, true, o, 1.0);
    let reasons = enforce(&mut rec);
    assert_eq!(reasons[0], "DOPE hash changed: dope_model.w");
    assert!(reasons.contains(&"2 DOPE parameters hold a .grad".to_owned()));
    assert!(reasons.contains(&"DOPE submodules in train mode: ['', 'drop']".to_owned()));
    assert_eq!(rec.get("trainable_jope_params"), Some(&J::Int(5)));
    assert!(DopeGuard::step_record(5, 0, vec![]).is_none());
    let mut step = DopeGuard::step_record(5, 1, vec![]).unwrap();
    assert_eq!(enforce(&mut step), vec!["1 DOPE parameters hold a .grad"]);
    assert!(g.should_sample(200) && !g.should_sample(201));
}

#[test]
fn counts_root_params_and_reconcile() {
    let c = param_counts(
        &[NamedParam {
            name: "bias".into(),
            numel: 3,
            requires_grad: true,
        }],
        3,
        1,
    );
    assert_eq!(c.by_submodule, vec![(ROOT_GROUP.to_owned(), 3, 0)]);
    assert!(!c.reconciled);
}
