use super::*;

#[test]
fn reference_optimizer_rejects_a_nonfinite_update_without_partial_change() {
    tch::set_num_threads(1);
    let vs = nn::VarStore::new(Device::Cpu);
    // BTreeMap order: "a" is updated before "b" is reached.
    let a = vs.root().var("a", &[4], nn::Init::Const(0.5));
    let b = vs.root().var("b", &[4], nn::Init::Const(-0.25));
    let mut optimizer = AdamW::new(1e-2, 0.01);
    (a.square().sum(Kind::Float) + b.square().sum(Kind::Float)).backward();
    optimizer.update_reference(&vs).unwrap();
    assert_eq!(optimizer.step, 1);
    let parameters = snapshot(&vs);
    let state = moments(&optimizer);
    // Finite gradient on "a", NaN on "b".
    (a.sum(Kind::Float) + (&b * f64::NAN).sum(Kind::Float)).backward();
    let error = optimizer.update_reference(&vs).unwrap_err();
    assert!(error.to_string().contains("nonfinite"), "{error}");
    assert_eq!(optimizer.step, 1, "a rejected update advanced the step");
    assert_eq!(
        snapshot(&vs),
        parameters,
        "a rejected update changed parameters"
    );
    assert_eq!(
        moments(&optimizer),
        state,
        "a rejected update changed moments"
    );
    // Recovery: clear the rejected gradients; the next valid update is step 2.
    for (_, mut p) in vs.variables() {
        p.zero_grad();
    }
    (a.square().sum(Kind::Float) + b.square().sum(Kind::Float)).backward();
    optimizer.update_reference(&vs).unwrap();
    assert_eq!(optimizer.step, 2);
    // A rejected first update leaves a fresh optimizer fresh.
    let fresh = nn::VarStore::new(Device::Cpu);
    let c = fresh.root().var("c", &[2], nn::Init::Const(1.0));
    let mut first = AdamW::new(1e-2, 0.01);
    (&c * f64::INFINITY).sum(Kind::Float).backward();
    assert!(first.update_reference(&fresh).is_err());
    assert_eq!(first.step, 0);
    assert!(first.moments.is_empty());
}
