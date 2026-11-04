use super::*;

#[test]
fn challenger_weights_load_bound_and_tampering_is_refused() {
    let dir = scratch("load");
    let (path, receipt, identity) = challenger_export(&dir, 0.0, "challenger state");
    let binding = challenger_binding(&receipt, &identity).unwrap();
    let model = PolicyModel::load_bound(&path, &binding, Device::Cpu).unwrap();
    let bound = model.bound_identity().unwrap();
    assert_eq!(bound.state_sha256(), identity.state_sha256);
    assert_eq!(bound.parameter_sha256(), identity.parameter_sha256);
    assert_eq!(
        bound.policy_safetensors_sha256(),
        identity.policy_safetensors_sha256
    );
    // One flipped bit in the last tensor's data.
    let mut bytes = std::fs::read(&path).unwrap();
    *bytes.last_mut().unwrap() ^= 0x01;
    std::fs::write(&path, &bytes).unwrap();
    let tampered = sha256_hex(&bytes);
    // (a) The original binding refuses the file.
    assert_refused(
        PolicyModel::load_bound(&path, &binding, Device::Cpu),
        "differs from its export receipt",
    );
    // (b) Receipt and identity re-pinned to the new file: the per-tensor inventory refuses it.
    let mut repinned = receipt.clone();
    repinned["policy"]["output"]["sha256"] = json!(tampered);
    let repinned_identity = ChallengerIdentity {
        policy_safetensors_sha256: tampered.clone(),
        ..identity.clone()
    };
    let binding = challenger_binding(&repinned, &repinned_identity).unwrap();
    assert_refused(
        PolicyModel::load_bound(&path, &binding, Device::Cpu),
        "differ from the export receipt inventory",
    );
    // (c) The inventory re-pinned as well: the recomputed parameter hash refuses it.
    let inventory = PolicyModel::load_safetensors(&path, tiny(), Device::Cpu)
        .unwrap()
        .tensor_inventory()
        .unwrap();
    repinned["policy"]["tensors"] = serde_json::to_value(inventory).unwrap();
    let binding = challenger_binding(&repinned, &repinned_identity).unwrap();
    assert_refused(
        PolicyModel::load_bound(&path, &binding, Device::Cpu),
        "recomputed parameter hash differs",
    );
    std::fs::remove_dir_all(&dir).unwrap();
}
