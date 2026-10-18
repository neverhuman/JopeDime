use jope_policy::{
    config::lineage::{validate_receipt_v2, EXPORT_RECEIPT_V2_SCHEMA},
    ACCEPTED_PARAMETER_SHA256, ACCEPTED_STATE_SHA256, REJECTED_STATE_SHA256,
};
use serde_json::{json, Value};

fn receipt(step: u64, state: &str) -> Value {
    json!({
        "schema": EXPORT_RECEIPT_V2_SCHEMA,
        "checkpoint": {
            "step": step,
            "state_sha256": state,
            "model_sha256": "1".repeat(64),
            "config_sha256": "2".repeat(64),
            "parameter_sha256": "3".repeat(64),
            "tensor_inventory_sha256": "4".repeat(64),
            "tensors": [{"name":"decoder.weight","dtype":"F32","shape":[2,2],"finite":true,"sha256":"5".repeat(64)}]
        },
        "lineage": {
            "accepted_anchor_state_sha256": ACCEPTED_STATE_SHA256,
            "parent_state_sha256": ACCEPTED_STATE_SHA256,
            "ancestors": [{"state_sha256":ACCEPTED_STATE_SHA256,"parameter_sha256":ACCEPTED_PARAMETER_SHA256}]
        },
        "delta": {
            "committed_updates": step,
            "optimizer_epoch_sha256": "6".repeat(64),
            "parent_parameter_sha256": ACCEPTED_PARAMETER_SHA256,
            "child_parameter_sha256": "3".repeat(64)
        }
    })
}

#[test]
fn verified_3000_and_6000_descendants_are_valid_challengers() {
    for step in [3000, 6000] {
        let value = receipt(step, &format!("{:064x}", step));
        validate_receipt_v2(&serde_json::to_vec(&value).unwrap()).unwrap();
    }
}

#[test]
fn high_step_and_rejected_checkpoint_are_refused() {
    let high = receipt(70000, &"7".repeat(64));
    assert!(validate_receipt_v2(&serde_json::to_vec(&high).unwrap()).is_err());
    let rejected = receipt(3000, REJECTED_STATE_SHA256);
    assert!(validate_receipt_v2(&serde_json::to_vec(&rejected).unwrap()).is_err());
}

#[test]
fn ancestry_and_delta_must_bind_the_accepted_parent() {
    let mut value = receipt(3000, &"7".repeat(64));
    value["delta"]["parent_parameter_sha256"] = json!("8".repeat(64));
    assert!(validate_receipt_v2(&serde_json::to_vec(&value).unwrap()).is_err());

    let mut value = receipt(6000, &"9".repeat(64));
    value["lineage"]["accepted_anchor_state_sha256"] = json!("a".repeat(64));
    assert!(validate_receipt_v2(&serde_json::to_vec(&value).unwrap()).is_err());
}

#[test]
fn malformed_tensor_inventory_is_refused() {
    let mut value = receipt(3000, &"7".repeat(64));
    value["checkpoint"]["tensors"][0]["finite"] = json!(false);
    assert!(validate_receipt_v2(&serde_json::to_vec(&value).unwrap()).is_err());

    let mut value = receipt(3000, &"7".repeat(64));
    value["checkpoint"]["tensors"][0]["dtype"] = json!("F64");
    assert!(validate_receipt_v2(&serde_json::to_vec(&value).unwrap()).is_err());
}

#[test]
fn unknown_v2_fields_are_refused() {
    let mut value = receipt(3000, &"7".repeat(64));
    value["checkpoint"]["unrecognized"] = json!(true);
    assert!(validate_receipt_v2(&serde_json::to_vec(&value).unwrap()).is_err());
}
