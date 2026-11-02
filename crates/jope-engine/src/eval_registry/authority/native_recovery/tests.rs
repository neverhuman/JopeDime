use super::*;

#[cfg(test)]
mod tests {
    use super::*;
    fn attempt() -> FencedAttempt {
        serde_json::from_value(
            json!({"job_id":jain_smartcluster::JobId::new(),"number":1,"fence":1}),
        )
        .unwrap()
    }
    #[test]
    fn original_preclaim_fences_cannot_be_replaced_by_same_process_or_historical_assertion() {
        let process = ProcessIdentity::current().unwrap();
        let custody = OriginalCustody {
            authority_attempt: attempt(),
            target_attempt: attempt(),
            authority_leader: process.clone(),
            target_leader: process.clone(),
            requesting_process: process.clone(),
        };
        let row = json!({"owner_json":serde_json::to_string(&process).unwrap(),"state":"claimed"});
        let request = json!({"details":{"requesting_process":process,"native_allocation":{"schema":"jope.native_claim_allocation.v1",
            "authority_attempt":custody.authority_attempt,"target_attempt":custody.target_attempt,
            "authority_leader":custody.authority_leader,"target_leader":custody.target_leader}}});
        let mut binding = HeldBinding {
            row_sha256: hash(&encoded(&row).unwrap()),
            request_sha256: hash(&encoded(&request).unwrap()),
            custody,
            historical_custody: None,
        };
        check_binding("key", &row, &request, &binding).unwrap();
        binding.custody.target_attempt = attempt();
        assert!(check_binding("key", &row, &request, &binding).is_err());
        let mut altered = row.clone();
        altered["state"] = json!("running");
        assert!(check_binding("key", &altered, &request, &binding).is_err());
    }
    #[test]
    fn original_live_process_blocks_recovery_but_distinct_pid_birth_is_not_the_original() {
        assert!(process_gone(&ProcessIdentity::current().unwrap()).is_err());
        let mut original = ProcessIdentity::current().unwrap();
        original.birth_ticks += 1;
        process_gone(&original).unwrap();
    }
    #[test]
    fn operator_writable_self_signed_policy_is_not_installed_authorization() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("policy.json");
        fs::write(&path, b"{}").unwrap();
        assert!(load_policy(&path, 0).is_err());
    }
}

#[cfg(test)]
#[path = "../native_recovery_tests.rs"]
mod signature_tests;
