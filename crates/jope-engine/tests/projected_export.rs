//! W1 follow-up 2c: the fitted-view export worker (`fit-memory`, hidden)
//! receives its projected fitting input as sealed memory and opens no
//! prepared path. The shipped binary still fences every fit until a
//! production namespace verifier exists; the positive export through the
//! fixture authority is
//! `gp::fit_contract_tests::the_export_worker_receives_its_projected_input_and_never_resolves_the_view_itself`.

use jope_parallel::{
    gp,
    manifest::hash,
    prepare::{csv_bytes, Preprocessing},
};
use std::os::fd::AsRawFd;
#[path = "support/fit_authority.rs"]
mod authority;

#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
fn shipped_fit_memory_is_fenced_and_refuses_a_mismatched_or_unsealed_projection() {
    let tmp = tempfile::tempdir().unwrap();
    let raw = (0..12)
        .map(|i| vec![i.to_string(), (100 + 3 * i).to_string()])
        .collect::<Vec<_>>();
    let pp = Preprocessing::fit(&raw, "regression").unwrap();
    let rows = pp.transform(&raw).unwrap();
    let pp_bytes = serde_json::to_vec_pretty(&pp).unwrap();
    let memory = gp::fit_memory_input(&rows, &pp_bytes).unwrap();
    let training = hash(&csv_bytes(&rows).unwrap());
    let preprocessing = hash(&pp_bytes);
    let mut authority = authority::Fixture::new(tmp.path(), vec![authority::training_hash(&rows)]);
    {
        let run = |fd: i32, training: &str, output: &std::path::Path| {
            authority
                .command()
                .args(["fit-memory", "--fd", &fd.to_string()])
                .args(["--training-sha256", training])
                .args(["--preprocessing-sha256", &preprocessing])
                .args(["--genome", authority::PROGRAM, "--output"])
                .arg(output)
                .output()
                .unwrap()
        };
        // Fully configured and in scope: still refused at namespace activation.
        let fenced = tmp.path().join("fenced.json");
        let result = run(memory.as_raw_fd(), &training, &fenced);
        let error = String::from_utf8_lossy(&result.stderr);
        assert!(
            !result.status.success(),
            "shipped fit-memory ran without activation"
        );
        assert!(error.contains(authority::UNVERIFIED_ACTIVATION), "{error}");
        assert!(!fenced.exists());
        // A projection that is not the declared cut is refused before any claim.
        let mismatched = tmp.path().join("mismatched.json");
        let result = run(memory.as_raw_fd(), &"0".repeat(64), &mismatched);
        let error = String::from_utf8_lossy(&result.stderr);
        assert!(!result.status.success());
        assert!(error.contains("training identity"), "{error}");
        assert!(!mismatched.exists());
        // An unsealed regular file is not a projected input.
        let plain = std::fs::File::create(tmp.path().join("not-memory")).unwrap();
        let unsealed = tmp.path().join("unsealed.json");
        let result = run(plain.as_raw_fd(), &training, &unsealed);
        let error = String::from_utf8_lossy(&result.stderr);
        assert!(!result.status.success());
        assert!(error.contains("sealed memory"), "{error}");
        assert!(!unsealed.exists());
        authority.assert_no_fit_starts();
    }
    let stats = authority.shutdown();
    assert_eq!(stats["worker_spawns"], 0);
    assert_eq!(stats["committed"], 0);
}
