#[cfg(test)]
mod campaign_identity_tests {
    use super::*;

    #[test]
    fn completed_cycle_rejects_changed_inputs_without_rewriting_receipts() {
        struct Environment(Option<std::ffi::OsString>);
        impl Drop for Environment {
            fn drop(&mut self) {
                match &self.0 {
                    Some(v) => std::env::set_var("JOPE_GP_EXECUTABLE", v),
                    None => std::env::remove_var("JOPE_GP_EXECUTABLE"),
                }
            }
        }
        let _environment = Environment(std::env::var_os("JOPE_GP_EXECUTABLE"));
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path();
        let evaluator = root.join("evaluator");
        let corpus = root.join("corpus");
        let cycle = root.join("cycle");
        let checkpoint = root.join("checkpoint");
        let contexts = root.join("contexts.json");
        for path in [&corpus, &cycle, &checkpoint] {
            fs::create_dir(path).unwrap();
        }
        for path in [
            &evaluator,
            &contexts,
            &corpus.join("index.json"),
            &checkpoint.join("state.json"),
        ] {
            fs::write(path, b"frozen input").unwrap();
        }
        std::env::set_var("JOPE_GP_EXECUTABLE", &evaluator);
        let snapshot = json!({"schema":"parallel.champion_snapshot.v1",
            "request":{"corpus_sha256":file_hash(&corpus.join("index.json")).unwrap(),"cuts":2,"cut_start":10},
            "evaluator_sha256":"e","vocabulary_sha256":"v","entries":[]});
        write_json(&cycle.join("champion-snapshot.json"), &snapshot).unwrap();
        write_json(
            &cycle.join("champion-snapshot-binding.json"),
            &json!({"sha256":file_hash(&cycle.join("champion-snapshot.json")).unwrap()}),
        )
        .unwrap();
        let observations = vec![root.join("observations")];
        let verify = |existing| {
            verify_cycle_identity(
                root,
                &corpus,
                &cycle,
                300,
                2,
                10,
                Some(&contexts),
                Some(&checkpoint),
                &observations,
                existing,
            )
        };
        verify(false).unwrap();
        let mut proposal = json!({"checkpoint_sha256":"frozen"});
        controller::bind_proposal_protocol(&mut proposal);
        for arm in ["neural", "shuffled"] {
            for name in ["run.json", "complete.json"] {
                write_json(&cycle.join(arm).join(name), &proposal).unwrap();
            }
        }
        fs::write(cycle.join("complete.json"), b"already complete").unwrap();
        let identities = ["cycle.json", "predictor-evaluator.json", "complete.json"]
            .map(|name| (name, fs::read(cycle.join(name)).unwrap()));
        verify(true).unwrap();
        for input in [
            &evaluator,
            &contexts,
            &corpus.join("index.json"),
            &checkpoint.join("state.json"),
        ] {
            fs::write(input, b"changed input").unwrap();
            assert!(verify(true).is_err());
            fs::write(input, b"frozen input").unwrap();
        }
        assert!(verify_cycle_identity(
            root,
            &corpus,
            &cycle,
            301,
            2,
            10,
            Some(&contexts),
            Some(&checkpoint),
            &observations,
            true
        )
        .is_err());
        assert!(verify_cycle_identity(
            root,
            &corpus,
            &cycle,
            300,
            2,
            12,
            Some(&contexts),
            Some(&checkpoint),
            &observations,
            true
        )
        .is_err());
        assert!(verify_cycle_identity(
            root,
            &corpus,
            &cycle,
            300,
            2,
            10,
            Some(&contexts),
            Some(&checkpoint),
            &[],
            true
        )
        .is_err());
        let current: Value = read_json(&cycle.join("cycle.json")).unwrap();
        for field in [
            "sampling_protocol",
            "pure_prediction_protocol",
            "gp_assembly_protocol",
            "proposal_generation_schema",
        ] {
            let mut legacy = current.clone();
            legacy.as_object_mut().unwrap().remove(field);
            write_json(&cycle.join("cycle.json"), &legacy).unwrap();
            let before = fs::read(cycle.join("cycle.json")).unwrap();
            assert!(verify(true)
                .unwrap_err()
                .to_string()
                .contains("cycle resume specification changed"));
            assert_eq!(fs::read(cycle.join("cycle.json")).unwrap(), before);
            write_json(&cycle.join("cycle.json"), &current).unwrap();
        }
        for (name, bytes) in identities {
            assert_eq!(fs::read(cycle.join(name)).unwrap(), bytes);
        }
    }

    #[test]
    fn completed_cycle_rejects_legacy_proposal_before_reusing_any_outputs() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path();
        let output = root.join("cycle");
        for arm in ["neural", "shuffled"] {
            write_json(
                &output.join(arm).join("run.json"),
                &json!({"sampling_protocol":"shared-incumbents-and-retained-v3"}),
            )
            .unwrap();
            write_json(&output.join(arm).join("complete.json"), &json!({"world":8})).unwrap();
        }
        fs::write(output.join("complete.json"), b"legacy completion").unwrap();
        for require_existing in [false, true] {
            let error = verify_cycle_identity(
                root,
                &root.join("absent-corpus"),
                &output,
                3000,
                2,
                0,
                None,
                None,
                &[],
                require_existing,
            )
            .unwrap_err();
            assert!(error
                .to_string()
                .contains("proposal prediction protocol changed"));
        }
        assert_eq!(
            fs::read(output.join("complete.json")).unwrap(),
            b"legacy completion"
        );
        assert!(!output.join("cycle.json").exists());
        assert!(!output.join("predictor-evaluator.json").exists());
    }
}
