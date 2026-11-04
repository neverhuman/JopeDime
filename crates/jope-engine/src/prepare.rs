use crate::manifest::*;
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

mod expansion;
mod folds;
mod reconstruction;
mod records;
mod run;
mod table;
mod views;

use reconstruction::{
    documented_entity_columns, entity_documentation, entity_values,
    reconstruct_classification_with, reconstruct_recorded, reviewed_entity_columns,
};

pub use expansion::{expand_cuts, index_existing};
pub use folds::{cap_indices, folds, folds_with_seed};
pub use reconstruction::{reconstruct_classification, reconstruct_numeric};
pub use records::{Cut, Prepared};
pub use run::{run, run_with_admission};
pub use table::{csv_bytes, missing, Column, Preprocessing, Table};
pub use views::{
    check_cut_membership, ensure_fitted_on_membership, load_bound_contained_view,
    load_bound_view_in_root, load_raw_view, load_raw_view_in_root, load_view, load_view_in_root,
    reject_target_copies, validate_view_source, workspace_root, ViewSource,
};

#[cfg(test)]
mod grouping_review_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn reviewed_grouping_binds_frozen_identity_and_rejects_holds_and_target_columns() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        fs::write(root.join("train.csv"), "0,0\n1,1\n").unwrap();
        fs::write(root.join("evidence.html"), "Documented repeated subjects").unwrap();
        let metadata = json!({"source_platform":"rdatasets","source_dataset_id":"example/repeated","feature_columns":["subject","x"]});
        let dataset: Dataset = serde_json::from_value(json!({"id":"example","task":"regression","source_dir":root,"source_key":"rdatasets:example/repeated","family":"example","parents":[],"admission":"approved","reason":"test fixture","evidence":[],"metadata_sha256":"a".repeat(64),"train":null,"test":null,"split_feature_overlap":0,"issues":[],"meta_partition":"meta_train"})).unwrap();
        let reconstruction = json!({"source_sha256":"b".repeat(64),"source_row_indices":[1,2]});
        let mut review = json!({"schema":"parallel.provenance_review.v2","datasets":[{"id":"example","source_key":"rdatasets:example/repeated","metadata":{"sha256":dataset.metadata_sha256},"reconstruction":{"metadata_sha256":dataset.metadata_sha256,"source_sha256":reconstruction["source_sha256"],"training_sha256":file_hash(&root.join("train.csv")).unwrap(),"row_mapping_sha256":hash(&serde_json::to_vec(&reconstruction["source_row_indices"]).unwrap())},"pilot_candidate_status":"eligible_for_fresh_cohort_preparation","pilot_hold_reason":null,"entity_columns":["subject"],"documentation":{"path":root.join("evidence.html"),"sha256":file_hash(&root.join("evidence.html")).unwrap(),"source_url":"https://example.test/source","finding":"Documented repeated subjects"}}]});
        let path = root.join("review.json");
        write_json(&path, &review).unwrap();
        let sha = file_hash(&path).unwrap();
        let (columns, binding) =
            reviewed_entity_columns(&path, &sha, &dataset, &metadata, &reconstruction).unwrap();
        assert_eq!(columns, ["subject"]);
        assert_eq!(binding["manifest_sha256"], sha);
        assert!(reviewed_entity_columns(
            &path,
            &"0".repeat(64),
            &dataset,
            &metadata,
            &reconstruction
        )
        .unwrap_err()
        .to_string()
        .contains("frozen family registry"));
        let changed_mapping = json!({"source_sha256":"b".repeat(64),"source_row_indices":[2,1]});
        assert!(
            reviewed_entity_columns(&path, &sha, &dataset, &metadata, &changed_mapping)
                .unwrap_err()
                .to_string()
                .contains("identity mismatch")
        );
        review["datasets"][0]["pilot_hold_reason"] = json!("unresolved_entity_mapping");
        write_json(&path, &review).unwrap();
        assert!(reviewed_entity_columns(
            &path,
            &file_hash(&path).unwrap(),
            &dataset,
            &metadata,
            &reconstruction
        )
        .unwrap_err()
        .to_string()
        .contains("withholds"));
        review["datasets"][0]["pilot_hold_reason"] = Value::Null;
        review["datasets"][0]["entity_columns"] = json!(["target"]);
        write_json(&path, &review).unwrap();
        assert!(reviewed_entity_columns(
            &path,
            &file_hash(&path).unwrap(),
            &dataset,
            &metadata,
            &reconstruction
        )
        .unwrap_err()
        .to_string()
        .contains("input features"));
    }
    #[test]
    fn explicit_backing_root_preserves_original_rows_and_refuses_fallback() {
        let parent = tempfile::tempdir().unwrap();
        let source = parent.path().join("source");
        let restored = source.join("recovery/restored");
        fs::create_dir_all(source.join("manifests")).unwrap();
        fs::write(source.join("Cargo.toml"), b"source marker").unwrap();
        fs::create_dir_all(&restored).unwrap();
        let original = b"x,target\n1,10\n2,20\n3,30\n";
        fs::write(source.join("table.csv"), original).unwrap();
        let view = ViewSource {
            source: "table.csv".into(),
            source_sha256: hash(original),
            feature_columns: vec!["x".into()],
            target_column: "target".into(),
            original_training_row_map: vec![2, 0],
        };
        // An outer checkout has valid bytes, but cannot satisfy the explicit relocated root.
        assert!(load_raw_view_in_root(&restored, &view, &[0, 1]).is_err());
        fs::write(restored.join("table.csv"), original).unwrap();
        assert_eq!(
            load_raw_view_in_root(&restored, &view, &[0, 1]).unwrap(),
            vec![
                vec!["3".to_string(), "30".to_string()],
                vec!["1".to_string(), "10".to_string()]
            ]
        );
        fs::write(restored.join("table.csv"), b"x,target\n1,99\n").unwrap();
        assert!(load_raw_view_in_root(&restored, &view, &[0]).is_err());
        fs::remove_file(restored.join("table.csv")).unwrap();
        std::os::unix::fs::symlink(source.join("table.csv"), restored.join("table.csv")).unwrap();
        assert!(load_raw_view_in_root(&restored, &view, &[0]).is_err());
    }
}

/// W1 consumer-map route R8: an object is consumed only as the bytes whose
/// digest was verified. The instrumentation hook replaces the file on disk
/// immediately after verification; a receiver that reopens the path would
/// consume the replacement.
#[cfg(test)]
mod verified_bytes_tests {
    use super::*;
    use crate::manifest::verification_hook;
    use serde_json::json;

    /// Replace `path` (new inode, via rename) the first time the hook fires for it.
    fn replace_after_verification(target: &Path, bytes: &'static [u8]) -> verification_hook::Guard {
        let target = target.to_path_buf();
        let mut done = false;
        verification_hook::install(move |path| {
            if !done && path == target {
                done = true;
                let staged = target.with_extension("replacement");
                fs::write(&staged, bytes).unwrap();
                fs::rename(&staged, &target).unwrap();
            }
        })
    }

    #[test]
    fn backing_bytes_replaced_after_verification_are_never_consumed() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().canonicalize().unwrap();
        let original: &[u8] = b"x,target\n1,10\n2,20\n3,30\n";
        fs::write(root.join("table.csv"), original).unwrap();
        let view = ViewSource {
            source: "table.csv".into(),
            source_sha256: hash(original),
            feature_columns: vec!["x".into()],
            target_column: "target".into(),
            original_training_row_map: vec![0, 1, 2],
        };
        let expected = load_raw_view_in_root(&root, &view, &[0, 1, 2]).unwrap();
        let _hook =
            replace_after_verification(&root.join("table.csv"), b"x,target\n1,99\n2,98\n3,97\n");
        let consumed = load_raw_view_in_root(&root, &view, &[0, 1, 2]);
        assert_eq!(
            consumed.ok(),
            Some(expected),
            "the receiver consumed bytes other than the verified backing table"
        );
    }

    #[test]
    fn preprocessing_replaced_after_verification_is_never_consumed() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().canonicalize().unwrap();
        let raw = (0..10)
            .map(|i| vec![i.to_string(), (100 + 3 * i).to_string()])
            .collect::<Vec<_>>();
        let mut csv = "x,target\n".to_string();
        for row in &raw {
            csv.push_str(&format!("{},{}\n", row[0], row[1]));
        }
        fs::write(root.join("raw.csv"), &csv).unwrap();
        let pp = Preprocessing::fit(&raw[..8], "regression").unwrap();
        let prepared = root.join("prepared");
        fs::create_dir_all(prepared.join("cut-0")).unwrap();
        let pp_path = prepared.join("cut-0/preprocessing.json");
        write_json(&pp_path, &pp).unwrap();
        let tr = csv_bytes(&pp.transform(&raw[..8]).unwrap()).unwrap();
        let va = csv_bytes(&pp.transform(&raw[8..]).unwrap()).unwrap();
        write_json(
            &prepared.join("prepared.json"),
            &json!({"schema":"parallel.prepared.v1","dataset":"d","family":"f","task":"regression",
                "meta_partition":"meta_train","seed":SEED,"clean_evaluation":true,"clean_status":"raw_reconstructed",
                "original_train_sha256":"0".repeat(64),"original_test_sha256":"1".repeat(64),
                "selected_train_indices":(0..10).collect::<Vec<_>>(),"selected_test_indices":[],"target_last":true,
                "rows_total":10,"cuts":[{"index":0,"train_indices":(0..8).collect::<Vec<_>>(),"validation_indices":[8,9],
                    "training_sha256":hash(&tr),"validation_sha256":hash(&va),
                    "preprocessing_sha256":file_hash(&pp_path).unwrap()}],
                "storage":{"source":"raw.csv","source_sha256":hash(csv.as_bytes()),"feature_columns":["x"],
                    "target_column":"target","original_training_row_map":(0..10).collect::<Vec<_>>()}}),
        )
        .unwrap();
        let expected = load_view_in_root(&root, &prepared, 0, true).unwrap();
        // A well-formed but different state (fitted on every row) replaces the
        // verified file after its digest check.
        let other =
            serde_json::to_vec_pretty(&Preprocessing::fit(&raw, "regression").unwrap()).unwrap();
        let _hook = replace_after_verification(&pp_path, Box::leak(other.into_boxed_slice()));
        let consumed = load_view_in_root(&root, &prepared, 0, true);
        assert_eq!(
            consumed.map_err(|e| format!("{e:#}")),
            Ok(expected),
            "the view receiver consumed preprocessing other than the verified state"
        );
    }
}
