//! Fit cohort transforms on meta-training families, then persist full contexts.
use crate::{encoding::Encoding, manifest::*, prepare::Prepared, service::GpJobs};
use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};
use tch::{Device, Kind, Tensor};
mod apply;
mod assemble;
mod union;
pub use apply::{apply, apply_binary_qualification};
#[cfg(test)]
use apply::{normalize_frozen, reference_component};
pub use assemble::{assemble, assemble_registered};
#[cfg(test)]
use assemble::{registered_prepared, training_parity_rows};
pub use union::union;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registered_assembly_accepts_old_metadata_only_with_preserved_assignment() {
        let registry: crate::families::Registry = serde_json::from_value(json!({
            "schema":"parallel.family_registry.v1","seed":SEED,"registry_id":"test-registry","qualification":true,"full_inventory_complete":false,"independent_selection_eligible":false,"parent":null,"audit":{"path":"audit.json","sha256":"audit"},"decisions":[],"reviews":[],"prepared_inputs":[],"held_review_datasets":{},
            "families":{"family":{"family":"family","partition":"meta_train","assignment_method":"preserved_audit_assignment","assignment_hash":null,"bucket":null,"source_keys":["real-source"],"holds":[],"datasets":[{"dataset":"dataset","source_key":"real-source","audit_family":"family","metadata_sha256":"metadata","entity_review_sha256":null,"preparation_eligible":true,"eligibility_reason":"verified","prepared":[],"evidence":[]}]}}
        })).unwrap();
        registry.validate().unwrap();
        let mut prepared: Prepared = serde_json::from_value(json!({"schema":"parallel.prepared.v1","dataset":"dataset","family":"family","task":"regression","meta_partition":"meta_train","seed":SEED,"family_partition_frozen":false,"clean_evaluation":false,"clean_status":"qualification","original_train_sha256":"training","original_test_sha256":"sealed","selected_train_indices":[],"selected_test_indices":[],"target_last":true,"rows_total":0,"cuts":[]})).unwrap();
        assert!(prepared.family_registry_binding.is_none());
        registered_prepared(&registry, &prepared).unwrap();
        prepared.meta_partition = "meta_validation".into();
        assert!(registered_prepared(&registry, &prepared).is_err());
        prepared.meta_partition = "meta_train".into();
        let mut held = registry.clone();
        held.families
            .get_mut("family")
            .unwrap()
            .holds
            .insert("unresolved origin".into());
        assert!(registered_prepared(&held, &prepared).is_err());
        prepared.dataset = "unregistered".into();
        assert!(registered_prepared(&registry, &prepared).is_err());
    }

    #[test]
    fn parity_selection_is_training_only_bounded_and_registry_checked() {
        let mut families = serde_json::Map::new();
        let mut rows = Vec::new();
        for name in ["alpha", "beta", "gamma"] {
            families.insert(name.into(), json!({"family":name,"partition":"meta_train","assignment_method":"preserved_audit_assignment","assignment_hash":null,"bucket":null,"source_keys":[name],"holds":[],"datasets":[{"dataset":name,"source_key":name,"audit_family":name,"metadata_sha256":"metadata","entity_review_sha256":null,"preparation_eligible":true,"eligibility_reason":"verified","prepared":[],"evidence":[]}]}));
            for cut in (0..6).rev() {
                rows.push(json!({"dataset":name,"family":name,"cut":cut,"partition":"meta_train","path":"never-open-this"}));
            }
        }
        // A protected reference need not have a readable path or valid dataset.
        rows.push(json!({"partition":"meta_test","path":"protected-file-does-not-exist"}));
        rows.push(json!({"partition":"meta_validation","path":"protected-file-does-not-exist"}));
        let registry: crate::families::Registry = serde_json::from_value(json!({"schema":"parallel.family_registry.v1","seed":SEED,"registry_id":"test","qualification":true,"full_inventory_complete":false,"independent_selection_eligible":false,"parent":null,"audit":{"path":"audit","sha256":"audit"},"decisions":[],"reviews":[],"prepared_inputs":[],"held_review_datasets":{},"families":families})).unwrap();
        let original = json!({"encodings":rows});
        let selected = training_parity_rows(&original, &registry).unwrap();
        assert_eq!(selected.len(), 8);
        assert!(selected.iter().all(|row| row["partition"] == "meta_train"));
        assert_eq!(selected[0]["family"], "alpha");
        assert_eq!(selected[0]["cut"], 0);
        assert_eq!(selected[7]["family"], "beta");
        assert_eq!(selected[7]["cut"], 3);
        let mut protected = registry;
        protected.families.get_mut("alpha").unwrap().partition = "meta_test".into();
        assert!(training_parity_rows(&original, &protected).is_err());
    }

    #[test]
    fn reference_component_rejects_unbound_and_wrong_cut_before_payload_access() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let encoding: Encoding = serde_json::from_value(json!({
            "schema":"parallel.encoding.v1", "dataset":"dataset", "family":"family",
            "partition":"meta_train", "cut":2, "prepared_sha256":"prepared",
            "training_sha256":"training", "selected_training_rows_sha256":"rows",
            "preprocessing_sha256":"preprocessing", "encoder_spec_sha256":"assembly",
            "numerical_backend":"fixture", "dataset_dimensions":1536,
            "target_dimensions":512, "output_sha256":"context",
        }))
        .unwrap();
        let err = reference_component(root, &json!({}), &encoding, "target", "target.f32", 512)
            .unwrap_err();
        assert!(err.to_string().contains("unbound fallback"));
        let receipt = root.join("target.json");
        let mut meta = json!({"dataset":"dataset", "family":"family", "cut":3,
            "prepared_sha256":"prepared", "training_sha256":"training",
            "preprocessing_sha256":"preprocessing", "final_test_targets_used":false,
            "shape":[512], "spec_sha256":"spec", "signature_sha256":"unopened"});
        write_json(&receipt, &meta).unwrap();
        let mut metadata = json!({"components":{"target":{
            "receipt":"target.json", "receipt_sha256":file_hash(&receipt).unwrap(),
            "signature":"target.f32", "signature_sha256":"unopened", "spec_sha256":"spec",
        }}});
        let err = reference_component(root, &metadata, &encoding, "target", "target.f32", 512)
            .unwrap_err();
        assert!(err.to_string().contains("exact training cut"));
        assert!(!root.join("target.f32").exists());
        meta["cut"] = 2.into();
        let bytes = vec![0_u8; 512 * 4];
        fs::write(root.join("target.f32"), &bytes).unwrap();
        meta["signature_sha256"] = hash(&bytes).into();
        write_json(&receipt, &meta).unwrap();
        metadata["components"]["target"]["receipt_sha256"] = file_hash(&receipt).unwrap().into();
        metadata["components"]["target"]["signature_sha256"] = hash(&bytes).into();
        assert_eq!(
            reference_component(root, &metadata, &encoding, "target", "target.f32", 512).unwrap(),
            vec![0_f32; 512]
        );
        fs::write(root.join("target.f32"), vec![1_u8; 512 * 4]).unwrap();
        assert!(
            reference_component(root, &metadata, &encoding, "target", "target.f32", 512)
                .unwrap_err()
                .to_string()
                .contains("bytes changed")
        );
    }

    #[test]
    fn frozen_transform_applies_stored_statistics_without_refitting() {
        tch::set_num_threads(2);
        let cpu = (Kind::Float, Device::Cpu);
        let transforms = BTreeMap::from([
            (
                "projection".into(),
                Tensor::cat(
                    &[Tensor::eye(1024, cpu), Tensor::zeros([1024, 1024], cpu)],
                    1,
                ),
            ),
            ("foundation_mean".into(), Tensor::ones([2048], cpu)),
            ("hyperion_mean".into(), Tensor::ones([512], cpu)),
            ("hyperion_scale".into(), Tensor::full([512], 2., cpu)),
            ("projected_mean".into(), Tensor::ones([1024], cpu)),
            ("projected_scale".into(), Tensor::full([1024], 2., cpu)),
            ("target_mean".into(), Tensor::full([512], 3., cpu)),
            ("target_scale".into(), Tensor::full([512], 4., cpu)),
        ]);
        let output = normalize_frozen(
            &Tensor::full([2, 512], 3., cpu),
            &Tensor::full([2, 2048], 4., cpu),
            &Tensor::full([2, 512], 7., cpu),
            &transforms,
            32.,
        );
        assert_eq!(output.size(), [2, 2048]);
        assert_eq!((output - 1.).abs().max().double_value(&[]), 0.);
    }

    #[test]
    fn union_references_existing_vectors_and_rejects_changed_sources() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let mut indexes = Vec::new();
        for cut in 0..2 {
            let path = root.join(format!("saved-{cut}/cut-{cut}.f32"));
            atomic_write(&path, &vec![0u8; 2048 * 4]).unwrap();
            let encoding = Encoding {
                schema: "parallel.encoding.v1".into(),
                dataset: "real-source".into(),
                family: "source-family".into(),
                partition: "meta_train".into(),
                cut,
                prepared_sha256: "prepared".into(),
                training_sha256: "train".into(),
                selected_training_rows_sha256: "rows".into(),
                preprocessing_sha256: "pp".into(),
                encoder_spec_sha256: "frozen-assembly".into(),
                numerical_backend: "test".into(),
                dataset_dimensions: 1536,
                target_dimensions: 512,
                output_sha256: file_hash(&path).unwrap(),
            };
            write_json(&path.with_extension("json"), &encoding).unwrap();
            let index = root.join(format!("index-{cut}.json"));
            write_json(&index,&json!({"schema":"parallel.context_index.v1","assembly_sha256":"frozen-assembly","qualification":true,"dimensions":2048,"encodings":[{"dataset":encoding.dataset,"family":encoding.family,"partition":encoding.partition,"cut":cut,"path":path.strip_prefix(root).unwrap(),"sha256":encoding.output_sha256,"prepared":"prepared","prepared_sha256":"prepared"}]})).unwrap();
            indexes.push(index);
        }
        let output = root.join("union/index.json");
        union(root, &indexes, &output).unwrap();
        let merged: Value = read_json(&output).unwrap();
        assert_eq!(merged["encodings"].as_array().unwrap().len(), 2);
        assert_eq!(merged["copied_vector_bytes"], 0);
        assert_eq!(fs::read_dir(output.parent().unwrap()).unwrap().count(), 1);
        assert_eq!(merged["assembly_sha256"], "frozen-assembly");
        fs::write(root.join("saved-1/cut-1.f32"), b"changed").unwrap();
        assert!(union(root, &indexes, &root.join("changed.json")).is_err());
    }
}
