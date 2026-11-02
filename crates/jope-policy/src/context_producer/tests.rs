use super::*;

/// Crate-internal unit-test fixtures: synthetic products and artifacts, never accepted output.
#[cfg(test)]
pub(crate) mod test_support {
    use super::*;
    /// Input scope of the synthetic products below.
    pub const SCOPE: &str = "3333333333333333333333333333333333333333333333333333333333333333";
    pub fn components(hyperion: Vec<f32>, foundation: Vec<f32>, target: Vec<f32>) -> Components {
        Components {
            hyperion: HyperionSignature::for_testing(hyperion, SCOPE),
            foundation: RawFoundation::for_testing(foundation, SCOPE),
            target: TargetBlock::for_testing(target, SCOPE),
        }
    }
    /// A producer-shaped artifact whose context is the assembly of its components under
    /// `transforms`, with the accepted encoder identity.
    pub fn synthetic_artifact(transforms: &FrozenTransforms) -> CpuContextArtifact {
        let components = components(
            (0..512).map(|j| j as f32 / 64.0).collect(),
            (0..2048).map(|m| m as f32 * 0.01).collect(),
            (0..512).map(|j| -(j as f32) / 64.0).collect(),
        );
        let context = transforms.assemble(&components).unwrap();
        let preprocessing = TrainingPreprocessing {
            schema: "synthetic".into(),
            columns: vec![],
            target_mean: 0.0,
            target_scale: 1.0,
        };
        let artifact = CpuContextArtifact {
            schema: SCHEMA.into(),
            backend: BACKEND.into(),
            qualification: "unqualified".into(),
            source_sha256: "e".repeat(40),
            producer_sha256: "f".repeat(64),
            input_sha256: "1".repeat(64),
            training: TrainingIdentity {
                dataset_sha256: "a".repeat(64),
                prepared_sha256: "b".repeat(64),
                training_sha256: "c".repeat(64),
                fitting_rows_sha256: "d".repeat(64),
                cut: 0,
            },
            selected_training_rows_sha256: "d".repeat(64),
            original_labels_sha256: "2".repeat(64),
            preprocessing_sha256: sha(&serde_json::to_vec(&preprocessing).unwrap()),
            prepared_rows_sha256: components.scope().into(),
            preprocessing,
            encoder: EncoderIdentity::accepted(),
            components: components.to_vectors(),
            vector: context.values().to_vec(),
            final_test_targets_used: false,
        };
        artifact.validate().unwrap();
        artifact
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::*;
    use super::*;
    #[test]
    fn accepted_manifest_uses_sha256_bytes_not_legacy_identifier() {
        let mut manifest = AssetFile {
            path: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/accepted-hyperion-manifest.json"),
            sha256: HYPERION_MANIFEST.into(),
        };
        let (by_path, legacy_id) =
            hyperion::hyperion::load_settings(&manifest.path, HYPERION_SEED).unwrap();
        assert_eq!(legacy_id.len(), 16);
        assert_ne!(legacy_id, manifest.sha256);
        let settings = verified_hyperion_settings(&manifest).unwrap();
        assert_eq!(settings.emb_dim, 128);
        assert_eq!(settings.seed, HYPERION_SEED);
        // Parsing the authenticated bytes yields exactly the path loader's settings.
        assert_eq!(settings, by_path);
        manifest.sha256 = "0".repeat(64);
        assert!(verified_hyperion_settings(&manifest).is_err());
    }
    fn pooled(flat: Vec<f32>) -> HyperionSignature {
        let rows = flat.len() / RowEmbeddings::WIDTH;
        HyperionSignature::pool(&RowEmbeddings::for_testing(flat, rows, SCOPE)).unwrap()
    }
    #[test]
    fn assembly_uses_all_three_frozen_normalizers_including_target() {
        let mut tensors = BTreeMap::new();
        for (name, n, value) in [
            ("foundation_mean", 2048, 0.0f32),
            ("hyperion_mean", 512, 1.0),
            ("hyperion_scale", 512, 2.0),
            ("projected_mean", 1024, 3.0),
            ("projected_scale", 1024, 4.0),
            ("target_mean", 512, 5.0),
            ("target_scale", 512, 6.0),
        ] {
            tensors.insert(
                name.into(),
                Tensor::from_vec(vec![value; n], n, &Device::Cpu).unwrap(),
            );
        }
        tensors.insert(
            "projection".into(),
            Tensor::zeros((1024, 2048), DType::F32, &Device::Cpu).unwrap(),
        );
        let result = FrozenTransforms { tensors }
            .assemble(&components(
                vec![3.0; 512],
                vec![1.0; 2048],
                vec![17.0; 512],
            ))
            .unwrap();
        assert_eq!(&result[..512], &vec![1.0; 512]);
        assert_eq!(&result[512..1536], &vec![-0.75; 1024]);
        assert_eq!(&result[1536..], &vec![2.0; 512]);
        assert_eq!(result.scope(), SCOPE);
    }
    #[test]
    fn hyperion_pool_keeps_512_component_contract() {
        let mut flat = vec![1.0; 128];
        flat.extend(vec![3.0; 128]);
        let pooled = pooled(flat);
        assert_eq!(&pooled[..128], &vec![2.0; 128]);
        assert_eq!(&pooled[128..256], &vec![1.0; 128]);
        assert_eq!(pooled[256], 1.2);
        assert_eq!(pooled[384], 2.8);
        assert_eq!(pooled.scope(), SCOPE);
    }

    fn close(actual: f32, expected: f64, what: &str) {
        assert!(
            (actual as f64 - expected).abs() <= 1e-5 * expected.abs().max(1.0),
            "{what}: {actual} vs {expected}"
        );
    }

    /// W2-T05 golden: one column is computed by hand, and every other column by independent
    /// f64 arithmetic below (not by `HyperionSignature::pool`). Rows are given out of order, so
    /// the percentiles only match if values are sorted first.
    #[test]
    fn golden_hyperion_pooling_uses_population_std_and_interpolated_percentiles() {
        let rows = [10.0f32, 1.0, 3.0, 2.0, 4.0];
        let value = |row: usize, j: usize| rows[row] * (1.0 + j as f32 / 64.0) - j as f32 / 8.0;
        let flat = (0..rows.len())
            .flat_map(|row| (0..128).map(move |j| value(row, j)))
            .collect::<Vec<_>>();
        let pooled = pooled(flat);
        // Column 0 by hand, values {1, 2, 3, 4, 10}: mean 4; population variance 50/5 = 10
        // (the sample variance would be 12.5); p10 at rank 0.4 is 1.4; p90 at rank 3.6 is 7.6.
        close(pooled[0], 4.0, "mean");
        close(pooled[128], 10f64.sqrt(), "population std");
        assert!(
            (pooled[128] as f64 - 12.5f64.sqrt()).abs() > 0.3,
            "sample std would differ"
        );
        close(pooled[256], 1.4, "p10");
        close(pooled[384], 7.6, "p90");
        for j in 0..128 {
            let mut v = (0..rows.len())
                .map(|row| value(row, j) as f64)
                .collect::<Vec<_>>();
            v.sort_by(f64::total_cmp);
            let mean = v.iter().sum::<f64>() / 5.0;
            let var = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / 5.0;
            close(pooled[j], mean, "mean block");
            close(pooled[128 + j], var.sqrt(), "std block");
            close(pooled[256 + j], 0.6 * v[0] + 0.4 * v[1], "p10 block");
            close(pooled[384 + j], 0.4 * v[3] + 0.6 * v[4], "p90 block");
        }
        // Row alignment: the same values laid out column-major pool differently.
        let transposed = (0..128)
            .flat_map(|j| (0..rows.len()).map(move |row| value(row, j)))
            .collect::<Vec<_>>();
        assert_ne!(pooled_values(transposed), pooled.values());
    }
    fn pooled_values(flat: Vec<f32>) -> Vec<f32> {
        pooled(flat).values().to_vec()
    }

    /// Synthetic frozen transforms: row k of the 1024x2048 projection selects foundation
    /// coordinate 2k + 1 (scaled by `projection_scale`); the golden values below are written
    /// out independently of this construction.
    fn golden_transforms(projection_scale: f32) -> FrozenTransforms {
        FrozenTransforms::synthetic_for_testing(projection_scale)
    }

    /// W2-T05 golden: assembly order, projection orientation, per-block normalization and the
    /// frozen clip, with every expected value written out independently of `assemble`.
    #[test]
    fn golden_assembly_order_projection_normalization_and_clip() {
        let transforms = golden_transforms(1.0);
        let (hyperion, foundation, target) = (
            (0..512).map(|j| j as f32).collect::<Vec<_>>(),
            (0..2048).map(|m| m as f32 * 0.01).collect::<Vec<_>>(),
            (0..512).map(|j| -(j as f32)).collect::<Vec<_>>(),
        );
        let components = components(hyperion.clone(), foundation.clone(), target.clone());
        let result = transforms.assemble(&components).unwrap();
        let clip = |x: f64| x.clamp(-32.0, 32.0);
        for j in 0..512 {
            close(result[j], clip((j as f64 - 1.0) / 4.0), "hyperion block");
            close(result[1536 + j], clip(-(j as f64) / 0.5), "target block");
        }
        for k in 0..1024 {
            let z = (2 * k + 1) as f64 * 0.01 - 0.5;
            close(
                result[512 + k],
                clip((z - 0.1) / 2.0),
                "projected foundation block",
            );
        }
        assert_eq!(
            result[511], 32.0,
            "hyperion values beyond the clip are clamped"
        );
        assert_eq!(
            result[2047], -32.0,
            "target values beyond the clip are clamped"
        );
        // Each of these changes is caught by the golden values above. (A Hyperion/Target swap
        // no longer type-checks; the serialized swap is refused by `replay`, tested below.)
        let mut values = hyperion.clone();
        values.swap(2, 3);
        let reordered = self::components(values, foundation.clone(), target.clone());
        assert_ne!(transforms.assemble(&reordered).unwrap(), result);
        let mut values = foundation.clone();
        values.rotate_left(1);
        let shifted = self::components(hyperion, values, target);
        assert_ne!(transforms.assemble(&shifted).unwrap(), result);
    }

    fn synthetic_artifact(transforms: &FrozenTransforms) -> CpuContextArtifact {
        let components = components(
            (0..512).map(|j| j as f32 / 64.0).collect(),
            (0..2048).map(|m| m as f32 * 0.01).collect(),
            (0..512).map(|j| -(j as f32) / 64.0).collect(),
        );
        let context = transforms.assemble(&components).unwrap();
        let preprocessing = TrainingPreprocessing {
            schema: "synthetic".into(),
            columns: vec![],
            target_mean: 0.0,
            target_scale: 1.0,
        };
        let artifact = CpuContextArtifact {
            schema: SCHEMA.into(),
            backend: BACKEND.into(),
            qualification: "unqualified".into(),
            source_sha256: "e".repeat(40),
            producer_sha256: "f".repeat(64),
            input_sha256: "1".repeat(64),
            training: TrainingIdentity {
                dataset_sha256: "a".repeat(64),
                prepared_sha256: "b".repeat(64),
                training_sha256: "c".repeat(64),
                fitting_rows_sha256: "d".repeat(64),
                cut: 0,
            },
            selected_training_rows_sha256: "d".repeat(64),
            original_labels_sha256: "2".repeat(64),
            preprocessing_sha256: sha(&serde_json::to_vec(&preprocessing).unwrap()),
            prepared_rows_sha256: components.scope().into(),
            preprocessing,
            encoder: EncoderIdentity::accepted(),
            components: components.to_vectors(),
            vector: context.values().to_vec(),
            final_test_targets_used: false,
        };
        artifact.validate().unwrap();
        artifact
    }

    /// W2-T03: products of different encodes never assemble together, and the typed wire
    /// layout is the historical one.
    #[test]
    fn products_are_scoped_to_one_encode_and_serialize_in_the_historical_layout() {
        let transforms = golden_transforms(1.0);
        let other = "4444444444444444444444444444444444444444444444444444444444444444";
        let mixed = Components {
            hyperion: HyperionSignature::for_testing(vec![0.5; 512], SCOPE),
            foundation: RawFoundation::for_testing(vec![0.5; 2048], other),
            target: TargetBlock::for_testing(vec![0.5; 512], SCOPE),
        };
        let error = transforms.assemble(&mixed).unwrap_err();
        assert!(
            format!("{error:#}").contains("different encodes"),
            "{error:#}"
        );
        let good = components(vec![0.5; 512], vec![0.5; 2048], vec![0.5; 512]);
        let projected = transforms.project(&good.foundation).unwrap();
        assert_eq!(projected.len(), ProjectedFoundation::WIDTH);
        assert_eq!(projected.scope(), SCOPE);
        // Typed products serialize exactly like the historical Vec<f32> fields.
        let typed = serde_json::to_value(&good).unwrap();
        assert_eq!(typed, serde_json::to_value(good.to_vectors()).unwrap());
        assert_eq!(
            typed,
            serde_json::json!({"hyperion": vec![0.5f32; 512], "foundation": vec![0.5f32; 2048], "target": vec![0.5f32; 512]})
        );
        // Layout of a product is checked when it is constructed.
        assert!(RawFoundation::from_foundation_encoder(vec![0.5; 2047], SCOPE).is_err());
        assert!(TargetBlock::from_target_encoder(vec![f32::NAN; 512], SCOPE).is_err());
        assert!(TargetBlock::from_target_encoder(vec![0.5; 512], "not a scope").is_err());
    }

    /// W2-T03 acceptance: serialized Target/signature and Foundation/final-context swaps,
    /// padding and truncation fail at the receiver; the correct producer output passes.
    #[test]
    fn serialized_swaps_padding_and_truncation_fail_at_the_receiver() {
        let transforms = golden_transforms(1.0);
        let artifact = synthetic_artifact(&transforms);
        // Positive control: the producer's own output replays, bit for bit.
        let received = artifact.replay_assembly(&transforms).unwrap();
        assert_eq!(received.values(), artifact.vector.as_slice());
        assert_eq!(received.scope(), artifact.prepared_rows_sha256);
        let bytes = serde_json::to_vec(&artifact).unwrap();
        let parsed: CpuContextArtifact = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(parsed.replay_assembly(&transforms).unwrap(), received);

        let refuse = |json: serde_json::Value, needle: &str| {
            let error = match serde_json::from_value::<CpuContextArtifact>(json) {
                Err(error) => format!("{error:#}"),
                Ok(parsed) => format!("{:#}", parsed.replay_assembly(&transforms).unwrap_err()),
            };
            assert!(error.contains(needle), "{needle}: {error}");
        };
        let original = serde_json::to_value(&artifact).unwrap();
        // Target/signature swap: same widths, so only the replay can see it.
        let mut json = original.clone();
        let hyperion = json["components"]["hyperion"].take();
        json["components"]["hyperion"] = json["components"]["target"].take();
        json["components"]["target"] = hyperion;
        refuse(json, "not the frozen assembly");
        // Foundation/final-context swap.
        let mut json = original.clone();
        let foundation = json["components"]["foundation"].take();
        json["components"]["foundation"] = json["vector"].take();
        json["vector"] = foundation;
        refuse(json, "not the frozen assembly");
        // Padding and truncation of each block and of the context.
        for (path, pad) in [
            ("hyperion", true),
            ("foundation", false),
            ("target", true),
            ("target", false),
        ] {
            let mut json = original.clone();
            let block = json["components"][path].as_array_mut().unwrap();
            if pad {
                block.push(serde_json::json!(0.0));
            } else {
                block.pop();
            }
            refuse(json, "layout");
        }
        for pad in [true, false] {
            let mut json = original.clone();
            let block = json["vector"].as_array_mut().unwrap();
            if pad {
                block.push(serde_json::json!(0.0));
            } else {
                block.pop();
            }
            refuse(json, "final context layout");
        }
        // Unknown component fields are refused.
        let mut json = original.clone();
        json["components"]["projected"] = serde_json::json!([]);
        refuse(json, "unknown field");
        // A substituted projection or normalization fails the replay of the correct artifact.
        assert!(artifact.replay_assembly(&golden_transforms(1.5)).is_err());
        // Nothing authority-bearing is produced by a refusal: no FinalContext is returned, and
        // the accepted artifact still replays to the same context afterwards.
        assert_eq!(artifact.replay_assembly(&transforms).unwrap(), received);
    }

    /// W2-T02 G-04: all eight encoder identity fields are held to the release decision.
    #[test]
    fn every_encoder_identity_field_is_pinned_to_the_release_decision() {
        let transforms = golden_transforms(1.0);
        let artifact = synthetic_artifact(&transforms);
        EncoderIdentity::accepted().require_accepted().unwrap();
        let changed = |f: &dyn Fn(&mut EncoderIdentity)| {
            let mut artifact = artifact.clone();
            f(&mut artifact.encoder);
            format!("{:#}", artifact.validate().unwrap_err())
        };
        for (field, error) in [
            (
                "hyperion_weights",
                changed(&|e| e.hyperion_weights = "0".repeat(64)),
            ),
            (
                "hyperion_manifest",
                changed(&|e| e.hyperion_manifest = "0".repeat(64)),
            ),
            (
                "component_bundle",
                changed(&|e| e.component_bundle = worker::COMPONENT_BUNDLE_MANIFEST_SHA256.into()),
            ),
            (
                "foundation_weights",
                changed(&|e| e.foundation_weights = "0".repeat(64)),
            ),
            (
                "target_weights",
                changed(&|e| e.target_weights = "0".repeat(64)),
            ),
            (
                "frozen_transforms",
                changed(&|e| e.frozen_transforms = "0".repeat(64)),
            ),
            ("assembly", changed(&|e| e.assembly = "0".repeat(64))),
            ("recipe", changed(&|e| e.recipe.push_str(";changed"))),
        ] {
            assert!(error.contains(field), "{field}: {error}");
        }
    }

    /// Evidence (needs the pinned asset): the accepted encoder identity is what `CpuProducer::load`
    /// derives from the pinned component-bundle manifest file.
    #[test]
    #[ignore = "needs JOPE_COMPONENT_BUNDLE_MANIFEST (the pinned model-bundle.v1.json)"]
    fn accepted_encoder_identity_rederives_from_the_pinned_component_bundle() {
        let path = std::env::var_os("JOPE_COMPONENT_BUNDLE_MANIFEST")
            .expect("set JOPE_COMPONENT_BUNDLE_MANIFEST to run this evidence test");
        let bytes = std::fs::read(path).unwrap();
        assert_eq!(sha(&bytes), worker::COMPONENT_BUNDLE_MANIFEST_SHA256);
        let manifest: starforge::bundle::ModelBundleV1 = serde_json::from_slice(&bytes).unwrap();
        let accepted = EncoderIdentity::accepted();
        assert_eq!(
            sha(&serde_json::to_vec(&manifest).unwrap()),
            accepted.component_bundle
        );
        assert_eq!(
            manifest.encoders["foundation"].sha256,
            accepted.foundation_weights
        );
        assert_eq!(manifest.encoders["target"].sha256, accepted.target_weights);
    }

    /// W2-T05 golden: the two label normalizations, including their constant-input conventions.
    #[test]
    fn golden_label_normalizations() {
        // Population deviation of {1, 2, 3, 4} is sqrt(1.25); 1e-8 is added before dividing.
        let sd = 1.25f64.sqrt() + 1e-8;
        let standardized = standardize_context_labels(&[1.0, 2.0, 3.0, 4.0]);
        for (value, raw) in standardized.iter().zip([1.0, 2.0, 3.0, 4.0]) {
            close(*value, (raw - 2.5) / sd, "standardized label");
        }
        assert_eq!(standardize_context_labels(&[7.0, 7.0]), vec![0.0, 0.0]);
        assert_eq!(minmax_targets(&[5.0, 7.0, 9.0]), vec![0.0, 0.5, 1.0]);
        assert_eq!(minmax_targets(&[9.0, 5.0, 7.0]), vec![1.0, 0.0, 0.5]);
        assert_eq!(minmax_targets(&[3.0, 3.0]), vec![0.0, 0.0]);
    }
}
