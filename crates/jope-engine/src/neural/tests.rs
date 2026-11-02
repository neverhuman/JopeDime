use super::*;

#[cfg(test)]
mod native_runtime_qualification {
    use super::*;
    #[test]
    fn cpu_optimizer_restore_completes_original_window_identically() -> Result<()> {
        tch::set_num_threads(1);
        let dir = tempfile::tempdir()?;
        let uninterrupted = nn::VarStore::new(Device::Cpu);
        let p = uninterrupted
            .root()
            .var("weight", &[4], nn::Init::Const(0.25));
        let mut optimizer = AdamW::new(1e-4, 0.01);
        for _ in 0..3 {
            p.square().sum(Kind::Float).backward();
            optimizer.update(&uninterrupted)?;
        }
        uninterrupted.save(dir.path().join("model.ot"))?;
        optimizer.save(&dir.path().join("optimizer.ot"))?;
        let original_target = optimizer.step + 3;
        let mut restored = nn::VarStore::new(Device::Cpu);
        let restored_p = restored.root().var("weight", &[4], nn::Init::Const(0.0));
        restored.load(dir.path().join("model.ot"))?;
        let mut recovered = AdamW::load(&dir.path().join("optimizer.ot"), Device::Cpu)?;
        recovered.validate_parameters(&restored)?;
        while optimizer.step < original_target {
            p.square().sum(Kind::Float).backward();
            optimizer.update(&uninterrupted)?;
        }
        while recovered.step < original_target {
            restored_p.square().sum(Kind::Float).backward();
            recovered.update(&restored)?;
        }
        assert_eq!(recovered.step, original_target);
        assert!(p.equal(&restored_p));
        for (name, (m, v)) in &optimizer.moments {
            let (rm, rv) = &recovered.moments[name];
            assert!(m.equal(rm) && v.equal(rv));
        }
        let mut corrupt = fs::read(dir.path().join("optimizer.ot"))?;
        corrupt[0] ^= 1;
        fs::write(dir.path().join("optimizer.ot"), corrupt)?;
        assert!(AdamW::load(&dir.path().join("optimizer.ot"), Device::Cpu).is_err());
        Ok(())
    }
}

/// W2-T04 (G-10c): a warm-start or resume consumes exactly the checkpoint
/// bytes its state binds. The instrumentation point
/// `manifest::after_verification` replaces the files between their check and
/// their load; the load must still be of the verified bytes.
#[cfg(test)]
mod bound_load_tests {
    use super::*;
    use crate::manifest::verification_hook;

    fn store(value: f64) -> nn::VarStore {
        let vs = nn::VarStore::new(Device::Cpu);
        let _ = vs.root().var("weight", &[4], nn::Init::Const(value));
        vs
    }

    fn weight(vs: &nn::VarStore) -> Vec<f32> {
        Vec::<f32>::try_from(vs.variables()["weight"].flatten(0, -1)).unwrap()
    }

    /// Replace each `(target, bytes)` (new inodes, via rename) the first time
    /// a receiver reports `trigger` verified.
    fn replace_on_verification(
        trigger: &Path,
        replacements: Vec<(std::path::PathBuf, Vec<u8>)>,
    ) -> verification_hook::Guard {
        let trigger = fs::canonicalize(trigger).unwrap();
        let mut done = false;
        verification_hook::install(move |path| {
            if !done && fs::canonicalize(path).ok().as_deref() == Some(trigger.as_path()) {
                done = true;
                for (target, bytes) in &replacements {
                    let staged = target.with_extension("replacement");
                    fs::write(&staged, bytes).unwrap();
                    fs::rename(&staged, target).unwrap();
                }
            }
        })
    }

    #[test]
    fn a_checkpoint_model_replaced_after_its_check_is_never_loaded() -> Result<()> {
        tch::set_num_threads(1);
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("model.ot");
        store(1.0).save(&path)?;
        let expected = serde_json::json!(file_hash(&path)?);
        let other = dir.path().join("other.ot");
        store(2.0).save(&other)?;
        // Positive control.
        let mut vs = store(0.0);
        load_bound_model(&mut vs, &path, &expected, "initialization model changed")?;
        assert_eq!(weight(&vs), [1.0; 4]);
        // Replaced after its check.
        let mut raced = store(0.0);
        let result = {
            let _hook = replace_on_verification(&path, vec![(path.clone(), fs::read(&other)?)]);
            load_bound_model(&mut raced, &path, &expected, "initialization model changed")
        };
        assert!(
            result.is_ok() && weight(&raced) == [1.0; 4],
            "a model other than the bound bytes was loaded: {result:?} {:?}",
            weight(&raced)
        );
        // Negative control: other bytes at the path are refused with the receiver's wording.
        let mut refused = store(0.0);
        let error = load_bound_model(
            &mut refused,
            &path,
            &expected,
            "initialization model changed",
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("initialization model changed"), "{error}");
        assert_eq!(weight(&refused), [0.0; 4]);
        Ok(())
    }

    #[test]
    fn a_checkpoint_optimizer_replaced_after_its_check_is_never_loaded() -> Result<()> {
        tch::set_num_threads(1);
        let dir = tempfile::tempdir()?;
        let optimizer = |step: u64, value: f64| AdamW {
            step,
            lr: 1e-4,
            decay: 0.01,
            moments: BTreeMap::from([(
                "weight".to_string(),
                (
                    Tensor::full([4], value, (Kind::Float, Device::Cpu)),
                    Tensor::full([4], value * 2.0, (Kind::Float, Device::Cpu)),
                ),
            )]),
            backend: "scalar_fp32_v1".into(),
        };
        let path = dir.path().join("optimizer.ot");
        optimizer(3, 1.0).save(&path)?;
        let expected = serde_json::json!(file_hash(&path)?);
        let expected_metadata = serde_json::json!(file_hash(&path.with_extension("json"))?);
        // A consistent replacement pair: another optimizer and the metadata naming it.
        let other = dir.path().join("other").join("optimizer.ot");
        fs::create_dir_all(other.parent().unwrap())?;
        optimizer(9, 5.0).save(&other)?;
        let replacements = vec![
            (path.clone(), fs::read(&other)?),
            (
                path.with_extension("json"),
                fs::read(other.with_extension("json"))?,
            ),
        ];
        let refusal = "warm-start optimizer missing or changed";
        // Positive control.
        let loaded = AdamW::load_bound(&path, &expected, &expected_metadata, Device::Cpu, refusal)?;
        assert_eq!(loaded.step, 3);
        // Replaced after its check.
        let result = {
            let _hook = replace_on_verification(&path, replacements);
            AdamW::load_bound(&path, &expected, &expected_metadata, Device::Cpu, refusal)
        };
        assert!(
            matches!(&result, Ok(raced) if raced.step == 3
                && raced.moments["weight"].0.equal(&Tensor::full([4], 1.0, (Kind::Float, Device::Cpu)))),
            "an optimizer other than the bound bytes was loaded: {:?}",
            result.as_ref().map(|o| o.step)
        );
        // Negative control: the replaced pair is refused with the receiver's wording.
        let error = AdamW::load_bound(&path, &expected, &expected_metadata, Device::Cpu, refusal)
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(error.contains(refusal), "{error}");
        Ok(())
    }

    /// W7 (961ad2333), landed on this API: a checkpoint restored through
    /// `load_bound_model` and `AdamW::load_bound` after one AdamW update is the
    /// saved state and continues training exactly as the original, although
    /// every file it came from is replaced right after its check. The
    /// replacements are refused on the next load, and optimizer bytes the
    /// metadata does not name are refused by `from_verified`.
    #[test]
    fn checkpoint_consumption_uses_verified_tensor_bytes_after_path_replacement() -> Result<()> {
        tch::set_num_threads(1);
        let dir = tempfile::tempdir()?;
        let vs = nn::VarStore::new(Device::Cpu);
        let p = vs.root().var("weight", &[2], nn::Init::Const(2.0));
        let mut optimizer = AdamW::new(1e-4, 0.01);
        p.square().sum(Kind::Float).backward();
        optimizer.update(&vs)?;
        let model = dir.path().join("model.ot");
        let optimizer_path = dir.path().join("optimizer.ot");
        vs.save(&model)?;
        optimizer.save(&optimizer_path)?;
        let state = serde_json::json!({
            "model_sha256": file_hash(&model)?,
            "optimizer_sha256": file_hash(&optimizer_path)?,
            "optimizer_metadata_sha256": file_hash(&optimizer_path.with_extension("json"))?,
        });
        let saved_step = optimizer.step;
        let metadata: serde_json::Value = read_json(&optimizer_path.with_extension("json"))?;
        let optimizer_bytes = fs::read(&optimizer_path)?;
        let replaced = || b"replaced after verification".to_vec();
        let (model_refusal, optimizer_refusal) = (
            "checkpoint model changed",
            "checkpoint optimizer or configuration changed",
        );
        // Each file is replaced as soon as it is verified; the load uses the verified bytes.
        let mut restored = nn::VarStore::new(Device::Cpu);
        let q = restored.root().var("weight", &[2], nn::Init::Const(0.0));
        {
            let _hook = replace_on_verification(&model, vec![(model.clone(), replaced())]);
            load_bound_model(&mut restored, &model, &state["model_sha256"], model_refusal)?;
        }
        let mut resumed = {
            let _hook = replace_on_verification(
                &optimizer_path,
                vec![
                    (optimizer_path.clone(), replaced()),
                    (optimizer_path.with_extension("json"), replaced()),
                ],
            );
            AdamW::load_bound(
                &optimizer_path,
                &state["optimizer_sha256"],
                &state["optimizer_metadata_sha256"],
                Device::Cpu,
                optimizer_refusal,
            )?
        };
        // The replacements are refused on the next load, with the receivers' wording.
        let mut other = nn::VarStore::new(Device::Cpu);
        let _ = other.root().var("weight", &[2], nn::Init::Const(0.0));
        let error = load_bound_model(&mut other, &model, &state["model_sha256"], model_refusal)
            .unwrap_err()
            .to_string();
        assert!(error.contains(model_refusal), "{error}");
        let error = AdamW::load_bound(
            &optimizer_path,
            &state["optimizer_sha256"],
            &state["optimizer_metadata_sha256"],
            Device::Cpu,
            optimizer_refusal,
        )
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
        assert!(error.contains(optimizer_refusal), "{error}");
        // The restored checkpoint is the saved one and continues identically.
        assert_eq!(resumed.step, saved_step);
        assert_eq!((&p - &q).abs().max().double_value(&[]), 0.0);
        p.square().sum(Kind::Float).backward();
        optimizer.update(&vs)?;
        q.square().sum(Kind::Float).backward();
        resumed.update(&restored)?;
        assert_eq!((&p - &q).abs().max().double_value(&[]), 0.0);
        // `from_verified` loads the verified bytes and refuses bytes its metadata does not name.
        assert_eq!(
            AdamW::from_verified(metadata.clone(), optimizer_bytes.clone(), Device::Cpu)?.step,
            saved_step
        );
        let mut corrupt = optimizer_bytes;
        corrupt[0] ^= 1;
        let error = AdamW::from_verified(metadata, corrupt, Device::Cpu)
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(error.contains("optimizer artifact changed"), "{error}");
        Ok(())
    }
}
