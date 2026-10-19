#[cfg(any(target_os = "linux", target_os = "macos"))]
mod cache_tests {
    use super::*;
    use crate::cache::ModelsCache;
    use std::fs;
    use std::os::unix::fs::{symlink, MetadataExt};
    use std::sync::{Arc, Barrier};

    fn snapshot() -> VerifiedReleasePayloads {
        let (m, bundles, payloads) = payload_fixture();
        m.verify_payloads(&bundles, PayloadLimits::default(), |c| {
            Ok(std::io::Cursor::new(payloads[&c.id].clone()))
        })
        .unwrap()
    }
    fn target(cache: &ModelsCache, m: &VerifiedReleaseManifest) -> std::path::PathBuf {
        cache.root.join("releases").join(&m.manifest.release_id)
    }

    #[test]
    fn fresh_install_offline_reload_and_frozen_snapshot_use_exact_declared_bytes() {
        let scratch = tempfile::tempdir().unwrap();
        let cache = ModelsCache::at(scratch.path().join("models"));
        let release = snapshot();
        let dir = release.install_cache(&cache).unwrap();
        assert_eq!(dir, target(&cache, release.metadata()));
        assert_eq!(
            fs::read(dir.join("manifest.json")).unwrap(),
            release.metadata.manifest_bytes
        );
        let loaded = release
            .metadata()
            .load_cache(&cache, PayloadLimits::default())
            .unwrap();
        assert_eq!(loaded.verified_bytes(), 33);
        fs::write(dir.join("payloads/component-0"), b"bad").unwrap();
        assert_eq!(loaded.payload("component-0"), Some([1, 1, 1].as_slice()));
        assert!(release
            .metadata()
            .load_cache(&cache, PayloadLimits::default())
            .is_err());
    }

    #[test]
    fn existing_complete_version_is_reverified_without_staging_or_overwrite() {
        let scratch = tempfile::tempdir().unwrap();
        let cache = ModelsCache::at(scratch.path().join("models"));
        let release = snapshot();
        let dir = release.install_cache(&cache).unwrap();
        let inode = fs::metadata(dir.join("payloads/component-0"))
            .unwrap()
            .ino();
        let path = release
            .install_cache_with_hook(&cache, |_| bail!("must not stage a complete version"))
            .unwrap();
        assert_eq!(path, dir);
        assert_eq!(
            fs::metadata(dir.join("payloads/component-0"))
                .unwrap()
                .ino(),
            inode
        );
        assert_eq!(
            fs::read_dir(cache.root.join("releases")).unwrap().count(),
            1
        );
    }

    #[test]
    fn corrupt_missing_trailing_and_nonregular_cached_entries_refuse() {
        for mode in 0..7 {
            let scratch = tempfile::tempdir().unwrap();
            let cache = ModelsCache::at(scratch.path().join("models"));
            let release = snapshot();
            let dir = release.install_cache(&cache).unwrap();
            let payload = dir.join("payloads/component-0");
            match mode {
                0 => fs::write(&payload, b"bad").unwrap(),
                1 => fs::remove_file(&payload).unwrap(),
                2 => fs::write(&payload, b"longer").unwrap(),
                3 => fs::write(dir.join("manifest.json"), b"{}").unwrap(),
                4 => fs::write(dir.join("bundles/bundle-0.json"), b"{}").unwrap(),
                5 => fs::remove_file(dir.join("bundles/bundle-0.json")).unwrap(),
                _ => {
                    fs::remove_file(&payload).unwrap();
                    fs::create_dir(&payload).unwrap();
                }
            }
            assert!(release
                .metadata()
                .load_cache(&cache, PayloadLimits::default())
                .is_err());
            assert!(release.install_cache(&cache).is_err());
        }
    }

    #[test]
    fn root_releases_and_final_version_symlinks_are_never_followed() {
        for mode in 0..3 {
            let scratch = tempfile::tempdir().unwrap();
            let outside = scratch.path().join("outside");
            fs::create_dir(&outside).unwrap();
            fs::write(outside.join("sentinel"), b"keep").unwrap();
            let cache = ModelsCache::at(scratch.path().join("models"));
            let release = snapshot();
            match mode {
                0 => symlink(&outside, &cache.root).unwrap(),
                1 => {
                    fs::create_dir(&cache.root).unwrap();
                    symlink(&outside, cache.root.join("releases")).unwrap();
                }
                _ => {
                    fs::create_dir_all(cache.root.join("releases")).unwrap();
                    symlink(&outside, target(&cache, release.metadata())).unwrap();
                }
            }
            assert!(release.install_cache(&cache).is_err());
            assert_eq!(fs::read(outside.join("sentinel")).unwrap(), b"keep");
            assert_eq!(fs::read_dir(&outside).unwrap().count(), 1);
        }
    }

    #[test]
    fn manifest_bundle_and_payload_symlink_substitution_refuses_reload() {
        for rel in [
            "manifest.json",
            "bundles/bundle-0.json",
            "payloads/component-0",
        ] {
            let scratch = tempfile::tempdir().unwrap();
            let cache = ModelsCache::at(scratch.path().join("models"));
            let release = snapshot();
            let dir = release.install_cache(&cache).unwrap();
            let outside = scratch.path().join("outside");
            fs::write(&outside, b"outside must remain intact").unwrap();
            fs::remove_file(dir.join(rel)).unwrap();
            symlink(&outside, dir.join(rel)).unwrap();
            assert!(release
                .metadata()
                .load_cache(&cache, PayloadLimits::default())
                .is_err());
            assert_eq!(fs::read(&outside).unwrap(), b"outside must remain intact");
        }
    }

    #[test]
    fn interrupted_and_corrupt_staging_never_publishes_a_version_and_is_retained() {
        for corrupt in [false, true] {
            let scratch = tempfile::tempdir().unwrap();
            let cache = ModelsCache::at(scratch.path().join("models"));
            let release = snapshot();
            let result = release.install_cache_with_hook(&cache, |stage| {
                if corrupt {
                    fs::write(stage.join("payloads/component-0"), b"bad")?;
                    Ok(())
                } else {
                    bail!("fixture interrupted write")
                }
            });
            assert!(result.is_err());
            assert!(!target(&cache, release.metadata()).exists());
            let stages: Vec<_> = fs::read_dir(cache.root.join("releases"))
                .unwrap()
                .map(|x| x.unwrap().path())
                .collect();
            assert_eq!(stages.len(), 1);
            assert!(stages[0].join("manifest.json").exists());
            release.install_cache(&cache).unwrap();
            assert!(stages[0].exists()); // No automatic deletion of evidence from the failed attempt.
        }
    }

    #[test]
    fn destination_created_after_staging_is_never_replaced_even_when_empty() {
        let scratch = tempfile::tempdir().unwrap();
        let cache = ModelsCache::at(scratch.path().join("models"));
        let release = snapshot();
        let destination = target(&cache, release.metadata());
        let inode = std::cell::Cell::new(0);
        let result = release.install_cache_with_hook(&cache, |_| {
            fs::create_dir(&destination)?;
            inode.set(fs::metadata(&destination)?.ino());
            Ok(())
        });
        assert!(result.is_err());
        assert_eq!(fs::metadata(&destination).unwrap().ino(), inode.get());
        assert_eq!(fs::read_dir(&destination).unwrap().count(), 0);
    }

    #[test]
    fn concurrent_publishers_have_one_atomic_winner_and_keep_the_losing_stage() {
        let scratch = tempfile::tempdir().unwrap();
        let cache = ModelsCache::at(scratch.path().join("models"));
        let release = Arc::new(snapshot());
        let barrier = Arc::new(Barrier::new(2));
        let workers: Vec<_> = (0..2)
            .map(|_| {
                let release = release.clone();
                let barrier = barrier.clone();
                let cache = cache.clone();
                std::thread::spawn(move || {
                    release.install_cache_with_hook(&cache, |_| {
                        barrier.wait();
                        Ok(())
                    })
                })
            })
            .collect();
        let outcomes: Vec<_> = workers.into_iter().map(|t| t.join().unwrap()).collect();
        assert_eq!(outcomes.iter().filter(|r| r.is_ok()).count(), 1);
        assert_eq!(outcomes.iter().filter(|r| r.is_err()).count(), 1);
        assert_eq!(
            release
                .metadata()
                .load_cache(&cache, PayloadLimits::default())
                .unwrap()
                .verified_bytes(),
            33
        );
        assert_eq!(
            fs::read_dir(cache.root.join("releases")).unwrap().count(),
            2
        );
    }

    #[test]
    fn unsafe_roots_and_reload_payload_budget_refuse() {
        let release = snapshot();
        assert!(release
            .install_cache(&ModelsCache::at("relative-root"))
            .is_err());
        let scratch = tempfile::tempdir().unwrap();
        assert!(release
            .install_cache(&ModelsCache::at(scratch.path().join("..")))
            .is_err());
        let cache = ModelsCache::at(scratch.path().join("models"));
        release.install_cache(&cache).unwrap();
        let limits = PayloadLimits {
            max_release_bytes: 32,
            ..Default::default()
        };
        assert!(release.metadata().load_cache(&cache, limits).is_err());
    }

    #[test]
    fn replaced_staging_name_cannot_publish_unverified_directory_or_symlink() {
        for as_symlink in [true, false] {
            let scratch = tempfile::tempdir().unwrap();
            let cache = ModelsCache::at(scratch.path().join("models"));
            let release = snapshot();
            let outside = scratch.path().join("outside");
            fs::create_dir(&outside).unwrap();
            fs::write(outside.join("sentinel"), b"keep").unwrap();
            let result = release.install_cache_with_hook(&cache, |stage| {
                fs::rename(stage, scratch.path().join("displaced-stage"))?;
                if as_symlink {
                    symlink(&outside, stage)?;
                } else {
                    fs::create_dir(stage)?;
                }
                Ok(())
            });
            assert!(
                result.is_err(),
                "unverified staging replacement was published"
            );
            assert!(!target(&cache, release.metadata()).exists());
            assert_eq!(fs::read(outside.join("sentinel")).unwrap(), b"keep");
        }
    }

    #[test]
    fn exact_noncanonical_manifest_bytes_survive_install_and_reload() {
        let (m, bundles, payloads) = payload_fixture();
        let mut bytes = m.manifest_bytes.clone();
        bytes.push(b'\n');
        let pin = ReleasePin {
            release_id: m.manifest.release_id.clone(),
            source_commit: m.manifest.source_commit.clone(),
            manifest_sha256: sha256_hex(&bytes),
        };
        let m = verify_against_pin(&pin, &bytes).unwrap();
        let release = m
            .verify_payloads(&bundles, PayloadLimits::default(), |c| {
                Ok(std::io::Cursor::new(payloads[&c.id].clone()))
            })
            .unwrap();
        let scratch = tempfile::tempdir().unwrap();
        let cache = ModelsCache::at(scratch.path().join("models"));
        let dir = release.install_cache(&cache).unwrap();
        assert_eq!(fs::read(dir.join("manifest.json")).unwrap(), bytes);
        m.load_cache(&cache, PayloadLimits::default()).unwrap();
        assert!(!format!("{m:?}").contains("manifest_bytes"));
    }

    #[test]
    fn unexpected_cached_entries_refuse_reload_and_reuse_without_repair() {
        for subdir in ["", "bundles", "payloads"] {
            let scratch = tempfile::tempdir().unwrap();
            let cache = ModelsCache::at(scratch.path().join("models"));
            let release = snapshot();
            let dir = release.install_cache(&cache).unwrap();
            let extra = dir.join(subdir).join("unexpected");
            fs::write(&extra, b"preserve unexpected bytes").unwrap();
            assert!(release
                .metadata()
                .load_cache(&cache, PayloadLimits::default())
                .is_err());
            assert!(release.install_cache(&cache).is_err());
            assert_eq!(fs::read(extra).unwrap(), b"preserve unexpected bytes");
        }
    }

    #[test]
    fn unexpected_staged_files_directories_and_symlinks_never_publish() {
        for subdir in ["", "bundles", "payloads"] {
            for kind in 0..3 {
                let scratch = tempfile::tempdir().unwrap();
                let cache = ModelsCache::at(scratch.path().join("models"));
                let release = snapshot();
                let outside = scratch.path().join("outside");
                fs::write(&outside, b"keep").unwrap();
                let result = release.install_cache_with_hook(&cache, |stage| {
                    let extra = stage.join(subdir).join("unexpected");
                    match kind {
                        0 => fs::write(extra, b"extra")?,
                        1 => fs::create_dir(extra)?,
                        _ => symlink(&outside, extra)?,
                    }
                    Ok(())
                });
                assert!(result.is_err());
                assert!(!target(&cache, release.metadata()).exists());
                assert_eq!(fs::read(&outside).unwrap(), b"keep");
                assert_eq!(
                    fs::read_dir(cache.root.join("releases")).unwrap().count(),
                    1
                );
            }
        }
    }

    #[test]
    fn unexpected_non_utf8_cached_names_are_refused() {
        use std::os::unix::ffi::OsStringExt;
        for subdir in ["", "bundles", "payloads"] {
            let scratch = tempfile::tempdir().unwrap();
            let cache = ModelsCache::at(scratch.path().join("models"));
            let release = snapshot();
            let dir = release.install_cache(&cache).unwrap();
            let extra = dir
                .join(subdir)
                .join(std::ffi::OsString::from_vec(vec![0xff]));
            fs::write(&extra, b"keep").unwrap();
            assert!(release
                .metadata()
                .load_cache(&cache, PayloadLimits::default())
                .is_err());
            assert!(release.install_cache(&cache).is_err());
            assert_eq!(fs::read(extra).unwrap(), b"keep");
        }
    }
}
