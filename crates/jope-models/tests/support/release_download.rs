// Deterministic exact-version download fixtures. No cloud, credentials or real weights.
mod download_tests {
    use super::*;
    use jope_datasets::store::{
        InjectedBackend, ObjPath, ObjectStoreScheme, PutOutcome, Store, UpdateVersion,
    };
    use std::sync::{Arc, Mutex};

    struct Backend {
        objects: BTreeMap<String, (Vec<u8>, Option<String>)>,
        reads: Mutex<Vec<(String, Option<String>)>>,
        fail_key: Option<String>,
        delay: std::time::Duration,
    }
    impl InjectedBackend for Backend {
        fn get_versioned(
            &self,
            key: &ObjPath,
            version: Option<&str>,
        ) -> Result<Option<(Vec<u8>, UpdateVersion)>> {
            self.reads
                .lock()
                .unwrap()
                .push((key.to_string(), version.map(str::to_owned)));
            std::thread::sleep(self.delay);
            ensure!(
                self.fail_key.as_deref() != Some(key.as_ref()),
                "fixture download failure"
            );
            Ok(self.objects.get(key.as_ref()).map(|(bytes, actual)| {
                (
                    bytes.clone(),
                    UpdateVersion {
                        e_tag: version.map(str::to_owned),
                        version: actual.clone(),
                    },
                )
            }))
        }
        fn put_create(
            &self,
            _: &ObjPath,
            _: Vec<u8>,
        ) -> Result<(PutOutcome, Option<UpdateVersion>)> {
            bail!("read-only fixture")
        }
        fn put_overwrite(&self, _: &ObjPath, _: Vec<u8>) -> Result<UpdateVersion> {
            bail!("read-only fixture")
        }
        fn list_keys(&self, _: &ObjPath) -> Result<Vec<ObjPath>> {
            bail!("read-only fixture")
        }
    }
    fn fixture_store() -> (VerifiedReleaseManifest, Backend) {
        let (metadata, manifests, payloads) = payload_fixture();
        let mut objects = BTreeMap::new();
        for c in &metadata.manifest().components {
            objects.insert(
                crate::manifest_key(&c.bundle_id),
                (
                    manifests[&c.bundle_id].clone(),
                    Some(c.bundle_manifest_version_id.clone()),
                ),
            );
            objects.insert(
                format!("blobs/sha256/{}", c.sha256),
                (payloads[&c.id].clone(), Some(c.object_version_id.clone())),
            );
        }
        (
            metadata,
            Backend {
                objects,
                reads: Mutex::new(Vec::new()),
                fail_key: None,
                delay: std::time::Duration::ZERO,
            },
        )
    }
    fn store(backend: Backend) -> (Store, Arc<Backend>) {
        let backend = Arc::new(backend);
        (
            Store::from_injected_backend(
                "s3://fixture.invalid",
                ObjPath::default(),
                ObjectStoreScheme::AmazonS3,
                backend.clone(),
            ),
            backend,
        )
    }
    fn run_download(
        m: &VerifiedReleaseManifest,
        s: &Store,
        limits: DownloadLimits,
        budget: std::time::Duration,
    ) -> Result<VerifiedReleasePayloads> {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(m.download_payloads(s, limits, budget))
    }
    #[test]
    fn all_exact_version_manifests_precede_payload_reads_and_return_frozen_bytes() {
        let (m, backend) = fixture_store();
        let (s, backend) = store(backend);
        let result = run_download(
            &m,
            &s,
            DownloadLimits::default(),
            std::time::Duration::from_secs(2),
        )
        .unwrap();
        assert_eq!(result.verified_bytes(), 33);
        assert_eq!(result.payload("component-0"), Some([1, 1, 1].as_slice()));
        let reads = backend.reads.lock().unwrap();
        assert_eq!(reads.len(), 12);
        assert!(reads[..6]
            .iter()
            .all(|(k, v)| k.starts_with("bundles/") && v.as_deref() == Some("manifest-version")));
        assert!(
            reads[6..]
                .iter()
                .all(|(k, v)| k.starts_with("blobs/sha256/")
                    && v.as_deref() == Some("object-version"))
        );
    }
    #[test]
    fn invalid_limits_and_aggregate_payload_overflow_refuse_before_any_get() {
        let (m, backend) = fixture_store();
        let (s, backend) = store(backend);
        for mode in 0..5 {
            let mut l = DownloadLimits::default();
            match mode {
                0 => l.payloads.max_release_bytes = 32,
                1 => l.payloads.max_component_bytes = 7,
                2 => l.max_total_manifest_bytes = 0,
                3 => l.max_bundle_manifest_bytes = MAX_MANIFEST_BYTES as u64 + 1,
                _ => l.payloads.max_component_bytes = 0,
            }
            assert!(run_download(&m, &s, l, std::time::Duration::from_secs(1)).is_err());
            assert!(backend.reads.lock().unwrap().is_empty());
        }
        assert!(
            run_download(&m, &s, DownloadLimits::default(), std::time::Duration::ZERO).is_err()
        );
        assert!(backend.reads.lock().unwrap().is_empty());
    }
    #[test]
    fn corrupt_missing_or_wrong_version_manifests_never_fetch_payloads() {
        for mode in 0..3 {
            let (m, mut b) = fixture_store();
            let key = crate::manifest_key("bundle-5");
            match mode {
                0 => b.objects.get_mut(&key).unwrap().0.push(b' '),
                1 => {
                    b.objects.remove(&key);
                }
                _ => b.objects.get_mut(&key).unwrap().1 = Some("wrong-version".into()),
            }
            let (s, b) = store(b);
            assert!(run_download(
                &m,
                &s,
                DownloadLimits::default(),
                std::time::Duration::from_secs(2)
            )
            .is_err());
            assert!(b
                .reads
                .lock()
                .unwrap()
                .iter()
                .all(|(k, _)| k.starts_with("bundles/")));
        }
    }
    #[test]
    fn missing_version_corrupt_short_and_trailing_payloads_refuse_the_whole_release() {
        for mode in 0..5 {
            let (m, mut b) = fixture_store();
            let c = &m.manifest().components[0];
            let key = format!("blobs/sha256/{}", c.sha256);
            match mode {
                0 => {
                    b.objects.remove(&key);
                }
                1 => b.objects.get_mut(&key).unwrap().1 = None,
                2 => b.objects.get_mut(&key).unwrap().0[0] ^= 1,
                3 => {
                    b.objects.get_mut(&key).unwrap().0.pop();
                }
                _ => b.objects.get_mut(&key).unwrap().0.push(0),
            }
            let (s, _) = store(b);
            assert!(run_download(
                &m,
                &s,
                DownloadLimits::default(),
                std::time::Duration::from_secs(2)
            )
            .is_err());
        }
    }
    #[test]
    fn aggregate_manifest_budget_and_transport_failure_do_not_return_a_partial_release() {
        let (m, b) = fixture_store();
        let first = b.objects[&crate::manifest_key("bundle-0")].0.len();
        let (s, b) = store(b);
        let l = DownloadLimits {
            max_total_manifest_bytes: first as u64,
            ..Default::default()
        };
        assert!(run_download(&m, &s, l, std::time::Duration::from_secs(2)).is_err());
        assert_eq!(b.reads.lock().unwrap().len(), 1);
        let (m, mut b) = fixture_store();
        b.fail_key = Some(format!(
            "blobs/sha256/{}",
            m.manifest().components[0].sha256
        ));
        let (s, _) = store(b);
        let error = run_download(
            &m,
            &s,
            DownloadLimits::default(),
            std::time::Duration::from_secs(2),
        )
        .unwrap_err();
        assert!(format!("{error:#}").contains("fixture download failure"));
    }
    #[test]
    fn one_wall_deadline_refuses_late_backend_results_without_starting_more_gets() {
        let (m, mut b) = fixture_store();
        b.delay = std::time::Duration::from_millis(10);
        let (s, b) = store(b);
        let error = run_download(
            &m,
            &s,
            DownloadLimits::default(),
            std::time::Duration::from_millis(1),
        )
        .unwrap_err();
        assert!(format!("{error:#}").contains("deadline exceeded"));
        assert_eq!(b.reads.lock().unwrap().len(), 1);
    }
}
