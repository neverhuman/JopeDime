use super::*;

pub(super) fn cases() -> Vec<Case> {
    vec![
        Case {
            name: "disconnect-before-upload",
            inject: |w| {
                let reason =
                    blocked(publish_c1_faulty(w, |f| f.fault_put(0, Fault::Disconnect)).unwrap());
                let rec = reconstruct(&w.e.remote, w.r.id(), w.o.as_ref()).unwrap();
                assert_eq!(rec.local_only.len(), w.second.len() + 1, "{rec:?}");
                reason
            },
            committed: Some(0),
            clear: nothing,
            resume: resume_c1,
        },
        Case {
            name: "disconnect-mid-delivery",
            inject: |w| {
                let reason =
                    blocked(publish_c1_faulty(w, |f| f.fault_put(2, Fault::Disconnect)).unwrap());
                let rec = reconstruct(&w.e.remote, w.r.id(), w.o.as_ref()).unwrap();
                assert_eq!(rec.local_only.len(), w.second.len() + 1 - 2);
                assert_eq!(rec.uncommitted.len(), 2);
                reason
            },
            committed: Some(0),
            clear: nothing,
            resume: resume_c1,
        },
        Case {
            name: "response-lost-mid-delivery",
            inject: |w| {
                let reason =
                    blocked(publish_c1_faulty(w, |f| f.fault_put(2, Fault::ResponseLost)).unwrap());
                let rec = reconstruct(&w.e.remote, w.r.id(), w.o.as_ref()).unwrap();
                assert_eq!(rec.unacknowledged.len(), 1, "{rec:?}");
                reason
            },
            committed: Some(0),
            clear: nothing,
            resume: resume_c1,
        },
        Case {
            name: "partial-upload",
            inject: |w| {
                let key = w.second[0].key();
                let full = w.second[0].encoded().bytes.clone();
                let path = w.remote_path(&key);
                fs::create_dir_all(path.parent().unwrap()).unwrap();
                fs::write(&path, &full[..full.len() / 2]).unwrap();
                let reason = blocked(publish_c1(w, None).unwrap());
                let rec = reconstruct(&w.e.remote, w.r.id(), w.o.as_ref()).unwrap();
                assert!(rec.quarantined.contains(&key), "{rec:?}");
                assert!(
                    rec.local_only.contains(&key),
                    "an item shadowed by a partial upload is local-only, not delivered"
                );
                assert_eq!(fs::read(&path).unwrap(), &full[..full.len() / 2]);
                reason
            },
            committed: Some(0),
            clear: |w| {
                let key = w.second[0].key();
                fs::remove_file(w.remote_path(&key)).unwrap();
            },
            resume: resume_c1,
        },
        Case {
            name: "duplicate-delivery",
            inject: |w| {
                let before = super::fixture::remote_keys(&w.e.remote);
                let (_dir, mut other, restored) = fresh_restore(&w.e.remote, w.r.id()).unwrap();
                let again = other.deliver(&w.e.remote, None).unwrap();
                for seq in 1..=w.o.as_ref().unwrap().records() {
                    let key = w.o.as_ref().unwrap().record(seq).unwrap().item.key();
                    if w.o.as_ref().unwrap().ack(seq).is_some() {
                        w.o.as_ref().unwrap().redeliver(&w.e.remote, &key).unwrap();
                    }
                }
                assert_eq!(super::fixture::remote_keys(&w.e.remote), before);
                format!(
                    "a second host restored {} items (all already present) and redelivered; \
                     {} attempted again",
                    restored.appended, again.attempted
                )
            },
            committed: Some(0),
            clear: nothing,
            resume: resume_c1,
        },
        Case {
            name: "conditional-conflict",
            inject: |w| {
                let (_dir, mut rival, _) = fresh_restore(&w.e.remote, w.r.id()).unwrap();
                let e = w.r.scheduled(2).encoded().unwrap();
                rival.append_object(&e).unwrap();
                let rc = commit(&w.r, Some(&w.c0), &[&e]);
                let p = publish(&mut rival, &w.e.remote, PUBLISHER, &w.r.manifest, &rc).unwrap();
                assert!(matches!(p, Publication::Advanced { height: 1, .. }));
                let p = publish_c1(w, None).unwrap();
                assert!(matches!(p, Publication::Conflict { .. }), "{p:?}");
                let rec = reconstruct(&w.e.remote, w.r.id(), w.o.as_ref()).unwrap();
                assert!(rec.divergent.contains(&w.c1.digest), "{rec:?}");
                format!("{p:?}")
            },
            committed: Some(1),
            clear: nothing,
            resume: |w| {
                // Reconcile: adopt the winner's committed closure into this host's outbox, then
                // commit the same content-addressed work on top of it.
                w.outbox();
                let run_id = w.r.id().to_string();
                let adopted = restore(&w.e.remote, &run_id, w.o.as_mut().unwrap())
                    .unwrap()
                    .unwrap();
                assert!(
                    adopted.appended >= 2,
                    "the winner's commit and event were adopted"
                );
                let head = w.head().unwrap();
                let winner = Encoded::from_bytes(
                    Kind::Commit,
                    &head.commit,
                    &w.e.remote
                        .get(&object_key(Kind::Commit, &head.commit))
                        .unwrap()
                        .unwrap(),
                )
                .unwrap();
                let objects: Vec<Encoded> = w
                    .second
                    .iter()
                    .filter_map(|i| match i {
                        Item::Object(e) => Some(e.clone()),
                        Item::Blob(_) => None,
                    })
                    .collect();
                let rebased = commit(&w.r, Some(&winner), &objects.iter().collect::<Vec<_>>());
                let p = publish_with(w, None, &rebased).unwrap();
                height_after(p, w)
            },
        },
        Case {
            name: "missing-dependency-remote-only",
            inject: |w| {
                fs::remove_file(w.remote_path(&result_blob_key())).unwrap();
                let err = reconstruct(&w.e.remote, w.r.id(), w.o.as_ref()).unwrap_err();
                format!("{err:#}")
            },
            committed: None,
            clear: nothing,
            // The surviving host backs the lost blob and re-uploads it during publication.
            resume: resume_c1,
        },
        Case {
            name: "missing-dependency-everywhere",
            inject: |w| {
                let key = result_blob_key();
                w.stash
                    .insert("blob", fs::read(w.remote_path(&key)).unwrap());
                fs::remove_file(w.remote_path(&key)).unwrap();
                let sha = BlobRef::of(&Run::result(0, 1)).sha256;
                fs::remove_file(w.e.state.blob_path(&sha)).unwrap();
                let reason = blocked(publish_c1(w, None).unwrap());
                assert!(reason.contains("Missing"), "{reason}");
                reason
            },
            committed: None,
            clear: |w| {
                let bytes = w.stash["blob"].clone();
                w.e.remote
                    .put_immutable(&result_blob_key(), &bytes)
                    .unwrap();
                w.e.state.put_blob(&bytes).unwrap();
            },
            resume: resume_c1,
        },
        Case {
            name: "corrupt-blob",
            inject: |w| {
                let path = w.remote_path(&result_blob_key());
                fs::write(&path, b"not the result").unwrap();
                let reason = blocked(publish_c1(w, None).unwrap());
                assert!(reason.contains("Corrupt"), "{reason}");
                assert_eq!(fs::read(&path).unwrap(), b"not the result", "overwritten");
                reason
            },
            committed: None,
            // The operator quarantines the corrupt object; the surviving host re-uploads it.
            clear: |w| fs::remove_file(w.remote_path(&result_blob_key())).unwrap(),
            resume: resume_c1,
        },
        Case {
            name: "stale-pointer",
            inject: |w| {
                let key = head_key(w.r.id()).unwrap();
                let at_c0 = fs::read(w.remote_path(&key)).unwrap();
                assert!(matches!(
                    publish_c1(w, None).unwrap(),
                    Publication::Advanced { height: 1, .. }
                ));
                w.stash
                    .insert("head", fs::read(w.remote_path(&key)).unwrap());
                // Someone rolls the pointer back to c0.
                fs::write(w.remote_path(&key), &at_c0).unwrap();
                let rec = reconstruct(&w.e.remote, w.r.id(), w.o.as_ref()).unwrap();
                assert!(rec.divergent.contains(&w.c1.digest), "{rec:?}");
                let e = w.r.scheduled(2).encoded().unwrap();
                w.outbox().append_object(&e).unwrap();
                let c2 = commit(&w.r, Some(&w.c1), &[&e]);
                let p = publish_with(w, None, &c2).unwrap();
                assert!(matches!(p, Publication::Conflict { .. }), "{p:?}");
                format!("{p:?}")
            },
            committed: Some(0),
            clear: |w| {
                let key = head_key(w.r.id()).unwrap();
                fs::write(w.remote_path(&key), &w.stash["head"]).unwrap();
            },
            resume: resume_c1,
        },
        Case {
            name: "dangling-pointer",
            inject: |w| {
                let key = head_key(w.r.id()).unwrap();
                w.stash
                    .insert("head", fs::read(w.remote_path(&key)).unwrap());
                let mut h = w.head().unwrap();
                h.commit = "0".repeat(64);
                fs::write(w.remote_path(&key), h.encode().unwrap()).unwrap();
                let p = publish_c1(w, None).unwrap();
                assert!(matches!(p, Publication::Conflict { .. }), "{p:?}");
                format!("{p:?}")
            },
            committed: None,
            clear: |w| {
                let key = head_key(w.r.id()).unwrap();
                fs::write(w.remote_path(&key), &w.stash["head"]).unwrap();
            },
            resume: resume_c1,
        },
        Case {
            name: "crash-after-commit-payload",
            inject: |w| crash_at(w, Boundary::Payload),
            committed: Some(0),
            clear: nothing,
            resume: resume_c1,
        },
        Case {
            name: "crash-after-commit-record",
            inject: |w| crash_at(w, Boundary::Record),
            committed: Some(0),
            clear: nothing,
            resume: resume_c1,
        },
        Case {
            name: "crash-after-remote-put-before-ack",
            inject: |w| {
                let what = crash_at(w, Boundary::RemotePut);
                let rec = reconstruct(&w.e.remote, w.r.id(), w.o.as_ref()).unwrap();
                assert_eq!(rec.unacknowledged.len(), 1, "{rec:?}");
                what
            },
            committed: Some(0),
            clear: nothing,
            resume: resume_c1,
        },
        Case {
            name: "crash-after-ack",
            inject: |w| crash_at(w, Boundary::Ack),
            committed: Some(0),
            clear: nothing,
            resume: resume_c1,
        },
        Case {
            name: "crash-before-pointer-write",
            inject: |w| {
                w.outbox();
                let c1 = w.c1.clone();
                let p = prepare(
                    w.o.as_mut().unwrap(),
                    &w.e.remote,
                    PUBLISHER,
                    &w.r.manifest,
                    &c1,
                )
                .unwrap();
                assert!(matches!(p, Prepared::Ready(_)));
                drop(p);
                w.crash();
                w.outbox();
                let rec = reconstruct(&w.e.remote, w.r.id(), w.o.as_ref()).unwrap();
                assert!(rec.uncommitted.contains(&w.c1.key()) && rec.local_only.is_empty());
                "everything uploaded and verified; the pointer was never written".into()
            },
            committed: Some(0),
            clear: nothing,
            resume: resume_c1,
        },
        Case {
            name: "pointer-disconnect",
            inject: |w| {
                let err =
                    publish_c1_faulty(w, |f| f.fault_pointer(0, Fault::Disconnect)).unwrap_err();
                format!("{err:#}")
            },
            committed: Some(0),
            clear: nothing,
            resume: resume_c1,
        },
        Case {
            name: "pointer-response-lost",
            inject: |w| {
                let p = publish_c1_faulty(w, |f| f.fault_pointer(0, Fault::ResponseLost)).unwrap();
                assert!(
                    matches!(p, Publication::Advanced { height: 1, .. }),
                    "{p:?}"
                );
                format!("{p:?}")
            },
            committed: Some(1),
            clear: nothing,
            resume: resume_c1,
        },
    ]
}
