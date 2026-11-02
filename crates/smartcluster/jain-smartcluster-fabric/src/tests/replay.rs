use super::*;

#[test]
fn replay_cancel_and_duplicate_reports_are_fenced() {
    let run_id = Ulid::from(8_u128);
    let root = tempfile::tempdir().unwrap();
    fs::set_permissions(root.path(), fs::Permissions::from_mode(0o700)).unwrap();
    let mut journal = ReplayJournal::open(root.path().to_path_buf()).unwrap();
    let assignment = AttemptFence {
        attempt: 1,
        fence: 4,
        hub_epoch: 2,
        control_position: 10,
    };
    journal
        .accept_assignment(run_id, assignment.clone())
        .unwrap();
    assert_eq!(
        journal.accept_assignment(run_id, assignment),
        Err(FabricError::ReplayedPosition)
    );
    assert_eq!(
        journal.accept_cancel(
            run_id,
            AttemptFence {
                attempt: 1,
                fence: 5,
                hub_epoch: 2,
                control_position: 11,
            }
        ),
        Err(FabricError::FenceRejected)
    );
    journal
        .accept_cancel(
            run_id,
            AttemptFence {
                attempt: 1,
                fence: 4,
                hub_epoch: 2,
                control_position: 11,
            },
        )
        .unwrap();
    assert!(journal.accept_terminal_receipt(run_id, [9; 32]).unwrap());
    assert!(!journal.accept_terminal_receipt(run_id, [9; 32]).unwrap());
    assert_eq!(
        journal.accept_terminal_receipt(run_id, [10; 32]),
        Err(FabricError::ConflictingWorkerReport)
    );
    drop(journal);
    let mut recovered = ReplayJournal::open(root.path().to_path_buf()).unwrap();
    assert!(!recovered.accept_terminal_receipt(run_id, [9; 32]).unwrap());

    let crashed_root = tempfile::tempdir().unwrap();
    fs::set_permissions(crashed_root.path(), fs::Permissions::from_mode(0o700)).unwrap();
    std::fs::write(
        crashed_root.path().join("replay-journal.pending"),
        b"incomplete transition",
    )
    .unwrap();
    assert!(matches!(
        ReplayJournal::open(crashed_root.path().to_path_buf()),
        Err(FabricError::JournalIo(_))
    ));
}

#[test]
fn replay_journal_rejects_unsafe_root_and_file_custody() {
    let broad_root = tempfile::tempdir().unwrap();
    fs::set_permissions(broad_root.path(), fs::Permissions::from_mode(0o755)).unwrap();
    assert!(matches!(
        ReplayJournal::open(broad_root.path().to_path_buf()),
        Err(FabricError::JournalIo(_))
    ));

    let mode_root = tempfile::tempdir().unwrap();
    fs::set_permissions(mode_root.path(), fs::Permissions::from_mode(0o700)).unwrap();
    let run_id = Ulid::from(81_u128);
    let mut journal = ReplayJournal::open(mode_root.path().to_path_buf()).unwrap();
    journal
        .accept_assignment(
            run_id,
            AttemptFence {
                attempt: 1,
                fence: 4,
                hub_epoch: 2,
                control_position: 10,
            },
        )
        .unwrap();
    drop(journal);
    let state = mode_root.path().join("replay-journal.postcard");
    fs::set_permissions(&state, fs::Permissions::from_mode(0o644)).unwrap();
    assert!(matches!(
        ReplayJournal::open(mode_root.path().to_path_buf()),
        Err(FabricError::JournalIo(_))
    ));

    let link_root = tempfile::tempdir().unwrap();
    fs::set_permissions(link_root.path(), fs::Permissions::from_mode(0o700)).unwrap();
    let mut journal = ReplayJournal::open(link_root.path().to_path_buf()).unwrap();
    journal
        .accept_assignment(
            Ulid::from(82_u128),
            AttemptFence {
                attempt: 1,
                fence: 4,
                hub_epoch: 2,
                control_position: 10,
            },
        )
        .unwrap();
    drop(journal);
    let state = link_root.path().join("replay-journal.postcard");
    fs::hard_link(&state, link_root.path().join("state-hardlink")).unwrap();
    assert!(matches!(
        ReplayJournal::open(link_root.path().to_path_buf()),
        Err(FabricError::JournalIo(_))
    ));

    let symlink_root = tempfile::tempdir().unwrap();
    fs::set_permissions(symlink_root.path(), fs::Permissions::from_mode(0o700)).unwrap();
    let mut journal = ReplayJournal::open(symlink_root.path().to_path_buf()).unwrap();
    journal
        .accept_assignment(
            Ulid::from(83_u128),
            AttemptFence {
                attempt: 1,
                fence: 4,
                hub_epoch: 2,
                control_position: 10,
            },
        )
        .unwrap();
    drop(journal);
    let state = symlink_root.path().join("replay-journal.postcard");
    let backing = symlink_root.path().join("state-backing");
    fs::rename(&state, &backing).unwrap();
    symlink(&backing, &state).unwrap();
    assert!(matches!(
        ReplayJournal::open(symlink_root.path().to_path_buf()),
        Err(FabricError::JournalIo(_))
    ));

    let pending_root = tempfile::tempdir().unwrap();
    fs::set_permissions(pending_root.path(), fs::Permissions::from_mode(0o700)).unwrap();
    symlink(
        pending_root.path().join("missing"),
        pending_root.path().join("replay-journal.pending"),
    )
    .unwrap();
    assert!(matches!(
        ReplayJournal::open(pending_root.path().to_path_buf()),
        Err(FabricError::JournalIo(_))
    ));
}

#[test]
fn replay_journal_rejects_root_replacement_after_open() {
    let parent = tempfile::tempdir().unwrap();
    let root = parent.path().join("journal");
    fs::create_dir(&root).unwrap();
    fs::set_permissions(&root, fs::Permissions::from_mode(0o700)).unwrap();
    let mut journal = ReplayJournal::open(root.clone()).unwrap();
    fs::rename(&root, parent.path().join("original-journal")).unwrap();
    fs::create_dir(&root).unwrap();
    fs::set_permissions(&root, fs::Permissions::from_mode(0o700)).unwrap();

    assert!(matches!(
        journal.accept_assignment(
            Ulid::from(84_u128),
            AttemptFence {
                attempt: 1,
                fence: 4,
                hub_epoch: 2,
                control_position: 10,
            },
        ),
        Err(FabricError::JournalIo(_))
    ));
}

#[test]
fn replay_journal_rejects_a_second_live_opener_and_preserves_all_state() {
    let root = tempfile::tempdir().unwrap();
    fs::set_permissions(root.path(), fs::Permissions::from_mode(0o700)).unwrap();
    let mut owner = ReplayJournal::open(root.path().to_path_buf()).unwrap();
    let lock = root.path().join("replay-journal.lock");
    let lock_inode = fs::metadata(&lock).unwrap().ino();

    assert!(matches!(
        ReplayJournal::open(root.path().to_path_buf()),
        Err(FabricError::JournalLocked)
    ));
    assert!(!root.path().join("replay-journal.postcard").exists());
    owner
        .accept_assignment(
            Ulid::from(101_u128),
            AttemptFence {
                attempt: 1,
                fence: 4,
                hub_epoch: 2,
                control_position: 10,
            },
        )
        .unwrap();
    drop(owner);
    assert_eq!(fs::metadata(&lock).unwrap().ino(), lock_inode);

    let mut successor = ReplayJournal::open(root.path().to_path_buf()).unwrap();
    assert_eq!(fs::metadata(&lock).unwrap().ino(), lock_inode);
    assert_eq!(
        successor.accept_assignment(
            Ulid::from(101_u128),
            AttemptFence {
                attempt: 1,
                fence: 4,
                hub_epoch: 2,
                control_position: 10,
            },
        ),
        Err(FabricError::ReplayedPosition)
    );
    successor
        .accept_assignment(
            Ulid::from(102_u128),
            AttemptFence {
                attempt: 1,
                fence: 5,
                hub_epoch: 2,
                control_position: 11,
            },
        )
        .unwrap();
    drop(successor);

    let mut recovered = ReplayJournal::open(root.path().to_path_buf()).unwrap();
    for (run_id, fence, position) in [(101_u128, 4, 10), (102_u128, 5, 11)] {
        assert_eq!(
            recovered.accept_assignment(
                Ulid::from(run_id),
                AttemptFence {
                    attempt: 1,
                    fence,
                    hub_epoch: 2,
                    control_position: position,
                },
            ),
            Err(FabricError::ReplayedPosition)
        );
    }
}

#[test]
fn replay_journal_rejects_unsafe_or_replaced_lock_file() {
    let broad_lock_root = tempfile::tempdir().unwrap();
    fs::set_permissions(broad_lock_root.path(), fs::Permissions::from_mode(0o700)).unwrap();
    let broad_lock = broad_lock_root.path().join("replay-journal.lock");
    fs::write(&broad_lock, []).unwrap();
    fs::set_permissions(&broad_lock, fs::Permissions::from_mode(0o644)).unwrap();
    assert!(matches!(
        ReplayJournal::open(broad_lock_root.path().to_path_buf()),
        Err(FabricError::JournalIo(_))
    ));

    let linked_lock_root = tempfile::tempdir().unwrap();
    fs::set_permissions(linked_lock_root.path(), fs::Permissions::from_mode(0o700)).unwrap();
    let linked_lock = linked_lock_root.path().join("replay-journal.lock");
    fs::write(&linked_lock, []).unwrap();
    fs::set_permissions(&linked_lock, fs::Permissions::from_mode(0o600)).unwrap();
    fs::hard_link(&linked_lock, linked_lock_root.path().join("lock-hardlink")).unwrap();
    assert!(matches!(
        ReplayJournal::open(linked_lock_root.path().to_path_buf()),
        Err(FabricError::JournalIo(_))
    ));

    let symlink_lock_root = tempfile::tempdir().unwrap();
    fs::set_permissions(symlink_lock_root.path(), fs::Permissions::from_mode(0o700)).unwrap();
    let lock_target = symlink_lock_root.path().join("lock-target");
    fs::write(&lock_target, []).unwrap();
    fs::set_permissions(&lock_target, fs::Permissions::from_mode(0o600)).unwrap();
    symlink(
        &lock_target,
        symlink_lock_root.path().join("replay-journal.lock"),
    )
    .unwrap();
    assert!(matches!(
        ReplayJournal::open(symlink_lock_root.path().to_path_buf()),
        Err(FabricError::JournalIo(_))
    ));

    let nonempty_lock_root = tempfile::tempdir().unwrap();
    fs::set_permissions(nonempty_lock_root.path(), fs::Permissions::from_mode(0o700)).unwrap();
    let nonempty_lock = nonempty_lock_root.path().join("replay-journal.lock");
    fs::write(&nonempty_lock, b"not-a-lock-marker").unwrap();
    fs::set_permissions(&nonempty_lock, fs::Permissions::from_mode(0o600)).unwrap();
    assert!(matches!(
        ReplayJournal::open(nonempty_lock_root.path().to_path_buf()),
        Err(FabricError::JournalIo(_))
    ));

    let replaced_lock_root = tempfile::tempdir().unwrap();
    fs::set_permissions(replaced_lock_root.path(), fs::Permissions::from_mode(0o700)).unwrap();
    let lock = replaced_lock_root.path().join("replay-journal.lock");
    fs::write(&lock, []).unwrap();
    fs::set_permissions(&lock, fs::Permissions::from_mode(0o600)).unwrap();
    let mut journal = ReplayJournal::open(replaced_lock_root.path().to_path_buf()).unwrap();
    fs::rename(&lock, replaced_lock_root.path().join("replaced-lock")).unwrap();
    fs::write(&lock, []).unwrap();
    fs::set_permissions(&lock, fs::Permissions::from_mode(0o600)).unwrap();
    assert!(matches!(
        journal.accept_assignment(
            Ulid::from(103_u128),
            AttemptFence {
                attempt: 1,
                fence: 6,
                hub_epoch: 2,
                control_position: 12,
            },
        ),
        Err(FabricError::JournalIo(_))
    ));
}
