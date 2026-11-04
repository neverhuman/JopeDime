use super::*;

#[test]
fn resident_worker_chunk_ladder_is_exact_and_redoes_at_most_one_dataset() {
    let tasks = ["regression", "binary", "multiclass", "timeseries"];
    let items = tasks
        .into_iter()
        .flat_map(|task| (0..8).map(move |index| (task, index)))
        .enumerate()
        .map(|(number, (task, index))| EncoderWorkItem {
            input_identity_sha256: format!("{:064x}", number + 1),
            task: task.to_owned(),
            rows: 128 + index,
            columns: 16,
            input_bytes: 4_096,
            estimated_vram_bytes: 1_024,
        })
        .collect::<Vec<_>>();
    let chunks = plan_chunks(items, ChunkMode::Proof, 2_048).unwrap();
    assert_eq!(
        chunks.iter().map(|chunk| chunk.items.len()).sum::<usize>(),
        32
    );
    assert!(plan_chunks(Vec::new(), ChunkMode::Replay, 2_048).is_err());

    let progress = ResidentWorkerProgress {
        worker_capsule_sha256: digest(250),
        loaded_expert_sha256: digest(251),
        committed_input_identities: vec![digest(1), digest(2)],
        active_input_identity: Some(digest(3)),
    };
    progress.validate().unwrap();
    assert_eq!(progress.maximum_redo_datasets(), 1);
}
