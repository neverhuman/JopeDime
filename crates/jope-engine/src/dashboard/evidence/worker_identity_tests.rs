use super::experimental_gpu_rank;

#[test]
fn recognizes_batch_workers_without_a_world_flag() {
    assert!(experimental_gpu_rank(
        b"jope-parallel\0qualify-batch-rank\0--rank\x000\0"
    ));
    assert!(experimental_gpu_rank(
        b"jope-parallel\0train-rank\0--rank\x007\0--world\08\0"
    ));
    assert!(experimental_gpu_rank(
        b"python3\0encode.py\0--rank\x000\0--world-size\08\0"
    ));
    assert!(!experimental_gpu_rank(b"python3\0batch-controller.py\0"));
    assert!(!experimental_gpu_rank(
        b"jope-parallel\0qualify-batch-rank\0--rank\0invalid\0"
    ));
    assert!(!experimental_gpu_rank(
        b"python3\0/tmp/qualify-batch-rank\0--rank\x000\0"
    ));
    assert!(!experimental_gpu_rank(
        b"python3\0script.py\0--rank\x000\0--world\x000\0"
    ));
}
