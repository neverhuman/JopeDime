use jope_parallel::{manifest::*, queue, service::{GpJob,GpJobs}};

#[test]
fn queue_merge_is_idempotent_and_preserves_other_work() {
    let root = tempfile::tempdir().unwrap();
    let job = |n| GpJob { prepared: format!("data/prepared/{n}").into(), prepared_sha256: "a".repeat(64), cut:0,
        output:format!("runs/search/{n}").into(), seeds:None,seeds_sha256:None };
    let mut incoming = GpJobs { schema:"parallel.gp_jobs.v1".into(),qualification:true,jobs:vec![job(1)],vocabulary:None };
    assert_eq!(queue::merge(root.path(),&incoming).unwrap(),1);
    assert_eq!(queue::merge(root.path(),&incoming).unwrap(),0);
    incoming.jobs=vec![job(2)];
    assert_eq!(queue::merge(root.path(),&incoming).unwrap(),1);
    let queue_path=root.path().join("runs/gp-priority-jobs.json");
    let previous=file_hash(&queue_path).unwrap();
    incoming.jobs[0].prepared_sha256="b".repeat(64);
    assert!(queue::merge(root.path(),&incoming).is_err());
    assert_eq!(file_hash(&queue_path).unwrap(),previous);
    let saved:GpJobs=read_json(&queue_path).unwrap();
    assert_eq!(saved.jobs.len(),2);
    assert_eq!(saved.jobs[0].output,job(2).output);
    assert_eq!(saved.jobs[1].output,job(1).output);
}

#[test]
fn unencoded_jobs_cannot_replace_a_live_queue() {
    let root = tempfile::tempdir().unwrap();
    let jobs=GpJobs {schema:"parallel.gp_jobs.v1".into(),qualification:true,jobs:vec![GpJob {
        prepared:"missing-prepared".into(),prepared_sha256:"a".repeat(64),cut:0,
        output:"runs/search/1".into(),seeds:None,seeds_sha256:None}],vocabulary:None};
    queue::merge(root.path(),&jobs).unwrap();
    let queue_path=root.path().join("runs/gp-priority-jobs.json");
    let previous=file_hash(&queue_path).unwrap();
    let job_path=root.path().join("jobs.json"); let context_path=root.path().join("contexts.json");
    write_json(&job_path,&jobs).unwrap();
    write_json(&context_path,&serde_json::json!({"schema":"parallel.context_index.v1","dimensions":2048,"qualification":true,"encodings":[]})).unwrap();
    assert!(queue::enqueue(root.path(),&job_path,&context_path).is_err());
    assert_eq!(file_hash(&queue_path).unwrap(),previous);
}
