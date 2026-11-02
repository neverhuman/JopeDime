SELECT state,reservation_owner,manifest_sha256,job_id
FROM pairs
WHERE task=? AND dataset_hash=? AND semantic_sha256=? AND protocol_sha256=?
