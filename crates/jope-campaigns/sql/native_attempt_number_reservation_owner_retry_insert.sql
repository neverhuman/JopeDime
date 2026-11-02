INSERT INTO attempts(task,dataset_hash,semantic_sha256,protocol_sha256,
attempt_number,state,reservation_owner,manifest_path,manifest_sha256,job_id,created_at_ms)
VALUES(?,?,?,?,?,'reserved',?,?,?,?,?)
