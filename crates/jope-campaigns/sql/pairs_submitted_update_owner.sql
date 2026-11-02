UPDATE pairs SET state='submitted',job_id=?,updated_at_ms=?
WHERE task=? AND dataset_hash=? AND semantic_sha256=? AND protocol_sha256=?
  AND owner=? AND manifest_sha256=?
  AND state IN ('reserved','infrastructure_retry')
