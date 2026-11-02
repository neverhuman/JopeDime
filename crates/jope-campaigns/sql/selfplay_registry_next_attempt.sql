SELECT COALESCE(MAX(attempt_number),0)+1 FROM attempts
WHERE task=? AND dataset_hash=? AND semantic_sha256=? AND protocol_sha256=?
