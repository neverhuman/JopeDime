WITH clean AS (
    SELECT o.dataset_hash, o.semantic_sha256, o.canonical_genome, o.score,
           coalesce(f.source_family, 'dataset:' || o.dataset_hash) AS source_family
    FROM observations o
    LEFT JOIN dataset_family f ON f.dataset_hash = o.dataset_hash
    WHERE o.task = 'regression'
      AND lower(candidate_source) NOT LIKE '%proposal%'
      AND lower(candidate_source) NOT LIKE '%model_sampled%'
      AND lower(candidate_source) NOT LIKE '%checkpoint_dl%'
      AND lower(origin_class) != 'dl_origin'
),
positioned AS (
    SELECT dataset_hash, source_family, semantic_sha256, canonical_genome, score,
           rank() OVER (PARTITION BY dataset_hash ORDER BY score ASC) AS asc_rank,
           count(*) OVER (PARTITION BY dataset_hash) AS dataset_n,
           max(score) OVER (PARTITION BY dataset_hash) AS dataset_max
    FROM clean
),
with_tops AS (
    SELECT *, sum(CASE WHEN score = dataset_max THEN 1 ELSE 0 END)
                  OVER (PARTITION BY dataset_hash) AS top_count
    FROM positioned
)
SELECT target.dataset_hash,
       candidate.semantic_sha256,
       candidate.canonical_genome,
       group_concat(candidate.dataset_hash),
       avg(CASE WHEN candidate.dataset_n = 1 THEN 1.0
                ELSE CAST(candidate.asc_rank - 1 AS REAL) / CAST(candidate.dataset_n - 1 AS REAL) END),
       min(CASE WHEN candidate.dataset_n = 1 THEN 1.0
                ELSE CAST(candidate.asc_rank - 1 AS REAL) / CAST(candidate.dataset_n - 1 AS REAL) END),
       sum(CASE WHEN candidate.score = candidate.dataset_max AND candidate.top_count = 1 THEN 1 ELSE 0 END),
       count(*)
FROM with_tops AS candidate
JOIN transfer_targets AS target
  ON candidate.dataset_hash != target.dataset_hash
 AND candidate.source_family != target.source_family
GROUP BY target.dataset_hash, candidate.semantic_sha256, candidate.canonical_genome
