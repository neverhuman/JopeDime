# [ASTRA] S3 metadata and clean-sample receipts

Only the brief-approved prefix `s3://veox-jopedime-learning-use2/experiments/flywheel-selfplay-20261023/` was queried. All CLI operations below were read-only with `--profile veox-jepson-prod`, under nice15. No score/table/checkpoint/outer/holdout body was downloaded. Three 806-byte retention receipts were sampled into `/tmp/rd_astra_s3_samples/`; all bodies and response metadata are reproduced below for durability. ETags are recorded, not treated as SHA256 proofs.

## Listing: s3://veox-jopedime-learning-use2/experiments/flywheel-selfplay-20261023/

UTC 2026-10-31T02:02:34.568533+00:00; exit 0.

```text
                           PRE checkpoints/
                           PRE corpus_v2_train_winner_integrated/
                           PRE corpus_v3_wider_strict_train_256/
                           PRE results/
                           PRE round-01/
                           PRE round-02/
                           PRE round-03/
2026-10-24 08:31:18      33767 terminal_manifest_20261024.json
2026-10-24 09:06:35      14842 terminal_manifest_v2_20261024.json
2026-10-24 10:10:11       4798 terminal_manifest_v3_partial_20261024.json
2026-10-24 10:32:24       1569 terminal_manifest_v3_partial_addendum_v2_20261024.json
2026-10-24 11:02:26       1864 terminal_manifest_v3_partial_addendum_v3_20261024.json
2026-10-24 11:50:29       5356 terminal_manifest_v3_partial_addendum_v4_20261024.json
```

## Listing: s3://veox-jopedime-learning-use2/experiments/flywheel-selfplay-20261023/round-01/

UTC 2026-10-31T02:02:35.748896+00:00; exit 0.

```text
                           PRE gp_distilled/
                           PRE hwm/
                           PRE scores/
2026-10-24 07:57:30        806 persistence_receipt.json
2026-10-24 08:14:02        806 persistence_receipt.post_sync_806B.json
2026-10-24 08:13:59        800 persistence_receipt.pre_sync_800B.json
```

## Listing: s3://veox-jopedime-learning-use2/experiments/flywheel-selfplay-20261023/round-02/

UTC 2026-10-31T02:02:36.874462+00:00; exit 0.

```text
                           PRE gp_distilled/
                           PRE hwm/
                           PRE scores/
2026-10-24 07:53:52        806 persistence_receipt.json
```

## Listing: s3://veox-jopedime-learning-use2/experiments/flywheel-selfplay-20261023/round-03/

UTC 2026-10-31T02:02:38.172437+00:00; exit 0.

```text
                           PRE gp_distilled/
                           PRE hwm/
                           PRE scores/
2026-10-24 07:53:54        806 persistence_receipt.json
```

## Listing: s3://veox-jopedime-learning-use2/experiments/flywheel-selfplay-20261023/round-01/scores/

UTC 2026-10-31T02:02:39.490057+00:00; exit 0.

```text
2026-10-24 05:52:19      27134 retained_scores_round_01.jsonl
```

## Listing: s3://veox-jopedime-learning-use2/experiments/flywheel-selfplay-20261023/round-02/scores/

UTC 2026-10-31T02:02:40.725411+00:00; exit 0.

```text
2026-10-24 06:15:13      27603 retained_scores_round_02.jsonl
```

## Listing: s3://veox-jopedime-learning-use2/experiments/flywheel-selfplay-20261023/round-03/scores/

UTC 2026-10-31T02:02:42.057108+00:00; exit 0.

```text
2026-10-24 06:42:50      26973 retained_scores_round_03.jsonl
```

## Listing: s3://veox-jopedime-learning-use2/experiments/flywheel-selfplay-20261023/corpus_v3_wider_strict_train_256/

UTC 2026-10-31T02:02:43.318826+00:00; exit 0.

```text
2026-10-24 10:09:51     718858 corpus_v3_manifest.json
```

## Listing: s3://veox-jopedime-learning-use2/experiments/flywheel-selfplay-20261023/checkpoints/

UTC 2026-10-31T02:02:45.229828+00:00; exit 0.

```text
2026-10-24 16:47:33    2107237 flywheel_baseline_frozen_pool_train_only_430_reconstructed.pt
2026-10-24 16:47:45    2105804 flywheel_multicycle_7fold_closed_loop_430.pt
2026-10-24 17:24:00    2107237 flywheel_p9_baseline_frozen_pool_train_only_430_preimport1.pt
2026-10-24 17:24:03    2105804 flywheel_p9_cycle_1_q25gate_carried_adamw.pt
2026-10-24 17:24:06    2105804 flywheel_p9_cycle_2_q25gate_carried_adamw.pt
2026-10-24 17:24:09    2105804 flywheel_p9_cycle_3_q25gate_carried_adamw.pt
2026-10-24 17:24:12    2105804 flywheel_p9_cycle_4_q25gate_carried_adamw.pt
2026-10-24 17:24:21    2106534 flywheel_p9_q25gate45_foldsafe_prior_430_preimport1.pt
2026-10-24 17:24:18    2107018 flywheel_p9_unfiltered117_foldsafe_prior_430_preimport1.pt
2026-10-24 17:24:15    2106680 flywheel_p9_unfiltered117_leaked_prior_430_preimport1.pt
2026-10-24 17:24:24    2106817 flywheel_p9_v3a_plus_q25gate45_foldsafe_430_preimport1.pt
2026-10-24 16:47:38    2104800 flywheel_part3_cycle_1_policy.pt
2026-10-24 16:47:40    2104800 flywheel_part3_cycle_2_policy.pt
2026-10-24 16:47:42    2104800 flywheel_part3_cycle_3_policy.pt
2026-10-24 16:47:44    2104800 flywheel_part3_cycle_4_policy.pt
2026-10-24 19:07:23    2105457 flywheel_phase10_guided_cycle_1_policy.pt
2026-10-24 19:07:27    2105457 flywheel_phase10_guided_cycle_2_policy.pt
2026-10-24 19:07:29    2105457 flywheel_phase10_guided_cycle_3_policy.pt
2026-10-24 10:09:58    2105950 flywheel_winner_pool_v3a_q25_train_only_430.pt
2026-10-24 16:47:35    2107018 flywheel_winner_pool_v3a_q25_train_only_430_repro_check.pt
2026-10-24 10:10:02    2106169 flywheel_winner_pool_v3b_median_train_only_430.pt
2026-10-24 16:47:37    2107237 flywheel_winner_pool_v3b_median_train_only_430_repro_check.pt
```

## Sample: experiments/flywheel-selfplay-20261023/round-01/persistence_receipt.json

```json
{
  "utc": "2026-10-31T02:04:05.680115+00:00",
  "command": [
    "aws",
    "--profile",
    "veox-jepson-prod",
    "s3api",
    "get-object",
    "--bucket",
    "veox-jopedime-learning-use2",
    "--key",
    "experiments/flywheel-selfplay-20261023/round-01/persistence_receipt.json",
    "/tmp/rd_astra_s3_samples/round-01-persistence_receipt.json"
  ],
  "exit": 0,
  "metadata": {
    "AcceptRanges": "bytes",
    "LastModified": "2026-10-24T07:57:30+00:00",
    "ContentLength": 806,
    "ETag": "\"5844329ef50022c89bc6cc00821b3014\"",
    "ChecksumCRC64NVME": "wkCMdE5wUmw=",
    "ChecksumType": "FULL_OBJECT",
    "VersionId": "75H4Wf31_uKWdyh39Ap9iY_yBvfW1KP5",
    "ContentType": "application/json",
    "ServerSideEncryption": "AES256",
    "Metadata": {}
  },
  "size": 806,
  "sha256": "4cdc11dcf994b105614055237afda2847310190dadd289ba79f2f34951683128",
  "body": {
    "round": 1,
    "local_round_dir": "/mnt/jope-archive/flywheel-20261023/s3_mirror/round-01",
    "scores_jsonl_path": "/mnt/jope-archive/flywheel-20261023/s3_mirror/round-01/scores/retained_scores_round_01.jsonl",
    "distilled_path": "/mnt/jope-archive/flywheel-20261023/s3_mirror/round-01/gp_distilled/distilled_round_01.json",
    "retention_summary": {
      "total_evaluated": 128,
      "pruned_failed": 0,
      "pruned_duplicates": 19,
      "pruned_dominated": 71,
      "retained_top_k": 32,
      "retained_hard_negatives": 6,
      "total_retained": 38,
      "new_hwm_count": 0,
      "pruned_garbage_total": 90,
      "storage_reduction_pct": 70.31
    },
    "s3_sync": {
      "uploaded": true,
      "s3_uri": "s3://veox-jopedime-learning-use2/experiments/flywheel-selfplay-20261023/round-01/",
      "returncode": 0
    }
  }
}
```

## Sample: experiments/flywheel-selfplay-20261023/round-02/persistence_receipt.json

```json
{
  "utc": "2026-10-31T02:04:07.195002+00:00",
  "command": [
    "aws",
    "--profile",
    "veox-jepson-prod",
    "s3api",
    "get-object",
    "--bucket",
    "veox-jopedime-learning-use2",
    "--key",
    "experiments/flywheel-selfplay-20261023/round-02/persistence_receipt.json",
    "/tmp/rd_astra_s3_samples/round-02-persistence_receipt.json"
  ],
  "exit": 0,
  "metadata": {
    "AcceptRanges": "bytes",
    "LastModified": "2026-10-24T07:53:52+00:00",
    "ContentLength": 806,
    "ETag": "\"e5f268e692ac3d2fe3a8762c067f1053\"",
    "ChecksumCRC64NVME": "jVAH77C8vrI=",
    "ChecksumType": "FULL_OBJECT",
    "VersionId": "UyMRjyqasrO3DKM0viJVr7Cc5Si_iDOz",
    "ContentType": "application/json",
    "ServerSideEncryption": "AES256",
    "Metadata": {}
  },
  "size": 806,
  "sha256": "563ecdad1f2d30288f47370a8c05dde93c9bb4dc6abd0806667f9246b4e0d72e",
  "body": {
    "round": 2,
    "local_round_dir": "/mnt/jope-archive/flywheel-20261023/s3_mirror/round-02",
    "scores_jsonl_path": "/mnt/jope-archive/flywheel-20261023/s3_mirror/round-02/scores/retained_scores_round_02.jsonl",
    "distilled_path": "/mnt/jope-archive/flywheel-20261023/s3_mirror/round-02/gp_distilled/distilled_round_02.json",
    "retention_summary": {
      "total_evaluated": 127,
      "pruned_failed": 0,
      "pruned_duplicates": 12,
      "pruned_dominated": 76,
      "retained_top_k": 32,
      "retained_hard_negatives": 7,
      "total_retained": 39,
      "new_hwm_count": 1,
      "pruned_garbage_total": 88,
      "storage_reduction_pct": 69.29
    },
    "s3_sync": {
      "uploaded": true,
      "s3_uri": "s3://veox-jopedime-learning-use2/experiments/flywheel-selfplay-20261023/round-02/",
      "returncode": 0
    }
  }
}
```

## Sample: experiments/flywheel-selfplay-20261023/round-03/persistence_receipt.json

```json
{
  "utc": "2026-10-31T02:04:08.954842+00:00",
  "command": [
    "aws",
    "--profile",
    "veox-jepson-prod",
    "s3api",
    "get-object",
    "--bucket",
    "veox-jopedime-learning-use2",
    "--key",
    "experiments/flywheel-selfplay-20261023/round-03/persistence_receipt.json",
    "/tmp/rd_astra_s3_samples/round-03-persistence_receipt.json"
  ],
  "exit": 0,
  "metadata": {
    "AcceptRanges": "bytes",
    "LastModified": "2026-10-24T07:53:54+00:00",
    "ContentLength": 806,
    "ETag": "\"77a85be4b06c75300b64547b5c8f36f3\"",
    "ChecksumCRC64NVME": "idsrlBaRkbA=",
    "ChecksumType": "FULL_OBJECT",
    "VersionId": "5e06unUgEx6YBaC1bABOLmMujxBWfI7N",
    "ContentType": "application/json",
    "ServerSideEncryption": "AES256",
    "Metadata": {}
  },
  "size": 806,
  "sha256": "ef5e1ace35b5ab298e259dbae4117493b8b7847703902a76fe916726550d3de0",
  "body": {
    "round": 3,
    "local_round_dir": "/mnt/jope-archive/flywheel-20261023/s3_mirror/round-03",
    "scores_jsonl_path": "/mnt/jope-archive/flywheel-20261023/s3_mirror/round-03/scores/retained_scores_round_03.jsonl",
    "distilled_path": "/mnt/jope-archive/flywheel-20261023/s3_mirror/round-03/gp_distilled/distilled_round_03.json",
    "retention_summary": {
      "total_evaluated": 128,
      "pruned_failed": 0,
      "pruned_duplicates": 13,
      "pruned_dominated": 77,
      "retained_top_k": 32,
      "retained_hard_negatives": 6,
      "total_retained": 38,
      "new_hwm_count": 4,
      "pruned_garbage_total": 90,
      "storage_reduction_pct": 70.31
    },
    "s3_sync": {
      "uploaded": true,
      "s3_uri": "s3://veox-jopedime-learning-use2/experiments/flywheel-selfplay-20261023/round-03/",
      "returncode": 0
    }
  }
}
```

## Interpretation

Across these three retention receipts, 383 evaluated records became 115 retained rows (96 passing, 19 hard negatives); 44 duplicates and 224 dominated records were pruned from this output. This is not an inventory of every possible source copy. Mixed score/distilled schemas include test fields (`s3_persistence.py:73-107`); their bodies were left unopened. File names such as `train_only` are not sufficient admission evidence. Historical receipts do not have the newer schema/hash fields of current source. No r882 checkpoint was observed in this checkpoint listing. Current fused retention uses a different prefix; it was not queried.
