# Read-only custody and retention dry run

This tool inventories every version under the existing native program-results
prefix. It classifies each version conservatively, creates a local compressed
sample of exact small-artifact versions, then verifies every member SHA in a
fresh process. The sample never establishes whole-prefix recovery or learning
admission. All results remain non-qualifying.

Unknown keeper decisions, legal holds, audit-salt commitments and noncurrent
versions retain their bytes. Score records and source/input evidence remain
permanent. A fully bound negative keeper record can produce an inert 14-day
`other` classification. This executable exposes only LIST and exact-version
GET; its before-send guard rejects any other S3 operation.

The daily guard uses 25 decimal GB of new-version bytes, 1 decimal TB raw hot,
and 1.5 million PUTs: alarm at 80%, refuse new scoring at 100% or missing
counters. It reports a decision only. Listing timestamps do not establish net
new storage or PUT counts, so those missing production measurements stay
explicit and no worker is paused by this tool.

Cost estimates price measured all-version bytes and physical HTTP requests
using a dated, hashed Ohio rate card. Storage uses binary GB as specified by
[Amazon S3 pricing](https://aws.amazon.com/s3/pricing/); rates come from the
[regional AWS offer catalog](https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonS3/current/us-east-2/index.json).
The storage estimate assumes the first 50 TiB tier and the current UTC calendar
month. Account-wide tiers, discounts, transfer and actual billed USD are
unmeasured. Unsupported storage classes receive no invented price.

Run fixtures without S3 credentials:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 deploy/dl-flywheel/custody_v2/run_offline.py --receipt /absolute/custody/offline.json
```

For an explicitly claimed read-only window, run with an absolute output outside
the checkout and a captured rate card:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 deploy/dl-flywheel/custody_v2/run_dry_run.py --output /absolute/custody/new-run --rates /absolute/custody/rates.json
```

The product profile/account and prefix are fixed. A new output directory is
required. At most 512 current, standard, small ledger/parity/evaluation objects
are selected lexically before body reads. Executables are retained in S3 and
are outside this bounded sample. Missing or failed reads abort the dry run;
failed output is retained and never reused as a successful receipt.
