# H125 reproducibility bundle

Preregistration commit: `6424a539a295aa02a5fb831c19b70c19d9be07b1` (PR#1083; merged `d3f22a1c5f6c679e3e27f6d5adf014169bab2303`). Shared H97/H105 source SHA pins are in manifest.json. The executed H125 CPU wrapper was frozen before execution as exact-hash source in the execution archive; it was not a previously merged runner. This bundle reproduces the reported readout without new training or a new seed attempt.

- Final manifest: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H125/astra-t0955/manifest-v2.json`; VersionId `YxMNr.xPc8JYEWyREIA7_a5IBNJXSyxN`; SHA256 `50cc7893b3d7285f968865f0700c3777c2f7a8d8023d5395d6542e71276cffb7`.
- Evidence: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H125/astra-t0955/closeout-evidence.tar.gz`; VersionId `FsDFlWk_rhbgnlLRVNQ9hL0bBQslesQ7`; SHA256 `c1df6b0589076aac2b89daaa341e8277163f85a66e4318ab07172fe580cf14a8`; 6232875bytes.
- Full source/inputs/protocol/fold/host/resource manifest and archive member hashes are included. Exact-version GET and every archive member hash were verified.
- `run/roster.json`, `run/calibration.json`, all three plans, regression/reference predictions, nine cell JSONs/adapters and update-by-update loss curves supply the calculation inputs when the wave completed. An incomplete run retains only its actual artifacts and failure evidence.
- Seeds67–69; fold seed125067; bootstrap seed125000,10000 paired family draws. CPU-only, source r882 SHA `0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`, frozen DOPE `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`.

Fetch the evidence by its exact VersionId into custody outside any repository, verify the SHA, extract there, and verify `closeout-inventory.json` before use. The archive contains no credentials and no dataset payloads. Do not invoke monitor.py again: this was one finite registered wave. The following receipt replay reads only the admitted assessment results already in the bundle:

```python
import json
from pathlib import Path
from science import ARMS, SEEDS, summarize
root = Path("run")
reference = json.loads((root / "reference.json").read_text())
cells = [json.loads((root / f"{arm}-{seed}.json").read_text())
         for arm in ARMS for seed in SEEDS]
replayed = summarize(reference, cells)
recorded = json.loads((root / "metrics.json").read_text())
assert all(recorded[key] == value for key, value in replayed.items())
```

Execution paths and environment are preserved in source/monitor/audit receipts. Original data stayed in `/home/ubuntu/jope-data/h50/input`; unique checkpoint copies and all bulk outputs stayed under external custody. No serving or GPU admission follows from this bundle.
