# HLT-033 immutable Git source execution exceptions

These three exact runner paths contain `exec(compile(...))` because each study
loads its own Python modules from Git blobs. The HLT-033 dynamic-code finding
does not represent execution of attacker-controlled input: each launch binds
the requested source to an immutable Git commit, and the loader obtains the
module bytes from that commit. The exact audit fingerprints below were reviewed
against the post-policy report `j_27137b5fb` (source commit
`27137b5fb8271c8314102691fcc00d1079a545eb`).

| Fingerprint | Path | Current audited runner SHA-256 | Source identity check |
|---|---|---|---|
| `sha256:1ee87131609d65b95dd5d1686b6fe715aa04dec55ee3f6f6a50294780babc9da` | `research/flywheel_selfplay/h222_lipschitz_adapter/runner.py` | `008159b546643094d0ef9d784f7059c6266cb6fde81f5ec8c64cd322082f629c` | `args.source` must equal the launch receipt's `source_commit`; the manifest blob digest must equal the launch receipt's `manifest_sha256`; the source commit must be an ancestor of `origin/main`. The runner blob at the pinned execution commit `12da44ef5fb97d74084d652109690c49f1ace854` has the same digest. |
| `sha256:17c505fe7ac01ce076aa132a71dad3b9596c568717215019db003c49f085ddc1` | `research/flywheel_selfplay/h224_learned_switch_gate/runner.py` | `ca3a8aee72a6158e600671a07e9870ac4db1b219c9a915cec95b5ecc2d11e64d` | The attempt claim binds `source_commit` to `args.commit`; the loader reads Git blobs at that commit and checks source module hashes from its manifest. The runner digest matches `manifest.json`'s `source_sha256.runner.py`. |
| `sha256:077c7542622c3787abf818a8ad5e17f1d3f81a0129bee8af673963daf9a76779` | `research/flywheel_selfplay/h225_multistep_regret/runner.py` | `a5642759d04fb476524bbfc9d44afbd6defadcd85964ea495cc7b1a03eaa2735` | The source commit must be an ancestor of `origin/main`; setup loads Git blobs from that commit and checks manifest-pinned module hashes. The runner digest matches `manifest.json`'s `source_sha256.runner.py`. |

The policy excludes only these three exact runner paths from scoring. It does not
change their bytes, alter the finding rules, or exclude any neighboring study
files. Revisit the exceptions if a runner stops binding executed source to its
committed identity.
