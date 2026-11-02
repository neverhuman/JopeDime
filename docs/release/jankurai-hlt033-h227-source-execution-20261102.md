# HLT-033 H227 Git-source execution exceptions

The post-policy report `rerun/j_27137b5fb.json` (source commit
`27137b5fb8271c8314102691fcc00d1079a545eb`, report fingerprint
`sha256:41a3ef2ba66482aac5122c6c623b2a560e78584a74e9fce1b1e0b3bb4a91d34d`)
flagged three exact H227 files for executing Git source. They no longer accept
unverified or arbitrary source identities before execution:

- `launcher.py` requires a full 40-character commit ID, verifies that it resolves
  to that exact commit, and checks ancestry under
  `refs/agent/res08/h227/main` before reading or executing a Git blob. The child
  bootstrap repeats those checks before loading its backend and runner blobs.
- `backend.py` requires a SHA-256 pin for every loaded source blob and verifies
  the digest before compiling or executing it. The pins are read from H227's
  preregistered `manifest.json` (`source_parent` is
  `31207ade1cb12da1df2036d1adbbfcc50c3d8dce`).
- `test_h227.py` extracts one named pure function from the manifest-pinned H212
  runner and verifies the source digest before compilation.

The exact fingerprints and audited source-file identities are:

| Fingerprint | Exact path | Audited SHA-256 at report source | Updated source SHA-256 |
|---|---|---|---|
| `sha256:c43e57f12cbc2c33652d968b8b7c55c945cedf0d886f55445ad0bfbb77a8bc42` | `research/flywheel_selfplay/h227_pool_headroom/backend.py` | `cbd651babb62e4f3b15ea0436018129457035d5a52725ed54a60511e9b2b2a36` | `5a38bf19195effe5a471c9dc209c26eefd8fe72762298105e1d35b55a9643eaf` |
| `sha256:a7656dd78423388734077b39a9db4ca190d1eee419621543a816047fdc2354c3` | `research/flywheel_selfplay/h227_pool_headroom/launcher.py` | `9f5f1539dce8d7264bee4f4fb62ef6b0a975ad54019700a8f10e7d39a719f72f` | `c5f68dfbc4069ba6399de759471bdbeeb54acac5eda84ee4d610ad0c2370ec4b` |
| `sha256:09fea9f7943d378dbbe22f402dfc656458c7f875a10ced63d2b81f1d5cba0527` | `research/flywheel_selfplay/h227_pool_headroom/test_h227.py` | `a7159ad2211c381fece0157ec6cce9062d5f3ca5377328ffe11396941993b938` | `e47784885ecf564eb6941f66b3ed7bf25c5b2391029bc8f3ca362105ebda0569` |
| `sha256:78c47028cdd5f4a341ee97f6feca52e248c9f4444bf020375fc3b6d739c807fe` | `research/flywheel_selfplay/h227_pool_headroom/test_h227.py` | `a7159ad2211c381fece0157ec6cce9062d5f3ca5377328ffe11396941993b938` | `e47784885ecf564eb6941f66b3ed7bf25c5b2391029bc8f3ca362105ebda0569` |

The scanner reports the intentional `exec(compile(...))` mechanism even though
the source is now constrained and checked before execution. The three exact-path
policy dispositions cover only this false-positive class; no neighboring H227
files, study inputs, manifest, preregistration or data are excluded or changed.
Revisit the dispositions if the commit-identity, ancestry or hash checks are
removed or relaxed.
