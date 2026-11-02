# H209 — GPU utility-delta calibration

**CLOSED INCOMPLETE; no scientific verdict.** The single registered xbabe2 GPU
attempt stopped with a resource-logging reentrancy error before training.
No retry was launched; the persistent attempt claim remains consumed.

| Observation | Result |
|---|---|
| Executed source | `d609dab46dc725925008e750e0813bfe40b44ab4` (PR #1743) |
| Attempt window | 2026-11-02 07:02:17–07:03:07.701 UTC |
| Wall time | 50.241 seconds supervisor; 44.374 seconds worker |
| Native r882 baseline | All 91 ordered utilities bit-exact; mean **1.1377941898164525** |
| Optimizer updates / completed heads | **0 / 0** (planned 3,840 / 30) |
| B selection lock / A endpoint scoring | Neither reached |
| Adoption, utility gain, Elo, U4 | **Unmeasured** |
| Terminal state | Worker exit 1; GPU released and qualification lock available |

The SIGALRM callback entered `check()` while a synchronous call was writing to
its buffered `resources.jsonl` stream. This raised `RuntimeError: reentrant call
inside <_io.BufferedWriter ...>` during TRAIN native-cost generation. It was
not a resource-limit refusal or evidence against the scientific hypothesis.

[REPORT.md](REPORT.md) records the failure, resource observations, CPU reproduction
and verified S3 custody. [metrics.json](metrics.json) keeps unmeasured outcomes
null. [manifest.json](manifest.json) links the closeout and the unchanged
[execution manifest](execution-manifest-attempt1.json); its historical source
hashes refer to the executed commit, including the original README.

The [preregistered protocol](PREREG.md), Python executor, [fold identities](fold_manifest.json)
and [real-S3 provenance](provenance.json) remain unchanged. This was historical
static-pool research with frozen r882/DOPE, not fresh native-CV labels or serving
qualification. The baseline success does not establish trained-head improvement.

Before any future attempt, repair and review asynchronous resource logging,
add a reentry regression test, and obtain a new prospective run decision.
The existing 13 CPU fixtures passed but did not exercise signal-time reentry.
No retrospective retry or budget reset is authorized by this closeout.
