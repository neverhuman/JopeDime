# H212 — frozen-r882 candidate-pool width sweep

**Status: PREREGISTERED; NOT RUN.** The exact protocol is in [PREREG.md](PREREG.md).
No candidate arm has been scored. This offline study compares the frozen r882
selector over width-specific GP proposal calls at 16, 32, and 64 candidates. It makes
no model update or serving change.

The approved real-S3 H139/H195 cohort is pinned to 1,219 DL-Train contexts across
205 families and 91 DL-Val contexts across 19 disjoint families. A 91/91 native
r882 parity check is required before TRAIN treatment scoring. The experiment is
one CPU attempt on xbabe2, capped at three hours and guarded by the recorded
resource limits.

No result, metric, verdict, or promotion claim is made by this preregistration.
