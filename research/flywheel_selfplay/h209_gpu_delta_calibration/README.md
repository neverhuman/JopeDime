# H209 — GPU utility-delta calibration

**NOT RUN.** [Preregistered protocol](PREREG.md), [source/input pins](manifest.json),
[fold identities](fold_manifest.json), [real-S3 provenance](provenance.json).

Train a small improvement head over frozen r882 inputs; compare raw choices,
B-gated choices and a shuffled-label control. This reacts to H204's reported
zero adoption while retaining its scientific qualification limits. Static
historical pools only; no fresh-label or serving-promotion claim.

[runner.py](runner.py) implements 30 CUDA fits / 3,840 updates and locks all B
choices before A endpoints. [launch.py](launch.py) requires an externally merged
preregistration, exclusive xbabe2 GPU ownership and disk<=93.8%, with one hour
and one attempt. Tests are CPU-only; actual CUDA execution remains pending.

Acceptance: `python3 -m pytest -q research/flywheel_selfplay/h209_gpu_delta_calibration`
with single-thread numeric environment, followed by the repository records gates
and Ruff. [test_h209.py](test_h209.py) exercises actual optimizer/forward change,
role refusal, baseline-vector identity, threshold boundaries and resource guards.

No H209 utility gain, Elo, adoption rate or scientific verdict exists yet.
