# H165: annealed gradient noise and adapter-input dropout

Hypothesis and fixed setup: [PREREG.md](PREREG.md). Registered four-arm continuation from a read-only r882 copy with frozen DOPE and 29,211 trainable adapter parameters; three seeds and five disjoint family assessment folds. Primary outcome is equal-family OOF utility versus r882; secondary is the historical 91-context DL-Val panel. No speed measurements or serving promotion.

Status: preregistration; no H165 scientific scores or updates yet. Data-free controls test selection reachability, numerical ties, family weighting, noise scale/decay/stream reproducibility, dropout train/eval behavior and every resource stop. Outcome, three-seed median/CI, verdict and next step will be written after the one authorized run. Large model/data/run artifacts remain outside Git.

Run the data-free suite with `PYTHONPATH=research/flywheel_selfplay/h165_gradient_noise_dropout pytest -q research/flywheel_selfplay/h165_gradient_noise_dropout/test_h165.py`. Launch smoke and study only through `monitor.py` under nice19, with an exact manifest-bound launch JSON and fresh exclusive output. The non-smoke launch also requires verified merged source and the sole attempt marker.
