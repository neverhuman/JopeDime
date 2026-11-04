# H297 — context-normalized clipped adapter loss

**PREREGISTERED; new arms unscored.**

Hypothesis: per-context median/IQR target normalization, clipping to [-3,3] and .05 smoothing allow the frozen-DOPE adapter to improve near-tie choices. Primary CTXNORM_CLIP; secondary without smoothing; paired baseline-loss CONTROL. Each seed29701–29703 gets identical noise, batches, reference LR/AdamW and eight TRAIN-only updates, with one terminal dev91 evaluation per candidate.

Both required regressions passed: r3175 identity e79a7715c1531ccb8c4055ab96971101cd4034ec3a71d83bca6d25b0e604ca83 at1.1389572403748425 (91/91 choices/utilities), and t1845 control seed184501 after two updates at1.1354387884248622 (91/91 choices/utilities; original order/slate hashes exact). Parameter bytes differ under numeric1; functional regression is exact.

Targets use cost +2 for failed utility -2. Smoothing follows per-context normalization/clipping; raw predicted costs and terminal argmin remain unchanged. DOPE c51a56d0 and every non-adapter state tensor stay frozen; 29,211 adapter trainables only. See [PREREG.md](PREREG.md), [plan.json](plan.json), [loss.py](loss.py) and [test_loss.py](test_loss.py).

DEV_ADVANCE requires all owner gates on all three primary seeds; it authorizes no adoption or fresh-cohort launch. The repeatedly reused dev91 cohort cannot establish independent generalization. Next: run the single frozen attempt after this prereg is verified on origin/v1/next.

Plan SHA256 `df471d5bd52a043d38a9bca8ee15e066b4a4c24a10bb185c66f253e0542743ef`. Six normalization/loss tests passed on xbabe1. Exact-version input/source archive: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H297/astra-t0240/inputs-source.tar.gz`, VersionId `PVoZZo09fFm.MegYE67gQIv1XobpbBik`; all 2674 member hashes verified. [metrics.json](metrics.json) carries S3 receipts.

Resource preparation: assigned60–63 are unavailable in the host CPU set; verified-idle100–103 are used. An initial startup was stopped at five aggregate threads before any score/update; PyTorch was reduced2→1 with the four-thread cap unchanged. A count assertion was corrected before updates because t1845 cycles family blocks (8,190 occurrences,1,219 unique contexts for seed184501). Original receipts are retained. Admitted computation uses CPU only, nice19, numeric1, aggregate<=4, RSS<=4GiB, own copies<=3GiB, and memory/CPU pause guards. No live writes, GPU, fresh/test/holdout/sealed inputs, main/fastlane or hosted CI.
