# Fleet readiness audit — 2026-11-01 8:35 PM Mountain

Observed UTC: 2026-11-02T03:32–03:35Z. Source audit read-only; screen hardcopies only to a unique external evidence directory. No scientific run, optimizer, model import, VM start, owner input, source mutation, cleanup, or process signal performed.

## Current state

| Host | Research GPU state | Resources at last sample | Action boundary |
| --- | --- | --- | --- |
| xbabe1 | RTX 4090, live flywheel PID 2297929, 17,258 MiB, 0% sampled | 91.774% user-visible disk used; 153.57 GB available; load1 18.16; ~38.8 GB memory available | H205 reserved and allowed to share GPU with live flywheel; never signal/renice it. |
| xbabe2 | RTX 3090 idle, no compute PID | 93.332% used; 523.87 GB available; load 68.10/72.56/75.91; ~90.2 GB memory available | H204 reserved, cache <=20 GB. About 36.8 GB growth before 93.8% guard; shared builds can consume it. |
| xbabe3 | RTX 3090 idle, no compute PID | 89.789% used; 99.27 GB available; load1 11.61; ~96.3 GB memory available | H206 reserved. Fails its <=~88% prelaunch requirement reserving search-cache headroom; no launch. |
| xbabe4 | Mac; existing Colima jope-ci Linux aarch64 VM running | Guest 24 vCPU, 269.72 GB RAM, 261.06 GB available, 89.29 GB root free. Mac backing data 98%, ~76 GiB free | Reuse existing guest only after exact CPU method/environment admission and owner capacity check; no expansion/new VM. |

All Linux shells reported nproc=102 allowed CPUs, while old research guards use 128 online cores. Retain each wave's preregistered guard and distinguish affinity/online counts.

## Actual flywheel progress

Latest completed round sampled: 3394; 98.07 seconds/round, 3.57 seconds training, 71.07 seconds GP, 10.26 seconds development gate. It is real GPU adapter training (2 optimizer steps, parameter L2 update 0.065745), while DOPE is frozen. Exactly 29,211 trainable parameters; 444,168 DOPE parameters frozen. Candidate development utility 1.130591866723114; adapter rejected and restored previous best. r882 reference 1.1377941898164525. Regression Elo unavailable because Golden45/panel fitting is outside the current per-round native-CV budget. Fixed, repeatedly selected 91-context development scores are not independent confirmation. GPU0% instantaneous samples do not prove CPU-only training.

## Ownership and readiness

- H204: gpu_tokbench_agy active in native Gemini screen, reading n2022 directive and implementing bigger proposer. No launch PID, prereg PR, committed H204 trainer or qualified ready command observed. Can smoke label on already-admitted 205-family TRAIN pool; D4 extension waits real-catalog admission.
- H205: rd_argon active, handling new FRESH/STATIC-DATA/STATIC-TEACHER design. Existing JRB-01 code is a different joint/regression-only/binary-only study (54 jobs across 3 seeds/6 fits), so launching JRB-01 alone does not prove H205 causal flywheel effect. New arms/equal compute need explicit preregistration. n2022 allows initial static infrastructure/native r882 parity before fresh-family admission; FRESH waits corrected catalog PR merge/admission PASS.
- Catalog unblock: luna_v1_tel active in own tel-01 lane, fix f0df3c05f followed by main merge fb9361329. Native screen showed 187 tests passed +2193 subtests; Ruff/lint/READY pending.
- DOPE/thread unblock: gpu_fused_redteam_codex active, inspecting freeze callers and lane sweep. Research-only opt-in must retain serving/flywheel refusals; <=4 numeric threads, CUDA runtime helpers exempt. No READY PR observed. Same owner then writes H206 prereg.
- H206: queued only; owner gpu_fused_redteam_codex, backup rd_astra. Disk guard with search reservation not met.
- H207: rd_astra active, singleton TTA followup; commit/bundle 9f149a88dd97b29b926488528f3e4c13c1862963 observed. No reserved GPU; preserve H204/H205/H206 allocations.
- H203: stopped at4/45 CPU fits by load guard. Owner retains one resume requiring load1<50 and load5<60; current values fail. Do not duplicate/bypass deadline.
- All observed source owners active; no duplicate writers or screen input delivered by this audit.

## Scientific and storage contract

Authority: /home/ubuntu/nudges/hwave_t2022_plan.md, exact launch authority string “Jepson via Grok Bot, card answer 8:22 PM MT”. Launch-before-review allowed, but committed PREREG.md and open PR required before scored inference. Native r882 parity 91/91 exact (1.1377941898164525), pooled family cross-fit LCB95>0, <=10% harmed families, final separately admitted sealed confirmation. Research-only; no serving promotion inferred.

Canonical Git destinations: research/flywheel_selfplay/h204_bigger_proposer/, h205_fresh_vs_static/, h206_search_distill/. Each gets PREREG.md, README.md with verdict/parent comparison/next experiment, metrics.json, manifest.json with immutable S3 evidence versions/hashes. Bulk outside Git. S3 prefixes experiments/rd_breakthrough_20261101/H204/, H205/, H206/ in veox-jopedime-learning-use2.

H204 ~30 GPU-hours +200 CPU-core-hours; H205 ~36 GPU-hours; H206 ~400 CPU-core-hours +18 GPU-hours. These are inherited n2022 plan budgets, not new admissions/resets. Nice19, CPU<=85%, disk93.8%, cache caps retained. Current n2022 lanes never self-merge; designated merger owns exact-head review/integration.

## Receipts

Local bounded snapshots; runner JIT configurations redacted. Remote screen copies: /home/ubuntu/jope_plan/evidence/codex-fleet-readiness-20261102T0335Z/. Source rules read fully from xbabe2:/home/ubuntu/JopeDime/AGENTS.md. This packet authorizes no launch by itself.
