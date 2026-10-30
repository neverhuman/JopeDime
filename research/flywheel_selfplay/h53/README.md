# H53: learn-from-zero pilot

## Preregistered hypothesis

A randomly initialized 29,211-parameter outer adapter, trained from scratch against the frozen DOPE reference program, can learn useful ranking behavior without inheriting r882 adapter weights. This is a bounded development pilot, not a serving or promotion claim.

## Cohort and arms

Use the pinned offline cohort only: 1,219 DL-Train contexts in 205 families and 91 DL-Val contexts in 19 disjoint families. Do not read sealed/holdout or outer payloads and do not access a live loop. Arms are seeds 12, 13, and 14 crossed with default class initialization and a 0.1x scale applied only to the freshly initialized trainable adapter. The architecture, fixed source program, and split are identical across arms. No r882 or DOPE adapter state is transferred into the trainable adapter; DOPE c51a56d0 remains frozen and is checksummed before and after each cell. The r882 checkpoint is used only as a separately scored comparison baseline and is not loaded into the candidate adapter.

The first pilot is capped at 30 minutes total. It uses three training rounds per cell, one optimizer update per round, family-balanced batches, and per-round DL-Val scoring. The DL-Val curve is diagnostic-only and cannot select a seed, scale, checkpoint, or continuation. All six cells are reported; no best-seed selection. Later training is a separate decision after the pilot and requires a new explicit wave record.

## Preregistered decision rule

H-wave admission (t537): admit only if the one-sided 95% family-bootstrap lower confidence bound of the mean family gain exceeds 0.001, the median family gain is at least zero, and sealed holdout remains untouched. Report the previous median-plus-LB95 rule alongside it. Family gains versus r882 are continuity diagnostics because r882 has seen the DL-Train cohort; they are not independent evidence. DL-Val metrics and Elo are diagnostic-only and repeated per-round reads carry selection bias. Ties fail.

## Required reporting

Report every seed/scale's DL-Val utility curve by round, round reaching utility 1.1377942 or best reached within the pilot, family-bootstrap median and CI, Elo versus r882 (or an explicit reason it is unavailable), admission result under both rules, and resource/identity checks. No best-of-seed summary may replace per-seed results.

## Pilot status

Preregistration only. No H53 training outputs have been read at commit time.
