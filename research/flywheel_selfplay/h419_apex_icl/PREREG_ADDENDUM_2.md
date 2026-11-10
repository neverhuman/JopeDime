# H419 PREREG addendum 2 — a 104M-parameter size replaces the 7.5M run

**Status.** Published on `origin/v1/next` before the run it describes starts. No prediction on a test cohort exists.

**Reason.** Owner instruction, 2026-11-10 11:37Z: look at the trends; if 7.5M to 24M moves in the right direction,
consider 100M.

## 1. The trend (equal hours of training; both runs saw about the same number of tasks)
| Hours of training | Synthetic held-out u, 7.5M / 24M | Real-table win rate vs ridge, 7.5M / 24M | Real-table median u, 7.5M / 24M (ridge 0.630) |
| --- | --- | --- | --- |
| 0.33 | 0.109 / 0.064 | — | — |
| 0.67 | 0.198 / 0.179 | — | — |
| 1.00 | 0.227 / 0.231 | — | — |
| 1.33 | 0.255 / 0.264 | — | — |
| 1.67 | 0.266 / 0.292 | 0.481 / 0.547 | 0.526 / 0.643 |
| 2.00 | 0.277 / 0.297 | 0.491 / 0.575 | 0.597 / 0.645 |

The larger model starts slower, passes the smaller one after about an hour, and is ahead on every real-table measure
at the two readings that exist, with fewer tasks seen (2.7M against 3.0M). That is two readings over 20 minutes: a
direction, not an established scaling law.

## 2. Change
- `ICL_S_r3` (7.5M) is stopped at 0.77 h (checkpoint of 11:32:59Z kept). It was behind ridge on every real-table
  measure and behind the 24M run on all of them.
- **`ICL_L_r4`** on the xbabe3 RTX 3090: size `L` = trunk width 768, 14 layers, 12 heads, row-encoder width 160;
  **104,360,576 parameters**. Trained from scratch with the addendum-1 data mix (column injection; FIT-ICL
  real-table tasks up to 40% of batches). 72 hours, peak LR 1.5e-4, token budget 550,000, row-token cap 22,000
  (stress test: peak 20.5 GB at 600,000 / 24,000 on the 24 GB card).
- `ICL_M_r3` (24M, xbabe1 RTX 4090) continues unchanged.

## 3. Honest expectation
`L` is 4.4 times larger on a card about 0.6 times as fast. Measured step times give roughly 40 tasks per second at
1,024 rows, so over its 72 hours it will see about half the tasks the 24M run sees in 46. It may not catch the 24M
model within this cycle.

## 4. Size rule (fixed now)
Hours count training time; the 24M run's hours include its 1.33 h before the restart. At `L`'s hours 12, 24 and 48,
`L` continues only if its real-table win rate against ridge is at least the 24M run's win rate at half as many hours
(6, 12, 24), minus 0.02. Otherwise `L` is stopped and the result is recorded as "larger is not better at this
compute". The hour-12 on-track rule of addendum 1 applies to the 24M run as written.

## 5. Part A
The kill rules that name ICL-S now apply to the best available checkpoint of the 24M or 104M run. Tests and
verdicts are unchanged.
