# H419 PREREG addendum 3 — the 104M run scaled across three GPUs by parameter averaging

**Status.** Published on `origin/v1/next` before any scaled worker starts. No prediction on a test cohort exists.

**Reason.** xbabe2 and xbabe0 rebooted at 12:02Z and 12:05Z (not by this wave); their RTX 3090s are now usable and
idle. Owner instruction relayed by ENG:watcher (05:11 and 05:19 MST): take full advantage with the 104M run, from its
latest checkpoint, without discarding the live run; publish an addendum before any change to the published
`ICL_L_r4` configuration.

## 1. Why not synchronous multi-GPU training
One GPU per host, joined by 1 Gb Ethernet. A synchronous step would move the 104M model's gradients (417 MB) between
hosts every step, about 4 seconds against a 0.5-second step. xbabe0 also runs a different torch build (2.10 against
2.7). So the faithful way to make this one run faster is local training with periodic parameter averaging.

## 2. Method
- **Workers.** `W1` on xbabe0 and `W2` on xbabe2 start first; `W0` is the live xbabe3 run, which joins at cut-over
  (§3). Every worker runs the published `ICL_L_r4` configuration unchanged: size `L` (104,360,576 parameters), peak
  LR 1.5e-4 on the same schedule, token budget 550,000, row-token cap 22,000, the addendum-1 data mix, the 72-hour
  horizon.
- **Start.** `W1` and `W2` clone `ICL_L_r4/ckpt_last.pt`: weights, EMA, optimizer state and schedule position
  (`icl_train.py --clone`). Each worker draws its own tasks (seed 2 and 3; `W0` keeps seed 1).
- **Averaging.** Every 300 seconds of wall-clock each worker publishes its weights; a coordinator on xbabe2
  (`icl_sync.sh`, `icl_avg.py`) averages the weights that arrive within 100 seconds and returns the mean, which each
  worker loads before continuing. Optimizer moments and EMA stay local. A worker that receives no mean within 240
  seconds continues on its own weights, so a dead host cannot stall the others.
- **What does not change.** Hours still count training time on the `L` schedule. The group sees about three times
  the tasks per hour. The size rule of addendum 2 is unchanged.

## 3. Cut-over (the live run is not discarded)
The single-GPU run on xbabe3 keeps training until all of these hold: `W1` and `W2` are training, at least 3
averaging rounds have completed, each has written a checkpoint, and check (b) below passes. Then the xbabe3 process is
restarted as `W0` from its own latest checkpoint and joins the rounds. If the checks fail, the group is stopped and
the single-GPU run simply continues.

## 4. Checks and stop rules (fixed now)
- (a) Every round logs each worker's relative distance to the mean and its training loss before the mean is loaded.
- (b) **Before cut-over:** at matched schedule hours, the group workers' training loss is not above the single run's
  by more than 0.05.
- (c) **At any time:** if a worker's training loss in the minute after loading the mean is above its loss before by
  more than 0.1 in 3 consecutive rounds, the group stops and one worker (or the latest checkpoint) continues alone.
- A mechanism smoke on the two idle GPUs with the 7.5M checkpoint (`SMOKE_SYNC_*`, discarded) preceded this addendum:
  3 rounds completed, both workers loaded the mean each time, relative distance to the mean 0.004-0.012, and the
  training loss moved within its batch noise (5.04-5.13) across the loads.

## 5. Budget
Bounded by the published horizon: at most three GPUs for the remaining 71 hours. Latest checkpoint every 20 minutes
on every worker; named checkpoints every 2 hours on xbabe0 and xbabe2 and every 6 hours on xbabe3 (82 GB free
there). Exchange files have fixed names and are replaced each round; checkpoints are never replaced by them.

## 6. Not in this addendum
xbabe4 (Mac Studio, no torch installed) is not used here. If it can run a worker, that is a further addendum.

## 7. Files
`icl_train.py` gains `--clone`, `--name`, `--sync-dir`, `--sync-s`, `--sync-wait`, `--ckpt-s`, `--named-h`;
`icl_avg.py` (mean of weights); `icl_sync.sh` (coordinator).
