# H438 PREREG addendum 1 — a code fix before any result: the sync record's distance is a list

**Status.** Published on `origin/v1/next` before any result of this wave exists. The first run (17:45Z, after
4 of 4 gates) stopped with an error in its first function, before it computed or printed anything: the result
file was never created and its text output is empty (0 bytes).

## 1. What failed
`lag_h438.py` read the `dist` field of an averaging record as one number. It is a list: the relative distance of
each averaged worker to the group mean (`icl_avg.py`), two or three numbers per round. The line that fits a trend
to it raised a type error.

## 2. The change (one function, nothing else)
In `facts()`, the distance of a round is now the mean of that list. The gaps, the loss profile after a sync, the
verdict rule, the window and every threshold are unchanged, and none of them uses this field. The distance stays a
reported number (PREREG section 5), not part of the verdict.

## 3. What was looked at between the PREREG and this addendum
The error message; the type and length of the `dist` field over the sync records of one log (342 lists of three,
9 lists of two); `icl_avg.py`'s one-line description of it. No evaluation value, loss or distance value.

## 4. Gates
`gates2_h438.sh` repeats the four gates on the new stage and adds: this addendum is published and no result
exists; the first attempt's exit status is kept (`logs/run_times_attempt1.json`).
