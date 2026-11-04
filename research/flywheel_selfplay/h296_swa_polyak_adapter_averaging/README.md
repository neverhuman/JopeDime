# H296: SWA / Polyak averaging of offline TRAIN-only adapter trajectories

**Status: preregistered, not executed.** [PREREG.md](PREREG.md) is the frozen protocol. It is published on `v1/next`
before any H296 arm is trained or scored.

**What it tests.** Three 16-update TRAIN-only trajectories (seeds 29601–29603; 4 cosine-restart cycles of 4 updates)
start from an r3175 copy, using t1845's harness and pinned helpers. The **primary SWA_CYCLE** (the average of the 4
cycle-end iterates) is compared against the **paired final iterate** and against r3175 on dev91. EMA_POLYAK and SOUP_RET
(r3100 + r3175) are descriptive.

## Files

| File | Contents |
|---|---|
| [h296.py](h296.py) | the prepare / regress / run stages (t1845 harness, own copy directory, pinned helpers) |
| [pinned/](pinned) | helper modules from t1845's source commit `e5c5b852` |
| [analysis.py](analysis.py) | the strict primary rule, Elo and family-bootstrap statistics |
| [test_h296.py](test_h296.py), [manifest.json](manifest.json) | data-free controls; pins |
| [metrics.json](metrics.json) | the metrics schema placeholder |
