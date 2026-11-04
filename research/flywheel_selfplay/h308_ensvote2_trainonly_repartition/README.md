# H308: H303 NESTED ENS_VOTE2 through a TRAIN-only loader, on two new outer partitions

**Status: preregistered, not executed.** [PREREG.md](PREREG.md) is the frozen protocol.

**Loader.** [trainonly_loader.py](trainonly_loader.py) reads only the verified TRAIN prefix of the mixed H58 cache NPZs; no DL-Val
feature row is decoded. [test_h308.py](test_h308.py) holds the negative fixture.

**Partitions.** P1 = SHA256("H308-outer-v1" NUL family) mod 4; P2 = SHA256("H308-outer-v2" NUL family) mod 4.

**Elo:** reference = r882 slot0 (DL-Train OOF); r3175-anchored Elo unmeasured.
