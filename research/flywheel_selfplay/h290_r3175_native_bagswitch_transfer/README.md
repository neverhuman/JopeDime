# H290 r3175-native BAG_SWITCH transfer

PREREGISTERED; scoring has not started. Hypothesis: fitting unchanged H279 BAG_SWITCH on r3175-native CPU top8 DL-Train pools improves SEMI_FRESH transfer against the frozen H284 r882-fit reference.

All three native/placebo fits are frozen and reload bit-exactly on all1219 training contexts. CPU DL-Val91 baseline and one H280 cell reproduced exactly. Primary H286 has200/200 families; secondary H283 has29/60 built families,31 unbuilt. The coverage amendment was fixed before scoring. No ODD/v3/remainder, GPU, neural updates/forwards or writer changes.

[Preregistration](PREREG.md), [original training protocol](FIT_PROTOCOL.md), [fit summary](fit-summary.json), [fit custody](fit-custody.json), [prepared input custody](input-custody.json), [manifest](manifest.json), [metrics](metrics.json). A PASS would remain shadow-only and require separately authorized fresh confirmation.
