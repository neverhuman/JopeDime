# H20 fixed wide state correction — completed, benefit not established

All 12 fresh adapters completed 2400 updates using six unchanged frozen bases.
Later separate independent replay passed with limits. Both prespecified primary
intervals include harm, so the fixed incremental-state criterion is not met.
Positive raw-fitness differences are adverse; lower is better.

| Primary contrast | Raw selected-fitness difference | Simultaneous 97.5% interval |
| --- | ---: | --- |
| Trained minus frozen base | +0.0232406036 | [-0.7472265233, +0.6894384189] |
| Trained minus matched random | +0.1744534973 | [-0.0091406014, +0.4830348439] |

Primary utility averages three seeds per context, contexts per component, then
162 components equally. Intervals use 100000 paired component draws, 98.75% each
for 97.5% familywise coverage. No selected seed/checkpoint/ensemble substitutes.
Failed picks were 7/2550 base, 11/2550 trained and 6/2550 random; literal counts differ
from the component-weighted rates in [all results](results.json).

[All 12 TRAIN curves](curves.json) retain 36 actual snapshots at 0/100/200.
OOF was evaluated only after all 12 fits closed; it has final metrics, not a
validation learning curve. TRAIN loss fell in every adapter, which does not
establish better hard GP choices. See the [frozen recipe and interpretation](PLAN.md).

The range is 10 times H18's, but widening also changes the initial score Jacobian
and optimizer/clipping dynamics. It is not a pure reachable-set experiment.
Base failure logits remain frozen; extra failed picks reflect changed choices,
not gate-head retraining. The 112 state coordinates include readout summaries.

The 850 contexts/162 components/979200 archived candidates are historically exposed
TRAIN/internal crossfit. CAL/OLDDEV and known reservations stay excluded.
No new independent validation, source-volume admission or promotion is claimed.
The original metadata QA FAIL/CLOSED_INCOMPLETE remains preserved. A later,
separate source-pin amendment and 137.773 s replay verify the same models with
zero new updates; their status does not rewrite the original closure.

Next priority is the [H21 exact-source admission plan](../h21/PLAN.md), rather
than an outcome-selected bound sweep. Root owns Git custody/publication;
this staging packet makes no merged-product or remote-storage durability claim.
