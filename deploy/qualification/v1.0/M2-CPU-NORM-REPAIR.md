# M2 CPU normalization repair

The CPU adapter previously reduced LayerNorm means and variances in float32.
For wide rows, rounding can discard variation or perturb the normalized program
embedding. The resulting anchor prior failed four of 1,600 frozen M2 costs even
though the multimodal corrections and structural conditioning agreed.

Compute CPU normalization, including its affine transform, in float64 and round
once back to the incoming model dtype. Stored tensors, model architecture,
epsilon, non-CPU arithmetic, and DOPE inference are unchanged. This is an
inference arithmetic repair; it performs no fitting or parameter update.

The regression checks translation invariance for 558-channel rows whose quarter
increments remain exactly representable after a large offset. The original code
fails with normalized values -2.5375578 versus -0.6469721. The repair preserves
the translation and the affine bias of constant rows. This property does not
depend on a particular checkpoint or the failing golden costs.

The same real TRAIN fixture replay then covers all 50 contexts, 100 slates and
1,600 costs at unchanged absolute/relative tolerances 1e-5/1e-4. Four cost
failures become zero; no admission set changes occur. Temporary layer tracing
used to locate the first divergence is excluded from this change.

Rejected alternatives include increasing tolerances, changing golden outputs,
adjusting final costs, and changing checkpoint weights. More precise reduction
may change CPU results by a few float32 units; the full frozen replay and adapter
tests bound that risk. GPU parity needs its own qualification. This source proof
does not upgrade RC1 or qualify RC2: the new signed bytes still need all gates
under the T0710 rules and spec section 14 M2.
