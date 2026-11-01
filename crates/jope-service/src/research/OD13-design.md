# OD13 reveal selection in research capture

## Approach

Apply the accepted OD13 rule inside `CaptureContext`: for rounds after zero,
rank the actual measured admitted pool by ascending surrogate F and select the
first four. Keep source order for equal F values. Round zero has no reveals.
The provider supplies program observations only; it cannot substitute a
different reveal rule. Validate finite measurements before ordering.

## Alternatives rejected

- Trusting a provider-returned index list would leave the accepted rule
  unenforced at the capture boundary.
- Re-sorting recorded reveals after selection could lose their measured-pool
  identity and order.
- Inferring F from an objective, gate, or user score would change the
  measurement contract.

## Risks and limits

This binds selection to the measured input order supplied by the engine. It
does not qualify the evaluator, establish how that pool was formed, or provide
the missing request-bound measurement receipts. Synthetic controls remain
non-qualifying. The positive engine-to-Python Done-when is still required.

## Spec and owner ruling

Implements the accepted S5 OD13 recommendation for the research capture
boundary under spec §13.2. Equal-F ties retain input order. No public schema,
worker protocol, discovery behavior, CLI, or default admission policy changes.
