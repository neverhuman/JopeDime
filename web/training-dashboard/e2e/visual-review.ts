/**
 * Source review ledger for the rendered UX screenshots. This records a manual
 * visual comparison alongside the automated visual regression snapshots. Each visual review is
 * limited to the synthetic fixture and the desktop/mobile captures in the
 * adjacent visual_review.json receipt. The implementing agent performed this
 * review; it is not independent human approval or live-service qualification.
 */
export const visualReviews = {
  overview: "pass: summary cards align on desktop and reflow into readable groups on mobile; fixture values are labeled synthetic.",
  training: "pass: loss chart and checkpoint summary remain legible; provenance content wraps within the mobile viewport.",
  fleet: "pass: unavailable telemetry is explicit and node cards retain consistent alignment and spacing.",
  "storage-datasets": "pass: dataset status and storage details remain distinguishable; dense values wrap without page overflow.",
  "gp-library": "pass: program cards and inspection action remain clear; the long program representation stays within the viewport.",
  "results-lineage": "pass: result status and lineage links remain ordered and readable at both captured widths.",
  "checkpoint-dialog": "pass: dialog has a clear title and close action, stays inside the viewport, and preserves the underlying training context.",
  "interrupted-feed": "pass: stale-feed state is visibly distinguished from live telemetry and retains its recovery guidance.",
} as const;

export type VisualReviewState = keyof typeof visualReviews;
