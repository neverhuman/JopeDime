import type { Observation, Snapshot } from "../src/types";

export const observedAt = Date.parse("2026-10-21T12:00:00Z") / 1000;
const pin = "a".repeat(64);
const observation = <T>(value: T): Observation<T> => ({
  status: "observed", source: "synthetic-browser-fixture", observed_at: observedAt,
  sha256: pin, value,
});

// Artificial UI data only. No fixture metrics establish scientific capability,
// real optimizer updates, custody, or production admission.
export function observerFixture(): Snapshot {
  return {
    schema: "jope.observatory/v1", generated_at: observedAt,
    service: { version: "0.3.0-fixture", refresh_seconds: 5, mode: "read-only", production_authority: false },
    runs: [{
      id: "synthetic-rendered-ui-only", config_sha256: pin, identity_matches: true,
      config: { host: "xbabe2", neural: { max_updates: 100, batch_size: 4, max_epochs: 1 } },
      progress: observation({
        run_id: "synthetic-rendered-ui-only", state: "paused", at: observedAt,
        optimizer_steps: 2, completed_epochs: 0, total_train_presentations: 8,
        unique_train_data_runs_seen: 8, unique_validation_data_runs_scored: 4,
        model_generated_programs: 0, novel_programs_executed: 0,
        published_results: 0, comparison_games: 0, performance: {}, outbox: {},
        latest_full_validation: null, scope: "synthetic UI fixture only",
      }),
      curve: [0, 1, 2].map((step) => ({
        step, train: 4 - step * 0.2, validation: 4.5 - step * 0.1,
        at: observedAt, panel_sha256: pin, objective: "synthetic-fixed",
        terms: { training: {}, validation: {} },
      })),
      counts: { partitions: { train: 400, validation: 40 } }, membership_sha256: pin,
      checkpoints: [{
        id: "synthetic-step-2", identity_matches: true,
        evidence: { source: "synthetic-fixture", sha256: pin, status: "fixture" },
        manifest: { state_sha256: pin, parameter_sha256: pin, state_bytes: 1024 },
        verification: "Synthetic checkpoint, not a runtime or custody proof",
      }],
      checkpoint_count_observed: 1, source_identity: { source_commit: pin }, terminal_receipt: observation(null),
    }],
    storage: {
      catalog_fallback: false,
      evidence: observation({
        buckets: [{ bucket: "synthetic-bucket", region: "fixture", role: "UI only", bytes: null,
          objects: 0, observed_at: "2026-10-21T12:00:00Z", status: "unavailable" }],
        observed_bytes: null, observed_objects: 0, complete: false, scope: "Synthetic partial coverage",
      }),
    },
    datasets: [], fleet: [],
    gp: { unique_programs: 1, sources: [], families: { synthetic: 1 }, status: "observed", global_complete: false,
      scope: "Synthetic UI fixture", programs: [{
        id: pin, dsl: "add(x0, 1)", operators: ["add"], family: "synthetic",
        context_occurrences: 1, source: "synthetic-fixture", scope: "UI only", score: null, score_note: "Unmeasured",
      }] },
    research: { scope: "Synthetic UI fixture", waves: [{
      id: "synthetic-wave", card: { interpretation: "UI fixture only", capture_status: "fixture", archive_status: "unqualified" },
      assertions: [], manifest_sha256: pin, metrics_sha256: pin, publication: "fixture", archive: "unqualified",
      source_commit: pin, training_identity_sha256: pin, archive_manifest_sha256: pin, archive_roots: [],
    }] },
    limits: {},
  };
}
