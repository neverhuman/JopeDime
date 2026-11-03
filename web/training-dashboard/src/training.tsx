import { useState } from "react";
import { bytes, count, number, text, when } from "./types";
import type { GP, Run } from "./types";
import {
  Badge,
  Hash,
  Metric,
  Empty,
  Details,
  LossChart,
  Dialog,
} from "./shared";

export function Training({
  run,
  compact = false,
}: {
  run: Run | undefined;
  compact?: boolean;
}) {
  const [checkpoint, setCheckpoint] = useState<string | null>(null);
  if (!run)
    return (
      <Empty>
        No campaign roots have been configured. Connect an owned run to see live
        training.
      </Empty>
    );
  const p = run.progress.value;
  const n = run.config.neural;
  const goal = number(n.max_updates);
  const train = run.counts.partitions?.train;
  const epochBatches =
    train && number(n.batch_size)
      ? Math.ceil(train / Number(n.batch_size))
      : null;
  const budget =
    goal && epochBatches
      ? Math.min(goal, epochBatches * Number(n.max_epochs ?? 1))
      : goal;
  const pct =
    p && budget ? Math.min(100, (p.optimizer_steps / budget) * 100) : 0;
  const selected = run.checkpoints.find((c) => c.id === checkpoint);
  return (
    <>
      <div className="run-strip">
        <div>
          <span className="eyebrow">
            Selected campaign · {run.config.host ?? "host unknown"}
          </span>
          <h2>{run.id}</h2>
          <span className="muted">
            Config <Hash value={run.config_sha256} /> · source{" "}
            <Hash
              value={
                run.source_identity.source_commit ??
                run.source_identity.source_main_sha
              }
            />
          </span>
        </div>
        <div className="status-stack">
          <Badge tone={run.progress.status === "observed" ? "green" : "amber"}>
            {run.progress.status === "stale"
              ? `Stale · ${p?.state ?? "unknown"}`
              : (p?.state.replaceAll("_", " ") ?? run.progress.status)}
          </Badge>
          <small>{when(run.progress.observed_at)}</small>
        </div>
      </div>
      <div className="training-layout">
        <LossChart points={run.curve} />
        <section className="panel">
          <span className="eyebrow">Bounded training pass</span>
          <h2 className="big-number">
            {count(p?.optimizer_steps)} <span>/ {count(budget)}</span>
          </h2>
          <div className="progress-track">
            <div style={{ width: `${pct}%` }} />
          </div>
          <p>{pct.toFixed(1)}% of configured step/epoch budget</p>
          <dl className="stat-list">
            <div>
              <dt>Completed epochs</dt>
              <dd>{count(p?.completed_epochs)}</dd>
            </div>
            <div>
              <dt>Unique train contexts seen</dt>
              <dd>{count(p?.unique_train_data_runs_seen)}</dd>
            </div>
            <div>
              <dt>Train contexts available</dt>
              <dd>{count(train)}</dd>
            </div>
            <div>
              <dt>Validation contexts scored</dt>
              <dd>
                {count(p?.unique_validation_data_runs_scored)} /{" "}
                {count(run.counts.partitions?.validation)}
              </dd>
            </div>
            <div>
              <dt>Full validation loss</dt>
              <dd>
                {number(p?.latest_full_validation?.loss)?.toFixed(4) ??
                  "Pending / unavailable"}
              </dd>
            </div>
            <div>
              <dt>Run-average throughput</dt>
              <dd>
                {number(
                  p?.performance.optimizer_steps_per_wall_second,
                )?.toFixed(3) ?? "—"}{" "}
                / s
              </dd>
            </div>
            <div>
              <dt>Peak GPU allocation</dt>
              <dd>{bytes(p?.performance.cuda_peak_allocated_bytes)}</dd>
            </div>
          </dl>
          <Badge>Research only</Badge>
          <p className="footnote">
            A completed optimizer pass does not establish production admission,
            an untouched holdout, or self-play improvement.
          </p>
        </section>
      </div>
      <div className="metrics compact">
        <Metric
          label="Generated proposals"
          value={count(p?.model_generated_programs)}
          detail="Durable emissions; not measured rewards"
        />
        <Metric
          label="Novel programs executed"
          value={count(p?.novel_programs_executed)}
          detail="Actual executions reported by run"
        />
        <Metric
          label="Published GP results"
          value={count(p?.published_results)}
          detail="Producer-reported publication receipts"
        />
        <Metric
          label="Comparison games"
          value={count(p?.comparison_games)}
          detail="No win rate inferred from loss"
        />
      </div>
      {!compact && (
        <>
          <section className="panel">
            <div className="panel-heading">
              <div>
                <span className="eyebrow">Model lineage</span>
                <h2>Checkpoints</h2>
              </div>
              <Badge>
                {count(run.checkpoint_count_observed)} directories observed
              </Badge>
            </div>
            <p className="footnote">
              Latest 24 manifests. Weight hashes are declared by the checkpoint
              producer; the dashboard does not repeatedly read model weights.
            </p>
            <div
              className="table-wrap"
              tabIndex={0}
              role="region"
              aria-label="Checkpoint records"
            >
              <table>
                <thead>
                  <tr>
                    <th>Step</th>
                    <th>Model state</th>
                    <th>Parameter identity</th>
                    <th>Size</th>
                    <th>Run binding</th>
                    <th />
                  </tr>
                </thead>
                <tbody>
                  {run.checkpoints.map((c) => (
                    <tr key={c.id}>
                      <td>
                        <code>{c.id}</code>
                      </td>
                      <td>
                        <Hash value={c.manifest.state_sha256} />
                      </td>
                      <td>
                        <Hash value={c.manifest.parameter_sha256} />
                      </td>
                      <td>{bytes(c.manifest.state_bytes)}</td>
                      <td>
                        <Badge tone={c.identity_matches ? "green" : "amber"}>
                          {c.identity_matches
                            ? "Config matched"
                            : "Unqualified"}
                        </Badge>
                      </td>
                      <td>
                        <button
                          className="text-button"
                          onClick={() => setCheckpoint(c.id)}
                        >
                          Inspect
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
          <div className="two-column">
            <section className="panel">
              <h2>Model & optimizer</h2>
              <Details fields={n} />
            </section>
            <section className="panel">
              <h2>Provenance & admission</h2>
              <Details
                fields={{
                  config_sha256: run.config_sha256,
                  membership_sha256: run.membership_sha256,
                  snapshot: run.config.snapshot,
                  source_identity: run.source_identity,
                  runtime_versions: run.config.runtime_versions,
                  research_admission: run.config.research_optimizer_admission,
                }}
              />
            </section>
          </div>
        </>
      )}
      {selected && (
        <Dialog
          title={`Checkpoint ${selected.id}`}
          close={() => setCheckpoint(null)}
        >
          <Details fields={selected.manifest} />
          <p>{selected.verification}</p>
          <Details fields={selected.evidence} />
        </Dialog>
      )}
    </>
  );
}
