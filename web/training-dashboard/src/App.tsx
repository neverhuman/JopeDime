import { useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import {
  bytes,
  count,
  number,
  short,
  text,
  validateSnapshot,
  when,
} from "./types";
import type { Fields, GP, Point, Program, Run, Snapshot } from "./types";

const tabs = [
  "Overview",
  "Training",
  "Fleet",
  "Storage & datasets",
  "GP library",
  "Results & lineage",
] as const;
type Tab = (typeof tabs)[number];
function Badge({
  children,
  tone = "",
}: {
  children: ReactNode;
  tone?: string;
}) {
  return <span className={`badge ${tone}`}>{children}</span>;
}
function Hash({ value }: { value: unknown }) {
  return <code title={text(value)}>{short(value)}</code>;
}
function Metric({
  label,
  value,
  detail,
  tone,
}: {
  label: string;
  value: string;
  detail: string;
  tone?: string;
}) {
  return (
    <article className="metric">
      <span className="eyebrow">{label}</span>
      <strong className={tone}>{value}</strong>
      <small>{detail}</small>
    </article>
  );
}
function Empty({ children }: { children: ReactNode }) {
  return <div className="empty">{children}</div>;
}
function Details({ fields }: { fields: Fields }) {
  return (
    <dl className="details">
      {Object.entries(fields).map(([k, v]) => (
        <div key={k}>
          <dt>{k.replaceAll("_", " ")}</dt>
          <dd>
            {v === null || v === undefined ? (
              "Unknown"
            ) : typeof v === "object" ? (
              <pre>{JSON.stringify(v, null, 2)}</pre>
            ) : (
              String(v)
            )}
          </dd>
        </div>
      ))}
    </dl>
  );
}

export function LossChart({ points }: { points: Point[] }) {
  const [hover, setHover] = useState<number | null>(null);
  const [series, setSeries] = useState<"both" | "train" | "validation">("both");
  // A changed evaluation panel starts a separate series; never connect unlike measurements.
  const groups = useMemo(() => {
    const result: Point[][] = [];
    let key = "";
    for (const p of points) {
      const next = `${p.panel_sha256}:${p.objective}`;
      if (next !== key) {
        result.push([]);
        key = next;
      }
      result[result.length - 1].push(p);
    }
    return result;
  }, [points]);
  const p = points[hover ?? points.length - 1];
  const values = points.flatMap((p) => [p.train, p.validation]);
  const lo = Math.max(0, Math.min(...values) * 0.88);
  const hi = Math.max(...values) * 1.08;
  const last = Math.max(1, ...points.map((p) => p.step));
  const x = (s: number) => 54 + (s / last) * 820;
  const y = (v: number) => 232 - ((v - lo) / (hi - lo || 1)) * 210;
  return (
    <section className="panel loss-panel">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">Learning signal</span>
          <h2>Train & validation loss</h2>
        </div>
        <select
          aria-label="Loss series"
          value={series}
          onChange={(e) => setSeries(e.target.value as typeof series)}
        >
          <option value="both">Both series</option>
          <option value="train">Train</option>
          <option value="validation">Validation</option>
        </select>
      </div>
      <div className="chart-key">
        <span className="train-dot" />
        Train <b>{p?.train.toFixed(4) ?? "—"}</b>
        <span className="validation-dot" />
        Validation <b>{p?.validation.toFixed(4) ?? "—"}</b>
        <span className="step-readout">Step {p ? count(p.step) : "—"}</span>
      </div>
      {!points.length ? (
        <Empty>
          No identity-qualified fixed-panel loss measurements are available.
        </Empty>
      ) : (
        <svg
          viewBox="0 0 900 275"
          role="img"
          aria-label="Training and validation loss by optimizer step"
          className="loss-chart"
          onMouseLeave={() => setHover(null)}
        >
          {[0, 1, 2, 3, 4].map((i) => {
            const v = lo + ((hi - lo) * i) / 4;
            return (
              <g key={i}>
                <line
                  x1="54"
                  x2="874"
                  y1={y(v)}
                  y2={y(v)}
                  className="gridline"
                />
                <text x="44" y={y(v) + 4} textAnchor="end">
                  {v.toFixed(2)}
                </text>
              </g>
            );
          })}
          {[0, 0.25, 0.5, 0.75, 1].map((f) => (
            <text key={f} x={x(last * f)} y="258" textAnchor="middle">
              {Math.round(last * f)}
            </text>
          ))}
          {groups.map((group, i) => (
            <g key={i}>
              {series !== "validation" && (
                <polyline
                  className="train-line"
                  points={group
                    .map((p) => `${x(p.step)},${y(p.train)}`)
                    .join(" ")}
                />
              )}{" "}
              {series !== "train" && (
                <polyline
                  className="validation-line"
                  points={group
                    .map((p) => `${x(p.step)},${y(p.validation)}`)
                    .join(" ")}
                />
              )}
            </g>
          ))}
          {p && (
            <g>
              <line
                className="hoverline"
                x1={x(p.step)}
                x2={x(p.step)}
                y1="18"
                y2="232"
              />
              {series !== "validation" && (
                <circle cx={x(p.step)} cy={y(p.train)} r="4" fill="#137b60" />
              )}
              {series !== "train" && (
                <circle
                  cx={x(p.step)}
                  cy={y(p.validation)}
                  r="4"
                  fill="#8181d5"
                />
              )}
            </g>
          )}
          {points.map((p, i) => (
            <rect
              key={`${p.step}-${i}`}
              x={x(p.step) - Math.max(5, 410 / points.length)}
              y="10"
              width={Math.max(10, 820 / points.length)}
              height="224"
              fill="transparent"
              onMouseEnter={() => setHover(i)}
            >
              <title>{`Step ${p.step}, train ${p.train.toFixed(5)}, validation ${p.validation.toFixed(5)}`}</title>
            </rect>
          ))}
        </svg>
      )}
      <p className="footnote">
        Fixed development panels · dropout disabled · lower is better within the
        same objective and panel. Full validation and verified self-play gains
        are tracked separately.
      </p>
    </section>
  );
}

function Training({
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
function Dialog({
  title,
  close,
  children,
}: {
  title: string;
  close: () => void;
  children: ReactNode;
}) {
  const container = useRef<HTMLElement>(null);
  // Capture before React commits autoFocus to the close button. An effect runs
  // after that commit and would otherwise remember a soon-detached dialog node.
  const opener = useRef(document.activeElement as HTMLElement | null);
  const closeRef = useRef(close);
  closeRef.current = close;
  useEffect(() => {
    const previous = opener.current;
    // Focus on each effect setup, including React StrictMode's setup/cleanup
    // probe. Restoring the opener in that probe must not leave focus outside.
    container.current?.querySelector<HTMLElement>("button")?.focus();
    const key = (e: KeyboardEvent) => {
      if (e.key === "Escape") closeRef.current();
      if (e.key === "Tab") {
        const items = container.current?.querySelectorAll<HTMLElement>(
          'button:not([disabled]),a[href],input,select,[tabindex="0"]',
        );
        if (!items?.length) return;
        const first = items[0],
          last = items[items.length - 1];
        if (!container.current?.contains(document.activeElement)) {
          e.preventDefault();
          (e.shiftKey ? last : first).focus();
        } else if (e.shiftKey && document.activeElement === first) {
          e.preventDefault();
          last.focus();
        } else if (!e.shiftKey && document.activeElement === last) {
          e.preventDefault();
          first.focus();
        }
      }
    };
    window.addEventListener("keydown", key);
    return () => {
      window.removeEventListener("keydown", key);
      previous?.focus();
    };
  }, []);
  return (
    <div
      className="modal-backdrop"
      onClick={(e) => {
        if (e.target === e.currentTarget) close();
      }}
    >
      <section
        ref={container}
        className="modal"
        role="dialog"
        aria-modal="true"
        aria-label={title}
      >
        <div className="panel-heading">
          <h2>{title}</h2>
          <button autoFocus onClick={close} aria-label="Close details">
            Close ×
          </button>
        </div>
        {children}
      </section>
    </div>
  );
}

function StorageView({ snapshot: s }: { snapshot: Snapshot }) {
  const v = s.storage.evidence.value;
  const [filter, setFilter] = useState("");
  return (
    <>
      <section className="panel">
        <div className="panel-heading">
          <div>
            <span className="eyebrow">S3 estate</span>
            <h2>Storage by bucket</h2>
          </div>
          <input
            aria-label="Filter storage buckets"
            placeholder="Filter buckets…"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
          />
        </div>
        <p className="footnote">
          {v.scope}. Observation times below describe the data, not the browser
          refresh. Physical replicas are included; unavailable buckets remain
          unknown.
        </p>
        <div
              className="table-wrap"
              tabIndex={0}
              role="region"
              aria-label="Storage bucket records"
            >
          <table>
            <thead>
              <tr>
                <th>Bucket / purpose</th>
                <th>Physical bytes</th>
                <th>Objects</th>
                <th>Region</th>
                <th>Measured at</th>
                <th>Coverage</th>
              </tr>
            </thead>
            <tbody>
              {v.buckets
                .filter((b) =>
                  b.bucket.toLowerCase().includes(filter.toLowerCase()),
                )
                .map((b) => (
                  <tr key={b.bucket}>
                    <td>
                      <b>{b.bucket}</b>
                      <small>{b.role}</small>
                    </td>
                    <td>{bytes(b.bytes)}</td>
                    <td>{count(b.objects)}</td>
                    <td>{b.region}</td>
                    <td>{when(b.observed_at)}</td>
                    <td>
                      <Badge tone={b.status === "unavailable" ? "amber" : ""}>
                        {b.status}
                        {b.replica_of ? " · replica" : ""}
                      </Badge>
                    </td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
        {v.errors?.map((e) => (
          <p className="notice" key={e}>
            {e}
          </p>
        ))}
      </section>
      <section className="panel">
        <span className="eyebrow">Dataset catalog</span>
        <h2>Datasets, coverage & licenses</h2>
        <p className="footnote">
          Dataset IDs, corpus contexts, shard records, and S3 objects are
          separate quantities.
        </p>
        <div
              className="table-wrap"
              tabIndex={0}
              role="region"
              aria-label="Dataset catalog records"
            >
          <table>
            <thead>
              <tr>
                <th>Catalog</th>
                <th>Dataset IDs</th>
                <th>License records</th>
                <th>Catalog identity</th>
                <th>Observation</th>
              </tr>
            </thead>
            <tbody>
              {s.datasets.map((d) => (
                <tr key={d.name}>
                  <td>{d.name}</td>
                  <td>{count(d.count)}</td>
                  <td>
                    {Object.entries(d.license_status)
                      .map(([k, v]) => `${k}: ${count(v)}`)
                      .join(" · ")}
                  </td>
                  <td>
                    <Hash value={d.sha256} />
                  </td>
                  <td>
                    {d.observed_at} · {d.status}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {!s.datasets.length && <Empty>No dataset census is available.</Empty>}
      </section>
    </>
  );
}

export function GpLibrary({ summary }: { summary: GP }) {
  const [library, setLibrary] = useState<GP | null>(null);
  const [error, setError] = useState("");
  const [query, setQuery] = useState("");
  const [family, setFamily] = useState("all");
  const [page, setPage] = useState(0);
  const [selected, setSelected] = useState<Program | null>(null);
  const identity = summary.sources.map((s) => s.sha256).join(":");
  useEffect(() => {
    const controller = new AbortController();
    setError("");
    setLibrary(null);
    fetch("/api/v1/gp", { signal: controller.signal })
      .then((r) => {
        if (!r.ok) throw new Error("GP library request failed");
        return r.json();
      })
      .then((v) => {
        if (!Array.isArray(v.programs))
          throw new Error("Library detail unavailable");
        setLibrary(v);
      })
      .catch((e) => {
        if (e.name !== "AbortError") setError(e.message);
      });
    return () => controller.abort();
  }, [identity]);
  const rows = (library?.programs ?? []).filter(
    (p) =>
      (family === "all" || p.family === family) &&
      `${p.id} ${p.dsl} ${p.source}`
        .toLowerCase()
        .includes(query.toLowerCase()),
  );
  return (
    <>
      <section className="panel">
        <div className="panel-heading">
          <div>
            <span className="eyebrow">Genetic programming</span>
            <h2>Program library</h2>
          </div>
          <Badge tone={summary.status === "observed" ? "green" : "amber"}>
            {summary.status}
          </Badge>
        </div>
        <p>{summary.scope}</p>
        <div className="family-list">
          {Object.entries(summary.families)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 12)
            .map(([f, n]) => (
              <button
                className={family === f ? "active-family" : ""}
                key={f}
                onClick={() => {
                  setFamily(family === f ? "all" : f);
                  setPage(0);
                }}
              >
                {f} <b>{count(n)}</b>
              </button>
            ))}
        </div>
        <div className="filters">
          <input
            aria-label="Search programs"
            placeholder="Search program hash, DSL or source…"
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setPage(0);
            }}
          />
          <select
            aria-label="Program family"
            value={family}
            onChange={(e) => {
              setFamily(e.target.value);
              setPage(0);
            }}
          >
            <option value="all">All families</option>
            {Object.keys(summary.families).map((f) => (
              <option key={f}>{f}</option>
            ))}
          </select>
          <span>{count(rows.length)} matching programs</span>
        </div>
        {error && (
          <p role="alert" className="notice">
            {error}
          </p>
        )}
        {!library && !error && (
          <Empty>Loading hash-pinned program records…</Empty>
        )}
        <div
              className="table-wrap"
              tabIndex={0}
              role="region"
              aria-label="Program library records"
            >
          <table>
            <thead>
              <tr>
                <th>Program identity</th>
                <th>Family</th>
                <th>Operators</th>
                <th>Occurrences</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {rows.slice(page * 25, page * 25 + 25).map((p) => (
                <tr key={p.id}>
                  <td>
                    <Hash value={p.id} />
                    <small>{p.source}</small>
                  </td>
                  <td>
                    <Badge>{p.family}</Badge>
                  </td>
                  <td>{p.operators.join(" · ")}</td>
                  <td>{count(p.context_occurrences)}</td>
                  <td>
                    <button
                      className="text-button"
                      onClick={() => setSelected(p)}
                    >
                      Inspect program ↗
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="pagination">
          <button disabled={page === 0} onClick={() => setPage(page - 1)}>
            ← Previous
          </button>
          <span>
            Page {page + 1} of {Math.max(1, Math.ceil(rows.length / 25))}
          </span>
          <button
            disabled={(page + 1) * 25 >= rows.length}
            onClick={() => setPage(page + 1)}
          >
            Next →
          </button>
        </div>
      </section>
      <section className="panel">
        <h2>Library provenance</h2>
        {summary.sources.map((s, i) => (
          <Details fields={s} key={i} />
        ))}
      </section>
      {selected && (
        <Dialog
          title={`Program ${short(selected.id)}`}
          close={() => setSelected(null)}
        >
          <Badge>{selected.family}</Badge>
          <pre className="dsl">{selected.dsl}</pre>
          <Details
            fields={{
              program_sha256: selected.id,
              source: selected.source,
              scope: selected.scope,
              operators: selected.operators,
              context_occurrences: selected.context_occurrences,
              evaluator_score: selected.score,
              score_note: selected.score_note,
            }}
          />
        </Dialog>
      )}
    </>
  );
}

function Fleet({ snapshot: s }: { snapshot: Snapshot }) {
  const hosts = ["xbabe1", "xbabe2", "xbabe3", "xbabe4"];
  return (
    <>
      <div className="two-column">
        {hosts.map((host) => {
          const o = s.fleet.find((o) => o.value?.host === host);
          const v = o?.value;
          const age = number(v?.observed_at);
          const stale = !age || Date.now() / 1000 - age > 90;
          const gpus = Array.isArray(v?.gpus) ? (v.gpus as Fields[]) : [];
          return (
            <section className="panel" key={host}>
              <div className="panel-heading">
                <h2>{host}</h2>
                <Badge
                  tone={
                    !v || stale || v.status === "unavailable"
                      ? "amber"
                      : "green"
                  }
                >
                  {!v
                    ? "No live feed"
                    : v.status === "unavailable"
                      ? "Unavailable"
                      : stale
                        ? "Stale"
                        : "Observed"}
                </Badge>
              </div>
              {!v ? (
                <Empty>
                  Connect an approved node telemetry snapshot. Missing telemetry
                  does not mean idle hardware.
                </Empty>
              ) : (
                <>
                  <div className="node-stats">
                    <div>
                      <span>CPU cores</span>
                      <b>{count(v.cpu_count)}</b>
                    </div>
                    <div>
                      <span>Memory available</span>
                      <b>{bytes(v.memory_available_bytes)}</b>
                    </div>
                    <div>
                      <span>Disk available</span>
                      <b>{bytes(v.disk_available_bytes)}</b>
                    </div>
                  </div>
                  <p>
                    Load averages:{" "}
                    {Array.isArray(v.load)
                      ? v.load
                          .map((v) => number(v)?.toFixed(2) ?? "—")
                          .join(" / ")
                      : "Unknown"}
                  </p>
                  {gpus.map((g, i) => (
                    <div className="gpu-row" key={i}>
                      <strong>{text(g.name)}</strong>
                      <span>
                        {count(g.utilization_percent)}% GPU ·{" "}
                        {bytes(g.memory_used_bytes)} /{" "}
                        {bytes(g.memory_total_bytes)}
                      </span>
                      <div className="progress-track">
                        <div
                          style={{
                            width: `${Math.max(0, Math.min(100, Number(g.utilization_percent) || 0))}%`,
                          }}
                        />
                      </div>
                    </div>
                  ))}
                  <p className="footnote">
                    Measured {when(v.observed_at)}.{" "}
                    {text(
                      v.workload_scope,
                      "Host-wide resource counters; no attribution to learning inferred.",
                    )}
                  </p>
                </>
              )}
            </section>
          );
        })}
      </div>
      <p className="notice">
        GPU activity is a resource observation. Optimizer updates, changed model
        parameters, evaluator-bound results, and self-play comparisons are
        separate evidence.
      </p>
    </>
  );
}

function Results({ snapshot: s }: { snapshot: Snapshot }) {
  const [selected, setSelected] = useState<string | null>(null);
  const wave = s.research.waves.find((w) => w.id === selected);
  return (
    <>
      <section className="panel">
        <div className="panel-heading">
          <div>
            <span className="eyebrow">Preserved research</span>
            <h2>Waves & analyzable results</h2>
          </div>
          <Badge>{s.research.waves.length} waves observed</Badge>
        </div>
        <p className="footnote">
          {s.research.scope}. Git files, archive claims, exact S3 versions, and
          restoration proofs are distinct preservation milestones.
        </p>
        <div
              className="table-wrap"
              tabIndex={0}
              role="region"
              aria-label="Research wave records"
            >
          <table>
            <thead>
              <tr>
                <th>Research wave</th>
                <th>Recorded assertions</th>
                <th>Capture</th>
                <th>Archive claim</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {s.research.waves.map((w) => (
                <tr key={w.id}>
                  <td>
                    <b>{w.id}</b>
                    <small>{text(w.card.interpretation)}</small>
                  </td>
                  <td>{count(w.assertions.length)}</td>
                  <td>
                    <Badge>{text(w.card.capture_status)}</Badge>
                  </td>
                  <td>{text(w.card.archive_status)}</td>
                  <td>
                    <button
                      className="text-button"
                      onClick={() => setSelected(w.id)}
                    >
                      Inspect results
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
      <section className="panel">
        <h2>Code & model identities</h2>
        <Details
          fields={{
            dashboard_version: s.service.version,
            dashboard_code: s.code_identity?.value ?? null,
            observer_mode: s.service.mode,
          }}
        />
        {s.runs.map((r) => (
          <Details
            key={r.id}
            fields={{
              campaign: r.id,
              source: r.source_identity,
              config_sha256: r.config_sha256,
              snapshot_sha256: r.config.snapshot?.sha256,
              membership_sha256: r.membership_sha256,
              latest_checkpoint: r.checkpoints[0]?.manifest,
            }}
          />
        ))}
      </section>
      {wave && (
        <Dialog title={wave.id} close={() => setSelected(null)}>
          <Details
            fields={{
              ...wave.card,
              manifest_sha256: wave.manifest_sha256,
              metrics_sha256: wave.metrics_sha256,
              publication: wave.publication,
              archive: wave.archive,
              source_commit: wave.source_commit,
              training_identity_sha256: wave.training_identity_sha256,
              archive_manifest_sha256: wave.archive_manifest_sha256,
              archive_roots: wave.archive_roots,
            }}
          />
          <div
              className="table-wrap"
              tabIndex={0}
              role="region"
              aria-label="Research measurements"
            >
            <table>
              <thead>
                <tr>
                  <th>Measurement</th>
                  <th>Value</th>
                  <th>Unit / coverage</th>
                  <th>Interpretation</th>
                </tr>
              </thead>
              <tbody>
                {wave.assertions.map((a, i) => (
                  <tr key={`${a.id}-${i}`}>
                    <td>
                      {a.id}
                      <small>
                        <Hash value={a.evidence_sha256} />
                      </small>
                    </td>
                    <td>
                      {number(a.value)?.toLocaleString(undefined, {
                        maximumFractionDigits: 6,
                      }) ?? "Unknown"}
                    </td>
                    <td>
                      {a.unit}
                      <small>
                        {count(a.numerator)} / {count(a.denominator)}
                      </small>
                    </td>
                    <td>{a.status.replaceAll("_", " ")}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Dialog>
      )}
    </>
  );
}

export default function App() {
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [error, setError] = useState("");
  const [tab, setTab] = useState<Tab>("Overview");
  const [runId, setRunId] = useState("");
  const [clock, setClock] = useState(Date.now());
  useEffect(() => {
    let stopped = false;
    let timer: ReturnType<typeof setTimeout>;
    let delay = 5000;
    const controller = new AbortController();
    async function poll() {
      try {
        const r = await fetch("/api/v1/snapshot", {
          signal: controller.signal,
          cache: "no-store",
        });
        if (!r.ok) throw new Error(`Observer HTTP ${r.status}`);
        const v = validateSnapshot(await r.json());
        if (!stopped) {
          setSnapshot(v);
          setError("");
        }
        delay = 5000;
      } catch (e) {
        if (!stopped) {
          setError(e instanceof Error ? e.message : "Observer unavailable");
          delay = Math.min(60000, delay * 2);
        }
      }
      if (!stopped) timer = setTimeout(poll, delay);
    }
    void poll();
    return () => {
      stopped = true;
      controller.abort();
      clearTimeout(timer);
    };
  }, []);
  useEffect(() => {
    const t = setInterval(() => setClock(Date.now()), 1000);
    return () => clearInterval(t);
  }, []);
  const run =
    snapshot?.runs.find((r) => r.id === runId) ??
    snapshot?.runs[snapshot.runs.length - 1];
  const p = run?.progress.value;
  const age = snapshot
    ? Math.max(0, Math.floor(clock / 1000 - snapshot.generated_at))
    : null;
  const disconnected = !!error || (age !== null && age > 20);
  const s = snapshot;
  const storage = s?.storage.evidence.value;
  const datasets = s?.datasets.length === 1 ? s.datasets[0].count : null;
  const assertions = s?.research.waves.reduce(
    (n, w) => n + w.assertions.length,
    0,
  );
  return (
    <div className="app">
      <aside>
        <a className="brand" href="/" aria-label="JopeDime home">
          <span className="brand-mark">
            J<span>◌</span>
          </span>
          <div>
            JopeDime<small>RESEARCH OBSERVATORY</small>
          </div>
        </a>
        <div className="nav-caption">WORKSPACE</div>
        <nav aria-label="Dashboard sections">
          {tabs.map((t, i) => (
            <button
              key={t}
              className={tab === t ? "selected" : ""}
              aria-current={tab === t ? "page" : undefined}
              onClick={() => setTab(t)}
            >
              <span className="nav-symbol" aria-hidden="true">
                {["◫", "↗", "▦", "▱", "⌘", "◷"][i]}
              </span>
              {t}
              {t === "Training" && p?.state === "running" && (
                <i className="live-dot" />
              )}
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <Badge tone="sidebar-badge">Read-only observatory</Badge>
          <p>
            Evidence first.
            <br />
            Every result has a lineage.
          </p>
          <small>v{s?.service.version ?? "—"} · Rust + React</small>
        </div>
      </aside>
      <main>
        <header>
          <div className="breadcrumb">
            JopeDime <span>/</span> {tab}
          </div>
          <div className="connection">
            <i className={`live-dot ${disconnected ? "offline" : ""}`} />
            {!s
              ? "Connecting"
              : disconnected
                ? "Feed interrupted"
                : "Observer live"}
            <span>{age === null ? "—" : `${age}s ago`}</span>
          </div>
        </header>
        <div className="page">
          <div className="page-title">
            <div>
              <span className="eyebrow">Fleet-wide research intelligence</span>
              <h1>{tab === "Overview" ? "Training observatory" : tab}</h1>
              <p>
                Follow the learning. Inspect the evidence. Keep the lineage.
              </p>
            </div>
            {s && (
              <div className="campaign-select">
                <label htmlFor="campaign">CAMPAIGN</label>
                <select
                  id="campaign"
                  value={run?.id ?? ""}
                  onChange={(e) => setRunId(e.target.value)}
                >
                  {s.runs.map((r) => (
                    <option key={r.id} value={r.id}>
                      {r.id}
                    </option>
                  ))}
                </select>
              </div>
            )}
          </div>
          {error && (
            <div className="notice error" role="alert">
              Live feed interrupted: {error}. Retaining the last observation;
              values may be stale.
            </div>
          )}
          {!s ? (
            <Empty>
              Connecting to the Rust telemetry service. Live metrics appear when
              evidence is available.
            </Empty>
          ) : (
            <>
              {(tab === "Overview" || tab === "Storage & datasets") && (
                <div className="metrics">
                  <Metric
                    label={
                      s.storage.catalog_fallback
                        ? "S3 cataloged payload"
                        : "S3 physical storage"
                    }
                    value={bytes(storage?.observed_bytes)}
                    detail={`${storage?.complete ? "Configured buckets covered" : "Partial / dated coverage"} · ${count(storage?.observed_objects)} objects`}
                  />
                  <Metric
                    label="Cataloged datasets"
                    value={s.datasets.length ? count(datasets) : "—"}
                    detail={
                      s.datasets.length
                        ? `Dataset IDs · ${s.datasets[0].status === "historical" ? "snapshot" : "verified"} ${s.datasets[0].observed_at.slice(0, 10)}`
                        : "No verified dataset census"
                    }
                  />
                  <Metric
                    label="GP library programs"
                    value={count(s.gp.unique_programs)}
                    detail="Unique IDs in pinned snapshots · drill down"
                  />
                  <Metric
                    label="Research waves"
                    value={count(s.research.waves.length)}
                    detail={`${count(assertions)} recorded result metrics · inspect lineage`}
                  />
                </div>
              )}
              {tab === "Overview" && (
                <>
                  <Training key={run?.id} run={run} compact />
                  <section className="panel preservation">
                    <div>
                      <span className="eyebrow">Durable research memory</span>
                      <h2>Preserve every wave, not just the winner.</h2>
                      <p>
                        Compact metrics and context in Git. Bulk outputs and
                        checkpoints in immutable, versioned storage with
                        restoration evidence.
                      </p>
                    </div>
                    <button onClick={() => setTab("Results & lineage")}>
                      Inspect research waves ↗
                    </button>
                  </section>
                </>
              )}
              {tab === "Training" && <Training key={run?.id} run={run} />}
              {tab === "Fleet" && <Fleet snapshot={s} />}
              {tab === "Storage & datasets" && <StorageView snapshot={s} />}
              {tab === "GP library" && <GpLibrary summary={s.gp} />}
              {tab === "Results & lineage" && <Results snapshot={s} />}
            </>
          )}
          <footer>
            <span>All times shown in your browser’s local timezone.</span>
            <span>Observation ≠ admission ≠ scientific acceptance</span>
          </footer>
        </div>
      </main>
    </div>
  );
}
