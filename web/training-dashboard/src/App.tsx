import { useEffect, useState } from "react";
import { bytes, count, validateSnapshot } from "./types";
import type { Snapshot } from "./types";
import { Badge, Empty, Metric } from "./shared";
import { Training } from "./training";
import { StorageView } from "./storage";
import { GpLibrary } from "./gp-library";
import { Fleet } from "./fleet";
import { Results } from "./results";
export { LossChart } from "./shared";


const tabs = [
  "Overview",
  "Training",
  "Fleet",
  "Storage & datasets",
  "GP library",
  "Results & lineage",
] as const;
type Tab = (typeof tabs)[number];

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
