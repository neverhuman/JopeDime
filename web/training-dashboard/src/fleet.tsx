import { bytes, count, number, text, when } from "./types";
import type { Fields, Snapshot } from "./types";
import { Badge, Empty } from "./shared";

export function Fleet({ snapshot: s }: { snapshot: Snapshot }) {
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
