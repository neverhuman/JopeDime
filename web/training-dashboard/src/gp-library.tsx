import { useEffect, useState } from "react";
import { count, short, text } from "./types";
import type { GP, Program } from "./types";
import { Badge, Hash, Empty, Details, Dialog } from "./shared";

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
