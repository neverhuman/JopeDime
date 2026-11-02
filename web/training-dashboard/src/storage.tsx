import { useState } from "react";
import { bytes, count, when } from "./types";
import type { Snapshot } from "./types";
import { Badge, Hash, Empty } from "./shared";

export function StorageView({ snapshot: s }: { snapshot: Snapshot }) {
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
