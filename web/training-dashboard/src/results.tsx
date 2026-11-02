import { useState } from "react";
import { count, number, text } from "./types";
import type { Snapshot } from "./types";
import { Badge, Hash, Details } from "./shared";

export function Results({ snapshot: s }: { snapshot: Snapshot }) {
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
