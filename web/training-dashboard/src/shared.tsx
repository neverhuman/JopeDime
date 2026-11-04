import { useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import { count, short, text } from "./types";
import type { Fields, Point } from "./types";

export function Badge({
  children,
  tone = "",
}: {
  children: ReactNode;
  tone?: string;
}) {
  return <span className={`badge ${tone}`}>{children}</span>;
}
export function Hash({ value }: { value: unknown }) {
  return <code title={text(value)}>{short(value)}</code>;
}
export function Metric({
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
export function Empty({ children }: { children: ReactNode }) {
  return <div className="empty">{children}</div>;
}
export function Details({ fields }: { fields: Fields }) {
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

export function Dialog({
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
