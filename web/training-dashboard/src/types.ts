export type Fields = Record<string, unknown>;
export interface Observation<T> {
  status: string;
  source: string;
  observed_at: number | string | null;
  sha256: string | null;
  error?: string | null;
  value: T;
}
export interface Point {
  step: number;
  train: number;
  validation: number;
  at: number;
  panel_sha256: string;
  objective: string;
  terms: { training: Fields; validation: Fields };
}
export interface Progress {
  run_id: string;
  state: string;
  at: number;
  optimizer_steps: number;
  completed_epochs: number;
  total_train_presentations: number;
  unique_train_data_runs_seen: number;
  unique_validation_data_runs_scored: number;
  model_generated_programs: number;
  novel_programs_executed: number;
  published_results: number;
  comparison_games: number;
  performance: Fields;
  outbox: Fields;
  latest_full_validation: Fields | null;
  scope: string;
}
export interface Checkpoint {
  id: string;
  evidence: { source: string; sha256: string; status: string };
  identity_matches: boolean;
  manifest: Fields;
  verification: string;
}
export interface Run {
  id: string;
  config_sha256: string | null;
  identity_matches: boolean;
  config: {
    host?: string;
    neural: Fields;
    runtime_versions?: Fields;
    research_optimizer_admission?: Fields;
    snapshot?: Fields;
    interpretation?: Fields;
    duration_seconds?: number;
  };
  progress: Observation<Progress | null>;
  curve: Point[];
  counts: {
    partitions?: { train?: number; validation?: number };
    pending_contexts?: number;
    loaded_kumo_contexts?: number;
  };
  membership_sha256: string | null;
  checkpoints: Checkpoint[];
  checkpoint_count_observed: number;
  source_identity: Fields;
  terminal_receipt: Observation<Fields | null>;
}
export interface Bucket {
  bucket: string;
  region: string;
  role: string;
  replica_of?: string | null;
  bytes: number | null;
  objects: number | null;
  observed_at: string | null;
  status: string;
  scope?: string;
}
export interface Storage {
  evidence: Observation<{
    buckets: Bucket[];
    observed_bytes: number | null;
    observed_objects: number | null;
    complete: boolean;
    scope: string;
    source?: string;
    captured_at?: string;
    errors?: string[];
  }>;
  catalog_fallback: boolean;
}
export interface Program {
  id: string;
  dsl: string;
  operators: string[];
  family: string;
  context_occurrences: number;
  source: string;
  scope: string;
  score: number | null;
  score_note: string;
}
export interface GP {
  status: string;
  unique_programs: number | null;
  global_complete: boolean;
  scope: string;
  families: Record<string, number>;
  sources: Fields[];
  programs?: Program[];
}
export interface Assertion {
  id: string;
  value: number | null;
  unit: string;
  numerator: number;
  denominator: number;
  status: string;
  evidence_sha256: string;
  source_field: string;
}
export interface Wave {
  id: string;
  card: Fields;
  assertions: Assertion[];
  manifest_sha256: string;
  metrics_sha256: string;
  publication: string;
  archive: string;
  source_commit: string;
  training_identity_sha256: string;
  archive_manifest_sha256: string;
  archive_roots: Fields[];
}
export interface Snapshot {
  schema: "jope.observatory/v1";
  generated_at: number;
  service: {
    version: string;
    refresh_seconds: number;
    mode: string;
    production_authority: boolean;
  };
  storage: Storage;
  datasets: {
    name: string;
    count: number;
    status: string;
    observed_at: string;
    sha256: string;
    license_status: Record<string, number>;
    scope: string;
  }[];
  runs: Run[];
  gp: GP;
  research: { waves: Wave[]; scope: string };
  fleet: Observation<Fields | null>[];
  limits: Fields;
  code_identity?: Observation<Fields | null>;
}
export const number = (v: unknown): number | null =>
  typeof v === "number" && Number.isFinite(v) ? v : null;
export const text = (v: unknown, fallback = "Unknown"): string =>
  typeof v === "string" ? v : fallback;
export const count = (v: unknown) =>
  number(v)?.toLocaleString(undefined, { maximumFractionDigits: 0 }) ?? "—";
export function bytes(v: unknown): string {
  const n = number(v);
  if (n === null) return "—";
  if (n === 0) return "0 B";
  const i = Math.min(5, Math.max(0, Math.floor(Math.log(n) / Math.log(1024))));
  return `${(n / 1024 ** i).toLocaleString(undefined, { maximumFractionDigits: i ? 2 : 0 })} ${["B", "KiB", "MiB", "GiB", "TiB", "PiB"][i]}`;
}
export function when(v: unknown): string {
  if (typeof v !== "string" && typeof v !== "number") return "Unknown";
  const d = new Date(typeof v === "number" ? v * 1000 : v);
  return Number.isNaN(d.getTime()) ? "Unknown" : d.toLocaleString();
}
export function short(v: unknown): string {
  return typeof v === "string" ? v.slice(0, 12) : "Unknown";
}
export function validateSnapshot(v: unknown): Snapshot {
  if (!v || typeof v !== "object") throw new Error("Invalid observer response");
  const s = v as Partial<Snapshot>;
  if (
    s.schema !== "jope.observatory/v1" ||
    !Array.isArray(s.runs) ||
    !s.storage?.evidence ||
    !s.gp ||
    !Array.isArray(s.research?.waves) ||
    !Array.isArray(s.fleet) ||
    !Array.isArray(s.datasets) ||
    typeof s.generated_at !== "number"
  )
    throw new Error("Unsupported or incomplete observer response");
  return s as Snapshot;
}
