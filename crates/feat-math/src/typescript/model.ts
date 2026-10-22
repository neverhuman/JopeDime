// Standalone typed model. Supply the compiled module bytes explicitly.
export type NumericVector = ReadonlyArray<number> | Float64Array;
export type NumericRows = ReadonlyArray<NumericVector>;
export type StringRows = ReadonlyArray<ReadonlyArray<string>>;
export type ModuleSource = Uint8Array | ArrayBuffer | WebAssembly.Module;
const boundHeaders: ReadonlyArray<string> | null = __BOUND_HEADERS__;
const maxElements = 16_777_216;

interface Kernel {
  memory: WebAssembly.Memory;
  model_abi_version(): number;
  model_values(len: number): number;
  model_labels(len: number): number;
  model_text(len: number): number;
  model_output(): number;
  model_output_len(): number;
  model_fit(rows: number, cols: number): number;
  model_predict(rows: number, cols: number): number;
  model_predict_text(): number;
  model_dispose(): void;
}

function checkedSize(value: number): void {
  if (!Number.isSafeInteger(value) || value < 0 || value > maxElements)
    throw new RangeError("input exceeds transfer limit");
}
function numericShape(rows: NumericRows, width?: number): number {
  const cols = width === undefined ? rows[0]?.length : width;
  if (cols === undefined || cols === 0) throw new RangeError("empty feature width");
  checkedSize(rows.length * cols);
  for (const row of rows) {
    if (row.length !== cols) throw new RangeError("row width mismatch");
    for (const value of row)
      if (typeof value !== "number") throw new TypeError("numeric features required");
  }
  return cols;
}
function quote(value: string): string {
  return '"' + value.replace(/"/g, '""') + '"';
}

export class Model {
  private width: number | undefined;
  private alive = true;
  private constructor(private readonly kernel: Kernel) {}

  static async create(source: ModuleSource): Promise<Model> {
    const compiled = source instanceof WebAssembly.Module
      ? source
      : await WebAssembly.compile(source instanceof ArrayBuffer ? source : new Uint8Array(source).buffer);
    if (WebAssembly.Module.imports(compiled).length !== 0)
      throw new TypeError("model module must not request host imports");
    const instance = await WebAssembly.instantiate(compiled, {});
    const exports = instance.exports;
    if (!(exports.memory instanceof WebAssembly.Memory))
      throw new TypeError("model memory missing");
    for (const name of ["model_abi_version", "model_values", "model_labels", "model_text",
      "model_output", "model_output_len", "model_fit", "model_predict",
      "model_predict_text", "model_dispose"])
      if (typeof exports[name] !== "function") throw new TypeError("model ABI incomplete");
    const kernel = exports as unknown as Kernel;
    if (kernel.model_abi_version() !== 1) throw new TypeError("model ABI version mismatch");
    return new Model(kernel);
  }

  private check(): void {
    if (!this.alive) throw new Error("model disposed or trapped");
  }
  private call(operation: () => number): void {
    this.check();
    let status: number;
    try { status = operation(); }
    catch (error) { this.alive = false; throw error; }
    if (status !== 0) throw new RangeError("model rejected input (status " + status + ")");
  }
  private copyRows(rows: NumericRows, cols: number): void {
    const len = rows.length * cols;
    const pointer = this.kernel.model_values(len);
    if (len > 0 && pointer === 0) throw new RangeError("feature allocation refused");
    const values = new Float64Array(this.kernel.memory.buffer, pointer, len);
    let offset = 0;
    for (const row of rows) { values.set(row, offset); offset += cols; }
  }
  private output(expected?: number): Float64Array {
    const length = this.kernel.model_output_len();
    if (expected !== undefined && length !== expected) throw new Error("prediction count mismatch");
    // Never expose a live view: the next call can allocate/grow linear memory.
    return new Float64Array(this.kernel.memory.buffer, this.kernel.model_output(), length).slice();
  }

  fit(rows: NumericRows, targets: NumericVector): this {
    this.check();
    if (boundHeaders !== null) throw new Error("fitted artifact cannot be refitted");
    const cols = numericShape(rows);
    if (rows.length === 0 || targets.length !== rows.length ||
        targets.some(value => typeof value !== "number" || !Number.isFinite(value)))
      throw new RangeError("invalid training targets");
    this.call(() => {
      this.copyRows(rows, cols);
      const pointer = this.kernel.model_labels(targets.length);
      if (pointer === 0) throw new RangeError("target allocation refused");
      new Float64Array(this.kernel.memory.buffer, pointer, targets.length).set(targets);
      return this.kernel.model_fit(rows.length, cols);
    });
    this.width = cols;
    return this;
  }

  predict(rows: NumericRows): Float64Array {
    this.check();
    if (boundHeaders !== null) throw new Error("use predictStrings for fitted artifacts");
    if (this.width === undefined) throw new Error("fit required before prediction");
    const cols = numericShape(rows, this.width);
    this.call(() => {
      this.copyRows(rows, cols);
      return this.kernel.model_predict(rows.length, cols);
    });
    return this.output(rows.length);
  }

  predictStrings(rows: StringRows): Float64Array {
    this.check();
    if (boundHeaders === null) throw new Error("string rows require a fitted artifact");
    const records = [boundHeaders.map(quote).join(",")];
    for (const row of rows) {
      if (row.length !== boundHeaders.length || row.some(value => typeof value !== "string"))
        throw new RangeError("invalid string row");
      records.push(row.map(quote).join(","));
    }
    return this.predictCsv(records.join("\n") + "\n", rows.length);
  }

  predictCsv(csv: string, expectedRows?: number): Float64Array {
    this.check();
    if (boundHeaders === null) throw new Error("CSV prediction requires a fitted artifact");
    const bytes = new TextEncoder().encode(csv);
    checkedSize(bytes.length);
    this.call(() => {
      const pointer = this.kernel.model_text(bytes.length);
      if (bytes.length > 0 && pointer === 0) throw new RangeError("text allocation refused");
      new Uint8Array(this.kernel.memory.buffer, pointer, bytes.length).set(bytes);
      return this.kernel.model_predict_text();
    });
    return this.output(expectedRows);
  }

  score(rows: NumericRows, targets: NumericVector): number {
    if (targets.length !== rows.length || targets.length === 0 || targets.some(v => !Number.isFinite(v)))
      throw new RangeError("invalid scoring targets");
    const predictions = this.predict(rows);
    let mean = 0;
    for (const value of targets) mean += value;
    mean /= targets.length;
    let residual = 0, total = 0;
    for (let i = 0; i < targets.length; i++) {
      const error = targets[i] - predictions[i], centered = targets[i] - mean;
      residual += error * error;
      total += centered * centered;
    }
    return total <= 0 ? 0 : 1 - residual / total;
  }

  dispose(): void {
    if (this.alive) { this.kernel.model_dispose(); this.alive = false; }
  }
}

export const createModel = Model.create;
